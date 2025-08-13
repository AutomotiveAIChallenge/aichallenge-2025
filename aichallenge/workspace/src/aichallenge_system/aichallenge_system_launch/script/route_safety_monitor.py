#!/usr/bin/env python3

import rclpy
import rclpy.node
import time
import xml.etree.ElementTree as ET
from matplotlib import path
import matplotlib.pyplot as plt
import os
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
try:
    from tier4_vehicle_msgs.msg import ActuationCommandStamped
except:
    None

class RouteDeviationSafetyMonitor:
    def __init__(self, osm_file_path=None):
        # OSMファイルを探す
        if osm_file_path:
            self.osm_file = osm_file_path
        else:
            try:
                from ament_index_python.packages import get_package_share_directory
                self.osm_file = os.path.join(
                    get_package_share_directory('aichallenge_system_launch'),
                    'map', 'route_area.osm'
                )
            except:
                # フォールバック: 相対パス
                script_dir = os.path.dirname(os.path.abspath(__file__))
                self.osm_file = os.path.join(script_dir, '..', 'map', 'route_area.osm')
        
        self.nodes = {}
        self.lane_polygons = []
        self.lane_coords = []  # 可視化用
        self._load_map()
    
    def _load_map(self):
        tree = ET.parse(self.osm_file)
        root = tree.getroot()
        
        # ノードの座標を取得
        for node in root.findall("node"):
            node_id = node.attrib['id']
            local_x = local_y = None
            
            for tag in node.findall('tag'):
                if tag.attrib['k'] == 'local_x':
                    local_x = float(tag.attrib['v'])
                elif tag.attrib['k'] == 'local_y':
                    local_y = float(tag.attrib['v'])
            
            if local_x is not None and local_y is not None:
                self.nodes[node_id] = {'x': local_x, 'y': local_y}
        
        # laneletからポリゴンを作成
        for relation in root.findall("relation"):
            if relation.find("tag[@k='type'][@v='lanelet']") is not None:
                left_way = right_way = None
                
                for member in relation.findall("member"):
                    role = member.attrib.get('role')
                    ref = member.attrib.get('ref')
                    
                    if role == 'left':
                        left_way = ref
                    elif role == 'right':
                        right_way = ref
                
                if left_way and right_way:
                    left_coords = self._get_way_coordinates(root, left_way)
                    right_coords = self._get_way_coordinates(root, right_way)
                    
                    if left_coords and right_coords:
                        polygon = self._create_polygon(left_coords, right_coords)
                        if polygon:
                            self.lane_polygons.append(path.Path(polygon))
                            self.lane_coords.append(polygon)  # 可視化用に保存
    
    def _get_way_coordinates(self, root, way_id):
        way = root.find(f"way[@id='{way_id}']")
        if way is None:
            return []
        
        coords = []
        for nd in way.findall('nd'):
            node_ref = nd.attrib['ref']
            if node_ref in self.nodes:
                node = self.nodes[node_ref]
                coords.append((node['x'], node['y']))
        return coords
    
    def _create_polygon(self, left_coords, right_coords):
        if not left_coords or not right_coords:
            return None
        
        # 左境界 + 右境界の逆順でポリゴンを作成
        polygon_points = left_coords + list(reversed(right_coords))
        
        if len(polygon_points) >= 3:
            polygon_points.append(polygon_points[0])  # ポリゴンを閉じる
            return polygon_points
        return None
    
    def is_in_any_lane(self, x, y):
        for polygon in self.lane_polygons:
            if polygon.contains_point((x, y)):
                return True
        return False
    
    def visualize_position(self, x, y, is_safe, save_path="route_status.png"):
        """現在位置とレーン境界を可視化"""
        plt.figure(figsize=(10, 8))
        
        # レーン境界を描画
        for lane_coords in self.lane_coords:
            if len(lane_coords) > 2:
                xs = [coord[0] for coord in lane_coords]
                ys = [coord[1] for coord in lane_coords]
                plt.plot(xs, ys, 'b-', alpha=0.6, linewidth=1)
                plt.fill(xs, ys, color='lightblue', alpha=0.3)
        
        # 現在位置を描画
        color = 'green' if is_safe else 'red'
        marker = 'o' if is_safe else 'x'
        label = 'Safe (In Lane)' if is_safe else 'Danger (Outside Lane)'
        
        plt.scatter(x, y, c=color, marker=marker, s=200, label=label, zorder=5)
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Route Safety Monitor - Current Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()  # メモリリーク防止

class RouteDeviationSafetyMonitorNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("route_deviation_safety_monitor")
        
        # パラメータ設定
        self.declare_parameter('enable_visualization', False)
        self.enable_viz = self.get_parameter('enable_visualization').get_parameter_value().bool_value
        
        # 安全監視装置の初期化
        self.safety_monitor = RouteDeviationSafetyMonitor()
        
        # 状態管理
        self.current_position = None
        self.current_velocity = 0.0  # 現在速度 [m/s]
        self.deviation_start_time = None
        self.is_outside_route = False
        self.position_count = 0  # 可視化用カウンタ
        
        # P制御パラメータ（比例制御のみ）
        self.target_speed_kmh = 5.0  # 目標速度 [km/h]
        self.target_speed_ms = self.target_speed_kmh / 3.6  # 目標速度 [m/s]
        self.max_acceleration = 2.0  # 最大加速度 [m/s²]
        self.min_acceleration = -3.0  # 最小加速度 [m/s²]
        
        # 安全制御状態管理
        self.safety_mode_active = False
        self.safety_control_active = False  # 介入状態フラグ
        self.safety_control_timer = None    # 300Hz制御タイマー
        self.low_speed_threshold = 4.0 / 3.6# 安全対策解除速度閾値 [m/s] (約4.0km/h)
        
        # ROS 2 トピック設定
        self.position_sub = self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.position_callback,
            10
        )
        
        self.safety_control_pub = self.create_publisher(
             ActuationCommandStamped,
            '/control/command/actuation_cmd',
            10
        )
        
        # 監視タイマー (0.5秒ごと)
        self.monitoring_timer = self.create_timer(0.5, self.monitor_position)
        
        self.get_logger().info("Route Deviation Safety Monitor started")
        if self.enable_viz:
            self.get_logger().info("Visualization enabled - saving route_status.png")
        
    def position_callback(self, msg: Odometry):
        self.current_position = {
            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'timestamp': time.time()
        }
        
        # 速度情報を更新
        linear_vel = msg.twist.twist.linear
        self.current_velocity = (linear_vel.x**2 + linear_vel.y**2)**0.5
        
    def monitor_position(self):
        if self.current_position is None:
            return
            
        x = self.current_position['x']
        y = self.current_position['y']
        current_time = time.time()
        
        # lanelet内にいるかチェック
        is_in_lane = self.safety_monitor.is_in_any_lane(x, y)
        
        # 可視化 (フラグが有効な場合のみ)
        if self.enable_viz:
            self.position_count += 1
            if self.position_count % 4 == 0:  # 2秒ごと (0.5秒×4)
                try:
                    self.safety_monitor.visualize_position(x, y, is_in_lane)
                    self.get_logger().info(f"Visualization saved: position=({x:.1f}, {y:.1f}), safe={is_in_lane}")
                except Exception as e:
                    self.get_logger().warn(f"Visualization failed: {e}")
        
        if is_in_lane:
            # OK - 経路内
            if self.is_outside_route:
                self.get_logger().info("Vehicle returned to route")
                self.is_outside_route = False
                self.deviation_start_time = None
                self._check_safety_release_condition()
        else:
            # 経路外
            if not self.is_outside_route:
                # 初回逸脱検出
                self.is_outside_route = True
                self.deviation_start_time = current_time
                self.get_logger().error("Route deviation detected")
            else:
                # 継続的な逸脱
                deviation_duration = current_time - self.deviation_start_time
                if deviation_duration >= 3.0:  # 3秒経過
                    self._activate_safety_intervention()
    
    def _compute_accel_brake_control(self):
        """ブレーキ制御による目標速度達成"""
        # 速度誤差
        speed_error = self.target_speed_ms - self.current_velocity
        
        # 制御パラメータ
        brake_gain = 0.8    # ブレーキ応答ゲイン
        deadband = 0.1      # 制御不感帯 [m/s]
        
        # ブレーキ値の初期化
        brake_cmd = 0.0
        
        if abs(speed_error) < deadband:
            # 目標速度に近い場合：制御しない
            brake_cmd = 0.0
            
        elif speed_error > 0:
            # 現在速度 < 目標速度：制御不要
            brake_cmd = 0.0
            
        else:
            # 現在速度 > 目標速度：減速が必要
            if abs(speed_error) > 1.0:
                # 大幅に速い場合：強めのブレーキ
                brake_cmd = max(0.0, min(0.8, brake_gain * abs(speed_error)))
            else:
                # 軽微なブレーキ
                brake_cmd = max(0.0, min(0.4, brake_gain * abs(speed_error) * 0.5))
        
        return brake_cmd
    
    def _activate_safety_intervention(self):
        """安全介入を開始（300Hz制御開始）"""
        if not self.safety_control_active:
            self.safety_control_active = True
            self.safety_mode_active = True
            
            # 300Hz制御タイマーを開始
            self.safety_control_timer = self.create_timer(0.00333, self._execute_safety_control)  # 300Hz
            
            self.get_logger().error("SAFETY INTERVENTION ACTIVATED - Starting 300Hz control")
    
    def _check_safety_release_condition(self):
        """安全制御解除条件をチェック"""
        if self.safety_control_active:
            # 条件1: 経路内復帰 AND 速度が閾値以下
            if not self.is_outside_route and self.current_velocity <= self.low_speed_threshold:
                self._deactivate_safety_intervention()
    
    def _deactivate_safety_intervention(self):
        """安全介入を停止"""
        if self.safety_control_active:
            self.safety_control_active = False
            self.safety_mode_active = False
            
            # 300Hz制御タイマーを停止
            if self.safety_control_timer is not None:
                self.safety_control_timer.cancel()
                self.destroy_timer(self.safety_control_timer)
                self.safety_control_timer = None
    
    def _execute_safety_control(self):
        # 安全制御が非活性の場合は何もしない
        if not self.safety_control_active:
            return
            
        # 安全制御解除条件をチェック
        self._check_safety_release_condition()
        
        # 現在速度が目標速度以下の場合は制御しない
        if self.current_velocity <= self.target_speed_ms:
            return
        
        # アクセル・ブレーキ制御計算
        brake_cmd = self._compute_accel_brake_control()
        
        # 制御コマンド生成（ActuationCommandStamped形式）
        safety_cmd = ActuationCommandStamped()
        safety_cmd.header.stamp = self.get_clock().now().to_msg()
        safety_cmd.header.frame_id = "base_link"
        
        # アクチュエーション制御値を設定
        safety_cmd.actuation.accel_cmd = 0.0
        safety_cmd.actuation.brake_cmd = brake_cmd
        safety_cmd.actuation.steer_cmd = 0.0  # 安全制御時は直進
        
        self.safety_control_pub.publish(safety_cmd)
        
        # ログ出力（300Hzでは制限）
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 0
            
        # 0.1Hzでログ出力（3000回に1回 = 10秒に1回）
        if self._log_counter % 3000 == 0:
            speed_error = self.target_speed_ms - self.current_velocity
            self.get_logger().info(
                f"[300Hz] Accel/Brake Control: vel={self.current_velocity:.2f}m/s, "
                f"target={self.target_speed_ms:.2f}m/s, "
                f"brake={brake_cmd:.3f}"
            )

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RouteDeviationSafetyMonitorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == "__main__":
    main()
