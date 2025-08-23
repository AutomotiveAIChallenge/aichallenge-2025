#!/usr/bin/env python3

import rclpy
import rclpy.node
import time
import xml.etree.ElementTree as ET
from matplotlib import path
import matplotlib.pyplot as plt
import os
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Bool

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
        self.deviation_start_time = None
        self.position_count = 0
        self.emergency_sent = False
        
        # ROS 2 トピック設定
        self.position_sub = self.create_subscription(
            Odometry,
            '/localization/kinematic_state',
            self.position_callback,
            10
        )
        
        self.safety_control_pub = self.create_publisher(
            Bool,
            '/vehicle/emergency/is_route_deviation',
            10
        )
        
        # 監視タイマー (0.5秒ごと)
        self.monitoring_timer = self.create_timer(0.5, self.monitor_position)
        
        self.get_logger().debug("Route Deviation Safety Monitor started")
        if self.enable_viz:
            self.get_logger().debug("Visualization enabled - saving route_status.png")
        
    def position_callback(self, msg: Odometry):
        self.current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        
    def monitor_position(self):
        if not self.current_position:
            return
        
        x, y = self.current_position
        is_in_lane = self.safety_monitor.is_in_any_lane(x, y)
        current_time = time.time()
        
        # ルート逸脱監視
        if is_in_lane:
            if self.deviation_start_time is not None:
                self.get_logger().debug("Returned to lane")
            self.deviation_start_time = None
            self.emergency_sent = False
        elif not self.deviation_start_time:
            self.deviation_start_time = current_time
            self.emergency_sent = False
            self.get_logger().debug("Route deviation detected")
        elif current_time - self.deviation_start_time >= 3.0 and not self.emergency_sent:
            self.safety_control_pub.publish(Bool(data=True))
            self.emergency_sent = True
            self.get_logger().debug("Emergency signal sent - route deviation for 3+ seconds")
        
        # 可視化
        if self.enable_viz:
            self.position_count += 1
            if self.position_count % 4 == 0:
                try:
                    self.safety_monitor.visualize_position(x, y, is_in_lane)
                except Exception as e:
                    self.get_logger().warn(f"Visualization failed: {e}")

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
