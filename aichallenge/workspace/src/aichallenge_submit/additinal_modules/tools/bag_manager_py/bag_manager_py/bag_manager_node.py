import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger 
import subprocess
import datetime
import signal
import os
import threading

class RosBagManagerNode(Node):
    def __init__(self):
        super().__init__('ros2_bag_manager_node')

        # === パラメータ宣言・取得 ===
        self.declare_parameter('output_dir', 'rosbag2_output')
        self.declare_parameter('all_topics', True)
        self.declare_parameter('topics', ['/rosbag2_recorder/trigger'])
        self.declare_parameter('storage_id', 'mcap') 
        
        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.all_topics = self.get_parameter('all_topics').get_parameter_value().bool_value
        self.topics = list(self.get_parameter('topics').get_parameter_value().string_array_value)
        self.storage_id = self.get_parameter('storage_id').get_parameter_value().string_value

        now = datetime.datetime.now()
        session_ts = now.strftime('%Y%m%d_%H%M%S')
        self.session_dir = os.path.join(self.output_dir, session_ts)
        os.makedirs(self.session_dir, exist_ok=True)
        self.get_logger().info(f"セッションディレクトリ作成: {self.session_dir}")

        self.trigger_sub = self.create_subscription(
            Bool,
            '/rosbag2_recorder/trigger',
            self.trigger_callback,
            10
        )

        self.status_pub = self.create_publisher(Bool, '~/status', 10)

        self.start_srv = self.create_service(
            Trigger, 
            '~/start_recording', 
            self.start_recording_callback
        )
        self.stop_srv = self.create_service(
            Trigger, 
            '~/stop_recording', 
            self.stop_recording_callback
        )

        self.recording_process = None
        self.is_recording = False
        self.lock = threading.Lock() # 排他制御用ロック
        
        # 起動時に現在のステータス (False) を一度発行する
        self.status_pub.publish(Bool(data=self.is_recording))

    def start_recording_callback(self, request, response):
        """録画開始サービス コールバック"""
        success, message = self._start_recording_logic()
        response.success = success
        response.message = message
        return response

    def stop_recording_callback(self, request, response):
        """録画停止サービス コールバック"""
        success, message = self._stop_recording_logic()
        response.success = success
        response.message = message
        return response

    # --- トピックコールバック (Joyトリガー用) ---

    def trigger_callback(self, msg: Bool):
        """録画トリガートピック コールバック"""
        if msg.data:
            # 録画開始
            success, message = self._start_recording_logic()
            if not success:
                self.get_logger().warn(f"トリガーによる録画開始失敗: {message}")
        else:
            # 録画停止
            success, message = self._stop_recording_logic()
            if not success:
                self.get_logger().warn(f"トリガーによる録画停止失敗: {message}")

    # --- 内部ロジック (共通化) ---

    def _start_recording_logic(self):
        """録画開始の共通ロジック (排他制御あり)"""
        with self.lock:
            if self.is_recording:
                self.get_logger().warn("すでに録画中です。新しい録画は開始しません。")
                return False, "Already recording"

            # === 録画開始処理 ===
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            record_dir = os.path.join(self.session_dir, ts)
            
            cmd = ['ros2', 'bag', 'record']
            if self.all_topics:
                cmd.append('-a')
            elif self.topics:
                cmd.extend(self.topics)
            else:
                self.get_logger().warn("all_topics=false かつ topics=[] のため、何も録画しません")
                return False, "No topics to record"

            cmd.extend(['-o', record_dir, '-s', self.storage_id])

            command_str = ' '.join(cmd)
            self.get_logger().info(f"[DEBUG] Command to be executed: {command_str}")

            self.get_logger().info(f"録画開始: {command_str}")
            self.recording_process = subprocess.Popen(cmd, preexec_fn=os.setsid)
            self.is_recording = True
    
            self.status_pub.publish(Bool(data=True))
            
            return True, f"Recording started: {record_dir}"

    def _stop_recording_logic(self):
        """録画停止の共通ロジック (排他制御あり)"""
        with self.lock:
            if not self.is_recording:
                self.get_logger().warn("録画は現在行われていません。停止要求は無視されます。")
                return False, "Not recording"

            # === 録画停止処理 ===
            self.get_logger().info("録画停止要求 — SIGINT を送信します")
            self._cleanup() 
            
            return True, "Recording stopped"

    def _cleanup(self):
        """録画プロセスが生きていたら SIGINT で停止し、wait する"""
        if self.recording_process:
            try:
                os.killpg(os.getpgid(self.recording_process.pid), signal.SIGINT)
                self.recording_process.wait(timeout=5)
            except Exception as e:
                self.get_logger().error(f"録画プロセス停止中にエラー: {e}")
            finally:
                self.recording_process = None
                self.is_recording = False
                
                self.status_pub.publish(Bool(data=False))
                self.get_logger().info("録画を停止しました")

    def destroy_node(self):
        self.get_logger().info("ノードシャットダウン: 録画プロセス停止処理を実行します")
        with self.lock: 
            if self.is_recording:
                self._cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RosBagManagerNode()
    
    def _signal_handler(signum, frame):
        node.get_logger().info(f"シグナル {signal.Signals(signum).name} 受信 — シャットダウンします")
        if rclpy.ok():
            node.destroy_node()
        rclpy.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _signal_handler)

    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"rclpy.spin() 中に予期せぬエラー: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()