import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time
from sensor_msgs.msg import LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand

from .model.tinylidarnet import TinyLidarNetNp, TinyLidarNetSmallNp


class TinyLidarNetNode(Node):
    def __init__(self):
        super().__init__('tiny_lidar_net_node')

        self.declare_parameter('log_interval_sec', 5.0)
        self.declare_parameter('model.input_dim', 1080)
        self.declare_parameter('model.output_dim', 2)
        self.declare_parameter('model.architecture', 'large')
        self.declare_parameter('model.ckpt_path', '')
        self.declare_parameter('acceleration', 0.1)
        self.declare_parameter('control_mode', 'ai')  # "ai" or "fixed"

        self.log_interval = self.get_parameter('log_interval_sec').value
        self.input_dim = self.get_parameter('model.input_dim').value
        self.output_dim = self.get_parameter('model.output_dim').value
        self.model_architecture = self.get_parameter('model.architecture').value
        self.ckpt_path = self.get_parameter('model.ckpt_path').value
        self.acceleration = self.get_parameter('acceleration').value
        self.control_mode = self.get_parameter('control_mode').value.lower()

        # --- モデル生成 ---
        if self.model_architecture == 'small':
            self.model = TinyLidarNetSmallNp(input_dim=self.input_dim, output_dim=self.output_dim)
        else:
            self.model = TinyLidarNetNp(input_dim=self.input_dim, output_dim=self.output_dim)

        # --- 重みロード ---
        if self.ckpt_path:
            try:
                self.load_weights(self.ckpt_path)
                self.get_logger().info(f"Loaded model weights from {self.ckpt_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load model weights: {e}")
        else:
            self.get_logger().warn("No weight file provided, using random initialized weights.")

        self.get_logger().info(
            f"NumPy model: {self.model_architecture} | mode={self.control_mode} | input_dim={self.input_dim}"
        )

        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST,
                         depth=1)
        self.create_subscription(LaserScan, "/scan", self.scan_callback, qos)
        self.control_pub = self.create_publisher(AckermannControlCommand, "/awsim/control_cmd", 1)
        self.get_logger().info("TinyLidarNetNode initialized (NumPy ver).")

    def load_weights(self, path):
        """NumPyで保存された重みファイル (.npy または .npz) をロード"""
        weights = np.load(path, allow_pickle=True)
        if isinstance(weights, np.lib.npyio.NpzFile):
            weights = dict(weights.items())
        elif isinstance(weights, np.ndarray) and weights.dtype == object:
            weights = weights.item()   #
        elif isinstance(weights, dict):
            pass
        else:
            raise ValueError(f"Unsupported weight format type: {type(weights)}")

        count = 0
        for k, v in weights.items():
            k_norm = k.replace('.', '_')
            if k_norm in self.model.params:
                if self.model.params[k_norm].shape == v.shape:
                    self.model.params[k_norm] = v
                    count += 1
                else:
                    self.get_logger().warn(
                        f"Shape mismatch for {k_norm}: expected {self.model.params[k_norm].shape}, got {v.shape}"
                    )
            else:
                self.get_logger().warn(f"Unused weight key: {k_norm}")

        self.get_logger().info(f"Loaded {count} parameters from {path}")


    def scan_callback(self, msg):
        start_time = time.monotonic()
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = 0.0

        processed = self.process_ranges_for_model(ranges)
        x = np.expand_dims(np.expand_dims(processed, axis=0), axis=1)  # (1, 1, 1080)

        outputs = self.model(x)[0]  # shape=(2,) expected
        if self.control_mode == "ai":
            accel = float(np.clip(outputs[0], -1.0, 1.0))
        else:
            accel = self.acceleration
        steer = float(np.clip(outputs[1], -1.0, 1.0))

        duration_ms = (time.monotonic() - start_time) * 1000.0
        self.inference_times.append(duration_ms)

        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.acceleration = accel
        cmd.lateral.steering_tire_angle = steer
        self.control_pub.publish(cmd)

        self.get_logger().info(
            f"[{self.control_mode.upper()}-{self.model_architecture}] accel={accel:.3f}, steer={steer:.3f}",
            throttle_duration_sec=1.0
        )

        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds / 1e9 > self.log_interval:
            if self.inference_times:
                avg = np.mean(self.inference_times)
                self.get_logger().info(
                    f"Avg {avg:.2f}ms ({1000.0/avg:.2f}Hz), Max {np.max(self.inference_times):.2f}ms, N={len(self.inference_times)}"
                )
                self.inference_times.clear()
            self.last_log_time = now

    def process_ranges_for_model(self, ranges):
        if len(ranges) > self.input_dim:
            idx = np.linspace(0, len(ranges) - 1, self.input_dim, dtype=int)
            ranges = ranges[idx]
        elif len(ranges) < self.input_dim:
            ranges = np.pad(ranges, (0, self.input_dim - len(ranges)), 'constant')
        return ranges / 30.0


def main(args=None):
    rclpy.init(args=args)
    node = TinyLidarNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
