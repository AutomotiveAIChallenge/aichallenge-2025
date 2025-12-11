#!/usr/bin/env python3
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from autoware_auto_control_msgs.msg import AckermannControlCommand

from tiny_lidar_net_controller.model.tinylidarnet import TinyLidarNetNp, TinyLidarNetSmallNp


class TinyLidarNetNode(Node):
    """
    ROS 2 node for autonomous driving control using TinyLidarNet (NumPy version).
    Takes LaserScan data as input and outputs AckermannControlCommand.
    """

    def __init__(self):
        super().__init__('tiny_lidar_net_node')

        # --- Parameter Declaration ---
        self.declare_parameter('log_interval_sec', 5.0)
        self.declare_parameter('model.input_dim', 1080)
        self.declare_parameter('model.output_dim', 2)
        self.declare_parameter('model.architecture', 'large')  # 'large' or 'small'
        self.declare_parameter('model.ckpt_path', '')
        self.declare_parameter('acceleration', 0.1)
        self.declare_parameter('scan_topic', '/sensing/lidar/pointcloud')
        self.declare_parameter('control_mode', 'ai')  # 'ai' or 'fixed'
        self.declare_parameter('debug', False)        # True/False

        # --- Parameter Retrieval ---
        self.log_interval = self.get_parameter('log_interval_sec').value
        self.input_dim = self.get_parameter('model.input_dim').value
        self.output_dim = self.get_parameter('model.output_dim').value
        self.model_architecture = self.get_parameter('model.architecture').value
        self.ckpt_path = self.get_parameter('model.ckpt_path').value
        self.acceleration = self.get_parameter('acceleration').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.control_mode = self.get_parameter('control_mode').value.lower()
        self.debug = self.get_parameter('debug').value

        # --- Model Initialization ---
        if self.model_architecture == 'small':
            self.model = TinyLidarNetSmallNp(input_dim=self.input_dim, output_dim=self.output_dim)
        else:
            self.model = TinyLidarNetNp(input_dim=self.input_dim, output_dim=self.output_dim)

        # --- Load Weights ---
        if self.ckpt_path:
            try:
                self.load_weights(self.ckpt_path)
                self.get_logger().info(f"Loaded model weights from {self.ckpt_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load model weights: {e}")
        else:
            self.get_logger().warn("No weight file provided. Using randomly initialized weights.")

        self.get_logger().info(
            f"Model: {self.model_architecture} | Mode: {self.control_mode} | Debug: {self.debug}"
        )

        # --- Metrics & Subscribers ---
        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        # QoS configuration for sensor data (High throughput, keep latest)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.create_subscription(PointCloud2, self.scan_topic, self.pointcloud_callback, qos)
        self.control_pub = self.create_publisher(AckermannControlCommand, "/awsim/control_cmd", 1)

        self.get_logger().info("TinyLidarNetNode initialized.")

    def load_weights(self, path: str):
        """
        Loads weights from a .npy or .npz file into the model parameters.
        Handles dictionary key mapping (replacing '.' with '_').
        """
        weights = np.load(path, allow_pickle=True)
        
        # Normalize weight format to dict
        if isinstance(weights, np.lib.npyio.NpzFile):
            weights = dict(weights.items())
        elif isinstance(weights, np.ndarray) and weights.dtype == object:
            weights = weights.item()
        elif isinstance(weights, dict):
            pass
        else:
            raise ValueError(f"Unsupported weight format type: {type(weights)}")

        count = 0
        for key, value in weights.items():
            # Normalize key to match model parameter naming convention
            key_norm = key.replace('.', '_')
            
            if key_norm in self.model.params:
                target_shape = self.model.params[key_norm].shape
                if target_shape == value.shape:
                    self.model.params[key_norm] = value
                    count += 1
                else:
                    self.get_logger().warn(
                        f"Shape mismatch for {key_norm}: expected {target_shape}, got {value.shape}"
                    )
            else:
                self.get_logger().debug(f"Unused weight key: {key_norm}")

        self.get_logger().info(f"Successfully loaded {count} parameters from {path}")

    def pointcloud_callback(self, msg: PointCloud2):
        """
        Converts PointCloud2 (RobotecGPULidar) to LaserScan-like ranges and reuses the existing pipeline.
        """
        angles = []
        distances = []
        for x, y, z in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            dist = math.hypot(x, y)
            if dist == 0.0:
                continue
            angles.append(math.atan2(y, x))
            distances.append(dist)

        if not distances:
            self.get_logger().warn("PointCloud2 had no valid points; skipping inference.")
            return

        ordered = sorted(zip(angles, distances), key=lambda t: t[0])
        angle_min = ordered[0][0]
        angle_max = ordered[-1][0]
        count = len(ordered)
        angle_increment = (angle_max - angle_min) / (count - 1) if count > 1 else 0.0

        scan_msg = LaserScan()
        scan_msg.header = msg.header
        scan_msg.angle_min = float(angle_min)
        scan_msg.angle_max = float(angle_max)
        scan_msg.angle_increment = float(angle_increment)
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.0
        scan_msg.range_min = float(min(distances))
        scan_msg.range_max = float(max(distances))
        scan_msg.ranges = [d for _, d in ordered]

        self.scan_callback(scan_msg)

    def scan_callback(self, msg: LaserScan):
        """
        Callback for LaserScan messages. 
        Preprocesses data, runs inference, and publishes control commands.
        """
        start_time = time.monotonic()
        
        # 1. Preprocess LiDAR data
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = 0.0

        processed_ranges = self.process_ranges_for_model(ranges)
        
        # Prepare input tensor: (1, 1, input_dim)
        x = np.expand_dims(np.expand_dims(processed_ranges, axis=0), axis=1)

        # 2. Inference
        outputs = self.model(x)[0]  # Expected output shape: (2,) -> [acceleration, steering]
        
        # 3. Process Control Command
        if self.control_mode == "ai":
            accel = float(np.clip(outputs[0], -1.0, 1.0))
        else:
            accel = self.acceleration

        steer = float(np.clip(outputs[1], -1.0, 1.0))

        # 4. Measure Latency (only if debug is enabled to save resources)
        if self.debug:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            self.inference_times.append(duration_ms)

        # 5. Publish Command
        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.acceleration = accel
        cmd.lateral.steering_tire_angle = steer
        self.control_pub.publish(cmd)

        # 6. Debug Logging
        if self.debug:
            self.get_logger().info(
                f"[{self.control_mode.upper()}-{self.model_architecture}] accel={accel:.3f}, steer={steer:.3f}",
                throttle_duration_sec=1.0
            )
            self._log_performance_metrics()

    def process_ranges_for_model(self, ranges: np.ndarray) -> np.ndarray:
        """
        Resizes and normalizes the LiDAR ranges to match the model's input dimension.
        Normalization factor: 30.0 (Hardcoded max range assumption)
        """
        current_len = len(ranges)
        
        if current_len > self.input_dim:
            # Downsample (e.g., 18000 -> 1080) with linear interpolation to evenly spread samples.
            target_pos = np.linspace(0, current_len - 1, self.input_dim, dtype=np.float32)
            ranges = np.interp(target_pos, np.arange(current_len), ranges).astype(np.float32)
        elif current_len < self.input_dim:
            ranges = np.pad(ranges, (0, self.input_dim - current_len), 'constant')
            
        return ranges / 30.0

    def _log_performance_metrics(self):
        """Logs average inference time at specified intervals."""
        now = self.get_clock().now()
        
        # Only log if enough time has passed
        if (now - self.last_log_time).nanoseconds / 1e9 > self.log_interval:
            if self.inference_times:
                avg = np.mean(self.inference_times)
                max_time = np.max(self.inference_times)
                fps = 1000.0 / avg if avg > 0 else 0.0
                
                self.get_logger().info(
                    f"DEBUG Stats: Avg {avg:.2f}ms ({fps:.2f}Hz), "
                    f"Max {max_time:.2f}ms, Samples={len(self.inference_times)}"
                )
                self.inference_times.clear()
            self.last_log_time = now


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
