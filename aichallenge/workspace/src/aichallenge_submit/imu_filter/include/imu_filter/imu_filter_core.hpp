// Copyright 2024 AI Challenge Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IMU_FILTER__IMU_FILTER_CORE_HPP_
#define IMU_FILTER__IMU_FILTER_CORE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>

#include <deque>
#include <memory>
#include <string>

namespace imu_filter
{

/**
 * @brief IMUデータの高度なフィルタリングを行うノード
 * 
 * このノードは以下の機能を提供します：
 * 1. 外れ値（ノイズスパイク）の検出と除去
 * 2. 移動平均による平滑化
 * 3. 停止時の自動キャリブレーション
 * 4. 加速度共分散の動的調整
 * 5. IMU方位角の信頼度低減
 */
class ImuFilter : public rclcpp::Node
{
public:
  explicit ImuFilter(const rclcpp::NodeOptions & node_options);

private:
  // コールバック関数
  void callbackImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg_ptr);
  void callbackVelocity(
    const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr velocity_msg_ptr);

  // フィルタリング関数
  bool isOutlier(const sensor_msgs::msg::Imu & imu_msg);
  sensor_msgs::msg::Imu applyMovingAverage(const sensor_msgs::msg::Imu & imu_msg);
  void updateCalibration(const sensor_msgs::msg::Imu & imu_msg);
  sensor_msgs::msg::Imu applyCalibration(const sensor_msgs::msg::Imu & imu_msg);
  void adjustCovariance(sensor_msgs::msg::Imu & imu_msg);
  void reduceOrientationConfidence(sensor_msgs::msg::Imu & imu_msg);

  // ヘルパー関数
  bool isVehicleStationary() const;
  double calculateStdDev(const std::deque<double> & data) const;
  double calculateMean(const std::deque<double> & data) const;

  // ROS通信
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr velocity_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;

  // データバッファ（移動窓として使用）
  std::deque<sensor_msgs::msg::Imu> imu_buffer_;
  sensor_msgs::msg::Imu last_valid_imu_;  // 外れ値検出時の補間用
  bool has_valid_imu_;

  // キャリブレーション用データ
  std::deque<double> calib_gyro_x_;
  std::deque<double> calib_gyro_y_;
  std::deque<double> calib_gyro_z_;
  double gyro_bias_x_;
  double gyro_bias_y_;
  double gyro_bias_z_;
  bool is_calibrated_;

  // 速度情報（停止判定用）
  double current_velocity_;
  bool velocity_received_;

  // パラメータ（yamlファイルから読み込み）
  // === 外れ値検出パラメータ ===
  int window_size_;           // 移動窓のサイズ（サンプル数）
  double outlier_threshold_;  // 外れ値判定の閾値（標準偏差の何倍か）
  
  // === 移動平均フィルタパラメータ ===
  int moving_average_size_;   // 移動平均の窓サイズ
  
  // === キャリブレーションパラメータ ===
  double stationary_velocity_threshold_;  // 停止判定の速度閾値 [m/s]
  int calibration_sample_size_;           // キャリブレーションに必要なサンプル数
  bool enable_auto_calibration_;          // 自動キャリブレーションの有効/無効
  
  // === 共分散パラメータ ===
  double acceleration_stddev_;            // 加速度の標準偏差 [m/s^2]
  double angular_velocity_stddev_;        // 角速度の標準偏差 [rad/s]
  double orientation_covariance_scale_;   // 方位角の共分散スケール（大きいほど信頼度低）
};

}  // namespace imu_filter

#endif  // IMU_FILTER__IMU_FILTER_CORE_HPP_
