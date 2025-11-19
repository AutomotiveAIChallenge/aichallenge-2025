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

#include "imu_filter/imu_filter_core.hpp"
#include <cmath>
#include <algorithm>

namespace imu_filter
{

ImuFilter::ImuFilter(const rclcpp::NodeOptions & node_options)
: Node("imu_filter", node_options),
  has_valid_imu_(false),
  gyro_bias_x_(0.0),
  gyro_bias_y_(0.0),
  gyro_bias_z_(0.0),
  is_calibrated_(false),
  current_velocity_(0.0),
  velocity_received_(false)
{
  // ==========================================
  // パラメータ読み込み
  // ==========================================
  
  // --- 外れ値検出パラメータ ---
  // window_size_: 外れ値判定に使う過去データの数
  // 大きくする → より多くのデータで判断（安定するが遅延増加）
  // 小さくする → 高速だが誤検出の可能性増加
  window_size_ = declare_parameter<int>("window_size", 10);
  
  // outlier_threshold_: 標準偏差の何倍を外れ値とするか
  // 大きくする → 外れ値判定が厳しくなり、正常値も残りやすい
  // 小さくする → 外れ値を積極的に除去（過剰除去に注意）
  // 推奨値: 3.0（3シグマルール）
  outlier_threshold_ = declare_parameter<double>("outlier_threshold", 3.0);
  
  // --- 移動平均フィルタパラメータ ---
  // moving_average_size_: 平滑化に使うサンプル数
  // 大きくする → より滑らかになるが応答が遅くなる
  // 小さくする → 応答が速いがノイズが残りやすい
  // 推奨値: 3-5（レーシングカートの高速応答に対応）
  moving_average_size_ = declare_parameter<int>("moving_average_size", 3);
  
  // --- キャリブレーションパラメータ ---
  // stationary_velocity_threshold_: この速度以下を停止とみなす [m/s]
  // 大きくする → 低速走行中もキャリブレーション（精度低下のリスク）
  // 小さくする → 完全停止時のみキャリブレーション（機会減少）
  // 推奨値: 0.1 m/s
  stationary_velocity_threshold_ = declare_parameter<double>(
    "stationary_velocity_threshold", 0.1);
  
  // calibration_sample_size_: キャリブレーションに必要なサンプル数
  // 大きくする → より正確なバイアス推定（キャリブレーション時間増加）
  // 小さくする → 高速キャリブレーション（精度低下）
  // 推奨値: 100サンプル（約3秒 @ 30Hz）
  calibration_sample_size_ = declare_parameter<int>("calibration_sample_size", 100);
  
  // enable_auto_calibration_: 自動キャリブレーションの有効化
  // true  → 停止時に自動でバイアス補正
  // false → 固定バイアス値を使用
  enable_auto_calibration_ = declare_parameter<bool>("enable_auto_calibration", true);
  
  // --- 共分散パラメータ ---
  // acceleration_stddev_: 加速度データの信頼性（標準偏差） [m/s^2]
  // 大きくする → 加速度データへの信頼度を下げる（他のセンサを優先）
  // 小さくする → 加速度データを重視（IMUの精度が高い場合）
  // 推奨値: 1.25（ノイズ解析結果に基づく）
  acceleration_stddev_ = declare_parameter<double>("acceleration_stddev", 1.25);
  
  // angular_velocity_stddev_: 角速度データの信頼性（標準偏差） [rad/s]
  // 大きくする → 角速度データへの信頼度を下げる
  // 小さくする → 角速度データを重視
  // 推奨値: 0.03（既存設定を踏襲）
  angular_velocity_stddev_ = declare_parameter<double>("angular_velocity_stddev", 0.03);
  
  // orientation_covariance_scale_: IMU方位角の信頼度を下げるスケール
  // 大きくする → 方位角を信用しない（推奨：実車では方位角が不正確）
  // 小さくする → 方位角を信用する
  // 推奨値: 1000.0（実質的に方位角を無視）
  orientation_covariance_scale_ = declare_parameter<double>(
    "orientation_covariance_scale", 1000.0);

  // ==========================================
  // ROS通信の初期化
  // ==========================================
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
  
  // IMUデータのサブスクライバー
  imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
    "input", qos,
    std::bind(&ImuFilter::callbackImu, this, std::placeholders::_1));
  
  // 速度データのサブスクライバー（停止判定用）
  velocity_sub_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
    "velocity", qos,
    std::bind(&ImuFilter::callbackVelocity, this, std::placeholders::_1));
  
  // フィルタリング後のIMUデータのパブリッシャー
  imu_pub_ = create_publisher<sensor_msgs::msg::Imu>("output", qos);

  RCLCPP_INFO(this->get_logger(), "IMU Filter node initialized");
  RCLCPP_INFO(this->get_logger(), "  Window size: %d", window_size_);
  RCLCPP_INFO(this->get_logger(), "  Outlier threshold: %.1f sigma", outlier_threshold_);
  RCLCPP_INFO(this->get_logger(), "  Moving average size: %d", moving_average_size_);
  RCLCPP_INFO(this->get_logger(), "  Auto calibration: %s",
    enable_auto_calibration_ ? "enabled" : "disabled");
}

// ==========================================
// コールバック関数
// ==========================================

void ImuFilter::callbackImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg_ptr)
{
  sensor_msgs::msg::Imu filtered_imu = *imu_msg_ptr;
  
  // ステップ1: 外れ値チェック
  // 急激な変化（ノイズスパイク）を検出
  if (isOutlier(filtered_imu)) {
    if (has_valid_imu_) {
      // 外れ値の場合、前回の正常値を使用（線形補間も可能）
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 1000,
        "Outlier detected! Using previous valid value.");
      filtered_imu.linear_acceleration = last_valid_imu_.linear_acceleration;
      filtered_imu.angular_velocity = last_valid_imu_.angular_velocity;
    }
  } else {
    // 正常値の場合、バッファに追加
    imu_buffer_.push_back(filtered_imu);
    if (imu_buffer_.size() > static_cast<size_t>(window_size_)) {
      imu_buffer_.pop_front();
    }
    last_valid_imu_ = filtered_imu;
    has_valid_imu_ = true;
  }
  
  // ステップ2: 移動平均による平滑化
  // 高周波ノイズを除去
  if (imu_buffer_.size() >= static_cast<size_t>(moving_average_size_)) {
    filtered_imu = applyMovingAverage(filtered_imu);
  }
  
  // ステップ3: 自動キャリブレーション
  // 停止時にジャイロバイアスを推定・補正
  if (enable_auto_calibration_ && isVehicleStationary()) {
    updateCalibration(filtered_imu);
  }
  if (is_calibrated_) {
    filtered_imu = applyCalibration(filtered_imu);
  }
  
  // ステップ4: 共分散の調整
  // センサの信頼性を適切に設定
  adjustCovariance(filtered_imu);
  
  // ステップ5: 方位角の信頼度を下げる
  // 実車では方位角が不正確なため
  reduceOrientationConfidence(filtered_imu);
  
  // フィルタリング後のデータをパブリッシュ
  imu_pub_->publish(filtered_imu);
}

void ImuFilter::callbackVelocity(
  const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr velocity_msg_ptr)
{
  // 車両速度を記録（停止判定に使用）
  current_velocity_ = std::abs(velocity_msg_ptr->twist.twist.linear.x);
  velocity_received_ = true;
}

// ==========================================
// フィルタリング関数
// ==========================================

bool ImuFilter::isOutlier(const sensor_msgs::msg::Imu & imu_msg)
{
  // バッファが十分に溜まっていない場合は正常値として扱う
  if (imu_buffer_.size() < static_cast<size_t>(window_size_)) {
    return false;
  }
  
  // 加速度の各軸について外れ値判定
  // 過去のデータから標準偏差を計算し、閾値を超えた場合は外れ値
  std::deque<double> accel_x_history, accel_y_history, accel_z_history;
  for (const auto & msg : imu_buffer_) {
    accel_x_history.push_back(msg.linear_acceleration.x);
    accel_y_history.push_back(msg.linear_acceleration.y);
    accel_z_history.push_back(msg.linear_acceleration.z);
  }
  
  double mean_x = calculateMean(accel_x_history);
  double mean_y = calculateMean(accel_y_history);
  double mean_z = calculateMean(accel_z_history);
  
  double std_x = calculateStdDev(accel_x_history);
  double std_y = calculateStdDev(accel_y_history);
  double std_z = calculateStdDev(accel_z_history);
  
  // 現在の値が平均から標準偏差×閾値以上離れていたら外れ値
  bool is_x_outlier = std::abs(imu_msg.linear_acceleration.x - mean_x) >
                      outlier_threshold_ * std_x;
  bool is_y_outlier = std::abs(imu_msg.linear_acceleration.y - mean_y) >
                      outlier_threshold_ * std_y;
  bool is_z_outlier = std::abs(imu_msg.linear_acceleration.z - mean_z) >
                      outlier_threshold_ * std_z;
  
  // いずれかの軸で外れ値が検出された場合
  return is_x_outlier || is_y_outlier || is_z_outlier;
}

sensor_msgs::msg::Imu ImuFilter::applyMovingAverage(
  const sensor_msgs::msg::Imu & imu_msg)
{
  sensor_msgs::msg::Imu averaged_imu = imu_msg;
  
  // バッファが十分にない場合は平均化せずそのまま返す
  if (imu_buffer_.size() < static_cast<size_t>(moving_average_size_)) {
    return averaged_imu;
  }
  
  // 最新のN個のデータで移動平均を計算
  double sum_accel_x = 0.0, sum_accel_y = 0.0, sum_accel_z = 0.0;
  double sum_gyro_x = 0.0, sum_gyro_y = 0.0, sum_gyro_z = 0.0;
  
  auto it = imu_buffer_.rbegin();  // 最新データから
  for (int i = 0; i < moving_average_size_ && it != imu_buffer_.rend(); ++i, ++it) {
    sum_accel_x += it->linear_acceleration.x;
    sum_accel_y += it->linear_acceleration.y;
    sum_accel_z += it->linear_acceleration.z;
    sum_gyro_x += it->angular_velocity.x;
    sum_gyro_y += it->angular_velocity.y;
    sum_gyro_z += it->angular_velocity.z;
  }
  
  // 平均値を算出
  int n = moving_average_size_;
  averaged_imu.linear_acceleration.x = sum_accel_x / n;
  averaged_imu.linear_acceleration.y = sum_accel_y / n;
  averaged_imu.linear_acceleration.z = sum_accel_z / n;
  averaged_imu.angular_velocity.x = sum_gyro_x / n;
  averaged_imu.angular_velocity.y = sum_gyro_y / n;
  averaged_imu.angular_velocity.z = sum_gyro_z / n;
  
  return averaged_imu;
}

void ImuFilter::updateCalibration(const sensor_msgs::msg::Imu & imu_msg)
{
  // 停止時のジャイロデータを蓄積
  calib_gyro_x_.push_back(imu_msg.angular_velocity.x);
  calib_gyro_y_.push_back(imu_msg.angular_velocity.y);
  calib_gyro_z_.push_back(imu_msg.angular_velocity.z);
  
  // 古いデータは削除（最新N個のみ保持）
  if (calib_gyro_x_.size() > static_cast<size_t>(calibration_sample_size_)) {
    calib_gyro_x_.pop_front();
    calib_gyro_y_.pop_front();
    calib_gyro_z_.pop_front();
  }
  
  // 十分なサンプルが集まったらバイアスを計算
  if (calib_gyro_x_.size() >= static_cast<size_t>(calibration_sample_size_)) {
    gyro_bias_x_ = calculateMean(calib_gyro_x_);
    gyro_bias_y_ = calculateMean(calib_gyro_y_);
    gyro_bias_z_ = calculateMean(calib_gyro_z_);
    
    if (!is_calibrated_) {
      RCLCPP_INFO(this->get_logger(),
        "Calibration completed: bias[x=%.4f, y=%.4f, z=%.4f] rad/s",
        gyro_bias_x_, gyro_bias_y_, gyro_bias_z_);
      is_calibrated_ = true;
    }
  }
}

sensor_msgs::msg::Imu ImuFilter::applyCalibration(
  const sensor_msgs::msg::Imu & imu_msg)
{
  sensor_msgs::msg::Imu calibrated_imu = imu_msg;
  
  // ジャイロバイアスを補正
  calibrated_imu.angular_velocity.x -= gyro_bias_x_;
  calibrated_imu.angular_velocity.y -= gyro_bias_y_;
  calibrated_imu.angular_velocity.z -= gyro_bias_z_;
  
  return calibrated_imu;
}

void ImuFilter::adjustCovariance(sensor_msgs::msg::Imu & imu_msg)
{
  // 加速度の共分散（対角成分のみ設定）
  // より大きい値 = センサの信頼性が低い
  imu_msg.linear_acceleration_covariance[0] = 
    acceleration_stddev_ * acceleration_stddev_;  // X軸
  imu_msg.linear_acceleration_covariance[4] = 
    acceleration_stddev_ * acceleration_stddev_;  // Y軸
  imu_msg.linear_acceleration_covariance[8] = 
    acceleration_stddev_ * acceleration_stddev_;  // Z軸
  
  // 角速度の共分散（対角成分のみ設定）
  imu_msg.angular_velocity_covariance[0] = 
    angular_velocity_stddev_ * angular_velocity_stddev_;  // X軸
  imu_msg.angular_velocity_covariance[4] = 
    angular_velocity_stddev_ * angular_velocity_stddev_;  // Y軸
  imu_msg.angular_velocity_covariance[8] = 
    angular_velocity_stddev_ * angular_velocity_stddev_;  // Z軸
}

void ImuFilter::reduceOrientationConfidence(sensor_msgs::msg::Imu & imu_msg)
{
  // IMU方位角の共分散を大きくして信頼度を下げる
  // 実車走行では方位角が不正確なため、他のセンサ（GNSS等）を優先させる
  for (int i = 0; i < 9; ++i) {
    imu_msg.orientation_covariance[i] *= orientation_covariance_scale_;
  }
}

// ==========================================
// ヘルパー関数
// ==========================================

bool ImuFilter::isVehicleStationary() const
{
  // 速度データが受信されていない場合は停止していないと判定
  if (!velocity_received_) {
    return false;
  }
  
  // 速度が閾値以下なら停止とみなす
  return current_velocity_ < stationary_velocity_threshold_;
}

double ImuFilter::calculateStdDev(const std::deque<double> & data) const
{
  if (data.empty()) {
    return 0.0;
  }
  
  double mean = calculateMean(data);
  double sum_sq_diff = 0.0;
  
  for (const auto & val : data) {
    double diff = val - mean;
    sum_sq_diff += diff * diff;
  }
  
  return std::sqrt(sum_sq_diff / data.size());
}

double ImuFilter::calculateMean(const std::deque<double> & data) const
{
  if (data.empty()) {
    return 0.0;
  }
  
  double sum = 0.0;
  for (const auto & val : data) {
    sum += val;
  }
  
  return sum / data.size();
}

}  // namespace imu_filter

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(imu_filter::ImuFilter)
