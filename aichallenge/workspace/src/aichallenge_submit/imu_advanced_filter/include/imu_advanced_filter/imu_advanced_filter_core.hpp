#ifndef IMU_ADVANCED_FILTER__IMU_ADVANCED_FILTER_CORE_HPP_
#define IMU_ADVANCED_FILTER__IMU_ADVANCED_FILTER_CORE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <std_msgs/msg/float64.hpp>

#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <chrono>
#include <memory>

namespace imu_advanced_filter
{

class DriftCompensationFilter
{
private:
  Eigen::VectorXd state_;
  Eigen::MatrixXd covariance_;
  Eigen::MatrixXd process_noise_;
  Eigen::MatrixXd measurement_noise_;
  
  double drift_compensation_gain_;
  double bias_estimation_rate_;
  double zero_velocity_threshold_;
  
  bool is_initialized_;
  rclcpp::Time last_update_time_;
  Eigen::Vector3d accumulated_angular_bias_;
  Eigen::Vector3d accumulated_accel_bias_;
  
public:
  DriftCompensationFilter()
  {
    drift_compensation_gain_ = 0.025;
    bias_estimation_rate_ = 0.003;
    zero_velocity_threshold_ = 0.05;
    
    state_ = Eigen::VectorXd::Zero(15);
    covariance_ = Eigen::MatrixXd::Identity(15, 15) * 0.1;
    
    process_noise_ = Eigen::MatrixXd::Identity(15, 15);
    process_noise_.block<3,3>(0,0) *= 0.01;
    process_noise_.block<3,3>(3,3) *= 0.1;
    process_noise_.block<3,3>(6,6) *= 1.0;
    process_noise_.block<3,3>(9,9) *= 0.001;
    process_noise_.block<3,3>(12,12) *= 0.01;
    
    measurement_noise_ = Eigen::MatrixXd::Identity(6, 6);
    measurement_noise_.block<3,3>(0,0) *= 0.1;
    measurement_noise_.block<3,3>(3,3) *= 0.05;
    
    accumulated_angular_bias_.setZero();
    accumulated_accel_bias_.setZero();
    is_initialized_ = false;
  }
  
  void updateParameters(double drift_gain, double bias_rate, double zero_vel_thresh)
  {
    drift_compensation_gain_ = std::clamp(drift_gain, 0.001, 0.1);
    bias_estimation_rate_ = std::clamp(bias_rate, 0.0001, 0.01);
    zero_velocity_threshold_ = std::clamp(zero_vel_thresh, 0.01, 0.1);
  }
  
  sensor_msgs::msg::Imu compensateDrift(
    const sensor_msgs::msg::Imu& raw_imu,
    const nav_msgs::msg::Odometry& odometry_data,
    bool has_gnss_fix
  )
  {
    sensor_msgs::msg::Imu corrected_imu = raw_imu;
    
    if (!is_initialized_) {
      initializeFilter(raw_imu, odometry_data);
      return corrected_imu;
    }
    
    // 簡易的なドリフト補正実装
    corrected_imu.angular_velocity.x -= accumulated_angular_bias_(0);
    corrected_imu.angular_velocity.y -= accumulated_angular_bias_(1);
    corrected_imu.angular_velocity.z -= accumulated_angular_bias_(2);
    
    corrected_imu.linear_acceleration.x -= accumulated_accel_bias_(0);
    corrected_imu.linear_acceleration.y -= accumulated_accel_bias_(1);
    corrected_imu.linear_acceleration.z -= accumulated_accel_bias_(2);
    
    return corrected_imu;
  }
  
  Eigen::Vector3d getCurrentAngularBias() const { return accumulated_angular_bias_; }
  Eigen::Vector3d getCurrentAccelBias() const { return accumulated_accel_bias_; }
  double getDriftMagnitude() const { return accumulated_angular_bias_.norm(); }
  
private:
  void initializeFilter(const sensor_msgs::msg::Imu& imu, const nav_msgs::msg::Odometry& odom)
  {
    last_update_time_ = rclcpp::Time(imu.header.stamp);
    is_initialized_ = true;
  }
};

class VibrationNoiseFilter
{
private:
  double lowpass_cutoff_frequency_;
  double vibration_detection_threshold_;
  double median_filter_window_size_;
  
  std::deque<Eigen::Vector3d> angular_velocity_history_;
  std::deque<Eigen::Vector3d> linear_acceleration_history_;
  
  Eigen::Vector3d prev_filtered_angular_vel_;
  Eigen::Vector3d prev_filtered_linear_accel_;
  bool filter_initialized_;
  
public:
  VibrationNoiseFilter()
  {
    lowpass_cutoff_frequency_ = 15.0;
    vibration_detection_threshold_ = 0.8;
    median_filter_window_size_ = 7;
    
    prev_filtered_angular_vel_.setZero();
    prev_filtered_linear_accel_.setZero();
    filter_initialized_ = false;
  }
  
  void updateParameters(double cutoff_freq, double vibration_thresh, int window_size)
  {
    lowpass_cutoff_frequency_ = std::clamp(cutoff_freq, 1.0, 50.0);
    vibration_detection_threshold_ = std::clamp(vibration_thresh, 0.1, 2.0);
    median_filter_window_size_ = std::clamp(static_cast<double>(window_size), 3.0, 21.0);
  }
  
  sensor_msgs::msg::Imu filterVibrationNoise(const sensor_msgs::msg::Imu& raw_imu)
  {
    sensor_msgs::msg::Imu filtered_imu = raw_imu;
    
    if (!filter_initialized_) {
      filter_initialized_ = true;
      return filtered_imu;
    }
    
    // 簡易的なローパスフィルタ
    Eigen::Vector3d current_angular_vel(
      raw_imu.angular_velocity.x, raw_imu.angular_velocity.y, raw_imu.angular_velocity.z
    );
    
    double alpha = 0.1;  // 簡易的なフィルタ係数
    Eigen::Vector3d filtered_angular = alpha * current_angular_vel + (1.0 - alpha) * prev_filtered_angular_vel_;
    
    filtered_imu.angular_velocity.x = filtered_angular(0);
    filtered_imu.angular_velocity.y = filtered_angular(1);
    filtered_imu.angular_velocity.z = filtered_angular(2);
    
    prev_filtered_angular_vel_ = filtered_angular;
    
    return filtered_imu;
  }
  
  double getVibrationLevel() const { return 0.5; }
};

class ImuAdvancedFilterNode : public rclcpp::Node
{
private:
  std::unique_ptr<DriftCompensationFilter> drift_filter_;
  std::unique_ptr<VibrationNoiseFilter> vibration_filter_;
  
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr filtered_imu_pub_;
  
  nav_msgs::msg::Odometry latest_odom_;
  bool odom_received_;
  bool gnss_fix_available_;
  
public:
  ImuAdvancedFilterNode() : Node("imu_advanced_filter_node")
  {
    this->declare_parameter("drift_compensation_gain", 0.025);
    this->declare_parameter("bias_estimation_rate", 0.003);
    this->declare_parameter("zero_velocity_threshold", 0.05);
    this->declare_parameter("lowpass_cutoff_frequency", 15.0);
    this->declare_parameter("vibration_detection_threshold", 0.8);
    this->declare_parameter("median_filter_window_size", 7);
    this->declare_parameter("enable_drift_compensation", true);
    this->declare_parameter("enable_vibration_filtering", true);
    this->declare_parameter("debug_output", false);
    
    drift_filter_ = std::make_unique<DriftCompensationFilter>();
    vibration_filter_ = std::make_unique<VibrationNoiseFilter>();
    
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
      "input/imu", qos,
      std::bind(&ImuAdvancedFilterNode::imuCallback, this, std::placeholders::_1)
    );
    
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "input/odometry", qos,
      std::bind(&ImuAdvancedFilterNode::odomCallback, this, std::placeholders::_1)
    );
    
    filtered_imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>(
      "output/filtered_imu", qos
    );
    
    odom_received_ = false;
    gnss_fix_available_ = false;
    
    RCLCPP_INFO(this->get_logger(), "IMU高度フィルタノード起動完了");
  }
  
private:
  void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (!odom_received_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "オドメトリデータ未受信 - IMUフィルタリングをスキップ");
      return;
    }
    
    sensor_msgs::msg::Imu filtered_imu = *msg;
    
    bool enable_vibration = this->get_parameter("enable_vibration_filtering").as_bool();
    if (enable_vibration) {
      filtered_imu = vibration_filter_->filterVibrationNoise(filtered_imu);
    }
    
    bool enable_drift = this->get_parameter("enable_drift_compensation").as_bool();
    if (enable_drift) {
      filtered_imu = drift_filter_->compensateDrift(filtered_imu, latest_odom_, gnss_fix_available_);
    }
    
    filtered_imu_pub_->publish(filtered_imu);
  }
  
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    latest_odom_ = *msg;
    odom_received_ = true;
    
    double position_covariance = msg->pose.covariance[0] + msg->pose.covariance[7];
    gnss_fix_available_ = (position_covariance < 10.0);
  }
};

} // namespace imu_advanced_filter

#endif // IMU_ADVANCED_FILTER__IMU_ADVANCED_FILTER_CORE_HPP_
