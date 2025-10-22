#include "imu_advanced_filter/imu_advanced_filter_core.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  RCLCPP_INFO(rclcpp::get_logger("main"), 
    "=== IMU高度フィルタシステム起動 ===");
  RCLCPP_INFO(rclcpp::get_logger("main"), 
    "解決する課題:");
  RCLCPP_INFO(rclcpp::get_logger("main"), 
    "  1. ドリフト現象: IMU積分計算による誤差累積");
  RCLCPP_INFO(rclcpp::get_logger("main"), 
    "  2. 振動ノイズ: 車両走行時の振動混入");
  
  auto node = std::make_shared<imu_advanced_filter::ImuAdvancedFilterNode>();
  
  try {
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("main"), 
      "IMUフィルタでエラー発生: %s", e.what());
    return 1;
  }
  
  RCLCPP_INFO(rclcpp::get_logger("main"), "IMUフィルタシステム終了");
  rclcpp::shutdown();
  return 0;
}

