cmake_minimum_required(VERSION 3.14)
project(imu_filter)

find_package(autoware_cmake REQUIRED)
autoware_package()

ament_auto_add_library(imu_filter_node SHARED
  src/imu_filter_core.cpp
  include/imu_filter/imu_filter_core.hpp
)

rclcpp_components_register_node(imu_filter_node
  PLUGIN "imu_filter::ImuFilter"
  EXECUTABLE imu_filter
)

ament_auto_package(INSTALL_TO_SHARE
  launch
  config
)
