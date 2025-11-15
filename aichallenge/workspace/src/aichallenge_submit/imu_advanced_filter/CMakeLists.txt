cmake_minimum_required(VERSION 3.5)
project(imu_advanced_filter)

# C++17標準を使用
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# コンパイラ警告設定
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 依存パッケージの検索
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# メインのIMUフィルタノード
add_executable(imu_advanced_filter_node
  src/imu_advanced_filter_node.cpp
)

target_include_directories(imu_advanced_filter_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

ament_target_dependencies(imu_advanced_filter_node
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  Eigen3
)

# インストール設定
install(TARGETS
  imu_advanced_filter_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include/
)

install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.xml"
)

install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

install(PROGRAMS
  scripts/quick_setup.sh
  scripts/imu_diagnostic_tool.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  std_msgs
  Eigen3
)

ament_package()
