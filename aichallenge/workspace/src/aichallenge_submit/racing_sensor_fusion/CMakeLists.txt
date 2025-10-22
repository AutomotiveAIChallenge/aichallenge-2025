cmake_minimum_required(VERSION 3.8)
project(racing_sensor_fusion)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# メインのセンサーフュージョンnode
add_executable(racing_sensor_fusion_node src/racing_sensor_fusion_node.cpp)
ament_target_dependencies(racing_sensor_fusion_node
  rclcpp
  geometry_msgs
  sensor_msgs
  nav_msgs
  std_msgs
)

# 診断ツール（オプション）
add_executable(fusion_diagnostic_tool src/fusion_diagnostic_tool.cpp)
ament_target_dependencies(fusion_diagnostic_tool
  rclcpp
)

# インストール設定
install(TARGETS
  racing_sensor_fusion_node
  fusion_diagnostic_tool
  DESTINATION lib/${PROJECT_NAME}
)

# 設定ファイルとlaunchファイルのインストール
install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
