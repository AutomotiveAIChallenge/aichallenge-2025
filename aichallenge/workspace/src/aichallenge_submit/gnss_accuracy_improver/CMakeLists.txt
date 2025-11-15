cmake_minimum_required(VERSION 3.8)
project(gnss_accuracy_improver)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Python実行ファイルのインストール
install(PROGRAMS
  scripts/gnss_accuracy_improver_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# 設定ファイルのインストール
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config/
)

# ソースファイルのインストール
install(DIRECTORY src/ gnss_accuracy_improver/
  DESTINATION share/${PROJECT_NAME}/src/
  FILES_MATCHING PATTERN "*.py"
)

ament_package()
