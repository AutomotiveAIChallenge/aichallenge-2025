cmake_minimum_required(VERSION 3.8)
project(goal_checker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 依存パッケージを探す
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

# 実行ファイル（ノード）を作成
add_executable(goal_line_checker_node src/goal_line_checker_node.cpp)

# 依存関係をリンク
ament_target_dependencies(goal_line_checker_node
  rclcpp
  geometry_msgs
  nav_msgs
)

# インストール設定
install(TARGETS
  goal_line_checker_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
