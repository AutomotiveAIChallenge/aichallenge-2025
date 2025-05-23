cmake_minimum_required(VERSION 3.8)
project(simple_trajectory_generator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_executable(simple_trajectory_generator_node
  src/simple_trajectory_generator.cpp
)

ament_auto_package(INSTALL_TO_SHARE
  data
)
