cmake_minimum_required(VERSION 3.14)
project(gnss_filter)

# Autowareの共通CMake設定を読み込み
find_package(autoware_cmake REQUIRED)
autoware_package()

# ==================== ヘッダーファイルの設定 ====================
set(GNSS_FILTER_HEADERS
  include/gnss_filter/gnss_filter_core.hpp
)

# ==================== ライブラリのビルド ====================
# 共有ライブラリとしてノードを作成
ament_auto_add_library(gnss_filter_node SHARED
  src/gnss_filter_core.cpp
  ${GNSS_FILTER_HEADERS}
)

# ==================== ノードコンポーネントの登録 ====================
# ROS2のコンポーネントとして登録（composition用）
rclcpp_components_register_node(gnss_filter_node
  PLUGIN "gnss_filter::GnssFilter"
  EXECUTABLE gnss_filter
)

# ==================== インストール設定 ====================
# launch、configフォルダをインストール
ament_auto_package(INSTALL_TO_SHARE
  launch
  config
)
