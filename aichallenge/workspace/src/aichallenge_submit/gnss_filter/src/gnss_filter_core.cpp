// Copyright 2024 Tier IV, Inc.
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

#include "gnss_filter/gnss_filter_core.hpp"

#include <cmath>
#include <algorithm>

namespace gnss_filter
{

GnssFilter::GnssFilter(const rclcpp::NodeOptions & node_options)
: rclcpp::Node("gnss_filter", node_options),
  total_received_(0),
  total_filtered_(0),
  total_published_(0)
{
  // ==================== パラメータの読み込み ====================
  // 移動平均フィルタのバッファサイズ
  // デフォルト: 5個のデータで平均化
  buffer_size_ = declare_parameter("buffer_size", 5);
  
  // 異常値判定：最大移動距離閾値 [m]
  // デフォルト: 30m（時速30km、1秒間隔、安全係数3を想定）
  max_distance_threshold_ = declare_parameter("max_distance_threshold", 30.0);
  
  // 異常値判定：最大共分散閾値 [m^2]
  // デフォルト: 100 (標準偏差10mに相当)
  max_covariance_threshold_ = declare_parameter("max_covariance_threshold", 100.0);
  
  // 異常値判定：最小衛星数
  // デフォルト: 4個（3D測位の最低限）
  min_satellites_ = declare_parameter("min_satellites", 4);

  // バッファの初期化
  gnss_buffer_.set_capacity(buffer_size_);

  // ==================== サブスクライバの作成 ====================
  // 入力: 生のGNSSデータ
  gnss_sub_ = create_subscription<sensor_msgs::msg::NavSatFix>(
    "gnss/fix", rclcpp::QoS{10},
    std::bind(&GnssFilter::callbackNavSatFix, this, std::placeholders::_1));

  // ==================== パブリッシャの作成 ====================
  // 出力: フィルタリング済みGNSSデータ
  filtered_gnss_pub_ = create_publisher<sensor_msgs::msg::NavSatFix>(
    "gnss/filtered_fix", rclcpp::QoS{10});

  // パラメータのログ出力
  RCLCPP_INFO(this->get_logger(), "GNSS Filter Node initialized");
  RCLCPP_INFO(this->get_logger(), "  buffer_size: %d", buffer_size_);
  RCLCPP_INFO(this->get_logger(), "  max_distance_threshold: %.2f m", max_distance_threshold_);
  RCLCPP_INFO(this->get_logger(), "  max_covariance_threshold: %.2f m^2", max_covariance_threshold_);
  RCLCPP_INFO(this->get_logger(), "  min_satellites: %d", min_satellites_);
}

void GnssFilter::callbackNavSatFix(const sensor_msgs::msg::NavSatFix::ConstSharedPtr msg)
{
  // 受信カウンタをインクリメント
  total_received_++;

  // ==================== ステップ1: 基本的な妥当性チェック ====================
  // GNSSデータのステータスが有効かチェック
  if (msg->status.status < sensor_msgs::msg::NavSatStatus::STATUS_FIX) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "GNSS status is not FIX. Skipping data. Status: %d", msg->status.status);
    total_filtered_++;
    return;
  }

  // ==================== ステップ2: 異常値検出 ====================
  // マルチパスや急激な位置ジャンプを検出
  if (isOutlier(*msg)) {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 1000,
      "Outlier detected. Position: (%.6f, %.6f), Distance from prev: %.2f m",
      msg->latitude, msg->longitude,
      prev_gnss_msg_ ? calculateDistance(
        msg->latitude, msg->longitude,
        prev_gnss_msg_->latitude, prev_gnss_msg_->longitude) : 0.0);
    total_filtered_++;
    return;
  }

  // ==================== ステップ3: 移動平均フィルタを適用 ====================
  // ノイズを低減するため、過去のデータと平均化
  auto filtered_msg = applyMovingAverage(*msg);

  // ==================== ステップ4: フィルタリング済みデータをパブリッシュ ====================
  publishFilteredData(filtered_msg);

  // 現在のデータを前回データとして保存
  prev_gnss_msg_ = std::make_shared<sensor_msgs::msg::NavSatFix>(*msg);
  total_published_++;

  // 定期的に統計情報をログ出力（5秒ごと）
  if (total_received_ % 50 == 0) {
    double filter_rate = 100.0 * total_filtered_ / total_received_;
    RCLCPP_INFO(
      this->get_logger(),
      "Stats - Received: %d, Filtered: %d (%.1f%%), Published: %d",
      total_received_, total_filtered_, filter_rate, total_published_);
  }
}

bool GnssFilter::isOutlier(const sensor_msgs::msg::NavSatFix & current_msg)
{
  // ==================== チェック1: 共分散の妥当性 ====================
  // 共分散が大きすぎる = 精度が低い測位結果
  if (!isCovarianceValid(current_msg)) {
    RCLCPP_DEBUG(
      this->get_logger(),
      "Covariance too large: %.2f > %.2f",
      current_msg.position_covariance[0], max_covariance_threshold_);
    return true;
  }

  // ==================== チェック2: 可視衛星数 ====================
  // 衛星数が少ないと測位精度が低下
  // NMEA 0183のGGAメッセージから取得される衛星数をチェック
  // 注: NavSatFixには直接衛星数フィールドがないため、
  //     実際の実装では別のメッセージから取得するか、
  //     ここではステータスで代用
  // この実装例では簡略化のためスキップ（必要に応じて追加）

  // ==================== チェック3: 前回位置からの距離 ====================
  // 前回データが存在する場合のみチェック
  if (prev_gnss_msg_) {
    // 現在位置と前回位置の距離を計算
    double distance = calculateDistance(
      current_msg.latitude, current_msg.longitude,
      prev_gnss_msg_->latitude, prev_gnss_msg_->longitude);

    // 距離が閾値を超える場合は異常値と判定
    // これにより、マルチパスによる急激な位置ジャンプを検出
    if (distance > max_distance_threshold_) {
      RCLCPP_DEBUG(
        this->get_logger(),
        "Distance too large: %.2f > %.2f",
        distance, max_distance_threshold_);
      return true;
    }

    // ==================== チェック4: 時間間隔をチェック ====================
    // 時間間隔が長すぎる場合は、閾値判定を緩和
    double time_diff = (rclcpp::Time(current_msg.header.stamp) -
                        rclcpp::Time(prev_gnss_msg_->header.stamp)).seconds();
    
    // 時間間隔が5秒以上の場合は、距離チェックをスキップ
    // （長時間の移動で大きな距離移動があっても正常と判断）
    if (time_diff > 5.0) {
      RCLCPP_DEBUG(
        this->get_logger(),
        "Large time gap detected: %.2f sec. Skipping distance check.", time_diff);
      return false;
    }
  }

  // すべてのチェックをパス = 正常なデータ
  return false;
}

sensor_msgs::msg::NavSatFix GnssFilter::applyMovingAverage(
  const sensor_msgs::msg::NavSatFix & msg)
{
  // 現在のデータをバッファに追加
  gnss_buffer_.push_back(msg);

  // バッファが空の場合は、そのまま返す
  if (gnss_buffer_.empty()) {
    return msg;
  }

  // ==================== 移動平均の計算 ====================
  // 緯度、経度、高度それぞれの平均を計算
  double sum_lat = 0.0;
  double sum_lon = 0.0;
  double sum_alt = 0.0;
  int count = gnss_buffer_.size();

  for (const auto & buffered_msg : gnss_buffer_) {
    sum_lat += buffered_msg.latitude;
    sum_lon += buffered_msg.longitude;
    sum_alt += buffered_msg.altitude;
  }

  // 平均値を計算
  sensor_msgs::msg::NavSatFix filtered_msg = msg;
  filtered_msg.latitude = sum_lat / count;
  filtered_msg.longitude = sum_lon / count;
  filtered_msg.altitude = sum_alt / count;

  // ==================== 共分散の調整 ====================
  // 移動平均によりノイズが低減されるため、共分散も低減
  // 理論上は1/sqrt(N)に低減されるが、ここでは簡易的に1/Nを使用
  double reduction_factor = 1.0 / count;
  for (int i = 0; i < 9; ++i) {
    filtered_msg.position_covariance[i] = msg.position_covariance[i] * reduction_factor;
  }

  return filtered_msg;
}

double GnssFilter::calculateDistance(
  double lat1, double lon1, double lat2, double lon2) const
{
  // ==================== Haversine公式による距離計算 ====================
  // 2点間の大圏距離を計算（球面上の最短距離）
  
  // 地球の平均半径 [m]
  const double earth_radius = 6371000.0;

  // 度をラジアンに変換
  double lat1_rad = lat1 * M_PI / 180.0;
  double lon1_rad = lon1 * M_PI / 180.0;
  double lat2_rad = lat2 * M_PI / 180.0;
  double lon2_rad = lon2 * M_PI / 180.0;

  // 緯度・経度の差
  double dlat = lat2_rad - lat1_rad;
  double dlon = lon2_rad - lon1_rad;

  // Haversine公式
  double a = std::sin(dlat / 2.0) * std::sin(dlat / 2.0) +
             std::cos(lat1_rad) * std::cos(lat2_rad) *
             std::sin(dlon / 2.0) * std::sin(dlon / 2.0);
  double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
  double distance = earth_radius * c;

  return distance;
}

bool GnssFilter::isCovarianceValid(const sensor_msgs::msg::NavSatFix & msg) const
{
  // ==================== 共分散のチェック ====================
  // position_covariance[0]: x方向(東)の分散
  // position_covariance[4]: y方向(北)の分散
  // position_covariance[8]: z方向(上)の分散
  
  // 共分散タイプが不明の場合は、チェックをスキップ
  if (msg.position_covariance_type == sensor_msgs::msg::NavSatFix::COVARIANCE_TYPE_UNKNOWN) {
    return true;  // 共分散情報がない場合は、他の条件でチェック
  }

  // x, y方向の共分散をチェック（水平精度）
  // どちらかが閾値を超える場合は無効と判定
  if (msg.position_covariance[0] > max_covariance_threshold_ ||
      msg.position_covariance[4] > max_covariance_threshold_) {
    return false;
  }

  // すべてのチェックをパス
  return true;
}

void GnssFilter::publishFilteredData(const sensor_msgs::msg::NavSatFix & msg)
{
  // フィルタリング済みデータをパブリッシュ
  filtered_gnss_pub_->publish(msg);

  // デバッグ情報をログ出力（詳細レベル）
  RCLCPP_DEBUG(
    this->get_logger(),
    "Published filtered GNSS: (%.6f, %.6f, %.2f)",
    msg.latitude, msg.longitude, msg.altitude);
}

}  // namespace gnss_filter

// ==================== ノードのコンポーネント登録 ====================
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(gnss_filter::GnssFilter)
