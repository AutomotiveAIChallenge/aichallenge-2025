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

#ifndef GNSS_FILTER__GNSS_FILTER_CORE_HPP_
#define GNSS_FILTER__GNSS_FILTER_CORE_HPP_

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <boost/circular_buffer.hpp>

#include <memory>
#include <string>
#include <deque>

namespace gnss_filter
{

/**
 * @brief GNSS信号の品質を表す構造体
 */
struct GnssQuality
{
  double position_variance;  // 位置の分散
  double velocity;           // 推定速度
  bool is_valid;            // データの有効性
  int satellites_visible;   // 可視衛星数
};

/**
 * @brief GNSSフィルタリングノード
 * 
 * マルチパス、信号遮断、精度限界などのGNSS課題を解決するためのフィルタリングを実行
 */
class GnssFilter : public rclcpp::Node
{
public:
  explicit GnssFilter(const rclcpp::NodeOptions & node_options);

private:
  /**
   * @brief NavSatFixメッセージのコールバック関数
   * @param msg 受信したGNSSデータ
   */
  void callbackNavSatFix(const sensor_msgs::msg::NavSatFix::ConstSharedPtr msg);

  /**
   * @brief 異常値を検出する
   * @param current_msg 現在のGNSSデータ
   * @return true: 正常値, false: 異常値
   * 
   * 以下の基準で異常値を判定:
   * - 前回位置からの移動距離が閾値を超える（物理的に不可能な移動）
   * - 共分散が大きすぎる（精度が低い）
   * - 可視衛星数が少なすぎる
   */
  bool isOutlier(const sensor_msgs::msg::NavSatFix & current_msg);

  /**
   * @brief 移動平均フィルタを適用
   * @param msg 現在のGNSSデータ
   * @return フィルタリング後のデータ
   * 
   * 過去N個のデータの平均を取ることでノイズを低減
   */
  sensor_msgs::msg::NavSatFix applyMovingAverage(
    const sensor_msgs::msg::NavSatFix & msg);

  /**
   * @brief 2点間の距離を計算（緯度経度から）
   * @param lat1, lon1 地点1の緯度経度
   * @param lat2, lon2 地点2の緯度経度
   * @return 距離[m]
   */
  double calculateDistance(
    double lat1, double lon1, double lat2, double lon2) const;

  /**
   * @brief 共分散の妥当性をチェック
   * @param msg GNSSデータ
   * @return true: 共分散が妥当, false: 共分散が大きすぎる
   */
  bool isCovarianceValid(const sensor_msgs::msg::NavSatFix & msg) const;

  /**
   * @brief フィルタリング済みデータをパブリッシュ
   * @param msg フィルタリング後のGNSSデータ
   */
  void publishFilteredData(const sensor_msgs::msg::NavSatFix & msg);

  // ==================== サブスクライバ・パブリッシャ ====================
  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gnss_sub_;
  rclcpp::Publisher<sensor_msgs::msg::NavSatFix>::SharedPtr filtered_gnss_pub_;

  // ==================== パラメータ ====================
  // ■ 移動平均フィルタのバッファサイズ
  // 【調整ガイド】
  // - 大きくすると: 平滑化が強くなり、ノイズは減るが応答が遅くなる
  // - 小さくすると: 応答は速いが、ノイズが残りやすくなる
  // - 推奨値: 5-10（動きが速い場合は小さく、ゆっくりの場合は大きく）
  int buffer_size_;

  // ■ 異常値判定：最大移動距離閾値 [m]
  // 【調整ガイド】
  // - 大きくすると: より大きな急激な位置変化も許容（誤検出が減る）
  // - 小さくすると: マルチパスなどの異常をより厳しく検出
  // - 推奨値: 車両の場合 10-50m、歩行者の場合 2-10m
  // - 計算式: max_speed(m/s) × update_interval(s) × safety_factor(2-3)
  double max_distance_threshold_;

  // ■ 異常値判定：最大共分散閾値 [m^2]
  // 【調整ガイド】
  // - 大きくすると: 精度が低いデータも許容（データ欠落が減る）
  // - 小さくすると: 高精度データのみを使用（信頼性が向上）
  // - 推奨値: 10-100（一般的なGNSSの標準偏差3-10mに対応）
  double max_covariance_threshold_;

  // ■ 異常値判定：最小衛星数
  // 【調整ガイド】
  // - 大きくすると: より多くの衛星が見える良好な環境のみ使用
  // - 小さくすると: 衛星が少ない環境でもデータを使用
  // - 推奨値: 4-6（4は最低限、6以上で高精度）
  int min_satellites_;

  // ==================== 内部状態 ====================
  // 過去のGNSSデータを保存するバッファ（移動平均用）
  boost::circular_buffer<sensor_msgs::msg::NavSatFix> gnss_buffer_;

  // 直前の有効なGNSSデータ（異常値検出用）
  sensor_msgs::msg::NavSatFix::SharedPtr prev_gnss_msg_;

  // フィルタリング統計情報
  int total_received_;    // 受信した総データ数
  int total_filtered_;    // 除外した異常値の数
  int total_published_;   // パブリッシュした有効データ数
};

}  // namespace gnss_filter

#endif  // GNSS_FILTER__GNSS_FILTER_CORE_HPP_
