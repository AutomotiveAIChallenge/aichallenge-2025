#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <deque>
#include <chrono>
#include <cmath>

/**
 * カートレース用改良センサーフュージョンnode
 * 
 * 【主な改良点】
 * 1. 運転状態に応じた動的センサー重み付け
 * 2. リアルタイム処理最適化
 * 3. カートレース特有の状況への対応
 * 4. 予測機能付き位置推定
 */
class RacingSensorFusion : public rclcpp::Node
{
public:
    RacingSensorFusion() : Node("racing_sensor_fusion")
    {
        // ============================================
        // パラメータ設定（初心者でもわかりやすく）
        // ============================================
        
        // 【重要】信号処理の基本設定
        declare_parameter("update_frequency", 50.0);        // 更新頻度 [Hz] - 大きくするほど応答が速い
        declare_parameter("prediction_time", 0.05);         // 予測時間 [秒] - 大きくするほど未来を予測
        
        // 【重要】センサー重み調整（0.0～1.0）
        declare_parameter("gnss_weight_default", 0.7);      // GNSS標準重み - 大きいほどGNSSを信頼
        declare_parameter("imu_weight_default", 0.3);       // IMU標準重み - 大きいほどIMUを信頼
        
        // 【実走行データ基準】速度帯別重み調整
        declare_parameter("high_speed_threshold", 34.5);    // 直線走行判定 [km/h]
        declare_parameter("corner_speed_threshold", 28.0);  // コーナー走行判定 [km/h]
        
        // 直線走行時（35km/h付近）- ハンドル振れ抑制重視
        declare_parameter("straight_gnss_weight", 0.85);    // 直線時GNSS重み
        declare_parameter("straight_imu_weight", 0.15);     // 直線時IMU重み
        
        // コーナー走行時（29-34km/h）- オーバステア抑制重視
        declare_parameter("corner_gnss_weight", 0.60);      // コーナー時GNSS重み
        declare_parameter("corner_imu_weight", 0.40);       // コーナー時IMU重み
        
        // ピットレーン走行時（10km/h以下）- 安全重視
        declare_parameter("pit_gnss_weight", 0.70);         // ピット時GNSS重み
        declare_parameter("pit_imu_weight", 0.30);          // ピット時IMU重み
        
        // データ保存設定
        declare_parameter("history_size", 10);              // 履歴保存数 - 大きいほど安定、小さいほど応答良
        
        // パラメータ取得
        update_frequency_ = get_parameter("update_frequency").as_double();
        prediction_time_ = get_parameter("prediction_time").as_double();
        gnss_weight_default_ = get_parameter("gnss_weight_default").as_double();
        imu_weight_default_ = get_parameter("imu_weight_default").as_double();
        high_speed_threshold_ = get_parameter("high_speed_threshold").as_double();
        corner_speed_threshold_ = get_parameter("corner_speed_threshold").as_double();
        straight_gnss_weight_ = get_parameter("straight_gnss_weight").as_double();
        straight_imu_weight_ = get_parameter("straight_imu_weight").as_double();
        corner_gnss_weight_ = get_parameter("corner_gnss_weight").as_double();
        corner_imu_weight_ = get_parameter("corner_imu_weight").as_double();
        pit_gnss_weight_ = get_parameter("pit_gnss_weight").as_double();
        pit_imu_weight_ = get_parameter("pit_imu_weight").as_double();
        history_size_ = get_parameter("history_size").as_int();
        
        // ============================================
        // ROSトピック設定
        // ============================================
        const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
        
        // パブリッシャー（出力）
        pub_fused_pose_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/localization/racing_sensor_fusion/pose_with_covariance", qos);
        pub_fused_odom_ = create_publisher<nav_msgs::msg::Odometry>(
            "/localization/racing_sensor_fusion/odometry", qos);
        
        // サブスクライバー（入力）
        sub_gnss_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/sensing/gnss/pose_with_covariance", qos,
            std::bind(&RacingSensorFusion::gnss_callback, this, std::placeholders::_1));
        sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
            "/sensing/imu/imu_data", qos,
            std::bind(&RacingSensorFusion::imu_callback, this, std::placeholders::_1));
        sub_velocity_ = create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
            "/sensing/vehicle_velocity_converter/twist_with_covariance", qos,
            std::bind(&RacingSensorFusion::velocity_callback, this, std::placeholders::_1));
        
        // タイマー（メイン処理ループ）
        timer_ = create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / update_frequency_)),
            std::bind(&RacingSensorFusion::main_process, this));
        
        // 初期化
        is_initialized_ = false;
        current_speed_kmh_ = 0.0;
        
        RCLCPP_INFO(this->get_logger(), "カートレース用センサーフュージョン開始");
        RCLCPP_INFO(this->get_logger(), "更新頻度: %.1f Hz", update_frequency_);
    }

private:
    // ============================================
    // センサーデータコールバック関数
    // ============================================
    
    /**
     * GNSS信号受信処理
     * GNSSから位置情報（緯度経度）を受信
     */
    void gnss_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // GNSSデータを履歴に保存
        gnss_history_.push_back(*msg);
        if (gnss_history_.size() > history_size_) {
            gnss_history_.pop_front();
        }
        
        gnss_last_time_ = this->now();
        
        // 初期化チェック
        if (!is_initialized_) {
            // 最初のGNSSデータで初期位置設定
            latest_fused_pose_ = *msg;
            is_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "初期位置設定完了");
        }
    }
    
    /**
     * IMU信号受信処理  
     * IMUから姿勢・角速度・加速度情報を受信
     */
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // IMUデータを履歴に保存
        imu_history_.push_back(*msg);
        if (imu_history_.size() > history_size_) {
            imu_history_.pop_front();
        }
        
        imu_last_time_ = this->now();
    }
    
    /**
     * 速度信号受信処理
     * 車両の現在速度を取得（重み付け計算に使用）
     */
    void velocity_callback(const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // 速度をkm/hに変換
        double speed_ms = std::sqrt(
            msg->twist.twist.linear.x * msg->twist.twist.linear.x +
            msg->twist.twist.linear.y * msg->twist.twist.linear.y
        );
        current_speed_kmh_ = speed_ms * 3.6; // m/s → km/h変換
        
        velocity_last_time_ = this->now();
    }
    
    // ============================================
    // メイン処理（センサーフュージョン実行）
    // ============================================
    
    /**
     * メイン処理ループ
     * 各センサーの情報を統合して最適な位置推定を実行
     */
    void main_process()
    {
        std::lock_guard<std::mutex> lock(data_mutex_);
        
        // 初期化チェック
        if (!is_initialized_ || gnss_history_.empty() || imu_history_.empty()) {
            return;
        }
        
        // データ新しさチェック（古いデータは使わない）
        auto current_time = this->now();
        const double timeout_sec = 1.0; // 1秒以上古いデータは無効
        
        bool gnss_valid = (current_time - gnss_last_time_).seconds() < timeout_sec;
        bool imu_valid = (current_time - imu_last_time_).seconds() < timeout_sec;
        
        if (!gnss_valid && !imu_valid) {
            RCLCPP_WARN(this->get_logger(), "センサーデータが古すぎます");
            return;
        }
        
        // ============================================
        // 運転状態判定（速度ベース）
        // ============================================
        DrivingState driving_state = getDrivingState(current_speed_kmh_);
        
        // ============================================
        // センサー重み計算（運転状態に応じて動的調整）
        // ============================================
        SensorWeights weights = calculateSensorWeights(driving_state);
        
        // ============================================
        // センサーフュージョン実行
        // ============================================
        geometry_msgs::msg::PoseWithCovarianceStamped fused_pose = 
            performSensorFusion(weights, gnss_valid, imu_valid);
        
        // ============================================
        // 予測処理（未来位置の推定）
        // ============================================
        if (!imu_history_.empty()) {
            fused_pose = applyPrediction(fused_pose);
        }
        
        // ============================================
        // 結果発行
        // ============================================
        latest_fused_pose_ = fused_pose;
        publishResults(fused_pose, driving_state, weights);
    }
    
    // ============================================
    // 運転状態判定関数
    // ============================================
    
    /**
     * 運転状態を速度で判定（実走行データ基準）
     * STRAIGHT_DRIVING: 直線走行中（GNSS重視でハンドル振れ抑制）
     * CORNER_DRIVING: コーナー走行中（バランス重視でオーバステア抑制）
     * PIT_LANE: ピットレーン走行中（安全重視）
     */
    enum class DrivingState {
        STRAIGHT_DRIVING,   // 直線走行（34.5km/h以上）
        CORNER_DRIVING,     // コーナー走行（28-34.5km/h）
        PIT_LANE           // ピットレーン（28km/h未満）
    };
    
    DrivingState getDrivingState(double speed_kmh)
    {
        if (speed_kmh >= high_speed_threshold_) {
            return DrivingState::STRAIGHT_DRIVING;  // 直線走行モード
        } else if (speed_kmh >= corner_speed_threshold_) {
            return DrivingState::CORNER_DRIVING;    // コーナー走行モード
        } else {
            return DrivingState::PIT_LANE;          // ピットレーンモード
        }
    }
    
    // ============================================
    // センサー重み計算
    // ============================================
    
    struct SensorWeights {
        double gnss_weight; // GNSS重み（位置精度重視）
        double imu_weight;  // IMU重み（姿勢・動的応答重視）
    };
    
    /**
     * 運転状態に応じてセンサー重みを動的調整（実走行最適化）
     * 
     * 【重み付けの考え方】
     * - 直線走行時：GNSS最重視（ハンドル振れ抑制）
     * - コーナー走行時：バランス重視（オーバステア抑制）
     * - ピットレーン時：安全重視（慎重な制御）
     */
    SensorWeights calculateSensorWeights(DrivingState state)
    {
        SensorWeights weights;
        
        switch (state) {
            case DrivingState::STRAIGHT_DRIVING:
                // 直線走行時：GNSS最重視（ハンドル振れ抑制）
                weights.gnss_weight = straight_gnss_weight_;
                weights.imu_weight = straight_imu_weight_;
                break;
                
            case DrivingState::CORNER_DRIVING:
                // コーナー走行時：バランス重視（オーバステア抑制）
                weights.gnss_weight = corner_gnss_weight_;
                weights.imu_weight = corner_imu_weight_;
                break;
                
            case DrivingState::PIT_LANE:
                // ピットレーン時：安全重視
                weights.gnss_weight = pit_gnss_weight_;
                weights.imu_weight = pit_imu_weight_;
                break;
                
            default:
                // フォールバック：通常設定
                weights.gnss_weight = gnss_weight_default_;
                weights.imu_weight = imu_weight_default_;
                break;
        }
        
        // 重み正規化（合計を1.0に）
        double total = weights.gnss_weight + weights.imu_weight;
        if (total > 0.0) {
            weights.gnss_weight /= total;
            weights.imu_weight /= total;
        }
        
        return weights;
    }
    
    // ============================================
    // センサーフュージョン実行関数
    // ============================================
    
    /**
     * 実際のセンサーフュージョン処理
     * GNSSとIMUの信号を重み付きで統合
     */
    geometry_msgs::msg::PoseWithCovarianceStamped performSensorFusion(
        const SensorWeights& weights, bool gnss_valid, bool imu_valid)
    {
        geometry_msgs::msg::PoseWithCovarianceStamped result = latest_fused_pose_;
        result.header.stamp = this->now();
        result.header.frame_id = "base_link";
        
        // ============================================
        // 位置情報の統合（主にGNSSベース）
        // ============================================
        if (gnss_valid && !gnss_history_.empty()) {
            const auto& latest_gnss = gnss_history_.back();
            
            // GNSS位置を重み付きで適用
            result.pose.pose.position.x = 
                weights.gnss_weight * latest_gnss.pose.pose.position.x +
                (1.0 - weights.gnss_weight) * result.pose.pose.position.x;
                
            result.pose.pose.position.y = 
                weights.gnss_weight * latest_gnss.pose.pose.position.y +
                (1.0 - weights.gnss_weight) * result.pose.pose.position.y;
                
            result.pose.pose.position.z = 
                weights.gnss_weight * latest_gnss.pose.pose.position.z +
                (1.0 - weights.gnss_weight) * result.pose.pose.position.z;
        }
        
        // ============================================
        // 姿勢情報の統合（主にIMUベース）
        // ============================================
        if (imu_valid && !imu_history_.empty()) {
            const auto& latest_imu = imu_history_.back();
            
            // IMU姿勢を重み付きで適用
            result.pose.pose.orientation.x = 
                weights.imu_weight * latest_imu.orientation.x +
                (1.0 - weights.imu_weight) * result.pose.pose.orientation.x;
                
            result.pose.pose.orientation.y = 
                weights.imu_weight * latest_imu.orientation.y +
                (1.0 - weights.imu_weight) * result.pose.pose.orientation.y;
                
            result.pose.pose.orientation.z = 
                weights.imu_weight * latest_imu.orientation.z +
                (1.0 - weights.imu_weight) * result.pose.pose.orientation.z;
                
            result.pose.pose.orientation.w = 
                weights.imu_weight * latest_imu.orientation.w +
                (1.0 - weights.imu_weight) * result.pose.pose.orientation.w;
        }
        
        // ============================================
        // 共分散（不確実性）の調整
        // ============================================
        updateCovariance(result, weights, gnss_valid, imu_valid);
        
        return result;
    }
    
    /**
     * 共分散行列更新（データの信頼性を表現）
     * 重みが高いセンサーほど不確実性を小さく設定
     */
    void updateCovariance(geometry_msgs::msg::PoseWithCovarianceStamped& pose,
                          const SensorWeights& weights, bool gnss_valid, bool imu_valid)
    {
        // 位置の不確実性（GNSSの重みで調整）
        double pos_variance = gnss_valid ? 0.1 / weights.gnss_weight : 1.0;
        pose.pose.covariance[0] = pos_variance;   // x
        pose.pose.covariance[7] = pos_variance;   // y
        pose.pose.covariance[14] = pos_variance;  // z
        
        // 姿勢の不確実性（IMUの重みで調整）
        double orientation_variance = imu_valid ? 0.1 / weights.imu_weight : 100.0;
        pose.pose.covariance[21] = orientation_variance;  // roll
        pose.pose.covariance[28] = orientation_variance;  // pitch
        pose.pose.covariance[35] = orientation_variance;  // yaw
    }
    
    // ============================================
    // 予測処理（未来位置推定）
    // ============================================
    
    /**
     * 予測処理：現在の動きから未来位置を推定
     * カートの動きを予測して制御の先読み効果を実現
     */
    geometry_msgs::msg::PoseWithCovarianceStamped applyPrediction(
        const geometry_msgs::msg::PoseWithCovarianceStamped& current_pose)
    {
        if (imu_history_.empty()) {
            return current_pose;
        }
        
        geometry_msgs::msg::PoseWithCovarianceStamped predicted_pose = current_pose;
        
        const auto& latest_imu = imu_history_.back();
        
        // 角速度から予測回転量計算
        double angular_velocity_z = latest_imu.angular_velocity.z;
        double predicted_yaw_change = angular_velocity_z * prediction_time_;
        
        // 現在の向きに予測回転量を加算
        // （簡略化：yaw回転のみ考慮）
        double current_yaw = extractYawFromQuaternion(current_pose.pose.pose.orientation);
        double predicted_yaw = current_yaw + predicted_yaw_change;
        
        // 予測位置計算（現在の速度ベース）
        double predicted_distance = (current_speed_kmh_ / 3.6) * prediction_time_; // km/h → m/s
        predicted_pose.pose.pose.position.x += predicted_distance * std::cos(predicted_yaw);
        predicted_pose.pose.pose.position.y += predicted_distance * std::sin(predicted_yaw);
        
        // 予測姿勢をクォータニオンに変換
        predicted_pose.pose.pose.orientation = createQuaternionFromYaw(predicted_yaw);
        
        return predicted_pose;
    }
    
    // ============================================
    // ユーティリティ関数
    // ============================================
    
    /**
     * クォータニオンからYaw角度を抽出
     */
    double extractYawFromQuaternion(const geometry_msgs::msg::Quaternion& quat)
    {
        double siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y);
        double cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z);
        return std::atan2(siny_cosp, cosy_cosp);
    }
    
    /**
     * Yaw角度からクォータニオンを生成
     */
    geometry_msgs::msg::Quaternion createQuaternionFromYaw(double yaw)
    {
        geometry_msgs::msg::Quaternion quat;
        quat.x = 0.0;
        quat.y = 0.0;
        quat.z = std::sin(yaw / 2.0);
        quat.w = std::cos(yaw / 2.0);
        return quat;
    }
    
    // ============================================
    // 結果発行
    // ============================================
    
    /**
     * 統合結果の発行とログ出力
     */
    void publishResults(const geometry_msgs::msg::PoseWithCovarianceStamped& fused_pose,
                       DrivingState state, const SensorWeights& weights)
    {
        // PoseWithCovarianceStamped発行
        pub_fused_pose_->publish(fused_pose);
        
        // Odometry形式でも発行
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header = fused_pose.header;
        odom_msg.pose = fused_pose.pose;
        pub_fused_odom_->publish(odom_msg);
        
        // 定期的にステータスログ出力
        static int log_counter = 0;
        if (++log_counter % 50 == 0) { // 50回に1回
            std::string state_str;
            switch (state) {
                case DrivingState::STRAIGHT_DRIVING: state_str = "直線"; break;
                case DrivingState::CORNER_DRIVING: state_str = "コーナー"; break;
                case DrivingState::PIT_LANE: state_str = "ピット"; break;
                default: state_str = "不明"; break;
            }
            
            RCLCPP_INFO(this->get_logger(),
                "状態: %s (%.1f km/h) | GNSS重み: %.2f | IMU重み: %.2f",
                state_str.c_str(), current_speed_kmh_,
                weights.gnss_weight, weights.imu_weight);
        }
    }
    
    // ============================================
    // メンバ変数
    // ============================================
    
    // パラメータ
    double update_frequency_;
    double prediction_time_;
    double gnss_weight_default_, imu_weight_default_;
    double high_speed_threshold_, corner_speed_threshold_;
    double straight_gnss_weight_, straight_imu_weight_;
    double corner_gnss_weight_, corner_imu_weight_;
    double pit_gnss_weight_, pit_imu_weight_;
    int history_size_;
    
    // ROSインターフェース
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_fused_pose_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_fused_odom_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_gnss_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr sub_velocity_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // データ管理
    std::mutex data_mutex_;
    std::deque<geometry_msgs::msg::PoseWithCovarianceStamped> gnss_history_;
    std::deque<sensor_msgs::msg::Imu> imu_history_;
    rclcpp::Time gnss_last_time_, imu_last_time_, velocity_last_time_;
    
    // 状態管理
    bool is_initialized_;
    double current_speed_kmh_;
    geometry_msgs::msg::PoseWithCovarianceStamped latest_fused_pose_;
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RacingSensorFusion>());
    rclcpp::shutdown();
    return 0;
}
