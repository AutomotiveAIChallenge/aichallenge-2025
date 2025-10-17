#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cmath>

/**
 * ゴールライン通過判定＆パラメータ変更ノード（非同期版）
 * 
 * 【機能】
 * - ゴールライン通過を検出
 * - ラップごとに速度を0.5m/s増加
 * - 非同期でパラメータを変更（タイムアウトなし）
 */

class GoalLineChecker : public rclcpp::Node
{
public:
  GoalLineChecker() : Node("goal_line_checker_node")
  {
    // ===========================================
    // パラメータ設定
    // ===========================================
    
    in_threshold_ = 2.0;
    out_threshold_ = 3.0;
    
    is_in_goal_area_ = false;
    lap_count_ = 0;
    goal_received_ = false;
    odom_count_ = 0;
    
    // 初期速度（ラップ0の速度）
    base_velocity_ = 4.0;
    velocity_increment_ = 0.5;  // ラップごとの速度増加量
    
    // パラメータ変更中フラグ
    is_updating_params_ = false;
    
    // ===========================================
    // パラメータクライアントの作成
    // ===========================================
    
    param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(
      this, 
      "/simple_pure_pursuit_node"
    );
    
    // ===========================================
    // サブスクライバー設定
    // ===========================================
    
    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/planning/mission_planning/goal",
      rclcpp::QoS(1).transient_local(),
      std::bind(&GoalLineChecker::goalCallback, this, std::placeholders::_1)
    );
    
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/localization/kinematic_state",
      10,
      std::bind(&GoalLineChecker::odomCallback, this, std::placeholders::_1)
    );
    
    RCLCPP_INFO(this->get_logger(), "===================================");
    RCLCPP_INFO(this->get_logger(), "ゴールライン通過判定ノード 起動");
    RCLCPP_INFO(this->get_logger(), "===================================");
    RCLCPP_INFO(this->get_logger(), "判定エリア進入閾値: %.2f m", in_threshold_);
    RCLCPP_INFO(this->get_logger(), "判定エリア退出閾値: %.2f m", out_threshold_);
    RCLCPP_INFO(this->get_logger(), "初期速度: %.1f m/s", base_velocity_);
    RCLCPP_INFO(this->get_logger(), "速度増加量: %.1f m/s/lap", velocity_increment_);
    RCLCPP_INFO(this->get_logger(), "");
  }

private:
  void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    goal_x_ = msg->pose.position.x;
    goal_y_ = msg->pose.position.y;
    
    if (!goal_received_) {
      goal_received_ = true;
      RCLCPP_INFO(this->get_logger(), "");
      RCLCPP_INFO(this->get_logger(), "✓✓✓ ゴール位置受信成功 ✓✓✓");
      RCLCPP_INFO(this->get_logger(), "  X座標: %.2f m", goal_x_);
      RCLCPP_INFO(this->get_logger(), "  Y座標: %.2f m", goal_y_);
      RCLCPP_INFO(this->get_logger(), "");
    }
  }
  
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    odom_count_++;
    
    if (!goal_received_) {
      if (odom_count_ % 10 == 0) {
        RCLCPP_WARN(this->get_logger(), 
                    "車両位置は受信中ですが、ゴール位置がまだ設定されていません");
      }
      return;
    }
    
    double vehicle_x = msg->pose.pose.position.x;
    double vehicle_y = msg->pose.pose.position.y;
    
    double dx = vehicle_x - goal_x_;
    double dy = vehicle_y - goal_y_;
    double distance = std::sqrt(dx * dx + dy * dy);
    
    if (odom_count_ % 50 == 0) {
      RCLCPP_INFO(this->get_logger(), 
                  "[デバッグ] 現在の距離: %.2f m (閾値: 進入=%.2f, 退出=%.2f)", 
                  distance, in_threshold_, out_threshold_);
    }
    
    if (!is_in_goal_area_) {
      if (distance < in_threshold_) {
        is_in_goal_area_ = true;
        RCLCPP_INFO(this->get_logger(), "");
        RCLCPP_INFO(this->get_logger(), "→→→ ゴールエリア進入！ ←←←");
        RCLCPP_INFO(this->get_logger(), "  距離: %.2f m", distance);
        RCLCPP_INFO(this->get_logger(), "");
      }
    }
    else {
      if (distance > out_threshold_) {
        is_in_goal_area_ = false;
        lap_count_++;
        
        RCLCPP_INFO(this->get_logger(), "");
        RCLCPP_INFO(this->get_logger(), "🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁");
        RCLCPP_INFO(this->get_logger(), "  ゴールライン通過検出！");
        RCLCPP_INFO(this->get_logger(), "  現在のラップ数: %d", lap_count_);
        RCLCPP_INFO(this->get_logger(), "🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁🏁");
        RCLCPP_INFO(this->get_logger(), "");
        
        // ★ パラメータ変更処理を非同期で実行
        if (!is_updating_params_) {
          updateControlParametersAsync();
        } else {
          RCLCPP_WARN(this->get_logger(), "⚠️  前回のパラメータ変更が完了していません。スキップします。");
        }
      }
    }
  }
  
  // ===========================================
  // パラメータ変更処理（非同期版）
  // ===========================================
  void updateControlParametersAsync()
  {
    // 新しい目標速度を計算
    double new_velocity = base_velocity_ + (velocity_increment_ * lap_count_);
    
    RCLCPP_INFO(this->get_logger(), "");
    RCLCPP_INFO(this->get_logger(), "🔧 パラメータ変更開始...");
    RCLCPP_INFO(this->get_logger(), "  ラップ数: %d", lap_count_);
    RCLCPP_INFO(this->get_logger(), "  新しい目標速度: %.1f m/s", new_velocity);
    
    // 変更中フラグを立てる
    is_updating_params_ = true;
    
    // パラメータの設定
    auto parameters = std::vector<rclcpp::Parameter>{
      rclcpp::Parameter("external_target_vel", new_velocity)
    };
    
    // 非同期でパラメータを設定し、完了時にコールバックを呼ぶ
    param_client_->set_parameters(
      parameters,
      [this, new_velocity](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future) {
        try {
          auto results = future.get();
          if (!results.empty() && results[0].successful) {
            RCLCPP_INFO(this->get_logger(), "✅ パラメータ変更成功！");
            RCLCPP_INFO(this->get_logger(), "  external_target_vel = %.1f m/s", new_velocity);
          } else {
            RCLCPP_ERROR(this->get_logger(), "❌ パラメータ変更失敗");
            if (!results.empty()) {
              RCLCPP_ERROR(this->get_logger(), "  理由: %s", results[0].reason.c_str());
            }
          }
        } catch (const std::exception& e) {
          RCLCPP_ERROR(this->get_logger(), "❌ パラメータ変更エラー: %s", e.what());
        }
        
        // 変更完了
        is_updating_params_ = false;
        RCLCPP_INFO(this->get_logger(), "");
      }
    );
  }
  
  // ===========================================
  // メンバ変数
  // ===========================================
  
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::shared_ptr<rclcpp::AsyncParametersClient> param_client_;
  
  double goal_x_;
  double goal_y_;
  bool goal_received_;
  
  double in_threshold_;
  double out_threshold_;
  
  bool is_in_goal_area_;
  int lap_count_;
  int odom_count_;
  
  double base_velocity_;
  double velocity_increment_;
  
  bool is_updating_params_;  // パラメータ変更中フラグ
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GoalLineChecker>());
  rclcpp::shutdown();
  return 0;
}
