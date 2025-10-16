#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cmath>

class GoalLineChecker : public rclcpp::Node
{
public:
  GoalLineChecker() : Node("goal_line_checker_node")
  {
    in_threshold_ = 2.0;
    out_threshold_ = 3.0;
    is_in_goal_area_ = false;
    lap_count_ = 0;
    goal_received_ = false;
    odom_count_ = 0;
    
    // ★★★ 初期値は後で取得するため、仮の値を設定 ★★★
    base_velocity_ = 3.0;  // デフォルト値（取得失敗時のフォールバック）
    velocity_increment_ = 0.5;
    is_updating_params_ = false;
    initial_velocity_acquired_ = false;
    
    param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, "/simple_pure_pursuit_node");
    
    goal_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/planning/mission_planning/goal", rclcpp::QoS(1).transient_local(),
      std::bind(&GoalLineChecker::goalCallback, this, std::placeholders::_1));
    
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/localization/kinematic_state", 10,
      std::bind(&GoalLineChecker::odomCallback, this, std::placeholders::_1));
    
    RCLCPP_INFO(this->get_logger(), "===================================");
    RCLCPP_INFO(this->get_logger(), "ゴールライン通過判定ノード 起動");
    RCLCPP_INFO(this->get_logger(), "===================================");
    
    // ★★★ launch ファイルから初期速度を取得 ★★★
    getInitialVelocity();
  }

private:
  // ★★★ 初期速度を取得する関数（新規追加）★★★
  void getInitialVelocity() {
    RCLCPP_INFO(this->get_logger(), "初期速度を取得中...");
    
    // パラメータクライアントが準備できるまで待機
    if (!param_client_->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_WARN(this->get_logger(), 
                  "⚠️  パラメータサービスが利用できません。デフォルト値 %.1f m/s を使用", 
                  base_velocity_);
      initial_velocity_acquired_ = true;
      return;
    }
    
    // external_target_vel パラメータを取得
    auto future = param_client_->get_parameters({"external_target_vel"});
    
    // 非同期で取得結果を処理
    future.then([this](std::shared_future<std::vector<rclcpp::Parameter>> result_future) {
      try {
        auto params = result_future.get();
        if (!params.empty()) {
          base_velocity_ = params[0].as_double();
          RCLCPP_INFO(this->get_logger(), 
                      "✅ 初期速度を取得: %.1f m/s (launch設定値)", 
                      base_velocity_);
        } else {
          RCLCPP_WARN(this->get_logger(), 
                      "⚠️  パラメータ取得失敗。デフォルト値 %.1f m/s を使用", 
                      base_velocity_);
        }
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), 
                     "❌ パラメータ取得エラー: %s。デフォルト値 %.1f m/s を使用", 
                     e.what(), base_velocity_);
      }
      
      initial_velocity_acquired_ = true;
      RCLCPP_INFO(this->get_logger(), "速度増加量: %.1f m/s/lap", velocity_increment_);
      RCLCPP_INFO(this->get_logger(), "");
    });
  }

  void goalCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    goal_x_ = msg->pose.position.x;
    goal_y_ = msg->pose.position.y;
    if (!goal_received_) {
      goal_received_ = true;
      RCLCPP_INFO(this->get_logger(), "✓ ゴール位置受信: (%.2f, %.2f)", goal_x_, goal_y_);
    }
  }
  
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    odom_count_++;
    
    // ★★★ 初期速度取得が完了するまで待機 ★★★
    if (!initial_velocity_acquired_) {
      if (odom_count_ % 50 == 0) {
        RCLCPP_INFO(this->get_logger(), "初期速度取得待機中...");
      }
      return;
    }
    
    if (!goal_received_) return;
    
    double dx = msg->pose.pose.position.x - goal_x_;
    double dy = msg->pose.pose.position.y - goal_y_;
    double distance = std::sqrt(dx * dx + dy * dy);
    
    // 実速度も計算
    double actual_vel = std::sqrt(
      msg->twist.twist.linear.x * msg->twist.twist.linear.x +
      msg->twist.twist.linear.y * msg->twist.twist.linear.y
    );
    
    if (odom_count_ % 50 == 0) {
      RCLCPP_INFO(this->get_logger(), "[距離] %.2f m [実速度] %.2f m/s", distance, actual_vel);
    }
    
    if (!is_in_goal_area_ && distance < in_threshold_) {
      is_in_goal_area_ = true;
      RCLCPP_INFO(this->get_logger(), "→ 進入 (実速度: %.2f m/s)", actual_vel);
    } else if (is_in_goal_area_ && distance > out_threshold_) {
      is_in_goal_area_ = false;
      lap_count_++;
      RCLCPP_INFO(this->get_logger(), "");
      RCLCPP_INFO(this->get_logger(), "🏁 通過！ラップ: %d", lap_count_);
      
      if (!is_updating_params_) {
        updateParams();
      }
    }
  }
  
  void updateParams() {
    double new_vel = base_velocity_ + (velocity_increment_ * lap_count_);
    double old_vel = base_velocity_ + (velocity_increment_ * (lap_count_ - 1));
    
    RCLCPP_INFO(this->get_logger(), "🔧 速度変更: %.1f → %.1f m/s", old_vel, new_vel);
    is_updating_params_ = true;
    
    param_client_->set_parameters(
      {rclcpp::Parameter("external_target_vel", new_vel)},
      [this, new_vel](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> f) {
        try {
          auto r = f.get();
          if (!r.empty() && r[0].successful) {
            RCLCPP_INFO(this->get_logger(), "✅ 成功: %.1f m/s", new_vel);
          } else {
            RCLCPP_ERROR(this->get_logger(), "❌ 失敗");
          }
        } catch (...) {
          RCLCPP_ERROR(this->get_logger(), "❌ 例外発生");
        }
        is_updating_params_ = false;
        RCLCPP_INFO(this->get_logger(), "");
      });
  }
  
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  std::shared_ptr<rclcpp::AsyncParametersClient> param_client_;
  
  double goal_x_, goal_y_;
  bool goal_received_, is_in_goal_area_, is_updating_params_;
  int lap_count_, odom_count_;
  double in_threshold_, out_threshold_;
  double base_velocity_, velocity_increment_;
  bool initial_velocity_acquired_;  // ★★★ 初期速度取得完了フラグ ★★★
};

int main(int argc, char * argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GoalLineChecker>());
  rclcpp::shutdown();
  return 0;
}
