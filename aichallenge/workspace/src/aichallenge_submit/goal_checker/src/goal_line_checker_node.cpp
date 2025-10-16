#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float32.hpp>
#include <cmath>
#include <chrono>

class GoalLineChecker : public rclcpp::Node
{
public:
    GoalLineChecker() : Node("goal_line_checker")
    {
        // パラメータの宣言
        this->declare_parameter<double>("goal_x", 89750.8);
        this->declare_parameter<double>("goal_y", 43132.5);
        this->declare_parameter<double>("goal_tolerance", 5.0);  // ゴール判定の許容範囲（メートル）
        this->declare_parameter<double>("stop_time_threshold", 2.0);  // 停止判定時間（秒）
        this->declare_parameter<double>("velocity_threshold", 0.1);  // 停止とみなす速度（m/s）
        
        // パラメータの取得
        goal_x_ = this->get_parameter("goal_x").as_double();
        goal_y_ = this->get_parameter("goal_y").as_double();
        goal_tolerance_ = this->get_parameter("goal_tolerance").as_double();
        stop_time_threshold_ = this->get_parameter("stop_time_threshold").as_double();
        velocity_threshold_ = this->get_parameter("velocity_threshold").as_double();
        
        // サブスクライバーの設定
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/localization/kinematic_state", 10,
            std::bind(&GoalLineChecker::odomCallback, this, std::placeholders::_1));
            
        // パブリッシャーの設定
        goal_reached_pub_ = this->create_publisher<std_msgs::msg::Bool>("/goal_reached", 10);
        goal_status_pub_ = this->create_publisher<std_msgs::msg::String>("/goal_status", 10);
        distance_to_goal_pub_ = this->create_publisher<std_msgs::msg::Float32>("/distance_to_goal", 10);
        
        // 初期速度の取得を試みる
        getInitialVelocity();
        
        RCLCPP_INFO(this->get_logger(), "Goal Line Checker Node initialized");
        RCLCPP_INFO(this->get_logger(), "Goal position: x=%.2f, y=%.2f", goal_x_, goal_y_);
    }

private:
    void getInitialVelocity()
    {
        // パラメータクライアントを作成して初期速度を取得
        auto param_client = std::make_shared<rclcpp::AsyncParametersClient>(
            this, "/aichallenge_awsim_adapter");
        
        // パラメータが利用可能になるまで待機
        if (!param_client->wait_for_service(std::chrono::seconds(5))) {
            RCLCPP_WARN(this->get_logger(), 
                "Parameter service not available. Using default initial velocity: %.2f m/s", 
                initial_velocity_);
            return;
        }
        
        // パラメータを非同期で取得
        auto future = param_client->get_parameters({"initial_velocity"});
        
        // 結果を取得（修正部分：then()の代わりに直接処理）
        try {
            // タイムアウト付きで結果を待つ（最大2秒）
            if (future.wait_for(std::chrono::seconds(2)) == std::future_status::ready) {
                auto result = future.get();
                
                if (!result.empty()) {
                    for (const auto& param : result) {
                        if (param.get_name() == "initial_velocity") {
                            initial_velocity_ = param.as_double();
                            RCLCPP_INFO(this->get_logger(), 
                                "Successfully retrieved initial velocity: %.2f m/s", 
                                initial_velocity_);
                            
                            // ラップタイムの計測開始
                            if (initial_velocity_ > 0) {
                                lap_start_time_ = this->now();
                                is_timing_ = true;
                                RCLCPP_INFO(this->get_logger(), "Lap time measurement started");
                            }
                            break;
                        }
                    }
                } else {
                    RCLCPP_WARN(this->get_logger(), 
                        "No parameters found. Using default initial velocity: %.2f m/s", 
                        initial_velocity_);
                }
            } else {
                RCLCPP_WARN(this->get_logger(), 
                    "Timeout waiting for parameters. Using default initial velocity: %.2f m/s", 
                    initial_velocity_);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), 
                "Error retrieving parameters: %s. Using default initial velocity: %.2f m/s", 
                e.what(), initial_velocity_);
        }
    }
    
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // 現在位置の取得
        double current_x = msg->pose.pose.position.x;
        double current_y = msg->pose.pose.position.y;
        double current_velocity = std::sqrt(
            msg->twist.twist.linear.x * msg->twist.twist.linear.x +
            msg->twist.twist.linear.y * msg->twist.twist.linear.y
        );
        
        // ゴールまでの距離を計算
        double distance = std::sqrt(
            std::pow(current_x - goal_x_, 2) + 
            std::pow(current_y - goal_y_, 2)
        );
        
        // 距離をパブリッシュ
        auto distance_msg = std_msgs::msg::Float32();
        distance_msg.data = distance;
        distance_to_goal_pub_->publish(distance_msg);
        
        // ステータスメッセージの作成
        auto status_msg = std_msgs::msg::String();
        auto goal_msg = std_msgs::msg::Bool();
        
        // ゴール判定
        if (distance < goal_tolerance_) {
            // ゴール範囲内にいる場合
            if (current_velocity < velocity_threshold_) {
                // 速度が閾値以下の場合（停止状態）
                if (!is_stopped_) {
                    // 停止開始時刻を記録
                    stop_start_time_ = this->now();
                    is_stopped_ = true;
                }
                
                // 停止継続時間を計算
                double stopped_duration = (this->now() - stop_start_time_).seconds();
                
                if (stopped_duration >= stop_time_threshold_) {
                    // 必要な時間停止した場合、ゴール達成
                    if (!goal_reached_) {
                        goal_reached_ = true;
                        
                        // ラップタイムの計算と表示
                        if (is_timing_) {
                            double lap_time = (this->now() - lap_start_time_).seconds();
                            RCLCPP_INFO(this->get_logger(), "");
                            RCLCPP_INFO(this->get_logger(), "========================================");
                            RCLCPP_INFO(this->get_logger(), "    🏁 GOAL REACHED! 🏁");
                            RCLCPP_INFO(this->get_logger(), "    Lap Time: %.2f seconds", lap_time);
                            RCLCPP_INFO(this->get_logger(), "    Initial Velocity: %.2f m/s", initial_velocity_);
                            RCLCPP_INFO(this->get_logger(), "========================================");
                            RCLCPP_INFO(this->get_logger(), "");
                        } else {
                            RCLCPP_INFO(this->get_logger(), "Goal reached!");
                        }
                    }
                    
                    status_msg.data = "GOAL_REACHED";
                    goal_msg.data = true;
                } else {
                    // まだ停止時間が不足
                    status_msg.data = "STOPPING_AT_GOAL (Time: " + 
                        std::to_string(stopped_duration) + "/" + 
                        std::to_string(stop_time_threshold_) + "s)";
                    goal_msg.data = false;
                }
            } else {
                // 速度が速すぎる（まだ動いている）
                is_stopped_ = false;
                status_msg.data = "IN_GOAL_AREA (Velocity: " + 
                    std::to_string(current_velocity) + " m/s)";
                goal_msg.data = false;
            }
        } else {
            // ゴール範囲外
            is_stopped_ = false;
            goal_reached_ = false;
            status_msg.data = "APPROACHING (Distance: " + 
                std::to_string(distance) + " m)";
            goal_msg.data = false;
        }
        
        // メッセージをパブリッシュ
        goal_status_pub_->publish(status_msg);
        goal_reached_pub_->publish(goal_msg);
        
        // デバッグ出力（1秒に1回程度）
        static auto last_print_time = this->now();
        if ((this->now() - last_print_time).seconds() >= 1.0) {
            RCLCPP_INFO(this->get_logger(), 
                "Distance to goal: %.2f m, Velocity: %.2f m/s, Status: %s",
                distance, current_velocity, status_msg.data.c_str());
            last_print_time = this->now();
        }
    }
    
    // メンバ変数
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr goal_reached_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr goal_status_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr distance_to_goal_pub_;
    
    // ゴール関連のパラメータ
    double goal_x_;
    double goal_y_;
    double goal_tolerance_;
    double stop_time_threshold_;
    double velocity_threshold_;
    
    // 状態管理
    bool is_stopped_ = false;
    bool goal_reached_ = false;
    rclcpp::Time stop_start_time_;
    
    // ラップタイム計測用
    double initial_velocity_ = 0.0;  // デフォルト値を設定
    rclcpp::Time lap_start_time_;
    bool is_timing_ = false;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GoalLineChecker>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
