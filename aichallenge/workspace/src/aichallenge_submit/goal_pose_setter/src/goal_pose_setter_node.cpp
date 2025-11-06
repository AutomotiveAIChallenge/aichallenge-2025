#include "goal_pose_setter_node.hpp"

GoalPosePublisher::GoalPosePublisher() : Node("goal_pose_publisher")
{
    const auto rt_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    const auto rv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
    
    // ゴール位置をパブリッシュするパブリッシャーを作成
    goal_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/planning/mission_planning/goal", rt_qos);
    
    ekf_trigger_client_ = this->create_client<std_srvs::srv::SetBool>("/localization/trigger_node");
    
    // パラメータからゴール位置を読み込み
    this->declare_parameter("goal.position.x", 89653.7);
    this->declare_parameter("goal.position.y", 43122.5);
    this->declare_parameter("goal.position.z", 0.0);
    this->declare_parameter("goal.orientation.x", 0.0);
    this->declare_parameter("goal.orientation.y", 0.0);
    this->declare_parameter("goal.orientation.z", -0.971732);
    this->declare_parameter("goal.orientation.w", 0.236088);
    
    goal_pose_.header.frame_id = "map";
    goal_pose_.pose.position.x = this->get_parameter("goal.position.x").as_double();
    goal_pose_.pose.position.y = this->get_parameter("goal.position.y").as_double();
    goal_pose_.pose.position.z = this->get_parameter("goal.position.z").as_double();
    goal_pose_.pose.orientation.x = this->get_parameter("goal.orientation.x").as_double();
    goal_pose_.pose.orientation.y = this->get_parameter("goal.orientation.y").as_double();
    goal_pose_.pose.orientation.z = this->get_parameter("goal.orientation.z").as_double();
    goal_pose_.pose.orientation.w = this->get_parameter("goal.orientation.w").as_double();
    
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "Goal Pose Setter 起動");
    RCLCPP_INFO(this->get_logger(), "========================================");
    RCLCPP_INFO(this->get_logger(), "ゴール位置: (%.2f, %.2f, %.2f)", 
                goal_pose_.pose.position.x, 
                goal_pose_.pose.position.y, 
                goal_pose_.pose.position.z);
    
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(300),
        std::bind(&GoalPosePublisher::on_timer, this));
}

void GoalPosePublisher::on_timer()
{
    if (++delay_count_ <= 10) {
        return;
    }

    // ゴール位置を定期的にパブリッシュ（起動後3秒後から）
    if (delay_count_ == 11) {
        goal_pose_.header.stamp = this->now();
        goal_pose_pub_->publish(goal_pose_);
        RCLCPP_INFO(this->get_logger(), "✓ ゴール位置をパブリッシュしました");
    }

    if (!stop_initializing_pose_) {
        if (ekf_trigger_client_->service_is_ready()) {
            const auto req = std::make_shared<std_srvs::srv::SetBool::Request>();
            req->data = true;
            ekf_trigger_client_->async_send_request(req, [this](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future)
            {
                stop_initializing_pose_ = future.get()->success;
                delay_count_ = 0;
                RCLCPP_INFO(this->get_logger(), "Complete localization trigger");
            });
            RCLCPP_INFO(this->get_logger(), "Call localization trigger");
        }
        return;
    }
    
    // ローカライゼーション完了後、ゴール位置を継続的にパブリッシュ（5秒ごと）
    if (delay_count_ % 50 == 0) {  // 300ms * 50 = 15秒ごと
        goal_pose_.header.stamp = this->now();
        goal_pose_pub_->publish(goal_pose_);
    }
}

int main(int argc, char const *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GoalPosePublisher>());
    rclcpp::shutdown();
    return 0;
}
