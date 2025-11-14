#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist_with_covariance_stamped.hpp"
#include "sensor_msgs/msg/imu.hpp"

class ImuGnssPoser : public rclcpp::Node
{

public:
    ImuGnssPoser() : Node("imu_gnss_poser")
    {
        const auto rv_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable();
        const auto rt_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

        pub_pose_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/localization/imu_gnss_poser/pose_with_covariance", rv_qos);
        pub_initial_pose_3d_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/localization/initial_pose3d", rt_qos);
        
        sub_twist_ = this->create_subscription<geometry_msgs::msg::TwistWithCovarianceStamped>(
            "/localization/twist_with_covariance", rv_qos,
            std::bind(&ImuGnssPoser::twist_callback, this, std::placeholders::_1));
        
        sub_gnss_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/sensing/gnss/pose_with_covariance", rv_qos,
            std::bind(&ImuGnssPoser::gnss_callback, this, std::placeholders::_1));
        
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/sensing/imu/imu_raw", rv_qos,
            std::bind(&ImuGnssPoser::imu_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "IMU GNSS Poser initialized");
        RCLCPP_INFO(this->get_logger(), "Strategy: Minimize IMU orientation trust, use GNSS position primarily");
    }

private:

    void gnss_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {

        // 共分散行列の設定
        // 位置(X,Y,Z)の共分散は小さく（GNSSは信頼できる）
        msg->pose.covariance[7*0] = 0.1;   // X position covariance
        msg->pose.covariance[7*1] = 0.1;   // Y position covariance
        msg->pose.covariance[7*2] = 0.1;   // Z position covariance
        
        // 姿勢(Roll, Pitch, Yaw)の共分散は非常に大きく（信頼しない）
        // 特にYaw（方位角）はIMUの信頼度を最小化
        msg->pose.covariance[7*3] = 100000.0;  // Roll covariance
        msg->pose.covariance[7*4] = 100000.0;  // Pitch covariance
        msg->pose.covariance[7*5] = 100000.0;  // Yaw covariance（最も重要！）

        // IMU orientationを初期値として使用するが、信頼度は極小
        // GNSSのorientationが無効な場合のみIMUを使用
        if (std::isnan(msg->pose.pose.orientation.x) ||
            std::isnan(msg->pose.pose.orientation.y) ||
            std::isnan(msg->pose.pose.orientation.z) ||
            std::isnan(msg->pose.pose.orientation.w) ||
            (msg->pose.pose.orientation.x == 0 &&
             msg->pose.pose.orientation.y == 0 &&
             msg->pose.pose.orientation.z == 0 &&
             msg->pose.pose.orientation.w == 0))
        {
            // IMUのorientationを使用（初期化のみ）
            msg->pose.pose.orientation.x = imu_msg_.orientation.x;
            msg->pose.pose.orientation.y = imu_msg_.orientation.y;
            msg->pose.pose.orientation.z = imu_msg_.orientation.z;
            msg->pose.pose.orientation.w = imu_msg_.orientation.w;
            
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Using IMU orientation for initialization (low trust)");
        }
        
        // Pose更新を配信
        pub_pose_->publish(*msg);
        
        // 初期位置は1回だけ送信（EKF初期化用）
        // この制御により、初期化が完了するまで初期位置が更新され続ける
        if (!is_ekf_initialized_) {
            pub_initial_pose_3d_->publish(*msg);
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for EKF initialization...");
        }
    }

    void imu_callback(sensor_msgs::msg::Imu::SharedPtr msg) {
        imu_msg_ = *msg;
    }

    void twist_callback(const geometry_msgs::msg::TwistWithCovarianceStamped::SharedPtr)
    {
        // Twistが配信されたらEKF初期化完了
        if (!is_ekf_initialized_) {
            is_ekf_initialized_ = true;
            RCLCPP_INFO(this->get_logger(), "EKF initialization completed!");
        }
    }

    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_initial_pose_3d_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_gnss_;
    rclcpp::Subscription<geometry_msgs::msg::TwistWithCovarianceStamped>::SharedPtr sub_twist_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    sensor_msgs::msg::Imu imu_msg_;
    bool is_ekf_initialized_ = {false};
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImuGnssPoser>());
    rclcpp::shutdown();
    return 0;
}
