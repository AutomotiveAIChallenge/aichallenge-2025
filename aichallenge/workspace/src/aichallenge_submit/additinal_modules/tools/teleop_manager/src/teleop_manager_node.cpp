#include "teleop_manager_node.hpp"

#include <algorithm>
#include <string>
#include <memory>
#include <utility> // for std::move

using namespace std::chrono_literals;
using std::placeholders::_1;

TeleopManagerNode::TeleopManagerNode()
: Node("teleop_manager_node"),
  joy_active_(false),
  ack_active_(false),
  joy_speed_(0.0),
  joy_steer_(0.0),
  current_lap_(0.0f),
  prev_start_pressed_(false),
  prev_stop_pressed_(false),
  prev_steer_inc_pressed_(false),
  prev_steer_dec_pressed_(false),
  prev_speed_inc_pressed_(false),
  prev_speed_dec_pressed_(false),
  prev_scale_inc_pressed_(false),
  prev_scale_dec_pressed_(false),
  prev_awsim_button_pressed_(false),
  prev_reset_button_pressed_(false)
{
  // シミュレーション時刻を使用するようにノードを設定します。
  this->set_parameter(rclcpp::Parameter("use_sim_time", true));

  // --- Parameter Declaration & Retrieval ---
  declare_parameter<double>("speed_scale", 1.0);
  declare_parameter<double>("steer_scale", 1.0);
  declare_parameter<int>("joy_button_index",   2);
  declare_parameter<int>("ack_button_index",   3);
  declare_parameter<int>("start_button_index", 9);
  declare_parameter<int>("stop_button_index",  8);
  declare_parameter<int>("awsim_button_index", 2);
  declare_parameter<int>("reset_button_index", 7);
  declare_parameter<double>("timer_hz", 40.0);
  declare_parameter<double>("joy_timeout_sec", 0.5);

  get_parameter("speed_scale", speed_scale_);
  get_parameter("steer_scale", steer_scale_);
  get_parameter("joy_button_index", joy_button_index_);
  get_parameter("ack_button_index", ack_button_index_);
  get_parameter("start_button_index", start_button_index_);
  get_parameter("stop_button_index", stop_button_index_);
  get_parameter("awsim_button_index", awsim_button_index_);
  get_parameter("reset_button_index", reset_button_index_);
  get_parameter("timer_hz", timer_hz_);
  get_parameter("joy_timeout_sec", joy_timeout_sec_);

  // Initialize with new message structure
  last_autonomy_msg_.longitudinal.speed = 0.0;
  last_autonomy_msg_.lateral.steering_tire_angle = 0.0;
  last_joy_msg_time_ = this->get_clock()->now();

  // --- Subscriber / Publisher Setup ---
  joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
    "/joy", 10, std::bind(&TeleopManagerNode::joy_callback, this, _1));

  ack_sub_ = create_subscription<autoware_auto_control_msgs::msg::AckermannControlCommand>(
    "/ackermann_cmd", 10, std::bind(&TeleopManagerNode::ack_callback, this, _1));

  status_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
    "/aichallenge/awsim/status", 10, std::bind(&TeleopManagerNode::status_callback, this, _1));

  drive_pub_   = create_publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>("/cmd_drive", 10);
  trigger_pub_ = create_publisher<std_msgs::msg::Bool>("/rosbag2_recorder/trigger", 10);

  awsim_trigger_pub_ = create_publisher<std_msgs::msg::Bool>("/awsim/control_mode_request_topic", 10);

  reset_publisher_ = create_publisher<std_msgs::msg::Empty>("/aichallenge/awsim/reset", 10);
  initialpose_publisher_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 10);

  steer_inc_pub_ = create_publisher<std_msgs::msg::Bool>("/steer_offset_inc", 10);
  steer_dec_pub_ = create_publisher<std_msgs::msg::Bool>("/steer_offset_dec", 10);
  speed_inc_pub_ = create_publisher<std_msgs::msg::Bool>("/speed_offset_inc", 10);
  speed_dec_pub_ = create_publisher<std_msgs::msg::Bool>("/speed_offset_dec", 10);

  // --- Timer (for periodic command output) ---
  timer_ = create_wall_timer(
    std::chrono::duration<double>(1.0 / timer_hz_),
    std::bind(&TeleopManagerNode::timer_callback, this));
}

bool TeleopManagerNode::check_button_press(bool curr, bool &prev_flag)
{
  if (curr && !prev_flag) {
    prev_flag = true;
    return true;
  } else if (!curr) {
    prev_flag = false;
  }
  return false;
}

void TeleopManagerNode::status_callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() >= 2) {
    current_lap_ = msg->data[1];
  }
}

void TeleopManagerNode::joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
  last_joy_msg_time_ = this->get_clock()->now();

  // 0) Start/stop buttons (with debounce)
  bool curr_start = (msg->buttons.size() > start_button_index_
                     && msg->buttons[start_button_index_] == 1);
  bool curr_stop  = (msg->buttons.size() > stop_button_index_
                     && msg->buttons[stop_button_index_]  == 1);
  if (check_button_press(curr_start, prev_start_pressed_)) {
    std_msgs::msg::Bool b; b.data = true;
    trigger_pub_->publish(b);
  }
  if (check_button_press(curr_stop, prev_stop_pressed_)) {
    std_msgs::msg::Bool b; b.data = false;
    trigger_pub_->publish(b);
  }

  bool curr_awsim_button = (msg->buttons.size() > awsim_button_index_
                            && msg->buttons[awsim_button_index_] == 1);
  if (check_button_press(curr_awsim_button, prev_awsim_button_pressed_)) {
    std_msgs::msg::Bool b; b.data = true;
    awsim_trigger_pub_->publish(b);
    RCLCPP_INFO(get_logger(), "Published true to /awsim/control_mode_request_topic");
  }

  bool curr_reset_button = (msg->buttons.size() > reset_button_index_
                            && msg->buttons[reset_button_index_] == 1);
  if (check_button_press(curr_reset_button, prev_reset_button_pressed_)) {
    auto empty_msg = std::make_unique<std_msgs::msg::Empty>();
    reset_publisher_->publish(std::move(empty_msg));
    RCLCPP_INFO(get_logger(), "Published Empty message to /aichallenge/awsim/reset");

    auto pose_msg = std::make_unique<geometry_msgs::msg::PoseWithCovarianceStamped>();
    pose_msg->header.stamp = this->get_clock()->now();
    pose_msg->header.frame_id = "map";
    pose_msg->pose.pose.position.x = 89666.01577151686;
    pose_msg->pose.pose.position.y = 43124.3307874416;
    pose_msg->pose.pose.position.z = 0.0;
    pose_msg->pose.pose.orientation.x = 0.0;
    pose_msg->pose.pose.orientation.y = 0.0;
    pose_msg->pose.pose.orientation.z = -0.9683930510846941;
    pose_msg->pose.pose.orientation.w = 0.24942914547196962;

    initialpose_publisher_->publish(std::move(pose_msg));
    RCLCPP_INFO(get_logger(), "Published initial pose to /initialpose");
  }

  // 1) Mode selection
  bool joy_pressed = (msg->buttons.size() > joy_button_index_
                      && msg->buttons[joy_button_index_] == 1);
  bool ack_pressed = (msg->buttons.size() > ack_button_index_
                      && msg->buttons[ack_button_index_] == 1);
  if (ack_pressed) {
    ack_active_ = true; joy_active_ = false;
  } else if (joy_pressed) {
    joy_active_ = true; ack_active_ = false;
  } else {
    joy_active_ = false; ack_active_ = false;
  }

  // 2) Calculate speed/steer in Joy mode
  if (joy_active_) {
    double raw_speed = (msg->axes.size() > 1 ? msg->axes[1] : 0.0);
    double raw_steer = (msg->axes.size() > 3 ? msg->axes[3] : 0.0);
    joy_speed_ = raw_speed * speed_scale_;
    joy_steer_ = raw_steer * steer_scale_;
  }

  // 3) D-pad for offset triggers (with debounce)
  double a6 = (msg->axes.size() > 6 ? msg->axes[6] : 0.0);
  double a7 = (msg->axes.size() > 7 ? msg->axes[7] : 0.0);

  bool steer_inc = std::abs(a6 + 1.0) < 1e-3;  // Right
  bool steer_dec = std::abs(a6 - 1.0) < 1e-3;  // Left
  bool speed_inc = std::abs(a7 - 1.0) < 1e-3;  // Up
  bool speed_dec = std::abs(a7 + 1.0) < 1e-3;  // Down

  if (check_button_press(steer_inc, prev_steer_inc_pressed_)) {
    std_msgs::msg::Bool b; b.data = true; steer_inc_pub_->publish(b);
  }
  if (check_button_press(steer_dec, prev_steer_dec_pressed_)) {
    std_msgs::msg::Bool b; b.data = true; steer_dec_pub_->publish(b);
  }
  if (check_button_press(speed_inc, prev_speed_inc_pressed_)) {
    std_msgs::msg::Bool b; b.data = true; speed_inc_pub_->publish(b);
  }
  if (check_button_press(speed_dec, prev_speed_dec_pressed_)) {
    std_msgs::msg::Bool b; b.data = true; speed_dec_pub_->publish(b);
  }

  // 4) R1/L1 for dynamic steer_scale adjustment (with debounce)
  bool scale_inc = (msg->buttons.size() > 5 && msg->buttons[5] == 1); // R1
  bool scale_dec = (msg->buttons.size() > 4 && msg->buttons[4] == 1); // L1
  if (check_button_press(scale_inc, prev_scale_inc_pressed_)) {
    steer_scale_ = std::round((steer_scale_ + 0.1) * 10.0) / 10.0;
    if (steer_scale_ < 0.1) steer_scale_ = 0.1;
    RCLCPP_INFO(get_logger(), "steer_scale = %.1f", steer_scale_);
  }
  if (check_button_press(scale_dec, prev_scale_dec_pressed_)) {
    steer_scale_ = std::max(steer_scale_ - 0.1, 0.0);
    steer_scale_ = std::round(steer_scale_ * 10.0) / 10.0;
    if (steer_scale_ < 0.1 && steer_scale_ != 0.0) steer_scale_ = 0.1;
    RCLCPP_INFO(get_logger(), "steer_scale = %.1f", steer_scale_);
  }
}

void TeleopManagerNode::ack_callback(const autoware_auto_control_msgs::msg::AckermannControlCommand::SharedPtr msg)
{
  last_autonomy_msg_ = *msg;
  ack_received_ = true;
}

void TeleopManagerNode::timer_callback()
{
  autoware_auto_control_msgs::msg::AckermannControlCommand out;
  rclcpp::Time current_time = this->get_clock()->now();

  if ((current_time - last_joy_msg_time_).seconds() > joy_timeout_sec_) {
    if (joy_active_ || ack_active_) {
      RCLCPP_WARN(get_logger(), "Joy message timed out! Stopping the vehicle.");
    }
    joy_active_ = false;
    ack_active_ = false;
  }

  if (joy_active_) {
    out.longitudinal.acceleration = joy_speed_;
    out.lateral.steering_tire_angle = joy_steer_;
    out.lateral.steering_tire_rotation_rate = 1.0;
  } else if (ack_active_) {
    out = last_autonomy_msg_;
    out.lateral.steering_tire_rotation_rate = 0.5;
  } else {
    out.longitudinal.acceleration = 0.0;
    out.lateral.steering_tire_angle = 0.0;
    out.lateral.steering_tire_rotation_rate = 0.0;
  }

  out.stamp = current_time;
  out.longitudinal.stamp = current_time;
  out.lateral.stamp = current_time;

  out.longitudinal.speed = current_lap_;

  drive_pub_->publish(out);
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TeleopManagerNode>());
  rclcpp::shutdown();
  return 0;
}