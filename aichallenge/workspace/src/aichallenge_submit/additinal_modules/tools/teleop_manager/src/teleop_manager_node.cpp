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
  // 古いデバウンスフラグを削除
  prev_awsim_button_pressed_(false),
  prev_reset_button_pressed_(false),
  // スケール調整用の新しいデバウンスフラグを初期化
  prev_steer_scale_inc_pressed_(false),
  prev_steer_scale_dec_pressed_(false),
  prev_speed_scale_inc_pressed_(false),
  prev_speed_scale_dec_pressed_(false)
{
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
  declare_parameter<int>("dpad_lr_axis_index", 6); 
  declare_parameter<int>("dpad_ud_axis_index", 7); 

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
  get_parameter("dpad_lr_axis_index", dpad_lr_axis_index_);
  get_parameter("dpad_ud_axis_index", dpad_ud_axis_index_);

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

  
  // --- Timer ---
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

  // 1) Start/stop/AWSIM/Reset buttons (with debounce)
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

  // 2) Mode selection
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

  // 3) Calculate speed/steer in Joy mode (using current scales)
  if (joy_active_) {
    double raw_speed = (msg->axes.size() > 1 ? msg->axes[1] : 0.0);
    double raw_steer = (msg->axes.size() > 3 ? msg->axes[3] : 0.0);
    joy_speed_ = raw_speed * speed_scale_;
    joy_steer_ = raw_steer * steer_scale_;
  }

  // 4) D-pad for dynamic scale adjustment (with debounce)
  // パラメータ化された軸インデックスを使用
  double a_lr = (msg->axes.size() > dpad_lr_axis_index_ ? msg->axes[dpad_lr_axis_index_] : 0.0);
  double a_ud = (msg->axes.size() > dpad_ud_axis_index_ ? msg->axes[dpad_ud_axis_index_] : 0.0);

  // コントローラによって軸の+1/-1が逆の場合があるため、元のロジックを踏襲
  bool steer_scale_inc = std::abs(a_lr + 1.0) < 1e-3;  // 右 (軸値 -1.0)
  bool steer_scale_dec = std::abs(a_lr - 1.0) < 1e-3;  // 左 (軸値  1.0)
  bool speed_scale_inc = std::abs(a_ud - 1.0) < 1e-3;  // 上 (軸値  1.0)
  bool speed_scale_dec = std::abs(a_ud + 1.0) < 1e-3;  // 下 (軸値 -1.0)

  // Steer Scale 調整 (左右)
  if (check_button_press(steer_scale_inc, prev_steer_scale_inc_pressed_)) {
    steer_scale_ = std::round((steer_scale_ + 0.1) * 10.0) / 10.0;
    RCLCPP_INFO(get_logger(), "steer_scale increased to: %.1f", steer_scale_);
  }
  if (check_button_press(steer_scale_dec, prev_steer_scale_dec_pressed_)) {
    steer_scale_ = std::max(steer_scale_ - 0.1, 0.0); // 0.0未満にならないように
    steer_scale_ = std::round(steer_scale_ * 10.0) / 10.0;
    RCLCPP_INFO(get_logger(), "steer_scale decreased to: %.1f", steer_scale_);
  }

  // Speed Scale 調整 (上下)
  if (check_button_press(speed_scale_inc, prev_speed_scale_inc_pressed_)) {
    speed_scale_ = std::round((speed_scale_ + 0.1) * 10.0) / 10.0;
    RCLCPP_INFO(get_logger(), "speed_scale increased to: %.1f", speed_scale_);
  }
  if (check_button_press(speed_scale_dec, prev_speed_scale_dec_pressed_)) {
    speed_scale_ = std::max(speed_scale_ - 0.1, 0.0); // 0.0未満にならないように
    speed_scale_ = std::round(speed_scale_ * 10.0) / 10.0;
    RCLCPP_INFO(get_logger(), "speed_scale decreased to: %.1f", speed_scale_);
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
    // joy_speed_ と joy_steer_ は joy_callback でスケール適用済みの値
    out.longitudinal.acceleration = joy_speed_;
    out.lateral.steering_tire_angle = joy_steer_;
    out.lateral.steering_tire_rotation_rate = 1.0;
  } else if (ack_active_) {
    out = last_autonomy_msg_;
    out.lateral.steering_tire_rotation_rate = 0.5;
  } else {
    // Stop
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