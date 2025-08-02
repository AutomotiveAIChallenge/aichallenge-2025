#!/bin/bash
timeout 3s ros2 topic pub --once /d1/awsim/control_mode_request_topic std_msgs/msg/Bool data:\ true 
sleep 4
timeout 3s ros2 topic pub --once /d2/awsim/control_mode_request_topic std_msgs/msg/Bool data:\ true 
sleep 4
timeout 3s ros2 topic pub --once /d3/awsim/control_mode_request_topic std_msgs/msg/Bool data:\ true 
sleep 4
timeout 3s ros2 topic pub --once /d4/awsim/control_mode_request_topic std_msgs/msg/Bool data:\ true 
sleep 4
timeout 3s ros2 topic pub --once /d5/awsim/control_mode_request_topic std_msgs/msg/Bool data:\ true 
sleep 4
timeout 3s ros2 topic pub --once /d6/awsim/control_mode_request_topic std_msgs/msg/Bool data:\ true 
sleep 4
timeout 3s ros2 topic pub --once /d7/awsim/control_mode_request_topic std_msgs/msg/Bool data:\ true 
sleep 4
timeout 3s ros2 topic pub --once /d8/awsim/control_mode_request_topic std_msgs/msg/Bool data:\ true 
sleep 4
