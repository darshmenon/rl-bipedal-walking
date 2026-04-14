#!/bin/bash
# Script to launch Gazebo simulation with bipedal robot

# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Source workspace
source "$(dirname "$0")/../ros2_ws/install/setup.bash"

# Launch Gazebo with robot
echo "Launching Gazebo with bipedal robot..."
ros2 launch bipedal_robot_description spawn_robot.launch.py
