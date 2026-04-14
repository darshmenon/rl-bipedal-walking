#!/bin/bash
# Script to install missing dependencies for Gazebo simulation

echo "Installing ROS2 Gazebo packages..."
sudo apt-get update
sudo apt-get install -y \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-tf-transformations

echo "Dependencies installed successfully!"
echo "Now you can launch Gazebo with: ./launch_gazebo.sh"
