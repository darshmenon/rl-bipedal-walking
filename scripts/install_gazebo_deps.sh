#!/bin/bash
# Script to install missing dependencies for Gazebo simulation

echo "Installing ROS2 Gazebo packages..."
sudo apt-get update
sudo apt-get install -y \
    ros-jazzy-gazebo-ros-pkgs \
    ros-jazzy-gazebo-ros \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-joint-state-publisher \
    ros-jazzy-joint-state-publisher-gui \
    ros-jazzy-tf-transformations

echo "Dependencies installed successfully!"
echo "Now you can launch Gazebo with: ./launch_gazebo.sh"
