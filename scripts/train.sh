#!/bin/bash
# Script to start RL training

# Activate Python virtual environment
source /home/darsh/rl-bipedal-walking/venv/bin/activate

# Add project to Python path
export PYTHONPATH="/home/darsh/rl-bipedal-walking/src:$PYTHONPATH"

# Source ROS2
source /opt/ros/jazzy/setup.bash
source /home/darsh/rl-bipedal-walking/ros2_ws/install/setup.bash

# Start training
echo "Starting RL training..."
cd /home/darsh/rl-bipedal-walking
python src/rl_bipedal_walking/training/train_walker.py --episodes 1000
