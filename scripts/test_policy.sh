#!/bin/bash
# Script to test trained RL policy

# Activate Python virtual environment
source /home/darsh/rl-bipedal-walking/venv/bin/activate

# Add project to Python path
export PYTHONPATH="/home/darsh/rl-bipedal-walking/src:$PYTHONPATH"

# Source ROS2
source /opt/ros/jazzy/setup.bash
source /home/darsh/rl-bipedal-walking/ros2_ws/install/setup.bash

# Find the latest training results directory
LATEST_RESULTS=$(ls -td /home/darsh/rl-bipedal-walking/training_results_* 2>/dev/null | head -1)

if [ -z "$LATEST_RESULTS" ]; then
    echo "No training results found!"
    exit 1
fi

MODEL_PATH="$LATEST_RESULTS/models/best_model.pth"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found at $MODEL_PATH"
    exit 1
fi

echo "Testing policy from: $MODEL_PATH"
echo "Running 10 evaluation episodes..."

cd /home/darsh/rl-bipedal-walking
python src/rl_bipedal_walking/evaluation/evaluate_model.py \
  --model "$MODEL_PATH" \
  --episodes 10 \
  --save-plots "${LATEST_RESULTS}/evaluation_results"
