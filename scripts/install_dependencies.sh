# scripts/install_dependencies.sh
#!/bin/bash

echo "Installing dependencies for RL Bipedal Walking..."

# Update system
sudo apt update

# Install ROS2 Humble dependencies
sudo apt install -y \
    ros-humble-gazebo-ros \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-rviz2 \
    ros-humble-xacro \
    ros-humble-tf-transformations \
    python3-pip \
    python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

echo "Dependencies installed successfully!"

# scripts/setup_workspace.sh
#!/bin/bash

echo "Setting up ROS2 workspace..."

# Source ROS2
source /opt/ros/humble/setup.bash

# Build ROS2 workspace
cd ros2_ws
colcon build --symlink-install

# Source workspace
source install/setup.bash

echo "Workspace setup complete!"

# scripts/launch_gazebo.sh
#!/bin/bash

echo "Launching Gazebo simulation..."

# Source ROS2 and workspace
source /opt/ros/humble/setup.bash
cd ros2_ws
source install/setup.bash

# Launch Gazebo with robot
ros2 launch bipedal_robot_description gazebo_world.launch.py

# scripts/launch_rviz.sh  
#!/bin/bash

echo "Launching RViz visualization..."

# Source ROS2 and workspace
source /opt/ros/humble/setup.bash
cd ros2_ws
source install/setup.bash

# Launch RViz
ros2 launch bipedal_robot_description rviz_display.launch.py

# scripts/start_training.sh
#!/bin/bash

echo "Starting RL training..."

# Activate virtual environment
source venv/bin/activate

# Source ROS2 and workspace  
source /opt/ros/humble/setup.bash
cd ros2_ws
source install/setup.bash
cd ..

# Start training
python src/rl_bipedal_walking/training/train_walker.py \
    --episodes 1000 \
    --lr 3e-4 \
    --save-dir training_results \
    --save-interval 50 \
    --plot-interval 25

# scripts/evaluate_model.sh
#!/bin/bash

echo "Evaluating trained model..."

if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_model>"
    echo "Example: $0 training_results_20231201_120000/models/best_model.pth"
    exit 1
fi

MODEL_PATH=$1

# Activate virtual environment
source venv/bin/activate

# Source ROS2 and workspace
source /opt/ros/humble/setup.bash
cd ros2_ws  
source install/setup.bash
cd ..

# Evaluate model
python src/rl_bipedal_walking/evaluation/evaluate_model.py \
    --model $MODEL_PATH \
    --episodes 10 \
    --save-plots evaluation_results

# Makefile
.PHONY: install setup train evaluate clean

# Install all dependencies
install:
	@echo "Installing dependencies..."
	@chmod +x scripts/install_dependencies.sh
	@./scripts/install_dependencies.sh

# Setup ROS2 workspace
setup:
	@echo "Setting up workspace..."
	@chmod +x scripts/setup_workspace.sh
	@./scripts/setup_workspace.sh

# Build ROS2 packages
build:
	@echo "Building ROS2 packages..."
	@cd ros2_ws && colcon build --symlink-install

# Start Gazebo simulation
gazebo:
	@echo "Starting Gazebo..."
	@chmod +x scripts/launch_gazebo.sh
	@./scripts/launch_gazebo.sh

# Start RViz
rviz:
	@echo "Starting RViz..."
	@chmod +x scripts/launch_rviz.sh
	@./scripts/launch_rviz.sh

# Start training
train:
	@echo "Starting training..."
	@chmod +x scripts/start_training.sh
	@./scripts/start_training.sh

# Evaluate model (requires MODEL_PATH variable)
evaluate:
	@echo "Evaluating model..."
	@chmod +x scripts/evaluate_model.sh
	@./scripts/evaluate_model.sh $(MODEL_PATH)

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -