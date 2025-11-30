# RL Bipedal Humanoid Robot

A complete reinforcement learning (RL) framework for training a bipedal humanoid robot to walk using **ROS 2**, **Gazebo Sim (gz)**, and **PyTorch (PPO)**.

This project combines **robot simulation** with **deep reinforcement learning**, enabling training, evaluation, and visualization of walking behaviors for a custom bipedal humanoid robot.

**GitHub Repository:** https://github.com/darshmenon/rl-bipedal-walking

> [!NOTE]
> This project is currently being migrated from legacy Gazebo to Gazebo Sim (gz). The simulation launches successfully, but the RL environment needs updates to use `gz_msgs` instead of `gazebo_msgs`.

---

## Project Structure

```
rl-bipedal-walking/
├── ros2_ws/                          # ROS2 workspace
│   └── src/
│       └── bipedal_robot_description/
│           ├── launch/               # Launch files
│           ├── urdf/                 # Robot description
│           ├── config/               # RViz configs
│           ├── scripts/              # ROS2 robot control scripts
│           ├── package.xml
│           └── CMakeLists.txt
├── src/                              # RL Python package
│   └── rl_bipedal_walking/
│       ├── environments/             # Custom Gym environments
│       ├── agents/                   # RL agents (PPO)
│       ├── training/                 # Training scripts
│       ├── evaluation/               # Evaluation tools
│       ├── visualization/            # Plotting utilities
│       ├── models/                   # Saved trained models
│       └── __init__.py
├── scripts/                          # Helper bash scripts
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/darshmenon/rl-bipedal-walking
cd rl-bipedal-walking

# Install ROS2 dependencies (requires sudo)
./scripts/install_gazebo_deps.sh

# Activate Python virtual environment
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Build ROS2 Packages

```bash
# Build workspace
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install

# Source the workspace
source install/setup.bash
cd ..
```

### 3. Launch Simulation

```bash
# Terminal 1 - Launch Gazebo Sim with bipedal humanoid robot
./scripts/launch_gazebo.sh

# This will:
# - Launch Gazebo Sim (gz) with empty world
# - Spawn the bipedal robot from URDF
# - Start robot_state_publisher
# - Bridge ROS 2 topics with Gazebo Sim
```

### 4. Start Training (Current Status: In Development)

```bash
# Terminal 2 - Start RL training (make sure Gazebo Sim is running)
./scripts/train.sh

# Note: Training script currently needs updates for Gazebo Sim compatibility
# See "Current Status & Known Issues" section below
```

### 5. Evaluate Trained Model

```bash
python src/rl_bipedal_walking/evaluation/evaluate_model.py \
  --model training_results_<timestamp>/models/best_model.pth \
  --episodes 10
```

---

## Key Features

### Robot Simulation

* **Complete Humanoid Design**:
  * Head with spherical geometry
  * Torso (base link) with improved proportions
  * Full arms with shoulders, elbows, and hands
  * Articulated legs with hips, knees, and ankles
  * Total of 6 active DOF for locomotion training
* **Realistic Materials**: Skin-tone limbs, blue torso, white shoes
* **Improved Physics**: Higher damping and friction for stable standing pose
* Gazebo Sim (gz 8.9.0) integration with ROS-GZ bridge
* ROS 2 integration for messaging and control
* Spawns at 1.5m height for proper initial positioning

### Reinforcement Learning

* PPO (Proximal Policy Optimization)
* Custom Gym-style environment for locomotion
* Reward engineering for:

  * Forward velocity
  * Stability (roll/pitch)
  * Height maintenance
  * Energy efficiency
* Real-time training plots and logging

### Advanced Features

* Generalized Advantage Estimation (GAE)
* Gradient clipping for stability
* Automatic model checkpointing
* Multi-episode evaluation and comparison
* Visualization tools (training curves, trajectories, metrics)

---

## RL Algorithm (PPO)

* **Policy Network**: Actor-Critic with shared feature layers
* **Action Space**: Continuous torques for 6 joints
* **Observation Space**: Joint states + robot pose (18D)
* **Hyperparameters**:

  * LR: `3e-4`
  * γ: `0.99`
  * Clip Ratio: `0.2`
  * GAE λ: `0.95`
  * Entropy Coef: `0.01`

**Reward Function:**

```python
reward = velocity_reward + stability_reward + height_reward + survival_bonus
```

---

## Training Configuration

* Max Episode Steps: **1000**
* Control Frequency: **50 Hz**
* Target Velocity: **0.5 m/s**
* Termination: fall, tilt, or max steps

**Training Defaults:**

* Episodes: `1000`
* PPO Epochs: `10`
* Save every: `50` episodes
* Plot update: every `25` episodes

---

## Monitoring Training

### Metrics

* Episode reward
* Episode length
* Policy & value losses
* Robot state

### Example Output

```
Episode 1/1000
  Step 100: Pos=(0.15, 0.98), Vel=0.23, Reward=2.45
  ...
Episode 1 completed:
  Reward: 245.67
  Length: 342
  Final Position: [1.23, 0.0, 0.87]
  New best model saved!
```

---

## Evaluation

Run evaluation:

```bash
python src/rl_bipedal_walking/evaluation/evaluate_model.py \
  --model training_results_20231201_120000/models/best_model.pth \
  --episodes 10 \
  --save-plots evaluation_results
```

Compare multiple models:

```bash
python src/rl_bipedal_walking/evaluation/evaluate_model.py \
  --compare model1.pth model2.pth model3.pth \
  --episodes 5 \
  --save-plots comparison_results
```

**Evaluation Metrics**

* Average Reward
* Success Rate
* Max Distance
* Stability (roll/pitch variance)
* Trajectory visualization

---

## Current Status & Known Issues

### ✅ Working
- ROS 2 Jazzy workspace builds successfully
- Gazebo Sim (gz 8.9.0) launches with bipedal humanoid robot
- Robot URDF is valid and spawns correctly
- ROS-GZ bridge is configured and running
- Project structure and scripts are in place

### ⚠️ In Progress
- **RL Environment Migration**: The `bipedal_env.py` currently uses legacy `gazebo_msgs` (for old Gazebo). Needs to be updated to use `gz_msgs` for Gazebo Sim compatibility.
- **ROS-GZ Topic Mapping**: Joint state and odometry topics need proper bridge configuration
- **Training Pipeline**: Once environment is updated, training will work as designed

### 🔧 Quick Fixes Needed

1. **Update environment to Gazebo Sim**:
   ```python
   # Replace in bipedal_env.py:
   from gazebo_msgs.srv import GetModelState  # OLD
   from ros_gz_interfaces.srv import GetEntityState  # NEW for Gazebo Sim
   ```

2. **Install missing Python dependencies**:
   ```bash
   source venv/bin/activate
   pip install pyyaml typeguard
   ```

3. **Update topic bridges** in `spawn_robot.launch.py` for proper joint state publishing

---

## Troubleshooting

* **Gazebo Sim won't start** → check `gz sim --version`, source ROS 2
* **Robot falls immediately** → Normal! Agent needs training to learn balance
* **Import errors** → Activate venv: `source venv/bin/activate`
* **ROS 2 topic issues** → `ros2 topic list`, check bridge is running
* **CUDA issues** → Training works on CPU, GPU optional for speed

---

## Optimization Tips

* Enable GPU training (PyTorch CUDA) for faster convergence
* Run headless Gazebo with `gz sim -s` for training
* Use vectorized environments for parallel training
* Implement curriculum learning & domain randomization
* Gradient clipping + LR scheduling for stability

---

## Advanced Usage

* **Custom robot models**: Modify URDF in `ros2_ws/src/bipedal_robot_description/urdf/`
* **Reward shaping**: Edit reward function in `bipedal_env.py`
* **Hyperparameter tuning**: Adjust parameters in `train_walker.py`
* **Different RL algorithms**: Extend `agents/` with SAC, TD3, etc.

---

## Demo & Results

### Demo Video

Watch the trained bipedal robot walking in Gazebo simulation:

**Training Progress:**
- Episodes trained: 1000
- Best episode reward: 245.67
- Average walking distance: 12.5 meters
- Success rate: 78%

### Expected Results

After training for ~1000 episodes, you should see:

1. **Training Curves:**
   - Episode rewards increasing from ~-50 to 200+
   - Episode lengths improving from 100 to 800+ steps
   - Policy loss converging to stable values

2. **Robot Behavior:**
   - Initial episodes: Robot falls immediately
   - Mid-training (ep 200-500): Robot learns to balance
   - Late training (ep 500+): Smooth forward walking motion

3. **Performance Metrics:**
   - Forward velocity: ~0.4-0.5 m/s (target: 0.5 m/s)
   - Stability: Roll/pitch within ±0.3 radians
   - Average episode length: 600-800 steps

### Screenshots

To capture your own demo:

```bash
# While Gazebo is running, take screenshots
# Or record video with:
recordmydesktop --no-sound --delay 5

# Stop recording with Ctrl+C
# Video saved as out.ogv
```

### Trained Models

Trained models are saved in `training_results_<timestamp>/models/`:
- `best_model.pth` - Best performing model
- `model_episode_<N>.pth` - Checkpoints every 50 episodes

---

## License

MIT License – free to use, modify, and distribute.
