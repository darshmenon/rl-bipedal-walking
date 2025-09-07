# 🤖 RL Bipedal Walking Robot

A complete reinforcement learning (RL) framework for training a bipedal robot to walk using **ROS 2**, **Gazebo**, and **PyTorch (PPO)**.

This project combines **robot simulation** with **deep reinforcement learning**, enabling training, evaluation, and visualization of walking behaviors for a custom bipedal robot.

---

## 📂 Project Structure

```
rl-bipedal-walking/
├── ros2_ws/                          # ROS2 workspace
│   └── src/
│       └── bipedal_robot_description/
│           ├── launch/               # Launch files
│           ├── urdf/                 # Robot description
│           ├── worlds/               # Gazebo worlds
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
├── Makefile
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd rl-bipedal-walking

# Install dependencies
make install

# Setup ROS2 workspace
make setup
```

### 2️⃣ Build ROS2 Packages

```bash
make build
```

### 3️⃣ Launch Simulation

```bash
# Terminal 1 - Gazebo
make gazebo

# Terminal 2 - RViz (optional)
make rviz
```

### 4️⃣ Start Training

```bash
# Terminal 3 - RL Training
make train
```

### 5️⃣ Evaluate Trained Model

```bash
make evaluate MODEL_PATH=training_results_<timestamp>/models/best_model.pth
```

---

## 🎯 Key Features

### 🦾 Robot Simulation

* 6-DOF bipedal robot (hip, knee, ankle per leg)
* Gazebo physics with collision dynamics
* ROS2 integration for messaging and control
* RViz for visualization

### 🧠 Reinforcement Learning

* PPO (Proximal Policy Optimization)
* Custom Gym-style environment for locomotion
* Reward engineering for:

  * Forward velocity
  * Stability (roll/pitch)
  * Height maintenance
  * Energy efficiency
* Real-time training plots and logging

### ⚡ Advanced Features

* Generalized Advantage Estimation (GAE)
* Gradient clipping for stability
* Automatic model checkpointing
* Multi-episode evaluation and comparison
* Visualization tools (training curves, trajectories, metrics)

---

## 🧠 RL Algorithm (PPO)

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

## 📊 Training Configuration

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

## 📈 Monitoring Training

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
  ✅ New best model saved!
```

---

## 🧪 Evaluation

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

## 🛠️ Troubleshooting

* **Gazebo won’t start** → check `gazebo --version`, source ROS2
* **Robot falls immediately** → validate URDF, physics params
* **Training stalls** → lower LR, extend episodes, tune rewards
* **ROS2 issues** → `ros2 topic list`, `ros2 param get ...`
* **CUDA issues** → check `torch.cuda.is_available()`, fallback to CPU

---

## ⚡ Optimization Tips

* Enable GPU training (PyTorch CUDA)
* Run headless (`--no-render`)
* Use vectorized environments
* Curriculum learning & domain randomization
* Gradient clipping + LR scheduling

---

## 🚀 Advanced Usage

* Custom robot models: add new URDF & update environment
* Modify rewards: edit `bipedal_env.py`
* Hyperparameter tuning: adjust in `train_walker.py`

---

## 📜 License

MIT License – free to use, modify, and distribute.

