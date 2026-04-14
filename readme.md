# Humanoid Robot Reinforcement Learning

A full-stack humanoid robot RL framework — from fast simulation training in Isaac Gym to MuJoCo sim-to-sim validation and ROS 2 deployment. Covers locomotion policy learning, zero-shot sim-to-real transfer, and hardware-ready deployment pipelines.

---

## Overview

This project implements reinforcement learning for humanoid robot locomotion with a focus on real-world transfer. The pipeline covers:

- **Training** — Fast policy learning in NVIDIA Isaac Gym (GPU-accelerated, parallel envs)
- **Sim-to-Sim Validation** — Transfer trained policies to MuJoCo for physics verification before hardware
- **Sim-to-Real Transfer** — Domain randomization, actuator modeling, and system identification
- **ROS 2 Deployment** — Integration with Gazebo Sim and real robot control stacks

---

## Project Structure

```
rl-bipedal-walking/
├── humanoid/                         # Isaac Gym RL training package
│   ├── envs/
│   │   ├── base/                     # Base legged robot environment
│   │   └── custom/
│   │       ├── humanoid_env.py       # Humanoid-specific env (rewards, obs, sim2real)
│   │       └── humanoid_config.py    # Training & environment hyperparameters
│   ├── scripts/
│   │   ├── train.py                  # Launch Isaac Gym training
│   │   ├── play.py                   # Visualize trained policy
│   │   └── sim2sim.py                # Transfer policy Isaac → MuJoCo
│   ├── algo/                         # PPO implementation
│   └── utils/                        # Logging, terrain, helpers
├── resources/
│   └── robots/
│       └── XBot/                     # Humanoid robot URDF + meshes
├── ros2_ws/                          # ROS 2 workspace
│   └── src/
│       └── bipedal_robot_description/
│           ├── urdf/                 # Robot description
│           ├── launch/               # Gazebo Sim launch files
│           └── config/               # RViz configs
├── src/                              # Custom Gym environments
│   └── rl_bipedal_walking/
│       ├── environments/             # Gymnasium-compatible env wrappers
│       ├── agents/                   # PPO / SAC agents
│       └── training/                 # Training scripts
├── scripts/                          # Shell helper scripts
├── logs/                             # Training run logs
├── setup.py                          # Package install
└── requirements.txt
```

---

## Quick Start

### 1. Prerequisites

- Ubuntu 22.04 / 24.04
- Python 3.8+
- NVIDIA GPU with CUDA 11.x+ (for Isaac Gym)
- [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym)
- ROS 2 Jazzy (for deployment)

### 2. Installation

```bash
git clone https://github.com/darshmenon/rl-bipedal-walking
cd rl-bipedal-walking

# Install Isaac Gym (download preview4 from NVIDIA, then):
pip install -e isaacgym/python

# Install this package
pip install -e .
pip install -r requirements.txt
```

### 3. Train a Locomotion Policy

```bash
python humanoid/scripts/train.py --task humanoid_ppo
```

Training runs in Isaac Gym with 4096 parallel environments. Policy checkpoints are saved to `logs/`.

### 4. Visualize the Policy

```bash
python humanoid/scripts/play.py --task humanoid_ppo --load_run <run_name>
```

### 5. Sim-to-Sim: Transfer to MuJoCo

```bash
python humanoid/scripts/sim2sim.py --load_model logs/<run_name>/model_<iter>.pt
```

Validates the trained policy in MuJoCo before deploying to hardware.

### 6. ROS 2 + Gazebo Sim

```bash
# Build ROS 2 workspace
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash

# Launch Gazebo Sim with humanoid robot
ros2 launch bipedal_robot_description gazebo.launch.py
```

---

## Training Pipeline

### Isaac Gym (Primary Training)

- **4096 parallel environments** for fast wall-clock convergence
- **PPO** with clipped surrogate objective
- **Reward shaping** for locomotion:
  - Forward velocity tracking
  - Base stability (roll/pitch penalty)
  - Foot clearance and contact timing
  - Torque and energy minimization
  - Alive bonus

### Sim-to-Real Transfer Techniques

| Technique | Description |
|---|---|
| Domain Randomization | Randomizes mass, friction, motor gains, external pushes |
| Actuator Modeling | PD controller with delay and torque limits matching hardware |
| Observation Noise | Gaussian noise added to simulated sensor readings |
| System Identification | Physics parameters tuned against real hardware measurements |

### MuJoCo Sim-to-Sim

The `sim2sim.py` script exports the trained Isaac Gym policy and runs it in MuJoCo to:
1. Verify behavior under different physics assumptions
2. Catch policy brittleness before real deployment
3. Profile controller frequency and latency

---

## RL Configuration

Key parameters in `humanoid/envs/custom/humanoid_config.py`:

```python
# Training
num_envs = 4096
max_iterations = 3000
num_steps_per_env = 24

# PPO
learning_rate = 1e-4
num_mini_batches = 4
num_learning_epochs = 5
clip_param = 0.2
gamma = 0.99
lam = 0.95
entropy_coef = 0.01

# Domain Randomization
randomize_friction = True        # [0.1, 3.0]
randomize_base_mass = True       # [-1.0, 3.0] kg
push_robots = True               # random pushes during training
push_interval_s = 15
```

---

## Algorithms

### PPO (Proximal Policy Optimization)
Default algorithm for fast, stable locomotion training. Actor-critic with shared MLP backbone.

- **Policy network**: 3-layer MLP, 512 hidden units, ELU activation
- **Value network**: Separate head on shared trunk
- **Observation**: 47D — joint positions/velocities, base orientation, base angular velocity, commands, previous actions
- **Action**: 12D continuous joint position targets (PD-controlled)

### SAC (Soft Actor-Critic)
Available in `src/rl_bipedal_walking/agents/` for off-policy training with Gymnasium environments. Better sample efficiency for fine-grained manipulation tasks.

---

## Robot Model

Default robot: **XBot-L** (1.65m humanoid) — 12 DOF leg joints, 7 DOF arms.

Additional models from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) can be dropped into `resources/robots/` with a matching config.

---

## ROS 2 Integration

The `ros2_ws` provides:
- **URDF/SDF** humanoid robot description
- **Gazebo Sim (gz)** launch files with ROS-GZ bridge
- **Joint state publisher** and **robot_state_publisher**
- **RViz** visualization config
- Scaffolding for policy inference node (bridge from trained model → joint torque commands)

```bash
# ROS 2 topics
/joint_states            # sensor_msgs/JointState
/odom                    # nav_msgs/Odometry
/cmd_vel                 # geometry_msgs/Twist
```

---

## Results

![Demo](images/demo.gif)

| Metric | Value |
|---|---|
| Training environment | Isaac Gym, 4096 envs |
| Wall-clock convergence | ~2 hours (RTX 3090) |
| MuJoCo sim-to-sim transfer | Zero-shot |
| Target forward velocity | 0.5 m/s |
| Walking stability | Roll/pitch < ±0.3 rad |

---

## Requirements

```
isaacgym>=preview4
torch>=1.13
mujoco==2.3.6
mujoco-python-viewer
gymnasium
stable-baselines3
numpy==1.23.5
wandb
tensorboard
opencv-python
matplotlib
tqdm
```

---

## References

- [Humanoid-Gym: Zero-Shot Sim2Real Transfer](https://arxiv.org/abs/2404.05695) — RobotEra / Tsinghua
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — ETH Zurich RSL
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — DeepMind
- [Isaac Gym](https://developer.nvidia.com/isaac-gym) — NVIDIA

---

## License

MIT License
