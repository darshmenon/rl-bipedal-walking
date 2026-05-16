# Humanoid RL — Bipedal Locomotion with Sim-to-Real Transfer

A reinforcement learning framework for training bipedal humanoid robots to walk, combining **MuJoCo** simulation with **ROS 2 Humble** deployment pipelines. Trained policies transfer from simulation to real hardware via domain randomization and system identification.

---

## Overview

This project implements end-to-end reinforcement learning for humanoid locomotion:

- **PPO training** — Parallel Isaac Gym environments with reward shaping for stable bipedal walking (XBot-L, Unitree H1 leg-only, H1 whole-body)
- **SAC training** — Off-policy training on a lightweight MuJoCo gym env — no Isaac Gym required
- **Terrain curriculum** — Progressive difficulty from flat plane to rough trimesh via `--terrain` flag
- **Sim-to-Sim Validation** — Transfer trained policies to MuJoCo for physics cross-validation before hardware deployment (XBot and H1)
- **Sim-to-Real Transfer** — Domain randomization, actuator modeling, and system identification for zero-shot transfer
- **ROS 2 Deployment** — Policy inference node publishes joint commands via `ros2_control` ForwardCommandController at 100 Hz; hardware safety guard mode included

---

## Project Structure

```
rl-bipedal-walking/
├── humanoid_descriptions/           # Vendored humanoid robot sources
│   ├── ros/                         # ROS/Gazebo-oriented packages
│   ├── rl/                          # RL training stacks from official sources
│   └── urdf_only/                   # Description-focused robot packages
├── humanoid/                         # RL training package (Isaac Gym / PPO)
│   ├── envs/
│   │   ├── base/                     # Base legged robot environment
│   │   └── custom/
│   │       ├── humanoid_env.py       # XBot env (rewards, obs, domain rand)
│   │       ├── humanoid_config.py    # XBot training hyperparameters
│   │       ├── h1_env.py             # Unitree H1 leg-only env (10 DOF)
│   │       ├── h1_config.py          # H1 leg-only config
│   │       ├── h1_wholebody_env.py   # H1 whole-body env (18 DOF, arms + legs)
│   │       └── h1_wholebody_config.py# H1 whole-body config
│   ├── scripts/
│   │   ├── train.py                  # PPO training (--terrain, --use_wandb flags)
│   │   ├── play.py                   # Visualize & export trained policy
│   │   └── sim2sim.py                # MuJoCo sim-to-sim (--robot xbot|h1)
│   ├── algo/                         # PPO + accuracy metrics
│   └── utils/                        # Logging, terrain, task registry
├── resources/
│   └── robots/
│       └── XBot/                     # Humanoid URDF, MJCF, meshes
│           ├── urdf/                 # URDF robot description
│           └── mjcf/                 # MuJoCo XML models
├── ros2_ws/                          # ROS 2 Humble workspace
│   └── src/
│       ├── bipedal_robot_description/
│       │   ├── urdf/                 # Robot description (ros2_control tags added)
│       │   ├── launch/               # Gazebo Sim + controller manager launch
│       │   └── config/               # RViz config + controllers.yaml
│       └── policy_runner/            # Policy inference node (100 Hz)
│           ├── policy_runner/
│           │   └── policy_inference_node.py
│           ├── config/xbot_joints.yaml
│           └── launch/policy_runner.launch.py
├── src/                              # Standalone (no Isaac Gym) stack
│   └── rl_bipedal_walking/
│       ├── environments/
│       │   ├── bipedal_env.py        # Gazebo-backed gym env (ROS2)
│       │   └── mujoco_env.py         # Standalone MuJoCo gym env (XBot / H1)
│       ├── agents/
│       │   ├── ppo_agent.py          # PPO agent
│       │   └── sac_agent.py          # SAC with twin-Q + auto-entropy
│       └── training/
│           ├── train_walker.py       # PPO training script
│           └── train_sac.py          # SAC training script
├── scripts/                          # Shell helper scripts
├── logs/                             # Training run logs & exported policies
├── setup.py                          # Package install
└── requirements.txt
```

---

## Current Status

| Area | Status |
|---|---|
| XBot-L PPO training | Working |
| Unitree H1 leg-only PPO (`h1_ppo`) | Config + env ready — requires Isaac Gym |
| Unitree H1 whole-body PPO (`h1_wholebody_ppo`) | Config + env ready — requires Isaac Gym |
| SAC training on MuJoCo gym env | Working — no Isaac Gym needed |
| Terrain curriculum (`--terrain`) | Working — enabled via CLI flag |
| TensorBoard + W&B logging | TensorBoard always on; W&B opt-in via `--use_wandb` |
| Accuracy metrics in TensorBoard | Velocity tracking + gait contact accuracy logged |
| Sim-to-sim MuJoCo (XBot) | Working |
| Sim-to-sim MuJoCo (H1) | Working via `--robot h1` |
| ROS 2 Gazebo Sim spawn | Working |
| `ros2_control` ForwardCommandController | Wired — `controllers.yaml` + updated URDF |
| Policy inference node (sim + hardware) | Working — `policy_runner` package |

---

## Quick Start

### 1. Prerequisites

- Ubuntu 22.04
- Python 3.8+
- NVIDIA GPU with CUDA 11.x+
- ROS 2 Humble
- MuJoCo 2.3.6+

### 2. Installation

```bash
git clone https://github.com/darshmenon/rl-bipedal-walking
cd rl-bipedal-walking

# Create virtual environment
conda create -n humanoid-rl python=3.8
conda activate humanoid-rl

# Install PyTorch with CUDA
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install the package
pip install -e .
pip install -r requirements.txt
```

### 3. Train a Locomotion Policy

**PPO — Isaac Gym parallel environments (requires GPU + Isaac Gym)**

```bash
# XBot-L — flat plane
python humanoid/scripts/train.py --task humanoid_ppo --run_name v1 --headless --num_envs 4096

# XBot-L — terrain curriculum + W&B logging
python humanoid/scripts/train.py --task humanoid_ppo --run_name v1_terrain --headless --terrain --use_wandb

# Unitree H1 leg-only
python humanoid/scripts/train.py --task h1_ppo --run_name h1_v1 --headless

# Unitree H1 whole-body (legs + arms)
python humanoid/scripts/train.py --task h1_wholebody_ppo --run_name h1_wb_v1 --headless
```

**SAC — MuJoCo gym environment (no Isaac Gym needed)**

```bash
# Train SAC on H1 (CPU or GPU)
python src/rl_bipedal_walking/training/train_sac.py --robot h1 --steps 1000000

# Train SAC on XBot
python src/rl_bipedal_walking/training/train_sac.py --robot xbot --device cuda

# With W&B
python src/rl_bipedal_walking/training/train_sac.py --robot h1 --use_wandb
```

Policy checkpoints are saved to `logs/`.

### 4. Visualize & Export the Policy

```bash
# Load trained policy and export JIT model
python humanoid/scripts/play.py --task humanoid_ppo --run_name v1
```

This exports a JIT-compiled policy to `logs/<experiment>/exported/policies/` for deployment.

### 5. Sim-to-Sim: Transfer to MuJoCo

```bash
# XBot-L validation (flat)
python humanoid/scripts/sim2sim.py --load_model logs/XBot_ppo/exported/policies/policy_1.pt

# XBot-L with terrain
python humanoid/scripts/sim2sim.py --load_model logs/XBot_ppo/exported/policies/policy_1.pt --terrain

# Unitree H1 validation
python humanoid/scripts/sim2sim.py --load_model logs/H1_ppo/exported/policies/policy_1.pt --robot h1
```

Validates the trained policy under MuJoCo physics before deploying to hardware.

### 6. ROS 2 + Gazebo Sim + ros2_control

```bash
# Build ROS 2 workspace (includes policy_runner package)
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Launch Gazebo Sim with humanoid robot + ros2_control controller manager
ros2 launch bipedal_robot_description spawn_robot.launch.py
```

### 7. Deploy Policy (Simulation or Hardware)

```bash
# Run trained policy in Gazebo Sim at 100 Hz
ros2 launch policy_runner policy_runner.launch.py \
  policy_path:=/abs/path/to/policy_1.pt

# Run on real hardware (enables per-step safety guard)
ros2 launch policy_runner policy_runner.launch.py \
  policy_path:=/abs/path/to/policy_1.pt hardware:=true
```

The `policy_runner` node subscribes to `/joint_states`, `/odom`, and `/cmd_vel`, then publishes joint position targets to `/position_controller/commands` at 100 Hz.

---

## RL Algorithm

### PPO (Proximal Policy Optimization)

The primary training algorithm uses an actor-critic architecture with the following specs:

| Parameter | Value |
|---|---|
| Policy Network | 3-layer MLP [512, 256, 128], ELU |
| Observation Space | 47D × 15 frames = 705D |
| Action Space | 12D continuous joint position targets |
| Learning Rate | 1e-5 |
| Discount (γ) | 0.994 |
| GAE (λ) | 0.9 |
| Clip Ratio | 0.2 |
| Entropy Coef | 0.001 |
| Parallel Envs | 4096 |
| Max Iterations | 3000 |

### SAC (Soft Actor-Critic)

Off-policy alternative that trains on the standalone MuJoCo gym env — no Isaac Gym or large parallel simulation required.

| Parameter | Value |
|---|---|
| Actor / Critic | 2-layer MLP [256, 256] |
| Twin-Q critics | Yes (reduces overestimation) |
| Entropy tuning | Automatic (target = −action_dim) |
| Learning Rate | 3e-4 |
| Discount (γ) | 0.99 |
| Soft update (τ) | 0.005 |
| Replay Buffer | 1M transitions |
| Batch Size | 256 |
| Warm-up steps | 10k random actions |

---

### Reward Function

The reward is a weighted sum of locomotion objectives:

- **Velocity tracking** — Forward/lateral/angular velocity command following
- **Gait phase** — Foot contact timing aligned to reference sinusoidal gait
- **Joint position tracking** — Penalize deviation from reference motion
- **Stability** — Base orientation, height, and acceleration penalties
- **Feet clearance** — Swing leg lift during gait cycle
- **Energy efficiency** — Torque, joint velocity, and action smoothness penalties
- **Collision** — Penalize undesired body contacts

### Domain Randomization

Applied during training for robust sim-to-real transfer:

- Friction: [0.1, 2.0]
- Base mass: [-5.0, +5.0] kg
- Random pushes every 4s (linear + angular)
- Action delay and noise injection

---

## Robot Model

**Default:** XBot-L — a 1.65m humanoid with 12 DOF legs.

| Joint Group | DOF | PD Gains (Kp/Kd) |
|---|---|---|
| Hip Roll | 2 | 200 / 10 |
| Hip Yaw | 2 | 200 / 10 |
| Hip Pitch | 2 | 350 / 10 |
| Knee | 2 | 350 / 10 |
| Ankle Pitch | 2 | 15 / 10 |
| Ankle Roll | 2 | 15 / 10 |

Robot assets live in `resources/robots/XBot/` with both URDF and MJCF descriptions.

---

## Sim-to-Sim Pipeline

The `sim2sim.py` script enables zero-shot policy transfer between simulators:

1. Train in parallel simulation → export JIT policy
2. Load policy in MuJoCo with matched robot model
3. Run PD control loop at 1000 Hz (policy at 100 Hz via decimation)
4. Verify walking behavior, contact forces, and stability

This catches policy brittleness and physics mismatches before real hardware deployment.

---

## ROS 2 Integration

The `ros2_ws` provides deployment infrastructure on ROS 2 Humble:

- **URDF/SDF** humanoid robot description
- **Gazebo Sim** launch files with ROS-GZ bridge
- **Joint state publisher** and `robot_state_publisher`
- **RViz** visualization config
- Scaffolding for **policy inference node** (trained model → joint torque commands)
- A lightweight spawn path for testing descriptions before integrating controllers

Current limitation:

`ros2_control` is not fully wired into the local `bipedal_robot_description` package yet, so the ROS workspace is currently better suited for description validation and spawn testing than for full controller bring-up.

```bash
# Key ROS 2 topics
/joint_states          # sensor_msgs/JointState
/odom                  # nav_msgs/Odometry
/cmd_vel               # geometry_msgs/Twist
```

---

## Testing Humanoids In Gazebo

There are two different simulation paths in this repo:

- `ros2_ws/` uses **ROS 2 Humble + Gazebo Sim**
- `humanoid_descriptions/ros/unitree_ros` and `humanoid_descriptions/urdf_only/berkeley_humanoid_description` are imported from **ROS 1 + classic Gazebo** ecosystems

### 1. Test the local ROS 2 humanoid

Use this when you want to validate the repo's current ROS 2 spawn flow:

```bash
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

ros2 launch bipedal_robot_description spawn_robot.launch.py
```

Alternate Gazebo entrypoint:

```bash
ros2 launch bipedal_robot_description gazebo_launch.py
```

### 2. Test imported Unitree humanoids

The imported Unitree packages live in:

- `humanoid_descriptions/ros/unitree_ros/robots/g1_description`
- `humanoid_descriptions/ros/unitree_ros/robots/h1_description`
- `humanoid_descriptions/ros/unitree_ros/robots/h1_2_description`
- `humanoid_descriptions/ros/unitree_ros/robots/h2_description`
- `humanoid_descriptions/ros/unitree_ros/robots/r1_description`
- `humanoid_descriptions/ros/unitree_ros/robots/r1_air_description`

These are ROS 1 packages. Start with the one that already includes a direct Gazebo launch:

```bash
cd humanoid_descriptions/ros/unitree_ros
# inside a ROS 1 catkin workspace
roslaunch h1_description gazebo.launch
```

For quick visual checks without Gazebo:

```bash
roslaunch h1_description display.launch
```

The imported `unitree_gazebo` and `unitree_controller` packages are also included for classic Gazebo and low-level controller experiments.

Important note:

The upstream Unitree stack is aimed at ROS 1 and classic Gazebo, not the local ROS 2 Gazebo Sim workspace in this repo.

### 3. Test Berkeley Humanoid

The Berkeley package is imported under:

- `humanoid_descriptions/urdf_only/berkeley_humanoid_description`

Classic Gazebo test:

```bash
# inside a ROS 1 catkin workspace
roslaunch berkeley_humanoid_description empty_world.launch
```

Standalone URDF/RViz test:

```bash
roslaunch berkeley_humanoid_description standalone.launch
```

### 4. Test Booster and RobotEra models

These imports are best treated as description sources first:

- `humanoid_descriptions/urdf_only/booster_assets`
- `humanoid_descriptions/urdf_only/robotera_models`

Useful files include:

- `humanoid_descriptions/urdf_only/booster_assets/robots/T1/T1_23dof.urdf`
- `humanoid_descriptions/urdf_only/booster_assets/robots/K1/K1_22dof.urdf`
- `humanoid_descriptions/urdf_only/robotera_models/star1`

They do not yet come with a ready-to-run local ROS 2 spawn wrapper in this repo, so the usual next step is:

1. Copy the chosen URDF and meshes into a ROS package
2. Point `robot_state_publisher` at that URDF
3. Spawn it with either `ros_gz_sim create` in ROS 2 or `gazebo_ros spawn_model` in ROS 1

If you want, the next integration step is to replace `ros2_ws/src/bipedal_robot_description/urdf/bipedal.urdf` with one of these imported robots and add a dedicated launch package for it.

---

## Adding a New Robot

1. Add URDF and MJCF assets to `resources/robots/<your_robot>/`
2. Create a config in `humanoid/envs/custom/` inheriting from `LeggedRobotCfg`
3. Set asset path, body names, default joint angles, and PD gains
4. Register the task in `humanoid/envs/__init__.py`
5. Update `sim2sim.py` joint mapping if needed

If you want to start from an existing robot instead of creating one from scratch, check `humanoid_descriptions/` first. The vendored sources include official Unitree stacks plus Berkeley, Booster, and RobotEra description packages.

---

## Training Tips

- **Headless mode**: Use `--headless` for faster training without rendering
- **GPU selection**: `--sim_device=cuda:0 --rl_device=0`
- **Resume training**: Set `resume=True` and `load_run` in config
- **Terrain curriculum**: Enable `mesh_type='trimesh'` in config for rough terrain training
- **Monitoring**: Training logs are compatible with TensorBoard and Weights & Biases

---

## Key Hyperparameters

Located in `humanoid/envs/custom/humanoid_config.py`:

```python
# Training
num_envs = 4096
max_iterations = 3000
episode_length_s = 24

# Domain Randomization
randomize_friction = True       # [0.1, 2.0]
randomize_base_mass = True      # [-5, +5] kg
push_robots = True              # random impulses
action_delay = 0.5              # simulate actuator lag
action_noise = 0.02             # observation noise

# Control
action_scale = 0.25
decimation = 10                 # 1000Hz sim → 100Hz policy
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `libpython3.8.so` not found | `export LD_LIBRARY_PATH="~/conda/envs/humanoid-rl/lib:$LD_LIBRARY_PATH"` |
| `AttributeError: module 'distutils'` | Install PyTorch 1.12+ with matching CUDA |
| `libstdc++` version mismatch | Move conda's libstdc++ to `lib/_unused/` |
| Robot falls immediately in MuJoCo | Normal — needs trained policy loaded |
| ROS 2 topic issues | Check `ros2 topic list`, verify GZ bridge is running |

---

## References

- [Humanoid-Gym: Zero-Shot Sim2Real Transfer](https://arxiv.org/abs/2404.05695) — RobotEra / Tsinghua
- [Advancing Humanoid Locomotion with Denoising World Model Learning](https://enriquecoronadozu.github.io/rssproceedings2024/rss20/p058.pdf) — RSS 2024
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — ETH Zurich RSL
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) — DeepMind

---

## License

BSD-3-Clause License
