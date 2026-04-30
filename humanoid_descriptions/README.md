# Humanoid Descriptions

This folder vendors official or widely used humanoid robot sources into the repo as plain files.

There are no nested `.git/` directories here.

## Layout

- `ros/unitree_ros`
  Official Unitree ROS package set with humanoid descriptions such as `g1`, `h1`, `h1_2`, `h2`, `r1`, and `r1_air`.
  Best when you want robot description files plus ROS/Gazebo-oriented packages.

- `rl/unitree_rl_gym`
  Official Unitree RL training stack for `g1`, `h1`, and `h1_2`.
  Best when you want a reference RL pipeline for humanoid locomotion.

- `rl/booster_gym`
  Official Booster Robotics RL framework for humanoid locomotion.

- `urdf_only/booster_assets`
  Booster Robotics description and motion assets.

- `urdf_only/robotera_models`
  RobotEra STAR1 model package.

- `urdf_only/berkeley_humanoid_description`
  Berkeley Humanoid description package focused on locomotion research.

## Suggested Starting Points

- Use `ros/unitree_ros` if you want a better humanoid description than the placeholder ROS model in `ros2_ws/`.
- Use `rl/unitree_rl_gym` if you want to train or adapt an existing humanoid RL stack.
- Use `urdf_only/robotera_models` or `urdf_only/berkeley_humanoid_description` if you mainly want a cleaner URDF package to integrate.
