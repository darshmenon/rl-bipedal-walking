#!/usr/bin/env bash
# Train SAC on the MuJoCo bipedal env (no Isaac Gym required).
# Usage:
#   ./scripts/train_sac.sh                      # H1, 1M steps, CPU
#   ./scripts/train_sac.sh --robot xbot         # XBot-L
#   ./scripts/train_sac.sh --device cuda        # GPU
#   ./scripts/train_sac.sh --frame_stack 15     # PPO-equivalent obs history

set -e
cd "$(dirname "$0")/.."

python3 src/rl_bipedal_walking/training/train_sac.py \
  --robot h1 \
  --steps 1000000 \
  --start_steps 10000 \
  --log_dir logs/sac \
  "$@"
