#!/usr/bin/env bash
# Evaluate and optionally export a trained SAC checkpoint.
# Usage:
#   ./scripts/eval_sac.sh logs/sac/h1_sac_*/best_model.pt
#   ./scripts/eval_sac.sh logs/sac/h1_sac_*/best_model.pt --export logs/h1_policy.pt
#   ./scripts/eval_sac.sh logs/sac/h1_sac_*/best_model.pt --render

set -e
MODEL="$1"; shift
cd "$(dirname "$0")/.."

python3 src/rl_bipedal_walking/evaluation/play_sac.py \
  --model "$MODEL" \
  --robot h1 \
  --episodes 10 \
  "$@"
