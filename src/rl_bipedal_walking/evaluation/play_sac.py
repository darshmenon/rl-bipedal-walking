"""
play_sac.py

Load a trained SAC checkpoint, run it in the MuJoCo env, and optionally
export the actor as a TorchScript (.pt) file so the policy_runner ROS2
node can load it with torch.jit.load().

Usage:
  # Visualise in viewer
  python src/rl_bipedal_walking/evaluation/play_sac.py \
      --model logs/sac/h1_sac_May16_*/best_model.pt \
      --robot h1 --render

  # Evaluate 20 episodes (no viewer)
  python src/rl_bipedal_walking/evaluation/play_sac.py \
      --model logs/sac/h1_sac_May16_*/best_model.pt \
      --robot h1 --episodes 20

  # Export as JIT for policy_runner deployment
  python src/rl_bipedal_walking/evaluation/play_sac.py \
      --model logs/sac/h1_sac_May16_*/best_model.pt \
      --robot h1 --export logs/sac/h1_policy.pt
"""

import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.sac_agent import SACAgent, GaussianPolicy
from environments.mujoco_env import MujocoBipedalEnv


# ── JIT wrapper ──────────────────────────────────────────────────────────────

class DeterministicPolicyJIT(torch.nn.Module):
    """Wraps GaussianPolicy for JIT export: always returns the deterministic action."""

    def __init__(self, policy: GaussianPolicy):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        _, _, action = self.policy.sample(obs)
        return action


# ── helpers ──────────────────────────────────────────────────────────────────

def load_agent(model_path: str, state_dim: int, action_dim: int, device: str) -> SACAgent:
    agent = SACAgent(state_dim, action_dim, device=device)
    agent.load(model_path)
    agent.policy.eval()
    return agent


def run_episode(env: MujocoBipedalEnv, agent: SACAgent, render: bool = False) -> dict:
    obs, _ = env.reset()
    total_reward = 0.
    steps = 0
    fwd_vels = []

    while True:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        fwd_vels.append(info.get('forward_vel', 0.))
        if terminated or truncated:
            break

    return {
        'reward': total_reward,
        'length': steps,
        'mean_fwd_vel': float(np.mean(fwd_vels)),
        'max_fwd_vel':  float(np.max(fwd_vels)),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',    required=True, help='Path to SAC .pt checkpoint.')
    p.add_argument('--robot',    default='h1',  choices=['xbot', 'h1'])
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--render',   action='store_true', help='Open MuJoCo viewer.')
    p.add_argument('--export',   default='',
                   help='If set, export deterministic policy as TorchScript to this path.')
    p.add_argument('--device',   default='cpu')
    args = p.parse_args()

    render_mode = 'human' if args.render else None
    env = MujocoBipedalEnv(robot=args.robot, render_mode=render_mode, domain_rand=False)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = load_agent(args.model, state_dim, action_dim, args.device)

    print(f"Loaded SAC policy from {args.model}")
    print(f"Robot: {args.robot}  |  obs={state_dim}  act={action_dim}")

    rewards, lengths, fwd_vels = [], [], []
    for ep in range(1, args.episodes + 1):
        result = run_episode(env, agent, render=args.render)
        rewards.append(result['reward'])
        lengths.append(result['length'])
        fwd_vels.append(result['mean_fwd_vel'])
        print(f"  ep {ep:>3d}  reward={result['reward']:>8.2f}  "
              f"len={result['length']:>5d}  fwd_vel={result['mean_fwd_vel']:.3f} m/s")

    print(f"\n{'─'*55}")
    print(f"  Mean reward : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean length : {np.mean(lengths):.0f}")
    print(f"  Mean fwd vel: {np.mean(fwd_vels):.3f} m/s")

    if args.export:
        wrapper = DeterministicPolicyJIT(agent.policy).to(args.device)
        wrapper.eval()
        dummy = torch.zeros(1, state_dim)
        scripted = torch.jit.trace(wrapper, dummy)
        scripted.save(args.export)
        print(f"\nTorchScript policy exported → {args.export}")
        print("Load in policy_runner with:")
        print(f"  ros2 launch policy_runner policy_runner.launch.py policy_path:={args.export}")

    env.close()


if __name__ == '__main__':
    main()
