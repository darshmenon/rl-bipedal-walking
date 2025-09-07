# src/rl_bipedal_walking/evaluation/evaluate_model.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import rclpy

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.bipedal_env import BipedalWalkingEnv
from agents.ppo_agent import PPOAgent

class ModelEvaluator:
    """Evaluate trained RL models"""
    
    def __init__(self, model_path, render=True):
        self.model_path = model_path
        self.render = render
        
        # Create environment
        print("Initializing environment...")
        self.env = BipedalWalkingEnv()
        
        # Create agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.agent = PPOAgent(state_dim, action_dim)
        
        # Load trained model
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.agent.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def evaluate(self, num_episodes=10):
        """Evaluate model over multiple episodes"""
        episode_rewards = []
        episode_lengths = []
        episode_data = []
        
        print(f"Evaluating model for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Run episode
            episode_reward, episode_length, trajectory = self.run_episode()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_data.append(trajectory)
            
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Final position: {trajectory[-1]['position']}")
            print(f"  Max distance: {max([d['position'][0] for d in trajectory]):.2f}m")
        
        # Print summary statistics
        print(f"\n=== Evaluation Summary ===")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
        print(f"Success rate: {sum([r > 0 for r in episode_rewards])/len(episode_rewards)*100:.1f}%")
        
        return {
            'rewards': episode_rewards,
            'lengths': episode_lengths,
            'trajectories': episode_data
        }
    
    def run_episode(self, max_steps=2000):
        """Run a single evaluation episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        trajectory = []
        
        while episode_length < max_steps:
            # Select action (deterministic)
            action, _, _ = self.agent.select_action(state, deterministic=True)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store trajectory data
            trajectory.append({
                'step': episode_length,
                'state': state.copy(),
                'action': action.copy(),
                'reward': reward,
                'position': info['robot_position'].copy(),
                'orientation': info['robot_orientation'].copy(),
                'velocity': info['forward_velocity']
            })
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        return episode_reward, episode_length, trajectory
    
    def plot_trajectory(self, trajectory, save_path=None):
        """Plot trajectory data"""
        steps = [d['step'] for d in trajectory]
        positions = np.array([d['position'] for d in trajectory])
        orientations = np.array([d['orientation'] for d in trajectory])
        rewards = [d['reward'] for d in trajectory]
        velocities = [d['velocity'] for d in trajectory]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Position trajectory
        axes[0, 0].plot(positions[:, 0], positions[:, 1])
        axes[0, 0].set_xlabel('X Position (m)')
        axes[0, 0].set_ylabel('Y Position (m)')
        axes[0, 0].set_title('XY Trajectory')
        axes[0, 0].grid(True)
        axes[0, 0].axis('equal')
        
        # Height over time
        axes[0, 1].plot(steps, positions[:, 2])
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Height (m)')
        axes[0, 1].set_title('Height over Time')
        axes[0, 1].grid(True)
        
        # Orientation over time
        axes[0, 2].plot(steps, orientations[:, 0], label='Roll')
        axes[0, 2].plot(steps, orientations[:, 1], label='Pitch') 
        axes[0, 2].plot(steps, orientations[:, 2], label='Yaw')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Angle (rad)')
        axes[0, 2].set_title('Orientation over Time')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Velocity over time
        axes[1, 0].plot(steps, velocities)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Forward Velocity (m/s)')
        axes[1, 0].set_title('Forward Velocity over Time')
        axes[1, 0].grid(True)
        
        # Rewards over time
        axes[1, 1].plot(steps, rewards)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Reward')
        axes[1, 1].set_title('Rewards over Time')
        axes[1, 1].grid(True)
        
        # Forward distance over time
        axes[1, 2].plot(steps, positions[:, 0])
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Forward Distance (m)')
        axes[1, 2].set_title('Forward Progress')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Trajectory plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_models(self, model_paths, labels=None, num_episodes=5):
        """Compare multiple models"""
        if labels is None:
            labels = [f"Model {i+1}" for i in range(len(model_paths))]
        
        results = {}
        
        for i, (model_path, label) in enumerate(zip(model_paths, labels)):
            print(f"\nEvaluating {label}...")
            
            # Load model
            self.agent.load_model(model_path)
            
            # Evaluate
            result = self.evaluate(num_episodes)
            results[label] = result
        
        # Plot comparison
        self.plot_comparison(results)
        
        return results
    
    def plot_comparison(self, results, save_path=None):
        """Plot comparison between models"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        labels = list(results.keys())
        rewards = [results[label]['rewards'] for label in labels]
        lengths = [results[label]['lengths'] for label in labels]
        
        # Box plot of rewards
        axes[0].boxplot(rewards, labels=labels)
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Reward Distribution')
        axes[0].grid(True)
        
        # Box plot of episode lengths
        axes[1].boxplot(lengths, labels=labels)
        axes[1].set_ylabel('Episode Length')
        axes[1].set_title('Episode Length Distribution')
        axes[1].grid(True)
        
        # Mean rewards comparison
        mean_rewards = [np.mean(rewards[i]) for i in range(len(labels))]
        std_rewards = [np.std(rewards[i]) for i in range(len(labels))]
        
        axes[2].bar(labels, mean_rewards, yerr=std_rewards, capsize=5)
        axes[2].set_ylabel('Mean Episode Reward')
        axes[2].set_title('Average Performance')
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate bipedal walking robot')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--save-plots', type=str, default=None,
                       help='Directory to save plots')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Compare multiple models')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Compare multiple models
            evaluator = ModelEvaluator(args.compare[0], render=not args.no_render)
            results = evaluator.compare_models(args.compare, num_episodes=args.episodes)
            
            if args.save_plots:
                os.makedirs(args.save_plots, exist_ok=True)
                evaluator.plot_comparison(results, 
                                        os.path.join(args.save_plots, 'model_comparison.png'))
        else:
            # Evaluate single model
            evaluator = ModelEvaluator(args.model, render=not args.no_render)
            results = evaluator.evaluate(args.episodes)
            
            # Plot best trajectory
            best_episode_idx = np.argmax(results['rewards'])
            best_trajectory = results['trajectories'][best_episode_idx]
            
            if args.save_plots:
                os.makedirs(args.save_plots, exist_ok=True)
                evaluator.plot_trajectory(best_trajectory,
                                        os.path.join(args.save_plots, 'best_trajectory.png'))
            else:
                evaluator.plot_trajectory(best_trajectory)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        raise
    finally:
        print("Evaluation session ended")

if __name__ == "__main__":
    main()