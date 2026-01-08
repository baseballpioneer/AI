"""
Evaluate and compare your trained model
Run: python evaluate_model.py models/ppo_final_test.zip
"""

import sys
import os
import numpy as np
from all_in_one import CityFlowGymEnv
from stable_baselines3 import PPO, DQN, A2C


def evaluate_model(model_path, num_episodes=10):
    """Evaluate a trained model and compare to random baseline"""
    
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Load model
    print(f"Loading model: {model_path}")
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "dqn" in model_path.lower():
        model = DQN.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    else:
        print("❌ Unknown model type")
        sys.exit(1)
    
    # Create environment
    env = CityFlowGymEnv("config.json", steps_per_episode=1000)
    
    print(f"\nRunning {num_episodes} evaluation episodes...\n")
    
    # Test trained model
    print("Testing TRAINED model:")
    print("-" * 70)
    trained_rewards = []
    trained_wait_times = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        trained_rewards.append(episode_reward)
        trained_wait_times.append(info['avg_waiting_time'])
        print(f"  Episode {ep+1}: Reward={episode_reward:7.1f}, Wait={info['avg_waiting_time']:.2f}s")
    
    # Test random baseline
    print("\nTesting RANDOM baseline (for comparison):")
    print("-" * 70)
    random_rewards = []
    random_wait_times = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        random_rewards.append(episode_reward)
        random_wait_times.append(info['avg_waiting_time'])
        print(f"  Episode {ep+1}: Reward={episode_reward:7.1f}, Wait={info['avg_waiting_time']:.2f}s")
    
    # Calculate statistics
    trained_mean_reward = np.mean(trained_rewards)
    trained_std_reward = np.std(trained_rewards)
    trained_mean_wait = np.mean(trained_wait_times)
    trained_std_wait = np.std(trained_wait_times)
    
    random_mean_reward = np.mean(random_rewards)
    random_std_reward = np.std(random_rewards)
    random_mean_wait = np.mean(random_wait_times)
    random_std_wait = np.std(random_wait_times)
    
    # Calculate improvement
    reward_improvement = ((trained_mean_reward - random_mean_reward) / abs(random_mean_reward)) * 100
    wait_improvement = ((random_mean_wait - trained_mean_wait) / random_mean_wait) * 100
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    print(f"\n{'Metric':<25} {'Trained Model':<20} {'Random Baseline':<20} {'Improvement'}")
    print("-" * 70)
    print(f"{'Mean Reward':<25} {trained_mean_reward:>8.1f} ± {trained_std_reward:<8.1f} "
          f"{random_mean_reward:>8.1f} ± {random_std_reward:<8.1f} "
          f"{reward_improvement:>+6.1f}%")
    print(f"{'Mean Wait Time (s)':<25} {trained_mean_wait:>8.2f} ± {trained_std_wait:<8.2f} "
          f"{random_mean_wait:>8.2f} ± {random_std_wait:<8.2f} "
          f"{wait_improvement:>+6.1f}%")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if reward_improvement > 10:
        print("✅ GOOD: Model performs significantly better than random!")
        print(f"   Reward improved by {reward_improvement:.1f}%")
    elif reward_improvement > 0:
        print("⚠️  OKAY: Model is slightly better, but needs more training")
        print(f"   Reward improved by {reward_improvement:.1f}%")
    else:
        print("❌ POOR: Model is not better than random")
        print("   Need more training or better hyperparameters")
    
    if wait_improvement > 10:
        print(f"✅ GOOD: Wait time reduced by {wait_improvement:.1f}%")
    elif wait_improvement > 0:
        print(f"⚠️  OKAY: Wait time reduced by {wait_improvement:.1f}% (could be better)")
    else:
        print(f"❌ POOR: Wait time not improved ({wait_improvement:.1f}%)")
    
    print("\nRecommendations:")
    if reward_improvement < 20:
        print("  • Train longer (try 'dev' or 'medium' mode)")
        print("  • Check reward function design")
        print("  • Try different algorithm (PPO vs DQN)")
    else:
        print("  • Model is learning well!")
        print("  • Try longer training for even better results")
    
    print("="*70 + "\n")
    
    return {
        'trained_reward': trained_mean_reward,
        'random_reward': random_mean_reward,
        'improvement': reward_improvement,
        'trained_wait': trained_mean_wait,
        'random_wait': random_mean_wait
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python evaluate_model.py [model_path]")
        print("\nExample:")
        print("  python evaluate_model.py models/ppo_final_test.zip")
        print("\nAvailable models:")
        if os.path.exists("models"):
            for f in os.listdir("models"):
                if f.endswith(".zip"):
                    print(f"  - models/{f}")
        sys.exit(0)
    
    model_path = sys.argv[1]
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    
    evaluate_model(model_path, num_episodes=10)