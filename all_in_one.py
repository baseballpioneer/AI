"""
Complete CityFlow RL Training - All-in-One Script
Everything you need in a single file
"""

import sys
import os
import json
import numpy as np

# Check dependencies first
try:
    import gymnasium as gym
    from gymnasium import spaces
    import cityflow
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    import torch
    print("✓ All dependencies installed\n")
except ImportError as e:
    print("\n❌ Missing dependencies!")
    print("\nPlease install:")
    print("  pip install gymnasium stable-baselines3[extra] tensorboard torch")
    print(f"\nError: {e}")
    sys.exit(1)


# ==============================================================================
# GYMNASIUM ENVIRONMENT
# ==============================================================================

class CityFlowGymEnv(gym.Env):
    """Gymnasium-compliant CityFlow environment"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config_file, steps_per_episode=3600):
        super().__init__()
        
        self.eng = cityflow.Engine(config_file, thread_num=1)
        self.steps_per_episode = steps_per_episode
        
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        with open(config['roadnetFile'], 'r') as f:
            roadnet = json.load(f)
        
        # Find intersection
        for intersection in roadnet['intersections']:
            if not intersection.get('virtual', False):
                self.intersection_id = intersection['id']
                if 'trafficLight' in intersection:
                    self.num_phases = len(intersection['trafficLight']['lightphases'])
                break
        
        # Initialize lanes
        for _ in range(10):
            self.eng.next_step()
        
        lane_vehicles = self.eng.get_lane_vehicles()
        self.lanes = sorted([lane for lane in lane_vehicles.keys() if '_' in lane])
        self.num_lanes = len(self.lanes)
        
        # Define spaces
        obs_dim = self.num_lanes * 2 + 1
        self.observation_space = spaces.Box(
            low=0, 
            high=50,
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self.num_phases)
        
        # State tracking
        self.current_phase = 0
        self.phase_time = 0
        self.min_phase_time = 10
        self.yellow_time = 3
        self.current_step = 0
        
        # Metrics
        self.episode_waiting_time = 0
        self.episode_vehicles = 0
        self.previous_waiting = 0
        self.previous_vehicles = 0
        
        print(f"Environment initialized:")
        print(f"  Intersection: {self.intersection_id}")
        print(f"  Lanes: {self.num_lanes}")
        print(f"  Phases: {self.num_phases}")
        print(f"  Observation space: {self.observation_space.shape}")
        print(f"  Action space: {self.action_space.n}")
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.eng.reset()
        self.current_phase = 0
        self.phase_time = 0
        self.current_step = 0
        self.episode_waiting_time = 0
        self.episode_vehicles = 0
        self.previous_waiting = 0
        self.previous_vehicles = 0
        
        self.eng.set_tl_phase(self.intersection_id, self.current_phase)
        
        for _ in range(10):
            self.eng.next_step()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        # Phase change logic
        if action != self.current_phase and self.phase_time >= self.min_phase_time:
            for _ in range(self.yellow_time):
                self.eng.next_step()
                self.current_step += 1
            
            self.current_phase = action
            self.eng.set_tl_phase(self.intersection_id, self.current_phase)
            self.phase_time = 0
        
        # Simulation step
        self.eng.next_step()
        self.phase_time += 1
        self.current_step += 1
        
        # Get observation and reward
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        # Episode termination
        terminated = self.current_step >= self.steps_per_episode
        truncated = False
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        
        obs = []
        for lane in self.lanes:
            num_vehicles = len(lane_vehicles.get(lane, []))
            num_waiting = lane_waiting.get(lane, 0)
            obs.extend([num_vehicles, num_waiting])
        
        obs.append(self.current_phase)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self):
        """Improved reward function with better scaling"""
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        lane_vehicles = self.eng.get_lane_vehicles()
        
        current_waiting = sum(lane_waiting.values())
        current_vehicles = sum(len(vehicles) for vehicles in lane_vehicles.values())
        
        # Normalize to make rewards more stable
        # Scale down large negative rewards
        waiting_penalty = -np.sqrt(current_waiting) if current_waiting > 0 else 0
        
        # Reward for throughput (vehicles passing through)
        if hasattr(self, 'previous_vehicles'):
            throughput = max(0, self.previous_vehicles - current_vehicles)
            throughput_reward = throughput * 0.5
        else:
            throughput_reward = 0
        
        # Small penalty for each waiting vehicle to encourage clearing
        queue_penalty = -current_vehicles * 0.01
        
        # Combined reward (scaled to reasonable range)
        reward = waiting_penalty + throughput_reward + queue_penalty
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -10, 10)
        
        # Track metrics
        self.episode_waiting_time += current_waiting
        self.episode_vehicles += current_vehicles
        self.previous_waiting = current_waiting
        self.previous_vehicles = current_vehicles
        
        return reward
    
    def _get_info(self):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        
        return {
            'step': self.current_step,
            'phase': self.current_phase,
            'waiting_vehicles': sum(lane_waiting.values()),
            'total_vehicles': self.eng.get_vehicle_count(),
            'episode_waiting_time': self.episode_waiting_time,
            'avg_waiting_time': self.episode_waiting_time / max(self.current_step, 1)
        }
    
    def render(self):
        pass
    
    def close(self):
        pass


# ==============================================================================
# TRAINING CONFIGURATIONS
# ==============================================================================

CONFIGS = {
    "test": {
        "timesteps": 5000,
        "episode_length": 300,
        "desc": "Quick test (5-10 min)"
    },
    "dev": {
        "timesteps": 50000,
        "episode_length": 1000,
        "desc": "Development (1-2 hours)"
    },
    "medium": {
        "timesteps": 500000,
        "episode_length": 3600,
        "desc": "Research results (12-24 hours)"
    },
    "full": {
        "timesteps": 1000000,
        "episode_length": 3600,
        "desc": "Full training (2-3 days)"
    }
}


# ==============================================================================
# TRAINING FUNCTION
# ==============================================================================

def train(config_file, mode="test", algorithm="PPO"):
    """Main training function"""
    
    if mode not in CONFIGS:
        print(f"❌ Unknown mode: {mode}")
        print(f"Available: {', '.join(CONFIGS.keys())}")
        sys.exit(1)
    
    config = CONFIGS[mode]
    
    print("\n" + "="*70)
    print(f"TRAINING: {mode.upper()} MODE with {algorithm}")
    print("="*70)
    print(f"Total steps:    {config['timesteps']:,}")
    print(f"Episode length: {config['episode_length']}")
    print(f"Description:    {config['desc']}")
    print(f"Est. time:      ~{config['timesteps']/1000/60:.1f} hours")
    print("="*70 + "\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = CityFlowGymEnv(config_file, steps_per_episode=config['episode_length'])
    env = Monitor(env)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix=f'{algorithm.lower()}_model'
    )
    
    # Create model
    print(f"\nInitializing {algorithm}...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.0001,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log="./tensorboard/",
            device=device
        )
    
    elif algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./tensorboard/",
            device=device
        )
    
    elif algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            verbose=1,
            tensorboard_log="./tensorboard/",
            device=device
        )
    
    else:
        print(f"❌ Unknown algorithm: {algorithm}")
        sys.exit(1)
    
    print("\nStarting training...")
    print("Press Ctrl+C to stop early (model will be saved)")
    print("="*70 + "\n")
    
    try:
        model.learn(
            total_timesteps=config['timesteps'],
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        model_path = f"models/{algorithm.lower()}_final_{mode}"
        model.save(model_path)
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)
        print(f"Model saved: {model_path}.zip")
        print(f"\nView training curves:")
        print(f"  tensorboard --logdir ./tensorboard/")
        print("="*70 + "\n")
        
        return model
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Saving model...")
        model_path = f"models/{algorithm.lower()}_interrupted_{mode}"
        model.save(model_path)
        print(f"✓ Model saved: {model_path}.zip")
        return model


# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================

def evaluate(model_path, config_file, num_episodes=10):
    """Evaluate trained model"""
    
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70 + "\n")
    
    env = CityFlowGymEnv(config_file, steps_per_episode=3600)
    
    # Load model
    if "dqn" in model_path.lower():
        model = DQN.load(model_path)
    elif "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    else:
        print("❌ Cannot determine model type from filename")
        sys.exit(1)
    
    print(f"Model loaded: {model_path}\n")
    
    episode_rewards = []
    episode_wait_times = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_wait_times.append(info['avg_waiting_time'])
        
        print(f"Episode {episode+1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"AvgWait={info['avg_waiting_time']:.2f}s")
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Mean Reward:    {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Wait Time: {np.mean(episode_wait_times):.2f}s ± {np.std(episode_wait_times):.2f}s")
    print(f"{'='*70}\n")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    if not os.path.exists("config.json"):
        print("❌ Error: config.json not found!")
        sys.exit(1)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("CITYFLOW RL TRAINING")
        print("="*70 + "\n")
        print("Usage:")
        print("  python all_in_one.py [mode] [algorithm]")
        print("\nModes:")
        for name, conf in CONFIGS.items():
            print(f"  {name:8} - {conf['desc']}")
        print("\nAlgorithms: DQN, PPO, A2C")
        print("\nExamples:")
        print("  python all_in_one.py test PPO")
        print("  python all_in_one.py medium DQN")
        print("  python all_in_one.py full PPO")
        print("\n" + "="*70 + "\n")
        sys.exit(0)
    
    mode = sys.argv[1]
    algorithm = sys.argv[2] if len(sys.argv) > 2 else "PPO"
    
    # Train
    train("config.json", mode, algorithm)