"""
Professional CityFlow RL Setup using Gymnasium and Stable-Baselines3
This is the proper way to do RL research
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cityflow
import json
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch


class CityFlowGymEnv(gym.Env):
    """
    Gymnasium-compliant CityFlow environment
    Compatible with Stable-Baselines3 and other RL libraries
    """
    
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
        
        # Define observation and action spaces (required by Gymnasium)
        # Observation: [vehicle_count, waiting_count] for each lane + current phase
        obs_dim = self.num_lanes * 2 + 1
        self.observation_space = spaces.Box(
            low=0, 
            high=50,  # Max vehicles per lane
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Action: which phase to use
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
        
        print(f"Environment initialized:")
        print(f"  Intersection: {self.intersection_id}")
        print(f"  Lanes: {self.num_lanes}")
        print(f"  Phases: {self.num_phases}")
        print(f"  Observation space: {self.observation_space.shape}")
        print(f"  Action space: {self.action_space.n}")
        
    def reset(self, seed=None, options=None):
        """Reset environment (Gymnasium API)"""
        super().reset(seed=seed)
        
        self.eng.reset()
        self.current_phase = 0
        self.phase_time = 0
        self.current_step = 0
        self.episode_waiting_time = 0
        self.episode_vehicles = 0
        
        self.eng.set_tl_phase(self.intersection_id, self.current_phase)
        
        # Warm-up
        for _ in range(10):
            self.eng.next_step()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute action (Gymnasium API)"""
        
        # Phase change logic
        if action != self.current_phase and self.phase_time >= self.min_phase_time:
            # Yellow phase
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
        """Get current state observation"""
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        
        obs = []
        for lane in self.lanes:
            num_vehicles = len(lane_vehicles.get(lane, []))
            num_waiting = lane_waiting.get(lane, 0)
            obs.extend([num_vehicles, num_waiting])
        
        # Add current phase
        obs.append(self.current_phase)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self):
        """Calculate reward"""
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        lane_vehicles = self.eng.get_lane_vehicles()
        
        # Negative reward for waiting (primary objective)
        waiting_time = sum(lane_waiting.values())
        
        # Negative reward for queue length (secondary)
        queue_length = sum(len(vehicles) for vehicles in lane_vehicles.values())
        
        # Combined reward
        reward = -(waiting_time * 1.0 + queue_length * 0.1)
        
        # Track metrics
        self.episode_waiting_time += waiting_time
        self.episode_vehicles += queue_length
        
        return reward
    
    def _get_info(self):
        """Additional info (Gymnasium API)"""
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
        """Render (optional)"""
        pass
    
    def close(self):
        """Cleanup"""
        pass


def train_with_stable_baselines3(config_file, algorithm="DQN", total_timesteps=100000):
    """
    Train using Stable-Baselines3 (industry standard)
    
    Args:
        config_file: CityFlow config
        algorithm: "DQN", "PPO", or "A2C"
        total_timesteps: Total training steps (100k = quick test, 1M+ = real training)
    """
    
    print("\n" + "="*70)
    print(f"PROFESSIONAL RL TRAINING - {algorithm}")
    print("="*70 + "\n")
    
    # Create environment
    env = CityFlowGymEnv(config_file, steps_per_episode=3600)
    env = Monitor(env)  # Wrap for logging
    
    # Callbacks for checkpointing and evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix=f'{algorithm.lower()}_model'
    )
    
    # Choose algorithm
    print(f"Initializing {algorithm}...")
    
    if algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.0001,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            tensorboard_log="./tensorboard/",
            device='cuda' if torch.cuda.is_available() else 'cpu'
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
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./tensorboard/",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    elif algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=0.0007,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            verbose=1,
            tensorboard_log="./tensorboard/",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"\nDevice: {model.device}")
    print(f"Total timesteps to train: {total_timesteps:,}")
    print(f"Estimated time: {total_timesteps / 3600 / 10:.1f} hours (rough estimate)")
    print("\nStarting training...")
    print("="*70 + "\n")
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save(f"models/{algorithm.lower()}_final")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final model saved: models/{algorithm.lower()}_final.zip")
    print(f"\nTo view training curves:")
    print(f"  tensorboard --logdir ./tensorboard/")
    
    return model


def evaluate_model(model_path, config_file, num_episodes=10):
    """Evaluate a trained model"""
    
    print("\n" + "="*70)
    print("EVALUATING MODEL")
    print("="*70 + "\n")
    
    # Load environment
    env = CityFlowGymEnv(config_file, steps_per_episode=3600)
    
    # Load model
    if "dqn" in model_path.lower():
        model = DQN.load(model_path)
    elif "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "a2c" in model_path.lower():
        model = A2C.load(model_path)
    
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
              f"Avg Wait={info['avg_waiting_time']:.2f}s")
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Wait Time: {np.mean(episode_wait_times):.2f}s ± {np.std(episode_wait_times):.2f}s")
    
    return episode_rewards, episode_wait_times


# Installation instructions
INSTALL_INSTRUCTIONS = """
INSTALLATION REQUIRED:
----------------------
pip install gymnasium
pip install stable-baselines3[extra]
pip install tensorboard

Optional (for GPU acceleration):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""


if __name__ == "__main__":
    try:
        import gymnasium
        import stable_baselines3
        print("✓ All libraries installed\n")
    except ImportError:
        print(INSTALL_INSTRUCTIONS)
        exit(1)
    
    config_file = "config.json"
    
    # Quick test (10k steps = ~30 minutes)
    print("QUICK TEST MODE - 10,000 steps")
    print("For research-grade training, use 1,000,000+ steps")
    
    model = train_with_stable_baselines3(
        config_file,
        algorithm="DQN",        # or "PPO" or "A2C"
        total_timesteps=10000   # Increase to 1000000 for real training
    )
    
    # Evaluate
    evaluate_model("models/dqn_final", config_file, num_episodes=5)