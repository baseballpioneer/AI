"""
Final Working CityFlow RL Training
3-lane intersection: Left turn | Straight | Straight+Right
Best practices from both approaches combined
"""

import sys
import os
import json
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    import cityflow
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    import torch
    print("✓ All dependencies installed\n")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)


class ProgressCallback(BaseCallback):
    """Show episode-by-episode progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_count = 0
        
    def _on_step(self):
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_count += 1
                    reward = info['episode']['r']
                    length = info['episode']['l']
                    self.episode_rewards.append(reward)
                    
                    # Print every episode
                    recent = self.episode_rewards[-5:] if len(self.episode_rewards) >= 5 else self.episode_rewards
                    print(f"Episode {self.episode_count}: "
                          f"Reward={reward:.1f}, "
                          f"Length={length}, "
                          f"Recent avg={np.mean(recent):.1f}")
        return True


class CityFlowGymEnv(gym.Env):
    """
    3-Lane Intersection Environment
    Each direction has: Left turn lane | Straight lane | Straight+Right lane
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config_file, steps_per_episode=3600):
        super().__init__()
        
        self.eng = cityflow.Engine(config_file, thread_num=4)  # Use 4 threads
        self.steps_per_episode = steps_per_episode
        
        # Load config
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
        
        # Warm up
        for _ in range(10):
            self.eng.next_step()
        
        # Get INBOUND lanes only (like Gemini's approach)
        all_lanes = self.eng.get_lane_vehicles().keys()
        self.in_lanes = sorted([lane for lane in all_lanes if '_in_' in lane])
        self.num_lanes = len(self.in_lanes)
        
        print(f"Environment initialized:")
        print(f"  Intersection: {self.intersection_id}")
        print(f"  Inbound lanes: {self.num_lanes}")
        print(f"  Phases: {self.num_phases}")
        print(f"  Lanes: {self.in_lanes}")
        
        # Observation: [vehicle_count, wait_count] per lane + phase + time_in_phase
        # All normalized to 0-1
        obs_dim = self.num_lanes * 2 + 2
        self.observation_space = spaces.Box(
            low=0, 
            high=1,
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self.num_phases)
        
        # State
        self.current_phase = 0
        self.phase_time = 0
        self.min_phase_time = 5
        self.yellow_time = 3
        self.current_step = 0
        
        # Metrics
        self.episode_waiting_time = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.eng.reset()
        self.current_phase = np.random.randint(0, self.num_phases)  # Random start
        self.phase_time = 0
        self.current_step = 0
        self.episode_waiting_time = 0
        self.prev_waiting = 0
        
        self.eng.set_tl_phase(self.intersection_id, self.current_phase)
        
        for _ in range(10):
            self.eng.next_step()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        # Yellow light transition
        if action != self.current_phase and self.phase_time >= self.min_phase_time:
            for _ in range(self.yellow_time):
                self.eng.next_step()
                self.current_step += 1
            
            self.current_phase = action
            self.eng.set_tl_phase(self.intersection_id, self.current_phase)
            self.phase_time = 0
        
        # Regular step
        self.eng.next_step()
        self.phase_time += 1
        self.current_step += 1
        
        # Get observation and reward
        observation = self._get_observation()
        reward = self._calculate_reward()
        
        # Termination
        terminated = self.current_step >= self.steps_per_episode
        truncated = False
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """Normalized observation like Gemini's approach"""
        lv = self.eng.get_lane_vehicles()
        lw = self.eng.get_lane_waiting_vehicle_count()
        
        obs = []
        for lane in self.in_lanes:
            # Normalize: max 20 vehicles per lane
            vehicle_count = min(len(lv.get(lane, [])), 20) / 20.0
            waiting_count = min(lw.get(lane, 0), 20) / 20.0
            obs.extend([vehicle_count, waiting_count])
        
        # Current phase (normalized)
        obs.append(self.current_phase / max(self.num_phases - 1, 1))
        
        # Time in phase (normalized, cap at 60s)
        obs.append(min(self.phase_time, 60) / 60.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self):
        """
        Reward based on PRESSURE REDUCTION - not absolute waiting
        This encourages phase switching!
        """
        lw = self.eng.get_lane_waiting_vehicle_count()
        lv = self.eng.get_lane_vehicles()
        
        # Get current state
        inbound_waits = [lw.get(lane, 0) for lane in self.in_lanes]
        total_waiting = sum(inbound_waits)
        
        # PRESSURE-BASED: Reward for REDUCING waiting from last step
        if hasattr(self, 'prev_waiting'):
            pressure_change = self.prev_waiting - total_waiting
            # Positive reward for reducing waiting!
            pressure_reward = pressure_change * 1.0
        else:
            pressure_reward = 0
        
        self.prev_waiting = total_waiting
        
        # Small penalty for total waiting (secondary)
        waiting_penalty = -total_waiting * 0.1
        
        # Bonus for serving the busiest direction
        if len(inbound_waits) > 0 and max(inbound_waits) > 5:
            # Which lanes does current phase serve?
            # Simplified: assume phase 0=lanes 0-2, phase 1=lanes 3-5, etc.
            phase_start = self.current_phase * 3
            phase_end = min(phase_start + 3, len(inbound_waits))
            phase_lanes = inbound_waits[phase_start:phase_end]
            
            if phase_lanes and sum(phase_lanes) > 0:
                serving_bonus = 1.0
            else:
                serving_bonus = 0
        else:
            serving_bonus = 0
        
        reward = pressure_reward + waiting_penalty + serving_bonus
        
        # Track total waiting
        self.episode_waiting_time += total_waiting
        
        # Clip
        reward = np.clip(reward, -10, 10)
        
        return float(reward)
    
    def _get_info(self):
        lw = self.eng.get_lane_waiting_vehicle_count()
        
        return {
            'step': self.current_step,
            'phase': self.current_phase,
            'phase_time': self.phase_time,
            'waiting_vehicles': sum(lw.get(lane, 0) for lane in self.in_lanes),
            'avg_waiting_time': self.episode_waiting_time / max(self.current_step, 1)
        }
    
    def render(self):
        pass
    
    def close(self):
        pass


CONFIGS = {
    "test": {"timesteps": 10000, "episode_length": 500, "desc": "Quick test (10-15 min)"},
    "dev": {"timesteps": 100000, "episode_length": 1000, "desc": "Development (2-3 hours)"},
    "medium": {"timesteps": 200000, "episode_length": 1800, "desc": "Research (4-6 hours)"},
    "full": {"timesteps": 500000, "episode_length": 3600, "desc": "Full training (10-15 hours)"}
}


def train(config_file, mode="test", algorithm="PPO"):
    """Train with best practices"""
    
    if mode not in CONFIGS:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)
    
    config = CONFIGS[mode]
    
    print("\n" + "="*70)
    print(f"FINAL TRAINING: {mode.upper()} MODE with {algorithm}")
    print("="*70)
    print(f"Total steps:    {config['timesteps']:,}")
    print(f"Episode length: {config['episode_length']}")
    print(f"Key features:")
    print(f"  • 3-lane intersection (left, straight, straight+right)")
    print(f"  • Squared waiting penalty (punishes congestion hard)")
    print(f"  • Inbound lanes only (cleaner state)")
    print(f"  • Yellow light transitions (realistic)")
    print(f"  • Normalized observations (0-1 range)")
    print("="*70 + "\n")
    
    os.makedirs("models", exist_ok=True)
    
    print("Creating environment...")
    env = CityFlowGymEnv(config_file, steps_per_episode=config['episode_length'])
    env = Monitor(env)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix=f'{algorithm.lower()}_final'
    )
    
    progress_callback = ProgressCallback(verbose=1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    if algorithm == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.0001,
            buffer_size=100000,
            learning_starts=5000,
            batch_size=64,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.3,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=1,
            device=device
        )
    
    elif algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,  # INCREASED from 0.01 - much more exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=device
        )
    
    else:
        print(f"❌ Unknown algorithm: {algorithm}")
        sys.exit(1)
    
    print("Starting training with progress monitoring...")
    print("="*70 + "\n")
    
    try:
        model.learn(
            total_timesteps=config['timesteps'],
            callback=[checkpoint_callback, progress_callback],
            progress_bar=True
        )
        
        model_path = f"models/{algorithm.lower()}_final_{mode}"
        model.save(model_path)
        
        print("\n" + "="*70)
        print("✓ TRAINING COMPLETE!")
        print("="*70)
        print(f"Model saved: {model_path}.zip")
        print("="*70 + "\n")
        
        return model
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving model...")
        model_path = f"models/{algorithm.lower()}_final_interrupted_{mode}"
        model.save(model_path)
        print(f"✓ Saved: {model_path}.zip")
        return model


if __name__ == "__main__":
    if not os.path.exists("config.json"):
        print("❌ config.json not found!")
        sys.exit(1)
    
    if len(sys.argv) < 2:
        print("\nUsage: python final_train.py [mode] [algorithm]")
        print("\nModes: test, dev, medium, full")
        print("Algorithms: DQN, PPO")
        print("\nExample: python final_train.py test PPO")
        sys.exit(0)
    
    mode = sys.argv[1]
    algorithm = sys.argv[2] if len(sys.argv) > 2 else "PPO"
    
    # Use config.json (not config_3lane.json)
    train("config.json", mode, algorithm)