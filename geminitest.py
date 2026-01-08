import sys
import os
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
import cityflow
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Progress bar for training
class TqdmCallback(BaseCallback):
    def __init__(self, total_steps):
        super().__init__()
        self.pbar = None
        self.total_steps = total_steps
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_steps, desc="Training High-Res Agent")
    def _on_step(self):
        self.pbar.update(1)
        return True
    def _on_training_end(self):
        self.pbar.close()

class CityFlowGymEnv(gym.Env):
    def __init__(self, config_file="config.json"):
        super().__init__()
        # Ensure config.json has "interval": 0.1
        self.eng = cityflow.Engine(config_file, thread_num=4) 
        self.in_lanes = [f"{r}_in_{i}" for r in ["road_north", "road_east", "road_south", "road_west"] for i in range(3)]
        
        # 26 Dims: (12 lane vehicle counts + 12 lane wait counts + phase + time)
        self.observation_space = spaces.Box(low=0, high=1, shape=(26,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.current_phase = 0
        self.phase_time = 0
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.eng.reset()
        self.current_phase = 0
        self.phase_time = 0
        self.current_step = 0
        return self._get_observation(), {}

    def step(self, action):
        # 1. Yellow Light Logic (3.0 seconds = 30 steps at 0.1 interval)
        if action != self.current_phase:
            # Simple Yellow phase (no state updates here)
            for _ in range(30): 
                self.eng.next_step()
            self.current_phase = int(action)
            self.eng.set_tl_phase("intersection_1_1", self.current_phase)
            self.phase_time = 0
        
        # 2. Decision Duration (5.0 seconds = 50 steps at 0.1 interval)
        for _ in range(50):
            self.eng.next_step()
            self.phase_time += 0.1
            self.current_step += 1
            
        obs = self._get_observation()
        
        # 3. Reward Calculation (Pressure-based with Squared Penalty)
        lw = self.eng.get_lane_waiting_vehicle_count()
        # Only count waiting cars on INBOUND lanes
        inbound_waits = [lw.get(l, 0) for l in self.in_lanes]
        reward = -0.1 * sum([v**2 for v in inbound_waits])
        
        # Penalize staying in one phase for > 60 seconds
        if self.phase_time > 60:
            reward -= 10.0
            
        # End episode after 1500 'seconds' (15,000 steps)
        terminated = self.current_step >= 15000 
        return obs, float(reward), terminated, False, {}

    def _get_observation(self):
        lv = self.eng.get_lane_vehicles()
        lw = self.eng.get_lane_waiting_vehicle_count()
        obs = []
        for l in self.in_lanes:
            # Normalize: Max 20 cars per lane
            obs.append(min(len(lv.get(l, [])), 20) / 20.0)
            obs.append(min(lw.get(l, 0), 20) / 20.0)
        obs.append(self.current_phase / 3.0)
        obs.append(min(self.phase_time, 60) / 60.0)
        return np.array(obs, dtype=np.float32)

if __name__ == "__main__":
    env = Monitor(CityFlowGymEnv("config.json"))
    
    # Tuned for high-res simulation
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048, 
        batch_size=128,
        ent_coef=0.01, # Stable behavior
        gamma=0.99
    )
    
    # 200,000 steps = roughly 13 episodes of training
    total_training_steps = 20000 
    model.learn(total_timesteps=total_training_steps, callback=TqdmCallback(total_training_steps))
    
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_realistic_test")
    print("✅ High-Res Model Saved.")