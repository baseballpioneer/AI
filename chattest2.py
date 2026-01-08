"""
FIXED CityFlow RL Training
Addresses the degenerate solution problem
"""

import sys
import os
import json
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    import cityflow
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    import torch
    print("✓ All dependencies installed\n")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)


class DebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    print(
                        f"Episode reward={info['episode']['r']:.1f}, "
                        f"length={info['episode']['l']}"
                    )
        return True


class CityFlowGymEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, config_file, steps_per_episode=3600):
        super().__init__()

        self.eng = cityflow.Engine(config_file, thread_num=1)
        self.steps_per_episode = steps_per_episode

        with open(config_file) as f:
            config = json.load(f)
        with open(config['roadnetFile']) as f:
            roadnet = json.load(f)

        for inter in roadnet['intersections']:
            if not inter.get('virtual', False):
                self.intersection_id = inter['id']
                self.num_phases = len(inter['trafficLight']['lightphases'])
                break

        for _ in range(10):
            self.eng.next_step()

        lane_vehicles = self.eng.get_lane_vehicles()
        self.lanes = sorted(lane_vehicles.keys())
        self.num_lanes = len(self.lanes)

        obs_dim = self.num_lanes * 2 + 2
        self.observation_space = spaces.Box(0, 100, (obs_dim,), np.float32)
        self.action_space = spaces.Discrete(self.num_phases)

        self.current_phase = 0
        self.phase_time = 0
        self.min_phase_time = 5
        self.yellow_time = 2
        self.current_step = 0
        self.episode_waiting_time = 0

        # 🔧 NEW: cache lane groups per phase (robust mapping)
        self.lanes_per_phase = max(1, self.num_lanes // self.num_phases)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.eng.reset()

        self.current_phase = np.random.randint(self.num_phases)
        self.phase_time = 0
        self.current_step = 0
        self.episode_waiting_time = 0

        self.eng.set_tl_phase(self.intersection_id, self.current_phase)

        for _ in range(5):
            self.eng.next_step()

        return self._get_observation(), {}

    def step(self, action):
        phase_changed = False

        if action != self.current_phase and self.phase_time >= self.min_phase_time:
            phase_changed = True
            for _ in range(self.yellow_time):
                self.eng.next_step()
                self.current_step += 1

            self.current_phase = action
            self.eng.set_tl_phase(self.intersection_id, action)
            self.phase_time = 0

        self.eng.next_step()
        self.phase_time += 1
        self.current_step += 1

        obs = self._get_observation()
        reward = self._calculate_reward(phase_changed)

        terminated = self.current_step >= self.steps_per_episode
        return obs, reward, terminated, False, self._get_info()

    def _get_observation(self):
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()

        obs = []
        for lane in self.lanes:
            obs.append(min(len(lane_vehicles.get(lane, [])), 50) * 2)
            obs.append(min(lane_waiting.get(lane, 0), 50) * 2)

        obs.append(self.current_phase * (100 / self.num_phases))
        obs.append(min(self.phase_time * 2, 100))
        return np.array(obs, dtype=np.float32)

    def _calculate_reward(self, phase_changed):
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        waits = [lane_waiting.get(l, 0) for l in self.lanes]

        total_wait = sum(waits)

        # Core objective
        reward = -0.1 * total_wait

        # 🔧 FIX 1: imbalance penalty
        if waits:
            reward -= 0.05 * (max(waits) - min(waits))

        # 🔧 FIX 2: correct phase alignment (robust)
        busiest_lane = np.argmax(waits) if waits else 0
        correct_phase = busiest_lane // self.lanes_per_phase
        if self.current_phase == correct_phase:
            reward += 0.5

        # 🔧 FIX 3: explicit incentive to switch when overloaded
        if phase_changed and total_wait > 10:
            reward += 1.0

        # 🔧 FIX 4: strong penalty for camping one phase
        if self.phase_time > 25:
            reward -= 0.1 * (self.phase_time - 25)

        reward = np.clip(reward, -10, 10)
        self.episode_waiting_time += total_wait
        return float(reward)

    def _get_info(self):
        return {
            "phase": self.current_phase,
            "phase_time": self.phase_time,
            "waiting": sum(self.eng.get_lane_waiting_vehicle_count().values())
        }


CONFIGS = {
    "test": {"timesteps": 10000, "episode_length": 500},
    "dev": {"timesteps": 100000, "episode_length": 1000},
    "medium": {"timesteps": 500000, "episode_length": 3600},
    "full": {"timesteps": 1000000, "episode_length": 3600}
}


def train(config_file, mode="test", algorithm="PPO"):
    cfg = CONFIGS[mode]
    env = Monitor(CityFlowGymEnv(config_file, cfg["episode_length"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        ent_coef=0.02,  # 🔧 encourages exploration
        verbose=1,
        device=device
    )

    model.learn(cfg["timesteps"], callback=DebugCallback())
    model.save(f"models/ppo_fixed_{mode}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    train("config.json", mode)
