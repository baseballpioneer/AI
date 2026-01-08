import os
import sys
import json
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import cityflow

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import torch


class DebugCallback(BaseCallback):
    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                print(
                    f"Episode | Reward: {info['episode']['r']:.1f} | "
                    f"Steps: {info['episode']['l']}"
                )
        return True


class CityFlowEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config_file, steps_per_episode=600):
        super().__init__()

        self.eng = cityflow.Engine(config_file, thread_num=1)
        self.steps_per_episode = steps_per_episode

        with open(config_file) as f:
            cfg = json.load(f)

        with open(cfg["roadnetFile"]) as f:
            roadnet = json.load(f)

        for inter in roadnet["intersections"]:
            if not inter.get("virtual", False):
                self.intersection_id = inter["id"]
                self.num_phases = len(inter["trafficLight"]["lightphases"])
                break

        self.lanes = sorted(self.eng.get_lane_vehicles().keys())
        self.num_lanes = len(self.lanes)

        self.action_space = spaces.Discrete(2)  # 0 = HOLD, 1 = SWITCH

        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(self.num_lanes * 2 + 1,),
            dtype=np.float32
        )

        self.min_phase_time = 5
        self.yellow_time = 2

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.eng.reset()
        self.current_phase = np.random.randint(self.num_phases)
        self.phase_time = 0
        self.current_step = 0

        self.eng.set_tl_phase(self.intersection_id, self.current_phase)

        for _ in range(5):
            self.eng.next_step()

        self.prev_total_wait = self._total_waiting()

        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0

        if action == 1 and self.phase_time < self.min_phase_time:
            reward -= 0.5

        if action == 1 and self.phase_time >= self.min_phase_time:
            for _ in range(self.yellow_time):
                self.eng.next_step()
                reward -= 0.1

            self.current_phase = (self.current_phase + 1) % self.num_phases
            self.eng.set_tl_phase(self.intersection_id, self.current_phase)
            self.phase_time = 0

        self.eng.next_step()
        self.phase_time += 1
        self.current_step += 1

        total_wait = self._total_waiting()
        delta = self.prev_total_wait - total_wait
        reward += delta * 0.2
        self.prev_total_wait = total_wait

        if self.phase_time > 30:
            reward -= 0.1 * (self.phase_time - 30)

        reward = float(np.clip(reward, -10, 10))

        terminated = self.current_step >= self.steps_per_episode

        info = {
            "phase": self.current_phase,
            "phase_time": self.phase_time,
            "waiting": total_wait
        }

        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()

        obs = []
        for lane in self.lanes:
            obs.append(min(len(lane_vehicles.get(lane, [])), 50) * 2)
            obs.append(min(lane_waiting.get(lane, 0), 50) * 2)

        obs.append(min(self.phase_time * 2, 100))

        return np.array(obs, dtype=np.float32)

    def _total_waiting(self):
        return sum(self.eng.get_lane_waiting_vehicle_count().values())


def train():
    env = CityFlowEnv("config.json", steps_per_episode=600)
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.05,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    model.learn(
        total_timesteps=200_000,
        callback=DebugCallback()
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_cityflow_fixed")


if __name__ == "__main__":
    train()
