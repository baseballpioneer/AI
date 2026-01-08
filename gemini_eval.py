import sys
import os
import numpy as np
import cityflow
from stable_baselines3 import PPO

def run_simulation(eng, mode="rl", model=None, steps_per_ep=1500, in_lanes=[]):
    eng.reset()
    current_phase = 0
    phase_time = 0.0
    total_reward = 0
    waits = []
    
    # Total steps = Seconds / Interval (1500 / 0.1 = 15000)
    total_steps = int(steps_per_ep * 10)

    for step in range(0, total_steps, 50): # Decision every 5 seconds (50 steps)
        # 1. Action Selection
        if mode == "rl" and model:
            lv = eng.get_lane_vehicles()
            lw = eng.get_lane_waiting_vehicle_count()
            obs = []
            for l in in_lanes:
                obs.append(min(len(lv.get(l, [])), 20) / 20.0)
                obs.append(min(lw.get(l, 0), 20) / 20.0)
            obs.append(current_phase / 3.0)
            obs.append(min(phase_time, 60) / 60.0)
            action, _ = model.predict(np.array(obs, dtype=np.float32), deterministic=True)
        
        elif mode == "fixed":
            action = (current_phase + 1) % 4 if phase_time >= 30 else current_phase
            
        elif mode == "green_wave":
            limit = 60 if current_phase in [0, 2] else 20
            action = (current_phase + 1) % 4 if phase_time >= limit else current_phase

        # 2. Execute Action with Yellow Light logic (30 steps = 3s)
        if action != current_phase:
            for _ in range(30): eng.next_step()
            current_phase = int(action)
            eng.set_tl_phase("intersection_1_1", current_phase)
            phase_time = 0.0
        
        # 3. Advance Simulation (50 steps = 5s)
        for _ in range(50):
            eng.next_step()
            phase_time += 0.1
        
        # 4. Data Collection
        lw_counts = eng.get_lane_waiting_vehicle_count()
        inbound_waits = [lw_counts.get(l, 0) for l in in_lanes]
        total_reward -= (0.1 * sum([v**2 for v in inbound_waits]))
        waits.append(sum(inbound_waits) / 12)

    return total_reward, np.mean(waits)

def evaluate_and_compare(model_path, config_file="config.json"):
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    # Set thread_num higher for smoother local execution
    eng = cityflow.Engine(config_file, thread_num=4)
    in_lanes = [f"{r}_in_{i}" for r in ["road_north", "road_east", "road_south", "road_west"] for i in range(3)]
    
    print("\n" + "="*85)
    print("🚦 SMOOTH SIMULATION BENCHMARK (0.1s Interval)")
    print("="*85)

    model = PPO.load(model_path)
    
    print("Running RL Agent...")
    rl_r, rl_w = run_simulation(eng, mode="rl", model=model, in_lanes=in_lanes)

    print("Running Fixed Timer...")
    fx_r, fx_w = run_simulation(eng, mode="fixed", in_lanes=in_lanes)

    print("Running Green Wave...")
    gw_r, gw_w = run_simulation(eng, mode="green_wave", in_lanes=in_lanes)

    # Output Table
    print("\n" + "-" * 85)
    print(f"{'Metric':<25} | {'Fixed Timer':<15} | {'Green Wave':<15} | {'RL Agent (PPO)':<15}")
    print("-" * 85)
    print(f"{'Avg Wait (Veh/Lane)':<25} | {fx_w:<15.2f} | {gw_w:<15.2f} | {rl_w:<15.2f}")
    print(f"{'Total Penalty':<25} | {fx_r:<15.1f} | {gw_r:<15.1f} | {rl_r:<15.1f}")
    print("-" * 85)
    
    best_w = min(fx_w, gw_w)
    if rl_w < best_w:
        imp = ((best_w - rl_w) / best_w) * 100
        print(f"🌟 RL Agent wins by {imp:.1f}%!")
    else:
        print("⚠️  Baselines are still stronger. More training steps recommended.")
    print("="*85 + "\n")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "models/ppo_realistic_test.zip"
    evaluate_and_compare(path)