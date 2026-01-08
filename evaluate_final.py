"""
Comprehensive Evaluator for CityFlow RL Models
Compares against: Random, Fixed-Time, Green Wave, and Actuated baselines
STANDALONE - No external imports needed
"""

import sys
import os
import numpy as np
import json
import cityflow
from stable_baselines3 import PPO, DQN


class TrafficEvaluator:
    """Evaluate different traffic control strategies"""
    
    def __init__(self, config_file):
        self.config_file = config_file
        self.eng = cityflow.Engine(config_file, thread_num=1)
        
        # Load config to get number of phases
        with open(config_file, 'r') as f:
            config = json.load(f)
        with open(config['roadnetFile'], 'r') as f:
            roadnet = json.load(f)
        
        # Find intersection and phases
        for intersection in roadnet['intersections']:
            if not intersection.get('virtual', False):
                self.intersection_id = intersection['id']
                if 'trafficLight' in intersection:
                    self.num_phases = len(intersection['trafficLight']['lightphases'])
                break
        
        # Get inbound lanes
        for _ in range(10):
            self.eng.next_step()
            
        all_lanes = self.eng.get_lane_vehicles().keys()
        
        # Try to find inbound lanes
        in_lanes_candidates = sorted([lane for lane in all_lanes if '_in_' in lane])
        
        # If we have more than 12, take first 12 (to match training)
        if len(in_lanes_candidates) > 12:
            self.in_lanes = in_lanes_candidates[:12]
        elif len(in_lanes_candidates) > 0:
            self.in_lanes = in_lanes_candidates
        else:
            # Fallback: just take first lanes with underscores
            self.in_lanes = sorted([lane for lane in all_lanes if '_' in lane])[:12]
        
        self.num_lanes = len(self.in_lanes)
        
        print(f"Evaluator initialized:")
        print(f"  Intersection: {self.intersection_id}")
        print(f"  Phases: {self.num_phases}")
        print(f"  Inbound lanes: {self.num_lanes}")
        print(f"  Expected obs dim: {self.num_lanes * 2 + 2}")
        
    def _get_observation(self, current_phase, phase_time):
        """Get normalized observation for RL agent"""
        lv = self.eng.get_lane_vehicles()
        lw = self.eng.get_lane_waiting_vehicle_count()
        
        obs = []
        for lane in self.in_lanes:
            vehicle_count = min(len(lv.get(lane, [])), 20) / 20.0
            waiting_count = min(lw.get(lane, 0), 20) / 20.0
            obs.extend([vehicle_count, waiting_count])
        
        obs.append(current_phase / max(self.num_phases - 1, 1))
        obs.append(min(phase_time, 60) / 60.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_metrics(self):
        """Calculate current traffic metrics"""
        lw = self.eng.get_lane_waiting_vehicle_count()
        lv = self.eng.get_lane_vehicles()
        
        inbound_waits = [lw.get(lane, 0) for lane in self.in_lanes]
        inbound_vehicles = [len(lv.get(lane, [])) for lane in self.in_lanes]
        
        # Squared penalty (like training)
        reward = -0.1 * sum([w**2 for w in inbound_waits])
        
        return {
            'reward': reward,
            'waiting': sum(inbound_waits),
            'vehicles': sum(inbound_vehicles),
            'avg_waiting_per_lane': sum(inbound_waits) / self.num_lanes if self.num_lanes > 0 else 0,
            'max_waiting': max(inbound_waits) if inbound_waits else 0
        }
    
    def evaluate_rl_agent(self, model, steps=3600, yellow_time=3):
        """Evaluate trained RL agent"""
        
        self.eng.reset()
        current_phase = 0
        phase_time = 0
        self.eng.set_tl_phase(self.intersection_id, current_phase)
        
        total_reward = 0
        metrics_history = []
        phase_changes = 0
        action_counts = {i: 0 for i in range(self.num_phases)}
        
        for step in range(steps):
            # Get observation and action
            obs = self._get_observation(current_phase, phase_time)
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_counts[action] += 1
            
            # Yellow light transition
            if action != current_phase:
                for _ in range(yellow_time):
                    self.eng.next_step()
                phase_changes += 1
                current_phase = action
                self.eng.set_tl_phase(self.intersection_id, current_phase)
                phase_time = 0
            
            # Regular step
            self.eng.next_step()
            phase_time += 1
            
            # Collect metrics
            metrics = self._calculate_metrics()
            total_reward += metrics['reward']
            metrics_history.append(metrics)
        
        # Calculate action distribution
        total_actions = sum(action_counts.values())
        action_dist = {p: 100 * count / total_actions for p, count in action_counts.items()}
        
        return {
            'name': 'RL Agent',
            'total_reward': total_reward,
            'avg_waiting': np.mean([m['avg_waiting_per_lane'] for m in metrics_history]),
            'max_waiting': max([m['max_waiting'] for m in metrics_history]),
            'phase_changes': phase_changes,
            'action_dist': action_dist,
            'metrics_history': metrics_history
        }
    
    def evaluate_random(self, steps=3600, yellow_time=3):
        """Random phase selection baseline"""
        
        self.eng.reset()
        current_phase = 0
        phase_time = 0
        min_phase_time = 5
        self.eng.set_tl_phase(self.intersection_id, current_phase)
        
        total_reward = 0
        metrics_history = []
        phase_changes = 0
        
        for step in range(steps):
            if phase_time >= min_phase_time:
                action = np.random.randint(0, self.num_phases)
            else:
                action = current_phase
            
            if action != current_phase:
                for _ in range(yellow_time):
                    self.eng.next_step()
                phase_changes += 1
                current_phase = action
                self.eng.set_tl_phase(self.intersection_id, current_phase)
                phase_time = 0
            
            self.eng.next_step()
            phase_time += 1
            
            metrics = self._calculate_metrics()
            total_reward += metrics['reward']
            metrics_history.append(metrics)
        
        return {
            'name': 'Random',
            'total_reward': total_reward,
            'avg_waiting': np.mean([m['avg_waiting_per_lane'] for m in metrics_history]),
            'max_waiting': max([m['max_waiting'] for m in metrics_history]),
            'phase_changes': phase_changes,
            'metrics_history': metrics_history
        }
    
    def evaluate_fixed_time(self, steps=3600, cycle_time=30, yellow_time=3):
        """Fixed-time control baseline"""
        
        self.eng.reset()
        current_phase = 0
        phase_time = 0
        self.eng.set_tl_phase(self.intersection_id, current_phase)
        
        total_reward = 0
        metrics_history = []
        phase_changes = 0
        
        for step in range(steps):
            if phase_time >= cycle_time:
                action = (current_phase + 1) % self.num_phases
            else:
                action = current_phase
            
            if action != current_phase:
                for _ in range(yellow_time):
                    self.eng.next_step()
                phase_changes += 1
                current_phase = action
                self.eng.set_tl_phase(self.intersection_id, current_phase)
                phase_time = 0
            
            self.eng.next_step()
            phase_time += 1
            
            metrics = self._calculate_metrics()
            total_reward += metrics['reward']
            metrics_history.append(metrics)
        
        return {
            'name': 'Fixed-Time (30s)',
            'total_reward': total_reward,
            'avg_waiting': np.mean([m['avg_waiting_per_lane'] for m in metrics_history]),
            'max_waiting': max([m['max_waiting'] for m in metrics_history]),
            'phase_changes': phase_changes,
            'metrics_history': metrics_history
        }
    
    def evaluate_green_wave(self, steps=3600, yellow_time=3):
        """Green wave baseline"""
        
        self.eng.reset()
        current_phase = 0
        phase_time = 0
        self.eng.set_tl_phase(self.intersection_id, current_phase)
        
        total_reward = 0
        metrics_history = []
        phase_changes = 0
        
        phase_durations = [40, 40, 20, 20]
        
        for step in range(steps):
            if phase_time >= phase_durations[current_phase]:
                action = (current_phase + 1) % self.num_phases
            else:
                action = current_phase
            
            if action != current_phase:
                for _ in range(yellow_time):
                    self.eng.next_step()
                phase_changes += 1
                current_phase = action
                self.eng.set_tl_phase(self.intersection_id, current_phase)
                phase_time = 0
            
            self.eng.next_step()
            phase_time += 1
            
            metrics = self._calculate_metrics()
            total_reward += metrics['reward']
            metrics_history.append(metrics)
        
        return {
            'name': 'Green Wave',
            'total_reward': total_reward,
            'avg_waiting': np.mean([m['avg_waiting_per_lane'] for m in metrics_history]),
            'max_waiting': max([m['max_waiting'] for m in metrics_history]),
            'phase_changes': phase_changes,
            'metrics_history': metrics_history
        }
    
    def evaluate_actuated(self, steps=3600, min_time=10, max_time=60, yellow_time=3):
        """Actuated control"""
        
        self.eng.reset()
        current_phase = 0
        phase_time = 0
        self.eng.set_tl_phase(self.intersection_id, current_phase)
        
        total_reward = 0
        metrics_history = []
        phase_changes = 0
        
        for step in range(steps):
            lw = self.eng.get_lane_waiting_vehicle_count()
            inbound_waits = [lw.get(lane, 0) for lane in self.in_lanes]
            
            max_waiting_other = 0
            best_phase = current_phase
            
            for phase in range(self.num_phases):
                if phase != current_phase:
                    phase_waiting = sum(inbound_waits[phase*3:(phase+1)*3]) if phase*3 < len(inbound_waits) else 0
                    if phase_waiting > max_waiting_other:
                        max_waiting_other = phase_waiting
                        best_phase = phase
            
            current_waiting = sum(inbound_waits[current_phase*3:(current_phase+1)*3]) if current_phase*3 < len(inbound_waits) else 0
            
            should_switch = (
                phase_time >= min_time and 
                (phase_time >= max_time or max_waiting_other > current_waiting * 2)
            )
            
            if should_switch:
                action = best_phase
            else:
                action = current_phase
            
            if action != current_phase:
                for _ in range(yellow_time):
                    self.eng.next_step()
                phase_changes += 1
                current_phase = action
                self.eng.set_tl_phase(self.intersection_id, current_phase)
                phase_time = 0
            
            self.eng.next_step()
            phase_time += 1
            
            metrics = self._calculate_metrics()
            total_reward += metrics['reward']
            metrics_history.append(metrics)
        
        return {
            'name': 'Actuated',
            'total_reward': total_reward,
            'avg_waiting': np.mean([m['avg_waiting_per_lane'] for m in metrics_history]),
            'max_waiting': max([m['max_waiting'] for m in metrics_history]),
            'phase_changes': phase_changes,
            'metrics_history': metrics_history
        }


def evaluate_all(model_path, config_file, num_runs=5):
    """Run comprehensive evaluation"""
    
    print("\n" + "="*85)
    print("COMPREHENSIVE TRAFFIC CONTROL EVALUATION")
    print("="*85 + "\n")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Loading model: {model_path}")
    if "ppo" in model_path.lower():
        model = PPO.load(model_path)
    elif "dqn" in model_path.lower():
        model = DQN.load(model_path)
    else:
        print("❌ Unknown model type")
        return
    
    print(f"Running {num_runs} evaluation episodes per strategy...\n")
    
    evaluator = TrafficEvaluator(config_file)
    
    strategies = {
        'RL Agent': [],
        'Random': [],
        'Fixed-Time (30s)': [],
        'Green Wave': [],
        'Actuated': []
    }
    
    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}:")
        
        result = evaluator.evaluate_rl_agent(model, steps=1000)
        strategies['RL Agent'].append(result)
        action_dist_str = ", ".join([f"P{p}:{pct:.0f}%" for p, pct in result['action_dist'].items()])
        print(f"  RL Agent:       Reward={result['total_reward']:8.1f}, Wait={result['avg_waiting']:.2f}, Changes={result['phase_changes']}, Actions=[{action_dist_str}]")
        
        result = evaluator.evaluate_random(steps=1000)
        strategies['Random'].append(result)
        print(f"  Random:         Reward={result['total_reward']:8.1f}, Wait={result['avg_waiting']:.2f}, Changes={result['phase_changes']}")
        
        result = evaluator.evaluate_fixed_time(steps=1000)
        strategies['Fixed-Time (30s)'].append(result)
        print(f"  Fixed-Time:     Reward={result['total_reward']:8.1f}, Wait={result['avg_waiting']:.2f}, Changes={result['phase_changes']}")
        
        result = evaluator.evaluate_green_wave(steps=1000)
        strategies['Green Wave'].append(result)
        print(f"  Green Wave:     Reward={result['total_reward']:8.1f}, Wait={result['avg_waiting']:.2f}, Changes={result['phase_changes']}")
        
        result = evaluator.evaluate_actuated(steps=1000)
        strategies['Actuated'].append(result)
        print(f"  Actuated:       Reward={result['total_reward']:8.1f}, Wait={result['avg_waiting']:.2f}, Changes={result['phase_changes']}")
        
        print()
    
    print("="*85)
    print("FINAL RESULTS (Average over all runs)")
    print("="*85 + "\n")
    
    print(f"{'Strategy':<20} | {'Avg Reward':<12} | {'Avg Wait':<12} | {'Max Wait':<12} | {'Phase Changes'}")
    print("-"*85)
    
    stats = {}
    for name, results in strategies.items():
        avg_reward = np.mean([r['total_reward'] for r in results])
        std_reward = np.std([r['total_reward'] for r in results])
        avg_wait = np.mean([r['avg_waiting'] for r in results])
        std_wait = np.std([r['avg_waiting'] for r in results])
        avg_max_wait = np.mean([r['max_waiting'] for r in results])
        avg_changes = np.mean([r['phase_changes'] for r in results])
        
        stats[name] = {
            'reward': avg_reward,
            'reward_std': std_reward,
            'wait': avg_wait,
            'wait_std': std_wait,
            'max_wait': avg_max_wait,
            'changes': avg_changes
        }
        
        print(f"{name:<20} | {avg_reward:>6.1f}±{std_reward:<4.1f} | "
              f"{avg_wait:>5.2f}±{std_wait:<4.2f} | "
              f"{avg_max_wait:>6.1f}      | {avg_changes:>5.0f}")
    
    print("\n" + "="*85)
    print("RL AGENT PERFORMANCE COMPARISON")
    print("="*85 + "\n")
    
    rl_reward = stats['RL Agent']['reward']
    rl_wait = stats['RL Agent']['wait']
    
    baseline_names = [n for n in strategies.keys() if n != 'RL Agent']
    best_baseline_reward = max([stats[n]['reward'] for n in baseline_names])
    best_baseline_wait = min([stats[n]['wait'] for n in baseline_names])
    best_baseline_name = [n for n in baseline_names if stats[n]['reward'] == best_baseline_reward][0]
    
    reward_improvement = ((rl_reward - best_baseline_reward) / abs(best_baseline_reward)) * 100
    wait_improvement = ((best_baseline_wait - rl_wait) / best_baseline_wait) * 100
    
    print(f"Best Baseline: {best_baseline_name}")
    print(f"  Reward: {best_baseline_reward:.1f}")
    print(f"  Avg Wait: {best_baseline_wait:.2f}s\n")
    
    print(f"RL Agent vs Best Baseline:")
    print(f"  Reward improvement: {reward_improvement:+.1f}%")
    print(f"  Wait time improvement: {wait_improvement:+.1f}%\n")
    
    if reward_improvement > 10:
        print("🌟 EXCELLENT: RL Agent significantly outperforms all baselines!")
    elif reward_improvement > 5:
        print("✅ GOOD: RL Agent beats baselines")
    elif reward_improvement > 0:
        print("⚠️  OK: RL Agent slightly better, more training may help")
    else:
        print("❌ POOR: Baselines are stronger. Consider:")
        print("   • Training longer (more timesteps)")
        print("   • Adjusting reward function")
        print("   • Trying different algorithm (DQN vs PPO)")
    
    print("\n" + "="*85 + "\n")
    
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python evaluate_final.py [model_path] [config_file] [num_runs]")
        print("\nExample:")
        print("  python evaluate_final.py models/ppo_final_test.zip config.json 10")
        print("\nAvailable models:")
        if os.path.exists("models"):
            for f in sorted(os.listdir("models")):
                if f.endswith(".zip"):
                    print(f"  - models/{f}")
        sys.exit(0)
    
    model_path = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else "config.json"
    num_runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    
    evaluate_all(model_path, config_file, num_runs)