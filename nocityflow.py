"""
CityFlow RL Traffic Signal Control - Corrected for Real CityFlow API
"""

import cityflow
import numpy as np
import json
from collections import deque
import random
import os


class TrafficEnvironment:
    """CityFlow RL Environment using correct API"""
    
    def __init__(self, config_file, thread_num=1):
        self.eng = cityflow.Engine(config_file, thread_num=thread_num)
        
        # Load config to get intersection ID
        with open(config_file, 'r') as f:
            config = json.load(f)
        with open(config['roadnetFile'], 'r') as f:
            roadnet = json.load(f)
        
        # Find the non-virtual intersection
        self.intersection_id = None
        for intersection in roadnet['intersections']:
            if not intersection.get('virtual', False):
                self.intersection_id = intersection['id']
                # Get number of phases from traffic light config
                if 'trafficLight' in intersection:
                    self.num_phases = len(intersection['trafficLight']['lightphases'])
                break
        
        if self.intersection_id is None:
            raise ValueError("No non-virtual intersection found in roadnet")
        
        print(f"Controlling intersection: {self.intersection_id}")
        print(f"Number of phases: {self.num_phases}")
        
        # Initialize by running a few steps to populate lanes
        for _ in range(10):
            self.eng.next_step()
        
        # Get lanes from the engine
        lane_vehicles = self.eng.get_lane_vehicles()
        self.lanes = sorted([lane for lane in lane_vehicles.keys() if '_' in lane])
        self.num_lanes = len(self.lanes)
        
        print(f"Number of lanes: {self.num_lanes}")
        print(f"Sample lanes: {self.lanes[:3]}")
        
        # State and action spaces
        self.state_size = self.num_lanes * 2  # vehicles + waiting per lane
        self.action_size = self.num_phases
        
        # Episode tracking
        self.current_phase = 0
        self.phase_time = 0
        self.min_phase_time = 10  # Minimum green time
        self.yellow_time = 3  # Yellow light duration
        
        # Metrics
        self.total_wait_time = 0
        self.total_vehicles_processed = 0
        self.step_count = 0
        
    def reset(self):
        """Reset environment"""
        self.eng.reset()
        self.current_phase = 0
        self.phase_time = 0
        self.total_wait_time = 0
        self.total_vehicles_processed = 0
        self.step_count = 0
        
        # Set initial phase
        self.eng.set_tl_phase(self.intersection_id, self.current_phase)
        
        # Run a few steps to populate
        for _ in range(10):
            self.eng.next_step()
        
        return self._get_state()
    
    def _get_state(self):
        """Get current traffic state"""
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        
        state = []
        for lane in self.lanes:
            num_vehicles = len(lane_vehicles.get(lane, []))
            num_waiting = lane_waiting.get(lane, 0)
            state.extend([num_vehicles, num_waiting])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Execute action and return observation"""
        # Only change phase if different and minimum time elapsed
        if action != self.current_phase and self.phase_time >= self.min_phase_time:
            # Yellow phase transition (simplified - just wait a few steps)
            for _ in range(self.yellow_time):
                self.eng.next_step()
                self.step_count += 1
            
            # Change to new phase
            self.current_phase = action
            self.eng.set_tl_phase(self.intersection_id, self.current_phase)
            self.phase_time = 0
        
        # Execute one simulation step
        self.eng.next_step()
        self.phase_time += 1
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get next state
        next_state = self._get_state()
        
        # Episode termination (can be customized)
        done = False
        
        info = {
            'current_phase': self.current_phase,
            'phase_time': self.phase_time
        }
        
        return next_state, reward, done, info
    
    def _calculate_reward(self):
        """Calculate reward based on traffic metrics"""
        lane_waiting = self.eng.get_lane_waiting_vehicle_count()
        lane_vehicles = self.eng.get_lane_vehicles()
        
        # Primary objective: minimize waiting vehicles
        total_waiting = sum(lane_waiting.values())
        
        # Secondary: minimize queue length
        total_queue = sum(len(vehicles) for vehicles in lane_vehicles.values())
        
        # Reward function (negative = penalty)
        reward = -(total_waiting * 2.0 + total_queue * 0.5)
        
        # Update metrics
        self.total_wait_time += total_waiting
        self.total_vehicles_processed += total_queue
        
        return reward
    
    def get_metrics(self):
        """Return episode metrics"""
        return {
            'total_wait_time': self.total_wait_time,
            'avg_wait_time': self.total_wait_time / max(self.step_count, 1),
            'total_vehicles': self.total_vehicles_processed,
            'avg_queue': self.total_vehicles_processed / max(self.step_count, 1),
            'steps': self.step_count
        }
    
    def get_current_vehicles(self):
        """Get current vehicle count"""
        return self.eng.get_vehicle_count()


class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.batch_size = 32
        
        # Simple neural network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()
        
        print(f"Agent initialized: state_size={state_size}, action_size={action_size}")
    
    def _build_network(self):
        """Build Q-network"""
        return {
            'w1': np.random.randn(self.state_size, 64) * 0.1,
            'b1': np.zeros(64),
            'w2': np.random.randn(64, 64) * 0.1,
            'b2': np.zeros(64),
            'w3': np.random.randn(64, self.action_size) * 0.1,
            'b3': np.zeros(self.action_size)
        }
    
    def update_target_network(self):
        """Copy Q-network weights to target network"""
        self.target_network = {k: v.copy() for k, v in self.q_network.items()}
    
    def _forward(self, state, network):
        """Forward pass"""
        h1 = np.maximum(0, np.dot(state, network['w1']) + network['b1'])
        h2 = np.maximum(0, np.dot(h1, network['w2']) + network['b2'])
        q_values = np.dot(h2, network['w3']) + network['b3']
        return q_values
    
    def get_action(self, state):
        """Select action using epsilon-greedy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self._forward(state.reshape(1, -1), self.q_network)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        minibatch = random.sample(self.memory, self.batch_size)
        total_loss = 0.0
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q = self._forward(next_state.reshape(1, -1), self.target_network)
                target = reward + self.gamma * np.max(next_q[0])
            
            # Get current Q-values
            current_q = self._forward(state.reshape(1, -1), self.q_network)
            target_q = current_q.copy()
            target_q[0][action] = target
            
            # Calculate loss
            loss = np.mean((current_q - target_q) ** 2)
            total_loss += loss
            
            # Update network (simplified gradient descent)
            self._backward(state, target_q)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return total_loss / self.batch_size
    
    def _backward(self, state, target_q):
        """Simplified backpropagation"""
        lr = self.learning_rate
        
        # Forward pass
        h1 = np.maximum(0, np.dot(state.reshape(1, -1), self.q_network['w1']) + self.q_network['b1'])
        h2 = np.maximum(0, np.dot(h1, self.q_network['w2']) + self.q_network['b2'])
        output = np.dot(h2, self.q_network['w3']) + self.q_network['b3']
        
        # Gradients
        d_output = 2 * (output - target_q) / self.batch_size
        
        # Update layer 3
        self.q_network['w3'] -= lr * np.dot(h2.T, d_output)
        self.q_network['b3'] -= lr * np.sum(d_output, axis=0)
        
        # Layer 2
        d_h2 = np.dot(d_output, self.q_network['w3'].T)
        d_h2[h2 <= 0] = 0
        
        self.q_network['w2'] -= lr * np.dot(h1.T, d_h2)
        self.q_network['b2'] -= lr * np.sum(d_h2, axis=0)
        
        # Layer 1
        d_h1 = np.dot(d_h2, self.q_network['w2'].T)
        d_h1[h1 <= 0] = 0
        
        self.q_network['w1'] -= lr * np.dot(state.reshape(-1, 1), d_h1)
        self.q_network['b1'] -= lr * np.sum(d_h1, axis=0)
    
    def save(self, filename):
        """Save model"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        np.savez(filename, **self.q_network)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load model"""
        data = np.load(filename)
        self.q_network = {k: data[k] for k in data.files}
        print(f"Model loaded from {filename}")


def train_agent(config_file, episodes=100, steps_per_episode=3600):
    """Training loop"""
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize environment and agent
    env = TrafficEnvironment(config_file)
    agent = DQNAgent(env.state_size, env.action_size)
    
    # Training metrics
    episode_rewards = []
    episode_wait_times = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        
        for step in range(steps_per_episode):
            # Select and perform action
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store and train
            agent.remember(state, action, reward, next_state, done)
            loss = agent.replay()
            if loss > 0:
                episode_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update target network periodically
        if (episode + 1) % 10 == 0:
            agent.update_target_network()
        
        # Log metrics
        metrics = env.get_metrics()
        episode_rewards.append(total_reward)
        episode_wait_times.append(metrics['avg_wait_time'])
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_wait = np.mean(episode_wait_times[-10:])
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward (10 ep): {avg_reward:.2f}")
            print(f"  Avg Wait Time: {avg_wait:.2f}s")
            print(f"  Current Vehicles: {env.get_current_vehicles()}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Loss: {avg_loss:.6f}")
            print("-" * 60)
        
        # Save checkpoints
        if (episode + 1) % 50 == 0:
            agent.save(f"models/dqn_ep{episode + 1}.npz")
    
    # Save final model
    agent.save("models/dqn_final.npz")
    
    # Save metrics
    metrics_data = {
        'episode_rewards': episode_rewards,
        'episode_wait_times': episode_wait_times
    }
    
    with open('logs/training_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Final avg wait time (last 10): {np.mean(episode_wait_times[-10:]):.2f}s")
    
    return agent, episode_rewards, episode_wait_times


def test_environment(config_file, num_steps=100):
    """Quick test of the environment"""
    
    print("\n" + "="*60)
    print("Testing Environment")
    print("="*60 + "\n")
    
    env = TrafficEnvironment(config_file)
    state = env.reset()
    
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state[:6]}...")  # Show first 6 values
    print(f"Action space size: {env.action_size}")
    print()
    
    total_reward = 0
    for step in range(num_steps):
        # Random action
        action = random.randint(0, env.action_size - 1)
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        
        if (step + 1) % 20 == 0:
            metrics = env.get_metrics()
            print(f"Step {step + 1}:")
            print(f"  Reward: {reward:.2f}")
            print(f"  Total Vehicles: {env.get_current_vehicles()}")
            print(f"  Avg Wait: {metrics['avg_wait_time']:.2f}s")
            print(f"  Phase: {info['current_phase']}, Time: {info['phase_time']}")
    
    print(f"\nTest completed! Total reward: {total_reward:.2f}")
    print("Environment is working correctly.\n")


if __name__ == "__main__":
    config_file = "config.json"
    
    # First test the environment
    print("Step 1: Testing environment...")
    test_environment(config_file, num_steps=100)
    
    # Then train
    print("\nStep 2: Starting training...")
    agent, rewards, wait_times = train_agent(
        config_file,
        episodes=100,
        steps_per_episode=1000  # Shorter episodes for faster testing
    )