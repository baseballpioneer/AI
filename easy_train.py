"""
Easy-to-use training script for CityFlow RL
Run this after setting up the professional environment
"""

import sys
import os
from train_professional import train_with_stable_baselines3, evaluate_model, CityFlowGymEnv
from train_config import TRAINING_CONFIGS, ALGORITHM_CONFIGS, print_config_info


def main():
    """Main training function with user-friendly interface"""
    
    print("\n" + "="*70)
    print("CITYFLOW RL TRAINING - PROFESSIONAL SETUP")
    print("="*70 + "\n")
    
    # Check if config files exist
    if not os.path.exists("config.json"):
        print("❌ Error: config.json not found!")
        print("Make sure you're in the correct directory with CityFlow config files.")
        sys.exit(1)
    
    # Show available configs
    if len(sys.argv) == 1:
        print("Usage: python easy_train.py [mode] [algorithm]")
        print("\nExamples:")
        print("  python easy_train.py test DQN      # Quick test")
        print("  python easy_train.py dev PPO       # Development")
        print("  python easy_train.py medium PPO    # Research paper")
        print("")
        print_config_info()
        
        # Interactive mode
        print("\n" + "="*70)
        print("INTERACTIVE MODE")
        print("="*70 + "\n")
        
        print("Available modes:")
        for i, (name, config) in enumerate(TRAINING_CONFIGS.items(), 1):
            print(f"  {i}. {name:12} - {config['description']}")
        
        mode_choice = input("\nSelect training mode (1-6) [default: 1]: ").strip() or "1"
        mode_names = list(TRAINING_CONFIGS.keys())
        training_mode = mode_names[int(mode_choice) - 1]
        
        print("\nAvailable algorithms:")
        for i, (name, config) in enumerate(ALGORITHM_CONFIGS.items(), 1):
            print(f"  {i}. {name:5} - {config['description']}")
        
        algo_choice = input("\nSelect algorithm (1-3) [default: 2 (PPO)]: ").strip() or "2"
        algo_names = list(ALGORITHM_CONFIGS.keys())
        algorithm = algo_names[int(algo_choice) - 1]
        
    else:
        # Command line mode
        training_mode = sys.argv[1] if len(sys.argv) > 1 else "test"
        algorithm = sys.argv[2] if len(sys.argv) > 2 else "PPO"
    
    # Validate inputs
    if training_mode not in TRAINING_CONFIGS:
        print(f"❌ Error: Unknown training mode '{training_mode}'")
        print(f"Available modes: {', '.join(TRAINING_CONFIGS.keys())}")
        sys.exit(1)
    
    if algorithm not in ALGORITHM_CONFIGS:
        print(f"❌ Error: Unknown algorithm '{algorithm}'")
        print(f"Available algorithms: {', '.join(ALGORITHM_CONFIGS.keys())}")
        sys.exit(1)
    
    # Get configuration
    train_config = TRAINING_CONFIGS[training_mode]
    algo_config = ALGORITHM_CONFIGS[algorithm]
    
    # Confirm with user
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Mode:           {training_mode}")
    print(f"Algorithm:      {algorithm}")
    print(f"Total Steps:    {train_config['total_timesteps']:,}")
    print(f"Episode Length: {train_config['steps_per_episode']} steps")
    print(f"Description:    {train_config['description']}")
    print(f"Estimated Time: ~{train_config['total_timesteps']/1000/60:.1f} hours")
    print("="*70 + "\n")
    
    response = input("Proceed with training? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("tensorboard", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("\n✓ Directories created")
    print("  - models/      (saved models)")
    print("  - tensorboard/ (training logs)")
    print("  - logs/        (evaluation logs)")
    
    # Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print("\nTips:")
    print("  - Training progress will be shown in terminal")
    print("  - Models are saved every 10,000 steps")
    print("  - Press Ctrl+C to stop training early (model will be saved)")
    print("  - View live training curves: tensorboard --logdir ./tensorboard/")
    print("\n" + "="*70 + "\n")
    
    try:
        model = train_with_stable_baselines3(
            config_file="config.json",
            algorithm=algorithm,
            total_timesteps=train_config['total_timesteps']
        )
        
        # Evaluate
        print("\n" + "="*70)
        print("EVALUATING TRAINED MODEL")
        print("="*70 + "\n")
        
        model_path = f"models/{algorithm.lower()}_final"
        evaluate_model(model_path, "config.json", num_episodes=10)
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\n✓ Model saved: {model_path}.zip")
        print(f"✓ Training logs: tensorboard/")
        print("\nNext steps:")
        print("  1. View training curves: tensorboard --logdir ./tensorboard/")
        print("  2. Evaluate more: python evaluate.py")
        print("  3. Compare algorithms: train with different algorithms")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("TRAINING INTERRUPTED")
        print("="*70)
        print("\nModel has been saved at last checkpoint.")
        print("You can resume training or evaluate the partially trained model.")
    
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Check dependencies
    try:
        import gymnasium
        import stable_baselines3
        import torch
        print("✓ All dependencies installed\n")
    except ImportError as e:
        print("\n❌ Missing dependencies!")
        print("\nPlease install:")
        print("  pip install gymnasium stable-baselines3[extra] tensorboard torch")
        print(f"\nError: {e}")
        sys.exit(1)
    
    main()