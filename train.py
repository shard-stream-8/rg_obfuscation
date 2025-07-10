import argparse
import sys
from reinforce.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with reinforcement learning using a config YAML file")
    parser.add_argument("config", help="Path to the config YAML file")
    
    args = parser.parse_args()
    
    try:
        train(args.config)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1) 