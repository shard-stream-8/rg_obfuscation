import argparse
import sys
from reinforce.trainer import train_multi_turn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with reinforcement learning using a config YAML file")
    parser.add_argument("config", help="Path to the config YAML file")
    
    args = parser.parse_args()
    
    try:
        train_multi_turn(args.config)
    except FileNotFoundError:
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)