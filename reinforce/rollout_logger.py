import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any

class RolloutLogger:
    def __init__(self, config):
        self.rollouts_dir = "rollouts"
        self._clear_rollouts_dir()
        self.config = config
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _clear_rollouts_dir(self):
        """Clear the rollouts directory and recreate it."""
        if os.path.exists(self.rollouts_dir):
            shutil.rmtree(self.rollouts_dir)
        os.makedirs(self.rollouts_dir, exist_ok=True)
        
    def log_rollout(self, episode: int, prompts: List[str], targets: List[str], 
                   thinking_contents: List[str], contents: List[str], 
                   rewards: List[float], loss: float, kl_penalty_mean: float,
                   thinking_penalties: List[float] = None):
        """Log a single rollout to JSON file."""
        
        # Helper function to convert tensors to serializable types
        def make_serializable(obj):
            if hasattr(obj, 'item'):  # PyTorch tensor
                return obj.item()
            elif hasattr(obj, 'tolist'):  # PyTorch tensor or numpy array
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        # Convert config to serializable format
        config_dict = {}
        for k, v in self.config.items():
            config_dict[k] = make_serializable(v)
        
        rollout_data = {
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            "config": config_dict,
            "rollout": {
                "prompts": prompts,
                "targets": targets,
                "thinking_contents": thinking_contents,
                "contents": contents,
                "rewards": rewards,
                "metrics": {
                    "loss": loss,
                    "reward_mean": sum(rewards) / len(rewards),
                    "kl_penalty_mean": kl_penalty_mean,
                    "reward_std": (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5
                }
            }
        }
        
        # Add thinking penalties if provided
        if thinking_penalties is not None:
            rollout_data["rollout"]["thinking_penalties"] = thinking_penalties
            rollout_data["rollout"]["metrics"]["thinking_penalty_mean"] = sum(thinking_penalties) / len(thinking_penalties) if thinking_penalties else 0.0
        
        filename = f"rollout_episode_{episode:06d}_{self.run_id}.json"
        filepath = os.path.join(self.rollouts_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(rollout_data, f, indent=2)
            
    def get_latest_rollouts(self, num_rollouts: int = 10) -> List[Dict[str, Any]]:
        """Get the latest N rollouts for analysis."""
        if not os.path.exists(self.rollouts_dir):
            return []
            
        files = [f for f in os.listdir(self.rollouts_dir) if f.endswith('.json')]
        files.sort(reverse=True)
        
        rollouts = []
        for filename in files[:num_rollouts]:
            filepath = os.path.join(self.rollouts_dir, filename)
            with open(filepath, 'r') as f:
                rollouts.append(json.load(f))
                
        return rollouts 