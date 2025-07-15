import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Any, Literal

class ReadableJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that preserves newlines and formatting for better readability."""
    
    def encode(self, obj):
        if isinstance(obj, str):
            # Preserve newlines and formatting in strings
            return super().encode(obj)
        return super().encode(obj)

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
        
    def _format_dialogue_for_readability(self, conversation_dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Format dialogue content to be more readable by preserving formatting."""
        formatted_dialogue = []
        for message in conversation_dialogue:
            content = message["content"]
            
            # Ensure proper line breaks and formatting
            # This will make the JSON output more readable when viewed
            formatted_content = content
            
            formatted_message = {
                "role": message["role"],
                "content": formatted_content
            }
            formatted_dialogue.append(formatted_message)
        
        return formatted_dialogue
        
    def log_rollout(self, episode: int, prompts: List[str], targets: List[str], 
                   thinking_contents: List[str], contents: List[str], 
                   rewards: List[float], loss: float, kl_penalty_mean: float,
                   thinking_penalties: List[float] = None,
                   output_word_penalties: List[Dict[str, float]] = None,
                   thinking_word_penalties: List[Dict[str, float]] = None,
                   output_word_counts: List[Dict[str, int]] = None,
                   thinking_word_counts: List[Dict[str, int]] = None,
                   # Multi-turn specific parameters
                   turn_count: int = None,
                   episode_complete: bool = None,
                   final_reward: float = None,
                   commands: List[str] = None,
                   command_outputs: List[str] = None,
                   terminal_context: str = None,
                   episode_rewards: List[float] = None,
                   conversation_dialogue: List[Dict[str, str]] = None,
                   format: Literal["json", "readable", "super_readable"] = "json"):
        """Log a single rollout with specified format (json, readable, or super_readable)."""
        
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
        
        # Calculate metrics
        reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
        reward_std = (sum((r - reward_mean)**2 for r in rewards) / len(rewards))**0.5 if len(rewards) > 1 else 0.0
        thinking_penalty_mean = sum(thinking_penalties) / len(thinking_penalties) if thinking_penalties else 0.0
        
        if format == "super_readable":
            self._log_rollout_super_readable(
                episode, rewards, loss, reward_mean, reward_std, kl_penalty_mean, 
                thinking_penalty_mean, turn_count, episode_complete, final_reward, 
                conversation_dialogue
            )
        else:
            # Build simplified rollout data focusing on essential information
            rollout_data = {
                "episode": episode,
                "timestamp": datetime.now().isoformat(),
                "rewards": rewards,
                "metrics": {
                    "loss": loss,
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "kl_penalty_mean": kl_penalty_mean,
                    "thinking_penalty_mean": thinking_penalty_mean
                }
            }
            
            # Add conversation dialogue if provided (this is the main focus)
            if conversation_dialogue is not None:
                if format == "readable":
                    # Format dialogue for maximum readability with turn numbers
                    formatted_dialogue = []
                    for i, message in enumerate(conversation_dialogue):
                        content = message["content"]
                        
                        # Create a more readable format for the dialogue
                        formatted_message = {
                            "turn": i + 1,
                            "role": message["role"],
                            "content": content  # Keep original formatting
                        }
                        formatted_dialogue.append(formatted_message)
                else:
                    # Format dialogue for better readability
                    formatted_dialogue = self._format_dialogue_for_readability(conversation_dialogue)
                rollout_data["conversation_dialogue"] = formatted_dialogue
            
            # Add multi-turn specific metrics if available
            if turn_count is not None:
                rollout_data["multi_turn_metrics"] = {
                    "turn_count": turn_count,
                    "episode_complete": episode_complete,
                    "final_reward": final_reward
                }
            
            # Determine filename based on format
            if format == "readable":
                filename = f"rollout_episode_{episode:06d}_{self.run_id}_readable.json"
            else:
                filename = f"rollout_episode_{episode:06d}_{self.run_id}.json"
            
            filepath = os.path.join(self.rollouts_dir, filename)
            
            # Write with better formatting for readability
            with open(filepath, 'w') as f:
                json.dump(rollout_data, f, indent=2, ensure_ascii=False, separators=(',', ': '))
    
    def _log_rollout_super_readable(self, episode: int, rewards: List[float], loss: float,
                                   reward_mean: float, reward_std: float, kl_penalty_mean: float,
                                   thinking_penalty_mean: float, turn_count: int = None,
                                   episode_complete: bool = None, final_reward: float = None,
                                   conversation_dialogue: List[Dict[str, str]] = None):
        """Log a single rollout in a super readable text format that preserves all formatting."""
        
        filename = f"rollout_episode_{episode:06d}_{self.run_id}_super_readable.txt"
        filepath = os.path.join(self.rollouts_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write header with episode info and metrics
            f.write("=" * 80 + "\n")
            f.write(f"EPISODE {episode} - {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write metrics
            f.write("📊 METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Rewards: {rewards}\n")
            f.write(f"Reward Mean: {reward_mean:.4f}\n")
            f.write(f"Reward Std: {reward_std:.4f}\n")
            f.write(f"Loss: {loss:.4f}\n")
            f.write(f"KL Penalty Mean: {kl_penalty_mean:.4f}\n")
            f.write(f"Thinking Penalty Mean: {thinking_penalty_mean:.4f}\n")
            
            if turn_count is not None:
                f.write(f"Turn Count: {turn_count}\n")
                f.write(f"Episode Complete: {episode_complete}\n")
                f.write(f"Final Reward: {final_reward}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("💬 CONVERSATION DIALOGUE:\n")
            f.write("=" * 80 + "\n\n")
            
            # Write conversation dialogue with proper formatting
            if conversation_dialogue:
                for i, message in enumerate(conversation_dialogue):
                    role = message["role"].upper()
                    content = message["content"]
                    
                    f.write(f"TURN {i+1} - {role}:\n")
                    f.write("-" * 60 + "\n")
                    f.write(content)
                    f.write("\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF EPISODE\n")
            f.write("=" * 80 + "\n")
            
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