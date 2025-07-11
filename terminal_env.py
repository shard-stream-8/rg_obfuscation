import os
import subprocess
import tempfile
import shutil
import json
from typing import Dict, Any, Optional

class TerminalEnvironment:
    """
    Terminal environment wrapper that provides an isolated directory with verifier.py
    and handles command execution for the model.
    """
    
    def __init__(self, verifier_code: str, ground_truth: Any, max_turns: int = 10):
        """
        Initialize the terminal environment.
        
        Args:
            verifier_code: The Python code for verifier.py
            ground_truth: The ground truth data for this episode
            max_turns: Maximum number of turns allowed per episode
        """
        self.verifier_code = verifier_code
        self.ground_truth = ground_truth
        self.max_turns = max_turns
        self.work_dir = None
        self.terminal_output = []
        self.episode_complete = False
        self.final_reward = 0.0
        self.turn_count = 0
        self._setup_environment()
    
    def _setup_environment(self):
        """Create the isolated working directory with verifier.py."""
        # Create temporary directory
        self.work_dir = tempfile.mkdtemp(prefix="terminal_env_")
        
        # Write verifier.py with ground truth embedded
        verifier_path = os.path.join(self.work_dir, "verifier.py")
        
        # Create verifier that includes the ground truth
        ground_truth_str = json.dumps(self.ground_truth)
        verifier_with_ground_truth = f"""
import json
import sys

# Ground truth for this episode
GROUND_TRUTH = {ground_truth_str}

{self.verifier_code}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verifier.py <answer>")
        sys.exit(1)
    
    answer = sys.argv[1]
    result = verifier(answer, {{"answer": GROUND_TRUTH}})
    print(result)
"""
        
        with open(verifier_path, 'w') as f:
            f.write(verifier_with_ground_truth)
    
    def execute_command(self, command: str) -> str:
        """
        Execute a command in the terminal environment.
        
        Args:
            command: The command to execute
            
        Returns:
            The command output as a string
        """
        try:
            # Change to the working directory
            original_cwd = os.getcwd()
            os.chdir(self.work_dir)
            
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            # Capture output
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR: {result.stderr}"
            
            if result.returncode != 0:
                output += f"\nCommand failed with return code: {result.returncode}"
            
            # Add to terminal output history
            self.terminal_output.append(f"$ {command}")
            self.terminal_output.append(output)
            
            return output
            
        except subprocess.TimeoutExpired:
            error_msg = "Command timed out after 30 seconds"
            self.terminal_output.append(f"$ {command}")
            self.terminal_output.append(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Command execution failed: {str(e)}"
            self.terminal_output.append(f"$ {command}")
            self.terminal_output.append(error_msg)
            return error_msg
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def execute_command_with_verification(self, command: str) -> Dict[str, Any]:
        """
        Execute command and check if it's a verifier command.
        
        Args:
            command: The command to execute
            
        Returns:
            Dict with output, is_verifier, reward, and completion status
        """
        output = self.execute_command(command)
        self.turn_count += 1
        
        result = {
            'output': output,
            'is_verifier': command.strip().startswith('python verifier.py'),
            'reward': 0.0,
            'episode_complete': False,
            'turn_count': self.turn_count,
            'reason': None
        }
        
        # Check if this was a verifier command
        if result['is_verifier']:
            try:
                reward = float(output.strip())
                result['reward'] = reward
                
                # If verifier returned positive, mark episode complete
                if reward > 0.0:
                    self.episode_complete = True
                    self.final_reward = reward
                    result['episode_complete'] = True
                    result['reason'] = 'verifier_positive'
            except (ValueError, TypeError):
                result['reason'] = 'verifier_error'
        
        # Check turn limit
        if self.turn_count >= self.max_turns:
            result['episode_complete'] = True
            result['reason'] = 'turn_limit'
        
        return result
    
    def is_episode_complete(self) -> bool:
        """Check if the episode is complete (verifier returned positive or turn limit reached)."""
        return self.episode_complete or self.turn_count >= self.max_turns
    
    def get_final_reward(self) -> float:
        """Get the final reward for the episode."""
        return self.final_reward
    
    def get_current_context(self) -> str:
        """Get current terminal context for next turn."""
        return self.get_terminal_context()
    
    def mark_episode_complete(self, reward: float):
        """Mark episode as complete with final reward."""
        self.episode_complete = True
        self.final_reward = reward
    
    def get_terminal_context(self) -> str:
        """
        Get the current terminal context (all previous commands and outputs).
        
        Returns:
            String containing all terminal history
        """
        return "\n".join(self.terminal_output)
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the episode for logging.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'turn_count': self.turn_count,
            'episode_complete': self.is_episode_complete(),
            'final_reward': self.final_reward,
            'terminal_context': self.get_terminal_context(),
            'max_turns': self.max_turns
        }
    
    def cleanup(self):
        """Clean up the temporary directory."""
        if self.work_dir and os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 