import os
import subprocess
import tempfile
import shutil
import json
from typing import Dict, Any, Optional
from prompts.terminal_prompts import (
    get_verifier_incorrect_message
)

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
                
                # Add explicit interpretation of verifier result
                if reward == 1.0:
                    result['output'] = f"{output.strip()}\n\n✅ SUCCESS: Your answer is CORRECT! The verifier returned 1.0, which means your solution is valid."
                    self.episode_complete = True
                    self.final_reward = reward
                    result['episode_complete'] = True
                    result['reason'] = 'verifier_positive'
                elif reward == 0.0:
                    result['output'] = f"{output.strip()}\n\n❌ INCORRECT: Your answer is WRONG. The verifier returned 0.0, which means your solution is not valid. Please try a different approach."
                    result['reason'] = 'verifier_negative'
                else:
                    result['output'] = f"{output.strip()}\n\n⚠️ UNEXPECTED: The verifier returned {reward}, which is neither 0.0 nor 1.0. This might indicate an error."
                    result['reason'] = 'verifier_unexpected'
            except (ValueError, TypeError):
                result['output'] = f"{output.strip()}\n\n⚠️ ERROR: The verifier output '{output.strip()}' could not be parsed as a number. Please check your verifier command."
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
    
    def get_context(self) -> str:
        """
        Get the current terminal context (all previous commands and outputs) with enhanced formatting and interpretation.
        This is the only context method and always returns the best-formatted context.
        """
        if not self.terminal_output:
            return "No commands executed yet."
        formatted_output = []
        for i in range(0, len(self.terminal_output), 2):
            if i + 1 < len(self.terminal_output):
                command = self.terminal_output[i]
                output = self.terminal_output[i + 1]
                # Add interpretation for verifier commands
                if command.strip().startswith('$ python verifier.py'):
                    try:
                        reward = float(output.strip())
                        if reward == 1.0:
                            interpretation = "✅ CORRECT - Your answer is valid!"
                        elif reward == 0.0:
                            interpretation = "❌ INCORRECT - Your answer is wrong, try again"
                        else:
                            interpretation = f"⚠️ UNEXPECTED - Verifier returned {reward}"
                    except (ValueError, TypeError):
                        interpretation = f"⚠️ ERROR - Could not parse verifier output: {output.strip()}"
                    formatted_output.append(f"{command}\n{output}\n{interpretation}")
                else:
                    formatted_output.append(f"{command}\n{output}")
            else:
                # Handle case where we have a command but no output
                formatted_output.append(self.terminal_output[i])
        return "\n\n".join(formatted_output)
    
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
            'terminal_context': self.get_context(),
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