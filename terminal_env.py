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
    
    def __init__(self, verifier_code: str, ground_truth: Any):
        """
        Initialize the terminal environment.
        
        Args:
            verifier_code: The Python code for verifier.py
            ground_truth: The ground truth data for this episode
        """
        self.verifier_code = verifier_code
        self.ground_truth = ground_truth
        self.work_dir = None
        self.terminal_output = []
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
    
    def get_terminal_context(self) -> str:
        """
        Get the current terminal context (all previous commands and outputs).
        
        Returns:
            String containing all terminal history
        """
        return "\n".join(self.terminal_output)
    
    def cleanup(self):
        """Clean up the temporary directory."""
        if self.work_dir and os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 