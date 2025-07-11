import re
from typing import List, Dict, Any, Optional
from terminal_env import TerminalEnvironment

class TerminalTaskBase:
    """
    Base class for terminal-based tasks.
    Handles the interaction between model and terminal environment.
    """
    
    def __init__(self, task_name: str, verifier_code: str, config=None):
        """
        Initialize the terminal task.
        
        Args:
            task_name: Name of the task
            verifier_code: Python code for the verifier
            config: Optional configuration object
        """
        self.task_name = task_name
        self.verifier_code = verifier_code
        self.config = config
        self.terminal_env = None
    
    def create_terminal_environment(self, ground_truth: Any) -> TerminalEnvironment:
        """
        Create a new terminal environment for an episode.
        
        Args:
            ground_truth: The ground truth data for this episode
            
        Returns:
            TerminalEnvironment instance
        """
        return TerminalEnvironment(self.verifier_code, ground_truth)
    
    def extract_commands(self, content: str) -> List[str]:
        """
        Extract commands from model output using <command></command> tags.
        
        Args:
            content: The model's output content
            
        Returns:
            List of commands to execute
        """
        # Find all <command>...</command> tags
        command_pattern = r'<command>(.*?)</command>'
        commands = re.findall(command_pattern, content, re.DOTALL)
        
        # Clean up commands (remove extra whitespace)
        cleaned_commands = [cmd.strip() for cmd in commands if cmd.strip()]
        return cleaned_commands
    
    def execute_commands(self, commands: List[str]) -> str:
        """
        Execute a list of commands in the terminal environment.
        
        Args:
            commands: List of commands to execute
            
        Returns:
            Combined output from all commands
        """
        if not self.terminal_env:
            raise RuntimeError("Terminal environment not initialized")
        
        outputs = []
        for command in commands:
            output = self.terminal_env.execute_command(command)
            outputs.append(output)
        
        return "\n".join(outputs)
    
    def get_reward_from_verifier(self, answer: str) -> float:
        """
        Get reward by running the verifier with the given answer.
        
        Args:
            answer: The answer to verify
            
        Returns:
            Reward value (1.0 for correct, 0.0 for incorrect)
        """
        if not self.terminal_env:
            raise RuntimeError("Terminal environment not initialized")
        
        # Run the verifier command
        command = f'python verifier.py "{answer}"'
        output = self.terminal_env.execute_command(command)
        
        try:
            # Parse the output as a float
            reward = float(output.strip())
            return reward
        except (ValueError, TypeError):
            # If output can't be parsed as float, return 0.0
            return 0.0
    
    def process_model_output(self, content: str, ground_truth: Any) -> Dict[str, Any]:
        """
        Process model output and execute commands to get reward.
        
        Args:
            content: The model's output content
            ground_truth: The ground truth for this episode
            
        Returns:
            Dictionary containing reward and other metrics
        """
        # Create fresh terminal environment for this episode
        self.terminal_env = self.create_terminal_environment(ground_truth)
        
        try:
            # Extract commands from model output
            commands = self.extract_commands(content)
            
            # Execute commands and track outputs
            command_outputs = []
            if commands:
                for command in commands:
                    output = self.terminal_env.execute_command(command)
                    command_outputs.append(output)
                command_output = "\n".join(command_outputs)
            else:
                command_output = ""
            
            # Look for verifier command to get reward
            reward = 0.0
            verifier_used = False
            
            for i, command in enumerate(commands):
                if command.strip().startswith('python verifier.py'):
                    # Extract answer from verifier command
                    parts = command.split()
                    if len(parts) >= 3:
                        answer = parts[2].strip('"\'')
                        # Get the output from the executed command
                        if i < len(command_outputs):
                            try:
                                reward = float(command_outputs[i].strip())
                            except (ValueError, TypeError):
                                reward = 0.0
                        verifier_used = True
                        break
            
            # Get terminal context for logging
            terminal_context = self.terminal_env.get_terminal_context()
            
            return {
                'reward': reward,
                'verifier_used': verifier_used,
                'commands_executed': len(commands),
                'terminal_context': terminal_context,
                'command_output': command_output
            }
            
        finally:
            # Clean up terminal environment
            if self.terminal_env:
                self.terminal_env.cleanup()
                self.terminal_env = None
    
    def __len__(self):
        """Return the number of samples in the task."""
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, idx):
        """Get a sample from the task."""
        raise NotImplementedError("Subclasses must implement __getitem__") 