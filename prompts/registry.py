import os
import importlib.util
from typing import Dict, Optional, Callable, Union

# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback for when running from within the prompts directory
    from base_prompt import BasePrompt

class PromptRegistry:
    """
    Registry for managing custom prompts.
    
    This class provides a centralized way to register and retrieve
    custom prompts for different tasks.
    """
    
    def __init__(self):
        self._prompts: Dict[str, Union[BasePrompt, Callable]] = {}
    
    def register(self, task_name: str, prompt: Union[BasePrompt, Callable]):
        """
        Register a prompt for a specific task.
        
        Args:
            task_name: Name of the task (e.g., 'acre')
            prompt: The prompt instance or callable to register
        """
        self._prompts[task_name] = prompt
    
    def get_prompt(self, task_name: str) -> Optional[Union[BasePrompt, Callable]]:
        """
        Get a prompt for a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Prompt instance/callable or None if not found
        """
        return self._prompts.get(task_name)
    
    def load_prompt_from_file(self, file_path: str) -> Optional[Union[BasePrompt, Callable]]:
        """
        Load a prompt from a Python file.
        
        Args:
            file_path: Path to the Python file containing the prompt
            
        Returns:
            Prompt instance/callable or None if loading fails
        """
        try:
            # Load the module from file
            spec = importlib.util.spec_from_file_location("custom_prompt", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for a prompt function or class
            if hasattr(module, 'prompt'):
                return module.prompt
            elif hasattr(module, 'Prompt'):
                return module.Prompt()
            else:
                # Look for any class that inherits from BasePrompt
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BasePrompt) and 
                        attr != BasePrompt):
                        return attr()
                
                return None
                
        except Exception as e:
            print(f"Error loading prompt from {file_path}: {e}")
            return None
    
    def list_available_prompts(self) -> list:
        """
        List all available prompt task names.
        
        Returns:
            List of task names that have registered prompts
        """
        return list(self._prompts.keys())

# Global registry instance
registry = PromptRegistry()

# Auto-register prompts from the prompts directory
def _auto_register_prompts():
    """Automatically register prompts from the prompts directory."""
    prompts_dir = os.path.dirname(__file__)
    
    for filename in os.listdir(prompts_dir):
        if filename.endswith('_prompt.py') and filename != '__init__.py':
            task_name = filename.replace('_prompt.py', '')
            file_path = os.path.join(prompts_dir, filename)
            
            # Try to load and register the prompt
            prompt = registry.load_prompt_from_file(file_path)
            if prompt is not None:
                registry.register(task_name, prompt)

# Auto-register on import
_auto_register_prompts() 