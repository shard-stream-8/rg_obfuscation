import os
import importlib.util
from typing import Dict, Optional, Callable, Union

# Import BasePrompt using absolute import
try:
    from prompts.base_prompt import BasePrompt
except ImportError:
    # Fallback for when running from within the prompts directory
    try:
        from base_prompt import BasePrompt
    except ImportError:
        # If we can't import BasePrompt, we'll use a generic type
        BasePrompt = type('BasePrompt', (), {})

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
    
    def get_prompt(self, task_name: str, use_terminal: bool = False) -> Optional[Union[BasePrompt, Callable]]:
        """
        Get a prompt for a specific task.
        
        Args:
            task_name: Name of the task
            use_terminal: Whether to use terminal mode
            
        Returns:
            Prompt instance/callable or None if not found
        """
        # Check for exact match first
        if task_name in self._prompts:
            prompt = self._prompts[task_name]
            
            # If it's a BasePrompt instance, configure it with terminal settings
            if isinstance(prompt, BasePrompt):
                prompt.use_terminal = use_terminal
                return prompt
            elif callable(prompt):
                # For callable prompts, we'll pass the terminal settings via metadata
                return prompt
            
        # Check for task name variations (e.g., acre_terminal -> acre)
        if task_name.endswith('_terminal'):
            base_task = task_name.replace('_terminal', '')
            if base_task in self._prompts:
                prompt = self._prompts[base_task]
                if isinstance(prompt, BasePrompt):
                    prompt.use_terminal = True  # Force terminal mode for _terminal tasks
                    return prompt
                elif callable(prompt):
                    return prompt
        
        return None
    
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
                        hasattr(attr, '__bases__') and
                        any('BasePrompt' in str(base) for base in attr.__bases__)):
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
            
            # Try to load the module
            spec = importlib.util.spec_from_file_location("custom_prompt", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Prefer to register a class that inherits from BasePrompt
            prompt_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    hasattr(attr, '__bases__') and
                    any('BasePrompt' in str(base) for base in attr.__bases__)):
                    prompt_class = attr
                    break
            if prompt_class is not None:
                registry.register(task_name, prompt_class())
            elif hasattr(module, 'prompt'):
                registry.register(task_name, module.prompt)

# Auto-register on import
_auto_register_prompts() 