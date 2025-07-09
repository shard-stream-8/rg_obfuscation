import os
import importlib.util
from typing import Dict, Type, Optional, Callable, Union

# Import BaseVerifier using absolute import
try:
    from verifiers.base_verifier import BaseVerifier
except ImportError:
    # Fallback for when running from within the verifiers directory
    from base_verifier import BaseVerifier

class VerifierRegistry:
    """
    Registry for managing custom verifiers.
    
    This class provides a centralized way to register and retrieve
    custom verifiers for different tasks.
    """
    
    def __init__(self):
        self._verifiers: Dict[str, Union[BaseVerifier, Callable]] = {}
    
    def register(self, task_name: str, verifier: Union[BaseVerifier, Callable]):
        """
        Register a verifier for a specific task.
        
        Args:
            task_name: Name of the task (e.g., 'leg_counting')
            verifier: The verifier instance or callable to register
        """
        self._verifiers[task_name] = verifier
    
    def get_verifier(self, task_name: str) -> Optional[Union[BaseVerifier, Callable]]:
        """
        Get a verifier for a specific task.
        
        Args:
            task_name: Name of the task
            
        Returns:
            Verifier instance/callable or None if not found
        """
        return self._verifiers.get(task_name)
    
    def load_verifier_from_file(self, file_path: str) -> Optional[Union[BaseVerifier, Callable]]:
        """
        Load a verifier from a Python file.
        
        Args:
            file_path: Path to the Python file containing the verifier
            
        Returns:
            Verifier class or callable or None if loading fails
        """
        try:
            # Load the module from file
            spec = importlib.util.spec_from_file_location("custom_verifier", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # First, look for any class that inherits from BaseVerifier (prioritize classes)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseVerifier) and 
                    attr != BaseVerifier):
                    return attr  # Return the class, not an instance
            
            # Then look for a verifier function or class
            if hasattr(module, 'verifier'):
                return module.verifier
            elif hasattr(module, 'Verifier'):
                return module.Verifier  # Return the class, not an instance
            
            return None
                
        except Exception as e:
            print(f"Error loading verifier from {file_path}: {e}")
            return None
    
    def list_available_verifiers(self) -> list:
        """
        List all available verifier task names.
        
        Returns:
            List of task names that have registered verifiers
        """
        return list(self._verifiers.keys())

# Global registry instance
registry = VerifierRegistry()

# Auto-register verifiers from the verifiers directory
def _auto_register_verifiers():
    """Automatically register verifiers from the verifiers directory."""
    verifiers_dir = os.path.dirname(__file__)
    
    for filename in os.listdir(verifiers_dir):
        if filename.endswith('_verifier.py') and filename != '__init__.py':
            task_name = filename.replace('_verifier.py', '')
            file_path = os.path.join(verifiers_dir, filename)
            
            # Try to load and register the verifier
            verifier = registry.load_verifier_from_file(file_path)
            if verifier is not None:
                registry.register(task_name, verifier)

# Auto-register on import
_auto_register_verifiers() 