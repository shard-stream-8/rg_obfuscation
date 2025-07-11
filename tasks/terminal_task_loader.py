import importlib
import importlib.util
import sys
import os
from typing import Any

def load_terminal_task(task_name: str, config=None):
    """
    Load a terminal-based task.
    
    Args:
        task_name: Name of the terminal task (e.g., 'acre_terminal')
        config: Optional configuration object
        
    Returns:
        Terminal task instance
    """
    # Map of terminal task names to their implementations
    terminal_tasks = {
        'acre_terminal': 'tasks.acre_terminal_task.AcreTerminalTask',
    }
    
    if task_name in terminal_tasks:
        # Import the task class
        module_path, class_name = terminal_tasks[task_name].rsplit('.', 1)
        
        try:
            module = importlib.import_module(module_path)
            task_class = getattr(module, class_name)
            return task_class(config)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to load terminal task '{task_name}': {str(e)}")
    else:
        # Try to load from a custom file
        custom_task_path = f"tasks/{task_name}_task.py"
        if os.path.exists(custom_task_path):
            try:
                spec = importlib.util.spec_from_file_location(task_name, custom_task_path)
                custom_task_module = importlib.util.module_from_spec(spec)
                sys.modules[task_name] = custom_task_module
                spec.loader.exec_module(custom_task_module)
                
                # Look for a class with the same name as the task
                task_class_name = ''.join(word.capitalize() for word in task_name.split('_'))
                if hasattr(custom_task_module, task_class_name):
                    task_class = getattr(custom_task_module, task_class_name)
                    return task_class(config)
                else:
                    raise ValueError(f"No task class '{task_class_name}' found in {custom_task_path}")
            except Exception as e:
                raise ValueError(f"Failed to load custom terminal task '{task_name}': {str(e)}")
        else:
            available_tasks = list(terminal_tasks.keys())
            raise ValueError(
                f"Terminal task '{task_name}' not found. Available terminal tasks: {available_tasks}"
            ) 