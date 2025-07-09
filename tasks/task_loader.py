import importlib
import importlib.util
import sys
import os
from reasoning_gym.factory import create_dataset

def load_task(task_name, custom_verifier_path=None, config=None):
    """
    Load a task using reasoning_gym's automatic factory system.
    
    Args:
        task_name: Name of the task (e.g., 'leg_counting')
        custom_verifier_path: Optional path to custom verifier module or 'registry' to use registry
        config: Optional configuration object for verifier settings (e.g., word penalties)
        
    Returns:
        Configured dataset instance
    """
    try:
        # Use reasoning_gym's automatic factory to create the dataset
        dataset = create_dataset(task_name)
        
        # Handle custom verifier
        if custom_verifier_path:
            if custom_verifier_path == "registry":
                # Use the registry system
                try:
                    from verifiers.registry import registry
                    verifier = registry.get_verifier(task_name)
                    if verifier is not None:
                        # If verifier is a class, instantiate it with config
                        if isinstance(verifier, type):
                            verifier_instance = verifier(config)
                        else:
                            verifier_instance = verifier
                        dataset.score_answer = verifier_instance
                        print(f"Loaded custom verifier for task '{task_name}' from registry")
                    else:
                        print(f"No custom verifier found in registry for task '{task_name}'")
                except ImportError:
                    print("Verifier registry not available, falling back to default verifier")
            else:
                # Use the old file-based approach
                spec = importlib.util.spec_from_file_location("custom_verifier", custom_verifier_path)
                custom_verifier = importlib.util.module_from_spec(spec)
                sys.modules["custom_verifier"] = custom_verifier
                spec.loader.exec_module(custom_verifier)
                
                # Replace the dataset's verifier with the custom one
                if hasattr(dataset, 'verifier'):
                    dataset.verifier = custom_verifier.verifier
                elif hasattr(dataset, 'score_answer'):
                    # If dataset has score_answer method, replace it
                    dataset.score_answer = custom_verifier.verifier
                else:
                    # If dataset doesn't have a verifier attribute, we might need to recreate it
                    # This depends on how the dataset class is implemented
                    print(f"Warning: Dataset {task_name} doesn't have a verifier attribute to override")
        
        return dataset
        
    except ValueError as e:
        # Provide helpful error message with available tasks
        from reasoning_gym.factory import DATASETS
        available_tasks = list(DATASETS.keys())
        raise ValueError(
            f"Task '{task_name}' not found. Available tasks: {available_tasks}\n"
            f"Original error: {str(e)}"
        )
    except Exception as e:
        # Handle other unexpected errors
        raise RuntimeError(f"Unexpected error loading task '{task_name}': {str(e)}") 