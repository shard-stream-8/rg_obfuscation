import importlib
import importlib.util
import sys
import os
from reasoning_gym.factory import create_dataset
from tasks.terminal_task_loader import load_terminal_task
from tasks.terminal_task_base import TerminalTaskBase

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
    def extract_grid_lines(question):
        lines = question.split('\n')
        grid_lines = []
        in_grid_section = False
        for line in lines:
            line = line.strip()
            if 'binary matrix grid:' in line:
                in_grid_section = True
                continue
            elif in_grid_section and line == '':
                break
            elif in_grid_section:
                grid_lines.append(line)
        return grid_lines

    def count_ones_in_grid(grid_lines):
        return sum(line.split().count('1') for line in grid_lines)

    try:
        # Check if terminal mode is enabled via config
        use_terminal = getattr(config, 'use_terminal', False) if config else False
        
        if use_terminal:
            # Load the base task first to get the dataset
            base_dataset = create_dataset(task_name)
            # FILTER for largest_island: only keep samples with at least 3 ones
            if task_name == "largest_island":
                base_dataset = [sample for sample in base_dataset if count_ones_in_grid(extract_grid_lines(sample["question"])) >= 3]
            # Get verifier code for the task
            verifier_code = get_verifier_code_for_task(task_name, custom_verifier_path, config)
            # Create a terminal task wrapper
            terminal_task = TerminalTaskWrapper(
                task_name=task_name,
                dataset=base_dataset,
                verifier_code=verifier_code,
                config=config
            )
            return terminal_task
        
        # Check if this is a legacy terminal task (for backward compatibility)
        if task_name.endswith('_terminal'):
            return load_terminal_task(task_name, config)
        
        # Use reasoning_gym's automatic factory to create the dataset
        dataset = create_dataset(task_name)
        # FILTER for largest_island: only keep samples with at least 3 ones
        if task_name == "largest_island":
            dataset = [sample for sample in dataset if count_ones_in_grid(extract_grid_lines(sample["question"])) >= 3]
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

def get_verifier_code_for_task(task_name, custom_verifier_path=None, config=None):
    """
    Get verifier code for a task, either from registry or default.
    
    Args:
        task_name: Name of the task
        custom_verifier_path: Optional path to custom verifier
        config: Optional configuration object
        
    Returns:
        Verifier code as string
    """
    # Try to get verifier from registry first
    if custom_verifier_path == "registry":
        try:
            from verifiers.registry import registry
            verifier = registry.get_verifier(task_name)
            if verifier is not None:
                # For terminal mode, we need the verifier code as a string
                # This is a simplified approach - in practice you might want to extract
                # the actual verifier function code
                return f"""
def verifier(content, batch_item):
    \"\"\"
    Verifier for {task_name} task.
    \"\"\"
    # This is a placeholder - the actual verifier logic would be extracted
    # from the registered verifier
    return 1.0 if content.strip() == str(batch_item["answer"]) else 0.0
"""
        except ImportError:
            pass
    
    # Default verifier code
    return f"""
def verifier(content, batch_item):
    \"\"\"
    Default verifier for {task_name} task.
    \"\"\"
    correct_answer = str(batch_item["answer"])
    return 1.0 if content.strip() == correct_answer else 0.0
"""

class TerminalTaskWrapper(TerminalTaskBase):
    """
    Wrapper that converts any reasoning gym task into a terminal task.
    """
    def __init__(self, task_name, dataset, verifier_code, config=None):
        self.task_name = task_name
        self.dataset = dataset
        self.max_turns = getattr(config, 'max_turns', 10) if config else 10
        super().__init__(task_name, verifier_code, config)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.dataset[idx]
        # Always return the raw question
        return {
            "question": sample["question"],
            "answer": sample["answer"],
            "original_question": sample["question"]
        }
    
    def score_answer(self, content: str, batch_item: dict) -> float:
        """
        Score the model's answer using the terminal environment.
        
        Args:
            content: The model's output content
            batch_item: The batch item containing ground truth
            
        Returns:
            Reward value (1.0 for correct, 0.0 for incorrect)
        """
        # Always use terminal processing when terminal mode is enabled
        result = self.process_model_output(content, batch_item["answer"])
        return result["reward"] 