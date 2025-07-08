import wandb
import os
import re
from pathlib import Path

def get_next_run_number(task_name, project_name):
    """
    Find the next available run number for a given task.
    
    Args:
        task_name: Name of the task (e.g., 'acre')
        project_name: Name of the wandb project
        
    Returns:
        int: Next available run number
    """
    # Get the wandb directory path
    wandb_dir = Path("wandb")
    if not wandb_dir.exists():
        return 1
    
    # Pattern to match existing runs for this task
    pattern = re.compile(rf"^{task_name}-(\d+)$")
    
    # Find all existing run numbers for this task
    existing_numbers = []
    for item in wandb_dir.iterdir():
        if item.is_dir() and item.name.startswith("run-"):
            # Extract the run name from the directory
            # wandb directories are named like: run-20250708_214858-t9or7go1
            # We need to check the actual run name inside the directory
            run_info_file = item / "wandb-metadata.json"
            if run_info_file.exists():
                try:
                    import json
                    with open(run_info_file, 'r') as f:
                        metadata = json.load(f)
                    run_name = metadata.get('name', '')
                    match = pattern.match(run_name)
                    if match:
                        existing_numbers.append(int(match.group(1)))
                except (json.JSONDecodeError, KeyError):
                    continue
    
    # Return the next available number
    if not existing_numbers:
        return 1
    return max(existing_numbers) + 1

class WandbLogger:
    def __init__(self, config):
        # Generate run name in format [task]-[number]
        task_name = config.task_name
        project_name = config.wandb_project
        run_number = get_next_run_number(task_name, project_name)
        run_name = f"{task_name}-{run_number}"
        
        self.run = wandb.init(
            project=config.wandb_project,
            entity=getattr(config, 'wandb_entity', None),
            name=run_name,
            config={k: v for k, v in config.items()},
            reinit=True
        )
        
    def log(self, metrics, step=None):
        wandb.log(metrics, step=step)
        
    def finish(self):
        wandb.finish() 