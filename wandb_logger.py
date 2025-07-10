import wandb
from wandb.errors import CommError
import re

def get_next_run_number_wandb_api(task_name, project, entity=None):
    """
    Use the wandb public API to find the next available run number for a given task.
    Args:
        task_name: Name of the task (e.g., 'acre')
        project: Name of the wandb project
        entity: (Optional) wandb entity (user or team)
    Returns:
        int: Next available run number
    """
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project
    try:
        runs = api.runs(project_path)
    except CommError:
        # If the project doesn't exist yet, create it and return 1 as the first run number
        api.create_project(name=project, entity=entity)
        return 1
    max_num = 0
    prefix = f"{task_name}-"
    for run in runs:
        name = run.name or ""
        if name.startswith(prefix):
            try:
                num = int(name[len(prefix):])
                if num > max_num:
                    max_num = num
            except ValueError:
                continue
    return max_num + 1

class WandbLogger:
    def __init__(self, config):
        # Generate run name in format [task]-[number] using wandb API
        task_name = config.task_name
        project_name = config.wandb_project
        entity = getattr(config, 'wandb_entity', None)
        run_number = get_next_run_number_wandb_api(task_name, project_name, entity)
        run_name = f"{task_name}-{run_number}"
        
        self.run = wandb.init(
            project=project_name,
            entity=entity,
            name=run_name,
            config={k: v for k, v in config.items()},
            reinit=True
        )
        
    def log(self, metrics, step=None):
        wandb.log(metrics, step=step)
        
    def finish(self):
        wandb.finish() 