import wandb
from wandb.errors import CommError
import re

# Helper to fetch default entity from wandb settings when not explicitly provided.
def _get_default_entity():
    try:
        return wandb.Api().default_entity
    except Exception:
        return None

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
    except (CommError, ValueError):
        # If the project doesn't exist yet, create it and return 1 as the first run number.
        # `create_project` requires a non-None entity when provided. Build kwargs dynamically to
        # avoid passing None and silence type checker complaints.
        # Always pass a non-None entity to satisfy wandb API requirements.
        effective_entity = entity or _get_default_entity()
        create_kwargs = {"name": project, "entity": effective_entity}
        api.create_project(**create_kwargs)
        # Flush API cache so that subsequent queries see the new project
        api.flush()
        project_path = f"{effective_entity}/{project}" if effective_entity else project
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
        # Determine the project name, allowing an explicit override via `wandb_project_name` in the config.
        # `wandb_project_name` takes precedence over the legacy `wandb_project` key for backward compatibility.
        project_name = getattr(config, "wandb_project_name", None) or getattr(config, "wandb_project", None)
        if project_name is None:
            self.run = None
            return
            
        # Generate run name in format [task]-[number] using wandb API
        task_name = getattr(config, "task_name_for_wandb", getattr(config, "task_name", "run"))
        entity = getattr(config, 'wandb_entity', None) or _get_default_entity()
        run_number = get_next_run_number_wandb_api(task_name, project_name, entity)
        run_name = f"{task_name}-{run_number}"
        # Try to initialise the W&B run; if the project was somehow still missing (race condition or
        # different default entity), fall back to creating it and retry once.
        try:
            self.run = wandb.init(
                project=project_name,
                entity=entity,
                name=run_name,
                config={k: v for k, v in config.items()},
                reinit=True
            )
        except (CommError, ValueError):
            # Create the project (again) just to be safe and retry.
            effective_entity = entity or _get_default_entity()
            create_kwargs = {"name": project_name, "entity": effective_entity}
            wandb.Api().create_project(**create_kwargs)
            wandb.Api().flush()
            self.run = wandb.init(
                project=project_name,
                entity=effective_entity,
                name=run_name,
                config={k: v for k, v in config.items()},
                reinit=True
            )
        
    def log(self, metrics, step=None):
        if self.run is not None:
            wandb.log(metrics, step=step)
        
    def finish(self):
        if self.run is not None:
            wandb.finish() 