import wandb

class WandbLogger:
    def __init__(self, config):
        self.run = wandb.init(
            project=config.wandb_project,
            entity=getattr(config, 'wandb_entity', None),
            config={k: v for k, v in config.items()},
            reinit=True
        )
    def log(self, metrics, step=None):
        wandb.log(metrics, step=step)
    def finish(self):
        wandb.finish() 