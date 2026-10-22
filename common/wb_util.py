"""Weights & Biases logging. Every call does nothing unless enabled=True."""


class WandbLogger:
    def __init__(self, enabled, project, name, config):
        self.enabled = enabled
        if enabled:
            import wandb

            self.wandb = wandb
            wandb.init(project=project, name=name, config=config)

    def log(self, metrics, step):
        if self.enabled:
            self.wandb.log(metrics, step=step)

    def finish(self):
        if self.enabled:
            self.wandb.finish()
