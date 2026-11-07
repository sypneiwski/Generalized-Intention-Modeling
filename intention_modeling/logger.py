import os
import numpy as np
import torch
import wandb

from torchrl.record.loggers import get_logger


class Logger:
    def __init__(self, config, project="project"):
        self.config = config

        self.logger = get_logger(
            logger_type="wandb",
            logger_name="logger",
            experiment_name=self.config.experiment_name,
            wandb_kwargs={
                "project": project,
                "settings": {"x_disable_stats": True}
            },
        )

        hparams = vars(self.config)
        self.logger.log_hparams(hparams)

    def log(
        self,
        logs: dict[str, int]
    ):
        self.logger.experiment.log(logs, commit=True)
