"""Optimizer class responsible of managing both optimizer and scheduler."""

import functools
from typing import Any

import torch.optim

from .base import AbstractOptimizer

TORCH_OPTIMIZERS = torch.optim.__dict__
TORCH_SCHEDULERS = torch.optim.lr_scheduler.__dict__


class TBOptimizer(AbstractOptimizer):
    """Optimizer class that manage both optimizer and scheduler, fully compatible with `torch.optim` classes.

    Parameters
    ----------
    optimizer_id : str
        Name of the torch optimizer class to be used.
    parameters : dict
        Parameters to be passed to the optimizer.
    scheduler : dict, optional
        Scheduler id and parameters to be used. Default is None.
    """

    def __init__(self, optimizer_id, parameters, scheduler=None) -> None:
        optimizer_id = optimizer_id
        self.optimizer = functools.partial(
            TORCH_OPTIMIZERS[optimizer_id], **parameters
        )

        # CHANGED: Store the scheduler config so we can access keys like 'monitor' later
        self.scheduler_config = scheduler

        if scheduler is not None:
            scheduler_id = scheduler.get("scheduler_id")
            scheduler_params = scheduler.get("scheduler_params")
            self.scheduler = functools.partial(
                TORCH_SCHEDULERS[scheduler_id], **scheduler_params
            )
        else:
            self.scheduler = None

    def __repr__(self) -> str:
        if self.scheduler is not None:
            return f"{self.__class__.__name__}(optimizer={self.optimizer.__name__}, scheduler={self.scheduler.__name__})"
        else:
            return f"{self.__class__.__name__}(optimizer={self.optimizer.__name__})"

    def configure_optimizer(self, model_parameters) -> dict[str:Any]:
        """Configure the optimizer and scheduler.

        Act as a wrapper to provide the LightningTrainer module the required config dict
        when it calls `TBModel`'s `configure_optimizers()` method.

        Parameters
        ----------
        model_parameters : dict
            The model parameters.

        Returns
        -------
        dict
            The optimizer and scheduler configuration.
        """
        optimizer = self.optimizer(params=model_parameters)
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)

            # CHANGED: Use .get() to read from the config, falling back to defaults if missing
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.scheduler_config.get(
                        "monitor", "val/loss"
                    ),
                    "interval": self.scheduler_config.get("interval", "epoch"),
                    "frequency": self.scheduler_config.get("frequency", 1),
                },
            }
        return {"optimizer": optimizer}
