"""This module defines the `TBModel` class."""

from typing import Any

import torch
from lightning import LightningModule
from torch_geometric.data import Data
from torchmetrics import MeanMetric


class TBModel(LightningModule):
    r"""A `LightningModule` to define a network.

    Parameters
    ----------
    backbone : torch.nn.Module
        The backbone model to train.
    readout : torch.nn.Module
        The readout class.
    loss : torch.nn.Module
        The loss class.
    backbone_wrapper : torch.nn.Module, optional
        The backbone wrapper class (default: None).
    feature_encoder : torch.nn.Module, optional
        The feature encoder (default: None).
    evaluator : Any, optional
        The evaluator class (default: None).
    optimizer : Any, optional
        The optimizer class (default: None).
    **kwargs : Any
        Additional keyword arguments.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        readout: torch.nn.Module,
        loss: torch.nn.Module,
        backbone_wrapper: torch.nn.Module | None = None,
        feature_encoder: torch.nn.Module | None = None,
        evaluator: Any = None,
        optimizer: Any = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # This line allows accessing init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False, ignore=["backbone", "readout", "feature_encoder"]
        )

        self.feature_encoder = (
            feature_encoder
            if feature_encoder is not None
            else torch.nn.Identity()
        )
        if backbone_wrapper is None:
            self.backbone = backbone
        else:
            self.backbone = backbone_wrapper(backbone)
        self.readout = readout

        # Evaluator
        self.evaluator = evaluator
        self.train_metrics_logged = False

        # Optimizer (it also internally manages Scheduler if provided)
        self.optimizer = optimizer

        # Loss function
        self.loss = loss
        self.task_level = self.readout.task_level

        # Tracking best so far validation accuracy
        self.val_acc_best = MeanMetric()
        self.metric_collector_val = []
        self.metric_collector_val2 = []
        self.metric_collector_test = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(backbone={self.backbone}, readout={self.readout}, loss={self.loss}, feature_encoder={self.feature_encoder})"

    def forward(self, batch: Data) -> dict:
        r"""Perform a forward pass through the model.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output, which includes the logits and other relevant information.
        """
        # Feature Encoder
        model_out = self.feature_encoder(batch)

        # Domain model
        model_out = self.backbone(model_out)

        # Readout
        model_out = self.readout(model_out=model_out, batch=batch)

        return model_out

    def model_step(self, batch: Data) -> dict:
        r"""Perform a single model step on a batch of data.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output and the loss.
        """
        # Allow batch object to know the phase of the training
        batch["model_state"] = self.state_str

        # Forward pass
        model_out = self.forward(batch)

        # Loss
        model_out = self.process_outputs(model_out=model_out, batch=batch)

        # Metric
        model_out = self.loss(model_out=model_out, batch=batch)

        # Add batch to model_out for evaluator access to target normalizer stats
        model_out["batch"] = batch

        self.evaluator.update(model_out)

        return model_out

    def training_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        r"""Perform a single training step on a batch of data.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            A tensor of losses between model predictions and targets.
        """
        self.state_str = "Training"
        model_out = self.model_step(batch)

        # Update and log metrics
        loss_value = model_out["loss"].item()
        self.log(
            "train/loss",
            loss_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

        # Return loss for backpropagation step
        return model_out["loss"]

    def validation_step(self, batch: Data, batch_idx: int) -> None:
        r"""Perform a single validation step on a batch of data.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
        """
        self.state_str = "Validation"
        model_out = self.model_step(batch)

        # Log Loss
        loss_value = model_out["loss"].item()
        self.log(
            "val/loss",
            loss_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

    def test_step(self, batch: Data, batch_idx: int) -> None:
        r"""Perform a single test step on a batch of data.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.
        batch_idx : int
            The index of the current batch.
        """
        self.state_str = "Test"
        model_out = self.model_step(batch)

        # Log loss
        loss_value = model_out["loss"].item()
        self.log(
            "test/loss",
            loss_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

    def process_outputs(self, model_out: dict, batch: Data) -> dict:
        r"""Handle model outputs.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        # Get the correct mask
        if self.state_str == "Training":
            mask = batch.train_mask
        elif self.state_str == "Validation":
            mask = batch.val_mask
        elif self.state_str == "Test":
            mask = batch.test_mask
        else:
            raise ValueError("Invalid state_str")

        if self.task_level == "node":
            # Keep only train data points
            for key, val in model_out.items():
                if key in ["logits", "labels"]:
                    model_out[key] = val[mask]

        return model_out

    def log_metrics(self, mode=None):
        r"""Log metrics.

        Parameters
        ----------
        mode : str, optional
            The mode of the model, either "train", "val", or "test" (default: None).
        """
        metrics_dict = self.evaluator.compute()

        # Log current metrics
        for key in metrics_dict:
            self.log(
                f"{mode}/{key}",
                metrics_dict[key],
                prog_bar=True,
                on_step=False,
            )

        # Reset evaluator for next epoch
        self.evaluator.reset()

    def on_validation_epoch_start(self) -> None:
        r"""Hook called when a validation epoch begins.

        According pytorch lightning documentation this hook is called at the beginning of the
        validation epoch.

        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        Note that the validation step is within the train epoch. Hence here we have to log the train metrics
        before we reset the evaluator to start the validation loop.
        """
        # Log train metrics and reset evaluator
        self.log_metrics(mode="train")
        self.train_metrics_logged = True

    def on_train_epoch_end(self) -> None:
        r"""Lightning hook that is called when a train epoch ends.

        This hook is used to log the train metrics.
        """
        # Log train metrics and reset evaluator
        if not self.train_metrics_logged:
            self.log_metrics(mode="train")
            self.train_metrics_logged = True

    def on_validation_epoch_end(self) -> None:
        r"""Lightning hook that is called when a validation epoch ends.

        This hook is used to log the validation metrics.
        """
        # Log validation metrics and reset evaluator
        self.log_metrics(mode="val")

    def on_test_epoch_end(self) -> None:
        r"""Lightning hook that is called when a test epoch ends.

        This hook is used to log the test metrics.
        """
        self.log_metrics(mode="test")

    def on_train_epoch_start(self) -> None:
        r"""Lightning hook that is called when a train epoch begins.

        This hook is used to reset the train metrics.
        """
        self.evaluator.reset()
        self.train_metrics_logged = False

    def on_val_epoch_start(self) -> None:
        r"""Lightning hook that is called when a validation epoch begins.

        This hook is used to reset the validation metrics.
        """
        self.evaluator.reset()

    def on_test_epoch_start(self) -> None:
        r"""Lightning hook that is called when a test epoch begins.

        This hook is used to reset the test metrics.
        """
        self.evaluator.reset()

    def setup(self, stage: str) -> None:
        r"""Hook to call torch.compile.

        Lightning hook that is called at the beginning of fit (train +
        validate), validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using
        DDP.

        Parameters
        ----------
        stage : str
            Either "fit", "validate", "test", or "predict".
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> dict[str, Any]:
        r"""Configure optimizers and learning-rate schedulers.

        Choose what optimizers and learning-rate schedulers to use in your
        optimization. Normally you'd need one. But in the case of GANs or
        similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns
        -------
        dict:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer_config = self.optimizer.configure_optimizer(
            list(self.backbone.parameters())
            + list(self.readout.parameters())
            + list(self.feature_encoder.parameters())
        )

        return optimizer_config
