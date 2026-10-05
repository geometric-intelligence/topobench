"""This module defines the `TBModel` class."""

from typing import Any

import torch
from lightning import LightningModule
from torch_geometric.data import Data
from torchmetrics import MeanMetric


class HostBatchTransferMixin:
    """Move batches to the device without making the host wait.

    Backbones listing ``host_fields`` receive CPU copies of those fields as
    ``batch.trawl_host`` (read before the transfer), so host-side work such as
    walk sampling never synchronizes with the device. Batch tensors are copied
    with ``non_blocking``, which overlaps with compute for pinned memory.
    """

    def on_before_batch_transfer(self, batch, dataloader_idx):
        """Attach host copies of the backbone's ``host_fields`` to the batch.

        Parameters
        ----------
        batch : Any
            The batch, still on the host.
        dataloader_idx : int
            Index of the dataloader that produced the batch.

        Returns
        -------
        Any
            The batch, with ``trawl_host`` set for ``Data`` batches.
        """
        backbone = getattr(self, "backbone", None)
        fields = getattr(backbone, "host_fields", ())
        if fields and isinstance(batch, Data):
            batch.trawl_host = {
                key: batch[key].numpy() for key in fields if key in batch
            }
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Copy the batch to the device, using ``non_blocking`` for ``Data``.

        Parameters
        ----------
        batch : Any
            The batch to transfer.
        device : torch.device
            Target device.
        dataloader_idx : int
            Index of the dataloader that produced the batch.

        Returns
        -------
        Any
            The batch on ``device``.
        """
        if isinstance(batch, Data):
            return batch.to(device, non_blocking=True)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)


class TBModel(HostBatchTransferMixin, LightningModule):
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
    evaluation_autocast : bool, optional
        If False, validation and test forward passes run with autocast
        disabled, so mixed-precision training still evaluates in full
        precision (default: True).
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
        evaluation_autocast: bool = True,
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
        self.evaluation_autocast = evaluation_autocast

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
        if self.training or self.evaluation_autocast:
            model_out = self.forward(batch)
        else:
            with torch.autocast(self.device.type, enabled=False):
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

        # Update and log metrics. Logging the tensor (not ``item()``) avoids
        # a device synchronization per step; Lightning stores the same value.
        loss_value = model_out["loss"].detach().float()
        self.log(
            "train/loss",
            loss_value,
            sync_dist=True,
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
        loss_value = model_out["loss"].detach().float()
        self.log(
            "val/loss",
            loss_value,
            sync_dist=True,
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
        loss_value = model_out["loss"].detach().float()
        self.log(
            "test/loss",
            loss_value,
            sync_dist=True,
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
        if self.task_level == "node":
            # Get the correct mask
            if self.state_str == "Training":
                mask = batch.train_mask
            elif self.state_str == "Validation":
                mask = batch.val_mask
            elif self.state_str == "Test":
                mask = batch.test_mask
            else:
                raise ValueError("Invalid state_str")

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
        if (
            self.trainer.state.fn == "fit"
            and not self.trainer.sanity_checking
            and not self.train_metrics_logged
        ):
            self.log_metrics(mode="train")
            self.train_metrics_logged = True
        self.evaluator.reset()

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
        if stage == "fit":
            self.configure_compilation()

    def configure_compilation(self):
        """Compile in place, preserving checkpoint keys and CPU walk sampling."""
        if not self.hparams.get("compile", False):
            return
        scope = self.hparams.get("compile_scope", "backbone")
        if scope == "layers":
            if not hasattr(self.backbone, "encoders"):
                raise ValueError("Layer compilation requires encoder stacks")
            modules = [
                layer for stack in self.backbone.encoders for layer in stack
            ]
        elif scope == "backbone":
            modules = [self.backbone]
        else:
            raise ValueError(f"Unknown compilation scope: {scope}")
        for module in modules:
            if getattr(module, "_compiled_call_impl", None) is None:
                module.compile()

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
        scheduler = optimizer_config.get("lr_scheduler")
        if scheduler is not None and isinstance(
            scheduler["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            # Step plateau schedulers only on epochs that produce the metric.
            trainer = self._trainer
            scheduler["frequency"] = int(
                getattr(trainer, "check_val_every_n_epoch", None) or 1
            )
        return optimizer_config
