"""Loss module for the topobench package."""

import torch
import torch_geometric

from topobench.loss.base import AbstractLoss


class DatasetLoss(AbstractLoss):
    r"""Defines the default model loss for the given task.

    Parameters
    ----------
    dataset_loss : dict
        Dictionary containing the dataset loss information.
    """

    def __init__(self, dataset_loss):
        super().__init__()
        self.task = dataset_loss["task"]
        self.loss_type = dataset_loss["loss_type"]
        # Dataset loss
        if self.task == "classification":
            assert self.loss_type == "cross_entropy", (
                "Invalid loss type for classification task,TB supports only cross_entropy loss for classification task"
            )
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.task == "multilabel classification":
            assert self.loss_type == "BCE", (
                "Invalid loss type for classification task,TB supports only BCE for multilabel classification task"
            )
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif (
            self.task == "regression"
            and self.loss_type == "mse"
            or (
                self.task == "multioutput classification"
                and self.loss_type == "mse"
            )
        ):
            self.criterion = torch.nn.MSELoss()

        elif self.task == "regression" and self.loss_type == "mae":
            self.criterion = torch.nn.L1Loss()
        else:
            raise Exception("Loss is not defined")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, loss_type={self.loss_type})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the loss function.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output with the loss.
        """
        logits = model_out["logits"]
        target = model_out["labels"]

        return self.forward_criterion(logits, target)

    def forward_criterion(self, logits, target):
        r"""Forward pass of the loss function.

        Parameters
        ----------
        logits : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        if self.task == "regression":
            target = target.unsqueeze(1)
            dataset_loss = self.criterion(logits, target)

        elif self.task == "multioutput classification":
            dataset_loss = self.criterion(logits, target.float())

        elif self.task == "classification":
            dataset_loss = self.criterion(logits, target)

        elif self.task == "multilabel classification":
            mask = ~torch.isnan(target)
            # Avoid NaN values in the target
            target = torch.where(mask, target, torch.zeros_like(target))
            loss = self.criterion(logits, target.float())
            # Mask out the loss for NaN values
            loss = loss * mask
            # Take out average
            dataset_loss = (loss.sum(dim=-1) / mask.sum(dim=-1)).mean()

        else:
            raise Exception("Loss is not defined")

        return dataset_loss
