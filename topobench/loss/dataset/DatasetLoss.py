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
            assert self.loss_type in ("BCE", "focal"), (
                "Invalid loss type for multilabel classification task, "
                "TB supports 'BCE' and 'focal'"
            )
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
            if self.loss_type == "focal":
                self.focal_gamma = dataset_loss.get("focal_gamma", 2.0)
                self.focal_alpha = dataset_loss.get("focal_alpha", None)
        elif self.task == "regression" and self.loss_type == "mse":
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

        elif self.task == "classification":
            dataset_loss = self.criterion(logits, target)

        elif self.task == "multilabel classification":
            mask = ~torch.isnan(target)
            # Avoid NaN values in the target
            target = torch.where(mask, target, torch.zeros_like(target))
            loss = self.criterion(logits, target)

            if self.loss_type == "focal":
                # Apply focal modulation: (1 - p_t)^gamma
                # p_t = sigmoid(logit) for positives, 1 - sigmoid(logit) for negatives
                p = torch.sigmoid(logits)
                p_t = p * target + (1.0 - p) * (1.0 - target)
                focal_weight = (1.0 - p_t) ** self.focal_gamma
                if self.focal_alpha is not None:
                    alpha_t = (
                        self.focal_alpha * target
                        + (1.0 - self.focal_alpha) * (1.0 - target)
                    )
                    focal_weight = alpha_t * focal_weight
                loss = focal_weight * loss

            # Mask out the loss for NaN values
            loss = loss * mask
            # Take out average
            dataset_loss = (loss.sum(dim=-1) / mask.sum(dim=-1)).mean()

        else:
            raise Exception("Loss is not defined")

        return dataset_loss
