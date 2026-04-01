"""Abstract base class for readout layers."""

from abc import abstractmethod

import torch
import torch_geometric
from torch_geometric.utils import scatter


class AbstractZeroCellReadOut(torch.nn.Module):
    r"""Readout layer for GNNs that operates on the batch level.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the GNN model.
    out_channels : int
        Number of output channels.
    task_level : str
        Task level for readout layer. Either "graph" or "node".
    pooling_type : str
        Pooling type for readout layer. Either "max", "sum" or "mean".
    logits_linear_layer : bool
        Whether to use a linear layer for getting the final logits.
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        task_level: str,
        pooling_type: str = "sum",
        logits_linear_layer: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.linear = (
            torch.nn.Linear(hidden_dim, out_channels)
            if hidden_dim != out_channels or logits_linear_layer
            else torch.nn.Identity()
        )
        assert task_level in ["graph", "node"] or (
            self.name == "NoReadOut" and task_level == "edge"
        ), "Invalid task_level"
        self.task_level = task_level
        self.logits_linear_layer = logits_linear_layer

        assert pooling_type in ["max", "sum", "mean"], "Invalid pooling_type"
        self.pooling_type = pooling_type

    def __repr__(self):
        return f"{self.__class__.__name__}(task_level={self.task_level}, pooling_type={self.pooling_type})"

    def __call__(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        """Readout logic based on model_output.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        model_out = self.forward(model_out, batch)

        if model_out.get("logits", None) is None:
            model_out["logits"] = self.compute_logits(
                model_out["x_0"], batch["batch_0"]
            )

        return model_out

    def compute_logits(self, x, batch):
        r"""Compute logits based on the readout layer.

        Parameters
        ----------
        x : torch.Tensor
            Node embeddings.
        batch : torch.Tensor
            Batch index tensor.

        Returns
        -------
        torch.Tensor
            Logits tensor.
        """
        if self.task_level == "graph":
            x = scatter(x, batch, dim=0, reduce=self.pooling_type)

        return self.linear(x)

    @abstractmethod
    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.
        """
