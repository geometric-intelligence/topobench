"""Readout layer that does not perform any operation on the node embeddings."""

import torch_geometric
from torch_geometric.utils import scatter

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class SklearnReadOut(AbstractZeroCellReadOut):
    r"""No readout layer.

    This readout layer does not perform any operation on the node embeddings.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass of the no readout layer.

        It returns the model output without any modification.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        return model_out

    def __call__(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        model_out = self.forward(model_out, batch)

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

        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
