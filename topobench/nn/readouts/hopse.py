"""Readout function for the HOPSE model."""

import topomodelx
import torch
import torch_geometric
from torch_scatter import scatter

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class HOPSEReadout(AbstractZeroCellReadOut):
    r"""Readout function for the HOPSE model.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments. It should contain the following keys:
        - complex_dim (int): Dimension of the simplicial complex.
        - max_hop (int): Maximum hop neighbourhood to consider.
        - hidden_dim_1 (int):  Dimension of the embeddings.
        - out_channels (int): Number of classes.
        - pooling_type (str): Type of pooling operationg
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.complex_dim = kwargs["complex_dim"]
        self.max_hop = kwargs["max_hop"]
        self.task_level = kwargs["task_level"]
        hidden_dim = kwargs["hidden_dim"]
        out_channels = kwargs["out_channels"]
        pooling_type = kwargs["pooling_type"]
        self.dimensions = range(kwargs["complex_dim"] - 1, 0, -1)

        if self.task_level == "node":
            self._node_level_task_inits(hidden_dim)
            self.linear = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, out_channels),
            )

        elif self.task_level == "graph":
            self._graph_level_task_inits(hidden_dim)
            self.linear = torch.nn.Sequential(
                # We add a +1 to complex dim because the complex dimension is 0-indexed
                torch.nn.Linear(
                    (self.complex_dim + 1) * hidden_dim, hidden_dim
                ),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, out_channels),
            )
        assert pooling_type in ["max", "sum", "mean"], "Invalid pooling_type"
        self.pooling_type = pooling_type

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

        x_all = model_out["x_all"]
        bt = batch["batch_0"]

        model_out["logits"] = self.compute_logits(x_all, bt)
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
        return self.linear(x)

    def forward(
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

        if self.task_level == "graph":
            x_out = []
            for i in range(self.complex_dim + 1):
                x_i = torch.cat(
                    [model_out[f"x{i}_{j}"] for j in range(self.max_hop)],
                    dim=1,
                )

                # MLP for aggregating  hops
                x_i = getattr(self, f"linear_rank_{i}")(x_i)

                # This is pooling per r
                x_i = scatter(
                    x_i,
                    batch[f"batch_{i}"],
                    dim=0,
                    reduce=self.pooling_type,
                    dim_size=batch.batch_0.max() + 1,
                )
                x_out.append(x_i)
            x_all_cat = torch.cat(x_out, dim=1)

        elif self.task_level == "node":
            for i in self.dimensions:
                for j in range(self.max_hop - 1, -1, -1):
                    x_i = getattr(self, f"agg_conv_{i}")(
                        model_out[f"x{i}_{j}"], batch[f"incidence_{i}"]
                    )
                    # x_i = getattr(self, f"ln_{i}")(x_i)
                    model_out[f"x{i - 1}_{j}"] = getattr(
                        self, f"projector_{i}"
                    )(torch.cat([x_i, model_out[f"x{i - 1}_{j}"]], dim=1))

            x_all_cat = model_out["x0_0"]

        model_out["x_all"] = x_all_cat

        return model_out

    def _node_level_task_inits(self, hidden_dim: int):
        """Initialize the node-level task.

        Parameters
        ----------
        hidden_dim : int
            Dimension of the embeddings.
        """

        for i in self.dimensions:
            setattr(
                self,
                f"agg_conv_{i}",
                topomodelx.base.conv.Conv(
                    hidden_dim, hidden_dim, aggr_norm=False
                ),
            )

            setattr(self, f"ln_{i}", torch.nn.LayerNorm(hidden_dim))

            setattr(
                self,
                f"projector_{i}",
                torch.nn.Linear(2 * hidden_dim, hidden_dim),
            )

    def _graph_level_task_inits(self, hidden_dim: int):
        """Initialize the node-level task.

        Parameters
        ----------
        hidden_dim : int
            Dimension of the embeddings.
        """

        for i in range(self.complex_dim + 1):
            setattr(
                self,
                f"linear_rank_{i}",
                torch.nn.Sequential(
                    torch.nn.Linear(self.max_hop * hidden_dim, hidden_dim),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.LeakyReLU(),
                ),
            )
