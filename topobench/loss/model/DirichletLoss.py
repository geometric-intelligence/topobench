"""Dirichlet-energy regularization loss for the Gauge model."""

import torch
from torch import Tensor
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter

from topobench.loss.base import AbstractLoss


class DirichletLoss(AbstractLoss):
    r"""Dirichlet-energy regularization loss for the Gauge model (cf. paper equation (13)).

    Measures how smoothly the node embeddings vary across edges once projected
    onto each node's local frame ``Q``. Each node embedding is projected onto
    its ``r`` frame vectors and L2-normalized; the projection of the current
    embedding is then averaged over neighbors and compared to the projection of
    the (detached) initial embedding. The resulting term is scaled by ``lamb``
    and added to the task loss as a regularizer.

    Parameters
    ----------
    lamb : float, optional
        Weight (lambda) of the regularizer (default: 0.1).
    reduction : str, optional
        Neighbor aggregation reduction, either "mean" or "sum" (default: "mean").
    """

    def __init__(self, lamb: float = 0.1, reduction: str = "mean"):
        super().__init__()

        if reduction not in ["mean", "sum"]:
            raise NotImplementedError(
                f"reduction '{reduction}' not implemented. Valid choices are 'mean', 'sum'."
            )

        self.lamb = lamb
        self.reduce = reduction

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lamb={self.lamb}, reduction={self.reduce})"

    def forward(self, model_out: dict, batch: Data) -> Tensor:
        r"""Compute the Dirichlet-energy regularization loss according to the paper's eq. 13.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output. Uses ``x_0`` (the final node
            embeddings ``[N, d]``), ``z_0`` (the initial node embeddings
            ``[N, d]``) and ``Q`` (the per-node frames ``[N, r, d]``).
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data. Uses
            ``edge_index`` for the neighbor aggregation.

        Returns
        -------
        Tensor
            Scalar regularization loss scaled by ``lamb``.
        """

        zL = model_out["x_0"]  # [N, d]
        Q = model_out["Q"]  # Q has shape [N, r, d]
        z0 = model_out["z_0"]  # [N, d]

        N = z0.size(0)
        # Removed here (and, consistently, in the Gauge backbone) so the loss
        # aggregates over the same self-loop-free neighborhoods as the model.
        edge_index, _ = remove_self_loops(batch.edge_index)
        src, dst = edge_index[0], edge_index[1]

        # this is essentially the zhat=StopGrad(z0)
        zhat = z0.detach()

        src_term = F.normalize(torch.einsum("ijk,ik->ij", Q, zhat), dim=-1)
        agg_term = F.normalize(torch.einsum("ijk,ik->ij", Q, zL), dim=-1)

        agg_term = scatter(
            agg_term[src], index=dst, dim=0, dim_size=N, reduce=self.reduce
        )

        loss = ((src_term - agg_term) ** 2).sum(dim=-1).mean()

        return self.lamb * loss
