"""HiD-Net backbone for TopoBench.

Adapted from:
Li et al., "A Generalized Neural Diffusion Framework on Graphs", AAAI 2024.
Official implementation: https://github.com/BUPT-GAMMA/HiD-Net,
we return node embeddings instead of log-softmax class predictions for general use
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_scatter import scatter


def feature_norm(x: Tensor, eps: float = 1e-12) -> Tensor:
    """Normalize each node feature vector by its L1 norm.
    Parameters
    ----------
    x : torch.Tensor
        Node feature matrix of shape ``[num_nodes, num_features]``.
    eps : float, optional
        Lower bound for each normalization denominator.

    Returns
    -------
    torch.Tensor
        Row-wise normalized node feature matrix.
    """
    norm = torch.norm(x, p=1, dim=1, keepdim=True).clamp_min(eps)
    return x / norm


def cal_g_gradient3(
    edge_index: Tensor,
    x: Tensor,
    edge_weight: Tensor,
) -> Tensor:
    """HiD-Net g3 high-order gradient term.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices of shape ``[2, num_edges]``.
    x : torch.Tensor
        Node feature matrix of shape ``[num_nodes, num_features]``.
    edge_weight : torch.Tensor
        Normalized edge weights of shape ``[num_edges, 1]``.

    Returns
    -------
    torch.Tensor
        Normalized high-order gradient features.
    """
    row, col = edge_index[0], edge_index[1]

    onestep = scatter(
        (x[col] - x[row]) * edge_weight,
        row,
        dim=-2,
        dim_size=x.size(0),
        reduce="add",
    )

    twostep = scatter(
        onestep[col] * edge_weight,
        row,
        dim=-2,
        dim_size=x.size(0),
        reduce="add",
    )

    return feature_norm(twostep)


class HiDNet(nn.Module):
    """High-order Graph Diffusion Network backbone.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden/output embedding dimension returned to TopoBench.
    num_layers : int
        Number of HiD diffusion iterations. Corresponds to `k` in the
        official implementation.
    alpha : float
        Initial-feature retention coefficient.
    beta : float
        Diffusion coefficient.
    gamma : float
        High-order gradient coefficient.
    dropout : float
        Dropout used before and inside the input MLP.
    add_self_loops : bool
        Whether to add self-loops to the normalized graph used for ordinary
        diffusion.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 10,
        alpha: float = 0.1,
        beta: float = 0.8,
        gamma: float = 0.5,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels

        self.num_layers = num_layers
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin1 = Linear(
            in_channels,
            hidden_channels,
            bias=False,
            weight_initializer="glorot",
        )
        self.lin2 = Linear(
            hidden_channels,
            hidden_channels,
            bias=False,
            weight_initializer="glorot",
        )

    def reset_parameters(self) -> None:
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def _get_norms(
        self,
        edge_index: Tensor,
        edge_weight: Tensor | None,
        num_nodes: int,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return normalized graph for diffusion and gradient term."""

        # Ordinary diffusion graph: with self-loops, as in official model.py.
        norm_edge_index, norm_edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes,
            improved=False,
            add_self_loops=self.add_self_loops,
            dtype=dtype,
        )

        # Gradient graph: no self-loops, as in official model.py.
        grad_edge_index, grad_edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes,
            improved=False,
            add_self_loops=False,
            dtype=dtype,
        )

        return (
            norm_edge_index,
            norm_edge_weight,
            grad_edge_index,
            grad_edge_weight,
        )

    @staticmethod
    def _spmm(edge_index: Tensor, edge_weight: Tensor, x: Tensor) -> Tensor:
        row, col = edge_index[0], edge_index[1]
        return scatter(
            x[col] * edge_weight.view(-1, 1),
            row,
            dim=0,
            dim_size=x.size(0),
            reduce="add",
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        batch: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Return node embeddings for TopoBench."""

        edge_index, edge_weight, edge_index_g, edge_weight_g = self._get_norms(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=x.size(0),
            dtype=x.dtype,
        )

        ew_g = edge_weight_g.view(-1, 1)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        h0 = x

        for _ in range(self.num_layers):
            g = cal_g_gradient3(edge_index_g, x, edge_weight=ew_g)

            ax = self._spmm(edge_index, edge_weight, x)
            gx = self._spmm(edge_index, edge_weight, g)

            x = (
                self.alpha * h0
                + (1.0 - self.alpha - self.beta) * x
                + self.beta * ax
                + self.beta * self.gamma * gx
            )

        return x
