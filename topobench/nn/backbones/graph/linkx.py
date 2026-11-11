"""LINKX backbone for TopoBench.

Implements an inductive adaptation of LINKX from:
Lim et al., "Large Scale Learning on Non-Homophilous Graphs:
New Benchmarks and Strong Simple Methods", NeurIPS 2021.
https://arxiv.org/abs/2110.14446

Reference implementation: https://github.com/CUAI/Non-Homophily-Large-Scale

Original (transductive): MLP_final(σ(W·[MLP_A(A), MLP_X(X)] + MLP_A(A) + MLP_X(X)))
    where A is the raw adjacency matrix (each row = a node's connections).

Inductive adaptation: replaces A (fixed-size adjacency rows) with AX
    (neighbor-aggregated features), preserving the key architectural insight
    of separately processing structural and feature information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter


class _MLP(nn.Module):
    """Multi-layer perceptron with BatchNorm.

    Matches the MLP design from the original LINKX implementation:
    Linear -> [BatchNorm -> ReLU -> Dropout -> Linear ->]* Linear

    Parameters
    ----------
    in_channels : int
        Input dimension.
    hidden_channels : int
        Hidden layer dimension.
    out_channels : int
        Output dimension.
    num_layers : int
        Total number of linear layers.
    dropout : float
        Dropout probability between layers.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout=0.0,
    ):
        super().__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [N, in_channels].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, out_channels].
        """
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


def _aggregate_neighbors(x, edge_index):
    """Mean neighbor aggregation via scatter.

    Computes AX where A is the row-normalized adjacency matrix,
    equivalent to averaging neighbor features for each node.

    Parameters
    ----------
    x : torch.Tensor
        Node features [num_nodes, dim].
    edge_index : torch.Tensor
        Edge indices [2, num_edges].

    Returns
    -------
    torch.Tensor
        Aggregated features [num_nodes, dim].
    """
    src, dst = edge_index
    return scatter(x[src], dst, dim=0, dim_size=x.size(0), reduce="mean")


class LINKX(nn.Module):
    """Inductive LINKX backbone.

    Separately processes neighbor-aggregated features (AX) and raw node
    features (X) through independent MLPs, then combines them with skip
    connections. This architecture is particularly effective for heterophilous
    graphs where neighboring nodes may have different labels.

    Formula (Eq. 4 from the paper, adapted for inductive setting):
        h_A = MLP_A(AX)
        h_X = MLP_X(X)
        output = MLP_final(σ(W·[h_A, h_X] + h_A + h_X))

    Parameters
    ----------
    in_channels : int
        Input feature dimension (from feature encoder).
    hidden_channels : int
        Hidden and output embedding dimension.
    num_layers : int, optional
        Number of layers in final MLP. Default is 2.
    num_edge_layers : int, optional
        Number of layers in the structure (adjacency) MLP. Default is 1.
    num_node_layers : int, optional
        Number of layers in the node feature MLP. Default is 1.
    dropout : float, optional
        Dropout rate for the final MLP. Default is 0.5.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        num_edge_layers=1,
        num_node_layers=1,
        dropout=0.5,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels

        # MLP_A: processes aggregated neighbor features
        self.mlp_a = _MLP(
            in_channels,
            hidden_channels,
            hidden_channels,
            num_edge_layers,
            dropout=0.0,
        )
        # MLP_X: processes raw node features
        self.mlp_x = _MLP(
            in_channels,
            hidden_channels,
            hidden_channels,
            num_node_layers,
            dropout=0.0,
        )
        # W: combines the two branches
        self.combine = nn.Linear(2 * hidden_channels, hidden_channels)
        # MLP_final: produces output embeddings
        self.mlp_final = _MLP(
            hidden_channels,
            hidden_channels,
            hidden_channels,
            num_layers,
            dropout=dropout,
        )

    def forward(self, x, edge_index, batch=None, edge_weight=None, **kwargs):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        batch : torch.Tensor, optional
            Batch vector of shape [num_nodes]. Default is None.
        edge_weight : torch.Tensor, optional
            Edge weights (unused, kept for interface compatibility). Default is None.
        **kwargs : dict
            Additional arguments (unused).

        Returns
        -------
        torch.Tensor
            Output node embeddings of shape [num_nodes, hidden_channels].
        """
        # Aggregate neighbor features: AX
        ax = _aggregate_neighbors(x, edge_index)

        # Process through separate MLPs (Eq. 4)
        h_a = self.mlp_a(ax)
        h_x = self.mlp_x(x)

        # Combine with skip connections: σ(W·[h_a, h_x] + h_a + h_x)
        h = torch.cat([h_a, h_x], dim=-1)
        h = self.combine(h)
        h = F.relu(h + h_a + h_x)

        # Final MLP
        h = self.mlp_final(h)
        return h
