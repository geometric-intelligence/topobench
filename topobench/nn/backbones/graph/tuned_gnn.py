"""Configurable multi-layer GNN backbone for TopoBench.

Provides a single backbone class that composes any supported PyG
convolution (GCN, GAT, SAGE) with per-layer architectural features:
linear residuals, normalization, dropout, jumping knowledge, and
pre-linear projection.  All features are independently toggleable via
Hydra config, making it easy to benchmark the effect of each feature on
any dataset.

Originally ported from TunedGNN (NeurIPS 2024, medium-graph variant).
"""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    to_undirected,
)


class ConfigurableGNN(nn.Module):
    """Multi-layer GNN backbone with configurable per-layer augmentations.

    Each toggle adds one architectural feature on top of a base PyG
    convolution layer.  This lets you reproduce any TunedGNN (NeurIPS
    2024) configuration, or create new ones, purely through config.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Number of hidden features (also the output dimensionality).
    num_layers : int, optional
        Number of message-passing layers (default: 7).
    dropout : float, optional
        Dropout probability (default: 0.5).
    in_dropout : float, optional
        Input dropout applied before the first layer (default: 0.0).
    heads : int, optional
        Number of attention heads for GAT (default: 1).
    pre_ln : bool, optional
        Apply LayerNorm before each convolution (default: False).
    pre_linear : bool, optional
        Project input to hidden_channels before convolutions (default: False).
    res : bool, optional
        Use per-layer learned linear residuals: ``conv(x) + Linear(x)``
        (default: False).
    ln : bool, optional
        Use LayerNorm after each convolution (default: False).
    bn : bool, optional
        Use BatchNorm after each convolution (default: False).
    jk : bool, optional
        Use jumping-knowledge (sum of all layer outputs) (default: False).
    gnn : str, optional
        GNN convolution type: 'gcn', 'gat', or 'sage' (default: 'gcn').
    preprocess_edges : bool, optional
        Apply ``to_undirected`` + ``remove_self_loops`` + ``add_self_loops``
        to the edge index in the forward pass, matching the original TunedGNN
        paper's preprocessing (default: True).
    **kwargs
        Ignored; absorbed for compatibility with Hydra instantiation.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=7,
        dropout=0.5,
        in_dropout=0.0,
        heads=1,
        pre_ln=False,
        pre_linear=False,
        res=False,
        ln=False,
        bn=False,
        jk=False,
        gnn="gcn",
        preprocess_edges=True,
        **kwargs,
    ):
        super().__init__()

        self.out_channels = hidden_channels
        self.dropout = dropout
        self.in_dropout = in_dropout
        self.pre_ln = pre_ln
        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.preprocess_edges = preprocess_edges

        self.h_lins = nn.ModuleList()
        self.local_convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.bns = nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = nn.ModuleList()

        self.lin_in = nn.Linear(in_channels, hidden_channels)

        layers_to_build = num_layers

        if not self.pre_linear:
            self._add_layer(gnn, in_channels, hidden_channels, heads)
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(nn.LayerNorm(in_channels))
            layers_to_build -= 1

        for _ in range(layers_to_build):
            self._add_layer(gnn, hidden_channels, hidden_channels, heads)
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(nn.LayerNorm(hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(nn.LayerNorm(hidden_channels))

    def _add_layer(self, gnn, in_ch, out_ch, heads):
        """Append a convolution layer to local_convs.

        Parameters
        ----------
        gnn : str
            Convolution type ('gcn', 'gat', or 'sage').
        in_ch : int
            Input channels.
        out_ch : int
            Output channels.
        heads : int
            Number of attention heads (GAT only).
        """
        if gnn == "gat":
            self.local_convs.append(
                GATConv(
                    in_ch,
                    out_ch,
                    heads=heads,
                    concat=True,
                    add_self_loops=False,
                    bias=False,
                )
            )
        elif gnn == "sage":
            self.local_convs.append(SAGEConv(in_ch, out_ch))
        else:
            self.local_convs.append(
                GCNConv(in_ch, out_ch, cached=False, normalize=True)
            )

    def reset_parameters(self):
        """Re-initialize all learnable parameters."""
        for conv in self.local_convs:
            conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for layer_norm in self.lns:
            layer_norm.reset_parameters()
        for batch_norm in self.bns:
            batch_norm.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()

    def forward(self, x, edge_index, **kwargs):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``(num_nodes, in_channels)``.
        edge_index : torch.Tensor
            Edge index of shape ``(2, num_edges)``.
        **kwargs
            Ignored; absorbs ``batch`` and ``edge_weight`` from GNNWrapper.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``(num_nodes, hidden_channels)``.
        """
        if self.preprocess_edges:
            edge_index = to_undirected(edge_index)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if self.in_dropout > 0:
            x = F.dropout(x, p=self.in_dropout, training=self.training)

        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_final = 0

        for i, local_conv in enumerate(self.local_convs):
            if self.pre_ln:
                x = self.pre_lns[i](x)
            if self.res:
                x = local_conv(x, edge_index) + self.lins[i](x)
            else:
                x = local_conv(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:  # noqa: SIM108
                x_final = x_final + x
            else:
                x_final = x

        return x_final
