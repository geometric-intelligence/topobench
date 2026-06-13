"""GATE backbone: graph attention that separates self from neighbours.

GATE extends GATv2 by giving a node's self-loop its own attention vector
and value transform, plus an optional learned gate ``omega`` that controls
how much a node retains its own signal versus aggregating (potentially
heterophilic, "intrusive") neighbours. This self/neighbour separation is
what makes the model robust on heterophilic graphs.

Notes
-----
The exact message routing (value transform, i/j assignment) is validated
by a numerical parity test against the official ``GATv2Conv`` reference;
see ``test/nn/backbones/graph/test_gate.py``.

References
----------
.. [1] "GATE: How to Keep Out Intrusive Neighbors." ICML 2024.
   Official code: https://github.com/RelationalML/GATE
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class GATEConv(MessagePassing):
    r"""Single GATE attention layer.

    The attention logit for an edge :math:`(i, j)` follows GATv2,
    :math:`\alpha_{ij} \propto a^\top \mathrm{LeakyReLU}(W_l x_i + W_r x_j)`,
    but self-loops are parameterized separately: they use a distinct
    attention vector ``att2`` and a distinct value transform ``lin_s``,
    and an optional gate :math:`\omega` rescales the per-edge contribution
    as :math:`x_j (\mathbb{1}_{ii} - \omega(\mathbb{1}_{ii} - \alpha))`.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features per head.
    heads : int, optional
        Number of attention heads (default: 1).
    concat : bool, optional
        Concatenate heads if True, else average them (default: True).
    negative_slope : float, optional
        LeakyReLU negative slope (default: 0.2).
    dropout : float, optional
        Dropout rate on attention coefficients (default: 0.0).
    share_att : bool, optional
        If True, self and neighbour edges share one attention vector
        (recovers GATv2); if False, self-loops get a separate ``att2``
        (the GATE variant) (default: False).
    has_omega : bool, optional
        If True, apply the learned self/neighbour gate ``omega``
        (default: True).
    omega_init : float, optional
        Initial value for ``omega`` (default: 1.0).
    **kwargs
        Additional arguments forwarded to
        :class:`torch_geometric.nn.MessagePassing`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0.0,
        share_att=False,
        has_omega=True,
        omega_init=1.0,
        **kwargs,
    ):
        super().__init__(node_dim=0, aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_att = share_att
        self.has_omega = has_omega
        self.omega_init = omega_init

        h, c = heads, out_channels
        self.lin_l = Linear(in_channels, h * c)  # query / target transform
        self.lin_r = Linear(in_channels, h * c)  # key / source transform
        self.lin_s = Linear(in_channels, h * c)  # self-loop value transform
        self.att = Parameter(torch.empty(1, h, c))  # neighbour attention
        self.att2 = (
            self.att if share_att else Parameter(torch.empty(1, h, c))
        )  # self-loop attention
        if has_omega:
            self.omega = Parameter(torch.empty(h * c))
        self.reset_parameters()

    def reset_parameters(self):
        """Reset all learnable parameters."""
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        self.lin_s.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att)
        if not self.share_att:
            torch.nn.init.xavier_uniform_(self.att2)
        if self.has_omega:
            torch.nn.init.constant_(self.omega, self.omega_init)

    def forward(self, x, edge_index):
        """Compute one round of GATE attention.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(num_nodes, in_channels)``.
        edge_index : torch.Tensor
            Graph connectivity of shape ``(2, num_edges)``.

        Returns
        -------
        torch.Tensor
            Updated node features of shape
            ``(num_nodes, heads * out_channels)`` if ``concat`` else
            ``(num_nodes, out_channels)``.
        """
        h, c = self.heads, self.out_channels
        x_l = self.lin_l(x).view(-1, h, c)
        x_r = self.lin_r(x).view(-1, h, c)
        x_s = self.lin_s(x).view(-1, h, c)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        out = self.propagate(
            edge_index, x_l=x_l, x_r=x_r, x_s=x_s, ei=edge_index
        )
        return out.reshape(-1, h * c) if self.concat else out.mean(dim=1)

    def message(self, x_l_i, x_r_j, x_s_j, ei, index, ptr, size_i):
        """Build attention-weighted messages with the self/neighbour split.

        Parameters
        ----------
        x_l_i : torch.Tensor
            Target-node query features per edge.
        x_r_j : torch.Tensor
            Source-node key/value features per edge.
        x_s_j : torch.Tensor
            Source-node self-value features per edge (used on self-loops).
        ei : torch.Tensor
            Edge index (to identify self-loops).
        index : torch.Tensor
            Target index per edge, for softmax normalization.
        ptr : torch.Tensor
            Optional CSR pointer for softmax.
        size_i : int
            Number of target nodes.

        Returns
        -------
        torch.Tensor
            Messages of shape ``(num_edges, heads, out_channels)``.
        """
        x = F.leaky_relu(x_l_i + x_r_j, self.negative_slope)
        self_mask = (ei[0] == ei[1]).to(x.dtype).unsqueeze(-1)
        nbr_mask = 1.0 - self_mask

        alpha = (x * self.att).sum(-1) * nbr_mask + (x * self.att2).sum(
            -1
        ) * self_mask
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Neighbours contribute their source value; self-loops contribute
        # the dedicated self transform.
        value = x_r_j * nbr_mask.unsqueeze(-1) + x_s_j * self_mask.unsqueeze(
            -1
        )
        if self.has_omega:
            omega = self.omega.view(self.heads, self.out_channels)
            sm = self_mask.unsqueeze(-1)
            a = alpha.unsqueeze(-1)
            return value * (sm - omega * (sm - a))
        return value * alpha.unsqueeze(-1)

    def __repr__(self):
        """Return a string representation of the layer."""
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class GATE(torch.nn.Module):
    r"""Stacked GATE backbone.

    Stacks :class:`GATEConv` layers with ELU activations and dropout,
    returning node embeddings of dimension ``hidden_channels``; the
    TopoBench readout produces the task logits.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Hidden width; also the returned embedding dimension.
    num_layers : int, optional
        Number of GATE layers (default: 2).
    heads : int, optional
        Number of attention heads (default: 1).
    dropout : float, optional
        Dropout rate between layers (default: 0.0).
    share_att : bool, optional
        Whether self/neighbour edges share an attention vector
        (default: False).
    has_omega : bool, optional
        Whether to use the learned self/neighbour gate (default: True).
    **kwargs
        Additional arguments (ignored), kept for config compatibility.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        heads=1,
        dropout=0.0,
        share_att=False,
        has_omega=True,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.out_channels = hidden_channels
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            in_dim = in_channels if layer == 0 else hidden_channels
            self.convs.append(
                GATEConv(
                    in_dim,
                    hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    share_att=share_att,
                    has_omega=has_omega,
                )
            )

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        """Compute node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``(num_nodes, in_channels)``.
        edge_index : torch.Tensor
            Graph connectivity of shape ``(2, num_edges)``.
        edge_weight : torch.Tensor, optional
            Unused; accepted for wrapper compatibility (default: None).
        **kwargs
            Additional arguments (e.g. ``batch``) ignored by this backbone.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``(num_nodes, hidden_channels)``.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
