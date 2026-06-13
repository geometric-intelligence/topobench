"""GATE backbone: graph attention that separates self from neighbours.

GATE extends GATv2 with a single change (paper Eq. 4): the attention
logit uses a *separate* learnable vector for self-loops (``a_t``) versus
genuine neighbours (``a_s``). This lets a node parameterize its own
self-attention independently of its neighbours, so it can suppress
aggregation from unrelated ("intrusive") neighbours -- which is what
makes the model robust on heterophilic graphs. Apart from the extra
attention vector, GATE adds no parameters over GATv2: the value and the
source/target transforms are the same as GATv2.

Notes
-----
Faithful to the paper's GATE (Eq. 1, 2, 4), not to the optional knobs in
the reference repo: the repo's ``omega`` gate and separate self-loop value
transform are *not* part of the published model (the paper adds only the
``d``-dimensional ``a_t`` vector and never uses ``omega``), so they are
intentionally omitted here. Initialization follows the paper: attention
vectors zero (no initial inductive bias, Thm. 4.3) and weight matrices
random-orthogonal. (The paper's full "looks-linear" channel-mirroring
construction is not applied; orthogonal init captures the random-orthogonal
specification, and the zero attention init provides the Thm. 4.3
aggregation property.) The base mechanism is validated against PyG's
``GATv2Conv`` in ``test/nn/backbones/graph/test_gate.py``.

References
----------
.. [1] Nimrah Mustafa, Rebekka Burkholz. "GATE: How to Keep Out
   Intrusive Neighbors." ICML 2024. https://arxiv.org/abs/2406.00418
   Official code: https://github.com/RelationalML/GATE
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax


class GATEConv(MessagePassing):
    r"""Single GATE attention layer (Mustafa & Burkholz, 2024).

    Implements the GATE update. For an edge :math:`(u \to v)` the logit is
    (Eq. 4)

    .. math::
        e_{uv} = (\mathbb{1}_{u \neq v}\, a_s
                  + \mathbb{1}_{u = v}\, a_t)^\top
                 \,\phi(U h_u + V h_v),

    i.e. a GATv2-style additive score whose attention vector switches
    between ``a_s`` (neighbours) and ``a_t`` (self-loops). Coefficients are
    normalized by softmax over the neighbourhood (Eq. 2) and aggregated as
    :math:`h_v = \sum_u \alpha_{uv} U h_u` (Eq. 1) -- the value transform is
    the same source matrix :math:`U` for every edge, self-loops included.

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
        LeakyReLU negative slope for :math:`\phi` (default: 0.2).
    dropout : float, optional
        Dropout rate on attention coefficients (default: 0.0).
    share_att : bool, optional
        If True, tie ``a_s`` and ``a_t`` into one vector (the paper's
        weight-sharing "GATES" variant / standard GATv2); if False, use the
        separate self/neighbour vectors of GATE (default: False).
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

        h, c = heads, out_channels
        self.lin_l = Linear(in_channels, h * c)  # V: target transform
        self.lin_r = Linear(in_channels, h * c)  # U: source transform & value
        self.att = Parameter(torch.empty(1, h, c))  # a_s (neighbours)
        self.att2 = (
            self.att if share_att else Parameter(torch.empty(1, h, c))
        )  # a_t (self-loops)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters following the paper's initialization scheme.

        Weight matrices use random orthogonal initialization; attention
        vectors and biases start at zero (zero ``a`` gives no initial
        inductive bias, Thm. 4.3).
        """
        torch.nn.init.orthogonal_(self.lin_l.weight)
        torch.nn.init.orthogonal_(self.lin_r.weight)
        if self.lin_l.bias is not None:
            torch.nn.init.zeros_(self.lin_l.bias)
            torch.nn.init.zeros_(self.lin_r.bias)
        torch.nn.init.zeros_(self.att)
        if not self.share_att:
            torch.nn.init.zeros_(self.att2)

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

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        out = self.propagate(edge_index, x_l=x_l, x_r=x_r, ei=edge_index)
        return out.reshape(-1, h * c) if self.concat else out.mean(dim=1)

    def message(self, x_l_i, x_r_j, ei, index, ptr, size_i):
        """Build attention-weighted messages (Eq. 4 logit, Eq. 1 value).

        Parameters
        ----------
        x_l_i : torch.Tensor
            Target-node transform :math:`V h_v` per edge.
        x_r_j : torch.Tensor
            Source-node transform :math:`U h_u` per edge (also the value).
        ei : torch.Tensor
            Edge index, used to flag self-loops (:math:`u = v`).
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
        # phi(U h_u + V h_v), then the self/neighbour attention split.
        pre = F.leaky_relu(x_l_i + x_r_j, self.negative_slope)
        self_mask = (ei[0] == ei[1]).to(pre.dtype).unsqueeze(-1)
        nbr_mask = 1.0 - self_mask
        alpha = (pre * self.att).sum(-1) * nbr_mask + (pre * self.att2).sum(
            -1
        ) * self_mask
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        # Value is U h_u for every edge (self-loops use U h_v with u = v).
        return x_r_j * alpha.unsqueeze(-1)

    def __repr__(self):
        """Return a string representation of the layer."""
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class GATE(torch.nn.Module):
    r"""Stacked GATE backbone.

    Stacks :class:`GATEConv` layers with a homogeneous activation
    (LeakyReLU, consistent with the paper's :math:`\phi`) and dropout
    between layers, returning node embeddings of dimension
    ``hidden_channels``; the TopoBench readout produces the task logits.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int
        Hidden width; also the returned embedding dimension. Must be
        divisible by ``heads``.
    num_layers : int, optional
        Number of GATE layers (default: 2).
    heads : int, optional
        Number of attention heads (default: 1).
    dropout : float, optional
        Dropout rate between layers (default: 0.0).
    share_att : bool, optional
        Whether self/neighbour edges share one attention vector
        (default: False).
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
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.negative_slope = 0.2
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
                x = F.leaky_relu(x, self.negative_slope)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
