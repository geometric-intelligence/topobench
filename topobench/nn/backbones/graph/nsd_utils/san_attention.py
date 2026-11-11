"""
GAT-style attention over the augmented sheaf edge set.

Sheaf Attention Networks (Barbero et al., 2022) build a row-stochastic
attention matrix Lambda whose entries are given by equation (2) of the
paper::

    Lambda_ij = a(x_i, x_j)
              = exp(LeakyReLU(a [W x_i || W x_j]))
                / sum_{k in N_i} exp(LeakyReLU(a [W x_i || W x_k]))

The paper states explicitly that "this is the same attention mechanism
utilised in GAT", i.e. the original GAT (Velickovic et al., 2018)
coefficient, and that is what ``attention_variant='gat'`` implements.

The extended version of this work (Barbero, "Attention-based Sheaf
Neural Networks", MLMI dissertation, 2022, section 3.1) notes instead
that "in our experiments we implement the GATv2 attention mechanism as
it is provably more powerful and empirically outperforms the GAT
attention mechanism". That setting is available as
``attention_variant='gatv2'``, which moves the LeakyReLU inside the
scoring function (Brody et al., 2022)::

    Lambda_ij ~ exp(a^T LeakyReLU(W [x_i || x_j]))

Both variants normalize over ``k in N_i``, so each row of Lambda is a
probability distribution over the neighbours aggregated into node i.
The score-vector parameter is split into a source half and a target
half to avoid materializing the concatenation of W x_i with W x_j.

Multi-head attention is computed by stacking H projection heads and
averaging the resulting per-edge probabilities, following the
dissertation ("the multiple attention heads are then aggregated by
taking their mean"). The self-loop edges appended to the edge index
participate in the same softmax, matching the "added self-loops" of the
sheaf adjacency A_F-hat in equations (3) and (4), so each node attends
to itself with a learned coefficient.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_softmax

ATTENTION_VARIANTS = ("gat", "gatv2")


class SheafGATAttention(nn.Module):
    """
    Multi-head GAT attention producing one scalar per directed edge.

    Implements equation (2) of Barbero et al. (2022) with a selectable
    scoring function; see the module docstring for why both variants
    exist.

    Parameters
    ----------
    in_channels : int
        Dimension of the per-node feature vectors at the layer input.
    num_heads : int, optional
        Number of attention heads. Default is 1.
    head_dim : int or None, optional
        Per-head projection dimension. If None, falls back to
        max(1, in_channels // num_heads) (floor division with min of 1).
    negative_slope : float, optional
        Slope of the LeakyReLU non-linearity. Default is 0.2.
    attention_variant : str, optional
        Either ``'gat'`` (equation (2) of the paper, LeakyReLU applied
        after the score projection, one shared W) or ``'gatv2'`` (the
        dissertation's experimental setting, LeakyReLU applied before
        the score projection, separate source/target W). Default is
        ``'gat'``.

    Raises
    ------
    ValueError
        If ``attention_variant`` is not one of ``'gat'``, ``'gatv2'``.
    """

    def __init__(
        self,
        in_channels,
        num_heads=1,
        head_dim=None,
        negative_slope=0.2,
        attention_variant="gat",
    ):
        super().__init__()
        assert num_heads >= 1
        if attention_variant not in ATTENTION_VARIANTS:
            raise ValueError(
                f"Unknown attention variant: {attention_variant}. "
                f"Expected one of {ATTENTION_VARIANTS}."
            )
        if head_dim is None:
            head_dim = max(1, in_channels // num_heads)
        assert head_dim >= 1

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.negative_slope = negative_slope
        self.attention_variant = attention_variant

        out_channels = num_heads * head_dim
        # GAT shares one W across both endpoints; GATv2 learns a
        # separate target projection so that the pre-activation score
        # is a genuine function of both endpoints.
        self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_tgt = (
            nn.Linear(in_channels, out_channels, bias=False)
            if attention_variant == "gatv2"
            else None
        )
        self.att_src = nn.Parameter(torch.empty(1, num_heads, head_dim))
        self.att_tgt = (
            None
            if attention_variant == "gatv2"
            else nn.Parameter(torch.empty(1, num_heads, head_dim))
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialize the projection and attention parameters."""
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.att_src)
        if self.lin_tgt is not None:
            nn.init.xavier_uniform_(self.lin_tgt.weight)
        if self.att_tgt is not None:
            nn.init.xavier_uniform_(self.att_tgt)

    def forward(self, x, edge_index):
        """
        Compute per-edge attention coefficients.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, in_channels].
        edge_index : torch.Tensor
            Augmented directed edge index of shape [2, num_edges]. The
            caller is responsible for appending self-loops if needed.

        Returns
        -------
        torch.Tensor
            Attention scalars of shape [num_edges], averaged across
            heads and row-stochastic over the neighbourhood N_i of each
            row node i (the source entry of the edge index).
        """
        src, tgt = edge_index
        n = x.size(0)

        if self.attention_variant == "gatv2":
            z_src = self.lin_src(x).view(n, self.num_heads, self.head_dim)
            z_tgt = self.lin_tgt(x).view(n, self.num_heads, self.head_dim)
            # a^T LeakyReLU(W [x_i || x_j]): non-linearity first.
            pre = F.leaky_relu(z_src[src] + z_tgt[tgt], self.negative_slope)
            scores = (pre * self.att_src).sum(dim=-1)
        else:
            z = self.lin_src(x).view(n, self.num_heads, self.head_dim)
            # LeakyReLU(a [W x_i || W x_j]): projection first.
            alpha_src = (z * self.att_src).sum(dim=-1)
            alpha_tgt = (z * self.att_tgt).sum(dim=-1)
            scores = F.leaky_relu(
                alpha_src[src] + alpha_tgt[tgt], self.negative_slope
            )

        # Row-stochastic normalization over k in N_i (equation (2)).
        alpha = scatter_softmax(scores, src, dim=0)
        return alpha.mean(dim=-1)
