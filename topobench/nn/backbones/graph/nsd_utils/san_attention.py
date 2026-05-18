"""
GAT-style attention over the augmented sheaf edge set.

The Sheaf Attention Network (Barbero et al., 2022, equation 2) reuses
the standard GAT attention coefficient::

    a(x_i, x_j) = softmax_{k in N_i}(
        LeakyReLU(a^T [W x_i || W x_j])
    )

so that each row of the resulting Lambda matrix is a probability
distribution over a node's outgoing edges. The score-vector parameter
a is split into a source half and a target half to avoid materializing
the concatenation of W x_i with W x_j.

Multi-head attention is computed by stacking H projection heads and
averaging the resulting per-edge probabilities. The self-loop edges
appended to the edge index participate in the same softmax, so each
node attends to itself with a learned coefficient.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_softmax


class SheafGATAttention(nn.Module):
    """
    Multi-head GAT attention producing one scalar per directed edge.

    Parameters
    ----------
    in_channels : int
        Dimension of the per-node feature vectors at the layer input.
    num_heads : int, optional
        Number of attention heads. Default is 1.
    head_dim : int or None, optional
        Per-head projection dimension. If None, falls back to
        in_channels // num_heads and requires divisibility.
    negative_slope : float, optional
        Slope of the LeakyReLU non-linearity. Default is 0.2.
    """

    def __init__(
        self,
        in_channels,
        num_heads=1,
        head_dim=None,
        negative_slope=0.2,
    ):
        super().__init__()
        assert num_heads >= 1
        if head_dim is None:
            head_dim = max(1, in_channels // num_heads)
        assert head_dim >= 1

        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_channels, num_heads * head_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, num_heads, head_dim))
        self.att_tgt = nn.Parameter(torch.empty(1, num_heads, head_dim))
        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialize the projection and attention parameters."""
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
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
            heads and row-stochastic per source node.
        """
        src, tgt = edge_index
        n = x.size(0)

        z = self.lin(x).view(n, self.num_heads, self.head_dim)

        alpha_src = (z * self.att_src).sum(dim=-1)
        alpha_tgt = (z * self.att_tgt).sum(dim=-1)
        # Per-edge raw score, shape [num_edges, num_heads].
        scores = alpha_src[src] + alpha_tgt[tgt]
        scores = F.leaky_relu(scores, self.negative_slope)
        # Row-stochastic normalization within each source bucket.
        alpha = scatter_softmax(scores, src, dim=0)
        return alpha.mean(dim=-1)
