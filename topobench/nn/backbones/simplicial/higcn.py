"""Higher-order Graph Convolutional Network (HiGCN) backbone.

This module implements HiGCN, the Higher-order Graph Convolutional Network with
Flower-Petals (FP) Laplacians introduced in

    Yiming Huang, Yujie Zeng, Qiang Wu, Linyuan Lu.
    "Higher-order Graph Convolutional Network with Flower-Petals Laplacians on
    Simplicial Complexes." AAAI 2024. https://arxiv.org/abs/2309.12971

Reference implementation (node classification): ``node_classify/models/HiGCN_model.py``
and ``node_classify/utils/gen_HoHLaplacian.py`` at https://github.com/Yiminghh/HiGCN

Notes
-----
For a simplicial complex, HiGCN defines, for every *petal order* ``k`` (k=1 are
edges, k=2 are triangles/filled 2-simplices, ...), a **Flower-Petals Laplacian**
that lives on the nodes ("hubs"):

.. math::

    \\mathcal{L}_k = \\frac{1}{k+1}\\, \\mathbf{D}_k^{-1/2}\\, \\mathbf{H}_k\\,
                    \\mathbf{H}_k^{\\top}\\, \\mathbf{D}_k^{-1/2},

where :math:`\\mathbf{H}_k \\in \\{0,1\\}^{N \\times n_k}` is the node-to-k-simplex
incidence ("hub-petal" incidence) and :math:`\\mathbf{D}_k` is the diagonal matrix
of node degrees with respect to k-simplices (Eq. 4 of the paper).

Node features are then filtered, independently per order, by a learnable
polynomial (GPR/APPNP-style) of the corresponding FP Laplacian:

.. math::

    \\mathbf{Z}_k = \\sum_{j=0}^{K} w_j^{(k)}\\, \\mathcal{L}_k^{\\,j}\\,
                    \\mathbf{X} \\mathbf{W}_{\\text{in}}^{(k)},

and the per-order representations are concatenated and projected, yielding a
multi-scale node embedding that fuses pairwise (k=1) and higher-order (k>=2)
topology.

Within TopoBench the FP Laplacians are reconstructed on the fly from the
boundary/incidence matrices produced by the (clique) graph-to-simplicial
lifting, so the backbone needs no precomputed operators beyond the standard
``incidence_k`` tensors carried on the batch.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_sparse import SparseTensor, matmul, mul


def _to_sparse_tensor(coo: torch.Tensor) -> SparseTensor:
    """Convert a (possibly signed) sparse COO incidence matrix to a SparseTensor.

    The absolute value of the entries is taken so that the result encodes
    unsigned membership (a node/edge belongs to a simplex), as required by the
    Flower-Petals incidence :math:`\\mathbf{H}_k`.

    Parameters
    ----------
    coo : torch.Tensor
        Sparse COO tensor of shape ``(n_rows, n_cols)``.

    Returns
    -------
    SparseTensor
        Unsigned sparse tensor with the same shape.
    """
    coo = coo.coalesce()
    idx = coo.indices()
    val = coo.values().abs()
    return SparseTensor(
        row=idx[0],
        col=idx[1],
        value=val,
        sparse_sizes=tuple(coo.shape),
    )


def _binarize(adj: SparseTensor) -> SparseTensor:
    """Set every stored entry of a SparseTensor to one (membership indicator).

    Parameters
    ----------
    adj : SparseTensor
        Sparse tensor whose nonzero pattern encodes membership.

    Returns
    -------
    SparseTensor
        Sparse tensor with identical sparsity pattern and unit values.
    """
    row, col, val = adj.coo()
    return SparseTensor(
        row=row,
        col=col,
        value=torch.ones_like(val),
        sparse_sizes=adj.sparse_sizes(),
    )


def build_flower_petals_laplacians(
    incidences: tuple[torch.Tensor, ...],
    order: int,
) -> list[SparseTensor]:
    """Build the Flower-Petals Laplacians for orders ``1..order``.

    The node-to-k-simplex incidence :math:`\\mathbf{H}_k` is obtained by chaining
    the boundary matrices: :math:`\\mathbf{H}_1 = |\\mathbf{B}_1|` (node-edge) and
    :math:`\\mathbf{H}_k = \\mathbb{1}[\\mathbf{H}_{k-1} |\\mathbf{B}_k| > 0]`
    (node-to-k-simplex membership). Each FP Laplacian is then the symmetrically
    normalized hub-petal operator :math:`\\tfrac{1}{k+1}
    \\mathbf{D}_k^{-1/2}\\mathbf{H}_k\\mathbf{H}_k^{\\top}\\mathbf{D}_k^{-1/2}`.

    Parameters
    ----------
    incidences : tuple of torch.Tensor
        Sparse incidence/boundary matrices ``(B_1, B_2, ...)`` where ``B_k`` maps
        (k-1)-simplices to k-simplices. ``B_1`` has shape ``(n_nodes, n_edges)``.
    order : int
        Highest petal order to build. Must satisfy ``1 <= order <= len(incidences)``.

    Returns
    -------
    list of SparseTensor
        ``[L_1, ..., L_order]``, each an ``(n_nodes, n_nodes)`` node operator.
    """
    assert order >= 1, "order must be a positive integer."
    assert order <= len(incidences), (
        f"order={order} requires {order} incidence matrices, "
        f"but only {len(incidences)} were provided."
    )

    # H_1 is the unsigned node-edge incidence.
    hub_petal = _to_sparse_tensor(incidences[0])
    laplacians: list[SparseTensor] = [_normalize_fp(hub_petal, order=1)]

    for k in range(2, order + 1):
        boundary_k = _to_sparse_tensor(incidences[k - 1])
        # node -> k-simplex membership counts, then binarized to {0, 1}.
        hub_petal = _binarize(matmul(hub_petal, boundary_k))
        laplacians.append(_normalize_fp(hub_petal, order=k))

    return laplacians


def _normalize_fp(hub_petal: SparseTensor, order: int) -> SparseTensor:
    """Symmetrically normalize a hub-petal incidence into an FP Laplacian.

    Parameters
    ----------
    hub_petal : SparseTensor
        Node-to-k-simplex incidence :math:`\\mathbf{H}_k`.
    order : int
        Petal order ``k`` (used for the ``1 / (k + 1)`` scaling).

    Returns
    -------
    SparseTensor
        The ``(n_nodes, n_nodes)`` Flower-Petals Laplacian for this order.
    """
    deg = hub_petal.sum(dim=1)
    lap = matmul(hub_petal, hub_petal.t())
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
    lap = mul(lap, deg_inv_sqrt.view(1, -1))
    lap = mul(lap, deg_inv_sqrt.view(-1, 1) / (order + 1))
    return lap


class HiGCNProp(nn.Module):
    """Flower-Petals polynomial propagation for a single petal order.

    Implements the GPR/APPNP-style learnable polynomial filter
    :math:`\\sum_{j=0}^{K} w_j \\mathcal{L}^{\\,j} \\mathbf{x}` over a fixed FP
    Laplacian, with the coefficients ``w_j`` initialized to the Personalized
    PageRank weights :math:`\\alpha (1-\\alpha)^j` (Eq. 6 of the paper).

    Parameters
    ----------
    K : int
        Number of propagation hops (polynomial degree).
    alpha : float
        Teleport probability used to initialize the filter coefficients.
    """

    def __init__(self, K: int, alpha: float):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.filter_weights = nn.Parameter(torch.empty(K + 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the filter weights with the PPR profile."""
        with torch.no_grad():
            for k in range(self.K + 1):
                self.filter_weights[k] = self.alpha * (1 - self.alpha) ** k
            self.filter_weights[-1] = (1 - self.alpha) ** self.K

    def forward(
        self, x: torch.Tensor, laplacian: SparseTensor
    ) -> torch.Tensor:
        """Apply the polynomial FP filter to node features.

        Parameters
        ----------
        x : torch.Tensor
            Node feature tensor of shape ``(n_nodes, channels)``.
        laplacian : SparseTensor
            Flower-Petals Laplacian for this order, shape ``(n_nodes, n_nodes)``.

        Returns
        -------
        torch.Tensor
            Filtered node features of shape ``(n_nodes, channels)``.
        """
        hidden = x * self.filter_weights[0]
        for k in range(self.K):
            x = matmul(laplacian, x)
            hidden = hidden + self.filter_weights[k + 1] * x
        return hidden

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K}, alpha={self.alpha})"


class HiGCN(nn.Module):
    """Higher-order Graph Convolutional Network (HiGCN) backbone.

    For each petal order ``1..order`` the encoded node features are linearly
    projected, filtered by an order-specific Flower-Petals polynomial
    (:class:`HiGCNProp`), and the resulting multi-scale node representations are
    concatenated and projected back to ``hidden_channels``. The output is a node
    embedding that the TopoBench readout maps to task logits.

    Parameters
    ----------
    in_channels : int
        Dimension of the (encoded) input node features.
    hidden_channels : int
        Width of every per-order branch and of the returned node embedding.
    K : int, optional
        Number of propagation hops in each polynomial filter (default: 10).
    alpha : float, optional
        Teleport probability for PPR filter initialization (default: 0.1).
    order : int, optional
        Highest petal order to use; ``order=2`` fuses edges and triangles
        (default: 2).
    dropout : float, optional
        Dropout applied to the input features and the fused embedding
        (default: 0.5).
    dprate : float, optional
        Dropout applied between projection and propagation in each branch
        (default: 0.0).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        K: int = 10,
        alpha: float = 0.1,
        order: int = 2,
        dropout: float = 0.5,
        dprate: float = 0.0,
    ):
        super().__init__()
        assert order >= 1, "order must be a positive integer."
        assert K >= 1, "K (propagation hops) must be a positive integer."

        self.order = order
        self.dropout = dropout
        self.dprate = dprate

        self.lin_in = nn.ModuleList(
            [nn.Linear(in_channels, hidden_channels) for _ in range(order)]
        )
        self.prop = nn.ModuleList([HiGCNProp(K, alpha) for _ in range(order)])
        self.lin_out = nn.Linear(hidden_channels * order, hidden_channels)

    def reset_parameters(self) -> None:
        """Reset all learnable parameters of the backbone."""
        for layer in self.lin_in:
            layer.reset_parameters()
        for prop in self.prop:
            prop.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        incidences: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Encoded node features of shape ``(n_nodes, in_channels)``.
        incidences : tuple of torch.Tensor
            Sparse incidence matrices ``(incidence_1, incidence_2, ...)`` with at
            least ``order`` entries, used to build the FP Laplacians.

        Returns
        -------
        torch.Tensor
            Node embedding of shape ``(n_nodes, hidden_channels)``.
        """
        laplacians = build_flower_petals_laplacians(incidences, self.order)

        branches = []
        for i in range(self.order):
            h = F.dropout(x, p=self.dropout, training=self.training)
            h = self.lin_in[i](h)
            if self.dprate > 0.0:
                h = F.dropout(h, p=self.dprate, training=self.training)
            h = self.prop[i](h, laplacians[i])
            branches.append(h)

        h = torch.cat(branches, dim=1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin_out(h)
        return h
