"""Cellular Transformer (CT) backbone.

Paper: Ballester, Hernandez-Garcia, Papillon, Battiloro, Miolane,
Birdal, Casacuberta, Escalera, Hajij,
"Attending to Topological Spaces: The Cellular Transformer", 2024.
https://arxiv.org/abs/2405.14094

There is no public reference implementation; this module implements the
architecture from the paper's equations (sparse pairwise cellular
attention, Eq. (2); prenorm layer structure, Appendix A; ConcatPE
preprocessing, Eq. (4); random walk positional encodings, Appendix C.1).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import softmax

# Pairwise attention routes (source rank -> target rank) used by the
# Cellular Transformer, Section 4.2.1: within-rank attention uses the
# upper adjacency for rank 0 and the lower adjacency for ranks > 0;
# across ranks, attention follows the (non-signed) incidence relations
# between adjacent ranks (the block structure of Eq. (3)).
CT_ROUTES = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
]


def sparse_row_normalize(matrix):
    r"""Row-normalize a sparse matrix into a random walk operator.

    Computes :math:`P = D^{-1} A` for a sparse non-negative matrix
    :math:`A`, where :math:`D` is the diagonal matrix of row sums; rows
    with zero sum are left as zero rows.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse COO tensor of shape [n, n].

    Returns
    -------
    torch.Tensor
        Sparse COO tensor of shape [n, n] with rows summing to 1 (or 0).
    """
    matrix = matrix.coalesce()
    n = matrix.size(0)
    if matrix._nnz() == 0:
        return matrix
    indices = matrix.indices()
    values = matrix.values().abs()
    row_sum = torch.zeros(n, device=values.device, dtype=values.dtype)
    row_sum.index_add_(0, indices[0], values)
    inv = torch.where(
        row_sum > 0, 1.0 / row_sum.clamp(min=1e-12), torch.zeros_like(row_sum)
    )
    return torch.sparse_coo_tensor(
        indices, values * inv[indices[0]], matrix.shape
    ).coalesce()


def random_walk_pe(matrix, steps):
    r"""Random walk positional encoding on a single rank (RWPe).

    Implements the local random walk encoding of the Cellular
    Transformer (Appendix C.1): for the random walk operator
    :math:`P = D^{-1} A` of the within-rank neighborhood matrix, the
    encoding of cell :math:`i` collects the return probabilities

    .. math::
        \mathrm{pe}_i = [P_{ii}, (P^2)_{ii}, \dots, (P^k)_{ii}].

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse COO within-rank neighborhood matrix of shape [n, n]
        (upper adjacency for rank 0, lower adjacency for ranks > 0).
    steps : int
        Number of random walk steps :math:`k`.

    Returns
    -------
    torch.Tensor
        Dense tensor of shape [n, steps] with return probabilities.
    """
    n = matrix.size(0)
    pe = torch.zeros(
        n, steps, device=matrix.device, dtype=torch.get_default_dtype()
    )
    if n == 0:
        return pe
    walk = sparse_row_normalize(matrix)
    if walk._nnz() == 0:
        return pe
    power = walk
    for step in range(steps):
        if step > 0:
            power = torch.sparse.mm(power, walk).coalesce()
        indices = power.indices()
        diag_mask = indices[0] == indices[1]
        pe[:, step].index_add_(
            0, indices[0][diag_mask], power.values()[diag_mask]
        )
    return pe


class SparseCellAttention(nn.Module):
    r"""Sparse pairwise cellular attention head block.

    Implements the sparse variant (:math:`\cdot = s`) of the pairwise
    cellular attention of the Cellular Transformer, Eq. (2) of
    `Ballester et al., 2024 <https://arxiv.org/abs/2405.14094>`_:

    .. math::
        \mathcal{A}^{s}_{k_s \to k_t}(X_{k_t}, X_{k_s}) =
        \mathrm{softmax}\left( X_{k_t} Q (X_{k_s} K)^\top \odot
        N_{k_s \to k_t} \right) X_{k_s} V,

    where the Hadamard product with the (binary) neighborhood matrix
    :math:`N_{k_s \to k_t}` restricts attention to neighboring cells:
    the softmax is computed over the support of :math:`N` only (masked
    softmax), which is the standard realization of sparse attention and
    keeps attention block-diagonal across batched complexes. Scores are
    scaled by :math:`1/\sqrt{p}` (head dimension :math:`p`) following
    standard transformer convention.

    Multi-head attention uses ``num_heads`` independent projections of
    head dimension ``hidden_channels // num_heads``; head outputs are
    concatenated, matching the value projection shape
    :math:`(d^h_s, d^h_t)` of the paper.

    Parameters
    ----------
    hidden_channels : int
        Hidden dimension :math:`d^h` (shared by all ranks).
    num_heads : int, optional
        Number of attention heads (default: 4).
    att_dropout : float, optional
        Dropout on attention coefficients (default: 0.0).
    """

    def __init__(self, hidden_channels, num_heads=4, att_dropout=0.0):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by num_heads, got "
                f"{hidden_channels} and {num_heads}."
            )
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.q = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.k = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.v = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.att_dropout = nn.Dropout(att_dropout)

    def forward(self, x_target, x_source, neighborhood):
        """Compute neighborhood-masked attention from source to target.

        Parameters
        ----------
        x_target : torch.Tensor
            Target-rank cell features of shape [n_target, hidden].
        x_source : torch.Tensor
            Source-rank cell features of shape [n_source, hidden].
        neighborhood : torch.Tensor
            Sparse COO neighborhood matrix of shape
            [n_target, n_source]; attention is restricted to its
            support.

        Returns
        -------
        torch.Tensor
            Attention output of shape [n_target, hidden].
        """
        n_target = x_target.size(0)
        out = torch.zeros_like(x_target)
        neighborhood = neighborhood.coalesce()
        if n_target == 0 or x_source.size(0) == 0:
            return out
        if neighborhood._nnz() == 0:
            return out

        target_idx, source_idx = neighborhood.indices()
        heads, dim = self.num_heads, self.head_dim
        queries = self.q(x_target).view(n_target, heads, dim)
        keys = self.k(x_source).view(-1, heads, dim)
        values = self.v(x_source).view(-1, heads, dim)

        scores = (queries[target_idx] * keys[source_idx]).sum(
            dim=-1
        ) / math.sqrt(dim)
        alpha = softmax(scores, target_idx, num_nodes=n_target)
        alpha = self.att_dropout(alpha)

        weighted = alpha.unsqueeze(-1) * values[source_idx]
        out = out.view(n_target, heads, dim).index_add(0, target_idx, weighted)
        return out.view(n_target, heads * dim)


class CTFeedForward(nn.Module):
    """Position-wise feed-forward block of the Cellular Transformer.

    Follows Appendix A of `Ballester et al., 2024
    <https://arxiv.org/abs/2405.14094>`_:
    ``FFN2(Dropout(ReLU(FFN1(x))))`` followed by dropout.

    Parameters
    ----------
    hidden_channels : int
        Input/output dimension.
    ffn_channels : int
        Inner dimension of the feed-forward block (the paper uses
        ``ffn_channels == hidden_channels``).
    dropout : float, optional
        Dropout rate (default: 0.0).
    """

    def __init__(self, hidden_channels, ffn_channels, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(hidden_channels, ffn_channels)
        self.lin2 = nn.Linear(ffn_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [n, hidden_channels].

        Returns
        -------
        torch.Tensor
            Output tensor of shape [n, hidden_channels].
        """
        x = self.dropout(F.relu(self.lin1(x)))
        return self.dropout(self.lin2(x))


class CellularTransformerLayer(nn.Module):
    r"""One prenorm layer of the Cellular Transformer.

    Implements the layer structure of Appendix A of
    `Ballester et al., 2024 <https://arxiv.org/abs/2405.14094>`_ for
    pairwise attention. For every target rank :math:`k_t`:

    .. math::
        X^1_{k} &= \mathrm{LN}_k(X_{k}) \\
        X^2_{k_s \to k_t} &= \mathcal{A}^{s}_{k_s \to k_t}
            (X^1_{k_t}, X^1_{k_s}) \\
        X^4_{k_t} &= X_{k_t} + \sum_{k_s}
            \mathrm{Dropout}(X^2_{k_s \to k_t}) \\
        X_{k_t}^{\mathrm{out}} &= X^4_{k_t} +
            \mathrm{FFN}(\mathrm{LN}(X^4_{k_t})).

    Parameters
    ----------
    hidden_channels : int
        Hidden dimension shared by all ranks.
    num_heads : int, optional
        Number of attention heads (default: 4).
    dropout : float, optional
        Dropout rate for attention outputs and FFN (default: 0.0).
    att_dropout : float, optional
        Dropout on attention coefficients (default: 0.0).
    """

    def __init__(
        self,
        hidden_channels,
        num_heads=4,
        dropout=0.0,
        att_dropout=0.0,
    ):
        super().__init__()
        self.pre_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(3)]
        )
        self.attentions = nn.ModuleDict(
            {
                f"{ks}_{kt}": SparseCellAttention(
                    hidden_channels,
                    num_heads=num_heads,
                    att_dropout=att_dropout,
                )
                for ks, kt in CT_ROUTES
            }
        )
        self.ffn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(3)]
        )
        self.ffns = nn.ModuleList(
            [
                CTFeedForward(
                    hidden_channels, hidden_channels, dropout=dropout
                )
                for _ in range(3)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, xs, neighborhoods):
        """Forward pass over all ranks.

        Parameters
        ----------
        xs : list of torch.Tensor
            Cell features per rank, ``[x_0, x_1, x_2]``.
        neighborhoods : dict
            Mapping ``"{source}_{target}"`` to the sparse neighborhood
            matrix of shape [n_target, n_source] for every route in
            ``CT_ROUTES``.

        Returns
        -------
        list of torch.Tensor
            Updated cell features per rank.
        """
        normed = [norm(x) for norm, x in zip(self.pre_norms, xs, strict=True)]
        residual = list(xs)
        for ks, kt in CT_ROUTES:
            attention = self.attentions[f"{ks}_{kt}"]
            contribution = attention(
                normed[kt], normed[ks], neighborhoods[f"{ks}_{kt}"]
            )
            residual[kt] = residual[kt] + self.dropout(contribution)
        return [
            x4 + ffn(norm(x4))
            for x4, norm, ffn in zip(
                residual, self.ffn_norms, self.ffns, strict=True
            )
        ]


class CellularTransformer(nn.Module):
    r"""Cellular Transformer (CT) backbone.

    Transformer for cell complexes from `"Attending to Topological
    Spaces: The Cellular Transformer" (Ballester et al., 2024)
    <https://arxiv.org/abs/2405.14094>`_. Cells of ranks 0 (nodes),
    1 (edges) and 2 (faces) exchange information through pairwise
    cellular attention (Eq. (2)) restricted to the neighborhood
    structure of the complex (Section 4.2.1): upper adjacency within
    rank 0, lower adjacency within ranks 1 and 2, and non-signed
    incidence across adjacent ranks (Eq. (3)).

    This implementation provides the *sparse* pairwise attention
    variant :math:`\mathcal{A}^s` (attention restricted to neighboring
    cells), which scales to batched complexes; the dense variants of
    the paper attend over all cell pairs and are not tractable under
    TopoBench's block-diagonal batching (modification allowed by the
    challenge rules to respect computational requirements).

    Input features are combined with positional encodings via ConcatPE
    (Eq. (4)): per rank, features and encodings are concatenated and
    processed jointly by a single linear projection ("shared weights"
    in the sense of Eq. (4), as opposed to SumPE's separate feature and
    encoding projections; one projection per rank since cochain spaces
    are rank-specific). Supported encodings: the local random walk
    encoding RWPe (Appendix C.1) and "zero" (no positional
    information). RWPe is computed on the fly from the within-rank
    neighborhood matrices under ``torch.no_grad()`` — a handful of
    sparse matrix products per batch — which keeps the preprocessing
    pipeline transform-free and the cached datasets lifting-only.

    Pooling/decoding is delegated to the TopoBench readout, matching
    the paper's global add pooling over rank-0 signals followed by an
    MLP head (Section 5 of the experimental setup).

    Parameters
    ----------
    in_channels : int
        Number of input features per cell (after the TopoBench feature
        encoder; shared by all ranks).
    hidden_channels : int
        Hidden dimension :math:`d^h` of the transformer.
    num_layers : int, optional
        Number of transformer layers (default: 4).
    num_heads : int, optional
        Number of attention heads (default: 4).
    dropout : float, optional
        Dropout rate for attention outputs and FFN (default: 0.0).
    att_dropout : float, optional
        Dropout on attention coefficients (default: 0.1).
    pe_type : str, optional
        Positional encoding type, "rwpe" or "zero" (default: "rwpe").
    pe_steps : int, optional
        Number of random walk steps of RWPe (default: 8).
    **kwargs
        Additional (ignored) keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=4,
        num_heads=4,
        dropout=0.0,
        att_dropout=0.1,
        pe_type="rwpe",
        pe_steps=8,
        **kwargs,
    ):
        super().__init__()
        if pe_type not in ("rwpe", "zero"):
            raise ValueError(
                f"Positional encoding '{pe_type}' is not supported."
            )
        self.out_channels = hidden_channels
        self.pe_type = pe_type
        self.pe_steps = pe_steps

        pe_dim = pe_steps if pe_type == "rwpe" else 0
        # ConcatPE preprocessing, Eq. (4): concatenate features and
        # positional encodings, then project with shared weights.
        self.preprocess = nn.ModuleList(
            [
                nn.Linear(in_channels + pe_dim, hidden_channels)
                for _ in range(3)
            ]
        )
        self.layers = nn.ModuleList(
            [
                CellularTransformerLayer(
                    hidden_channels,
                    num_heads=num_heads,
                    dropout=dropout,
                    att_dropout=att_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        adjacency_0,
        coadjacency_1,
        coadjacency_2,
        incidence_1,
        incidence_2,
        rwpe_0=None,
        rwpe_1=None,
        rwpe_2=None,
    ):
        """Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node (rank 0) features of shape [n_0, in_channels].
        x_1 : torch.Tensor
            Edge (rank 1) features of shape [n_1, in_channels].
        x_2 : torch.Tensor
            Face (rank 2) features of shape [n_2, in_channels].
        adjacency_0 : torch.Tensor
            Sparse upper adjacency of rank 0, shape [n_0, n_0].
        coadjacency_1 : torch.Tensor
            Sparse lower adjacency of rank 1, shape [n_1, n_1].
        coadjacency_2 : torch.Tensor
            Sparse lower adjacency of rank 2, shape [n_2, n_2].
        incidence_1 : torch.Tensor
            Sparse incidence between ranks 0 and 1, shape [n_0, n_1].
        incidence_2 : torch.Tensor
            Sparse incidence between ranks 1 and 2, shape [n_1, n_2].
        rwpe_0 : torch.Tensor, optional
            Precomputed random-walk PE for rank 0, shape [n_0, pe_steps].
        rwpe_1 : torch.Tensor, optional
            Precomputed random-walk PE for rank 1, shape [n_1, pe_steps].
        rwpe_2 : torch.Tensor, optional
            Precomputed random-walk PE for rank 2, shape [n_2, pe_steps].

        Returns
        -------
        tuple of torch.Tensor
            Updated features ``(x_0, x_1, x_2)`` with
            ``hidden_channels`` channels each.
        """
        neighborhoods = {
            "0_0": adjacency_0,
            "1_1": coadjacency_1,
            "2_2": coadjacency_2,
            "1_0": incidence_1,
            "0_1": incidence_1.transpose(0, 1),
            "2_1": incidence_2,
            "1_2": incidence_2.transpose(0, 1),
        }

        xs = [x_0, x_1, x_2]
        if self.pe_type == "rwpe":
            if rwpe_0 is not None:
                # Fast path: precomputed by CellRandomWalkPE transform
                pes = [rwpe_0, rwpe_1, rwpe_2]
            else:
                # Fallback: compute on-the-fly (backward compat)
                intra = [adjacency_0, coadjacency_1, coadjacency_2]
                with torch.no_grad():
                    pes = [
                        random_walk_pe(matrix, self.pe_steps)
                        for matrix in intra
                    ]
            xs = [
                torch.cat([x, pe.to(x.dtype)], dim=-1)
                for x, pe in zip(xs, pes, strict=True)
            ]
        xs = [lin(x) for lin, x in zip(self.preprocess, xs, strict=True)]

        for layer in self.layers:
            xs = layer(xs, neighborhoods)
        return xs[0], xs[1], xs[2]
