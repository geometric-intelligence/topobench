"""Class implementing the DPHGNN model.

Reference: Saxena, Ghatak, Kolla, Mukherjee, Chakraborty. "DPHGNN: A Dual
Perspective Hypergraph Neural Networks", KDD'24, arXiv:2405.16616.

All structures (clique, star and HyperGCN expansions, and the three
hypergraph Laplacians) are recomputed at every forward pass directly from
``incidence_hyperedges``, so that a block-diagonal incidence (a batch of
several disjoint hypergraphs) yields exactly the concatenation of the
per-hypergraph outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_softmax

# --------------------------------------------------------------------- #
# Low-level sparse-algebra helpers (module-private).
# --------------------------------------------------------------------- #


def _inv_pow(degree, power):
    """Raise `degree` to `power`, mapping zero entries to zero.

    Parameters
    ----------
    degree : torch.Tensor
        Degree vector, shape ``[k]``.
    power : float
        Exponent applied to strictly positive entries.

    Returns
    -------
    torch.Tensor
        ``degree ** power`` where ``degree > 0``, else ``0``, shape ``[k]``.
    """
    out = torch.zeros_like(degree)
    mask = degree > 0
    out[mask] = degree[mask].pow(power).to(out.dtype)
    return out


def _scale_sparse_rows(matrix, row_scale):
    """Left-multiply a sparse COO matrix by a diagonal row scaling.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse COO matrix, shape ``[k, l]``.
    row_scale : torch.Tensor
        Diagonal scaling factors, shape ``[k]``.

    Returns
    -------
    torch.Tensor
        Sparse COO matrix ``diag(row_scale) @ matrix``, shape ``[k, l]``.
    """
    indices = matrix.indices()
    values = matrix.values() * row_scale[indices[0]]
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def _scale_sparse_cols(matrix, col_scale):
    """Right-multiply a sparse COO matrix by a diagonal column scaling.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse COO matrix, shape ``[k, l]``.
    col_scale : torch.Tensor
        Diagonal scaling factors, shape ``[l]``.

    Returns
    -------
    torch.Tensor
        Sparse COO matrix ``matrix @ diag(col_scale)``, shape ``[k, l]``.
    """
    indices = matrix.indices()
    values = matrix.values() * col_scale[indices[1]]
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def _sparse_identity(size, device, dtype):
    """Build a sparse COO identity matrix.

    Parameters
    ----------
    size : int
        Number of rows and columns ``k``.
    device : torch.device
        Device of the returned tensor.
    dtype : torch.dtype
        Data type of the returned tensor.

    Returns
    -------
    torch.Tensor
        Sparse COO identity matrix, shape ``[k, k]``.
    """
    idx = torch.arange(size, device=device)
    indices = torch.stack([idx, idx])
    values = torch.ones(size, device=device, dtype=dtype)
    return torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()


def _degree_diagonal(degree):
    """Build a sparse diagonal matrix from a degree vector.

    Parameters
    ----------
    degree : torch.Tensor
        Degree vector, shape ``[k]``.

    Returns
    -------
    torch.Tensor
        Sparse COO diagonal matrix, shape ``[k, k]``.
    """
    idx = torch.arange(degree.shape[0], device=degree.device)
    indices = torch.stack([idx, idx])
    return torch.sparse_coo_tensor(
        indices, degree, (degree.shape[0], degree.shape[0])
    ).coalesce()


def _sparse_combine(first, second, beta=1.0):
    """Compute ``first + beta * second`` for two sparse COO matrices.

    Parameters
    ----------
    first : torch.Tensor
        Sparse COO matrix, shape ``[k, l]``.
    second : torch.Tensor
        Sparse COO matrix, shape ``[k, l]``.
    beta : float, optional
        Scalar multiplier applied to `second`. Defaults to ``1.0``.

    Returns
    -------
    torch.Tensor
        Sparse COO matrix ``first + beta * second``, shape ``[k, l]``.
    """
    indices = torch.cat([first.indices(), second.indices()], dim=1)
    values = torch.cat([first.values(), beta * second.values()])
    return torch.sparse_coo_tensor(indices, values, first.shape).coalesce()


def _compute_degrees(incidence):
    """Compute hypernode and hyperedge degrees from a sparse incidence.

    Parameters
    ----------
    incidence : torch.Tensor
        Sparse COO incidence matrix, shape ``[n, m]``.

    Returns
    -------
    tuple of torch.Tensor
        ``(d_v [n], d_e [m])``, the hypernode and hyperedge degrees.
    """
    d_v = torch.sparse.sum(incidence, dim=1).to_dense()
    d_e = torch.sparse.sum(incidence, dim=0).to_dense()
    return d_v, d_e


def _sym_normalize(adjacency):
    """Symmetrically normalize a sparse adjacency: D^{-1/2} A D^{-1/2}.

    Parameters
    ----------
    adjacency : torch.Tensor
        Sparse COO adjacency matrix, shape ``[k, k]``.

    Returns
    -------
    torch.Tensor
        Symmetrically normalized sparse COO matrix, shape ``[k, k]``.
    """
    degree = torch.sparse.sum(adjacency, dim=1).to_dense()
    deg_inv_sqrt = _inv_pow(degree, -0.5)
    scaled = _scale_sparse_rows(adjacency, deg_inv_sqrt)
    return _scale_sparse_cols(scaled, deg_inv_sqrt)


def _combinatorial_laplacian(adjacency):
    """Compute the combinatorial Laplacian ``D - A`` of a sparse matrix.

    Parameters
    ----------
    adjacency : torch.Tensor
        Sparse COO adjacency matrix, shape ``[k, k]``.

    Returns
    -------
    torch.Tensor
        Sparse COO Laplacian matrix, shape ``[k, k]``.
    """
    degree = torch.sparse.sum(adjacency, dim=1).to_dense()
    return _sparse_combine(_degree_diagonal(degree), adjacency, beta=-1.0)


# --------------------------------------------------------------------- #
# Structure derivation: hypergraph expansions and Laplacians.
# --------------------------------------------------------------------- #


def _clique_adjacency(incidence, num_nodes):
    r"""Compute the binarized clique-expansion adjacency.

    Implements :math:`A_c = \mathbb{1}[H H^\top > 0] - I`: every hyperedge
    becomes a clique among its members, without multiplicity, diagonal set
    to zero.

    Parameters
    ----------
    incidence : torch.Tensor
        Sparse COO incidence matrix, shape ``[n, m]``.
    num_nodes : int
        Number of hypernodes ``n``.

    Returns
    -------
    torch.Tensor
        Sparse COO adjacency matrix, shape ``[n, n]``.
    """
    product = torch.sparse.mm(incidence, incidence.transpose(0, 1)).coalesce()
    indices = product.indices()
    row, col = indices[0], indices[1]
    keep = row != col
    row, col = row[keep], col[keep]
    values = torch.ones(
        row.shape[0], device=incidence.device, dtype=incidence.dtype
    )
    return torch.sparse_coo_tensor(
        torch.stack([row, col]), values, (num_nodes, num_nodes)
    ).coalesce()


def _star_adjacency(incidence, num_nodes, num_edges):
    r"""Compute the star-expansion adjacency.

    Builds the bipartite adjacency used by the star expansion::

        A_* = [[0, H], [H^T, 0]]

    on ``n + m`` nodes, ordered ``[hypernodes 0..n-1, supernodes
    n..n+m-1]`` (:math:`\phi_V`, :math:`\phi_S` convention).

    Parameters
    ----------
    incidence : torch.Tensor
        Sparse COO incidence matrix, shape ``[n, m]``.
    num_nodes : int
        Number of hypernodes ``n``.
    num_edges : int
        Number of hyperedges ``m``.

    Returns
    -------
    torch.Tensor
        Sparse COO adjacency matrix, shape ``[n + m, n + m]``.
    """
    indices = incidence.indices()
    row, col = indices[0], indices[1]
    top_right = torch.stack([row, col + num_nodes])
    bottom_left = torch.stack([col + num_nodes, row])
    all_indices = torch.cat([top_right, bottom_left], dim=1)
    values = torch.ones(
        all_indices.shape[1], device=incidence.device, dtype=incidence.dtype
    )
    size = num_nodes + num_edges
    return torch.sparse_coo_tensor(
        all_indices, values, (size, size)
    ).coalesce()


@torch.no_grad()
def _hypergcn_adjacency(incidence, x, num_nodes, with_mediators):
    r"""Compute the HyperGCN-expansion adjacency.

    For each hyperedge with cardinality :math:`\geq 2`, connects the pair
    of members farthest apart in feature space; optionally adds mediator
    edges to the other members, weighted :math:`1 / (2|e| - 3)`
    (the HyperGCN "with mediators" construction).

    Parameters
    ----------
    incidence : torch.Tensor
        Sparse COO incidence matrix, shape ``[n, m]``.
    x : torch.Tensor
        Node features used to pick the farthest pair, shape ``[n, d]``.
    num_nodes : int
        Number of hypernodes ``n``.
    with_mediators : bool
        Whether to add mediator edges.

    Returns
    -------
    torch.Tensor
        Sparse COO adjacency matrix, shape ``[n, n]``, symmetric.

    Notes
    -----
    Cardinality-1 hyperedges are ignored (no edge produced). The pairwise
    arg max is not differentiable, so this whole computation runs under
    ``no_grad``: only the resulting structure is used downstream. Ties in
    the arg max would break permutation equivariance; the toy test fixture
    uses continuous features to avoid this in practice.

    Implementation is batched: every hyperedge's member list is right-padded
    into a single ``[num_valid_hyperedges, k_max, d]`` tensor (``k_max`` the
    largest cardinality present), so a single batched ``torch.cdist`` and
    ``argmax`` replace what would otherwise be a per-hyperedge Python loop
    with a ``.item()`` CPU-GPU sync at every iteration — the latter is the
    dominant cost for hypergraphs with many hyperedges. Padded entries are
    masked to ``-inf`` before the arg max so they can never be selected;
    this is proven (and unit-tested against a reference loop implementation)
    to reproduce the unpadded arg max exactly, tie-breaking included, since
    padding is always appended after the real members of a hyperedge.
    """
    indices = incidence.indices()
    row, col = indices[0], indices[1]
    order = torch.argsort(col, stable=True)
    row, col = row[order], col[order]
    _, counts = torch.unique_consecutive(col, return_counts=True)

    valid = counts >= 2
    if not bool(valid.any()):
        empty = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        return torch.sparse_coo_tensor(
            empty,
            torch.zeros(0, device=x.device, dtype=x.dtype),
            (num_nodes, num_nodes),
        ).coalesce()

    offsets = torch.cumsum(counts, dim=0) - counts
    counts_valid = counts[valid]
    offsets_valid = offsets[valid]
    num_groups = counts_valid.shape[0]
    k_max = int(counts_valid.max().item())

    pos = (
        torch.arange(k_max, device=x.device)
        .unsqueeze(0)
        .expand(num_groups, k_max)
    )
    member_mask = pos < counts_valid.unsqueeze(1)
    flat_pos = torch.clamp(
        offsets_valid.unsqueeze(1) + pos, max=row.shape[0] - 1
    )
    members = row[flat_pos]  # [num_groups, k_max]; garbage where ~member_mask

    member_x = x[members]
    dist = torch.cdist(member_x, member_x)
    pair_mask = member_mask.unsqueeze(2) & member_mask.unsqueeze(1)
    dist = dist.masked_fill(~pair_mask, float("-inf"))
    flat_argmax = torch.argmax(dist.reshape(num_groups, -1), dim=1)
    i_local, j_local = flat_argmax // k_max, flat_argmax % k_max
    group_idx = torch.arange(num_groups, device=x.device)
    node_i = members[group_idx, i_local]
    node_j = members[group_idx, j_local]

    edge_row = [node_i, node_j]
    edge_col = [node_j, node_i]
    ones = torch.ones(num_groups, device=x.device, dtype=x.dtype)
    edge_val = [ones, ones]

    if with_mediators:
        has_mediators = counts_valid > 2
        is_pivot = (pos == i_local.unsqueeze(1)) | (
            pos == j_local.unsqueeze(1)
        )
        mediator_mask = member_mask & ~is_pivot & has_mediators.unsqueeze(1)
        # Matches the reference `1.0 / (2 * count - 3)` (Python floats are
        # float64), rounding to `x.dtype` only once, at the very end.
        weight = (1.0 / (2 * counts_valid.double() - 3)).to(x.dtype)

        g_idx, k_idx = mediator_mask.nonzero(as_tuple=True)
        if g_idx.numel() > 0:
            node_k = members[g_idx, k_idx]
            node_i_sel, node_j_sel, w_sel = (
                node_i[g_idx],
                node_j[g_idx],
                weight[g_idx],
            )
            edge_row += [node_i_sel, node_k, node_j_sel, node_k]
            edge_col += [node_k, node_i_sel, node_k, node_j_sel]
            edge_val += [w_sel, w_sel, w_sel, w_sel]

    edge_indices = torch.stack([torch.cat(edge_row), torch.cat(edge_col)])
    values = torch.cat(edge_val)
    return torch.sparse_coo_tensor(
        edge_indices, values, (num_nodes, num_nodes)
    ).coalesce()


def _hypergraph_laplacians(incidence, dv_inv, dv_inv_sqrt, de_inv):
    r"""Compute the three hypergraph Laplacians used by SIB.

    .. math::
        \Delta_{HGNN} &= D_v^{-1/2} H D_e^{-1} H^\top D_v^{-1/2} \\
        \Delta_{sym}  &= I - \Delta_{HGNN} \\
        \Delta_{rw}   &= I - D_v^{-1} H D_e^{-1} H^\top

    the HGNN, symmetric-normalized, and random-walk hypergraph Laplacians.

    Parameters
    ----------
    incidence : torch.Tensor
        Sparse COO incidence matrix, shape ``[n, m]``.
    dv_inv : torch.Tensor
        Guarded :math:`D_v^{-1}`, shape ``[n]``.
    dv_inv_sqrt : torch.Tensor
        Guarded :math:`D_v^{-1/2}`, shape ``[n]``.
    de_inv : torch.Tensor
        Guarded :math:`D_e^{-1}`, shape ``[m]``.

    Returns
    -------
    tuple of torch.Tensor
        ``(delta_hgnn, delta_sym, delta_rw)``, each sparse COO ``[n, n]``.
    """
    num_nodes = incidence.shape[0]
    incidence_t = incidence.transpose(0, 1)

    hgnn_scaled = _scale_sparse_cols(
        _scale_sparse_rows(incidence, dv_inv_sqrt), de_inv
    )
    delta_hgnn = torch.sparse.mm(hgnn_scaled, incidence_t).coalesce()
    delta_hgnn = _scale_sparse_cols(delta_hgnn, dv_inv_sqrt)

    identity = _sparse_identity(num_nodes, incidence.device, incidence.dtype)
    delta_sym = _sparse_combine(identity, delta_hgnn, beta=-1.0)

    rw_scaled = _scale_sparse_cols(
        _scale_sparse_rows(incidence, dv_inv), de_inv
    )
    rw_product = torch.sparse.mm(rw_scaled, incidence_t).coalesce()
    delta_rw = _sparse_combine(identity, rw_product, beta=-1.0)

    return delta_hgnn, delta_sym, delta_rw


_VALID_EXPANSIONS = frozenset({"clique", "star", "hypergcn"})
_VALID_FUSIONS = ("gate", "sum")


def _validate_expansions(expansions):
    """Validate an Experiment-E2 ``expansions`` ablation tuple.

    Parameters
    ----------
    expansions : tuple of str
        Candidate value for ``DPHGNN(expansions=...)``.

    Raises
    ------
    ValueError
        If ``expansions`` contains an unknown name, or omits ``"clique"``
        (structurally required: the TAA neighborhood, decision D-3, and
        every other block's fallback view are both defined in terms of
        the clique expansion, so dropping it has no clean counterpart in
        this architecture).
    """
    seen = set(expansions)
    unknown = seen - _VALID_EXPANSIONS
    if unknown:
        raise ValueError(
            f"Unknown expansions: {sorted(unknown)!r}; must be a subset "
            f"of {sorted(_VALID_EXPANSIONS)!r}"
        )
    if "clique" not in seen:
        raise ValueError(
            "expansions must include 'clique': it is structurally "
            "required (TAA neighborhood, decision D-3) and has no "
            "clean ablation counterpart in this architecture, got "
            f"{expansions!r}"
        )


def _clique_or_self_edges(adjacency, num_nodes, neighborhood):
    """Build the neighbor edge index used by topology-aware attention.

    Parameters
    ----------
    adjacency : torch.Tensor
        Sparse COO clique adjacency, shape ``[n, n]``.
    num_nodes : int
        Number of hypernodes ``n``.
    neighborhood : {"clique", "self"}
        ``"clique"`` uses the clique-expansion neighborhood plus self
        loops (decision D-3); ``"self"`` degenerates the attention into
        a per-node cross-gate.

    Returns
    -------
    torch.Tensor
        Edge index, shape ``[2, E]``.
    """
    self_loops = torch.arange(num_nodes, device=adjacency.device)
    self_index = torch.stack([self_loops, self_loops])
    if neighborhood == "self":
        return self_index
    if neighborhood == "clique":
        return torch.cat([adjacency.indices(), self_index], dim=1)
    raise ValueError(f"Unknown taa_neighborhood: {neighborhood!r}")


# --------------------------------------------------------------------- #
# Sub-modules.
# --------------------------------------------------------------------- #


class CliqueGCN(nn.Module):
    r"""Stacked convolution over the clique expansion :math:`G_c`.

    Implements the clique-expansion convolution::

        X_c = sigma[(I + D_v^{-1} A_c) X theta_c]

    stacked ``n_layers`` times, where :math:`D_v` is the degree of the
    clique graph itself.

    Parameters
    ----------
    hidden_channels : int
        Input and output feature dimension ``d``.
    n_layers : int
        Number of stacked layers.

    Notes
    -----
    The paper is not fully explicit about which degree normalizes this
    convolution. The corresponding star-expansion convolution normalizes
    by the degree of :math:`G_*` (:math:`D_{v_*}`), which is dimensionally
    required since :math:`G_*` has :math:`n+m` nodes; by consistency, and
    to match the graph-specific degrees used for the Laplacians, this
    implementation normalizes by the clique graph's own degree rather
    than the raw hypergraph incidence degree.
    """

    def __init__(self, hidden_channels, n_layers):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(n_layers)
        )

    def __repr__(self):
        """Represent the module as a string.

        Returns
        -------
        str
            String representation of the module.
        """
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"n_layers={self.n_layers})"
        )

    def forward(self, x, adjacency):
        """Apply the stacked clique convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape ``[n, hidden_channels]``.
        adjacency : torch.Tensor
            Sparse COO clique adjacency, shape ``[n, n]``.

        Returns
        -------
        torch.Tensor
            Output features, shape ``[n, hidden_channels]``.
        """
        degree = torch.sparse.sum(adjacency, dim=1).to_dense()
        deg_inv = _inv_pow(degree, -1.0)
        for layer in self.layers:
            h = layer(x)
            propagated = torch.sparse.mm(adjacency, h)
            x = F.relu(h + deg_inv.unsqueeze(-1) * propagated)
        return x


class StarGCN(nn.Module):
    r"""Stacked convolution over the star expansion :math:`G_*`.

    Implements the star-expansion convolution::

        X_* = sigma[(I + D_{v_*}^{-1} A_*) X_{G_*}^{(0)} theta_*]

    stacked ``n_layers`` times, where :math:`D_{v_*}` is the degree within
    :math:`G_*` (size :math:`n + m`).

    Parameters
    ----------
    hidden_channels : int
        Input and output feature dimension ``d``.
    n_layers : int
        Number of stacked layers.
    """

    def __init__(self, hidden_channels, n_layers):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(n_layers)
        )

    def __repr__(self):
        """Represent the module as a string.

        Returns
        -------
        str
            String representation of the module.
        """
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"n_layers={self.n_layers})"
        )

    def forward(self, x_star0, adjacency):
        """Apply the stacked star convolution.

        Parameters
        ----------
        x_star0 : torch.Tensor
            Initial star-graph features, shape ``[n + m, hidden_channels]``.
        adjacency : torch.Tensor
            Sparse COO star adjacency, shape ``[n + m, n + m]``.

        Returns
        -------
        torch.Tensor
            Output features, shape ``[n + m, hidden_channels]``.
        """
        degree = torch.sparse.sum(adjacency, dim=1).to_dense()
        deg_inv = _inv_pow(degree, -1.0)
        x = x_star0
        for layer in self.layers:
            h = layer(x)
            propagated = torch.sparse.mm(adjacency, h)
            x = F.relu(h + deg_inv.unsqueeze(-1) * propagated)
        return x


class HyperGCNConv(nn.Module):
    r"""Stacked convolution over the HyperGCN expansion :math:`G_{hyp}`.

    Implements the HyperGCN-expansion convolution::

        X_hyp = sigma[D_hat^{-1/2} A_hat_hyp D_hat^{-1/2} X theta_hyp]

    stacked ``n_layers`` times, the standard Kipf & Welling GCN
    normalization with self loops.

    Parameters
    ----------
    hidden_channels : int
        Input and output feature dimension ``d``.
    n_layers : int
        Number of stacked layers.
    """

    def __init__(self, hidden_channels, n_layers):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.layers = nn.ModuleList(
            nn.Linear(hidden_channels, hidden_channels)
            for _ in range(n_layers)
        )

    def __repr__(self):
        """Represent the module as a string.

        Returns
        -------
        str
            String representation of the module.
        """
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"n_layers={self.n_layers})"
        )

    def forward(self, x, adjacency):
        """Apply the stacked HyperGCN convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input features, shape ``[n, hidden_channels]``.
        adjacency : torch.Tensor
            Sparse COO HyperGCN adjacency (no self loops), shape
            ``[n, n]``.

        Returns
        -------
        torch.Tensor
            Output features, shape ``[n, hidden_channels]``.
        """
        num_nodes = adjacency.shape[0]
        identity = _sparse_identity(num_nodes, x.device, x.dtype)
        adjacency_hat = _sparse_combine(adjacency, identity, beta=1.0)
        norm_adjacency = _sym_normalize(adjacency_hat)
        for layer in self.layers:
            x = F.relu(torch.sparse.mm(norm_adjacency, layer(x)))
        return x


class TopologyAwareAttention(nn.Module):
    r"""Topology-Aware Attention (TAA) block.

    Implements additive GAT-style cross-attention between a query, key
    and value view of the same node set, as used by DPHGNN's
    Topology-Aware Attention formulation::

        alpha(i,j) = LeakyReLU_0.2(delta^T [W q_i | W k_j])
        alpha_hat  = softmax_j(alpha(i,j))   over j in N(i)
        out_i      = sum_j alpha_hat(i,j) * W v_j

    with a single shared projection ``W`` applied to the query, key and
    value inputs, split into ``heads`` heads and concatenated back.

    Parameters
    ----------
    hidden_channels : int
        Feature dimension ``d`` (must be divisible by ``heads``).
    heads : int
        Number of attention heads ``h``.
    dropout : float
        Dropout applied to the attention coefficients.
    negative_slope : float, optional
        Negative slope of the LeakyReLU, per GAT. Defaults to ``0.2``.

    Notes
    -----
    The neighborhood :math:`N(i)` is passed in explicitly as an
    ``edge_index`` at call time (decision D-3): the clique-expansion
    neighborhood plus self loops by default.
    """

    def __init__(self, hidden_channels, heads, dropout, negative_slope=0.2):
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by heads, got "
                f"hidden_channels={hidden_channels}, heads={heads}"
            )
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.head_dim = hidden_channels // heads
        self.negative_slope = negative_slope
        self.proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.attn = nn.Parameter(torch.empty(heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the learnable parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn)

    def __repr__(self):
        """Represent the module as a string.

        Returns
        -------
        str
            String representation of the module.
        """
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, heads={self.heads})"
        )

    def forward(self, query, key, value, edge_index):
        """Apply topology-aware cross-attention.

        Parameters
        ----------
        query : torch.Tensor
            Query features, shape ``[n, hidden_channels]``.
        key : torch.Tensor
            Key features, shape ``[n, hidden_channels]``.
        value : torch.Tensor
            Value features, shape ``[n, hidden_channels]``.
        edge_index : torch.Tensor
            Neighborhood edge index, shape ``[2, E]``, row ``i`` gathers
            over its neighbors ``j`` in ``edge_index[1]``.

        Returns
        -------
        torch.Tensor
            Attended features, shape ``[n, hidden_channels]``.
        """
        num_nodes = query.shape[0]
        q = self.proj(query).view(num_nodes, self.heads, self.head_dim)
        k = self.proj(key).view(num_nodes, self.heads, self.head_dim)
        v = self.proj(value).view(num_nodes, self.heads, self.head_dim)

        target, source = edge_index[0], edge_index[1]
        edge_features = torch.cat([q[target], k[source]], dim=-1)
        scores = F.leaky_relu(
            (edge_features * self.attn).sum(-1), self.negative_slope
        )
        alpha = scatter_softmax(scores, target, dim=0)
        alpha = self.dropout(alpha)
        out = scatter_add(
            alpha.unsqueeze(-1) * v[source],
            target,
            dim=0,
            dim_size=num_nodes,
        )
        return out.reshape(num_nodes, self.hidden_channels)


class SpectralInductiveBias(nn.Module):
    r"""Spectral Inductive Biases (SIB) block.

    Implements the SIB block's spectral smoothing (Saxena et al.,
    KDD'24, Sec. 3.2)::

        Delta_spectral X = [(Delta_rw + Delta_sym) X | Delta_HGNN X]
        X_spectral = sigma([X | X] + lambda * Delta_spectral X)

    Parameters
    ----------
    sib_lambda : float
        Scalar weight of the spectral term.

    Notes
    -----
    The output has :math:`2d` channels, not :math:`d`: this is required
    for the :math:`MLP_2` signature in Eq. (2) of the paper (decision D-4).
    The paper's claimed identity
    :math:`\Delta_{HGNN} \oplus \Delta_{sym} \oplus \Delta_{rw} = 3I -
    D_v^{-1}HD_e^{-1}H^\top` does not hold algebraically (decision D-7);
    concatenating, rather than summing, the three Laplacian terms is used
    instead, which preserves the symmetric-normalized information that a
    sum would cancel.
    """

    def __init__(self, sib_lambda):
        super().__init__()
        self.sib_lambda = sib_lambda

    def __repr__(self):
        """Represent the module as a string.

        Returns
        -------
        str
            String representation of the module.
        """
        return f"{self.__class__.__name__}(sib_lambda={self.sib_lambda})"

    def forward(self, x, delta_rw, delta_sym, delta_hgnn):
        """Compute the spectrally-biased features.

        Parameters
        ----------
        x : torch.Tensor
            Input features (encoder output), shape ``[n, d]``.
        delta_rw : torch.Tensor
            Sparse COO random-walk Laplacian, shape ``[n, n]``.
        delta_sym : torch.Tensor
            Sparse COO symmetric Laplacian, shape ``[n, n]``.
        delta_hgnn : torch.Tensor
            Sparse COO HGNN Laplacian, shape ``[n, n]``.

        Returns
        -------
        torch.Tensor
            Spectrally-biased features, shape ``[n, 2 * d]``.
        """
        rw_sym = torch.sparse.mm(delta_rw, x) + torch.sparse.mm(delta_sym, x)
        hgnn = torch.sparse.mm(delta_hgnn, x)
        delta_spectral = torch.cat([rw_sym, hgnn], dim=-1)
        residual = torch.cat([x, x], dim=-1)
        return F.relu(residual + self.sib_lambda * delta_spectral)


class FeatureMixtureModule(nn.Module):
    r"""Feature Mixture Module — Eq. (2) of the paper.

    Implements the following feature mixture (Eq. 2 of the paper)::

        X_eqv = MLP_1(x_kappa | x_Z) (*) sigmoid(ReLU(MLP_2(X_spectral)))
        X_static = X_eqv | MLP_3(X)

    Parameters
    ----------
    hidden_channels : int
        Feature dimension ``d`` of the backbone.
    fusion : {"gate", "sum"}, optional
        How ``MLP_1(x_kappa | x_Z)`` is combined with the SIB-derived gate
        in Eq. (2). ``"gate"`` (default) reproduces the paper's Hadamard
        product. ``"sum"`` is a component-ablation flag (arm A3,
        hypothesis: does the *learned multiplicative gate* matter, or does
        "use both branches" already explain the paper's gain?): it
        replaces the product with elementwise addition. Validity is
        enforced by the caller (``DPHGNN.__init__``). Defaults to
        ``"gate"``.

    Notes
    -----
    :math:`\sigma \circ \mathrm{ReLU}` in Eq. (2) is read as sigmoid then
    ReLU, turning the second factor into a multiplicative gate in
    :math:`[0.5, 1]` (decision D-5), consistent with the Hadamard product.
    :math:`MLP_1`, :math:`MLP_2` output :math:`\lfloor d/2 \rfloor`
    channels and :math:`MLP_3` outputs :math:`\lceil d/2 \rceil`, so the
    concatenation is exactly ``d`` wide even for odd ``d``.
    """

    def __init__(self, hidden_channels, fusion="gate"):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.fusion = fusion
        half_floor = hidden_channels // 2
        half_ceil = hidden_channels - half_floor
        self.mlp1 = nn.Linear(2 * hidden_channels, half_floor)
        self.mlp2 = nn.Linear(2 * hidden_channels, half_floor)
        self.mlp3 = nn.Linear(hidden_channels, half_ceil)

    def __repr__(self):
        """Represent the module as a string.

        Returns
        -------
        str
            String representation of the module.
        """
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"fusion={self.fusion!r})"
        )

    def forward(self, x_kappa, x_z, x_spectral, x):
        """Compute the static, equivariant node representation.

        Parameters
        ----------
        x_kappa : torch.Tensor
            Spatial TAA output, shape ``[n, d]``.
        x_z : torch.Tensor
            Spectral TAA output, shape ``[n, d]``.
        x_spectral : torch.Tensor
            SIB output, shape ``[n, 2 * d]``.
        x : torch.Tensor
            Input features (encoder output), shape ``[n, d]``.

        Returns
        -------
        torch.Tensor
            Static node representation, shape ``[n, d]``.
        """
        gate = torch.sigmoid(F.relu(self.mlp2(x_spectral)))
        mlp1_out = self.mlp1(torch.cat([x_kappa, x_z], dim=-1))
        x_eqv = mlp1_out * gate if self.fusion == "gate" else mlp1_out + gate
        return torch.cat([x_eqv, self.mlp3(x)], dim=-1)


class DynamicFeatureFusion(nn.Module):
    r"""Dynamic Feature Fusion (DFF) block — Eq. (3) of the paper.

    Implements the following dynamic fusion (Eq. 3 of the paper)::

        M_V  = H^T D_v^{-1/2} X_static             # V -> E
        M_S  = D_e^{-1} phi_S{A_* X_{G_*}}          # S -> E
        X_E  = M_V + M_S
        X_dp = sigma(X_static + H D_e^{-1} X_E Theta)

    Parameters
    ----------
    hidden_channels : int
        Feature dimension ``d``.

    Notes
    -----
    :math:`X_{G_*}` in the ``S -> E`` term is the *output* of the star
    convolution, not the initial star features (decision D-6).

    Component-ablation (arm A5, ``expansions`` without ``"star"`` in
    ``DPHGNN``): when ``star_adjacency`` is ``None``, the ``S -> E`` term
    has no supernode representation to draw on and is dropped entirely,
    so :math:`X_E = M_V` alone. This is a genuine change to the fused
    quantity, not a zero-masked approximation of it: dropping an
    expansion changes what is fused, so it has no clean output-masking
    counterpart (unlike the other ablation flags).
    """

    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.theta = nn.Linear(hidden_channels, hidden_channels)

    def __repr__(self):
        """Represent the module as a string.

        Returns
        -------
        str
            String representation of the module.
        """
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels})"
        )

    def forward(
        self,
        x_static,
        incidence,
        star_adjacency,
        x_star,
        phi_s,
        dv_inv_sqrt,
        de_inv,
    ):
        """Fuse node and supernode information into a hyperedge signal.

        Parameters
        ----------
        x_static : torch.Tensor
            Static node representation, shape ``[n, d]``.
        incidence : torch.Tensor
            Sparse COO incidence matrix, shape ``[n, m]``.
        star_adjacency : torch.Tensor or None
            Sparse COO star adjacency, shape ``[n + m, n + m]``, or
            ``None`` if the star expansion is excluded (``expansions``
            ablation, arm A5): the ``S -> E`` term is then dropped.
        x_star : torch.Tensor or None
            Star convolution output, shape ``[n + m, d]``, or ``None``
            iff ``star_adjacency`` is ``None``.
        phi_s : torch.Tensor or None
            Boolean mask selecting the ``m`` supernode rows, or ``None``
            iff ``star_adjacency`` is ``None``.
        dv_inv_sqrt : torch.Tensor
            Guarded :math:`D_v^{-1/2}`, shape ``[n]``.
        de_inv : torch.Tensor
            Guarded :math:`D_e^{-1}`, shape ``[m]``.

        Returns
        -------
        tuple of torch.Tensor
            ``(x_dp [n, d], x_e [m, d])``.
        """
        incidence_t = incidence.transpose(0, 1)
        scaled_static = dv_inv_sqrt.unsqueeze(-1) * x_static
        m_v = torch.sparse.mm(incidence_t, scaled_static)
        if star_adjacency is not None:
            m_s = (
                de_inv.unsqueeze(-1)
                * torch.sparse.mm(star_adjacency, x_star)[phi_s]
            )
            x_e = m_v + m_s
        else:
            x_e = m_v
        propagated = torch.sparse.mm(incidence, de_inv.unsqueeze(-1) * x_e)
        x_dp = F.relu(x_static + self.theta(propagated))
        return x_dp, x_e


class HypergraphOutputLayer(nn.Module):
    r"""Output hypergraph convolution — last line of Algorithm 1.

    Implements the following UniGCN-style output convolution::

        X_out = sigma(D_v^{-1/2} H D_e_bar^{-1/2} D_e^{-1} H^T D_v^{-1/2}
                      X_dp Theta_out)

    the UniGCN normalization (:math:`\bar{D}_e^{-1/2}`, the mean member
    degree), applied without dynamic fusion.

    Parameters
    ----------
    hidden_channels : int
        Feature dimension ``d``.

    Notes
    -----
    In the reference code this layer output ``num_classes`` directly; in
    TopoBench, classification is the readout's responsibility, so
    ``Theta_out`` maps ``d`` to ``d`` and the backbone returns embeddings.
    """

    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.theta_out = nn.Linear(hidden_channels, hidden_channels)

    def __repr__(self):
        """Represent the module as a string.

        Returns
        -------
        str
            String representation of the module.
        """
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels})"
        )

    def forward(
        self,
        x_dp,
        incidence,
        dv_inv_sqrt,
        de_inv,
        de_bar_inv_sqrt,
    ):
        r"""Apply the final hypergraph convolution.

        Parameters
        ----------
        x_dp : torch.Tensor
            Dynamically-fused node representation, shape ``[n, d]``.
        incidence : torch.Tensor
            Sparse COO incidence matrix, shape ``[n, m]``.
        dv_inv_sqrt : torch.Tensor
            Guarded :math:`D_v^{-1/2}`, shape ``[n]``.
        de_inv : torch.Tensor
            Guarded :math:`D_e^{-1}`, shape ``[m]``.
        de_bar_inv_sqrt : torch.Tensor
            Guarded :math:`\bar{D}_e^{-1/2}`, shape ``[m]``.

        Returns
        -------
        torch.Tensor
            Output node embeddings, shape ``[n, d]``.
        """
        incidence_t = incidence.transpose(0, 1)
        theta = self.theta_out(x_dp)
        v_to_e = de_inv.unsqueeze(-1) * torch.sparse.mm(
            incidence_t, dv_inv_sqrt.unsqueeze(-1) * theta
        )
        v_to_e = de_bar_inv_sqrt.unsqueeze(-1) * v_to_e
        e_to_v = torch.sparse.mm(incidence, v_to_e)
        return F.relu(dv_inv_sqrt.unsqueeze(-1) * e_to_v)


# --------------------------------------------------------------------- #
# Full model.
# --------------------------------------------------------------------- #


class DPHGNN(nn.Module):
    r"""DPHGNN: A Dual Perspective Hypergraph Neural Network.

    Saxena, Ghatak, Kolla, Mukherjee, Chakraborty, KDD'24,
    arXiv:2405.16616. Combines three hypergraph expansions (clique, star,
    HyperGCN), topology-aware attention, spectral inductive biases and
    dynamic feature fusion, following the paper's Algorithm 1.

    Parameters
    ----------
    hidden_channels : int
        Hidden feature dimension ``d`` (input and output of the
        backbone; both come from/feed into standard TopoBench encoder
        and readout).
    n_gnn_layers : int, optional
        Number of stacked layers in each expansion convolution. Defaults
        to ``2``.
    taa_heads : int, optional
        Number of attention heads in the TAA blocks. Defaults to ``4``.
    dropout : float, optional
        Dropout rate applied after each expansion convolution and inside
        the attention blocks. Defaults to ``0.5``.
    sib_lambda : float, optional
        Scalar weight of the spectral term in SIB. Defaults to ``0.5``.
    n_dff_layers : int, optional
        Number of stacked Dynamic Feature Fusion layers. Defaults to
        ``1``. The ablation in the paper (Table 6) shows a collapse at 4
        and 8 layers: do not increase this without reason.
    supernode_init : {"zeros", "mean"}, optional
        Initialization of the star-expansion supernodes. ``"zeros"``
        matches the reference code; ``"mean"`` initializes each supernode
        as the mean of its members' features. Defaults to ``"zeros"``.
    with_mediators : bool, optional
        Whether the HyperGCN expansion adds mediator edges. Defaults to
        ``False``.
    taa_neighborhood : {"clique", "self"}, optional
        Neighborhood used by the TAA blocks (decision D-3). Defaults to
        ``"clique"``.
    use_spectral : bool, optional
        Component-ablation flag (arm A1 ``spatial_only``). If ``False``,
        the spectral TAA-branch output (:math:`\hat{x}_Z`) is zeroed
        before the Feature Mixture Module fuses it (Eq. 2). The default
        (``True``) reproduces the published dual spatial/spectral
        architecture.
    use_spatial : bool, optional
        Component-ablation flag (arm A2 ``spectral_only``). If
        ``False``, the spatial TAA-branch output (:math:`\hat{x}_\kappa`)
        is zeroed before fusion (Eq. 2). The default (``True``)
        reproduces the published architecture.
    use_sib : bool, optional
        Component-ablation flag (arm A4 ``no_sib``). If ``False``, the
        Spectral Inductive Bias block's Laplacian smoothing term is
        disabled: the SIB output collapses to its :math:`\lambda \to 0`
        limit, :math:`X_{spectral} = \sigma([X|X])` -- the residual
        alone, at the same :math:`2d` shape so the downstream
        :math:`MLP_2` (Eq. 2) is unaffected. The default (``True``)
        reproduces the published architecture.
    fusion : {"gate", "sum"}, optional
        Component-ablation flag (arm A3 ``fusion_sum`` -- tests whether
        the learned multiplicative gate is load-bearing, or whether
        "use both branches" alone already explains the paper's gain).
        Passed to :class:`FeatureMixtureModule`; controls whether Eq.
        (2)'s second factor is combined with :math:`MLP_1`'s output by a
        Hadamard product (``"gate"``, default, paper-faithful) or by
        elementwise addition (``"sum"``). Raises ``ValueError`` if not
        one of ``{"gate", "sum"}``.
    expansions : tuple of str, optional
        Component-ablation flag (arm A5 ``clique_only``). Which
        of the three hypergraph expansions feed the model;
        must be a subset of ``{"clique", "star", "hypergcn"}`` that
        always includes ``"clique"`` (see ``_validate_expansions``).
        When ``"star"`` is absent, the star-conv branch is skipped
        entirely: the TAA query view (:math:`X_*^V`) and its
        Laplacian-smoothed counterpart (:math:`Y_*^V`) both fall back to
        the clique-conv view, and the Dynamic Feature Fusion ``S -> E``
        term is dropped, leaving :math:`X_E = M_V` alone. When
        ``"hypergcn"`` is absent, the TAA value view
        (:math:`X_{hyp}`/:math:`Y_{hyp}`) likewise falls back to the
        clique-conv view. Defaults to
        ``("clique", "star", "hypergcn")`` (the full, paper-faithful
        architecture). Raises ``ValueError`` on an unknown expansion
        name or if ``"clique"`` is missing.

    Notes
    -----
    The reference implementation (https://github.com/mr-siddy/DPHGNN)
    concatenates a noisy one-hot encoding of the *labels* to the TAA
    features; this has no counterpart in the paper and is a data leak
    (decision D-2). It is not reproduced here: ``forward`` only takes
    ``(x_0, incidence_hyperedges)``. Learning rates and dropout rates
    per sub-module (Table 10) are not exposed individually (decision
    D-8): a single ``dropout`` hyperparameter is shared across blocks.

    ``use_spectral``/``use_spatial``/``use_sib``/``fusion``/``expansions``
    are component-ablation flags used by a supplementary study isolating
    which architectural piece of DPHGNN is load-bearing. They default to
    the values that reproduce the published architecture exactly, and
    every submodule stays registered regardless of the flags (masking,
    not deletion): parameter count (``n_params``) is therefore identical
    across every ablation arm, and a reader must not infer that a masked
    arm is a smaller model -- some of its parameters simply receive no
    gradient. One documented consequence of masking rather than deleting:
    ``use_spectral=False`` / ``use_spatial=False`` zero the corresponding
    TAA branch's *output*, not its contribution to :math:`MLP_1` in Eq.
    (2) -- because :math:`MLP_1` has a learned bias and shares weights
    across both halves of its concatenated input, a small bias-driven
    signal from the "disabled" branch can still reach :math:`X_{eqv}`.
    Note also that, in this implementation, the multiplicative gate in
    Eq. (2) is derived from SIB (:math:`X_{spectral}`), not from either
    TAA branch, so masking a TAA branch does not halve the gate -- only
    the bias caveat above applies.
    """

    def __init__(
        self,
        hidden_channels,
        n_gnn_layers=2,
        taa_heads=4,
        dropout=0.5,
        sib_lambda=0.5,
        n_dff_layers=1,
        supernode_init="zeros",
        with_mediators=False,
        taa_neighborhood="clique",
        *,
        use_spectral: bool = True,
        use_spatial: bool = True,
        use_sib: bool = True,
        fusion: str = "gate",
        expansions: tuple = ("clique", "star", "hypergcn"),
    ):
        super().__init__()
        if supernode_init not in ("zeros", "mean"):
            raise ValueError(f"Unknown supernode_init: {supernode_init!r}")
        if fusion not in _VALID_FUSIONS:
            raise ValueError(
                f"Unknown fusion: {fusion!r}; must be one of "
                f"{_VALID_FUSIONS!r}"
            )
        _validate_expansions(expansions)
        self.hidden_channels = hidden_channels
        self.n_gnn_layers = n_gnn_layers
        self.taa_heads = taa_heads
        self.dropout_rate = dropout
        self.sib_lambda = sib_lambda
        self.n_dff_layers = n_dff_layers
        self.supernode_init = supernode_init
        self.with_mediators = with_mediators
        self.taa_neighborhood = taa_neighborhood
        self.use_spectral = use_spectral
        self.use_spatial = use_spatial
        self.use_sib = use_sib
        self.fusion = fusion
        self.expansions = tuple(expansions)

        self.clique_conv = CliqueGCN(hidden_channels, n_gnn_layers)
        self.star_conv = StarGCN(hidden_channels, n_gnn_layers)
        self.hyp_conv = HyperGCNConv(hidden_channels, n_gnn_layers)

        self.taa_spatial = TopologyAwareAttention(
            hidden_channels, taa_heads, dropout
        )
        self.taa_spectral = TopologyAwareAttention(
            hidden_channels, taa_heads, dropout
        )

        self.sib = SpectralInductiveBias(sib_lambda)
        self.feature_mixture = FeatureMixtureModule(
            hidden_channels, fusion=fusion
        )

        self.dff_layers = nn.ModuleList(
            DynamicFeatureFusion(hidden_channels) for _ in range(n_dff_layers)
        )
        self.output_layer = HypergraphOutputLayer(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        """Represent the module as a string.

        Returns
        -------
        str
            String representation of the module.
        """
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"n_gnn_layers={self.n_gnn_layers}, "
            f"taa_heads={self.taa_heads}, "
            f"dropout={self.dropout_rate}, "
            f"sib_lambda={self.sib_lambda}, "
            f"n_dff_layers={self.n_dff_layers}, "
            f"supernode_init={self.supernode_init!r}, "
            f"with_mediators={self.with_mediators}, "
            f"taa_neighborhood={self.taa_neighborhood!r}, "
            f"use_spectral={self.use_spectral}, "
            f"use_spatial={self.use_spatial}, "
            f"use_sib={self.use_sib}, "
            f"fusion={self.fusion!r}, "
            f"expansions={self.expansions!r})"
        )

    def forward(self, x_0, incidence_hyperedges):
        """Compute node and hyperedge embeddings.

        Parameters
        ----------
        x_0 : torch.Tensor
            Dense hypernode features (encoder output), shape
            ``[n, hidden_channels]``.
        incidence_hyperedges : torch.Tensor
            Sparse COO incidence matrix, shape ``[n, m]``.

        Returns
        -------
        tuple of torch.Tensor
            ``(x_0_out [n, hidden_channels], x_1_out [m,
            hidden_channels])``: the output node embeddings and the
            hyperedge representation :math:`X_E` produced by the last
            Dynamic Feature Fusion layer.
        """
        incidence = incidence_hyperedges.coalesce().to(x_0.dtype)
        num_nodes, num_edges = incidence.shape

        d_v, d_e = _compute_degrees(incidence)
        dv_inv = _inv_pow(d_v, -1.0)
        dv_inv_sqrt = _inv_pow(d_v, -0.5)
        de_inv = _inv_pow(d_e, -1.0)
        de_bar = de_inv * torch.sparse.mm(
            incidence.transpose(0, 1), d_v.unsqueeze(-1)
        ).squeeze(-1)
        de_bar_inv_sqrt = _inv_pow(de_bar, -0.5)

        # Experiment-E2 ablation (arm A5 `clique_only`): "clique" is
        # structurally required (validated in __init__) and always
        # computed; "star"/"hypergcn" are only computed when selected,
        # with the surviving clique view standing in for the dropped
        # one downstream (see the `expansions` docstring above). When
        # both are enabled (the default), the RNG-consuming dropout
        # calls below (x_c, x_star, x_hyp, in that order) reproduce the
        # exact pre-ablation call sequence.
        use_star = "star" in self.expansions
        use_hypergcn = "hypergcn" in self.expansions

        a_c = _clique_adjacency(incidence, num_nodes)
        delta_hgnn, delta_sym, delta_rw = _hypergraph_laplacians(
            incidence, dv_inv, dv_inv_sqrt, de_inv
        )

        x_c = self.dropout(self.clique_conv(x_0, a_c))
        l_c = _combinatorial_laplacian(a_c)
        y_c = torch.sparse.mm(l_c, x_c)

        phi_v = torch.zeros(
            num_nodes + num_edges, dtype=torch.bool, device=x_0.device
        )
        phi_v[:num_nodes] = True
        phi_s = ~phi_v

        if use_star:
            if self.supernode_init == "zeros":
                s_0 = torch.zeros(
                    num_edges,
                    self.hidden_channels,
                    device=x_0.device,
                    dtype=x_0.dtype,
                )
            else:
                s_0 = de_inv.unsqueeze(-1) * torch.sparse.mm(
                    incidence.transpose(0, 1), x_0
                )
            x_star0 = torch.cat([x_0, s_0], dim=0)
            a_star = _star_adjacency(incidence, num_nodes, num_edges)
            x_star = self.dropout(self.star_conv(x_star0, a_star))
            x_star_v = x_star[phi_v]
            l_star = _combinatorial_laplacian(a_star)
            y_star_v = torch.sparse.mm(l_star, x_star)[phi_v]
        else:
            a_star, x_star = None, None
            x_star_v, y_star_v = x_c, y_c

        if use_hypergcn:
            a_hyp = _hypergcn_adjacency(
                incidence, x_0, num_nodes, self.with_mediators
            )
            x_hyp = self.dropout(self.hyp_conv(x_0, a_hyp))
            l_hyp = _combinatorial_laplacian(a_hyp)
            y_hyp = torch.sparse.mm(l_hyp, x_hyp)
        else:
            x_hyp, y_hyp = x_c, y_c

        edge_index = _clique_or_self_edges(
            a_c, num_nodes, self.taa_neighborhood
        )
        x_kappa = self.taa_spatial(x_star_v, x_c, x_hyp, edge_index)
        x_z = self.taa_spectral(y_star_v, y_c, y_hyp, edge_index)

        # Experiment-E2 ablation (arms A1/A2): output-masking, applied
        # after the branch is computed and before it reaches fusion
        # (see the `use_spectral`/`use_spatial` docstrings above for the
        # documented gate-interaction caveat).
        if not self.use_spatial:
            x_kappa = torch.zeros_like(x_kappa)
        if not self.use_spectral:
            x_z = torch.zeros_like(x_z)

        # Experiment-E2 ablation (arm A4 `no_sib`): the lambda -> 0
        # limit of Eq. 6.2, at the same 2d shape.
        if self.use_sib:
            x_spectral = self.sib(x_0, delta_rw, delta_sym, delta_hgnn)
        else:
            x_spectral = F.relu(torch.cat([x_0, x_0], dim=-1))
        x_static = self.feature_mixture(x_kappa, x_z, x_spectral, x_0)

        x_dp = x_static
        x_e = None
        for dff in self.dff_layers:
            x_dp, x_e = dff(
                x_dp,
                incidence,
                a_star,
                x_star,
                phi_s if use_star else None,
                dv_inv_sqrt,
                de_inv,
            )

        x_out = self.output_layer(
            x_dp, incidence, dv_inv_sqrt, de_inv, de_bar_inv_sqrt
        )
        return x_out, x_e
