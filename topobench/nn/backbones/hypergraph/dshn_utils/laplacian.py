r"""The Directed Sheaf Hypergraph Laplacian.

This module builds the operator at the heart of

    E. Mule, S. Fiorini, A. Purificato, F. Siciliano, S. Coniglio and
    F. Silvestri, *"Directional Sheaf Hypergraph Networks: Unifying Learning
    on Directed and Undirected Hypergraphs"*, ICLR 2026,
    https://arxiv.org/abs/2510.04727

Given a hypergraph whose hyperedges are partitioned into a *tail set*
:math:`T(e)` and a *head set* :math:`H(e)` (Gallo et al., 1993 -- an
undirected hyperedge is one with :math:`H(e) = \emptyset`), and learned
real restriction maps :math:`F_{u \lhd e} \in \mathbb{R}^{d \times d}`, the
construction is:

**Charge matrix** (Definition 1, p. 3)

.. math::
    S^{(q)}_{u \lhd e} = \begin{cases}
        1 & u \in H(e) \\
        e^{-2\pi i q} & u \in T(e) \\
        0 & \text{otherwise.}
    \end{cases}

**Charged incidence matrix** (Eq. 1, p. 4)

.. math::
    B^{(q)}_{eu} = S^{(q)}_{u \lhd e} F_{u \lhd e} \in \mathbb{C}^{d\times d}

**Degrees** (p. 5), both real-valued because :math:`|S^{(q)}| = 1`:

.. math::
    D_E := \operatorname{diag}(\delta_1 I_d, \dots, \delta_m I_d), \quad
    D_{u} := \sum_{e \ni u} \bar{F}^\dagger_{u \lhd e}\bar{F}_{u \lhd e}
           = \sum_{e \ni u} F^\top_{u \lhd e}F_{u \lhd e}

**Laplacian** (Eq. 2, p. 4) and its normalization (Eq. 7, p. 6)

.. math::
    L^F := D_V - Q^F, \quad Q^F := B^{(q)\dagger} D_E^{-1} B^{(q)}
.. math::
    L^F_N := D_V^{-1/2} L^F D_V^{-1/2} = I_{nd} - Q^F_N, \quad
    Q^F_N := D_V^{-1/2} Q^F D_V^{-1/2}

Expanding per block (Eq. 3, p. 5):

.. math::
    (L^F)_{uv} = \begin{cases}
        \sum_{e \ni u}\left(1 - \tfrac{1}{\delta_e}\right)
            F^\top_{u \lhd e}F_{u \lhd e} & u = v \\
        -\sum_{e \ni u,v} \tfrac{1}{\delta_e}
            \overline{S^{(q)}_{u \lhd e}} S^{(q)}_{v \lhd e}
            F^\top_{u \lhd e}F_{v \lhd e} & u \neq v.
    \end{cases}

The :math:`\left(1 - \tfrac{1}{\delta_e}\right)` diagonal coefficient is the
single point of departure from the Sheaf Hypergraph Laplacian of Duta et al.
(2023) (which uses :math:`\tfrac{1}{\delta_e}`), and it is exactly what makes
this operator positive semidefinite -- see Appendix E of the paper, and
:func:`test_appendix_e_counterexample` in the test module.

.. note::
    The operator is real whenever the hypergraph is undirected. With
    :math:`H(e) = \emptyset` every incidence is a tail, so every phase
    product :math:`\overline{S^{(q)}_{u \lhd e}} S^{(q)}_{v \lhd e}` equals
    :math:`1` regardless of :math:`q` (Eq. 4, p. 5), and the paper states
    (p. 5) that :math:`L^F` is then "real-valued". Theorem 6 makes this
    precise: the normalized operator recovers the undirected hypergraph
    Laplacian of Zhou et al. (2006). So :math:`q` has no effect on undirected
    input. The phase cancellation is algebraic, so that holds exactly rather
    than to within floating-point error.

    We keep a single complex code path anyway, since the same code then serves
    directed input (see :func:`derive_orientation`) and the dtype saving is
    not worth branching the whole model over. A real-only specialization would
    halve the memory of the operator if that ever matters.
"""

import torch
from torch_scatter import scatter_add

ORIENTATIONS = ("none", "star")


def derive_orientation(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_hyperedges: int,
    mode: str = "none",
) -> torch.Tensor:
    r"""Assign each node-hyperedge incidence to the head or the tail set.

    TopoBench liftings produce *undirected* hypergraphs, so a direction has to
    be induced if the directional part of DSHN is to be exercised at all.

    ``"none"``
        Every incidence is a tail, i.e. :math:`H(e) = \emptyset` for all
        :math:`e`. This is the paper's definition of an undirected hyperedge
        (p. 3, after Gallo et al., 1993) and makes the operator real.

    ``"star"``
        The orientation of Appendix D.5: :math:`T(e_v) = \{v\}` and
        :math:`H(e_v) = N(v)`. The paper applies it to *directed* source
        graphs (:math:`H(e_v) = N_{out}(v)`); applied to an undirected source
        it still yields a genuinely directed hypergraph, with each hyperedge
        oriented away from the node it is centred on.

        This requires a **node-centred** lifting, one hyperedge per node with
        hyperedge :math:`j` centred on node :math:`j`. In TopoBench that is
        what :class:`HypergraphKHopLifting` produces
        (``incidence_1[n, neighbors] = 1``, one hyperedge per node), and also
        :class:`HypergraphKNNLifting`.

    Parameters
    ----------
    edge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]``, where row 0 holds node ids and
        row 1 holds hyperedge ids.
    num_nodes : int
        Number of nodes :math:`n`.
    num_hyperedges : int
        Number of hyperedges :math:`m`.
    mode : str, optional
        One of :data:`ORIENTATIONS` (default: ``"none"``).

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``[nnz]``, ``True`` where the incidence
        belongs to the head set :math:`H(e)`.

    Raises
    ------
    ValueError
        If ``mode`` is unknown, or if ``mode="star"`` is requested for a
        hypergraph that is not node-centred (``num_hyperedges != num_nodes``).
    """
    if mode not in ORIENTATIONS:
        raise ValueError(
            f"Unknown orientation {mode!r}; expected one of {ORIENTATIONS}."
        )
    if mode == "none":
        return torch.zeros(
            edge_index.size(1), dtype=torch.bool, device=edge_index.device
        )

    if num_hyperedges != num_nodes:
        raise ValueError(
            "orientation='star' needs a node-centred lifting with one "
            f"hyperedge per node, but got {num_nodes} nodes and "
            f"{num_hyperedges} hyperedges. Use a k-hop or knn lifting, or "
            "orientation='none'."
        )
    # Tail = the centre node of the hyperedge; head = everything else.
    return edge_index[0] != edge_index[1]


def charge_phase(
    is_head: torch.Tensor, q: float, dtype: torch.dtype = torch.cfloat
) -> torch.Tensor:
    r"""Evaluate the charge :math:`S^{(q)}` on each incidence.

    Implements Definition 1 (p. 3): :math:`1` on the head set and
    :math:`e^{-2\pi i q}` on the tail set.

    .. note::
        The reference implementation uses :math:`e^{+2\pi i q}` on tails
        (``models/sheafgedi.py``, ``complex_phase``), i.e. the complex
        conjugate of the published definition. The two agree on undirected
        input and differ only by the sign of :math:`\Im(L^F)` on directed
        input; we follow the paper.

    Parameters
    ----------
    is_head : torch.Tensor
        Boolean tensor of shape ``[nnz]``, as returned by
        :func:`derive_orientation`.
    q : float
        The charge parameter :math:`q`. ``q = 0`` disregards directions
        entirely (p. 5).
    dtype : torch.dtype, optional
        Complex dtype of the result (default: ``torch.cfloat``).

    Returns
    -------
    torch.Tensor
        Complex tensor of shape ``[nnz]`` with unit-modulus entries.
    """
    tail = torch.exp(
        torch.tensor(-2.0j * torch.pi * q, dtype=dtype, device=is_head.device)
    )
    one = torch.ones((), dtype=dtype, device=is_head.device)
    return torch.where(is_head, one, tail)


def incidence_pairs(
    edge_index: torch.Tensor, num_hyperedges: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Enumerate all ordered co-incidence pairs within each hyperedge.

    :math:`Q^F = B^{(q)\dagger} D_E^{-1} B^{(q)}` couples every pair of nodes
    that share a hyperedge, so assembling it requires the ordered pairs
    :math:`(i, j)` of incidences belonging to the same hyperedge -- including
    :math:`i = j`, which produces the :math:`\tfrac{1}{\delta_e}` term that
    turns the diagonal into :math:`1 - \tfrac{1}{\delta_e}`.

    The returned indices address ``edge_index`` **after sorting by hyperedge**,
    so callers must apply the same permutation to any per-incidence tensor;
    :func:`signless_laplacian_blocks` does this internally.

    Parameters
    ----------
    edge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]``.
    num_hyperedges : int
        Number of hyperedges :math:`m`.

    Returns
    -------
    order : torch.Tensor
        Permutation of shape ``[nnz]`` sorting incidences by hyperedge.
    left : torch.Tensor
        Index of shape ``[P]`` into the sorted incidences, the :math:`u` side.
    right : torch.Tensor
        Index of shape ``[P]`` into the sorted incidences, the :math:`v` side.

    Notes
    -----
    ``P`` equals :math:`\sum_e \delta_e^2`, so cost is quadratic in hyperedge
    size -- the dominant term in the paper's complexity analysis (p. 8). Dense
    hyperedges are the thing to watch: the k-hop liftings used here keep
    :math:`\delta_e` at roughly the average node degree.
    """
    nnz = edge_index.size(1)
    device = edge_index.device
    order = torch.argsort(edge_index[1], stable=True)
    hedges = edge_index[1][order]

    delta = torch.bincount(hedges, minlength=num_hyperedges)
    deg = delta[hedges]
    starts = torch.cumsum(delta, 0) - delta

    left = torch.repeat_interleave(torch.arange(nnz, device=device), deg)
    # Segmented arange: position of each pair within its own left-incidence.
    block_start = torch.cumsum(deg, 0) - deg
    within = torch.arange(
        int(deg.sum()), device=device
    ) - torch.repeat_interleave(block_start, deg)
    right = torch.repeat_interleave(starts[hedges], deg) + within

    return order, left, right


def restriction_degrees(
    edge_index: torch.Tensor, f_blocks: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    r"""Compute the block node-degree matrix :math:`D_V`.

    Implements :math:`D_u = \sum_{e \ni u} F^\top_{u \lhd e} F_{u \lhd e}`
    (p. 5). This is real even for a directed hypergraph, because
    :math:`\bar{F}^\dagger \bar{F} = |S^{(q)}|^2 F^\top F = F^\top F`; the
    paper types :math:`D_V \in \mathbb{R}^{nd \times nd}` for exactly this
    reason.

    Parameters
    ----------
    edge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]``.
    f_blocks : torch.Tensor
        Real restriction maps of shape ``[nnz, d, d]``.
    num_nodes : int
        Number of nodes :math:`n`.

    Returns
    -------
    torch.Tensor
        Real tensor of shape ``[num_nodes, d, d]``; block :math:`u` is
        :math:`D_u`. Note this is block-diagonal, not diagonal, unless the
        restriction maps are themselves diagonal.
    """
    prod = torch.bmm(f_blocks.transpose(1, 2), f_blocks)
    return scatter_add(prod, edge_index[0], dim=0, dim_size=num_nodes)


def block_inv_sqrt(
    blocks: torch.Tensor, eps: float = 1e-8, diagonal: bool = False
) -> torch.Tensor:
    r"""Invert the square root of a batch of symmetric PSD blocks.

    Used for :math:`D_V^{-1/2}` in Eq. 7. Because :math:`D_u` is symmetric
    positive semidefinite but may be *singular* -- an isolated node has
    :math:`D_u = 0` -- this returns the Moore-Penrose pseudo-inverse of the
    square root: eigenvalues at or below ``eps`` are mapped to zero rather
    than to infinity, which keeps isolated nodes finite instead of poisoning
    the whole operator with NaNs.

    Parameters
    ----------
    blocks : torch.Tensor
        Real symmetric PSD tensor of shape ``[k, d, d]``.
    eps : float, optional
        Eigenvalue floor below which the inverse is set to zero (default:
        ``1e-8``).
    diagonal : bool, optional
        If ``True``, treat each block as diagonal and invert its diagonal
        elementwise, skipping the eigendecomposition (default: ``False``).

    Returns
    -------
    torch.Tensor
        Real tensor of shape ``[k, d, d]``.

    Notes
    -----
    The ``diagonal`` flag exists for numerical reasons, not just speed.
    :math:`D_u^{-1/2}` is a smooth function of :math:`D_u` even where
    eigenvalues coincide, but the *eigendecomposition* is not: the backward
    pass of :func:`torch.linalg.eigh` divides by
    :math:`\lambda_i - \lambda_j` and returns NaN for repeated eigenvalues.

    Orthogonal restriction maps hit that case every time. With
    :math:`F^\top F = I` we get :math:`D_u = \deg(u) I`, which is maximally
    degenerate, so the eigenvector route NaNs on the backward pass while the
    elementwise route is exact. The reference implementation sidesteps the
    same problem by silently downgrading its ``block_norm`` to ``degree_norm``
    for orthogonal sheaves, which for :math:`D_u = \deg(u) I` is the same
    quantity.

    Callers therefore set ``diagonal=True`` whenever the restriction maps are
    structurally diagonal or orthogonal; see
    :func:`directed_sheaf_laplacian`.
    """
    if diagonal:
        diag = blocks.diagonal(dim1=-2, dim2=-1)
        inv = torch.where(
            diag > eps,
            diag.clamp_min(eps).pow(-0.5),
            torch.zeros_like(diag),
        )
        return torch.diag_embed(inv)

    evals, evecs = torch.linalg.eigh(blocks)
    inv = torch.where(
        evals > eps,
        evals.clamp_min(eps).pow(-0.5),
        torch.zeros_like(evals),
    )
    return evecs @ torch.diag_embed(inv) @ evecs.transpose(-2, -1)


def signless_laplacian_blocks(
    edge_index: torch.Tensor,
    f_blocks: torch.Tensor,
    phase: torch.Tensor,
    num_hyperedges: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Build the per-pair blocks of :math:`Q^F = B^{(q)\dagger}D_E^{-1}B^{(q)}`.

    Each ordered co-incidence pair :math:`(u \lhd e, v \lhd e)` contributes

    .. math::
        \frac{1}{\delta_e}\,\overline{S^{(q)}_{u \lhd e}}\,S^{(q)}_{v \lhd e}\,
        F^\top_{u \lhd e} F_{v \lhd e}

    to block :math:`(u, v)`, which is the off-diagonal term of Eq. 3 without
    its leading minus sign, and for :math:`u = v` the :math:`1/\delta_e`
    term subtracted from :math:`D_u`.

    Parameters
    ----------
    edge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]``.
    f_blocks : torch.Tensor
        Real restriction maps of shape ``[nnz, d, d]``.
    phase : torch.Tensor
        Complex charges of shape ``[nnz]``, from :func:`charge_phase`.
    num_hyperedges : int
        Number of hyperedges :math:`m`.

    Returns
    -------
    pair_u : torch.Tensor
        Node index of shape ``[P]`` for the row side of each block.
    pair_v : torch.Tensor
        Node index of shape ``[P]`` for the column side of each block.
    blocks : torch.Tensor
        Complex tensor of shape ``[P, d, d]``. Repeated ``(u, v)`` pairs
        arising from different hyperedges are *not* summed here; assembly
        does that via ``coalesce``.
    """
    order, left, right = incidence_pairs(edge_index, num_hyperedges)
    nodes = edge_index[0][order]
    hedges = edge_index[1][order]
    f_sorted = f_blocks[order]
    phase_sorted = phase[order]

    delta = torch.bincount(hedges, minlength=num_hyperedges)[hedges]

    prod = torch.bmm(f_sorted[left].transpose(1, 2), f_sorted[right]).to(
        phase.dtype
    )
    coef = (
        phase_sorted[left].conj()
        * phase_sorted[right]
        / delta[left].to(phase.real.dtype)
    )
    blocks = coef.view(-1, 1, 1) * prod
    return nodes[left], nodes[right], blocks


def assemble_blocks(
    pair_u: torch.Tensor,
    pair_v: torch.Tensor,
    blocks: torch.Tensor,
    num_nodes: int,
    d: int,
) -> torch.Tensor:
    r"""Scatter ``d x d`` blocks into a sparse :math:`nd \times nd` operator.

    Block :math:`(u, v)` occupies rows :math:`ud \dots ud + d - 1` and columns
    :math:`vd \dots vd + d - 1`, i.e. the stalk of each node is a contiguous
    slice -- the ordering assumed by :math:`I_n \otimes W_1` in Eq. 9.
    Duplicate blocks are summed by ``coalesce``.

    Parameters
    ----------
    pair_u : torch.Tensor
        Row node index of shape ``[P]``.
    pair_v : torch.Tensor
        Column node index of shape ``[P]``.
    blocks : torch.Tensor
        Tensor of shape ``[P, d, d]``.
    num_nodes : int
        Number of nodes :math:`n`.
    d : int
        Stalk dimension :math:`d`.

    Returns
    -------
    torch.Tensor
        Coalesced sparse COO tensor of shape ``[num_nodes * d, num_nodes * d]``
        with the dtype of ``blocks``.
    """
    device = blocks.device
    ar = torch.arange(d, device=device)
    rows = (pair_u.view(-1, 1, 1) * d + ar.view(1, -1, 1)).expand(-1, d, d)
    cols = (pair_v.view(-1, 1, 1) * d + ar.view(1, 1, -1)).expand(-1, d, d)
    index = torch.stack((rows.reshape(-1), cols.reshape(-1)))
    nd = num_nodes * d
    return torch.sparse_coo_tensor(
        index, blocks.reshape(-1), (nd, nd)
    ).coalesce()


def directed_sheaf_laplacian(
    edge_index: torch.Tensor,
    f_blocks: torch.Tensor,
    phase: torch.Tensor,
    num_nodes: int,
    num_hyperedges: int,
    normalized: bool = True,
    add_identity: bool = False,
    diagonal_degrees: bool = False,
) -> torch.Tensor:
    r"""Assemble the Directed Sheaf Hypergraph Laplacian.

    Returns either :math:`Q^F_N` (Eq. 7 -- the diffusion operator applied by
    Eq. 9, since :math:`I - L^F_N = Q^F_N`) or the unnormalized
    :math:`L^F = D_V - Q^F` of Eq. 2.

    Parameters
    ----------
    edge_index : torch.Tensor
        Incidence index of shape ``[2, nnz]``, row 0 nodes, row 1 hyperedges.
    f_blocks : torch.Tensor
        Real restriction maps of shape ``[nnz, d, d]``.
    phase : torch.Tensor
        Complex charges of shape ``[nnz]``, from :func:`charge_phase`.
    num_nodes : int
        Number of nodes :math:`n`.
    num_hyperedges : int
        Number of hyperedges :math:`m`.
    normalized : bool, optional
        If ``True`` (default) return :math:`Q^F_N = D_V^{-1/2}Q^F D_V^{-1/2}`.
        If ``False`` return :math:`L^F = D_V - Q^F`.
    add_identity : bool, optional
        If ``True``, use :math:`D_V + I` in place of :math:`D_V` when
        normalizing (default: ``False``).

        This is **not** part of Eq. 2 or Eq. 7. It is a stability trick
        enabled by default in the reference implementation
        (``--add_identity``, default ``true``); we default to the published
        definition and expose the deviation as a flag.
    diagonal_degrees : bool, optional
        Whether :math:`D_V`'s blocks are structurally diagonal, which is the
        case for diagonal and orthogonal restriction maps (default:
        ``False``). Required for orthogonal sheaves, whose degree blocks are
        degenerate; see :func:`block_inv_sqrt`.

    Returns
    -------
    torch.Tensor
        Coalesced sparse complex COO tensor of shape ``[n * d, n * d]``.
    """
    d = f_blocks.size(-1)
    pair_u, pair_v, blocks = signless_laplacian_blocks(
        edge_index, f_blocks, phase, num_hyperedges
    )
    d_v = restriction_degrees(edge_index, f_blocks, num_nodes)

    if not normalized:
        q_raw = assemble_blocks(pair_u, pair_v, blocks, num_nodes, d)
        deg = assemble_blocks(
            torch.arange(num_nodes, device=d_v.device),
            torch.arange(num_nodes, device=d_v.device),
            d_v.to(blocks.dtype),
            num_nodes,
            d,
        )
        return (deg - q_raw).coalesce()

    if add_identity:
        d_v = d_v + torch.eye(d, device=d_v.device).expand_as(d_v)
    d_inv = block_inv_sqrt(d_v, diagonal=diagonal_degrees).to(blocks.dtype)
    # D^{-1/2} is block diagonal, so it folds into each block before
    # assembly: D^{-1/2}(sum_e B_e)D^{-1/2} = sum_e D^{-1/2}B_e D^{-1/2}.
    blocks = d_inv[pair_u] @ blocks @ d_inv[pair_v]
    return assemble_blocks(pair_u, pair_v, blocks, num_nodes, d)
