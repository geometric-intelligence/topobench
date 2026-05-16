"""Deterministic connection on a graph via local PCA + optimal alignment.

This module implements Algorithm 1 of Barbero et al. (2022) — the scientific
heart of Conn-NSD. The Bodnar et al. (2022) NSD diffusion machinery in the
sibling ``nsd_utils`` package is reused unchanged; only the construction of
the orthogonal restriction maps changes.

Mathematical setting
--------------------
Assume node features ``X ∈ ℝ^{N×p}`` are sampled from a ``d``-dimensional
Riemannian manifold ``M`` embedded in ``ℝ^p``, with ``d ≪ p``. At every
node ``v`` we approximate the tangent space ``T_{x_v} M`` by local PCA
over its 1-hop neighbourhood (Singer & Wu, 2012, §3.1; Barbero et al.,
2022, §3.2). The resulting orthonormal basis is a column-orthonormal
matrix ``O_v ∈ ℝ^{p×d}``.

For each directed edge ``v → u`` we then build the orthogonal map
``F_{vu} = U V^⊤`` from the SVD ``O_v^⊤ O_u = U Σ V^⊤``. This is the
optimal alignment in Frobenius norm between the two tangent bases and is,
when ``v`` and ``u`` are nearby on ``M``, an approximation to the parallel
transport ``T_{x_v} M → T_{x_u} M``.

In the language of cellular sheaves (Hansen & Ghrist, 2019), ``F_{vu}`` is
the restriction map from the node stalk ``ℝ^d`` at ``v`` to the edge stalk
at the unoriented edge ``{v,u}``. The discrete ``O(d)``-bundle laplacian
``Δ_F = δ^⊤ δ`` is then assembled by ``NormConnectionLaplacianBuilder``
from the sibling package — same formula as Bodnar's NSD-O(d), but with
``F`` fixed instead of learned.

Shape contract
--------------
All shapes are documented in NumPy style. Notation:

    N   number of nodes
    E   number of *directed* edges (``edge_index`` is the PyG bidirectional form)
    p   ambient feature dimension
    d   stalk / tangent-space dimension

We use the dtype of the input features throughout. No tensors are moved
across devices implicitly.

References
----------
[1] F. Barbero et al. "Sheaf Neural Networks with Connection Laplacians."
    ICML 2022 TAG-ML Workshop. arXiv:2206.08702. Algorithm 1.
[2] A. Singer and H.-T. Wu. "Vector Diffusion Maps and the Connection
    Laplacian." Communications on Pure and Applied Mathematics, 2012.
[3] C. Bodnar et al. "Neural Sheaf Diffusion." ICLR 2022 Workshop.
    arXiv:2202.04579.
"""

from __future__ import annotations

import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Step 1 — Local tangent basis (Algorithm 1, lines 3–6).
# -----------------------------------------------------------------------------


def local_tangent_basis(
    node_features: Tensor,
    edge_index: Tensor,
    stalk_dim: int,
) -> Tensor:
    """Approximate the tangent space at each node by local PCA.

    For every node ``v`` we collect its 1-hop neighbourhood ``N(v)`` from the
    edge set, centre the neighbours at ``x_v``, stack into a matrix
    ``X̂_v ∈ ℝ^{p × |N(v)|}``, and take the first ``d`` left singular vectors
    as an orthonormal basis of the local tangent space.

    When ``|N(v)| < d`` the 1-hop set is *topped up* with the
    Euclidean-nearest non-neighbour nodes (Barbero et al., 2022, §3.2,
    "To solve the problem for nodes which have less than d neighbours…").
    Fully isolated nodes (and the unrealistic case ``N ≤ d``) are treated
    by ranking every other node by Euclidean distance.

    Parameters
    ----------
    node_features : torch.Tensor, shape ``[N, p]``
        Raw node feature matrix ``X``. Must be 2-D.
    edge_index : torch.Tensor, shape ``[2, E]``
        Edge indices in PyG convention. May be directed or bidirectional;
        we look at the row ``edge_index[0]`` for source vertices to build
        ``N(v)``.
    stalk_dim : int
        Tangent-space dimension ``d`` (``> 0``).

    Returns
    -------
    torch.Tensor, shape ``[N, p, d]``
        ``tangent_basis[v]`` has orthonormal columns spanning the local
        tangent space at node ``v``. Concretely:
        ``tangent_basis[v].T @ tangent_basis[v] == I_d`` up to fp tolerance.

    Notes
    -----
    Complexity is ``O(N · SVD(p × max_deg))``. The function has no learnable
    parameters and never enters the autograd graph: ``tangent_basis`` is
    detached from ``node_features``.
    """
    assert node_features.dim() == 2, (
        f"node_features must be [N, p], got {tuple(node_features.shape)}"
    )
    assert edge_index.dim() == 2 and edge_index.size(0) == 2, (
        f"edge_index must be [2, E], got {tuple(edge_index.shape)}"
    )
    assert stalk_dim > 0, f"stalk_dim must be positive, got {stalk_dim}"

    num_nodes, ambient_dim = node_features.shape
    device = node_features.device
    dtype = node_features.dtype

    # Build 1-hop neighbour lists from the (possibly directed) edge index.
    # We treat the edge set as undirected for the purposes of tangent
    # estimation, matching Barbero et al. §3.2.
    src, dst = edge_index[0], edge_index[1]
    neighbours: list[list[int]] = [[] for _ in range(num_nodes)]
    for s, t in zip(src.tolist(), dst.tolist()):
        if s != t:
            neighbours[s].append(t)
            neighbours[t].append(s)
    # Dedupe (handles bidirectional edge_index without double-counting).
    neighbours = [sorted(set(ns)) for ns in neighbours]

    # Precompute features on CPU index lists; SVD is fastest on small dense
    # matrices and we never see graphs large enough here to make this hot.
    tangent_basis = torch.zeros(
        num_nodes, ambient_dim, stalk_dim, device=device, dtype=dtype
    )

    # Distance to all other nodes (needed for the fallback). Computed lazily.
    all_pair_dist: Tensor | None = None

    for v in range(num_nodes):
        own = node_features[v]
        ns = neighbours[v]

        if len(ns) < stalk_dim:
            # Top-up rule: take Euclidean-nearest non-neighbours, excluding v
            # itself, until we have stalk_dim candidates (Barbero §3.2).
            if all_pair_dist is None:
                all_pair_dist = torch.cdist(node_features, node_features)
            assert all_pair_dist is not None  # for type-checkers
            dists = all_pair_dist[v].clone()
            dists[v] = float("inf")
            for n in ns:
                dists[n] = float("inf")
            needed = stalk_dim - len(ns)
            extra = torch.topk(dists, k=needed, largest=False).indices.tolist()
            ns = list(ns) + extra

        # Centred neighbour matrix:  X̂_v[:, k] = x_{n_k} − x_v   ∈  ℝ^p
        centred = node_features[ns] - own  # [|N|, p]
        centred = centred.t()  # [p, |N|]

        # SVD: X̂_v = U Σ Vᵀ, basis is first d columns of U.
        # torch.linalg.svd handles fat / thin matrices uniformly. We use
        # full_matrices=False so U is [p, min(p,|N|)] — large enough since
        # we ensured |N| ≥ d above and we assume p ≥ d (the manifold
        # assumption).
        u, _, _ = torch.linalg.svd(centred, full_matrices=False)
        # If p < d the manifold assumption fails; we still take the first d
        # columns, padding with zeros — but in practice the AllCellFeature
        # encoder lifts the input to p = hidden_dim ≫ d.
        if u.size(1) >= stalk_dim:
            tangent_basis[v] = u[:, :stalk_dim]
        else:
            tangent_basis[v, :, : u.size(1)] = u

    return tangent_basis.detach()


# -----------------------------------------------------------------------------
# Step 2 — Optimal alignment between tangent spaces (Algorithm 1, lines 8–10).
# -----------------------------------------------------------------------------


def optimal_alignment(
    tangent_basis_src: Tensor,
    tangent_basis_dst: Tensor,
) -> Tensor:
    """Solve the orthogonal Procrustes problem between two tangent bases.

    Given two column-orthonormal matrices ``O_v, O_u ∈ ℝ^{p×d}``, returns the
    orthogonal matrix ``F = U V^⊤`` from the SVD
    ``O_v^⊤ O_u = U Σ V^⊤``. This is the closest element of ``O(d)`` to
    ``O_v^⊤ O_u`` in Frobenius norm (Schönemann, 1966), equivalently the
    minimiser of ``‖O_v F − O_u‖_F^2``.

    Parameters
    ----------
    tangent_basis_src : torch.Tensor, shape ``[*, p, d]``
        Source orthonormal basis (or a batch thereof).
    tangent_basis_dst : torch.Tensor, shape ``[*, p, d]``
        Destination orthonormal basis (same leading dims as ``src``).

    Returns
    -------
    torch.Tensor, shape ``[*, d, d]``
        Orthogonal alignment matrix ``F``.

    Notes
    -----
    For each batch element ``F`` satisfies ``F^⊤ F = F F^⊤ = I_d`` up to fp
    tolerance. The map is computed without gradient flow; this is a
    pre-processing step, not a learnable layer.
    """
    assert tangent_basis_src.shape == tangent_basis_dst.shape, (
        f"basis shape mismatch: {tuple(tangent_basis_src.shape)} vs "
        f"{tuple(tangent_basis_dst.shape)}"
    )
    cross_gram = tangent_basis_src.transpose(-1, -2) @ tangent_basis_dst
    u, _, vt = torch.linalg.svd(cross_gram, full_matrices=False)
    return (u @ vt).detach()


# -----------------------------------------------------------------------------
# Step 3 — Top-level constructor: features + edges → restriction maps.
# -----------------------------------------------------------------------------


def build_connection(
    node_features: Tensor,
    edge_index: Tensor,
    stalk_dim: int,
) -> Tensor:
    """Build orthogonal restriction maps for every edge — Algorithm 1.

    This is the only function the model ever calls. It returns a tensor
    of shape ``[E, d, d]`` containing the orthogonal restriction map for
    each directed edge in ``edge_index``, in the same order.

    Parameters
    ----------
    node_features : torch.Tensor, shape ``[N, p]``
        Raw node feature matrix. We do not normalise these — the manifold
        assumption is on the features as given.
    edge_index : torch.Tensor, shape ``[2, E]``
        Edge index in PyG convention. The ``e``-th column ``(v, u)`` produces
        ``restriction_maps[e] = F_{v → u}``. The caller is responsible for
        passing a bidirectional edge index if a symmetric sheaf is desired.
    stalk_dim : int
        Tangent-space / stalk dimension ``d``.

    Returns
    -------
    torch.Tensor, shape ``[E, d, d]``
        Orthogonal restriction maps, one per directed edge.

    Invariants
    ----------
    For every edge ``e``:

        F[e] @ F[e].T  ≈  I_d                              (orthogonality)

    If both ``(v, u)`` and ``(u, v)`` appear in ``edge_index`` at positions
    ``e₁`` and ``e₂`` respectively, then:

        F[e₁] @ F[e₂]  ≈  I_d                              (transport inverse)

    These are verified in ``test/nn/backbones/graph/test_conn_nsd.py``.
    """
    tangent_basis = local_tangent_basis(
        node_features, edge_index, stalk_dim
    )  # [N, p, d]
    src, dst = edge_index[0], edge_index[1]
    return optimal_alignment(
        tangent_basis[src], tangent_basis[dst]
    )  # [E, d, d]
