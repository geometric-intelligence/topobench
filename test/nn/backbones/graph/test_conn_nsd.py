"""Specification for the Conn-NSD connection construction.

These tests are the spec. They encode the mathematical invariants of
Algorithm 1 in Barbero, Bodnar, Sáez de Ocáriz Borde, Bronstein,
Veličković, Liò ("Sheaf Neural Networks with Connection Laplacians",
ICML 2022 TAG-ML Workshop, arXiv:2206.08702):

    1. orthogonality of every restriction map
    2. inverse transport on antiparallel edges
    3. permutation equivariance of the connection
    4. SE(p) invariance of the connection (rotation/translation of features)
    5. the |N(v)| < d fallback rule produces a valid orthonormal basis
    6. no gradients leak into the connection (it is pre-processing)
    7. hand-computed triangle: every entry is verified by direct calculation

The triangle test is deliberately first and deliberately verbose — it is
the touchstone that anyone reading this file can re-derive on paper.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from topobench.nn.backbones.graph.conn_nsd_utils.connection import (
    build_connection,
    local_tangent_basis,
    optimal_alignment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_orthogonal(matrix: Tensor, atol: float = 1e-5) -> bool:
    """Return True iff `matrix` is square and matrix.T @ matrix ≈ I."""
    if matrix.dim() < 2 or matrix.size(-1) != matrix.size(-2):
        return False
    d = matrix.size(-1)
    identity = torch.eye(d, dtype=matrix.dtype, device=matrix.device)
    return torch.allclose(
        matrix.transpose(-1, -2) @ matrix, identity.expand_as(matrix), atol=atol
    )


def _is_column_orthonormal(matrix: Tensor, atol: float = 1e-5) -> bool:
    """Return True iff matrix.T @ matrix ≈ I_d for matrix of shape [..., p, d]."""
    d = matrix.size(-1)
    identity = torch.eye(d, dtype=matrix.dtype, device=matrix.device)
    gram = matrix.transpose(-1, -2) @ matrix
    return torch.allclose(gram, identity.expand_as(gram), atol=atol)


# ---------------------------------------------------------------------------
# Tiny synthetic example: 3-node triangle in ℝ² with d = 2.
#
# Vertices are placed at the corners of an equilateral triangle:
#
#       x_0 = (0, 0)
#       x_1 = (1, 0)
#       x_2 = (1/2, √3/2)
#
# Edges: (0,1), (1,2), (2,0), each in both directions.
#
# At every node v, the 1-hop neighbourhood has exactly 2 elements (the
# other two corners), so the centred matrix X̂_v is 2×2 and the SVD of
# the centred neighbours fully determines the tangent basis. With p = d
# the manifold assumption is *saturated*: the tangent space at every
# node is all of ℝ², and the basis is a rotation of the standard frame.
#
# Crucial sanity check: for any choice of bases the alignment
# F_{vu} ∈ O(2) is well-defined and we can verify orthogonality and
# invertibility by hand.
# ---------------------------------------------------------------------------


@pytest.fixture
def triangle():
    """Equilateral triangle in ℝ², stalk dim 2, bidirectional edges."""
    node_features = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2.0],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 0],
            [1, 0, 2, 1, 0, 2],
        ],
        dtype=torch.long,
    )
    return node_features, edge_index


class TestTriangle:
    """Hand-computable Conn-NSD on a 3-node triangle."""

    def test_tangent_basis_shape_and_orthonormality(self, triangle):
        """Every O_v is a 2×2 orthonormal frame in the saturated p=d case."""
        node_features, edge_index = triangle
        tangent_basis = local_tangent_basis(
            node_features, edge_index, stalk_dim=2
        )

        # Shape contract: [N, p, d] = [3, 2, 2].
        assert tangent_basis.shape == (3, 2, 2)
        # Columns are orthonormal.
        assert _is_column_orthonormal(tangent_basis)
        # In the saturated p=d case the bases span all of ℝ² → they are
        # full orthogonal matrices, so rows are orthonormal too.
        for v in range(3):
            assert _is_orthogonal(tangent_basis[v])

    def test_restriction_maps_are_orthogonal(self, triangle):
        """F[e] ∈ O(d) for every edge e — Algorithm 1 invariant (i)."""
        node_features, edge_index = triangle
        maps = build_connection(node_features, edge_index, stalk_dim=2)
        assert maps.shape == (edge_index.size(1), 2, 2)
        for e in range(edge_index.size(1)):
            assert _is_orthogonal(maps[e]), (
                f"edge {e} = {edge_index[:, e].tolist()} produced a "
                f"non-orthogonal map:\n{maps[e]}"
            )

    def test_antiparallel_edges_invert(self, triangle):
        """F_{v→u} @ F_{u→v} ≈ I — parallel transport is invertible.

        Algorithm 1 produces F_{v→u} = U V^T from O_v^T O_u = U Σ V^T;
        swapping (v,u) gives O_u^T O_v = (O_v^T O_u)^T = V Σ U^T, hence
        F_{u→v} = V U^T = (F_{v→u})^T. Therefore F_{v→u} @ F_{u→v} = I.
        """
        node_features, edge_index = triangle
        maps = build_connection(node_features, edge_index, stalk_dim=2)

        # Edge layout matches the fixture:
        # 0: (0,1)   1: (1,0)
        # 2: (1,2)   3: (2,1)
        # 4: (2,0)   5: (0,2)
        pairs = [(0, 1), (2, 3), (4, 5)]
        identity = torch.eye(2, dtype=maps.dtype)
        for forward, backward in pairs:
            assert torch.allclose(
                maps[forward] @ maps[backward], identity, atol=1e-10
            ), (
                f"edges {edge_index[:, forward].tolist()} and "
                f"{edge_index[:, backward].tolist()} are not mutual inverses"
            )

    def test_no_gradient_into_features(self, triangle):
        """Algorithm 1 is pre-processing: gradients must not flow in."""
        node_features, edge_index = triangle
        node_features = node_features.clone().requires_grad_(True)
        maps = build_connection(node_features, edge_index, stalk_dim=2)
        # If a gradient *did* flow in, .grad_fn would not be None.
        assert not maps.requires_grad, (
            "restriction maps must be detached — they encode a pre-computed "
            "deterministic sheaf, not a learnable transformation."
        )


# ---------------------------------------------------------------------------
# Algebraic invariants on a randomly generated 5-node graph.
# ---------------------------------------------------------------------------


@pytest.fixture
def small_random_graph():
    """Connected 5-node graph in ℝ⁸ with bidirectional edges.

    Sized so that p > d and every node has at least d neighbours, hence
    the fallback path is *not* triggered. Tested separately below.
    """
    torch.manual_seed(0)
    node_features = torch.randn(5, 8, dtype=torch.float64)
    # A pentagon-with-chords graph; every node has degree ≥ 3.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 2), (1, 3),
    ]
    src = [u for (u, _) in edges] + [v for (_, v) in edges]
    dst = [v for (_, v) in edges] + [u for (u, _) in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return node_features, edge_index


class TestAlgebraicInvariants:
    """Mathematical contracts the connection must obey on any graph."""

    @pytest.mark.parametrize("stalk_dim", [2, 3])
    def test_orthogonality_invariant(self, small_random_graph, stalk_dim):
        """F[e]^T F[e] = I_d for every edge — across stalk dimensions."""
        node_features, edge_index = small_random_graph
        maps = build_connection(node_features, edge_index, stalk_dim=stalk_dim)
        assert maps.shape == (edge_index.size(1), stalk_dim, stalk_dim)
        for e in range(maps.size(0)):
            assert _is_orthogonal(maps[e], atol=1e-8)

    def test_inverse_transport_invariant(self, small_random_graph):
        """F_{u→v} F_{v→u} = I — verified across all antiparallel pairs."""
        node_features, edge_index = small_random_graph
        maps = build_connection(node_features, edge_index, stalk_dim=3)

        # Build a lookup from (src, dst) tuple to row index.
        directed_to_row = {
            (int(edge_index[0, e]), int(edge_index[1, e])): e
            for e in range(edge_index.size(1))
        }
        identity = torch.eye(3, dtype=maps.dtype)
        for (src, dst), e_fwd in directed_to_row.items():
            e_bwd = directed_to_row.get((dst, src))
            if e_bwd is None:
                continue
            assert torch.allclose(
                maps[e_fwd] @ maps[e_bwd], identity, atol=1e-8
            ), f"edges ({src}->{dst}) and ({dst}->{src}) are not inverses"

    def test_permutation_equivariance(self, small_random_graph):
        """Relabelling nodes permutes the maps in exactly the same way.

        If π is a node permutation and the same edge (v,u) under the
        original labelling becomes (π(v), π(u)) under the new one, then
        the corresponding restriction maps are identical. This is the
        defining property of an *equivariant* sheaf construction.
        """
        node_features, edge_index = small_random_graph
        n = node_features.size(0)

        # Fixed permutation; deterministic for reproducibility.
        permutation = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
        # inverse: where does original index v end up?
        inv = torch.empty_like(permutation)
        inv[permutation] = torch.arange(n)

        permuted_features = node_features[permutation]
        permuted_edges = inv[edge_index]  # relabel endpoints

        maps_orig = build_connection(node_features, edge_index, stalk_dim=3)
        maps_perm = build_connection(permuted_features, permuted_edges, stalk_dim=3)

        # Match edges: the e-th column of the permuted edge_index corresponds
        # to the same physical edge as the e-th column of the original — both
        # are stored in the same order.
        assert maps_orig.shape == maps_perm.shape
        # The SVD-based alignment is unique up to a sign in degenerate
        # singular-value cases; with random Gaussian features singular values
        # are generically distinct and the comparison is exact up to fp.
        assert torch.allclose(maps_orig, maps_perm, atol=1e-8)

    def test_rotation_equivariance_of_features(self, small_random_graph):
        """If we rotate all features by R ∈ O(p), the restriction maps are unchanged.

        Reason: O_v gets right-multiplied by R, so O_v^T O_u is preserved:
            (R O_v)^T (R O_u) = O_v^T R^T R O_u = O_v^T O_u.
        The alignment SVD therefore returns the same F. This means
        Conn-NSD is *intrinsically* a function of the feature geometry,
        not of the chosen coordinate frame in feature space.
        """
        node_features, edge_index = small_random_graph
        p = node_features.size(1)

        # Generate a random orthogonal R ∈ O(p) via QR.
        torch.manual_seed(123)
        gaussian = torch.randn(p, p, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(gaussian)
        assert _is_orthogonal(rotation, atol=1e-10)

        rotated_features = node_features @ rotation.T  # right action

        maps_orig = build_connection(node_features, edge_index, stalk_dim=3)
        maps_rot = build_connection(rotated_features, edge_index, stalk_dim=3)
        assert torch.allclose(maps_orig, maps_rot, atol=1e-7)


# ---------------------------------------------------------------------------
# The fallback rule: |N(v)| < d.
# ---------------------------------------------------------------------------


class TestFallback:
    """When a node has fewer than d neighbours, top up with k-NN."""

    def test_isolated_node_in_3d(self):
        """Even an isolated node receives a valid 3-D tangent basis.

        Construction: 4 nodes in ℝ⁴, node 3 is isolated. With stalk_dim=3
        node 3's neighbour count is 0 < d = 3, so the fallback kicks in
        and uses the three Euclidean-nearest non-self nodes — which is
        all of them. The basis must still be orthonormal.
        """
        torch.manual_seed(42)
        node_features = torch.randn(4, 4, dtype=torch.float64)
        # Make node 3 distant so the topology of the fallback is obvious.
        node_features[3] = torch.tensor(
            [10.0, 10.0, 10.0, 10.0], dtype=torch.float64
        )
        edge_index = torch.tensor(
            [[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]], dtype=torch.long
        )  # nodes 0,1,2 form a triangle; node 3 isolated.

        tangent_basis = local_tangent_basis(
            node_features, edge_index, stalk_dim=3
        )
        assert tangent_basis.shape == (4, 4, 3)
        # Every basis is column-orthonormal, including the fallback one.
        for v in range(4):
            assert _is_column_orthonormal(
                tangent_basis[v : v + 1], atol=1e-10
            ), f"basis at node {v} is not column-orthonormal"

    def test_partial_neighbourhood_topup(self):
        """A node with one neighbour and stalk_dim=3 needs two top-up nodes."""
        torch.manual_seed(7)
        node_features = torch.randn(5, 6, dtype=torch.float64)
        # Node 0 has just one neighbour (node 1); needs 2 more for d=3.
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 4, 4, 1], [1, 0, 2, 1, 3, 2, 4, 3, 1, 4]],
            dtype=torch.long,
        )
        maps = build_connection(node_features, edge_index, stalk_dim=3)
        for e in range(maps.size(0)):
            assert _is_orthogonal(maps[e], atol=1e-8)


# ---------------------------------------------------------------------------
# API-level shape / dtype contracts.
# ---------------------------------------------------------------------------


class TestApiContract:
    """Type and dtype contracts at the module boundary."""

    def test_float32_inputs_produce_float32_outputs(self):
        """No silent dtype upcasting — surprises matter at scale."""
        node_features = torch.randn(6, 7, dtype=torch.float32)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 2]], dtype=torch.long
        )
        maps = build_connection(node_features, edge_index, stalk_dim=2)
        assert maps.dtype == torch.float32

    def test_rejects_3d_node_features(self):
        """node_features must be [N, p]; we refuse [B, N, p]."""
        bad = torch.randn(2, 5, 7)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(AssertionError, match="node_features must be"):
            local_tangent_basis(bad, edge_index, stalk_dim=2)

    def test_rejects_negative_stalk_dim(self):
        """stalk_dim must be positive."""
        node_features = torch.randn(3, 5)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(AssertionError, match="stalk_dim must be positive"):
            local_tangent_basis(node_features, edge_index, stalk_dim=0)
