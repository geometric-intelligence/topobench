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
from unittest.mock import Mock

import pytest
import torch
from torch import Tensor

from topobench.nn.backbones.graph.conn_nsd import ConnNSDEncoder
from topobench.nn.backbones.graph.conn_nsd_utils.connection import (
    build_connection,
    local_tangent_basis,
)
from topobench.nn.backbones.graph.conn_nsd_utils.fixed_laplacian_builder import (
    FixedConnectionLaplacianBuilder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_orthogonal(matrix: Tensor, atol: float = 1e-5) -> bool:
    """Return True iff `matrix` is square and matrix.T @ matrix ≈ I.

    Parameters
    ----------
    matrix : torch.Tensor
        Candidate matrix; need not be square.
    atol : float, default 1e-5
        Absolute tolerance for the equality check.

    Returns
    -------
    bool
        Whether the matrix is orthogonal within ``atol``.
    """
    if matrix.dim() < 2 or matrix.size(-1) != matrix.size(-2):
        return False
    d = matrix.size(-1)
    identity = torch.eye(d, dtype=matrix.dtype, device=matrix.device)
    return torch.allclose(
        matrix.transpose(-1, -2) @ matrix,
        identity.expand_as(matrix),
        atol=atol,
    )


def _is_column_orthonormal(matrix: Tensor, atol: float = 1e-5) -> bool:
    """Return True iff matrix.T @ matrix ≈ I_d for matrix of shape [..., p, d].

    Parameters
    ----------
    matrix : torch.Tensor, shape ``[..., p, d]``
        Candidate basis matrix; only the trailing two dims matter.
    atol : float, default 1e-5
        Absolute tolerance for the equality check.

    Returns
    -------
    bool
        Whether columns are orthonormal within ``atol``.
    """
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
    """Equilateral triangle in ℝ², stalk dim 2, bidirectional edges.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(node_features, edge_index)`` for the equilateral triangle.
    """
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
        """Every O_v is a 2×2 orthonormal frame in the saturated p=d case.

        Parameters
        ----------
        triangle : tuple[torch.Tensor, torch.Tensor]
            Test fixture supplying ``(node_features, edge_index)``.
        """
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
        """F[e] ∈ O(d) for every edge e — Algorithm 1 invariant (i).

        Parameters
        ----------
        triangle : tuple[torch.Tensor, torch.Tensor]
            Test fixture supplying ``(node_features, edge_index)``.
        """
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

        Parameters
        ----------
        triangle : tuple[torch.Tensor, torch.Tensor]
            Test fixture supplying ``(node_features, edge_index)``.
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
        """Algorithm 1 is pre-processing: gradients must not flow in.

        Parameters
        ----------
        triangle : tuple[torch.Tensor, torch.Tensor]
            Test fixture supplying ``(node_features, edge_index)``.
        """
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


def test_fixed_laplacian_uses_algorithm_transport_directly():
    """The off-diagonal block is the Algorithm 1 transport, not its square.

    ``build_connection`` returns ``O_ij`` and ``O_ji = O_ij.T`` for the two
    orientations of an edge.  For the block at row ``i``, column ``j``, the
    connection Laplacian therefore contains ``-O_ij``.  Treating ``O_ij``
    and ``O_ji`` as independent endpoint-to-edge restriction maps would
    instead form ``-O_ij.T @ O_ji`` and incorrectly square the transport.
    """
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    transport_01 = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64)
    transports = torch.stack((transport_01, transport_01.T))

    builder = FixedConnectionLaplacianBuilder(2, edge_index, d=2)
    (indices, values), _ = builder(transports)
    actual = torch.sparse_coo_tensor(indices, values, (4, 4)).to_dense()

    # Paper Definition 2.5 uses D = diag(L_F), with no added self-loop.
    # Both vertices have degree one, so normalisation leaves these blocks
    # unchanged.
    expected = torch.zeros((4, 4), dtype=torch.float64)
    expected[:2, :2] = torch.eye(2, dtype=torch.float64)
    expected[2:, 2:] = torch.eye(2, dtype=torch.float64)
    expected[:2, 2:] = -transport_01
    expected[2:, :2] = -transport_01.T

    assert torch.allclose(actual, expected, atol=1e-12)


def _dense_sheaf_laplacian(
    restriction_maps: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    stalk_dim: int,
) -> Tensor:
    """Assemble the un-normalised sheaf Laplacian ``L_F = δ^⊤ δ`` densely.

    For the connection representation used by Algorithm 1 this is the block
    matrix:

        L[v, v] = deg(v) · I_d
        L[v, u] = − O_{vu}              (for v ≠ u, when edge (v,u) ∈ E)

    Used only in tests, where transparency beats efficiency.

    Parameters
    ----------
    restriction_maps : torch.Tensor, shape ``[E, d, d]``
        Orthogonal restriction maps, one per directed edge.
    edge_index : torch.Tensor, shape ``[2, E]``
        Edge index in PyG convention (directed, bidirectional pairs).
    num_nodes : int
        Number of nodes in the graph.
    stalk_dim : int
        Dimension ``d`` of each stalk.

    Returns
    -------
    torch.Tensor, shape ``[N·d, N·d]``
        Dense sheaf Laplacian, symmetric and positive semi-definite.
    """
    d = stalk_dim
    dim = num_nodes * d
    laplacian = torch.zeros(dim, dim, dtype=restriction_maps.dtype)

    directed = {
        (int(edge_index[0, e]), int(edge_index[1, e])): e
        for e in range(edge_index.size(1))
    }

    # Diagonal: deg(v) · I_d, where deg counts directed out-edges from v.
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for v in edge_index[0].tolist():
        degrees[v] += 1
    for v in range(num_nodes):
        rng = slice(v * d, (v + 1) * d)
        laplacian[rng, rng] = degrees[v].to(laplacian.dtype) * torch.eye(
            d, dtype=laplacian.dtype
        )

    # Off-diagonal blocks for every unordered edge with both directions present.
    seen: set[frozenset[int]] = set()
    for (v, u), e_fwd in directed.items():
        if v == u or frozenset({v, u}) in seen:
            continue
        e_bwd = directed.get((u, v))
        if e_bwd is None:
            continue
        block = -restriction_maps[e_fwd]
        rv = slice(v * d, (v + 1) * d)
        ru = slice(u * d, (u + 1) * d)
        laplacian[rv, ru] = block
        laplacian[ru, rv] = block.T
        seen.add(frozenset({v, u}))

    return laplacian


@pytest.fixture
def small_random_graph():
    """Connected 5-node graph in ℝ⁸ with bidirectional edges.

    Sized so that p > d and every node has at least d neighbours, hence
    the fallback path is *not* triggered. Tested separately below.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(node_features, edge_index)`` for the pentagon-with-chords graph.
    """
    torch.manual_seed(0)
    node_features = torch.randn(5, 8, dtype=torch.float64)
    # A pentagon-with-chords graph; every node has degree ≥ 3.
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 2),
        (1, 3),
    ]
    src = [u for (u, _) in edges] + [v for (_, v) in edges]
    dst = [v for (_, v) in edges] + [u for (u, _) in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return node_features, edge_index


class TestAlgebraicInvariants:
    """Mathematical contracts the connection must obey on any graph."""

    @pytest.mark.parametrize("stalk_dim", [2, 3])
    def test_orthogonality_invariant(self, small_random_graph, stalk_dim):
        """F[e]^T F[e] = I_d for every edge — across stalk dimensions.

        Parameters
        ----------
        small_random_graph : tuple[torch.Tensor, torch.Tensor]
            Test fixture supplying ``(node_features, edge_index)``.
        stalk_dim : int
            Parametrised stalk dimension.
        """
        node_features, edge_index = small_random_graph
        maps = build_connection(node_features, edge_index, stalk_dim=stalk_dim)
        assert maps.shape == (edge_index.size(1), stalk_dim, stalk_dim)
        for e in range(maps.size(0)):
            assert _is_orthogonal(maps[e], atol=1e-8)

    def test_inverse_transport_invariant(self, small_random_graph):
        """F_{u→v} F_{v→u} = I — verified across all antiparallel pairs.

        Parameters
        ----------
        small_random_graph : tuple[torch.Tensor, torch.Tensor]
            Test fixture supplying ``(node_features, edge_index)``.
        """
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

    def test_permutation_invariance_of_laplacian_spectrum(
        self, small_random_graph
    ):
        """The sheaf-Laplacian spectrum is invariant under node relabelling.

        Subtlety
        --------
        Local PCA bases can differ by node-wise orthogonal gauges ``Q_v``.
        Algorithm 1 then transforms ``O_vu`` to ``Q_v^T O_vu Q_u``, which
        is a block-diagonal orthogonal conjugation of the connection
        Laplacian. Its entries are gauge-dependent, but its spectrum is not.

        Parameters
        ----------
        small_random_graph : tuple[torch.Tensor, torch.Tensor]
            Test fixture supplying ``(node_features, edge_index)``.
        """
        node_features, edge_index = small_random_graph
        n = node_features.size(0)
        d = 3

        permutation = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
        inv = torch.empty_like(permutation)
        inv[permutation] = torch.arange(n)
        permuted_features = node_features[permutation]
        permuted_edges = inv[edge_index]

        maps_orig = build_connection(node_features, edge_index, stalk_dim=d)
        maps_perm = build_connection(
            permuted_features, permuted_edges, stalk_dim=d
        )

        l_orig = _dense_sheaf_laplacian(maps_orig, edge_index, n, d)
        l_perm = _dense_sheaf_laplacian(maps_perm, permuted_edges, n, d)

        eig_orig = torch.linalg.eigvalsh(l_orig).sort().values
        eig_perm = torch.linalg.eigvalsh(l_perm).sort().values
        assert torch.allclose(eig_orig, eig_perm, atol=1e-8), (
            "Connection-Laplacian spectrum changed under node relabelling."
        )

    def test_fixed_normalised_laplacian_is_symmetric_psd(
        self, small_random_graph
    ):
        """Paper Definition 2.5 yields a symmetric PSD matrix.

        Parameters
        ----------
        small_random_graph : tuple[torch.Tensor, torch.Tensor]
            Test fixture supplying ``(node_features, edge_index)``.
        """
        node_features, edge_index = small_random_graph
        maps = build_connection(node_features, edge_index, stalk_dim=3)
        builder = FixedConnectionLaplacianBuilder(
            node_features.size(0), edge_index, d=3
        )
        (indices, values), _ = builder(maps)
        laplacian = torch.sparse_coo_tensor(
            indices, values, (15, 15)
        ).to_dense()

        assert torch.allclose(laplacian, laplacian.T, atol=1e-12)
        assert torch.linalg.eigvalsh(laplacian).min() >= -1e-8

    def test_determinism(self, small_random_graph):
        """Algorithm 1 is deterministic: same input → same output.

        Two calls on identical tensors must produce bit-identical maps.
        This is the strongest form of reproducibility available given the
        SVD sign gauge, and is what the diffusion actually needs at
        training time.

        Parameters
        ----------
        small_random_graph : tuple[torch.Tensor, torch.Tensor]
            Test fixture supplying ``(node_features, edge_index)``.
        """
        node_features, edge_index = small_random_graph
        maps_a = build_connection(node_features, edge_index, stalk_dim=3)
        maps_b = build_connection(node_features, edge_index, stalk_dim=3)
        assert torch.equal(maps_a, maps_b), (
            "build_connection is not deterministic on identical inputs — "
            "the diffusion would see a different operator each forward pass."
        )


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

    def test_fallback_respects_batch_boundaries(self):
        """Fallback nearest neighbours must not cross PyG graph boundaries."""
        torch.manual_seed(0)
        node_features = torch.randn(5, 5)
        node_features[2:] = node_features[2:] + 50.0
        edge_index = torch.tensor(
            [[0, 1, 1, 0, 2, 3, 3, 2], [1, 0, 0, 1, 3, 2, 2, 3]],
            dtype=torch.long,
        )
        second_graph_features = node_features + 0.01
        batched_features = torch.cat(
            [node_features, second_graph_features], dim=0
        )
        batched_edges = torch.cat([edge_index, edge_index + 5], dim=1)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

        single = build_connection(node_features, edge_index, stalk_dim=4)
        batched = build_connection(
            batched_features,
            batched_edges,
            stalk_dim=4,
            batch=batch,
        )

        assert torch.allclose(single, batched[: edge_index.size(1)], atol=1e-6)

    @pytest.mark.parametrize("num_nodes,stalk_dim", [(3, 3), (3, 4)])
    def test_too_small_graph_raises_clear_error(self, num_nodes, stalk_dim):
        """A graph must have at least d + 1 nodes for the fallback rule.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the test graph.
        stalk_dim : int
            Tangent-space dimension requested from the fallback rule.
        """
        node_features = torch.randn(num_nodes, 5)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(ValueError, match="graph too small"):
            local_tangent_basis(node_features, edge_index, stalk_dim=stalk_dim)


class TestBatching:
    """Conn-NSD must be invariant to PyG mini-batch grouping."""

    def test_encoder_matches_unbatched_outputs_with_low_degree_fallback(self):
        """The same graph alone or inside a batch receives the same output."""
        torch.manual_seed(0)
        model = ConnNSDEncoder(
            input_dim=5,
            hidden_dim=8,
            stalk_dim=4,
            num_layers=1,
            dropout=0.0,
            input_dropout=0.0,
        ).eval()

        node_features = torch.randn(5, 5)
        node_features[2:] = node_features[2:] + 50.0
        edge_index = torch.tensor(
            [[0, 1, 1, 0, 2, 3, 3, 2], [1, 0, 0, 1, 3, 2, 2, 3]],
            dtype=torch.long,
        )
        second_graph_features = node_features + 0.01
        batched_features = torch.cat(
            [node_features, second_graph_features], dim=0
        )
        batched_edges = torch.cat([edge_index, edge_index + 5], dim=1)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)

        with torch.no_grad():
            single = model(
                node_features,
                edge_index,
                batch=torch.zeros(5, dtype=torch.long),
            )
            batched = model(batched_features, batched_edges, batch=batch)[:5]

        assert torch.allclose(single, batched, atol=1e-6)

    def test_connection_is_preprocessed_once_per_raw_graph(self, monkeypatch):
        """Repeated optimizer steps reuse the fixed Algorithm 1 connection.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Patch fixture used to count Algorithm 1 executions.
        """
        torch.manual_seed(0)
        model = ConnNSDEncoder(
            input_dim=5,
            hidden_dim=8,
            stalk_dim=2,
            num_layers=1,
            dropout=0.0,
            input_dropout=0.0,
        ).eval()
        raw_features = torch.randn(5, 5)
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                [1, 0, 2, 1, 3, 2, 4, 3, 0, 4],
            ],
            dtype=torch.long,
        )

        counted_build_connection = Mock(wraps=build_connection)

        monkeypatch.setattr(
            "topobench.nn.backbones.graph.conn_nsd.build_connection",
            counted_build_connection,
        )

        batch = torch.zeros(5, dtype=torch.long)
        with torch.no_grad():
            model(
                raw_features,
                edge_index,
                batch=batch,
                connection_x=raw_features,
            )
            model(
                raw_features + 0.1,
                edge_index,
                batch=batch,
                connection_x=raw_features,
            )

        assert counted_build_connection.call_count == 1

    def test_connection_cache_survives_minibatch_reshuffling(
        self, monkeypatch
    ):
        """Each graph is cached independently of its mini-batch position.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Patch fixture used to count Algorithm 1 executions.
        """
        torch.manual_seed(0)
        model = ConnNSDEncoder(
            input_dim=5,
            hidden_dim=8,
            stalk_dim=2,
            num_layers=1,
            dropout=0.0,
            input_dropout=0.0,
        ).eval()
        graph_a = torch.randn(3, 5)
        graph_b = torch.randn(3, 5)
        triangle_edges = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        features = torch.cat((graph_a, graph_b))
        edges = torch.cat((triangle_edges, triangle_edges + 3), dim=1)
        batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        counted_build_connection = Mock(wraps=build_connection)

        monkeypatch.setattr(
            "topobench.nn.backbones.graph.conn_nsd.build_connection",
            counted_build_connection,
        )

        with torch.no_grad():
            model(features, edges, batch=batch, connection_x=features)
        assert counted_build_connection.call_count == 2

        permutation = torch.tensor([3, 4, 5, 0, 1, 2])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(6)
        shuffled_features = features[permutation]
        shuffled_edges = inverse[edges]
        with torch.no_grad():
            model(
                shuffled_features,
                shuffled_edges,
                batch=batch,
                connection_x=shuffled_features,
            )

        assert counted_build_connection.call_count == 2

    def test_connection_cache_is_shared_across_training_seeds(
        self, monkeypatch
    ):
        """Repeated model instances reuse identical graph preprocessing.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Patch fixture used to count Algorithm 1 executions.
        """
        torch.manual_seed(0)
        model_a = ConnNSDEncoder(
            input_dim=5, hidden_dim=8, stalk_dim=2, num_layers=1
        ).eval()
        model_b = ConnNSDEncoder(
            input_dim=5, hidden_dim=8, stalk_dim=2, num_layers=1
        ).eval()
        model_a._connection_cache.clear()
        features = torch.randn(3, 5)
        edges = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        batch = torch.zeros(3, dtype=torch.long)
        counted_build_connection = Mock(wraps=build_connection)
        monkeypatch.setattr(
            "topobench.nn.backbones.graph.conn_nsd.build_connection",
            counted_build_connection,
        )

        with torch.no_grad():
            model_a(features, edges, batch=batch, connection_x=features)
            model_b(features, edges, batch=batch, connection_x=features)

        assert counted_build_connection.call_count == 1

    def test_connection_cache_isolated_by_stalk_dimension(self, monkeypatch):
        """Paper variants with different ``d`` never share cached maps.

        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Patch fixture used to count Algorithm 1 executions.
        """
        torch.manual_seed(0)
        model_d2 = ConnNSDEncoder(
            input_dim=5, hidden_dim=8, stalk_dim=2, num_layers=1
        ).eval()
        model_d3 = ConnNSDEncoder(
            input_dim=5, hidden_dim=9, stalk_dim=3, num_layers=1
        ).eval()
        model_d2._connection_cache.clear()
        features = torch.randn(4, 5)
        edges = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 0, 0, 2],
                [1, 0, 2, 1, 3, 2, 0, 3, 2, 0],
            ],
            dtype=torch.long,
        )
        batch = torch.zeros(4, dtype=torch.long)
        counted_build_connection = Mock(wraps=build_connection)
        monkeypatch.setattr(
            "topobench.nn.backbones.graph.conn_nsd.build_connection",
            counted_build_connection,
        )

        with torch.no_grad():
            model_d2(features, edges, batch=batch, connection_x=features)
            model_d3(features, edges, batch=batch, connection_x=features)

        assert counted_build_connection.call_count == 2


# ---------------------------------------------------------------------------
# API-level shape / dtype contracts.
# ---------------------------------------------------------------------------


class TestApiContract:
    """Type and dtype contracts at the module boundary."""

    def test_diffusion_has_only_paper_equation_parameters(self):
        """Equation 5 has learnable ``W1`` and ``W2``, but no residual gate."""
        model = ConnNSDEncoder(
            input_dim=4, hidden_dim=4, stalk_dim=2, num_layers=2
        )

        parameter_names = set(dict(model.named_parameters()))
        assert parameter_names == {
            "lin1.weight",
            "lin1.bias",
            "lin2.weight",
            "lin2.bias",
            "lin_left_weights.0.weight",
            "lin_left_weights.1.weight",
            "lin_right_weights.0.weight",
            "lin_right_weights.1.weight",
        }

    def test_float32_inputs_produce_float32_outputs(self):
        """No silent dtype upcasting — surprises matter at scale."""
        node_features = torch.randn(6, 7, dtype=torch.float32)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0, 2]], dtype=torch.long
        )
        maps = build_connection(node_features, edge_index, stalk_dim=2)
        assert maps.dtype == torch.float32

    def test_rejects_3d_node_features(self):
        """Reject ``node_features`` of shape ``[B, N, p]``."""
        bad = torch.randn(2, 5, 7)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(AssertionError, match="node_features must be"):
            local_tangent_basis(bad, edge_index, stalk_dim=2)

    def test_rejects_negative_stalk_dim(self):
        """Reject ``stalk_dim`` ≤ 0."""
        node_features = torch.randn(3, 5)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        with pytest.raises(AssertionError, match="stalk_dim must be positive"):
            local_tangent_basis(node_features, edge_index, stalk_dim=0)

    def test_rejects_ambient_dimension_below_stalk_dimension(self):
        """Algorithm 1 requires ambient feature dimension ``p >= d``."""
        node_features = torch.randn(4, 2)
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
            dtype=torch.long,
        )
        with pytest.raises(ValueError, match="ambient feature dimension"):
            local_tangent_basis(node_features, edge_index, stalk_dim=3)
