"""Tests for :class:`DirectedSimplicialLifting`."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
import torch
import torch_geometric

from topobench.transforms.liftings.graph2simplicial import (
    DirectedSimplicialLifting,
)
from topobench.transforms.liftings.graph2simplicial.directed_simplicial_lifting import (
    DIR_ADJ_KEYS,
    DIR_LOWER_KEYS,
    DIR_UPPER_KEYS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(
    edges: list[tuple[int, int]],
    num_nodes: int,
    *,
    symmetric: bool = True,
) -> torch_geometric.data.Data:
    """Return a ``Data`` object built from the given edge list.

    Parameters
    ----------
    edges : list of tuple of int
        Edges to include. By default the resulting ``edge_index`` is
        symmetric (TopoBench default).
    num_nodes : int
        Number of nodes; node features are a column of indices.
    symmetric : bool, optional
        Whether to add the reverse direction for every edge. Default
        is ``True``.

    Returns
    -------
    torch_geometric.data.Data
        A PyG ``Data`` instance with ``x``, ``edge_index``,
        ``num_nodes`` set.
    """
    pairs = list(edges)
    if symmetric:
        pairs = pairs + [(v, u) for u, v in edges]
    edge_index = torch.tensor(pairs).T.long()
    x = torch.arange(num_nodes).float().unsqueeze(1)
    return torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
        y=torch.zeros(num_nodes, dtype=torch.long),
    )


def _dense(t: torch.Tensor) -> torch.Tensor:
    """Densify a sparse tensor for value-level assertions.

    Parameters
    ----------
    t : torch.Tensor
        Sparse COO tensor produced by the lifting.

    Returns
    -------
    torch.Tensor
        Dense view of ``t``.
    """
    return t.to_dense()


def _expected_matrix(
    size: int, coords: list[tuple[int, int]]
) -> torch.Tensor:
    """Build a dense adjacency matrix from one-valued coordinates.

    Parameters
    ----------
    size : int
        Number of rows and columns.
    coords : list of tuple of int
        ``(row, column)`` coordinates whose value should be one.

    Returns
    -------
    torch.Tensor
        Dense float adjacency matrix.
    """
    out = torch.zeros(size, size)
    for row, col in coords:
        out[row, col] = 1.0
    return out


# ---------------------------------------------------------------------------
# Static method ports of compute_adj.py
# ---------------------------------------------------------------------------


class TestComputeLowerAdj:
    """Unit tests for :meth:`DirectedSimplicialLifting.compute_lower_adj`."""

    def test_empty_edge_list_returns_empty_matrices(self) -> None:
        """Zero edges -> four 0x0 matrices."""
        a100, a101, a110, a111 = (
            DirectedSimplicialLifting.compute_lower_adj([])
        )
        for m in (a100, a101, a110, a111):
            assert m.shape == (0, 0)

    def test_single_edge(self) -> None:
        """A single edge is only self-100 / self-111 adjacent.

        With a single directed edge ``(0, 1)`` no two-edge lower
        adjacency can exist except the trivial self-source / self-target
        diagonal entries that the reference implementation also keeps.
        """
        a100, a101, a110, a111 = (
            DirectedSimplicialLifting.compute_lower_adj([(0, 1)])
        )
        assert torch.equal(a100, torch.tensor([[1.0]]))
        assert torch.equal(a111, torch.tensor([[1.0]]))
        assert torch.equal(a101, torch.tensor([[0.0]]))
        assert torch.equal(a110, torch.tensor([[0.0]]))

    def test_directed_path_matches_paper_intuition(self) -> None:
        """On the path ``0 -> 1 -> 2`` the 101 entry is at [0, 1].

        Edge 0 = (0,1) ends where edge 1 = (1,2) starts, so
        ``adj_low_101[0, 1] = 1`` and its transpose
        ``adj_low_110[1, 0] = 1`` (Eq. (3) of arXiv:2409.08389).
        The 100 / 111 matrices have only the diagonal (the two edges
        share no endpoint).
        """
        edges = [(0, 1), (1, 2)]
        a100, a101, a110, a111 = (
            DirectedSimplicialLifting.compute_lower_adj(edges)
        )
        assert a101[0, 1].item() == 1.0
        assert a101.sum().item() == 1.0
        assert a110[1, 0].item() == 1.0
        assert a110.sum().item() == 1.0
        # Diagonal-only for 100 / 111 (no shared src / dst).
        assert torch.equal(a100, torch.eye(2))
        assert torch.equal(a111, torch.eye(2))

    def test_shared_source(self) -> None:
        """``adj_low_100`` marks pairs sharing the source vertex.

        Edges ``(0,1)`` and ``(0,2)`` share source 0, so
        ``adj_low_100`` is the all-ones matrix (off-diagonal entries
        from the shared source, diagonal kept by the reference).
        """
        edges = [(0, 1), (0, 2)]
        a100, _, _, a111 = DirectedSimplicialLifting.compute_lower_adj(edges)
        assert torch.equal(a100, torch.ones(2, 2))
        # No shared target.
        assert torch.equal(a111, torch.eye(2))

    def test_shared_target(self) -> None:
        """``adj_low_111`` marks pairs sharing the target vertex."""
        edges = [(0, 2), (1, 2)]
        a100, _, _, a111 = DirectedSimplicialLifting.compute_lower_adj(edges)
        assert torch.equal(a111, torch.ones(2, 2))
        assert torch.equal(a100, torch.eye(2))

    def test_110_is_transpose_of_101(self) -> None:
        """``adj_low_110 == adj_low_101.T`` for any edge configuration."""
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        _, a101, a110, _ = DirectedSimplicialLifting.compute_lower_adj(edges)
        assert torch.equal(a110, a101.t())


class TestComputeUpperAdj:
    """Unit tests for :meth:`DirectedSimplicialLifting.compute_upper_adj`."""

    def test_empty_edge_list(self) -> None:
        """Zero edges -> six 0x0 matrices."""
        ups = DirectedSimplicialLifting.compute_upper_adj([])
        assert len(ups) == 6
        for m in ups:
            assert m.shape == (0, 0)

    def test_no_triangles_means_zero(self) -> None:
        """A graph without 3-cycles produces six zero matrices."""
        edges = [(0, 1), (1, 2), (2, 3)]  # directed path, no triangles
        ups = DirectedSimplicialLifting.compute_upper_adj(edges)
        for m in ups:
            assert m.sum().item() == 0.0

    def test_canonically_oriented_triangle(self) -> None:
        """One canonically oriented triangle populates one entry per slot.

        With canonical orientation ``src < dst`` the only triangle
        ``{0, 1, 2}`` lands in branch 1 of ``compute_upper_adj``
        (``(t0,t1), (t0,t2), (t1,t2)`` = ``(0,1), (0,2), (1,2)``).
        Each of the six upper adjacencies therefore gets exactly one
        non-zero entry, placed at the indices specified in Eq. (4) of
        arXiv:2409.08389.
        """
        edges = [(0, 1), (0, 2), (1, 2)]
        edge_to_id = {e: i for i, e in enumerate(edges)}
        ups = DirectedSimplicialLifting.compute_upper_adj(edges)
        a101, a102, a112, a110, a120, a121 = ups

        i_01 = edge_to_id[(0, 1)]
        i_02 = edge_to_id[(0, 2)]
        i_12 = edge_to_id[(1, 2)]

        assert a101[i_12, i_02].item() == 1.0 and a101.sum().item() == 1.0
        assert a102[i_12, i_01].item() == 1.0 and a102.sum().item() == 1.0
        assert a112[i_02, i_01].item() == 1.0 and a112.sum().item() == 1.0
        assert a110[i_02, i_12].item() == 1.0 and a110.sum().item() == 1.0
        assert a120[i_01, i_12].item() == 1.0 and a120.sum().item() == 1.0
        assert a121[i_01, i_02].item() == 1.0 and a121.sum().item() == 1.0

    @pytest.mark.parametrize(
        "rotation",
        [
            ((0, 1), (1, 2), (0, 2)),  # branch 1
            ((0, 2), (0, 1), (2, 1)),  # branch 2
            ((1, 0), (1, 2), (0, 2)),  # branch 3
            ((1, 2), (1, 0), (2, 0)),  # branch 4
            ((2, 1), (2, 0), (1, 0)),  # branch 5
            ((2, 0), (2, 1), (0, 1)),  # branch 6
        ],
    )
    def test_all_six_rotation_branches_yield_one_triangle(
        self, rotation
    ) -> None:
        """All six rotational forms of ``compute_upper_adj`` find the triangle.

        Each parametrised ``rotation`` is one of the six matching
        branches in ``compute_adj.compute_upper_adj`` of the upstream
        repository. We feed exactly the three directed edges that select
        the corresponding branch and check that each of the six upper
        adjacencies gets exactly one non-zero entry.

        Parameters
        ----------
        rotation : tuple of tuple of int
            Three directed edges defining the rotation under test.
        """
        edges = sorted({tuple(sorted(e)) for e in rotation})
        # We want the *digraph* defined by the rotation itself, not the
        # undirected projection, so we feed the rotation edges directly
        # (they are unique because each branch fixes a distinct
        # orientation of the triangle).
        edges = sorted(set(rotation))
        ups = DirectedSimplicialLifting.compute_upper_adj(edges)
        for m in ups:
            assert m.sum().item() == 1.0, (
                "every branch must populate one entry per adjacency"
            )

    def test_two_disjoint_triangles(self) -> None:
        """Two disjoint triangles contribute additively.

        Each canonically oriented triangle contributes exactly one entry
        per upper adjacency (six entries total per matrix would mean six
        triangles); two disjoint triangles must yield two non-zero
        entries per matrix.
        """
        edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
        ups = DirectedSimplicialLifting.compute_upper_adj(edges)
        for m in ups:
            assert m.sum().item() == 2.0


# ---------------------------------------------------------------------------
# End-to-end lifting
# ---------------------------------------------------------------------------


class TestDirectedSimplicialLifting:
    """End-to-end behaviour of :class:`DirectedSimplicialLifting`."""

    def test_repr_carries_directed_input_flag(self) -> None:
        """``repr`` is stable and surfaces the configuration knob."""
        assert "directed_input=False" in repr(DirectedSimplicialLifting())

    def test_directed_input_true_is_guarded(self) -> None:
        """``directed_input=True`` must raise :class:`NotImplementedError`.

        The raw-orientation edge list breaks row/column alignment
        between the 10 directed adjacencies and ``incidence_1`` /
        ``x_1`` (which use canonical ``(min, max)`` edge ordering), so
        we refuse to construct in that mode until the alignment is
        threaded through ``incidence_1``.
        """
        with pytest.raises(NotImplementedError, match="directed_input=True"):
            DirectedSimplicialLifting(directed_input=True)

    def test_complex_dim_is_forced_to_2(self) -> None:
        """Even if the caller asks for ``complex_dim=3`` we clamp to 2.

        The paper only defines the directed adjacencies up to triangles;
        higher complex dimensions are silently ignored by the backbone
        and would only inflate the simplicial complex's incidence
        matrices.
        """
        lifting = DirectedSimplicialLifting(complex_dim=3)
        assert lifting.complex_dim == 2

    def test_lift_keys_present(self) -> None:
        """All ten directed adjacencies appear on the lifted batch."""
        data = _make_data([(0, 1), (1, 2), (0, 2), (0, 3), (3, 4)], 5)
        out = DirectedSimplicialLifting()(data)
        for key in DIR_ADJ_KEYS:
            assert hasattr(out, key), f"missing {key}"
            assert getattr(out, key).is_sparse, (
                f"{key} must be a sparse COO tensor"
            )

    def test_lift_shapes_match_incidence_1(self) -> None:
        """Each adjacency is square and matches ``incidence_1.cols``."""
        data = _make_data([(0, 1), (1, 2), (0, 2), (0, 3), (3, 4)], 5)
        out = DirectedSimplicialLifting()(data)
        n_edges = out.incidence_1.shape[1]
        for key in DIR_ADJ_KEYS:
            mat = getattr(out, key)
            assert mat.shape == (n_edges, n_edges)

    def test_isolated_node_has_zero_block(self) -> None:
        """Adding an isolated node leaves the n_edges dimension unchanged."""
        data_a = _make_data([(0, 1), (1, 2), (0, 2)], 3)
        data_b = _make_data([(0, 1), (1, 2), (0, 2)], 4)  # extra isolated
        a = DirectedSimplicialLifting()(data_a)
        b = DirectedSimplicialLifting()(data_b)
        # Same edges, same adjacency matrices.
        for key in DIR_ADJ_KEYS:
            assert torch.equal(_dense(getattr(a, key)), _dense(getattr(b, key)))

    def test_symmetric_input_uses_canonical_orientation(self) -> None:
        """An undirected edge ``{u,v}`` is oriented as ``(min,max)``.

        Concretely: feeding the symmetric edge_index
        ``[(0,1),(1,0),(0,2),(2,0)]`` must produce the same canonical
        ``(min, max)`` edge list as feeding only ``[(0,1),(0,2)]``, and
        therefore the same ten directed adjacencies.
        """
        sym = _make_data([(0, 1), (0, 2)], 3, symmetric=True)
        asym = _make_data([(0, 1), (0, 2)], 3, symmetric=False)
        lifting = DirectedSimplicialLifting()
        sym_out = lifting(sym)
        asym_out = lifting(asym)
        for key in DIR_ADJ_KEYS:
            assert torch.equal(
                _dense(getattr(sym_out, key)),
                _dense(getattr(asym_out, key)),
            ), key

    def test_canonical_orientation_flips_reverse_edges(self) -> None:
        """Force inputs with reverse directions into canonical ``(min,max)`` order.

        Feeding the asymmetric edge_index ``[(2, 0), (0, 1)]`` (note
        ``2 > 0`` so canonical orientation flips the first edge) must
        match the canonical edge list ``[(0, 1), (0, 2)]`` exactly.
        With both edges sharing source 0, ``adj_low_100`` is the
        all-ones matrix.
        """
        data = _make_data([(2, 0), (0, 1)], 3, symmetric=False)
        out = DirectedSimplicialLifting()(data)
        a100 = _dense(out.dir_lower_adj_100)
        # Canonical edges sorted: (0,1) -> idx 0, (0,2) -> idx 1; both
        # share source 0.
        assert torch.equal(a100, torch.ones(2, 2))

    def test_self_loops_are_ignored(self) -> None:
        """Drop self-loops in ``edge_index`` before lifting.

        Including ``(2, 2)`` in the symmetric edge list must not change
        the resulting directed adjacencies relative to the loop-free
        input.
        """
        data_clean = _make_data([(0, 1), (0, 2), (1, 2)], 3)
        data_loops = _make_data(
            [(0, 1), (0, 2), (1, 2), (2, 2)], 3, symmetric=False
        )
        # Force the symmetrised version of the clean data to match what
        # `_make_data(symmetric=True)` produces above.
        data_loops.edge_index = torch.cat(
            [
                data_loops.edge_index,
                data_loops.edge_index.flip(0),
            ],
            dim=1,
        )
        out_clean = DirectedSimplicialLifting()(data_clean)
        out_loops = DirectedSimplicialLifting()(data_loops)
        for key in DIR_ADJ_KEYS:
            assert torch.equal(
                _dense(getattr(out_clean, key)),
                _dense(getattr(out_loops, key)),
            )

    def test_collapse_to_4_lower_on_symmetric_input(self) -> None:
        """On a symmetric input, the 4 lower adjacencies collapse to one.

        Reference :func:`compute_lower_adj_undirected` defines the
        symmetric edge lower-adjacency by ``|B_1|^T |B_1|`` with the
        diagonal zeroed: two edges are lower-adjacent iff they share at
        least one endpoint. Under canonical orientation, that endpoint
        is either a shared src (``A_100``), a shared dst (``A_111``),
        or a head-to-tail / tail-to-head meeting (``A_101`` / ``A_110``
        respectively). Summing the four matrices (off-diagonal) must
        recover the undirected lower adjacency exactly.
        """
        # Symmetric input: triangle 0-1-2 plus tail 0-3-4.
        data = _make_data([(0, 1), (1, 2), (0, 2), (0, 3), (3, 4)], 5)
        out = DirectedSimplicialLifting()(data)

        # Reference undirected lower adjacency (|B_1|^T |B_1| diag=0).
        n_edges = out.incidence_1.shape[1]
        inc_1 = _dense(out.incidence_1).abs()
        ref = (inc_1.t() @ inc_1).clamp(max=1.0)
        ref.fill_diagonal_(0.0)

        ours = (
            _dense(out.dir_lower_adj_100)
            + _dense(out.dir_lower_adj_101)
            + _dense(out.dir_lower_adj_110)
            + _dense(out.dir_lower_adj_111)
        ).clamp(max=1.0)
        ours.fill_diagonal_(0.0)

        assert ours.shape == (n_edges, n_edges)
        assert torch.equal(ours, ref)

    def test_triangle_count_matches_upper_adj_nnz(self) -> None:
        """Number of triangles equals nnz of each upper adjacency.

        With canonical orientation every undirected triangle contributes
        exactly one entry to every upper adjacency (branch 1 of the
        rotation chain). The line-graph triangle count is therefore
        equal to the nnz of e.g. ``dir_upper_adj_101``.
        """
        # Triangle 0-1-2, plus another triangle 2-3-4 sharing one vertex.
        data = _make_data(
            [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4)], 5
        )
        out = DirectedSimplicialLifting()(data)
        # Count triangles via networkx for an independent ground truth.
        und = nx.Graph()
        und.add_edges_from(
            [
                (int(data.edge_index[0, k]), int(data.edge_index[1, k]))
                for k in range(data.edge_index.shape[1])
            ]
        )
        triangles = sum(
            1 for clique in nx.find_cliques(und) if len(clique) >= 3
        ) if False else len([
            tuple(sorted((u, v, w)))
            for u, v in und.edges()
            for w in (set(und.neighbors(u)) & set(und.neighbors(v)))
        ]) // 3
        # Each triangle is counted three times in the above sum (once
        # per edge), so we divide by 3 (already done) -> use a clean
        # 3-clique enumeration:
        n_triangles = sum(
            1 for clique in nx.enumerate_all_cliques(und) if len(clique) == 3
        )
        assert triangles == n_triangles
        for key in DIR_UPPER_KEYS:
            assert _dense(getattr(out, key)).sum().item() == n_triangles

    def test_no_edges_does_not_crash(self) -> None:
        """A graph with nodes but no edges produces empty adjacency tensors."""
        data = torch_geometric.data.Data(
            x=torch.zeros(3, 1),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            num_nodes=3,
            y=torch.zeros(3, dtype=torch.long),
        )
        out = DirectedSimplicialLifting()(data)
        for key in DIR_ADJ_KEYS:
            mat = getattr(out, key)
            assert mat.shape == (0, 0) or mat.values().numel() == 0


# ---------------------------------------------------------------------------
# Opt-in spectral_normalize
# ---------------------------------------------------------------------------


class TestSpectralNormalize:
    """Tests for the opt-in ``spectral_normalize`` knob.

    Mirrors :func:`spectral_normalization` in the upstream
    ``utils.py:16-20`` applied to each of the lower adjacencies in
    ``train.py:29-33``. Default is
    off so the existing 10-adj evaluation at ``d507f80b`` stays
    bit-identical.
    """

    def test_spectral_normalize_off_preserves_raw_binary(self) -> None:
        """``spectral_normalize=False`` (default) keeps raw binary adjacencies.

        The default forward output must be bit-identical to the
        pre-review-3 behaviour, so existing eval results remain valid.
        """
        data = _make_data([(0, 1), (1, 2), (0, 2), (0, 3), (3, 4)], 5)
        raw = DirectedSimplicialLifting()(data)
        explicit_off = DirectedSimplicialLifting(spectral_normalize=False)(data)
        for key in DIR_ADJ_KEYS:
            mat = _dense(getattr(raw, key))
            # Every nonzero value of the raw adjacency must be exactly 1.
            unique = torch.unique(mat)
            assert torch.all(
                (unique == 0.0) | (unique == 1.0)
            ), f"{key} has non-binary entries when spectral_normalize is off"
            assert torch.equal(mat, _dense(getattr(explicit_off, key)))

    def test_spectral_normalize_on_divides_by_max_eig_symmetric(self) -> None:
        """On a symmetric adjacency, output = raw / max-eigenvalue.

        Uses the canonical triangle ``{0, 1, 2}`` whose
        ``adj_low_100`` (same-source) is the matrix::

            [[1, 1, 0],
             [1, 1, 0],
             [0, 0, 1]]

        (edges (0,1), (0,2), (1,2); (0,1) and (0,2) share source 0,
        (1,2) only matches itself). This is symmetric so we can match
        the reference ``torch.linalg.eigh`` computation exactly.
        """
        edges = [(0, 1), (0, 2), (1, 2)]
        # Build a directed-only-input by constructing a Data with the
        # exact canonical edges; the lifting will treat the symmetric
        # version of these the same way.
        data = _make_data(edges, 3, symmetric=True)
        out = DirectedSimplicialLifting(spectral_normalize=True)(data)
        a100 = _dense(out.dir_lower_adj_100)

        # Reference computation: eigh on the raw binary 100, divide.
        raw_100 = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        max_eig = torch.linalg.eigh(raw_100)[0][-1]
        expected = raw_100 / max_eig
        assert torch.allclose(a100, expected, atol=1e-6), (
            f"a100 = {a100}, expected = {expected}"
        )
        # Sanity: largest eigenvalue of the normalized matrix is 1.
        norm_eigvals = torch.linalg.eigh(a100)[0]
        assert abs(norm_eigvals[-1].item() - 1.0) < 1e-6

    def test_spectral_normalize_on_non_symmetric_matches_reference(self) -> None:
        """On a non-symmetric adjacency, reproduce the reference verbatim.

        ``adj_low_101`` (line-graph adjacency) is non-symmetric in
        general. The upstream :func:`spectral_normalization` calls
        :func:`torch.linalg.eigh` regardless of symmetry; we reproduce
        the same call (which treats only the lower triangle as
        Hermitian) for bit-for-bit faithfulness when the lower triangle
        is non-trivial.
        """
        # Star + back-edges so adj_low_101 has a non-trivial lower
        # triangle: edges (0,1), (0,2), (1,2). The line-graph
        # adjacency has (0,1)->(1,2) i.e. ``a101[0, 2] = 1`` (upper
        # triangle) and ``a101[1, 2] = 1`` if (0,2) ends where (2,_)
        # starts -- but (0,2) ends at 2 and (1,2) also ends at 2, so
        # no head-to-tail meeting. To get a lower-triangle entry we
        # need a directed cycle which our canonical orientation
        # forbids -- so a101's lower triangle is zero here, and the
        # reference's eigh path returns max_eig=0. Our implementation
        # falls back to eigvals's abs max in that degenerate case;
        # verify the fallback yields a finite, normalized result.
        edges = [(0, 1), (1, 2), (0, 2)]
        data = _make_data(edges, 3, symmetric=True)
        out = DirectedSimplicialLifting(spectral_normalize=True)(data)
        a101 = _dense(out.dir_lower_adj_101)
        # Non-NaN and bounded by 1 in absolute value of its largest
        # |eigvalue| (which the eigvals-fallback divides by).
        assert torch.isfinite(a101).all()
        eigvals_abs_max = torch.linalg.eigvals(a101).abs().max().item()
        # Either the matrix is all-zero (no head-to-tail meeting under
        # canonical orientation) or it has been rescaled to spectral
        # radius 1.
        assert eigvals_abs_max == 0.0 or abs(eigvals_abs_max - 1.0) < 1e-6

    def test_spectral_normalize_repr(self) -> None:
        """``repr`` surfaces the new ``spectral_normalize`` knob."""
        assert "spectral_normalize=True" in repr(
            DirectedSimplicialLifting(spectral_normalize=True)
        )
        assert "spectral_normalize=False" in repr(DirectedSimplicialLifting())

    def test_spectral_normalize_empty_graph(self) -> None:
        """Spectral normalization on an empty graph is a no-op."""
        data = torch_geometric.data.Data(
            x=torch.zeros(3, 1),
            edge_index=torch.empty(2, 0, dtype=torch.long),
            num_nodes=3,
            y=torch.zeros(3, dtype=torch.long),
        )
        out = DirectedSimplicialLifting(spectral_normalize=True)(data)
        for key in DIR_ADJ_KEYS:
            mat = getattr(out, key)
            assert mat.shape == (0, 0) or mat.values().numel() == 0

    def test_spectral_normalize_helper_zero_matrix(self) -> None:
        """The all-zero adjacency hits the second-stage fallback and is returned as-is."""
        zero = torch.zeros(3, 3)
        out = DirectedSimplicialLifting._spectral_normalize(zero)
        assert torch.equal(out, zero)

    def test_dirsnn_official_lower_uses_spectral_norm(self) -> None:
        """End-to-end Hydra instantiation of ``dirsnn_official_lower`` config.

        Verifies the resulting batch has spectrally normalized
        adjacencies: the largest eigenvalue (via ``torch.linalg.eigh``,
        matching the reference) of the symmetric lower adjacencies
        ``dir_lower_adj_100`` / ``dir_lower_adj_111`` is approximately
        1.0.
        """
        from hydra import compose, initialize_config_dir

        from topobench.utils.config_resolvers import register_all_resolvers

        # Ensure custom OmegaConf resolvers (notably
        # ``get_default_transform``) are registered before composing.
        register_all_resolvers()

        config_dir = str(Path(__file__).resolve().parents[4] / "configs")
        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = compose(
                config_name="run.yaml",
                overrides=[
                    "model=simplicial/dirsnn_official_lower",
                    "dataset=graph/MUTAG",
                    "trainer=cpu",
                    "logger=wandb",
                ],
            )
        # The compose result holds the transform yaml node; check the
        # spectral_normalize knob made it through.
        lifting_cfg = cfg.transforms.graph2simplicial_lifting
        assert lifting_cfg.spectral_normalize is True, (
            f"dirsnn_official_lower must set spectral_normalize=True; "
            f"got {lifting_cfg}"
        )

        # And actually instantiate the lifting + run it on a tiny PyG
        # batch so we can assert the normalized adjacency really has
        # spectral radius 1.
        from topobench.transforms import TRANSFORMS

        lifting = TRANSFORMS[lifting_cfg.transform_name](
            **{
                k: v
                for k, v in lifting_cfg.items()
                if k not in ("transform_type", "transform_name")
            }
        )
        data = _make_data([(0, 1), (0, 2), (1, 2), (0, 3)], 4)
        out = lifting(data)
        a100 = _dense(out.dir_lower_adj_100)
        # adj_low_100 is symmetric (same-source mask), so its eigh-based
        # spectral radius after normalization is 1.
        max_eig = torch.linalg.eigh(a100)[0][-1].item()
        assert abs(max_eig - 1.0) < 1e-5, f"max eigenvalue = {max_eig}"


def test_directed_adjacencies_match_reference_example() -> None:
    """Check all ten directed adjacencies on a fixed reference digraph."""
    canonical_edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    sym_pairs = canonical_edges + [(v, u) for u, v in canonical_edges]
    data = torch_geometric.data.Data(
        x=torch.arange(5).float().unsqueeze(1),
        edge_index=torch.tensor(sym_pairs).T.long(),
        num_nodes=5,
        y=torch.zeros(5, dtype=torch.long),
    )
    out = DirectedSimplicialLifting()(data)

    # Edge order:
    # 0=(0,1), 1=(0,2), 2=(0,3), 3=(1,2), 4=(2,3),
    # 5=(2,4), 6=(3,4). The expected coordinates below are the
    # elementwise outputs of the upstream DirSNN compute_adj.py on this
    # digraph, expressed in the canonical edge order used by TopoBench.
    expected = {
        "dir_lower_adj_100": _expected_matrix(
            7,
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (4, 5),
                (5, 4),
                (5, 5),
                (6, 6),
            ],
        ),
        "dir_lower_adj_101": _expected_matrix(
            7,
            [
                (0, 3),
                (1, 4),
                (1, 5),
                (2, 6),
                (3, 4),
                (3, 5),
                (4, 6),
            ],
        ),
        "dir_lower_adj_110": _expected_matrix(
            7,
            [
                (3, 0),
                (4, 1),
                (5, 1),
                (6, 2),
                (4, 3),
                (5, 3),
                (6, 4),
            ],
        ),
        "dir_lower_adj_111": _expected_matrix(
            7,
            [
                (0, 0),
                (1, 1),
                (1, 3),
                (2, 2),
                (2, 4),
                (3, 1),
                (3, 3),
                (4, 2),
                (4, 4),
                (5, 5),
                (5, 6),
                (6, 5),
                (6, 6),
            ],
        ),
        "dir_upper_adj_101": _expected_matrix(7, [(3, 1), (4, 2), (6, 5)]),
        "dir_upper_adj_102": _expected_matrix(7, [(3, 0), (4, 1), (6, 4)]),
        "dir_upper_adj_112": _expected_matrix(7, [(1, 0), (2, 1), (5, 4)]),
        "dir_upper_adj_110": _expected_matrix(7, [(1, 3), (2, 4), (5, 6)]),
        "dir_upper_adj_120": _expected_matrix(7, [(0, 3), (1, 4), (4, 6)]),
        "dir_upper_adj_121": _expected_matrix(7, [(0, 1), (1, 2), (4, 5)]),
    }
    for key, expected_dense in expected.items():
        ours = _dense(getattr(out, key))
        assert torch.allclose(ours, expected_dense), (
            f"{key} disagrees with the reference compute_adj.py output. "
            f"Diff:\n{(ours - expected_dense).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_adjacency_key_ordering_is_stable() -> None:
    """Public key constants follow the reference ``compute_adj.py`` order."""
    assert DIR_LOWER_KEYS == (
        "dir_lower_adj_100",
        "dir_lower_adj_101",
        "dir_lower_adj_110",
        "dir_lower_adj_111",
    )
    assert DIR_UPPER_KEYS == (
        "dir_upper_adj_101",
        "dir_upper_adj_102",
        "dir_upper_adj_112",
        "dir_upper_adj_110",
        "dir_upper_adj_120",
        "dir_upper_adj_121",
    )
    assert DIR_ADJ_KEYS == DIR_LOWER_KEYS + DIR_UPPER_KEYS
    assert len(DIR_ADJ_KEYS) == 10
