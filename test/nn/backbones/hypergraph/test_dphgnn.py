"""Unit tests for DPHGNN."""

import pytest
import torch

from topobench.nn.backbones.hypergraph import dphgnn
from topobench.nn.backbones.hypergraph.dphgnn import DPHGNN


def _toy_incidence():
    """Build a toy incidence matrix.

    n=6 hypernodes, m=4 hyperedges: e0={0,1,2}, e1={2,3}, e2={4}
    (singleton), e3={} (empty column), node 5 is isolated.
    """
    rows = [0, 1, 2, 2, 3, 4]
    cols = [0, 0, 0, 1, 1, 2]
    values = torch.ones(len(rows))
    indices = torch.tensor([rows, cols], dtype=torch.long)
    return torch.sparse_coo_tensor(indices, values, (6, 4)).coalesce()


def _toy_features(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(6, 8, generator=generator, dtype=torch.float64)


def _permute_incidence(incidence, perm):
    dense = incidence.to_dense()
    return dense[perm].to_sparse().coalesce()


@torch.no_grad()
def _hypergcn_adjacency_reference(incidence, x, num_nodes, with_mediators):
    """Unvectorized, per-hyperedge Python-loop reference implementation.

    This is the original ``_hypergcn_adjacency`` body, kept here (rather
    than in ``dphgnn.py``) purely as an oracle for
    ``test_hypergcn_adjacency_matches_reference``: it is exact by
    construction (no batching/padding tricks) but pays a CPU-GPU sync
    (``.item()``) per hyperedge, which is why ``dphgnn._hypergcn_adjacency``
    was vectorized.
    """
    indices = incidence.indices()
    row, col = indices[0], indices[1]
    order = torch.argsort(col, stable=True)
    row, col = row[order], col[order]
    _, counts = torch.unique_consecutive(col, return_counts=True)

    edge_row, edge_col, edge_val = [], [], []
    start = 0
    for count in counts.tolist():
        members = row[start : start + count]
        start += count
        if count < 2:
            continue
        member_x = x[members]
        dist = torch.cdist(member_x, member_x)
        flat_argmax = torch.argmax(dist).item()
        i_local, j_local = flat_argmax // count, flat_argmax % count
        i, j = members[i_local].item(), members[j_local].item()
        edge_row += [i, j]
        edge_col += [j, i]
        edge_val += [1.0, 1.0]
        if with_mediators and count > 2:
            weight = 1.0 / (2 * count - 3)
            for k_local in range(count):
                if k_local in (i_local, j_local):
                    continue
                k = members[k_local].item()
                edge_row += [i, k, j, k]
                edge_col += [k, i, k, j]
                edge_val += [weight, weight, weight, weight]

    if not edge_row:
        empty = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        return torch.sparse_coo_tensor(
            empty,
            torch.zeros(0, device=x.device, dtype=x.dtype),
            (num_nodes, num_nodes),
        ).coalesce()

    edge_indices = torch.tensor(
        [edge_row, edge_col], dtype=torch.long, device=x.device
    )
    values = torch.tensor(edge_val, device=x.device, dtype=x.dtype)
    return torch.sparse_coo_tensor(
        edge_indices, values, (num_nodes, num_nodes)
    ).coalesce()


def _ragged_incidence(cardinalities, seed):
    """Build a random incidence matrix with the given hyperedge sizes.

    A cardinality of 0 produces an unused hyperedge column (no rows), to
    exercise the "hyperedge absent from `col`" degenerate case alongside
    genuine singleton (cardinality 1) and larger hyperedges.
    """
    generator = torch.Generator().manual_seed(seed)
    num_nodes = sum(cardinalities)
    rows, cols = [], []
    node = 0
    for edge, card in enumerate(cardinalities):
        members = torch.randperm(num_nodes, generator=generator)[:card]
        for m in members.tolist():
            rows.append(m)
            cols.append(edge)
        node += card
    values = torch.ones(len(rows), dtype=torch.float64)
    indices = torch.tensor([rows, cols], dtype=torch.long)
    incidence = torch.sparse_coo_tensor(
        indices, values, (num_nodes, len(cardinalities))
    ).coalesce()
    x = torch.randn(num_nodes, 8, generator=generator, dtype=torch.float64)
    return incidence, x, num_nodes


class TestDPHGNN:
    """Unit tests for DPHGNN."""

    def setup_method(self):
        """Set up a toy hypergraph and a default model instance."""
        self.incidence = _toy_incidence()
        self.x = _toy_features()
        self.model = DPHGNN(hidden_channels=8).double()

    def test_output_shapes(self):
        """Output shapes must match the wrapper contract."""
        x_0, x_1 = self.model(self.x, self.incidence)
        assert x_0.shape == (6, 8)
        assert x_1.shape == (4, 8)

    def test_permutation_equivariance(self):
        """model(PX, PH) == P . model(X, H) on x_0 (Proposition 4.1)."""
        self.model.eval()
        perm = torch.tensor([3, 1, 4, 0, 5, 2])
        x_perm = self.x[perm]
        h_perm = _permute_incidence(self.incidence, perm)

        with torch.no_grad():
            x_0, _ = self.model(self.x, self.incidence)
            x_0_perm, _ = self.model(x_perm, h_perm)

        assert torch.allclose(x_0_perm, x_0[perm], atol=1e-5)

    def test_block_diagonal(self):
        """Two disjoint hypergraphs give the concatenation of outputs."""
        self.model.eval()
        incidence_2 = _toy_incidence()
        x_2 = _toy_features(seed=1)

        with torch.no_grad():
            x_0_a, x_1_a = self.model(self.x, self.incidence)
            x_0_b, x_1_b = self.model(x_2, incidence_2)

        n, m = 6, 4
        combined_x = torch.cat([self.x, x_2], dim=0)
        dense_a = self.incidence.to_dense()
        dense_b = incidence_2.to_dense()
        top = torch.cat(
            [dense_a, torch.zeros(n, m, dtype=torch.float64)], dim=1
        )
        bottom = torch.cat(
            [torch.zeros(n, m, dtype=torch.float64), dense_b], dim=1
        )
        combined_incidence = (
            torch.cat([top, bottom], dim=0).to_sparse().coalesce()
        )

        with torch.no_grad():
            x_0_combined, x_1_combined = self.model(
                combined_x, combined_incidence
            )

        assert torch.allclose(
            x_0_combined, torch.cat([x_0_a, x_0_b], dim=0), atol=1e-5
        )
        assert torch.allclose(
            x_1_combined, torch.cat([x_1_a, x_1_b], dim=0), atol=1e-5
        )

    def test_degenerate_cases_no_nan(self):
        """Isolated node, singleton and empty hyperedges must not yield nan."""
        x_0, x_1 = self.model(self.x, self.incidence)
        assert not torch.isnan(x_0).any()
        assert not torch.isinf(x_0).any()
        assert not torch.isnan(x_1).any()
        assert not torch.isinf(x_1).any()

    def test_gradient_flow(self):
        """Every parameter must receive a gradient after backward."""
        x_0, x_1 = self.model(self.x, self.incidence)
        loss = x_0.sum() + x_1.sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            assert param.grad is not None, f"no grad for {name}"

    def test_determinism_in_eval(self):
        """Two eval-mode forwards with no randomness give the same output."""
        self.model.eval()
        with torch.no_grad():
            x_0_first, _ = self.model(self.x, self.incidence)
            x_0_second, _ = self.model(self.x, self.incidence)
        assert torch.equal(x_0_first, x_0_second)

    def test_structures_by_hand(self):
        """Derived structures must match hand-computed values."""
        incidence = self.incidence
        d_v, d_e = dphgnn._compute_degrees(incidence)
        assert torch.equal(d_v, torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 0.0]))
        assert torch.equal(d_e, torch.tensor([3.0, 2.0, 1.0, 0.0]))

        a_c = dphgnn._clique_adjacency(incidence, 6).to_dense()
        expected_a_c = torch.zeros(6, 6)
        for i, j in [(0, 1), (0, 2), (1, 2), (2, 3)]:
            expected_a_c[i, j] = 1.0
            expected_a_c[j, i] = 1.0
        assert torch.equal(a_c, expected_a_c)

    def test_no_label_dependency(self):
        """forward only takes (x_0, incidence): no access to labels."""
        import inspect

        params = inspect.signature(DPHGNN.forward).parameters
        assert list(params.keys()) == ["self", "x_0", "incidence_hyperedges"]

    @pytest.mark.parametrize("supernode_init", ["zeros", "mean"])
    @pytest.mark.parametrize("with_mediators", [False, True])
    @pytest.mark.parametrize("taa_neighborhood", ["clique", "self"])
    def test_config_options(
        self, supernode_init, with_mediators, taa_neighborhood
    ):
        """Every config option must produce a valid forward pass."""
        model = DPHGNN(
            hidden_channels=8,
            supernode_init=supernode_init,
            with_mediators=with_mediators,
            taa_neighborhood=taa_neighborhood,
        ).double()
        x_0, x_1 = model(self.x, self.incidence)
        assert x_0.shape == (6, 8)
        assert x_1.shape == (4, 8)

    def test_invalid_supernode_init_raises(self):
        """An unknown supernode_init must raise ValueError."""
        with pytest.raises(ValueError):
            DPHGNN(hidden_channels=8, supernode_init="invalid")

    def test_invalid_taa_neighborhood_raises(self):
        """An unknown taa_neighborhood must raise ValueError at forward."""
        model = DPHGNN(hidden_channels=8, taa_neighborhood="invalid").double()
        with pytest.raises(ValueError):
            model(self.x, self.incidence)

    def test_heads_must_divide_hidden_channels(self):
        """A head count not dividing hidden_channels must raise."""
        with pytest.raises(ValueError):
            DPHGNN(hidden_channels=8, taa_heads=3)

    def test_repr(self):
        """__repr__ must return a non-empty string."""
        assert "DPHGNN" in repr(self.model)
        for name, module in self.model.named_children():
            assert module.__class__.__name__ in repr(module), name

    @pytest.mark.parametrize("with_mediators", [False, True])
    @pytest.mark.parametrize(
        "cardinalities",
        [
            [3, 2, 1, 0],
            [2, 2, 2],
            [5, 4, 3, 2, 1, 0],
            [6],
        ],
    )
    def test_hypergcn_adjacency_matches_reference(
        self, cardinalities, with_mediators
    ):
        """Vectorized `_hypergcn_adjacency` must exactly match the
        per-hyperedge Python-loop reference, including tie-breaking, for
        hyperedges of ragged cardinality (this is what the padding/masking
        in the vectorized version must reproduce bit-for-bit).
        """
        incidence, x, num_nodes = _ragged_incidence(cardinalities, seed=0)

        vectorized = dphgnn._hypergcn_adjacency(
            incidence, x, num_nodes, with_mediators
        ).to_dense()
        reference = _hypergcn_adjacency_reference(
            incidence, x, num_nodes, with_mediators
        ).to_dense()

        assert torch.equal(vectorized, reference)

    def test_hypergcn_adjacency_all_singletons(self):
        """No hyperedge with cardinality >= 2 yields an empty adjacency."""
        rows = [0, 1, 2]
        cols = [0, 1, 2]
        values = torch.ones(3, dtype=torch.float64)
        incidence = torch.sparse_coo_tensor(
            torch.tensor([rows, cols]), values, (3, 3)
        ).coalesce()
        x = torch.randn(3, 8, dtype=torch.float64)
        adjacency = dphgnn._hypergcn_adjacency(incidence, x, 3, False)
        assert adjacency._nnz() == 0
        assert adjacency.shape == (3, 3)
