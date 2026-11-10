"""Unit tests for the HODGNN backbone (HOD-GNN, arXiv:2510.02565).

The load-bearing test is the analytic-vs-autograd Jacobian cross-check: the
model's hand-propagated first-order derivatives (Algorithm 1 of the paper)
must match ``torch.autograd`` exactly. Further tests cover the Theorem-4.2
correspondence with random-walk return probabilities under the paper's
initialisation, batch isolation, permutation equivariance, gradient flow
through the derivative computation, and the expressivity motivation
(distinguishing C6 from 2xC3, which 1-WL message passing cannot).
"""

import math

import pytest
import torch
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.graph.hod_gnn import HODGNN

IN_DIM = 16


def _ring_graph(num_nodes, seed, feat_dim=IN_DIM, extra_edges=4):
    """Create a ring graph with random chords (no isolated nodes).

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed for chords and features.
    feat_dim : int, optional
        Node feature dimension.
    extra_edges : int, optional
        Number of random chord edges added on top of the ring.

    Returns
    -------
    torch_geometric.data.Data
        The undirected graph.
    """
    g = torch.Generator().manual_seed(seed)
    ring = torch.stack(
        [
            torch.arange(num_nodes),
            (torch.arange(num_nodes) + 1) % num_nodes,
        ]
    )
    chords = torch.randint(0, num_nodes, (2, extra_edges), generator=g)
    chords = chords[:, chords[0] != chords[1]]
    ei = torch.cat([ring, chords], dim=1)
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    ei = torch.unique(ei, dim=1)  # the sparse operator sums duplicates
    x = torch.randn(num_nodes, feat_dim, generator=g)
    return Data(x=x, edge_index=ei, num_nodes=num_nodes)


def _cycle_graph(cycle_sizes, feat_value=1.0, feat_dim=IN_DIM):
    """Create a disjoint union of cycles with constant features.

    Parameters
    ----------
    cycle_sizes : list of int
        Length of each cycle.
    feat_value : float, optional
        Constant node feature value.
    feat_dim : int, optional
        Node feature dimension.

    Returns
    -------
    torch_geometric.data.Data
        The undirected graph.
    """
    edges, offset = [], 0
    for size in cycle_sizes:
        idx = torch.arange(size)
        edges.append(torch.stack([idx, (idx + 1) % size]) + offset)
        offset += size
    ei = torch.cat(edges, dim=1)
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    x = torch.full((offset, feat_dim), feat_value)
    return Data(x=x, edge_index=ei, num_nodes=offset)


def _make_model(**kwargs):
    """Build a small HODGNN with test-friendly defaults.

    Parameters
    ----------
    **kwargs : dict
        Overrides for the HODGNN constructor.

    Returns
    -------
    HODGNN
        The instantiated model.
    """
    defaults = dict(
        in_channels=IN_DIM,
        hidden_channels=24,
        out_channels=20,
        base_channels=4,
        base_layers=3,
        downstream_layers=2,
        encoder_hidden=16,
        encoder_channels=8,
        structural_init=False,
    )
    defaults.update(kwargs)
    # Seed the global RNG so randomly initialised weights are reproducible
    # (an unlucky init can e.g. kill every ReLU in a layer and zero the
    # gradients that TestTraining asserts to be nonzero).
    torch.manual_seed(0)
    return HODGNN(**defaults)


def _base_internals(model, data_list):
    """Run the plumbing of ``forward`` up to the base propagation.

    Parameters
    ----------
    model : HODGNN
        The model (must be in double precision).
    data_list : list of torch_geometric.data.Data
        Graphs to batch together.

    Returns
    -------
    tuple
        ``(z, adj, idx_local, max_graph_size, batch_vector)`` as consumed
        by ``HODGNN._propagate_base``.
    """
    batch = Batch.from_data_list(data_list)
    x = batch.x.double()
    n = x.size(0)
    counts = torch.bincount(batch.batch)
    offsets = torch.cumsum(counts, dim=0) - counts
    idx_local = torch.arange(n) - offsets[batch.batch]
    adj = model._adjacency(batch.edge_index, n, None, x.dtype)
    z = model.lin_in(x)
    return z, adj, idx_local, int(counts.max().item()), batch.batch


class TestJacobianCorrectness:
    """Analytic derivative propagation must match torch.autograd exactly."""

    @pytest.mark.parametrize("activation", ["relu", "silu"])
    @pytest.mark.parametrize("aggregation", ["sum", "mean"])
    def test_diagonals_match_autograd(self, activation, aggregation):
        """Per-layer diagonal Jacobians equal autograd's, in a batch."""
        model = _make_model(
            activation=activation, aggregation=aggregation
        ).double()
        model.eval()
        z, adj, idx_local, s_max, _ = _base_internals(
            model, [_ring_graph(7, 1), _ring_graph(5, 2)]
        )
        z = z.detach().requires_grad_(True)
        k, t_layers = model.base_channels, model.base_layers

        feats, diags = model._propagate_base(z, adj, idx_local, s_max)
        jac_auto = torch.autograd.functional.jacobian(
            lambda inp: model._propagate_base(inp, adj, idx_local, s_max)[0],
            z,
        )  # [n, T*k, n, k]; rows carry the same 1/t! scale as `diags`.

        n = z.size(0)
        for t in range(t_layers):
            for v in range(n):
                ours = diags[v, t * k * k : (t + 1) * k * k].reshape(k, k)
                auto = jac_auto[v, t * k : (t + 1) * k, v, :]
                assert torch.allclose(ours, auto, atol=1e-10)

    def test_no_cross_graph_derivatives(self):
        """Autograd Jacobian is exactly zero across graph boundaries."""
        model = _make_model(activation="silu").double()
        model.eval()
        z, adj, idx_local, s_max, batch_vec = _base_internals(
            model, [_ring_graph(6, 3), _ring_graph(4, 4)]
        )
        jac_auto = torch.autograd.functional.jacobian(
            lambda inp: model._propagate_base(inp, adj, idx_local, s_max)[0],
            z.detach(),
        )
        cross = jac_auto[batch_vec == 0][:, :, batch_vec == 1, :]
        assert cross.abs().max().item() == 0.0


class TestStructuralInit:
    """Paper initialisation reproduces walk-based encodings (Theorem 4.2)."""

    def _diag_blocks(self, model, data):
        """Return the per-layer diagonal Jacobian blocks of ``data``.

        Parameters
        ----------
        model : HODGNN
            A double-precision model.
        data : torch_geometric.data.Data
            A single graph.

        Returns
        -------
        list of torch.Tensor
            Per-layer ``[n, k, k]`` diagonal blocks, rescaled by ``t!``.
        """
        z, adj, idx_local, s_max, _ = _base_internals(model, [data])
        _, diags = model._propagate_base(z, adj, idx_local, s_max)
        k = model.base_channels
        n = z.size(0)
        return [
            diags[:, t * k * k : (t + 1) * k * k].reshape(n, k, k)
            * math.factorial(t + 1)
            for t in range(model.base_layers)
        ]

    def _dense_adj(self, data):
        """Return the dense adjacency matrix of ``data``.

        Parameters
        ----------
        data : torch_geometric.data.Data
            A single graph.

        Returns
        -------
        torch.Tensor
            Dense ``[n, n]`` adjacency matrix.
        """
        n = data.num_nodes
        a = torch.zeros(n, n, dtype=torch.double)
        a[data.edge_index[1], data.edge_index[0]] = 1.0
        return a

    def test_mean_aggregation_gives_rw_return_probabilities(self):
        """Diagonals at init equal (P^t)_vv for P the random walk."""
        model = _make_model(
            structural_init=True, aggregation="mean", base_channels=3
        ).double()
        model.eval()
        data = _ring_graph(9, 5)
        a = self._dense_adj(data)
        p = a / a.sum(dim=1, keepdim=True)
        eye = torch.eye(3, dtype=torch.double)
        for t, block in enumerate(self._diag_blocks(model, data), start=1):
            returns = torch.diagonal(torch.linalg.matrix_power(p, t))
            expected = returns[:, None, None] * eye
            assert torch.allclose(block, expected, atol=1e-10)

    def test_sum_aggregation_gives_closed_walk_counts(self):
        """Diagonals at init equal (A^t)_vv (closed-walk centrality)."""
        model = _make_model(
            structural_init=True, aggregation="sum", base_channels=3
        ).double()
        model.eval()
        data = _ring_graph(8, 6)
        a = self._dense_adj(data)
        eye = torch.eye(3, dtype=torch.double)
        for t, block in enumerate(self._diag_blocks(model, data), start=1):
            walks = torch.diagonal(torch.linalg.matrix_power(a, t))
            expected = walks[:, None, None] * eye
            assert torch.allclose(block, expected, atol=1e-10)

    def test_distinguishes_c6_from_two_triangles(self):
        """Derivative features separate C6 from 2xC3 (1-WL cannot).

        Both graphs are 2-regular with identical constant features, so
        message passing sees identical multisets at every round. The
        3-step closed-walk derivative channel is nonzero exactly on the
        triangles — the paper's A^3 motivating example.
        """
        model = _make_model(
            structural_init=True, aggregation="sum", base_channels=3
        ).double()
        model.eval()
        c6 = _cycle_graph([6])
        c3c3 = _cycle_graph([3, 3])
        block_c6 = self._diag_blocks(model, c6)[2]
        block_c3 = self._diag_blocks(model, c3c3)[2]
        assert block_c6.abs().max().item() == 0.0  # no triangles in C6
        assert torch.allclose(
            torch.diagonal(block_c3, dim1=1, dim2=2),
            torch.full((6, 3), 2.0, dtype=torch.double),
        )  # every C3 node closes 2 directed 3-walks


class TestBatchIsolation:
    """A graph's output is identical alone and inside a batch."""

    @pytest.mark.parametrize("structural_init", [False, True])
    def test_batched_equals_individual(self, structural_init):
        """Forward on a batch matches per-graph forwards (diff == 0)."""
        model = _make_model(structural_init=structural_init).double()
        model.eval()
        g_a, g_b = _ring_graph(11, 7), _ring_graph(6, 8)
        batch = Batch.from_data_list([g_a, g_b])
        with torch.no_grad():
            out_batch = model(batch.x.double(), batch.edge_index, batch.batch)
            out_a = model(g_a.x.double(), g_a.edge_index)
            out_b = model(g_b.x.double(), g_b.edge_index)
        assert torch.allclose(out_batch[:11], out_a, atol=1e-12)
        assert torch.allclose(out_batch[11:], out_b, atol=1e-12)


class TestPermutationEquivariance:
    """Node relabelling permutes the output embeddings."""

    def test_permutation_equivariance(self):
        """f(P x, P edge_index) == P f(x, edge_index)."""
        model = _make_model().double()
        model.eval()
        data = _ring_graph(10, 9)
        perm = torch.randperm(10, generator=torch.Generator().manual_seed(0))
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(10)
        x_p = data.x.double()[perm]
        ei_p = inv[data.edge_index]
        with torch.no_grad():
            out = model(data.x.double(), data.edge_index)
            out_p = model(x_p, ei_p)
        assert torch.allclose(out[perm], out_p, atol=1e-10)


class TestTraining:
    """Gradient flow and stochastic-regularisation branches."""

    @pytest.mark.parametrize("structural_init", [False, True])
    def test_gradients_flow_through_derivative_computation(
        self, structural_init
    ):
        """Base weights receive nonzero, finite gradients via the loss."""
        model = _make_model(structural_init=structural_init)
        batch = Batch.from_data_list([_ring_graph(8, 10), _ring_graph(7, 11)])
        out = model(batch.x, batch.edge_index, batch.batch)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
        assert model.base_lin1[0].weight.grad.abs().sum() > 0
        assert model.base_eps.grad.abs().sum() > 0
        assert model.encoder[0].weight.grad.abs().sum() > 0

    def test_dropout_branches_run(self):
        """Base and downstream dropout paths execute in training mode."""
        model = _make_model(base_dropout=0.5, dropout=0.5)
        model.train()
        data = _ring_graph(9, 12)
        out = model(data.x, data.edge_index)
        assert out.shape == (9, 20)
        assert torch.isfinite(out).all()


class TestInterfaceAndBranches:
    """Constructor validation and remaining forward branches."""

    def test_invalid_activation_raises(self):
        """Unknown activation names are rejected."""
        with pytest.raises(ValueError, match="activation"):
            _make_model(activation="gelu")

    def test_invalid_aggregation_raises(self):
        """Unknown aggregation names are rejected."""
        with pytest.raises(ValueError, match="aggregation"):
            _make_model(aggregation="max")

    def test_edge_weight_is_used(self):
        """Doubling edge weights changes the output under sum aggregation."""
        model = _make_model()
        model.eval()
        data = _ring_graph(8, 13)
        ones = torch.ones(data.edge_index.size(1))
        with torch.no_grad():
            out_none = model(data.x, data.edge_index)
            out_ones = model(data.x, data.edge_index, None, ones)
            out_two = model(data.x, data.edge_index, None, 2 * ones)
        assert torch.allclose(out_none, out_ones, atol=1e-6)
        assert not torch.allclose(out_none, out_two, atol=1e-3)

    def test_include_input_false(self):
        """The raw-feature skip can be disabled."""
        model = _make_model(include_input=False)
        model.eval()
        data = _ring_graph(7, 14)
        with torch.no_grad():
            out = model(data.x, data.edge_index)
        assert out.shape == (7, 20)

    def test_edgeless_graph(self):
        """A graph with no edges still produces embeddings."""
        model = _make_model()
        model.eval()
        x = torch.randn(5, IN_DIM)
        ei = torch.empty(2, 0, dtype=torch.long)
        with torch.no_grad():
            out = model(x, ei)
        assert out.shape == (5, 20)
        assert torch.isfinite(out).all()

    def test_reset_parameters_applies_structural_init(self):
        """reset_parameters restores the Appendix F initialisation."""
        model = _make_model(structural_init=True)
        with torch.no_grad():
            model.lin_in.bias.fill_(3.0)
            model.base_eps.fill_(0.5)
        model.reset_parameters()
        assert torch.all(model.lin_in.bias == 1.0)
        assert torch.all(model.lin_in.weight == 0.0)
        assert torch.all(model.base_eps == -1.0)
        assert torch.allclose(
            model.base_lin1[0].weight,
            torch.eye(model.base_channels),
        )

    def test_reset_parameters_standard_init(self):
        """Without structural_init, eps resets to zero."""
        model = _make_model(structural_init=False)
        with torch.no_grad():
            model.base_eps.fill_(0.7)
        model.reset_parameters()
        assert torch.all(model.base_eps == 0.0)
