"""Unit tests for the TopoPolynormer graph backbone.

Covers the topological substructure encoding (triangle counts, clustering,
random-walk return probabilities) and the full TopoPolynormer backbone,
including batch-aware behaviour (arXiv:2403.01232 + GSN arXiv:2006.09252).
"""

import networkx as nx
import pytest
import torch
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.graph.topo_polynormer import (
    DEFAULT_RW_STEPS,
    NUM_BASE_STRUCT_CHANNELS,
    TRIANGLE_SCALE,
    TopoPolynormer,
    _GlobalLinearAttention,
    structural_encoding,
)


def _rand_graph(num_nodes, seed, feat_dim=16):
    """Create a random undirected graph data object.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    seed : int
        Random seed.
    feat_dim : int, optional
        Node feature dimension.

    Returns
    -------
    torch_geometric.data.Data
        The random graph.
    """
    g = torch.Generator().manual_seed(seed)
    ei = torch.randint(0, num_nodes, (2, num_nodes * 3), generator=g)
    ei = ei[:, ei[0] != ei[1]]
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    x = torch.randn(num_nodes, feat_dim, generator=g)
    return Data(x=x, edge_index=ei, num_nodes=num_nodes)


class TestStructuralEncoding:
    """Tests for the topological substructure encoding."""

    def test_channel_count(self):
        """The encoding has the documented number of channels."""
        g = _rand_graph(15, 1)
        s = structural_encoding(g.edge_index, 15, None, DEFAULT_RW_STEPS)
        assert s.shape == (15, NUM_BASE_STRUCT_CHANNELS + len(DEFAULT_RW_STEPS))

    def test_triangle_channel_matches_networkx(self):
        """The triangle channel equals the per-node triangle count."""
        g = _rand_graph(25, 2)
        s = structural_encoding(g.edge_index, 25, None, DEFAULT_RW_STEPS)
        tri_model = s[:, 1] * TRIANGLE_SCALE
        G = nx.Graph()
        G.add_nodes_from(range(25))
        G.add_edges_from(g.edge_index.t().tolist())
        tri_nx = torch.tensor(
            [nx.triangles(G)[i] for i in range(25)], dtype=torch.float
        )
        assert torch.allclose(tri_model, tri_nx, atol=1e-4)

    def test_sum_equals_three_times_total_triangles(self):
        """Sum of per-node triangle counts equals 3x total triangles.

        This is the property that makes graph-level triangle counting a
        near-linear readout under sum pooling.
        """
        g = _rand_graph(30, 3)
        s = structural_encoding(g.edge_index, 30, None, DEFAULT_RW_STEPS)
        total_model = (s[:, 1] * TRIANGLE_SCALE).sum().item()
        G = nx.Graph()
        G.add_nodes_from(range(30))
        G.add_edges_from(g.edge_index.t().tolist())
        total_nx = sum(nx.triangles(G).values()) // 3
        assert abs(total_model - 3 * total_nx) < 1e-3

    def test_clustering_in_unit_interval(self):
        """The clustering-coefficient channel lies in [0, 1]."""
        g = _rand_graph(20, 4)
        s = structural_encoding(g.edge_index, 20, None, DEFAULT_RW_STEPS)
        c = s[:, 3]
        assert (c >= 0).all() and (c <= 1).all()

    def test_batch_isolation(self):
        """Per-graph features are unchanged by batching with another graph."""
        g_a = _rand_graph(18, 5)
        g_b = _rand_graph(11, 6)
        s_a = structural_encoding(g_a.edge_index, 18, None, DEFAULT_RW_STEPS)
        batch = Batch.from_data_list([g_a, g_b])
        s_batched = structural_encoding(
            batch.edge_index, batch.num_nodes, batch.batch, DEFAULT_RW_STEPS
        )
        assert torch.allclose(s_a, s_batched[:18], atol=1e-5)

    def test_no_rw_steps(self):
        """With empty rw_steps only the base channels are returned."""
        g = _rand_graph(12, 7)
        s = structural_encoding(g.edge_index, 12, None, rw_steps=())
        assert s.shape == (12, NUM_BASE_STRUCT_CHANNELS)

    def test_empty_graph(self):
        """Zero nodes returns an empty feature tensor of correct width."""
        ei = torch.empty((2, 0), dtype=torch.long)
        s = structural_encoding(ei, 0, None, DEFAULT_RW_STEPS)
        assert s.shape == (0, NUM_BASE_STRUCT_CHANNELS + len(DEFAULT_RW_STEPS))

    def test_graph_without_edges(self):
        """A graph with no edges yields all-zero structural features."""
        ei = torch.empty((2, 0), dtype=torch.long)
        s = structural_encoding(ei, 5, None, DEFAULT_RW_STEPS)
        assert torch.count_nonzero(s) == 0

    def test_missing_batch_id(self):
        """A gap in batch ids (a graph with no nodes) is skipped cleanly."""
        # nodes 0,1,2 -> graph 0 (a triangle) ; nodes 3,4 -> graph 2 (an edge) ;
        # graph id 1 has no nodes and must be skipped cleanly.
        batch = torch.tensor([0, 0, 0, 2, 2])
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 0, 2, 3, 4], [1, 0, 2, 1, 2, 0, 4, 3]],
            dtype=torch.long,
        )
        s = structural_encoding(edge_index, 5, batch, DEFAULT_RW_STEPS)
        assert s.shape == (5, NUM_BASE_STRUCT_CHANNELS + len(DEFAULT_RW_STEPS))
        # the triangle 0-1-2 gives each node one triangle; graph 2 (an edge) none
        assert s[1, 1] * TRIANGLE_SCALE == pytest.approx(1.0)
        assert s[3, 1] * TRIANGLE_SCALE == pytest.approx(0.0)


class TestGlobalLinearAttention:
    """Tests for the private global linear-attention module."""

    def test_forward_shapes(self):
        """Forward returns the working width and handles batch=None."""
        attn = _GlobalLinearAttention(8, 2, 2, -1.0, 0.0).eval()
        x = torch.randn(6, 16)
        assert attn(x).shape == (6, 16)
        assert attn(x, torch.zeros(6, dtype=torch.long)).shape == (6, 16)

    def test_separate_qk_and_const_beta(self):
        """Forward and reset work with separate q/k and constant beta."""
        attn = _GlobalLinearAttention(8, 2, 2, 0.4, 0.0, qk_shared=False)
        attn.reset_parameters()
        assert attn.q_lins is not None
        out = attn.eval()(torch.randn(5, 16), torch.zeros(5, dtype=torch.long))
        assert out.shape == (5, 16)

    def test_reset_learnable_beta(self):
        """Reset runs for the learnable-beta path."""
        attn = _GlobalLinearAttention(8, 2, 2, -1.0, 0.0)
        attn.reset_parameters()


class TestTopoPolynormer:
    """Tests for the full TopoPolynormer backbone."""

    def setup_method(self):
        """Set common dimensions."""
        self.in_channels = 16
        self.hidden = 8
        self.heads = 2
        self.out_channels = 16

    def _model(self, **kwargs):
        """Build a TopoPolynormer with test defaults.

        Parameters
        ----------
        **kwargs : dict
            Overrides forwarded to :class:`TopoPolynormer`.

        Returns
        -------
        TopoPolynormer
            The constructed backbone.
        """
        params = dict(
            in_channels=self.in_channels,
            hidden_channels=self.hidden,
            out_channels=self.out_channels,
            local_layers=2,
            global_layers=2,
            heads=self.heads,
        )
        params.update(kwargs)
        return TopoPolynormer(**params)

    def test_initialization_default(self):
        """Default construction wires the expected submodules."""
        model = self._model()
        assert model.use_global is True
        assert model.use_struct is True
        assert model.struct_in is not None
        assert model.struct_dim == NUM_BASE_STRUCT_CHANNELS + len(
            DEFAULT_RW_STEPS
        )
        assert model.global_attn is not None
        assert model.pre_lns is None

    def test_initialization_no_struct(self):
        """Disabling the encoding drops the structural projection."""
        model = self._model(use_struct=False)
        assert model.use_struct is False
        assert model.struct_in is None

    def test_initialization_local_only_and_pre_ln_and_beta(self):
        """Local-only, pre-ln and constant-beta construction."""
        model = self._model(global_layers=0, pre_ln=True, beta=0.5)
        assert model.use_global is False
        assert model.global_attn is None
        assert model.pre_lns is not None
        assert torch.allclose(model.betas, torch.full_like(model.betas, 0.5))

    def test_forward_shape(self, simple_graph_0):
        """Forward returns ``[num_nodes, out_channels]``.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model().eval()
        x = torch.randn(simple_graph_0.num_nodes, self.in_channels)
        out = model(
            x,
            simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(use_struct=False),
            dict(global_layers=0),
            dict(pre_ln=True),
            dict(beta=0.5),
            dict(qk_shared=False),
            dict(rw_steps=()),
        ],
    )
    def test_forward_variants(self, simple_graph_0, kwargs):
        """Forward works across configuration branches.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        kwargs : dict
            Construction overrides.
        """
        model = TopoPolynormer(
            self.in_channels,
            self.hidden,
            self.out_channels,
            local_layers=2,
            global_layers=kwargs.pop("global_layers", 1),
            heads=self.heads,
            **kwargs,
        ).eval()
        x = torch.randn(simple_graph_0.num_nodes, self.in_channels)
        out = model(x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_edge_weight_accepted_and_ignored(self, simple_graph_0):
        """An ``edge_weight`` argument does not change the output.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model().eval()
        x = torch.randn(simple_graph_0.num_nodes, self.in_channels)
        batch = torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        out_a = model(x, simple_graph_0.edge_index, batch=batch)
        ew = torch.rand(simple_graph_0.edge_index.shape[1])
        out_b = model(
            x, simple_graph_0.edge_index, batch=batch, edge_weight=ew, foo=1
        )
        assert torch.allclose(out_a, out_b)

    def test_batch_isolation(self):
        """The full model does not leak across graphs in a batch."""
        torch.manual_seed(0)
        model = self._model().eval()
        g_a = _rand_graph(9, 11, self.in_channels)
        g_b = _rand_graph(14, 12, self.in_channels)
        out_a = model(
            g_a.x, g_a.edge_index, batch=torch.zeros(9, dtype=torch.long)
        )
        batch = Batch.from_data_list([g_a, g_b])
        out_batched = model(batch.x, batch.edge_index, batch=batch.batch)
        assert torch.allclose(out_a, out_batched[:9], atol=1e-5)

    def test_eval_is_deterministic(self, simple_graph_0):
        """Repeated eval-mode calls match.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model(dropout=0.5, in_dropout=0.5).eval()
        x = torch.randn(simple_graph_0.num_nodes, self.in_channels)
        batch = torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        out1 = model(x, simple_graph_0.edge_index, batch=batch)
        out2 = model(x, simple_graph_0.edge_index, batch=batch)
        assert torch.allclose(out1, out2)

    def test_backward_pass(self, simple_graph_0):
        """Gradients flow to the structural projection and parameters.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model().train()
        x = torch.randn(simple_graph_0.num_nodes, self.in_channels)
        out = model(
            x,
            simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        out.sum().backward()
        assert model.struct_in.weight.grad is not None
        assert model.betas.grad is not None

    @pytest.mark.parametrize("kwargs", [dict(), dict(use_struct=False), dict(global_layers=0), dict(pre_ln=True), dict(beta=0.5)])
    def test_reset_parameters(self, kwargs):
        """``reset_parameters`` runs across configuration branches.

        Parameters
        ----------
        kwargs : dict
            Construction overrides.
        """
        model = self._model(**kwargs)
        model.reset_parameters()

    @pytest.mark.parametrize("heads", [1, 2, 4])
    def test_parametrized_heads(self, simple_graph_0, heads):
        """Forward works for several head counts.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        heads : int
            Number of attention heads.
        """
        model = TopoPolynormer(
            self.in_channels,
            self.hidden,
            self.out_channels,
            local_layers=2,
            global_layers=1,
            heads=heads,
        ).eval()
        x = torch.randn(simple_graph_0.num_nodes, self.in_channels)
        out = model(
            x,
            simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)
