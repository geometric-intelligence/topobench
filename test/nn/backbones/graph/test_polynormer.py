"""Unit tests for the Polynormer graph backbone.

These tests cover the global linear-attention module
(:class:`PolynormerAttention`) and the full :class:`Polynormer` backbone,
including the batch-aware behaviour that keeps the global attention within the
boundaries of each graph in a mini-batch (arXiv:2403.01232).
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.graph.polynormer import (
    Polynormer,
    PolynormerAttention,
)


def _features(num_nodes, dim):
    """Create a random feature tensor.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    dim : int
        Feature dimension.

    Returns
    -------
    torch.Tensor
        Random features of shape ``[num_nodes, dim]``.
    """
    return torch.randn(num_nodes, dim)


class TestPolynormerAttention:
    """Tests for the batch-aware global linear-attention module."""

    def setup_method(self):
        """Set common dimensions used across tests."""
        self.hidden = 8
        self.heads = 2
        self.width = self.hidden * self.heads

    def test_initialization_shared_qk(self):
        """Default initialization shares the query/key projections."""
        attn = PolynormerAttention(self.hidden, self.heads, 2, -1.0, 0.0)
        assert attn.qk_shared is True
        assert attn.q_lins is None
        assert len(attn.k_lins) == 2
        assert len(attn.v_lins) == 2
        assert attn.betas.shape == (2, self.width)

    def test_initialization_separate_qk(self):
        """With ``qk_shared=False`` separate query projections are created."""
        attn = PolynormerAttention(
            self.hidden, self.heads, 3, -1.0, 0.0, qk_shared=False
        )
        assert attn.q_lins is not None
        assert len(attn.q_lins) == 3

    def test_initialization_constant_beta(self):
        """A non-negative ``beta`` initialises the gates to that constant."""
        attn = PolynormerAttention(self.hidden, self.heads, 2, 0.5, 0.0)
        assert torch.allclose(attn.betas, torch.full_like(attn.betas, 0.5))

    def test_forward_shape_single_graph(self):
        """Forward returns features of the working width."""
        attn = PolynormerAttention(self.hidden, self.heads, 2, -1.0, 0.0)
        attn.eval()
        x = _features(6, self.width)
        out = attn(x, torch.zeros(6, dtype=torch.long))
        assert out.shape == (6, self.width)
        assert not torch.isnan(out).any()

    def test_forward_batch_none_equals_single_segment(self):
        """``batch=None`` matches an all-zeros batch (single graph)."""
        attn = PolynormerAttention(self.hidden, self.heads, 2, -1.0, 0.0)
        attn.eval()
        x = _features(5, self.width)
        out_none = attn(x)
        out_zero = attn(x, torch.zeros(5, dtype=torch.long))
        assert torch.allclose(out_none, out_zero, atol=1e-6)

    def test_forward_separate_qk_and_constant_beta(self):
        """Forward works with separate q/k and a constant beta."""
        attn = PolynormerAttention(
            self.hidden, self.heads, 2, 0.3, 0.0, qk_shared=False
        )
        attn.eval()
        x = _features(7, self.width)
        out = attn(x, torch.zeros(7, dtype=torch.long))
        assert out.shape == (7, self.width)

    def test_reset_parameters_runs(self):
        """``reset_parameters`` runs for both shared and separate q/k."""
        for qk_shared in (True, False):
            for beta in (-1.0, 0.5):
                attn = PolynormerAttention(
                    self.hidden, self.heads, 2, beta, 0.0,
                    qk_shared=qk_shared,
                )
                attn.reset_parameters()
                if beta >= 0:
                    assert torch.allclose(
                        attn.betas, torch.full_like(attn.betas, beta)
                    )


class TestPolynormer:
    """Tests for the full Polynormer backbone."""

    def setup_method(self):
        """Set common dimensions used across tests."""
        self.in_channels = 16
        self.hidden = 8
        self.heads = 2
        self.out_channels = 16

    def _model(self, **kwargs):
        """Build a Polynormer with test defaults overridden by ``kwargs``.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to :class:`Polynormer`.

        Returns
        -------
        Polynormer
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
        return Polynormer(**params)

    def test_initialization_default(self):
        """Default construction wires the expected submodules."""
        model = self._model()
        assert model.use_global is True
        assert model.global_attn is not None
        assert len(model.local_convs) == 2
        assert model.pre_lns is None
        assert model.betas.shape == (2, self.hidden * self.heads)

    def test_initialization_local_only(self):
        """``global_layers=0`` disables the global module."""
        model = self._model(global_layers=0)
        assert model.use_global is False
        assert model.global_attn is None

    def test_initialization_pre_ln(self):
        """``pre_ln=True`` creates the per-layer input norms."""
        model = self._model(pre_ln=True)
        assert model.pre_lns is not None
        assert len(model.pre_lns) == 2

    def test_initialization_constant_beta(self):
        """A non-negative ``beta`` initialises the local gates to a constant."""
        model = self._model(beta=0.5)
        assert torch.allclose(model.betas, torch.full_like(model.betas, 0.5))

    def test_forward_shape(self, simple_graph_0):
        """Forward returns ``[num_nodes, out_channels]``.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model().eval()
        x = _features(simple_graph_0.num_nodes, self.in_channels)
        out = model(
            x,
            simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_local_only(self, simple_graph_0):
        """Local-only variant produces valid embeddings.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model(global_layers=0).eval()
        x = _features(simple_graph_0.num_nodes, self.in_channels)
        out = model(x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_forward_pre_ln_and_constant_beta(self, simple_graph_0):
        """Forward works with pre-layer-norm and a constant beta.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model(pre_ln=True, beta=0.4).eval()
        x = _features(simple_graph_0.num_nodes, self.in_channels)
        out = model(
            x,
            simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_forward_separate_qk(self, simple_graph_0):
        """Forward works when the global module uses separate q/k.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model(qk_shared=False).eval()
        x = _features(simple_graph_0.num_nodes, self.in_channels)
        out = model(x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_edge_weight_accepted_and_ignored(self, simple_graph_0):
        """An ``edge_weight`` argument is accepted but does not change output.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model().eval()
        x = _features(simple_graph_0.num_nodes, self.in_channels)
        batch = torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        out_a = model(x, simple_graph_0.edge_index, batch=batch)
        edge_weight = torch.rand(simple_graph_0.edge_index.shape[1])
        out_b = model(
            x, simple_graph_0.edge_index, batch=batch, edge_weight=edge_weight
        )
        assert torch.allclose(out_a, out_b)

    def test_extra_kwargs_ignored(self, simple_graph_0):
        """Unknown keyword arguments are ignored gracefully.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model().eval()
        x = _features(simple_graph_0.num_nodes, self.in_channels)
        out = model(
            x, simple_graph_0.edge_index, unused="x", another=1
        )
        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    def test_batch_isolation(self, simple_graph_0, simple_graph_1):
        """Global attention must not leak across graphs in a batch.

        Running the model on graph A alone must give the same node embeddings
        as running it on the batch ``[A, B]`` (in eval mode), proving the
        batch-aware global attention restricts attention to each graph.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            First test graph fixture.
        simple_graph_1 : torch_geometric.data.Data
            Second test graph fixture.
        """
        torch.manual_seed(0)
        model = self._model().eval()

        n_a = simple_graph_0.num_nodes
        n_b = simple_graph_1.num_nodes
        x_a = _features(n_a, self.in_channels)
        x_b = _features(n_b, self.in_channels)
        data_a = Data(x=x_a, edge_index=simple_graph_0.edge_index, num_nodes=n_a)
        data_b = Data(x=x_b, edge_index=simple_graph_1.edge_index, num_nodes=n_b)

        out_a_alone = model(
            data_a.x, data_a.edge_index, batch=torch.zeros(n_a, dtype=torch.long)
        )

        batch = Batch.from_data_list([data_a, data_b])
        out_batched = model(batch.x, batch.edge_index, batch=batch.batch)
        out_a_in_batch = out_batched[:n_a]

        assert torch.allclose(out_a_alone, out_a_in_batch, atol=1e-5)

    def test_eval_is_deterministic(self, simple_graph_0):
        """In eval mode (dropout off) repeated calls match.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model(dropout=0.5, in_dropout=0.5).eval()
        x = _features(simple_graph_0.num_nodes, self.in_channels)
        batch = torch.zeros(simple_graph_0.num_nodes, dtype=torch.long)
        out1 = model(x, simple_graph_0.edge_index, batch=batch)
        out2 = model(x, simple_graph_0.edge_index, batch=batch)
        assert torch.allclose(out1, out2)

    def test_backward_pass(self, simple_graph_0):
        """Gradients flow to the input and the parameters.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        model = self._model().train()
        x = _features(simple_graph_0.num_nodes, self.in_channels)
        x.requires_grad_(True)
        out = model(
            x,
            simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        out.sum().backward()
        assert x.grad is not None
        assert model.betas.grad is not None
        assert model.lin_in.weight.grad is not None

    def test_reset_parameters_runs(self, simple_graph_0):
        """``reset_parameters`` runs across configuration branches.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        """
        for kwargs in (
            dict(),
            dict(global_layers=0),
            dict(pre_ln=True),
            dict(beta=0.5),
        ):
            model = self._model(**kwargs)
            model.reset_parameters()

    def test_batched_two_graphs_shape(self, simple_graph_0, simple_graph_1):
        """Forward on a batch returns one embedding per node.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            First test graph fixture.
        simple_graph_1 : torch_geometric.data.Data
            Second test graph fixture.
        """
        model = self._model().eval()
        n = simple_graph_0.num_nodes + simple_graph_1.num_nodes
        x = _features(n, self.in_channels)
        batch_data = Batch.from_data_list([simple_graph_0, simple_graph_1])
        out = model(x, batch_data.edge_index, batch=batch_data.batch)
        assert out.shape == (n, self.out_channels)

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
        model = Polynormer(
            self.in_channels,
            self.hidden,
            self.out_channels,
            local_layers=2,
            global_layers=1,
            heads=heads,
        ).eval()
        x = _features(simple_graph_0.num_nodes, self.in_channels)
        out = model(
            x,
            simple_graph_0.edge_index,
            batch=torch.zeros(simple_graph_0.num_nodes, dtype=torch.long),
        )
        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)

    @pytest.mark.parametrize("local_layers", [1, 2, 4])
    def test_parametrized_local_layers(self, simple_graph_0, local_layers):
        """Forward works for several local-layer counts.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            Test graph fixture.
        local_layers : int
            Number of local attention layers.
        """
        model = Polynormer(
            self.in_channels,
            self.hidden,
            self.out_channels,
            local_layers=local_layers,
            global_layers=1,
            heads=self.heads,
        ).eval()
        x = _features(simple_graph_0.num_nodes, self.in_channels)
        out = model(x, simple_graph_0.edge_index)
        assert out.shape == (simple_graph_0.num_nodes, self.out_channels)
        assert len(model.local_convs) == local_layers
