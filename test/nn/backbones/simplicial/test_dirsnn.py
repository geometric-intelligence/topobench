"""Unit tests for DirSNN backbone."""

import torch
import pytest
from topobench.nn.backbones.simplicial.dirsnn import DirSNNLayer, DirSNN


class DummyBatch:
    """Mock TopoBench batch using correct framework field names."""
    def __init__(self, x_1, incidence_1, incidence_2):
        self.x_1 = x_1
        self.incidence_1 = incidence_1
        self.incidence_2 = incidence_2


def make_valid_sparse(rows, cols, n_rows, n_cols, values=None):
    """Helper to create a valid sparse COO tensor."""
    idx = torch.tensor([rows, cols], dtype=torch.long)
    val = torch.ones(len(rows)) if values is None else torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (n_rows, n_cols)).coalesce()


class TestDirSNNLayer:
    """Tests for DirSNNLayer."""

    def test_output_shape(self):
        """Test output shape is correct."""
        layer = DirSNNLayer(16, 32)
        x = torch.randn(5, 16)
        L_down = make_valid_sparse([0,1,2], [1,2,3], 5, 5)
        L_up = make_valid_sparse([0,1], [1,2], 5, 5)
        out = layer(x, L_down, L_up)
        assert out.shape == (5, 32)

    def test_no_nans(self):
        """Test output contains no NaNs."""
        layer = DirSNNLayer(8, 16)
        x = torch.randn(4, 8)
        L_down = make_valid_sparse([0,1], [1,2], 4, 4)
        L_up = make_valid_sparse([0,1], [1,0], 4, 4)
        out = layer(x, L_down, L_up)
        assert not torch.isnan(out).any()

    def test_source_sink_no_crash(self):
        """Test that isolated edges (all-zero rows in L_down/L_up) do not crash."""
        layer = DirSNNLayer(8, 16)
        x = torch.randn(4, 8)
        # Empty adjacency — all edges are isolated (worst-case source/sink)
        L_down = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0),
            (4, 4)
        ).coalesce()
        L_up = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0),
            (4, 4)
        ).coalesce()
        out = layer(x, L_down, L_up)
        assert out.shape == (4, 16)
        assert not torch.isnan(out).any()


class TestDirSNN:
    """Tests for the full DirSNN backbone."""

    def test_output_shape(self):
        """Test full backbone output shape."""
        backbone = DirSNN(16, 32, 8, num_layers=2)
        B1 = make_valid_sparse([0,1,1,2], [0,0,1,1], 4, 4,
                                values=[-1.,1.,-1.,1.])
        B2 = make_valid_sparse([0,1,2], [0,0,0], 4, 1)
        batch = DummyBatch(torch.randn(4, 16), B1, B2)
        out = backbone(batch)
        assert out.shape == (4, 8)

    def test_sparsification_branch(self):
        """Test _sparsify_boundary fires when max_upper_degree=0."""
        backbone = DirSNN(8, 16, 4, num_layers=2, max_upper_degree=0)
        backbone.train()
        B1 = make_valid_sparse([0,1,1,2], [0,0,1,1], 4, 4,
                                values=[-1.,1.,-1.,1.])
        B2 = make_valid_sparse([0,1,2], [0,0,0], 4, 1)
        batch = DummyBatch(torch.randn(4, 8), B1, B2)
        out = backbone(batch)
        assert out.shape == (4, 4)

    def test_single_layer(self):
        """Test num_layers=1 does not crash."""
        backbone = DirSNN(8, 16, 4, num_layers=1)
        B1 = make_valid_sparse([0,1,1,2], [0,0,1,1], 4, 4,
                                values=[-1.,1.,-1.,1.])
        B2 = make_valid_sparse([0,1,2], [0,0,0], 4, 1)
        batch = DummyBatch(torch.randn(4, 8), B1, B2)
        out = backbone(batch)
        assert out.shape == (4, 4)
class TestDepthAdaptiveGating:
    """Tests for the depth-adaptive gating mechanism in DirSNN."""

    def test_gates_initialized_to_one(self):
        """Test layer_gates exist and initialize to 1.0 (fully 'on')."""
        backbone = DirSNN(8, 16, 4, num_layers=2)
        assert len(backbone.layer_gates) == 2
        for gate in backbone.layer_gates:
            assert torch.isclose(gate, torch.tensor(1.0))

    def test_gates_are_trainable(self):
        """Test gates receive gradients during backprop.

        Uses num_layers=3, not 2, because gating is a documented no-op at
        num_layers<=2 (see test_gating_noop_at_two_layers) -- with only an
        input-changing first layer and an output-changing last layer, no
        layer's output ever matches its input shape, so the gate is never
        actually used in the forward computation graph and correctly
        receives no gradient. num_layers=3 has one interior hidden->hidden
        layer where gating is genuinely active.
        """
        backbone = DirSNN(8, 16, 4, num_layers=3)
        backbone.train()
        B1 = make_valid_sparse(
            [0, 1, 1, 2], [0, 0, 1, 1], 4, 4, values=[-1.0, 1.0, -1.0, 1.0]
        )
        B2 = make_valid_sparse([0, 1, 2], [0, 0, 0], 4, 1)
        batch = DummyBatch(torch.randn(4, 8, requires_grad=False), B1, B2)

        out = backbone(batch)
        loss = out.sum()
        loss.backward()

        # Only the interior gate (index 1, the hidden->hidden layer) is
        # guaranteed to receive a gradient; gates on shape-changing layers
        # (first and last) correctly receive none, same as at num_layers=2.
        interior_gate = backbone.layer_gates[1]
        assert interior_gate.grad is not None, (
            "Interior gate did not receive a gradient at num_layers=3"
        )

    def test_gating_noop_at_two_layers(self):
        """Test gating has no effect when num_layers=2 (both layers change
        dimensionality, so the shape-matching condition never fires).

        This confirms the documented no-op behavior: with only 1 or 2
        layers there is no interior layer whose contribution is ambiguous,
        so gates exist but never actually blend anything until
        num_layers >= 3.
        """
        torch.manual_seed(0)
        backbone_a = DirSNN(8, 16, 4, num_layers=2)
        torch.manual_seed(0)
        backbone_b = DirSNN(8, 16, 4, num_layers=2)

        # Force backbone_b's gates far from 1.0; output should be identical
        # to backbone_a (whose gates remain at 1.0) if gating is truly a
        # no-op at this depth.
        with torch.no_grad():
            for gate in backbone_b.layer_gates:
                gate.fill_(0.1)

        B1 = make_valid_sparse(
            [0, 1, 1, 2], [0, 0, 1, 1], 4, 4, values=[-1.0, 1.0, -1.0, 1.0]
        )
        B2 = make_valid_sparse([0, 1, 2], [0, 0, 0], 4, 1)
        x = torch.randn(4, 8)

        batch_a = DummyBatch(x.clone(), B1, B2)
        batch_b = DummyBatch(x.clone(), B1, B2)

        backbone_a.eval()
        backbone_b.eval()

        out_a = backbone_a(batch_a)
        out_b = backbone_b(batch_b)

        assert torch.allclose(out_a, out_b), (
            "Gating changed output at num_layers=2, but shape-matching "
            "should make gating a no-op at this depth"
        )

    def test_gating_active_at_three_layers(self):
        """Test gating actually blends output when an interior layer exists
        (num_layers=3 has one hidden->hidden layer eligible for gating).
        """
        torch.manual_seed(0)
        backbone_a = DirSNN(8, 16, 4, num_layers=3)
        torch.manual_seed(0)
        backbone_b = DirSNN(8, 16, 4, num_layers=3)

        # Push backbone_b's interior gate far from 1.0.
        with torch.no_grad():
            for gate in backbone_b.layer_gates:
                gate.fill_(0.1)

        B1 = make_valid_sparse(
            [0, 1, 1, 2], [0, 0, 1, 1], 4, 4, values=[-1.0, 1.0, -1.0, 1.0]
        )
        B2 = make_valid_sparse([0, 1, 2], [0, 0, 0], 4, 1)
        x = torch.randn(4, 8)

        batch_a = DummyBatch(x.clone(), B1, B2)
        batch_b = DummyBatch(x.clone(), B1, B2)

        backbone_a.eval()
        backbone_b.eval()

        out_a = backbone_a(batch_a)
        out_b = backbone_b(batch_b)

        assert not torch.allclose(out_a, out_b), (
            "Gating should change output at num_layers=3, where an "
            "interior hidden->hidden layer is eligible for gating"
        )
