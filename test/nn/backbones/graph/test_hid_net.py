"""Unit tests for the HiD-Net graph backbone."""

import torch
from topobench.nn.backbones.graph import HiDNet


class TestHiDNet:
    """Test HiD-Net graph backbone."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.num_nodes = 5
        self.in_channels = 8
        self.hidden_channels = 16
        self.edge_index = torch.tensor(
            [
                [0, 1, 2, 3, 4, 0],
                [1, 2, 3, 4, 0, 2],
            ],
            dtype=torch.long,
        )

    def test_forward_shape(self):
        """Test that the backbone returns finite node embeddings."""
        x = torch.randn(self.num_nodes, self.in_channels)
        model = HiDNet(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=2,
        )

        out = model(x, self.edge_index)

        assert out.shape == (self.num_nodes, self.hidden_channels)
        assert torch.isfinite(out).all()

    def test_accepts_topobench_wrapper_kwargs(self):
        """Test compatibility with arguments supplied by ``GNNWrapper``."""
        x = torch.randn(4, self.in_channels)
        edge_index = torch.tensor(
            [[0, 1, 2], [1, 2, 3]], dtype=torch.long
        )
        batch = torch.zeros(4, dtype=torch.long)

        model = HiDNet(
            in_channels=self.in_channels,
            hidden_channels=self.in_channels,
            num_layers=2,
        )
        out = model(x, edge_index, batch=batch, edge_weight=None)

        assert out.shape == x.shape

    def test_supports_weighted_cached_propagation(self):
        """Test weighted diffusion and normalized adjacency caching."""
        x = torch.randn(4, self.in_channels)
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], dtype=torch.long
        )
        edge_weight = torch.tensor([1.0, 0.5, 1.0, 0.5, 0.75])

        model = HiDNet(
            in_channels=self.in_channels,
            hidden_channels=self.in_channels,
            num_layers=2,
            cached=True,
        )
        model.eval()

        out_first = model(x, edge_index, edge_weight=edge_weight)
        out_cached = model(x, edge_index, edge_weight=edge_weight)

        assert model._cached_edge_index is not None
        assert model._cached_grad_edge_index is not None
        assert torch.allclose(out_first, out_cached)

    def test_backward_produces_finite_gradients(self):
        """Test that gradients propagate through the backbone."""
        x = torch.randn(
            self.num_nodes,
            self.in_channels,
            requires_grad=True,
        )

        model = HiDNet(
            in_channels=self.in_channels,
            hidden_channels=self.in_channels,
            num_layers=2,
        )
        model(x, self.edge_index).square().mean().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert all(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
