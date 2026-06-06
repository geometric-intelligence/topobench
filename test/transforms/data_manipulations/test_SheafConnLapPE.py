"""Test SheafConnLapPE (Sheaf Connection Laplacian Positional Encoding) Transform."""

import warnings

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations import SheafConnLapPE


class TestSheafConnLapPE:
    """Test SheafConnLapPE transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # max_pe_dim=9 is divisible by stalk_dim=3 → k=3 eigenvectors
        self.transform = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=True)

    # ── Initialisation ────────────────────────────────────────────────────────

    def test_initialization_valid(self):
        """Test that valid parameter combinations are stored correctly."""
        t = SheafConnLapPE(
            max_pe_dim=12,
            stalk_dim=4,
            include_first=True,
            concat_to_x=False,
            eps=1e-5,
        )
        assert t.max_pe_dim == 12
        assert t.stalk_dim == 4
        assert t.k == 3          # 12 // 4
        assert t.include_first is True
        assert t.concat_to_x is False
        assert t.eps == 1e-5

    def test_initialization_invalid_divisibility(self):
        """Raise ValueError when max_pe_dim is not divisible by stalk_dim."""
        with pytest.raises(ValueError, match="divisible"):
            SheafConnLapPE(max_pe_dim=10, stalk_dim=3)

    # ── Feature-dim guard─────────────────────────────────────────────

    def test_feature_dim_less_than_stalk_dim_raises(self):
        """Raise ValueError when feature_dim < stalk_dim."""
        t = SheafConnLapPE(max_pe_dim=6, stalk_dim=3)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
        # Only 2 features but stalk_dim=3
        x = torch.randn(3, 2)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)
        with pytest.raises(ValueError, match="feature_dim"):
            t(data)

    def test_no_x_raises(self):
        """Raise ValueError when data.x is None."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        data = Data(edge_index=edge_index, num_nodes=2)
        with pytest.raises(ValueError, match="data.x"):
            t(data)

    # ── Forward — basic correctness ───────────────────────────────────────────

    def test_forward_concat_to_x(self):
        """PE is concatenated to data.x when concat_to_x=True."""
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = torch.randn(3, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = self.transform(data)

        assert out.x.shape == (3, 5 + self.transform.max_pe_dim)
        # Original features preserved
        assert torch.allclose(out.x[:, :5], x)

    def test_forward_no_concat(self):
        """PE is stored in data.SheafConnLapPE when concat_to_x=False."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = torch.randn(3, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)

        assert hasattr(out, "SheafConnLapPE")
        assert out.SheafConnLapPE.shape == (3, 9)
        # Original x unchanged
        assert torch.equal(out.x, x)

    def test_output_shape_always_max_pe_dim(self):
        """Output PE always has exactly max_pe_dim columns (zero-padded if needed)."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = torch.randn(3, 4)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)
        assert out.SheafConnLapPE.shape == (3, 9)

    def test_pe_not_all_zeros_on_connected_graph(self):
        """For a connected graph with varied features, PE should be non-trivial."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = torch.randn(3, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)
        assert not torch.allclose(out.SheafConnLapPE, torch.zeros_like(out.SheafConnLapPE))

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_graph(self):
        """Empty edge_index returns zero PE."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        x = torch.randn(4, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        out = t(data)

        assert out.SheafConnLapPE.shape == (4, 9)
        assert torch.allclose(out.SheafConnLapPE, torch.zeros(4, 9))

    def test_single_node_graph(self):
        """Single-node graph returns zero PE."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        x = torch.randn(1, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=1)

        out = t(data)

        assert out.SheafConnLapPE.shape == (1, 9)
        assert torch.allclose(out.SheafConnLapPE, torch.zeros(1, 9))

    def test_isolated_nodes(self):
        """Graph with isolated nodes runs without errors; shape is correct."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        # Nodes 0 and 1 connected; node 2 is isolated
        edge_index = torch.tensor([[0, 1], [1, 0]])
        x = torch.randn(3, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)

        assert out.SheafConnLapPE.shape == (3, 9)
        assert not torch.isnan(out.SheafConnLapPE).any()
        assert not torch.isinf(out.SheafConnLapPE).any()

    def test_directed_graph_symmetrised(self):
        """A one-directional edge_index is symmetrised; no NaN/Inf."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        # Only forward direction stored
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        x = torch.randn(3, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)

        assert out.SheafConnLapPE.shape == (3, 9)
        assert not torch.isnan(out.SheafConnLapPE).any()
        assert not torch.isinf(out.SheafConnLapPE).any()

    def test_disconnected_graph(self):
        """Disconnected graph: two components, no errors."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        x = torch.randn(4, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        out = t(data)

        assert out.SheafConnLapPE.shape == (4, 9)
        assert not torch.isnan(out.SheafConnLapPE).any()

    def test_max_pe_dim_larger_than_eigenvectors(self):
        """When fewer eigenvectors are available than k, output is zero-padded."""
        # 2-node graph: at most 2*3=6 eigenvectors, but we ask for max_pe_dim=9 (k=3)
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.tensor([[0, 1], [1, 0]])
        x = torch.randn(2, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=2)

        out = t(data)

        assert out.SheafConnLapPE.shape == (2, 9)
        # Some columns must be zero padding
        non_zero_cols = (out.SheafConnLapPE.abs().sum(dim=0) > 1e-6).sum()
        assert non_zero_cols < 9

    # ── Sign canonicalisation ─────────────────────────────────────────

    def test_sign_canonicalisation(self):
        """Verify that the max-abs entry of each eigenvector block is non-negative."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
        x = torch.randn(4, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        out = t(data)
        pe = out.SheafConnLapPE   # (n, max_pe_dim)

        # Each block of stalk_dim columns corresponds to one eigenvector reshaped.
        # The max-abs entry within each block-column should be positive.
        d = t.stalk_dim
        k = t.k
        for i in range(k):
            block = pe[:, i * d:(i + 1) * d]   # (n, d)
            if block.abs().max() > 1e-6:         # skip zero padding blocks
                flat = block.reshape(-1)
                max_idx = flat.abs().argmax()
                assert flat[max_idx] >= 0, (
                    f"Eigenvector block {i}: max-abs entry is negative"
                )

    # ── Numerical sanity ──────────────────────────────────────────────────────

    def test_no_nan_inf(self):
        """PE contains no NaN or Inf values."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = torch.randn(3, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)

        assert not torch.isnan(out.SheafConnLapPE).any()
        assert not torch.isinf(out.SheafConnLapPE).any()

    def test_duplicate_features_no_nan(self):
        """All-identical node features (zero variance) do not produce NaN/Inf."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = torch.ones(3, 5)  # identical features → zero local variance
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)

        assert not torch.isnan(out.SheafConnLapPE).any()
        assert not torch.isinf(out.SheafConnLapPE).any()

    # ── Memory warning ────────────────────────────────────────────────
    # Test might crash if run in an environment with limited memory, so it's commented out by default.
    # def test_memory_warning_emitted(self):
    #     """A ResourceWarning is emitted when num_nodes * stalk_dim > 10,000."""
    #     t = SheafConnLapPE(max_pe_dim=3, stalk_dim=3, concat_to_x=False)
    #     num_nodes = 3400
    #     src = list(range(num_nodes - 1))
    #     dst = list(range(1, num_nodes))
    #     edge_index = torch.tensor([src + dst, dst + src])
    #     x = torch.randn(num_nodes, 4)
    #     data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

    #     with warnings.catch_warnings(record=True) as w:
    #         warnings.simplefilter("always")
    #         t(data)

    #     resource_warnings = [
    #         warning for warning in w
    #         if issubclass(warning.category, ResourceWarning)
    #     ]
    #     assert len(resource_warnings) >= 1
    #     assert "SheafConnLapPE" in str(resource_warnings[0].message)

    # ── Device consistency ────────────────────────────────────────────────────

    def test_device_consistency_cpu(self):
        """PE tensor is on the same device as the input edge_index (CPU)."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = torch.randn(3, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)

        assert out.SheafConnLapPE.device == edge_index.device

    def test_device_consistency_cuda(self):
        """PE tensor is on the same CUDA device as the input."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]).cuda()
        x = torch.randn(3, 5).cuda()
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)

        assert out.SheafConnLapPE.is_cuda

    # ── Backward compatibility ────────────────────────────────────────────────

    def test_other_attributes_preserved(self):
        """All other data attributes are left unchanged after the transform."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=True)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = torch.randn(3, 5)
        y = torch.tensor([0, 1, 0])
        custom = torch.tensor([10, 20, 30])
        data = Data(x=x, edge_index=edge_index, y=y, custom=custom, num_nodes=3)

        out = t(data)

        assert torch.equal(out.y, y)
        assert torch.equal(out.custom, custom)
        assert torch.equal(out.edge_index, edge_index)

    # ── Parametrised tests ────────────────────────────────────────────────────

    @pytest.mark.parametrize("max_pe_dim,stalk_dim", [
        (3, 3),
        (6, 3),
        (9, 3),
        (8, 4),
        (6, 2),
    ])
    def test_parametrised_dimensions(self, max_pe_dim, stalk_dim):
        """Verify PE shape is always (num_nodes, max_pe_dim) for valid param pairs.

        Parameters
        ----------
        max_pe_dim : int
            Total PE output dimension.
        stalk_dim : int
            Stalk dimension for the connection Laplacian.
        """
        t = SheafConnLapPE(max_pe_dim=max_pe_dim, stalk_dim=stalk_dim, concat_to_x=False)
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = torch.randn(3, max(stalk_dim + 1, 5))
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)

        assert out.SheafConnLapPE.shape == (3, max_pe_dim)
        assert not torch.isnan(out.SheafConnLapPE).any()
        assert not torch.isinf(out.SheafConnLapPE).any()

    @pytest.mark.parametrize("include_first", [True, False])
    def test_parametrised_include_first(self, include_first):
        """Toggle include_first and verify output shape remains constant.

        Parameters
        ----------
        include_first : bool
            Whether to include trivial (near-zero eigenvalue) eigenvectors.
        """
        t = SheafConnLapPE(
            max_pe_dim=9,
            stalk_dim=3,
            include_first=include_first,
            concat_to_x=False,
        )
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        x = torch.randn(3, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)

        assert out.SheafConnLapPE.shape == (3, 9)
        assert not torch.isnan(out.SheafConnLapPE).any()

    # ── Larger graph ──────────────────────────────────────────────────────────

    def test_larger_graph(self):
        """Transform runs without error on a larger random graph."""
        t = SheafConnLapPE(max_pe_dim=9, stalk_dim=3, concat_to_x=False)
        num_nodes = 50
        src = torch.randint(0, num_nodes, (200,))
        dst = torch.randint(0, num_nodes, (200,))
        edge_index = torch.stack([src, dst])
        x = torch.randn(num_nodes, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

        out = t(data)

        assert out.SheafConnLapPE.shape == (num_nodes, 9)
        assert not torch.isnan(out.SheafConnLapPE).any()
        assert not torch.isinf(out.SheafConnLapPE).any()
