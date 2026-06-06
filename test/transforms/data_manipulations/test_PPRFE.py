"""Test PPRFE (Personalized Page Rank Feature Encoding) Transform."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.ppr_feature_encodings import PPRFE


class TestPPRFE:
    """Test PPRFE (Personalized Page Rank Feature Encoding) transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        self.x = torch.randn(3, 5)
        self.num_nodes = 3

    # ── Initialization ─────────────────────────────────────────────────────

    def test_initialization_defaults(self):
        """Test default initialization."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5))
        assert t.alpha_param_PPRFE == (0.1, 5)
        assert t.concat_to_x is True
        assert t.aggregation == "mean"
        assert t.self_loop is True
        assert t.fe_dim == 5

    def test_initialization_custom(self):
        """Test custom initialization."""
        t = PPRFE(
            alpha_param_PPRFE=(0.2, 10),
            concat_to_x=False,
            aggregation="sum",
            self_loop=False,
        )
        assert t.alpha_param_PPRFE == (0.2, 10)
        assert t.concat_to_x is False
        assert t.aggregation == "sum"
        assert t.self_loop is False
        assert t.fe_dim == 10

    def test_initialization_invalid_aggregation(self):
        """Test that invalid aggregation raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            PPRFE(alpha_param_PPRFE=(0.1, 5), aggregation="invalid")

    # ── Output shape ───────────────────────────────────────────────────────

    def test_output_shape_concat(self):
        """Test output shape with concat_to_x=True."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=True)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        # Original features (5) + PPRFE (5)
        assert out.x.shape == (3, 10)

    def test_output_shape_separate(self):
        """Test output shape with concat_to_x=False."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False)
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        # Original features unchanged
        assert out.x.shape == (3, 5)
        # PPRFE stored separately
        assert hasattr(out, "PPRFE")
        assert out.PPRFE.shape == (3, 5)

    def test_different_fe_dims(self):
        """Test different fe_dim values."""
        for num_alphas in [3, 5, 10, 20]:
            t = PPRFE(alpha_param_PPRFE=(0.1, num_alphas), concat_to_x=False)
            data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

            out = t(data)

            assert out.PPRFE.shape == (3, num_alphas)

    # ── Edge cases ─────────────────────────────────────────────────────────

    def test_empty_graph(self):
        """Test with empty edge_index."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        x = torch.randn(3, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        out = t(data)

        assert out.PPRFE.shape == (3, 5)
        # Empty graph should return zeros
        assert torch.allclose(out.PPRFE, torch.zeros(3, 5))

    def test_single_node(self):
        """Test with single node graph."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        x = torch.randn(1, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=1)

        out = t(data)

        assert out.PPRFE.shape == (1, 5)

    def test_no_features_raises(self):
        """Test that None features raise error."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5))
        data = Data(x=None, edge_index=self.edge_index, num_nodes=3)

        with pytest.raises(ValueError, match="requires node features"):
            t(data)

    # ── Numerical properties ───────────────────────────────────────────────

    def test_no_nan_values(self):
        """Test that output contains no NaN values."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 10), concat_to_x=False)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert not torch.isnan(out.PPRFE).any()

    def test_no_inf_values(self):
        """Test that output contains no Inf values."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 10), concat_to_x=False)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert not torch.isinf(out.PPRFE).any()

    def test_output_dtype_float32(self):
        """Test that output dtype is float32."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert out.PPRFE.dtype == torch.float32

    # ── Self-loop behavior ─────────────────────────────────────────────────

    def test_with_self_loop(self):
        """Test with self_loop=True."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False, self_loop=True)
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert out.PPRFE.shape == (3, 5)
        assert not torch.isnan(out.PPRFE).any()

    def test_without_self_loop(self):
        """Test with self_loop=False."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False, self_loop=False)
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert out.PPRFE.shape == (3, 5)
        assert not torch.isnan(out.PPRFE).any()

    def test_self_loop_affects_output(self):
        """Test that self_loop parameter affects the output."""
        t_with = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False, self_loop=True)
        t_without = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False, self_loop=False)

        data_with = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)
        data_without = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        out_with = t_with(data_with)
        out_without = t_without(data_without)

        # Outputs should be different
        assert not torch.allclose(out_with.PPRFE, out_without.PPRFE)

    # ── Aggregation functions ──────────────────────────────────────────────

    def test_aggregation_mean(self):
        """Test mean aggregation."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False, aggregation="mean")
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert out.PPRFE.shape == (3, 5)

    def test_aggregation_sum(self):
        """Test sum aggregation."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False, aggregation="sum")
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert out.PPRFE.shape == (3, 5)

    def test_aggregation_max(self):
        """Test max aggregation."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False, aggregation="max")
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert out.PPRFE.shape == (3, 5)

    def test_aggregation_min(self):
        """Test min aggregation."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False, aggregation="min")
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert out.PPRFE.shape == (3, 5)

    def test_aggregations_produce_different_results(self):
        """Test that different aggregations produce different results."""
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        results = {}
        for agg in ["mean", "sum", "max", "min"]:
            t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False, aggregation=agg)
            out = t(Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes))
            results[agg] = out.PPRFE.clone()

        # At least some pairs should be different
        assert not torch.allclose(results["mean"], results["sum"])
        assert not torch.allclose(results["max"], results["min"])

    # ── Larger graphs ──────────────────────────────────────────────────────

    def test_larger_graph(self):
        """Test with a larger graph."""
        num_nodes = 50
        # Create random edges
        src = torch.randint(0, num_nodes, (100,))
        dst = torch.randint(0, num_nodes, (100,))
        edge_index = torch.stack([src, dst])
        x = torch.randn(num_nodes, 10)

        t = PPRFE(alpha_param_PPRFE=(0.1, 8), concat_to_x=False)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

        out = t(data)

        assert out.PPRFE.shape == (num_nodes, 8)
        assert not torch.isnan(out.PPRFE).any()
        assert not torch.isinf(out.PPRFE).any()

    def test_complete_graph(self):
        """Test with a complete graph."""
        num_nodes = 5
        # Complete graph edges
        src = []
        dst = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst])
        x = torch.randn(num_nodes, 4)

        t = PPRFE(alpha_param_PPRFE=(0.1, 6), concat_to_x=False)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

        out = t(data)

        assert out.PPRFE.shape == (num_nodes, 6)
        assert not torch.isnan(out.PPRFE).any()

    # ── Device consistency ─────────────────────────────────────────────────

    def test_device_consistency_cpu(self):
        """Test that output is on the same CPU device as input."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert out.PPRFE.device == self.edge_index.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_consistency_cuda(self):
        """Test that output is on the same CUDA device as input."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=False)
        edge_index = self.edge_index.cuda()
        x = self.x.cuda()
        data = Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes)

        out = t(data)

        assert out.PPRFE.is_cuda

    # ── Backward compatibility / other attributes ──────────────────────────

    def test_other_attributes_preserved(self):
        """Test that other data attributes are preserved."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=True)
        y = torch.tensor([0, 1, 0])
        custom = torch.tensor([10, 20, 30])
        data = Data(
            x=self.x.clone(),
            edge_index=self.edge_index,
            y=y,
            custom=custom,
            num_nodes=self.num_nodes,
        )

        out = t(data)

        assert torch.equal(out.y, y)
        assert torch.equal(out.custom, custom)
        assert torch.equal(out.edge_index, self.edge_index)

    def test_original_features_preserved_at_start(self):
        """Test that original features are at the start when concatenating."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), concat_to_x=True)
        x_orig = self.x.clone()
        data = Data(x=x_orig.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        out = t(data)

        # First 5 columns should be original features
        assert torch.allclose(out.x[:, :5], x_orig)

    def test_debug_mode(self, capsys):
        """Test PPRFE with debug=True."""
        t = PPRFE(alpha_param_PPRFE=(0.1, 5), debug=True)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        # This should run without error and print debug info
        out = t(data)

        captured = capsys.readouterr()
        # Verify that some debug information was printed
        assert "PPRFE Debug Report" in captured.out
        assert out.x.shape == (3, 10)
