"""Test CombinedEncodings Transform."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.all_encodings import (
    ALL_ENCODINGS,
    FE_ENCODINGS,
    PSE_ENCODINGS,
    CombinedEncodings,
    SelectDestinationEncodings,
)


class TestCombinedEncodings:
    """Test CombinedEncodings transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.edge_index = torch.tensor([[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]])
        self.x = torch.randn(3, 5)
        self.num_nodes = 3

    # ========== Encoding Sets Tests ==========

    def test_encoding_sets_defined(self):
        """Test that encoding sets are properly defined."""
        assert PSE_ENCODINGS == {"LapPE", "RWSE", "ElectrostaticPE", "HKdiagSE"}
        assert FE_ENCODINGS == {"HKFE", "KHopFE", "SheafConnLapPE", "PPRFE"}
        assert ALL_ENCODINGS == PSE_ENCODINGS | FE_ENCODINGS

    def test_encoding_sets_no_overlap(self):
        """Test that PSE and FE sets don't overlap."""
        assert PSE_ENCODINGS & FE_ENCODINGS == set()

    # ========== Initialization Tests ==========

    def test_initialization_with_fes_only(self):
        """Test initialization with only FE encodings."""
        transform = CombinedEncodings(encodings=["HKFE", "KHopFE"])
        assert transform.fe_encodings == ["HKFE", "KHopFE"]
        assert transform.pse_encodings == []

    def test_initialization_with_pses_only(self):
        """Test initialization with only PSE encodings."""
        transform = CombinedEncodings(encodings=["LapPE", "RWSE"])
        assert transform.fe_encodings == []
        assert transform.pse_encodings == ["LapPE", "RWSE"]

    def test_initialization_with_mixed(self):
        """Test initialization with mixed FE and PSE encodings."""
        transform = CombinedEncodings(encodings=["HKFE", "LapPE", "KHopFE", "RWSE"])
        assert transform.fe_encodings == ["HKFE", "KHopFE"]
        assert transform.pse_encodings == ["LapPE", "RWSE"]

    def test_initialization_preserves_order_within_groups(self):
        """Test that order is preserved within FE and PSE groups."""
        transform = CombinedEncodings(
            encodings=["RWSE", "HKFE", "LapPE", "KHopFE"]
        )
        assert transform.fe_encodings == ["HKFE", "KHopFE"]
        assert transform.pse_encodings == ["RWSE", "LapPE"]

    def test_initialization_invalid_encoding(self):
        """Test that invalid encoding raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported encoding"):
            CombinedEncodings(encodings=["InvalidEncoding"])

    def test_initialization_with_parameters(self):
        """Test initialization with parameters."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5)},
            "LapPE": {"max_pe_dim": 4},
        }
        transform = CombinedEncodings(
            encodings=["HKFE", "LapPE"],
            parameters=params
        )
        assert transform.parameters == params

    # ========== Forward Tests ==========

    def test_forward_fes_only(self):
        """Test forward with only FE encodings."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "KHopFE": {"max_hop": 3, "concat_to_x": False},
        }
        transform = CombinedEncodings(encodings=["HKFE", "KHopFE"], parameters=params)
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        result = transform(data)

        assert hasattr(result, "HKFE")
        assert hasattr(result, "KHopFE")
        assert result.HKFE.shape[0] == 3
        assert result.KHopFE.shape[0] == 3

    def test_forward_pses_only(self):
        """Test forward with only PSE encodings."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": False},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": False},
        }
        transform = CombinedEncodings(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        result = transform(data)

        assert hasattr(result, "LapPE")
        assert hasattr(result, "RWSE")

    def test_forward_mixed_concat(self):
        """Test forward with mixed encodings and concat_to_x=True."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
        }
        transform = CombinedEncodings(encodings=["HKFE", "LapPE"], parameters=params)
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        result = transform(data)

        # Original 5 + HKFE 4 + LapPE 4 = 13
        assert result.x.shape == (3, 13)

    def test_forward_mixed_separate_storage(self):
        """Test forward with mixed encodings and separate storage."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "LapPE": {"max_pe_dim": 4, "concat_to_x": False},
        }
        transform = CombinedEncodings(encodings=["HKFE", "LapPE"], parameters=params)
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        result = transform(data)

        assert hasattr(result, "HKFE")
        assert hasattr(result, "LapPE")
        assert result.HKFE.shape == (3, 4)
        assert result.LapPE.shape == (3, 4)

    def test_fes_applied_before_pses(self):
        """Test that FEs are applied before PSEs regardless of input order."""
        # This test verifies the ordering by checking that FEs use original features
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "LapPE": {"max_pe_dim": 4, "concat_to_x": False},
        }

        # PSE listed first, but FE should still run first
        transform = CombinedEncodings(encodings=["LapPE", "HKFE"], parameters=params)

        # Compute HKFE alone for comparison
        from topobench.transforms.data_manipulations import CombinedFEs
        fe_only = CombinedFEs(
            encodings=["HKFE"],
            parameters={"HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False}}
        )

        data1 = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)
        data2 = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        result_combined = transform(data1)
        result_fe_only = fe_only(data2)

        # HKFE should be identical in both cases (computed on original features)
        assert torch.allclose(result_combined.HKFE, result_fe_only.HKFE)

    def test_empty_encodings_list(self):
        """Test forward with empty encodings list."""
        transform = CombinedEncodings(encodings=[])
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        result = transform(data)

        assert torch.equal(result.x, self.x)

    def test_all_encodings(self):
        """Test with multiple encodings from both categories."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "KHopFE": {"max_hop": 3, "concat_to_x": False},
            "LapPE": {"max_pe_dim": 4, "concat_to_x": False},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": False},
        }
        transform = CombinedEncodings(
            encodings=["HKFE", "KHopFE", "LapPE", "RWSE"],
            parameters=params
        )
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        result = transform(data)

        assert hasattr(result, "HKFE")
        assert hasattr(result, "KHopFE")
        assert hasattr(result, "LapPE")
        assert hasattr(result, "RWSE")

    def test_numerical_stability(self):
        """Test that combined encodings don't produce NaN or Inf."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
        }
        transform = CombinedEncodings(encodings=["HKFE", "LapPE"], parameters=params)
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        result = transform(data)

        assert not torch.isnan(result.x).any()
        assert not torch.isinf(result.x).any()

    def test_with_sheaf_encoding(self):
        """Test with SheafConnLapPE encoding."""
        params = {
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": False},
            "LapPE": {"max_pe_dim": 4, "concat_to_x": False},
        }
        transform = CombinedEncodings(
            encodings=["SheafConnLapPE", "LapPE"],
            parameters=params
        )
        data = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)

        result = transform(data)

        assert hasattr(result, "SheafConnLapPE")
        assert hasattr(result, "LapPE")
        assert result.SheafConnLapPE.shape == (3, 6)


class TestSelectDestinationEncodings:
    """Test SelectDestinationEncodings transform."""

    def setup_method(self):
        """Set up test fixtures."""
        self.num_dst = 3
        self.num_src = 2
        self.num_total = self.num_dst + self.num_src

    def test_initialization(self):
        """Test initialization."""
        transform = SelectDestinationEncodings(encodings=["HKFE", "LapPE"])
        assert transform.encodings == ["HKFE", "LapPE"]

    def test_select_single_encoding(self):
        """Test selecting a single encoding."""
        transform = SelectDestinationEncodings(encodings=["HKFE"])

        # Create data with expanded graph (dst + src nodes)
        data = Data(
            x=torch.randn(self.num_total, 5),
            HKFE=torch.randn(self.num_total, 4),
        )

        result = transform(data, self.num_dst)

        assert result.x.shape == (self.num_dst, 5)
        assert result.HKFE.shape == (self.num_dst, 4)

    def test_select_multiple_encodings(self):
        """Test selecting multiple encodings."""
        transform = SelectDestinationEncodings(encodings=["HKFE", "LapPE", "RWSE"])

        data = Data(
            x=torch.randn(self.num_total, 5),
            HKFE=torch.randn(self.num_total, 4),
            LapPE=torch.randn(self.num_total, 6),
            RWSE=torch.randn(self.num_total, 8),
        )

        result = transform(data, self.num_dst)

        assert result.x.shape == (self.num_dst, 5)
        assert result.HKFE.shape == (self.num_dst, 4)
        assert result.LapPE.shape == (self.num_dst, 6)
        assert result.RWSE.shape == (self.num_dst, 8)

    def test_missing_encoding_raises(self):
        """Test that missing encoding raises ValueError."""
        transform = SelectDestinationEncodings(encodings=["HKFE", "MissingEnc"])

        data = Data(
            x=torch.randn(self.num_total, 5),
            HKFE=torch.randn(self.num_total, 4),
        )

        with pytest.raises(ValueError, match="MissingEnc.*not found"):
            transform(data, self.num_dst)

    def test_preserves_correct_rows(self):
        """Test that correct rows (first n_dst) are preserved."""
        transform = SelectDestinationEncodings(encodings=["HKFE"])

        # Create data with known values
        hkfe_data = torch.arange(self.num_total * 4).reshape(self.num_total, 4).float()
        data = Data(
            x=torch.randn(self.num_total, 5),
            HKFE=hkfe_data,
        )

        result = transform(data, self.num_dst)

        # Should have first 3 rows
        expected_hkfe = hkfe_data[:self.num_dst]
        assert torch.equal(result.HKFE, expected_hkfe)

    def test_x_none_handling(self):
        """Test handling when data.x is None."""
        transform = SelectDestinationEncodings(encodings=["HKFE"])

        data = Data(
            x=None,
            HKFE=torch.randn(self.num_total, 4),
        )

        result = transform(data, self.num_dst)

        assert result.x is None
        assert result.HKFE.shape == (self.num_dst, 4)

    def test_callable_interface(self):
        """Test that __call__ works correctly."""
        transform = SelectDestinationEncodings(encodings=["HKFE"])

        data = Data(
            x=torch.randn(self.num_total, 5),
            HKFE=torch.randn(self.num_total, 4),
        )

        # Should work via __call__
        result = transform(data, self.num_dst)

        assert result.HKFE.shape == (self.num_dst, 4)
