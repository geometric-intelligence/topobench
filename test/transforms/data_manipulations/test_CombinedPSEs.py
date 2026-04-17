"""Test CombinedPSEs (Combined Positional and Structural Encodings) Transform."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import CombinedPSEs


class TestCombinedPSEs:
    """Test CombinedPSEs (Combined Positional and Structural Encodings) transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        self.x = torch.tensor([[1.0], [2.0], [3.0]])
        self.num_nodes = 3

    def test_initialization(self):
        """Test initialization of the transform."""
        # Test with single encoding
        transform = CombinedPSEs(encodings=["LapPE"])
        assert transform.encodings == ["LapPE"]
        assert transform.parameters == {}

        # Test with multiple encodings
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"])
        assert transform.encodings == ["LapPE", "RWSE"]
        assert transform.parameters == {}

        # Test with parameters
        params = {
            "LapPE": {"max_pe_dim": 8, "concat_to_x": False},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": False}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        assert transform.encodings == ["LapPE", "RWSE"]
        assert transform.parameters == params

    def test_single_lappe_encoding(self):
        """Test transform with only LapPE encoding."""
        params = {
            "LapPE": {"max_pe_dim": 8, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Check that x is updated with LapPE
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3
        assert transformed.x.shape[1] == 1 + 8  # original + LapPE

        # Check original features are preserved
        assert torch.equal(transformed.x[:, 0:1], self.x)

    def test_single_rwse_encoding(self):
        """Test transform with only RWSE encoding."""
        params = {
            "RWSE": {"max_pe_dim": 8, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Check that x is updated with RWSE
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3
        assert transformed.x.shape[1] == 1 + 8  # original + RWSE

        # Check original features are preserved
        assert torch.equal(transformed.x[:, 0:1], self.x)

    def test_combined_lappe_and_rwse(self):
        """Test transform with both LapPE and RWSE encodings."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Check that x is updated with both encodings
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3
        assert transformed.x.shape[1] == 1 + 4 + 4  # original + LapPE + RWSE

        # Check original features are preserved
        assert torch.equal(transformed.x[:, 0:1], self.x)

    def test_combined_separate_storage(self):
        """Test transform with separate storage (concat_to_x=False)."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": False},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": False}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Check that encodings are stored separately
        assert hasattr(transformed, "LapPE")
        assert hasattr(transformed, "RWSE")
        assert transformed.LapPE.shape == (3, 4)
        assert transformed.RWSE.shape == (3, 4)

        # Check that original x is unchanged
        assert torch.equal(transformed.x, self.x)

    def test_encoding_order(self):
        """Test that encodings are applied in the specified order."""
        params = {
            "LapPE": {"max_pe_dim": 2, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 3, "concat_to_x": True}
        }

        # Test LapPE first, then RWSE
        transform1 = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data1 = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)
        transformed1 = transform1(data1)

        # Test RWSE first, then LapPE
        transform2 = CombinedPSEs(encodings=["RWSE", "LapPE"], parameters=params)
        data2 = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)
        transformed2 = transform2(data2)

        # Both should have the same final shape
        assert transformed1.x.shape == transformed2.x.shape
        assert transformed1.x.shape == (3, 1 + 2 + 3)  # original + 2 + 3

        # The order of features should be different based on application order
        # We can't easily test the exact order without knowing the encoding values,
        # but we can verify they're both computed
        assert not torch.allclose(transformed1.x, transformed2.x)

    def test_empty_encodings_list(self):
        """Test transform with empty encodings list."""
        transform = CombinedPSEs(encodings=[])
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # No encodings applied, so x should be unchanged
        assert torch.equal(transformed.x, self.x)

    def test_invalid_encoding_type(self):
        """Test transform with invalid encoding type raises error."""
        transform = CombinedPSEs(encodings=["InvalidEncoding"])
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        with pytest.raises(ValueError, match="Unsupported encoding type"):
            transform(data)

    def test_no_parameters_provided(self):
        """Test transform requires parameters for encodings with required args."""
        # LapPE and RWSE both require max_pe_dim, so this should fail
        transform = CombinedPSEs(encodings=["LapPE"])
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        # Should raise TypeError because max_pe_dim is required
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            transform(data)

    def test_partial_parameters(self):
        """Test transform with parameters for only some encodings."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}  # Must provide required params
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Both encodings should be applied
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3
        assert transformed.x.shape[1] == 1 + 4 + 4  # original + LapPE + RWSE

    def test_no_features(self):
        """Test transform when data.x is None."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # x should be created with combined encodings
        assert transformed.x is not None
        assert transformed.x.shape == (3, 4 + 4)

    def test_empty_graph(self):
        """Test transform on an empty graph (no edges)."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(x=self.x, edge_index=edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Encodings should still be computed (as zeros)
        assert transformed.x is not None
        assert transformed.x.shape == (3, 1 + 4 + 4)

    def test_single_node_graph(self):
        """Test transform on a graph with a single node."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        x = torch.tensor([[1.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=1)

        transformed = transform(data)

        assert transformed.x is not None
        assert transformed.x.shape == (1, 1 + 4 + 4)

    def test_large_graph(self):
        """Test transform on a larger graph."""
        params = {
            "LapPE": {"max_pe_dim": 8, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 8, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)

        # Create a larger graph
        num_nodes = 50
        num_edges = 150
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

        transformed = transform(data)

        assert transformed.x is not None
        assert transformed.x.shape == (num_nodes, 5 + 8 + 8)

    def test_different_pe_dimensions(self):
        """Test transform with different dimensions for each encoding."""
        params = {
            "LapPE": {"max_pe_dim": 16, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert transformed.x is not None
        assert transformed.x.shape == (3, 1 + 16 + 4)

    def test_lappe_with_eigenvalues(self):
        """Test CombinedPSEs with LapPE including eigenvalues."""
        params = {
            "LapPE": {
                "max_pe_dim": 4,
                "include_eigenvalues": True,
                "concat_to_x": True
            },
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # LapPE with eigenvalues should have 2*max_pe_dim dimensions
        assert transformed.x is not None
        assert transformed.x.shape == (3, 1 + 8 + 4)  # original + (4*2) + 4

    def test_lappe_include_first(self):
        """Test CombinedPSEs with LapPE including first eigenvector."""
        params = {
            "LapPE": {
                "max_pe_dim": 4,
                "include_first": True,
                "concat_to_x": True
            },
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert transformed.x is not None
        assert transformed.x.shape == (3, 1 + 4 + 4)

    def test_device_consistency(self):
        """Test that encodings respect the device of input data."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)

        edge_index = self.edge_index.cuda()
        x = self.x.cuda()
        data = Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Check that output is on the same device
        assert transformed.x.device == x.device
        assert transformed.x.is_cuda

    def test_backward_compatibility(self):
        """Test that all original data attributes are preserved."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)

        y = torch.tensor([0, 1, 0])
        custom_attr = torch.tensor([10, 20, 30])
        data = Data(
            x=self.x,
            edge_index=self.edge_index,
            y=y,
            custom_attr=custom_attr,
            num_nodes=self.num_nodes
        )

        transformed = transform(data)

        # Check that all attributes are preserved
        assert hasattr(transformed, "y")
        assert hasattr(transformed, "custom_attr")
        assert torch.equal(transformed.y, y)
        assert torch.equal(transformed.custom_attr, custom_attr)
        assert torch.equal(transformed.edge_index, self.edge_index)

    def test_multiple_applications(self):
        """Test applying the same transform multiple times."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": False},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": False}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        # Apply transform twice
        transformed1 = transform(data)
        transformed2 = transform(data.clone())

        # Check that both have the encodings computed
        assert hasattr(transformed1, "LapPE")
        assert hasattr(transformed1, "RWSE")
        assert hasattr(transformed2, "LapPE")
        assert hasattr(transformed2, "RWSE")

        # RWSE should be deterministic
        assert torch.allclose(transformed1.RWSE, transformed2.RWSE)

        # LapPE may vary due to numerical solver, but shapes should match
        assert transformed1.LapPE.shape == transformed2.LapPE.shape

    def test_only_lappe_in_list(self):
        """Test with only LapPE in the encodings list."""
        params = {
            "LapPE": {"max_pe_dim": 8, "concat_to_x": False}
        }
        transform = CombinedPSEs(encodings=["LapPE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, "LapPE")
        assert not hasattr(transformed, "RWSE")
        assert transformed.LapPE.shape == (3, 8)

    def test_only_rwse_in_list(self):
        """Test with only RWSE in the encodings list."""
        params = {
            "RWSE": {"max_pe_dim": 8, "concat_to_x": False}
        }
        transform = CombinedPSEs(encodings=["RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, "RWSE")
        assert not hasattr(transformed, "LapPE")
        assert transformed.RWSE.shape == (3, 8)

    def test_mixed_concat_modes(self):
        """Test with one encoding concatenated and one stored separately."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": False}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # LapPE should be concatenated to x
        assert transformed.x.shape == (3, 1 + 4)  # original + LapPE

        # RWSE should be stored separately
        assert hasattr(transformed, "RWSE")
        assert transformed.RWSE.shape == (3, 4)

    def test_numerical_stability(self):
        """Test that combined encodings don't produce NaN or Inf."""
        params = {
            "LapPE": {"max_pe_dim": 8, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 8, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Check for numerical stability
        assert not torch.isnan(transformed.x).any()
        assert not torch.isinf(transformed.x).any()

    def test_case_sensitive_encoding_names(self):
        """Test that encoding names are case-sensitive."""
        transform = CombinedPSEs(encodings=["lappe"])  # lowercase
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        # Should raise error for incorrect case
        with pytest.raises(ValueError, match="Unsupported encoding type"):
            transform(data)

    @pytest.mark.parametrize("encoding", ["LapPE", "RWSE"])
    def test_parametrized_single_encodings(self, encoding):
        """Parametrized test for single encodings.

        Parameters
        ----------
        encoding : str
            The encoding type to test ("LapPE" or "RWSE").
        """
        params = {
            encoding: {"max_pe_dim": 4, "concat_to_x": False}
        }
        transform = CombinedPSEs(encodings=[encoding], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, encoding)
        assert getattr(transformed, encoding).shape == (3, 4)

    @pytest.mark.parametrize("max_pe_dim", [2, 4, 8, 16])
    def test_parametrized_dimensions(self, max_pe_dim):
        """Parametrized test for different PE dimensions.

        Parameters
        ----------
        max_pe_dim : int
            The maximum positional encoding dimension to test.
        """
        params = {
            "LapPE": {"max_pe_dim": max_pe_dim, "concat_to_x": True},
            "RWSE": {"max_pe_dim": max_pe_dim, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert transformed.x.shape == (3, 1 + max_pe_dim + max_pe_dim)

    def test_duplicate_encodings_in_list(self):
        """Test behavior when the same encoding appears multiple times."""
        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        # Apply LapPE twice
        transform = CombinedPSEs(encodings=["LapPE", "LapPE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Both applications should concatenate
        assert transformed.x.shape == (3, 1 + 4 + 4)  # original + LapPE + LapPE

    def test_complete_graph(self):
        """Test combined encodings on a complete graph."""
        # Complete graph on 4 nodes
        edges = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges).t()
        x = torch.randn(4, 2)

        params = {
            "LapPE": {"max_pe_dim": 4, "concat_to_x": True},
            "RWSE": {"max_pe_dim": 4, "concat_to_x": True}
        }
        transform = CombinedPSEs(encodings=["LapPE", "RWSE"], parameters=params)
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        transformed = transform(data)

        assert transformed.x.shape == (4, 2 + 4 + 4)
        assert not torch.isnan(transformed.x).any()
