"""Test CombinedFEs (Combined Feature Encodings) Transform."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import CombinedFEs


class TestCombinedFEs:
    """Test CombinedFEs (Combined Feature Encodings) transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        self.x = torch.tensor([[1.0], [2.0], [3.0]])
        self.num_nodes = 3

    def test_initialization(self):
        """Test initialization of the transform."""
        # Test with single encoding
        transform = CombinedFEs(encodings=["HKFE"])
        assert transform.encodings == ["HKFE"]
        assert transform.parameters == {}

        # Test with multiple encodings
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"])
        assert transform.encodings == ["HKFE", "KHopFE"]
        assert transform.parameters == {}

        # Test with all supported encodings
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"])
        assert transform.encodings == ["HKFE", "KHopFE"]
        assert transform.parameters == {}

        # Test with parameters
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "KHopFE": {"max_hop": 4, "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        assert transform.encodings == ["HKFE", "KHopFE"]
        assert transform.parameters == params

    def test_single_hkfe_encoding(self):
        """Test transform with only HKFE encoding."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # HKFE with kernel_param (1, 5) produces fe_dim=4 fixed dimensions
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3
        assert transformed.x.shape[1] == 1 + 4  # original + HKFE (fe_dim)

        # Check original features are preserved
        assert torch.equal(transformed.x[:, 0:1], self.x)

    def test_single_khopfe_encoding(self):
        """Test transform with only KHopFE encoding."""
        params = {
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # KHopFE with max_hop=4 produces (4-1)=3 fixed dimensions
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3
        assert transformed.x.shape[1] == 1 + 3  # original + KHopFE (max_hop-1)

        # Check original features are preserved
        assert torch.equal(transformed.x[:, 0:1], self.x)

    def test_combined_hkfe_and_khopfe(self):
        """Test transform with both HKFE and KHopFE encodings."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Check that x is updated with both encodings (fixed dimensions)
        # HKFE: fe_dim=4, KHopFE: max_hop-1=3
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3
        # After HKFE: 1 + 4 = 5 features
        # After KHopFE: 5 + 3 = 8 features (fixed dimension output)
        assert transformed.x.shape[1] == 1 + 4 + 3

        # Check original features are preserved
        assert torch.equal(transformed.x[:, 0:1], self.x)

    def test_combined_separate_storage(self):
        """Test transform with separate storage (concat_to_x=False)."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "KHopFE": {"max_hop": 4, "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Check that encodings are stored separately (fixed dimensions)
        assert hasattr(transformed, "HKFE")
        assert hasattr(transformed, "KHopFE")
        assert transformed.HKFE.shape == (3, 4)  # fe_dim (fixed)
        assert transformed.KHopFE.shape == (3, 3)  # max_hop-1 (fixed)

        # Check that original x is unchanged
        assert torch.equal(transformed.x, self.x)

    def test_encoding_order(self):
        """Test that encodings are concatenated in the specified order."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 3), "concat_to_x": True},
            "KHopFE": {"max_hop": 3, "concat_to_x": True}
        }

        # Test HKFE first, then KHopFE
        transform1 = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        data1 = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)
        transformed1 = transform1(data1)

        # Test KHopFE first, then HKFE
        transform2 = CombinedFEs(encodings=["KHopFE", "HKFE"], parameters=params)
        data2 = Data(x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes)
        transformed2 = transform2(data2)

        # Shapes should be the same
        assert transformed1.x.shape == transformed2.x.shape

        # Original features should be the same (first column)
        assert torch.allclose(transformed1.x[:, :1], transformed2.x[:, :1])

        # The encoding values differ in position due to concatenation order:
        # Order 1: [x, HKFE (2 cols), KHopFE (2 cols)]
        # Order 2: [x, KHopFE (2 cols), HKFE (2 cols)]
        # So the full tensors should not be equal
        assert not torch.allclose(transformed1.x, transformed2.x)

        # But the HKFE columns in order1 should match HKFE columns in order2
        # (both computed on original features)
        hkfe_dim = 2  # kernel_param (1, 3) -> 2 dimensions
        khop_dim = 2  # max_hop 3 -> 2 dimensions
        hkfe_order1 = transformed1.x[:, 1:1+hkfe_dim]
        hkfe_order2 = transformed2.x[:, 1+khop_dim:1+khop_dim+hkfe_dim]
        assert torch.allclose(hkfe_order1, hkfe_order2)

    def test_empty_encodings_list(self):
        """Test transform with empty encodings list."""
        transform = CombinedFEs(encodings=[])
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # No encodings applied, so x should be unchanged
        assert torch.equal(transformed.x, self.x)

    def test_invalid_encoding_type(self):
        """Test transform with invalid encoding type raises error."""
        transform = CombinedFEs(encodings=["InvalidEncoding"])
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        with pytest.raises(ValueError, match="Unsupported encoding type"):
            transform(data)

    def test_no_parameters_provided(self):
        """Test transform requires parameters for encodings with required args."""
        # HKFE requires kernel_param_HKFE, so this should fail
        transform = CombinedFEs(encodings=["HKFE"])
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        # Should raise TypeError because kernel_param_HKFE is required
        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            transform(data)

    def test_partial_parameters(self):
        """Test transform with parameters for only some encodings."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Both encodings should be applied
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3

    def test_no_features_raises_error(self):
        """Test transform when data.x is None raises error."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE"], parameters=params)
        data = Data(edge_index=self.edge_index, num_nodes=self.num_nodes)

        # Feature encodings require data.x
        with pytest.raises(ValueError, match="requires node features"):
            transform(data)

    def test_empty_graph(self):
        """Test transform on an empty graph (no edges)."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        data = Data(x=self.x, edge_index=edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Encodings should still be computed (as zeros for empty graph)
        assert transformed.x is not None
        assert transformed.x.shape[0] == 3

    def test_single_node_graph(self):
        """Test transform on a graph with a single node."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        x = torch.tensor([[1.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=1)

        transformed = transform(data)

        assert transformed.x is not None
        assert transformed.x.shape[0] == 1

    def test_large_graph(self):
        """Test transform on a larger graph."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)

        # Create a larger graph
        num_nodes = 50
        num_edges = 150
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, 5)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

        transformed = transform(data)

        assert transformed.x is not None
        assert transformed.x.shape[0] == num_nodes

    def test_different_fe_dimensions(self):
        """Test transform with different dimensions for each encoding."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 9), "concat_to_x": True},
            "KHopFE": {"max_hop": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert transformed.x is not None
        # HKFE: fe_dim=8, KHopFE: max_hop-1=2 (both fixed dimensions)
        assert transformed.x.shape == (3, 1 + 8 + 2)

    def test_hkfe_different_kernel_params(self):
        """Test HKFE with different kernel parameter ranges."""
        # kernel_param (1, 9) => fe_dim=8 dimensions
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 9), "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # 1 original + 8 HKFE (fixed dimension)
        assert transformed.x.shape == (3, 1 + 8)
        assert not torch.isnan(transformed.x).any()

    def test_khopfe_different_max_hop(self):
        """Test KHopFE with different max_hop values."""
        params = {
            "KHopFE": {"max_hop": 6, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # 1 original + 5 KHopFE (max_hop-1, fixed dimension)
        assert transformed.x.shape == (3, 1 + 5)
        assert not torch.isnan(transformed.x).any()

    def test_device_consistency(self):
        """Test that encodings respect the device of input data."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)

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
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)

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
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "KHopFE": {"max_hop": 4, "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        # Apply transform twice
        transformed1 = transform(data)
        transformed2 = transform(data.clone())

        # Check that both have the encodings computed
        assert hasattr(transformed1, "HKFE")
        assert hasattr(transformed1, "KHopFE")
        assert hasattr(transformed2, "HKFE")
        assert hasattr(transformed2, "KHopFE")

        # Encodings should be deterministic
        assert torch.allclose(transformed1.HKFE, transformed2.HKFE)
        assert torch.allclose(transformed1.KHopFE, transformed2.KHopFE)

    def test_only_hkfe_in_list(self):
        """Test with only HKFE in the encodings list."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["HKFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, "HKFE")
        assert not hasattr(transformed, "KHopFE")
        assert transformed.HKFE.shape == (3, 4)  # fe_dim (fixed dimension)

    def test_only_khopfe_in_list(self):
        """Test with only KHopFE in the encodings list."""
        params = {
            "KHopFE": {"max_hop": 4, "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, "KHopFE")
        assert not hasattr(transformed, "HKFE")
        assert transformed.KHopFE.shape == (3, 3)  # max_hop-1 (fixed dimension)

    def test_mixed_concat_modes(self):
        """Test with one encoding concatenated and one stored separately."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # HKFE should be concatenated to x
        assert transformed.x.shape == (3, 1 + 4)  # original + HKFE

        # KHopFE should be stored separately (fixed dimension output)
        assert hasattr(transformed, "KHopFE")
        assert transformed.KHopFE.shape == (3, 3)  # max_hop-1 (fixed dimension)

    def test_numerical_stability(self):
        """Test that combined encodings don't produce NaN or Inf."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Check for numerical stability
        assert not torch.isnan(transformed.x).any()
        assert not torch.isinf(transformed.x).any()

    def test_case_sensitive_encoding_names(self):
        """Test that encoding names are case-sensitive."""
        transform = CombinedFEs(encodings=["hkfe"])  # lowercase
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        # Should raise error for incorrect case
        with pytest.raises(ValueError, match="Unsupported encoding type"):
            transform(data)

    @pytest.mark.parametrize("encoding,params,expected_dim", [
        ("HKFE", {"kernel_param_HKFE": (1, 5), "concat_to_x": False}, 4),
        ("KHopFE", {"max_hop": 4, "concat_to_x": False}, 3),
    ])
    def test_parametrized_single_encodings(self, encoding, params, expected_dim):
        """Parametrized test for single encodings.

        Parameters
        ----------
        encoding : str
            The encoding type to test.
        params : dict
            Parameters for the encoding.
        expected_dim : int
            Expected fixed output dimension.
        """
        transform = CombinedFEs(encodings=[encoding], parameters={encoding: params})
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        attr_name = "HKFE" if encoding == "HKFE" else "KHopFE"
        assert hasattr(transformed, attr_name)
        assert getattr(transformed, attr_name).shape == (3, expected_dim)

    @pytest.mark.parametrize("kernel_param", [(1, 3), (1, 5), (1, 9), (2, 10)])
    def test_parametrized_hkfe_dimensions(self, kernel_param):
        """Parametrized test for different HKFE kernel parameters.

        Parameters
        ----------
        kernel_param : tuple
            The kernel parameter tuple (start, end) to test.
        """
        params = {
            "HKFE": {"kernel_param_HKFE": kernel_param, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        fe_dim = kernel_param[1] - kernel_param[0]
        assert transformed.x.shape == (3, 1 + fe_dim)  # fixed dimension output

    @pytest.mark.parametrize("max_hop", [2, 3, 4, 5])
    def test_parametrized_khopfe_dimensions(self, max_hop):
        """Parametrized test for different KHopFE max_hop values.

        Parameters
        ----------
        max_hop : int
            The maximum hop value to test.
        """
        params = {
            "KHopFE": {"max_hop": max_hop, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert transformed.x.shape == (3, 1 + (max_hop - 1))  # fixed dimension output

    def test_encodings_use_original_features(self):
        """Test that all encodings are computed on original features, not modified ones."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }

        # Compute HKFE alone
        transform_hkfe = CombinedFEs(
            encodings=["HKFE"],
            parameters={"HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False}}
        )
        data_hkfe = transform_hkfe(Data(
            x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes
        ))
        hkfe_alone = data_hkfe.HKFE.clone()

        # Compute KHopFE alone
        transform_khop = CombinedFEs(
            encodings=["KHopFE"],
            parameters={"KHopFE": {"max_hop": 4, "concat_to_x": False}}
        )
        data_khop = transform_khop(Data(
            x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes
        ))
        khop_alone = data_khop.KHopFE.clone()

        # Compute both together
        transform_both = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        data_both = Data(
            x=self.x.clone(), edge_index=self.edge_index, num_nodes=self.num_nodes
        )
        transformed = transform_both(data_both)

        # Extract the encoding portions from the combined result
        # Shape: [original (1), HKFE (4), KHopFE (3)]
        hkfe_combined = transformed.x[:, 1:5]
        khop_combined = transformed.x[:, 5:8]

        # Both should match their standalone versions
        assert torch.allclose(hkfe_combined, hkfe_alone), \
            "HKFE encoding differs when combined vs standalone"
        assert torch.allclose(khop_combined, khop_alone), \
            "KHopFE encoding differs when combined vs standalone"

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
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        transformed = transform(data)

        assert transformed.x.shape[0] == 4
        assert not torch.isnan(transformed.x).any()

    def test_hkfe_separate_storage(self):
        """Test HKFE with separate storage."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["HKFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, "HKFE")
        assert transformed.HKFE.shape == (3, 4)  # fe_dim (fixed dimension)
        assert torch.equal(transformed.x, self.x)

    def test_khopfe_separate_storage(self):
        """Test KHopFE with separate storage."""
        params = {
            "KHopFE": {"max_hop": 4, "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, "KHopFE")
        assert transformed.KHopFE.shape == (3, 3)  # max_hop-1 (fixed dimension)
        assert torch.equal(transformed.x, self.x)

    def test_both_encodings_combined(self):
        """Test transform with both encoding types combined."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True},
        }
        transform = CombinedFEs(
            encodings=["HKFE", "KHopFE"],
            parameters=params,
        )
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert transformed.x is not None
        assert not torch.isnan(transformed.x).any()
        assert not torch.isinf(transformed.x).any()
        assert transformed.x.dtype == torch.float32

    def test_both_encodings_separate_storage(self):
        """Test both encodings with separate storage."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "KHopFE": {"max_hop": 4, "concat_to_x": False},
        }
        transform = CombinedFEs(
            encodings=["HKFE", "KHopFE"],
            parameters=params,
        )
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, "HKFE")
        assert hasattr(transformed, "KHopFE")
        assert transformed.HKFE.shape == (3, 4)  # fe_dim (fixed dimension)
        assert transformed.KHopFE.shape == (3, 3)  # max_hop-1 (fixed dimension)
        assert torch.equal(transformed.x, self.x)

    def test_hkfe_numerical_stability(self):
        """Test HKFE doesn't produce NaN or Inf."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert not torch.isnan(transformed.x).any()
        assert not torch.isinf(transformed.x).any()

    def test_khopfe_numerical_stability(self):
        """Test KHopFE doesn't produce NaN or Inf."""
        params = {
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["KHopFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert not torch.isnan(transformed.x).any()
        assert not torch.isinf(transformed.x).any()

    def test_hkfe_complete_graph(self):
        """Test HKFE on a complete graph."""
        edges = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges).t()
        x = torch.randn(4, 2)

        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE"], parameters=params)
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        transformed = transform(data)

        assert transformed.x.shape == (4, 2 + 4)  # original + fe_dim (fixed)
        assert not torch.isnan(transformed.x).any()

    def test_khopfe_complete_graph(self):
        """Test KHopFE on a complete graph."""
        edges = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges).t()
        x = torch.randn(4, 2)

        params = {
            "KHopFE": {"max_hop": 4, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["KHopFE"], parameters=params)
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        transformed = transform(data)

        assert transformed.x.shape == (4, 2 + 3)  # original + max_hop-1 (fixed)
        assert not torch.isnan(transformed.x).any()

    def test_encoding_output_dtype_is_float32(self):
        """Test that all encodings produce float32 output."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "KHopFE": {"max_hop": 4, "concat_to_x": False},
        }
        transform = CombinedFEs(
            encodings=["HKFE", "KHopFE"],
            parameters=params,
        )
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert transformed.HKFE.dtype == torch.float32
        assert transformed.KHopFE.dtype == torch.float32

    def test_with_multiple_input_features(self):
        """Test encodings with multiple input features."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "KHopFE": {"max_hop": 4, "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["HKFE", "KHopFE"], parameters=params)
        x_multi = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        data = Data(x=x_multi, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Fixed dimension output - fe_dim=4 regardless of input features
        assert transformed.HKFE.shape == (3, 4)
        # Fixed dimension output - max_hop-1=3 regardless of input features
        assert transformed.KHopFE.shape == (3, 3)

    # ========== SheafConnLapPE Tests ==========

    def test_initialization_with_sheaf(self):
        """Test initialization with SheafConnLapPE encoding."""
        transform = CombinedFEs(encodings=["SheafConnLapPE"])
        assert transform.encodings == ["SheafConnLapPE"]
        assert transform.parameters == {}

        # Test with all three encodings
        transform = CombinedFEs(encodings=["HKFE", "KHopFE", "SheafConnLapPE"])
        assert transform.encodings == ["HKFE", "KHopFE", "SheafConnLapPE"]

    def test_single_sheaf_encoding(self):
        """Test transform with only SheafConnLapPE encoding."""
        # SheafConnLapPE requires feature_dim >= stalk_dim (default 3)
        x = torch.randn(self.num_nodes, 5)
        params = {
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert transformed.x is not None
        assert transformed.x.shape[0] == 3
        # Original 5 features + max_pe_dim=6
        assert transformed.x.shape[1] == 5 + 6

    def test_sheaf_separate_storage(self):
        """Test SheafConnLapPE with separate storage."""
        x = torch.randn(self.num_nodes, 5)
        params = {
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, "SheafConnLapPE")
        assert transformed.SheafConnLapPE.shape == (3, 6)
        # Original features unchanged
        assert transformed.x.shape == (3, 5)

    def test_sheaf_combined_with_hkfe(self):
        """Test SheafConnLapPE combined with HKFE."""
        x = torch.randn(self.num_nodes, 5)
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Original 5 + HKFE 4 + SheafConnLapPE 6 = 15
        assert transformed.x.shape == (3, 5 + 4 + 6)
        assert not torch.isnan(transformed.x).any()

    def test_sheaf_combined_with_khopfe(self):
        """Test SheafConnLapPE combined with KHopFE."""
        x = torch.randn(self.num_nodes, 5)
        params = {
            "KHopFE": {"max_hop": 4, "concat_to_x": True},
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["KHopFE", "SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Original 5 + KHopFE 3 + SheafConnLapPE 6 = 14
        assert transformed.x.shape == (3, 5 + 3 + 6)
        assert not torch.isnan(transformed.x).any()

    def test_all_three_encodings(self):
        """Test all three encodings combined."""
        x = torch.randn(self.num_nodes, 5)
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True},
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(
            encodings=["HKFE", "KHopFE", "SheafConnLapPE"],
            parameters=params
        )
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Original 5 + HKFE 4 + KHopFE 3 + SheafConnLapPE 6 = 18
        assert transformed.x.shape == (3, 5 + 4 + 3 + 6)
        assert not torch.isnan(transformed.x).any()
        assert not torch.isinf(transformed.x).any()

    def test_all_three_separate_storage(self):
        """Test all three encodings with separate storage."""
        x = torch.randn(self.num_nodes, 5)
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "KHopFE": {"max_hop": 4, "concat_to_x": False},
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": False}
        }
        transform = CombinedFEs(
            encodings=["HKFE", "KHopFE", "SheafConnLapPE"],
            parameters=params
        )
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, "HKFE")
        assert hasattr(transformed, "KHopFE")
        assert hasattr(transformed, "SheafConnLapPE")
        assert transformed.HKFE.shape == (3, 4)
        assert transformed.KHopFE.shape == (3, 3)
        assert transformed.SheafConnLapPE.shape == (3, 6)
        # Original features unchanged
        assert transformed.x.shape == (3, 5)

    def test_sheaf_numerical_stability(self):
        """Test SheafConnLapPE doesn't produce NaN or Inf."""
        x = torch.randn(self.num_nodes, 5)
        params = {
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert not torch.isnan(transformed.x).any()
        assert not torch.isinf(transformed.x).any()

    def test_sheaf_requires_sufficient_features(self):
        """Test SheafConnLapPE raises error when feature_dim < stalk_dim."""
        # Only 2 features but stalk_dim=3 (default)
        x = torch.randn(self.num_nodes, 2)
        params = {
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        with pytest.raises(ValueError, match="feature_dim.*must be >= stalk_dim"):
            transform(data)

    def test_sheaf_max_pe_dim_divisibility(self):
        """Test SheafConnLapPE requires max_pe_dim divisible by stalk_dim."""
        x = torch.randn(self.num_nodes, 5)
        params = {
            "SheafConnLapPE": {"max_pe_dim": 7, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        with pytest.raises(ValueError, match="must be divisible by"):
            transform(data)

    @pytest.mark.parametrize("max_pe_dim,stalk_dim", [
        (6, 2),
        (6, 3),
        (9, 3),
        (12, 4),
    ])
    def test_parametrized_sheaf_dimensions(self, max_pe_dim, stalk_dim):
        """Parametrized test for different SheafConnLapPE dimensions."""
        x = torch.randn(self.num_nodes, 5)
        params = {
            "SheafConnLapPE": {
                "max_pe_dim": max_pe_dim,
                "stalk_dim": stalk_dim,
                "concat_to_x": False
            }
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert hasattr(transformed, "SheafConnLapPE")
        assert transformed.SheafConnLapPE.shape == (3, max_pe_dim)

    def test_sheaf_on_larger_graph(self):
        """Test SheafConnLapPE on a larger graph."""
        num_nodes = 20
        num_edges = 60
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, 8)

        params = {
            "SheafConnLapPE": {"max_pe_dim": 9, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

        transformed = transform(data)

        assert transformed.x.shape == (num_nodes, 8 + 9)
        assert not torch.isnan(transformed.x).any()

    def test_sheaf_complete_graph(self):
        """Test SheafConnLapPE on a complete graph."""
        edges = []
        for i in range(4):
            for j in range(4):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges).t()
        x = torch.randn(4, 5)

        params = {
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        transformed = transform(data)

        assert transformed.x.shape == (4, 5 + 6)
        assert not torch.isnan(transformed.x).any()

    def test_sheaf_empty_graph(self):
        """Test SheafConnLapPE on an empty graph (no edges)."""
        x = torch.randn(self.num_nodes, 5)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        params = {
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Should return zero PE for degenerate graph
        assert transformed.x.shape == (3, 5 + 6)

    def test_sheaf_single_node(self):
        """Test SheafConnLapPE on a single node graph."""
        x = torch.randn(1, 5)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        params = {
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=edge_index, num_nodes=1)

        transformed = transform(data)

        assert transformed.x.shape == (1, 5 + 6)

    def test_sheaf_output_dtype_is_float32(self):
        """Test that SheafConnLapPE produces float32 output."""
        x = torch.randn(self.num_nodes, 5)
        params = {
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["SheafConnLapPE"], parameters=params)
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert transformed.SheafConnLapPE.dtype == torch.float32

    # ========== PPRFE Tests ==========

    def test_initialization_with_pprfe(self):
        """Test initialization with PPRFE encoding."""
        transform = CombinedFEs(encodings=["PPRFE"])
        assert transform.encodings == ["PPRFE"]

        # Test with all encodings including PPRFE
        transform = CombinedFEs(encodings=["HKFE", "KHopFE", "SheafConnLapPE", "PPRFE"])
        assert transform.encodings == ["HKFE", "KHopFE", "SheafConnLapPE", "PPRFE"]

    def test_single_pprfe_encoding(self):
        """Test transform with only PPRFE encoding."""
        params = {
            "PPRFE": {"alpha_param_PPRFE": (0.1, 5), "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["PPRFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Original 1 + PPRFE 5 = 6
        assert transformed.x.shape == (3, 6)

    def test_pprfe_separate_storage(self):
        """Test PPRFE with separate storage."""
        params = {
            "PPRFE": {"alpha_param_PPRFE": (0.1, 5), "concat_to_x": False}
        }
        transform = CombinedFEs(encodings=["PPRFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        assert transformed.x.shape == (3, 1)  # Original unchanged
        assert hasattr(transformed, "PPRFE")
        assert transformed.PPRFE.shape == (3, 5)

    def test_pprfe_combined_with_hkfe(self):
        """Test PPRFE combined with HKFE."""
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "PPRFE": {"alpha_param_PPRFE": (0.1, 5), "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["HKFE", "PPRFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Original 1 + HKFE 4 + PPRFE 5 = 10
        assert transformed.x.shape == (3, 10)

    def test_pprfe_combined_with_khopfe(self):
        """Test PPRFE combined with KHopFE."""
        params = {
            "KHopFE": {"max_hop": 4, "concat_to_x": True},
            "PPRFE": {"alpha_param_PPRFE": (0.1, 5), "concat_to_x": True}
        }
        transform = CombinedFEs(encodings=["KHopFE", "PPRFE"], parameters=params)
        data = Data(x=self.x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Original 1 + KHopFE 3 + PPRFE 5 = 9
        assert transformed.x.shape == (3, 9)

    def test_all_four_encodings(self):
        """Test all four encodings together."""
        x = torch.randn(3, 5)  # Need more features for SheafConnLapPE
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": True},
            "KHopFE": {"max_hop": 4, "concat_to_x": True},
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": True},
            "PPRFE": {"alpha_param_PPRFE": (0.1, 5), "concat_to_x": True}
        }
        transform = CombinedFEs(
            encodings=["HKFE", "KHopFE", "SheafConnLapPE", "PPRFE"],
            parameters=params
        )
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Original 5 + HKFE 4 + KHopFE 3 + SheafConnLapPE 6 + PPRFE 5 = 23
        assert transformed.x.shape == (3, 23)

    def test_all_four_separate_storage(self):
        """Test all four encodings with separate storage."""
        x = torch.randn(3, 5)
        params = {
            "HKFE": {"kernel_param_HKFE": (1, 5), "concat_to_x": False},
            "KHopFE": {"max_hop": 4, "concat_to_x": False},
            "SheafConnLapPE": {"max_pe_dim": 6, "stalk_dim": 3, "concat_to_x": False},
            "PPRFE": {"alpha_param_PPRFE": (0.1, 5), "concat_to_x": False}
        }
        transform = CombinedFEs(
            encodings=["HKFE", "KHopFE", "SheafConnLapPE", "PPRFE"],
            parameters=params
        )
        data = Data(x=x, edge_index=self.edge_index, num_nodes=self.num_nodes)

        transformed = transform(data)

        # Original unchanged
        assert transformed.x.shape == (3, 5)
        # All stored separately
        assert hasattr(transformed, "HKFE")
        assert hasattr(transformed, "KHopFE")
        assert hasattr(transformed, "SheafConnLapPE")
        assert hasattr(transformed, "PPRFE")
        assert transformed.HKFE.shape == (3, 4)
        assert transformed.KHopFE.shape == (3, 3)
        assert transformed.SheafConnLapPE.shape == (3, 6)
        assert transformed.PPRFE.shape == (3, 5)


class TestSelectDestinationFEs:
    """Test SelectDestinationFEs transform."""

    def setup_method(self):
        """Set up shared fixtures."""
        from topobench.transforms.data_manipulations.combined_feature_encodings import SelectDestinationFEs
        self.cls = SelectDestinationFEs
        self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    def _make_data_with_encodings(self, n_nodes=5):
        """Build a minimal Data object that carries HKFE and KHopFE attributes."""
        x = torch.randn(n_nodes, 3)
        hkfe = torch.randn(n_nodes, 4)
        khopfe = torch.randn(n_nodes, 2)
        from torch_geometric.data import Data
        data = Data(x=x, HKFE=hkfe, KHopFE=khopfe)
        return data

    def test_forward_slices_to_n_dst(self):
        """forward() keeps only the first n_dst_nodes rows of each encoding."""
        transform = self.cls(encodings=["HKFE", "KHopFE"])
        data = self._make_data_with_encodings(n_nodes=5)
        n_dst = 3

        out = transform.forward(data, n_dst)

        assert out.x.shape == (3, 3)
        assert out.HKFE.shape == (3, 4)
        assert out.KHopFE.shape == (3, 2)

    def test_call_delegates_to_forward(self):
        """__call__ is equivalent to forward."""
        transform = self.cls(encodings=["HKFE"])
        data = self._make_data_with_encodings(n_nodes=4)

        out_call = transform(data, 2)
        out_fwd = transform.forward(data, 2)

        assert torch.equal(out_call.x, out_fwd.x)
        assert torch.equal(out_call.HKFE, out_fwd.HKFE)

    def test_missing_encoding_raises_value_error(self):
        """forward() raises ValueError when a requested encoding is absent."""
        transform = self.cls(encodings=["HKFE", "MISSING_KEY"])
        data = self._make_data_with_encodings(n_nodes=4)

        with pytest.raises(ValueError, match="MISSING_KEY"):
            transform.forward(data, 2)

    def test_none_x_handled(self):
        """forward() handles data.x = None without crashing."""
        from torch_geometric.data import Data
        transform = self.cls(encodings=["HKFE"])
        data = Data(x=None, HKFE=torch.randn(5, 4))

        out = transform(data, 3)

        assert out.x is None
        assert out.HKFE.shape == (3, 4)
