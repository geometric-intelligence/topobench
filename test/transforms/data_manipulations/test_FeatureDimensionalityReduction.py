"""Test FeatureDimensionalityReduction transform."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import FeatureDimensionalityReduction


class TestFeatureDimensionalityReduction:
    """Test FeatureDimensionalityReduction transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Using the default values from config
        self.reduced_dim = 3  # example value, would be from dataset.parameters.num_features
        self.svd_iter = 20
        self.svd_seed = 42
        self.transform = FeatureDimensionalityReduction(
            reduced_dim=self.reduced_dim,
            svd_iter=self.svd_iter,
            svd_seed=self.svd_seed
        )

    @staticmethod
    def _make_sparse_data(num_nodes: int=5, num_features: int=30):
        """Create a simple Data object with sparse node features."""
        torch.manual_seed(42)
        x_dense = torch.randn(num_nodes, num_features, dtype=torch.float32)
        x_sparse = x_dense.to_sparse()
        
        if num_nodes > 1:
            #num_edges = 0.5 * num_nodes * (num_nodes - 1)
            #edge_index = torch.randint(0, num_nodes, (2, num_edges))
            row = torch.arange(0, num_nodes - 1)
            col = torch.arange(1, num_nodes)
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(
            x=x_sparse,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        return data

    def test_initialization(self):
        """Test initialization with different parameters."""
        assert self.transform.type == "feature_dim_reduction"
        assert self.transform.reduced_dim == self.reduced_dim
        assert self.transform.svd_iter == self.svd_iter
        assert self.transform.svd_seed == self.svd_seed

        # Check that the internal TruncatedSVD is configured correctly
        assert self.transform.svd.n_components == self.reduced_dim
        assert self.transform.svd.n_iter == self.svd_iter
        assert self.transform.svd.random_state == self.svd_seed

    def test_repr(self):
        """Test string representation of the transform."""
        repr_str = repr(self.transform)
        assert "FeatureDimensionalityReduction" in repr_str
        assert "feature_dim_reduction" in repr_str
        assert f"reduced_dim={self.reduced_dim}" in repr_str
        assert f"svd_iter={self.svd_iter}" in repr_str
        assert f"svd_seed={self.svd_seed}" in repr_str
        # __repr__ includes the underlying SVD object as `svd_red=...`
        assert "svd_red=" in repr_str

    def test_forward_basic(self):
        """Test basic forward pass on sparse node features."""
        num_nodes = 5
        num_features = 30
        data = self._make_sparse_data(
            num_nodes=num_nodes,
            num_features=num_features
        )

        transformed = self.transform(data)

        # Feature matrix should be dense and reduced to (num_nodes, reduced_dim)
        assert transformed.x.size() == (num_nodes, self.reduced_dim)
        assert transformed.x.dtype == torch.float32

        # Check other attributes are preserved
        assert transformed.num_nodes == data.num_nodes
        assert torch.equal(transformed.edge_index, data.edge_index)

    def test_forward_without_x_attribute(self):
        """Test transform on a graph without node features (x is None)."""
        data = Data(
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            num_nodes=2,
        )

        transformed = self.transform(data)

        # Check that is the same object
        assert data.x is None
        assert transformed.x is None
        assert torch.equal(transformed.edge_index, data.edge_index)
        assert transformed.num_nodes == data.num_nodes

    def test_deterministic_with_same_seed(self):
        """Test deterministic behavior with the same seed."""
        data1 = self._make_sparse_data(
            num_nodes=6,
            num_features=7
        )
        data2 = data1.clone()

        t1 = FeatureDimensionalityReduction(
            reduced_dim=self.reduced_dim,
            svd_iter=self.svd_iter,
            svd_seed=self.svd_seed,
        )
        t2 = FeatureDimensionalityReduction(
            reduced_dim=self.reduced_dim,
            svd_iter=self.svd_iter,
            svd_seed=self.svd_seed,
        )
        result1 = t1(data1).x
        result2 = t2(data2).x

        assert result1.shape == result2.shape
        assert torch.allclose(result1, result2, atol=1e-6)

    def test_invalid_reduced_dim_raises(self):
        """reduced_dim > num_features should raise a ValueError from SVD."""
        num_nodes = 5
        num_features = 3
        data = self._make_sparse_data(
            num_nodes=num_nodes,
            num_features=num_features
        )

        bad_transform = FeatureDimensionalityReduction(
            reduced_dim=num_features + 1,
            svd_iter=self.svd_iter,
            svd_seed=self.svd_seed,
        )

        with pytest.raises(ValueError):
            _ = bad_transform(data)
    
    def test_attribute_preservation(self):
        """Test preservation of additional attributes besides x."""
        num_nodes = 4
        num_features = 5
        data = self._make_sparse_data(
            num_nodes=num_nodes,
            num_features=num_features
        )

        data.edge_attr = torch.randn(data.edge_index.size(1), 2)
        data.custom_attr = "test"

        transformed = self.transform(data)

        # x is changed, but other attributes should be preserved
        assert transformed.x.size() == (num_nodes, self.reduced_dim)
        assert torch.equal(transformed.edge_index, data.edge_index)
        assert torch.equal(transformed.edge_attr, data.edge_attr)
        assert transformed.custom_attr == data.custom_attr