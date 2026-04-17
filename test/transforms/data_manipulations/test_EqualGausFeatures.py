"""Test EqualGausFeatures transform."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import EqualGausFeatures


class TestEqualGausFeatures:
    """Test EqualGausFeatures transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Using the default values from config
        self.mean = 0.0
        self.std = 0.1
        self.num_features = 3  # example value, would be from dataset.parameters
        self.transform = EqualGausFeatures(
            mean=self.mean,
            std=self.std,
            num_features=self.num_features
        )

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test with config default parameters
        transform = EqualGausFeatures(
            mean=0.0,
            std=0.1,
            num_features=5
        )
        assert transform.type == "generate_non_informative_features"
        assert transform.mean == 0.0
        assert transform.std == 0.1
        assert transform.feature_vector is not None

        # Test with custom parameters
        custom_transform = EqualGausFeatures(
            mean=1.0,
            std=0.5,
            num_features=2
        )
        assert custom_transform.mean == 1.0
        assert custom_transform.std == 0.5
        assert custom_transform.feature_vector is not None

    def test_repr(self):
        """Test string representation of the transform."""
        repr_str = repr(self.transform)
        assert "EqualGausFeatures" in repr_str
        assert f"mean={self.mean}" in repr_str
        assert f"std={self.std}" in repr_str
        assert "feature_vector=" in repr_str

    def test_forward_basic(self):
        """Test basic forward pass."""
        data = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            num_nodes=2
        )

        transformed = self.transform(data)

        # Check output dimensions
        assert transformed.x.size() == (2, self.num_features)  # num_nodes x num_features
        assert transformed.num_nodes == 2

        # Check other attributes are preserved
        assert torch.equal(transformed.edge_index, data.edge_index)

    def test_feature_consistency(self):
        """Test that features are consistent for same input."""
        data = Data(
            x=torch.tensor([[1.0], [2.0]]),
            num_nodes=2
        )

        # Transform same data twice
        result1 = self.transform(data)
        result2 = self.transform(data)

        # Should get exactly same features since using same feature vector
        assert torch.equal(result1.x, result2.x)

    def test_different_node_counts(self):
        """Test with different numbers of nodes."""
        node_counts = [1, 10, 100]

        for n in node_counts:
            data = Data(
                x=torch.randn(n, 2),
                edge_index=torch.zeros((2, 0)),
                num_nodes=n
            )

            transformed = self.transform(data)
            assert transformed.x.size() == (n, self.num_features)
            assert transformed.num_nodes == n

    def test_empty_graph(self):
        """Test transform on empty graph."""
        data = Data(
            x=torch.tensor([], dtype=torch.float).reshape(0, 1),
            edge_index=torch.tensor([[],[]]),
            num_nodes=0
        )

        transformed = self.transform(data)
        assert transformed.x.size() == (0, self.num_features)
        assert transformed.num_nodes == 0

    def test_feature_values(self):
        """Test the generated feature values."""
        num_nodes = 1000  # Large number for stable statistics
        data = Data(
            x=torch.randn(num_nodes, 1),
            num_nodes=num_nodes
        )

        transformed = self.transform(data)
        features = transformed.x

        # Basic value checks
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()

        # Approximate distribution checks (should be loose enough)
        mean_diff = torch.abs(features.mean() - self.mean)
        std_diff = torch.abs(features.std() - self.std)
        assert mean_diff < 0.5, f"Mean differs by {mean_diff}"
        assert std_diff < 0.5, f"Std differs by {std_diff}"

    def test_attribute_preservation(self):
        """Test preservation of additional attributes."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.tensor([[1.0], [1.0]])
        custom_attr = "test"

        data = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=edge_index,
            edge_attr=edge_attr,
            custom_attr=custom_attr,
            num_nodes=2
        )

        transformed = self.transform(data)

        assert torch.equal(transformed.edge_index, edge_index)
        assert torch.equal(transformed.edge_attr, edge_attr)
        assert transformed.custom_attr == custom_attr
