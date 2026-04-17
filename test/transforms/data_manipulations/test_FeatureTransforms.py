"""Test feature manipulation transforms."""

import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import (
    NodeFeaturesToFloat,
    OneHotDegreeFeatures,
    NodeDegrees,
)


class TestFeatureTransforms:
    """Test feature manipulation transforms."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.data = Data(
            num_nodes=4,
            x=torch.tensor([[10], [20], [30], [40]]),
            edge_index=torch.tensor([[0, 0, 0, 2], [1, 2, 3, 3]])
        )

        self.node_degrees = NodeDegrees(selected_fields=["edge_index"])
        self.node_feature_float = NodeFeaturesToFloat()
        self.one_hot_degree_features = OneHotDegreeFeatures(
            max_degree=3,
            degrees_fields="node_degrees",
            features_fields="one_hot_degree"
        )

    def test_node_degrees(self):
        """Test node degrees calculation."""
        data = self.node_degrees(self.data.clone())
        expected_degrees = torch.tensor([[3], [0], [1], [0]]).float()
        assert (
            data.node_degrees == expected_degrees
        ).all(), "Node degrees do not match"

    def test_node_feature_float(self):
        """Test node features to float conversion."""
        data = self.node_feature_float(self.data.clone())
        assert data.x.is_floating_point(), "Node features are not float"

    def test_one_hot_degree_features(self):
        """Test one-hot degree features encoding."""
        data = self.node_degrees(self.data.clone())
        data = self.one_hot_degree_features(data)
        expected_vals = torch.tensor([
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ])
        assert (data.one_hot_degree == expected_vals).all()
