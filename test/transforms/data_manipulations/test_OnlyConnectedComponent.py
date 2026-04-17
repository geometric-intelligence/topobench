"""Test KeepOnlyConnectedComponent transform."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import KeepOnlyConnectedComponent


class TestKeepOnlyConnectedComponent:
    """Test KeepOnlyConnectedComponent transform."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.num_components = 1
        self.transform = KeepOnlyConnectedComponent(
            num_components=self.num_components
        )

    def test_initialization(self):
        """Test initialization of the transform."""
        assert self.transform.type == "keep_connected_component"
        assert self.transform.parameters["num_components"] == self.num_components

        # Test default initialization
        transform = KeepOnlyConnectedComponent()
        assert transform.type == "keep_connected_component"
        assert isinstance(transform.parameters, dict)

    def test_single_component(self):
        """Test transform on a graph with single connected component."""
        # Create a simple connected graph
        edge_index = torch.tensor([
            [0, 1, 1, 2],
            [1, 0, 2, 1]
        ])
        x = torch.tensor([[1.0], [2.0], [3.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=3)

        transformed = self.transform(data.clone())
        assert transformed.num_nodes == 3
        assert transformed.edge_index.size(1) == 4
        assert torch.equal(transformed.x, x)

    def test_multiple_components(self):
        """Test transform on a graph with multiple connected components."""
        # Create a graph with two components: (0,1,2) and (3,4)
        edge_index = torch.tensor([
            [0, 1, 1, 2, 3, 4],
            [1, 0, 2, 1, 4, 3]
        ])
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=5)

        transformed = self.transform(data.clone())
        # Should keep the largest component (0,1,2)
        assert transformed.num_nodes == 3
        assert transformed.edge_index.size(1) == 4

    def test_equal_size_components(self):
        """Test transform on a graph with components of equal size."""
        # Create a graph with two equal-sized components
        edge_index = torch.tensor([
            [0, 1, 2, 3],
            [1, 0, 3, 2]
        ])
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        transformed = self.transform(data.clone())
        assert transformed.num_nodes == 2  # Should keep one component
        assert transformed.edge_index.size(1) == 2

    def test_multiple_num_components(self):
        """Test transform with num_components > 1."""
        transform = KeepOnlyConnectedComponent(num_components=2)

        # Create a graph with three components
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5],  # Three components: (0,1), (2,3), (4,5)
            [1, 0, 3, 2, 5, 4]
        ])
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=6)

        transformed = transform(data.clone())
        assert transformed.num_nodes == 4  # Should keep two components
        assert transformed.edge_index.size(1) == 4

    def test_repr(self):
        """Test string representation of the transform."""
        repr_str = repr(self.transform)
        assert "KeepOnlyConnectedComponent" in repr_str
        assert "keep_connected_component" in repr_str
        assert "num_components" in repr_str
        assert str(self.num_components) in repr_str

    def test_disconnected_nodes(self):
        """Test transform on a graph with disconnected nodes."""
        # Create a graph with connected component and isolated nodes
        edge_index = torch.tensor([
            [0, 1],
            [1, 0]
        ])
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # 2 isolated nodes
        data = Data(x=x, edge_index=edge_index, num_nodes=4)

        transformed = self.transform(data.clone())
        assert transformed.num_nodes == 2  # Should only keep connected component
        assert transformed.edge_index.size(1) == 2
        assert torch.equal(transformed.x, x[:2])

    def test_attributes_preservation(self):
        """Test preservation of additional attributes after transform."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        x = torch.tensor([[1.0], [2.0]])
        data = Data(
            x=x,
            edge_index=edge_index,
            num_nodes=2,
            edge_attr=torch.tensor([[1.0], [1.0]]),
            test_attr="test"
        )

        transformed = self.transform(data.clone())
        assert hasattr(transformed, "edge_attr")
        assert hasattr(transformed, "test_attr")
        assert transformed.test_attr == "test"
