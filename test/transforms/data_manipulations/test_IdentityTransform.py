"""Test IdentityTransform class."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import IdentityTransform


class TestIdentityTransform:
    """Test IdentityTransform class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.transform = IdentityTransform()

    def test_initialization(self):
        """Test initialization of the transform."""
        assert self.transform.type == "domain2domain"
        assert isinstance(self.transform.parameters, dict)
        assert len(self.transform.parameters) == 0

        # Test with custom parameters
        params = {"param1": "value1", "param2": 42}
        transform = IdentityTransform(**params)
        assert transform.parameters == params

    def test_repr(self):
        """Test string representation of the transform."""
        repr_str = repr(self.transform)
        assert "IdentityTransform" in repr_str
        assert "domain2domain" in repr_str
        assert "parameters={}" in repr_str

        # Test repr with parameters
        transform = IdentityTransform(param="test")
        repr_str = repr(transform)
        assert "param" in repr_str
        assert "test" in repr_str

    def test_forward_simple_graph(self):
        """Test transform on a simple graph."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        x = torch.tensor([[1.0], [2.0]])
        data = Data(x=x, edge_index=edge_index, num_nodes=2)

        transformed = self.transform(data)

        # Check that all attributes are equal
        assert torch.equal(transformed.edge_index, data.edge_index)
        assert torch.equal(transformed.x, data.x)
        assert transformed.num_nodes == data.num_nodes

    def test_forward_with_attributes(self):
        """Test transform on a graph with multiple attributes."""
        edge_index = torch.tensor([[0, 1], [1, 0]])
        x = torch.tensor([[1.0], [2.0]])
        edge_attr = torch.tensor([[1.0], [1.0]])
        custom_tensor = torch.randn(5, 3)
        custom_string = "test_string"
        custom_int = 42

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=2,
            custom_tensor=custom_tensor,
            custom_string=custom_string,
            custom_int=custom_int
        )

        transformed = self.transform(data)

        # Check all attributes remain equal
        assert torch.equal(transformed.x, data.x)
        assert torch.equal(transformed.edge_index, data.edge_index)
        assert torch.equal(transformed.edge_attr, data.edge_attr)
        assert torch.equal(transformed.custom_tensor, data.custom_tensor)
        assert transformed.custom_string == data.custom_string
        assert transformed.custom_int == data.custom_int
        assert transformed.num_nodes == data.num_nodes

    def test_forward_empty_graph(self):
        """Test transform on an empty graph."""
        data = Data(
            x=torch.tensor([], dtype=torch.float).reshape((0, 1)),
            edge_index=torch.tensor([[],[]]),
            num_nodes=0
        )

        transformed = self.transform(data)

        assert transformed.num_nodes == 0
        assert transformed.edge_index.size() == data.edge_index.size()
        assert transformed.x.size() == data.x.size()
        assert torch.equal(transformed.edge_index, data.edge_index)
        assert torch.equal(transformed.x, data.x)

    def test_forward_large_graph(self):
        """Test transform on a large graph."""
        num_nodes = 1000
        num_edges = 5000

        x = torch.randn(num_nodes, 10)  # 10 features per node
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.randn(num_edges, 5)  # 5 features per edge

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes
        )

        transformed = self.transform(data)

        assert torch.equal(transformed.x, data.x)
        assert torch.equal(transformed.edge_index, data.edge_index)
        assert torch.equal(transformed.edge_attr, data.edge_attr)
        assert transformed.num_nodes == data.num_nodes

    def test_data_consistency(self):
        """Test that transform preserves data consistency."""
        data = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            num_nodes=2
        )

        transformed = self.transform(data)

        # Check key attributes remain equal
        for key in data.keys():  # Changed from data.keys to data.keys()
            assert hasattr(transformed, key)
            if torch.is_tensor(getattr(data, key)):
                assert torch.equal(getattr(transformed, key), getattr(data, key))
            else:
                assert getattr(transformed, key) == getattr(data, key)
    def test_with_different_dtypes(self):
        """Test transform with different data types."""
        data = Data(
            x=torch.tensor([[1], [2]], dtype=torch.long),
            y=torch.tensor([1.0, 2.0], dtype=torch.float),
            z=torch.tensor([True, False], dtype=torch.bool),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            num_nodes=2
        )

        transformed = self.transform(data)

        assert transformed.x.dtype == torch.long
        assert transformed.y.dtype == torch.float
        assert transformed.z.dtype == torch.bool
        assert torch.equal(transformed.x, data.x)
        assert torch.equal(transformed.y, data.y)
        assert torch.equal(transformed.z, data.z)
