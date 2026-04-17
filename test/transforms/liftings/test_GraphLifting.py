"""Test the GraphLifting class."""
import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.liftings import GraphLifting


class ConcreteGraphLifting(GraphLifting):
    """Concrete implementation of GraphLifting for testing."""

    def lift_topology(self, data):
        """Implement the abstract lift_topology method.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            Empty dictionary for testing purposes.
        """
        return {}


class TestGraphLifting:
    """Test the GraphLifting class."""

    def setup_method(self):
        """Set up test fixtures before each test method.

        Creates an instance of ConcreteGraphLifting with default parameters.
        """
        self.lifting = ConcreteGraphLifting(
            feature_lifting="ProjectionSum",
            preserve_edge_attr=False
        )

    def test_data_has_edge_attr(self):
        """Test _data_has_edge_attr method with different data configurations."""

        # Test case 1: Data with edge attributes
        data_with_edge_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.tensor([[1.0], [1.0]])
        )
        assert self.lifting._data_has_edge_attr(data_with_edge_attr) is True

        # Test case 2: Data without edge attributes
        data_without_edge_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]])
        )
        assert self.lifting._data_has_edge_attr(data_without_edge_attr) is False

        # Test case 3: Data with edge_attr set to None
        data_with_none_edge_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=None
        )
        assert self.lifting._data_has_edge_attr(data_with_none_edge_attr) is False

    def test_data_has_edge_attr_empty_data(self):
        """Test _data_has_edge_attr method with empty data object."""
        empty_data = Data()
        assert self.lifting._data_has_edge_attr(empty_data) is False

    def test_data_has_edge_attr_different_edge_formats(self):
        """Test _data_has_edge_attr method with different edge attribute formats."""

        # Test with float edge attributes
        data_float_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.tensor([[0.5], [0.5]])
        )
        assert self.lifting._data_has_edge_attr(data_float_attr) is True

        # Test with integer edge attributes
        data_int_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.tensor([[1], [1]], dtype=torch.long)
        )
        assert self.lifting._data_has_edge_attr(data_int_attr) is True

        # Test with multi-dimensional edge attributes
        data_multidim_attr = Data(
            x=torch.tensor([[1.0], [2.0]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        )
        assert self.lifting._data_has_edge_attr(data_multidim_attr) is True

    @pytest.mark.parametrize("preserve_edge_attr", [True, False])
    def test_init_preserve_edge_attr(self, preserve_edge_attr):
        """Test initialization with different preserve_edge_attr values.

        Parameters
        ----------
        preserve_edge_attr : bool
            Boolean value to test initialization with True and False values.
        """
        lifting = ConcreteGraphLifting(
            feature_lifting="ProjectionSum",
            preserve_edge_attr=preserve_edge_attr
        )
        assert lifting.preserve_edge_attr == preserve_edge_attr
