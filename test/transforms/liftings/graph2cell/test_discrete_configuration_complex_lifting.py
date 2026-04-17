"""Test the discrete config. complex lifting."""
import pytest
import torch
import torch_geometric
from unittest.mock import MagicMock
from topobench.transforms.liftings.graph2cell.discrete_configuration_complex_lifting import DiscreteConfigurationComplexLifting

@pytest.fixture
def mock_data():
    """Create a mock data object for testing.

    Returns
    -------
    MagicMock
        A mock object representing a graph data.
    """
    data = MagicMock(spec=torch_geometric.data.Data)
    data.y = torch.tensor([1, 0])
    data.edge_index = torch.tensor([[0, 1], [1, 0]])
    data.x = torch.tensor([[1, 2], [3, 4]])
    data.num_nodes = 2
    return data

def test_discrete_configuration_complex_lifting_init():
    """Test the initialization of the DiscreteConfigurationComplexLifting class."""
    lifting = DiscreteConfigurationComplexLifting(k=2, complex_dim=2)
    assert lifting.k == 2
    assert lifting.complex_dim == 2
    assert lifting.feature_aggregation == "concat"

def test_discrete_configuration_complex_lifting_forward(mock_data):
    """Test the forward method of the DiscreteConfigurationComplexLifting class.

    Parameters
    ----------
    mock_data : MagicMock
        A mock object representing a graph data.
    """
    lifting = DiscreteConfigurationComplexLifting(k=2, complex_dim=2)
    lifted_data = lifting.forward(mock_data)
    assert isinstance(lifted_data, torch_geometric.data.Data)
    assert torch.equal(lifted_data.y, mock_data.y)

def test_discrete_configuration_complex_lifting_lift_topology(mock_data):
    """Test the lift_topology method of the DiscreteConfigurationComplexLifting class.

    Parameters
    ----------
    mock_data : MagicMock
        A mock object representing a graph data.
    """
    lifting = DiscreteConfigurationComplexLifting(k=2, complex_dim=2)
    lifted_topology = lifting.lift_topology(mock_data)
    assert isinstance(lifted_topology, dict)
    assert "adjacency_0" in lifted_topology
    assert "adjacency_1" in lifted_topology

class TestDiscreteConfigurationComplexLifting:
    """Test the DiscreteConfigurationComplexLifting class."""

    def setup_method(self):
        """Initialise the DiscreteConfigurationComplexLifting class."""
        self.lifting_concat = DiscreteConfigurationComplexLifting(k=2, complex_dim=2, feature_aggregation="concat")
        self.lifting_sum = DiscreteConfigurationComplexLifting(k=2, complex_dim=2, feature_aggregation="sum")
        self.lifting_mean = DiscreteConfigurationComplexLifting(k=2, complex_dim=2, feature_aggregation="mean")

    def test_lift_topology(self, simple_graph_1):
        """Test the lift_topology method.

        Parameters
        ----------
        simple_graph_1 : Data
            A simple graph used for testing.
        """
        data = simple_graph_1

        assert self.lifting_concat.forward(data.clone()).incidence_1.shape[1] == 156, "Something is wrong with incidence_1."
        assert self.lifting_sum.forward(data.clone()).incidence_1.shape[1] == 156, "Something is wrong with incidence_1."
        assert self.lifting_mean.forward(data.clone()).incidence_1.shape[1] == 156, "Something is wrong with incidence_1."



# import torch

# from topobench.data.utils.utils import load_simple_configuration_graphs
# from topobench.transforms.liftings.graph2cell.discrete_configuration_complex_lifting import (
#     DiscreteConfigurationComplexLifting,
# )


# class TestDiscreteConfigurationComplexLifting:
#     """Test the DiscreteConfigurationComplexLifting class."""

#     def setup_method(self):
#         # Load the graph
#         self.dataset = load_simple_configuration_graphs()

#         # Initialise the DiscreteConfigurationComplexLifting class
#         self.liftings = [
#             DiscreteConfigurationComplexLifting(
#                 k=2, preserve_edge_attr=True, feature_aggregation="mean"
#             ),
#             DiscreteConfigurationComplexLifting(
#                 k=2, preserve_edge_attr=True, feature_aggregation="sum"
#             ),
#             DiscreteConfigurationComplexLifting(
#                 k=2, preserve_edge_attr=True, feature_aggregation="concat"
#             ),
#         ]

#     def test_lift_topology(self):
#         # Test the lift_topology method

#         expected_incidences_data_0 = (
#             torch.tensor(
#                 [
#                     [
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         1.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                     ],
#                 ]
#             ),
#             torch.tensor([]),
#         )

#         expected_incidences_data_1 = (
#             torch.tensor(
#                 [
#                     [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
#                     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
#                     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#                 ]
#             ),
#             torch.tensor([]),
#         )

#         expected_incidences_data_2 = (
#             torch.tensor(
#                 [
#                     [
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         1.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                         0.0,
#                         1.0,
#                     ],
#                     [
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         0.0,
#                         1.0,
#                         1.0,
#                     ],
#                 ]
#             ),
#             torch.tensor(
#                 [
#                     [0.0, 0.0],
#                     [0.0, 0.0],
#                     [1.0, 0.0],
#                     [1.0, 0.0],
#                     [1.0, 0.0],
#                     [0.0, 0.0],
#                     [0.0, 0.0],
#                     [0.0, 0.0],
#                     [1.0, 0.0],
#                     [0.0, 0.0],
#                     [0.0, 1.0],
#                     [0.0, 1.0],
#                     [0.0, 1.0],
#                     [0.0, 1.0],
#                     [0.0, 0.0],
#                     [0.0, 0.0],
#                 ]
#             ),
#         )

#         for lifting in self.liftings:
#             lifted_data = lifting.forward(self.dataset[0].clone())
#             assert (
#                 expected_incidences_data_0[0] == lifted_data.incidence_1.to_dense()
#             ).all(), f"Something is wrong with incidence_1 for graph 0, {lifting.feature_aggregation} aggregation."

#             assert (
#                 expected_incidences_data_0[1] == lifted_data.incidence_2.to_dense()
#             ).all(), f"Something is wrong with incidence_2 for graph 0, {lifting.feature_aggregation} aggregation."

#             lifted_data = lifting.forward(self.dataset[1].clone())
#             assert (
#                 expected_incidences_data_1[0] == lifted_data.incidence_1.to_dense()
#             ).all(), f"Something is wrong with incidence_1 for graph 1, {lifting.feature_aggregation} aggregation."

#             assert (
#                 expected_incidences_data_1[1] == lifted_data.incidence_2.to_dense()
#             ).all(), f"Something is wrong with incidence_2 for graph 1, {lifting.feature_aggregation} aggregation."

#             lifted_data = lifting.forward(self.dataset[2].clone())
#             assert (
#                 expected_incidences_data_2[0] == lifted_data.incidence_1.to_dense()
#             ).all(), f"Something is wrong with incidence_1 for graph 2, {lifting.feature_aggregation} aggregation."

#             assert (
#                 expected_incidences_data_2[1] == lifted_data.incidence_2.to_dense()
#             ).all(), f"Something is wrong with incidence_2 for graph 2, {lifting.feature_aggregation} aggregation."
