"""Test the knn lifting."""

import pytest
import torch
from torch_geometric.data import Data
from topobench.transforms.liftings.graph2hypergraph import (
    HypergraphKNNLifting,
)


class TestHypergraphKNNLifting:
    """Test the HypergraphKNNLifting class."""

    def setup_method(self):
        """Set up test fixtures before each test method.

        Creates instances of HypergraphKNNLifting with different k values
        and loop settings.
        """
        self.lifting_k2 = HypergraphKNNLifting(k_value=2, loop=True)
        self.lifting_k3 = HypergraphKNNLifting(k_value=3, loop=True)
        self.lifting_no_loop = HypergraphKNNLifting(k_value=2, loop=False)

    def test_initialization(self):
        """Test initialization with different parameters."""
        # Test default parameters
        lifting_default = HypergraphKNNLifting()
        assert lifting_default.k == 1
        assert lifting_default.loop is True

        # Test custom parameters
        lifting_custom = HypergraphKNNLifting(k_value=5, loop=False)
        assert lifting_custom.k == 5
        assert lifting_custom.loop is False

    def test_lift_topology_k2(self, simple_graph_2):
        """Test the lift_topology method with k=2.

        Parameters
        ----------
        simple_graph_2 : torch_geometric.data.Data
            A simple graph fixture with 9 nodes arranged in a line pattern.
        """
        lifted_data_k2 = self.lifting_k2.lift_topology(simple_graph_2.clone())

        expected_n_hyperedges = 9
        expected_incidence_1 = torch.tensor([
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ])

        assert torch.equal(
            lifted_data_k2["incidence_hyperedges"].to_dense(),
            expected_incidence_1
        ), "Incorrect incidence_hyperedges for k=2"

        assert lifted_data_k2["num_hyperedges"] == expected_n_hyperedges
        assert torch.equal(lifted_data_k2["x_0"], simple_graph_2.x)

    def test_lift_topology_k3(self, simple_graph_2):
        """Test the lift_topology method with k=3.

        Parameters
        ----------
        simple_graph_2 : torch_geometric.data.Data
            A simple graph fixture with 9 nodes arranged in a line pattern.
        """
        lifted_data_k3 = self.lifting_k3.lift_topology(simple_graph_2.clone())

        expected_n_hyperedges = 9
        expected_incidence_1 = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ])

        assert torch.equal(
            lifted_data_k3["incidence_hyperedges"].to_dense(),
            expected_incidence_1
        ), "Incorrect incidence_hyperedges for k=3"

        assert lifted_data_k3["num_hyperedges"] == expected_n_hyperedges
        assert torch.equal(lifted_data_k3["x_0"], simple_graph_2.x)

    def test_lift_topology_no_loop(self, simple_graph_2):
        """Test the lift_topology method with loop=False.

        Parameters
        ----------
        simple_graph_2 : torch_geometric.data.Data
            A simple graph fixture with 9 nodes arranged in a line pattern.
        """
        lifted_data = self.lifting_no_loop.lift_topology(simple_graph_2.clone())

        # Verify no self-loops in the incidence matrix
        incidence_matrix = lifted_data["incidence_hyperedges"].to_dense()
        diagonal = torch.diag(incidence_matrix)
        assert not torch.any(diagonal), "Self-loops found when loop=False"

    def test_lift_topology_with_equal_features(self):
        """Test lift_topology with nodes having equal features."""
        # Create a graph where some nodes have identical features
        data = Data(
            x=torch.tensor([[1.0], [1.0], [2.0], [2.0]]),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        )

        lifted_data = self.lifting_k2.lift_topology(data)

        # Verify the shape of the output
        assert lifted_data["incidence_hyperedges"].size() == (4, 4)
        assert lifted_data["num_hyperedges"] == 4
        assert torch.equal(lifted_data["x_0"], data.x)

    @pytest.mark.parametrize("k_value", [1, 2, 3, 4])
    def test_different_k_values(self, k_value, simple_graph_2):
        """Test lift_topology with different k values.

        Parameters
        ----------
        k_value : int
            The number of nearest neighbors to consider.
        simple_graph_2 : torch_geometric.data.Data
            A simple graph fixture with 9 nodes arranged in a line pattern.
        """
        lifting = HypergraphKNNLifting(k_value=k_value, loop=True)
        lifted_data = lifting.lift_topology(simple_graph_2.clone())

        # Verify basic properties
        assert lifted_data["num_hyperedges"] == simple_graph_2.x.size(0)
        incidence_matrix = lifted_data["incidence_hyperedges"].to_dense()

        # Check that each node is connected to at most k nodes
        assert torch.all(incidence_matrix.sum(dim=1) <= k_value), \
            f"Some nodes are connected to more than {k_value} neighbors"

    def test_invalid_inputs(self):
        """Test handling of invalid inputs and edge cases."""
        # Test with no x attribute (this should raise AttributeError)
        data_no_x = Data(edge_index=torch.tensor([[0, 1], [1, 0]]))
        with pytest.raises(AttributeError):
            self.lifting_k2.lift_topology(data_no_x)

        # Test single node case (edge case that should work)
        single_node_data = Data(
            x=torch.tensor([[1.0]], dtype=torch.float),
            edge_index=torch.tensor([[0], [0]])
        )
        lifted_single = self.lifting_k2.lift_topology(single_node_data)
        assert lifted_single["num_hyperedges"] == 1
        assert lifted_single["incidence_hyperedges"].size() == (1, 1)
        assert torch.equal(lifted_single["x_0"], single_node_data.x)

        # Test with identical nodes (edge case that should work)
        identical_nodes_data = Data(
            x=torch.tensor([[1.0], [1.0]], dtype=torch.float),
            edge_index=torch.tensor([[0, 1], [1, 0]])
        )
        lifted_identical = self.lifting_k2.lift_topology(identical_nodes_data)
        assert lifted_identical["num_hyperedges"] == 2
        assert lifted_identical["incidence_hyperedges"].size() == (2, 2)
        assert torch.equal(lifted_identical["x_0"], identical_nodes_data.x)

        # Test with missing edge_index (this should work as KNNGraph will create edges)
        data_no_edges = Data(
            x=torch.tensor([[1.0], [2.0]], dtype=torch.float)
        )
        lifted_no_edges = self.lifting_k2.lift_topology(data_no_edges)
        assert lifted_no_edges["num_hyperedges"] == 2
        assert lifted_no_edges["incidence_hyperedges"].size() == (2, 2)
        assert torch.equal(lifted_no_edges["x_0"], data_no_edges.x)

        # Test with no data (should raise AttributeError)
        with pytest.raises(AttributeError):
            self.lifting_k2.lift_topology(None)

        # Test with empty tensor for x (should work but result in empty outputs)
        empty_data = Data(
            x=torch.tensor([], dtype=torch.float).reshape(0, 1),
            edge_index=torch.tensor([], dtype=torch.long).reshape(2, 0)
        )
        lifted_empty = self.lifting_k2.lift_topology(empty_data)
        assert lifted_empty["num_hyperedges"] == 0
        assert lifted_empty["incidence_hyperedges"].size(0) == 0

    def test_invalid_initialization(self):
        """Test invalid initialization parameters."""
        # Test with non-integer k_value
        with pytest.raises(TypeError, match="k_value must be an integer"):
            HypergraphKNNLifting(k_value=1.5)

        # Test with zero k_value
        with pytest.raises(ValueError, match="k_value must be greater than or equal to 1"):
            HypergraphKNNLifting(k_value=0)

        # Test with negative k_value
        with pytest.raises(ValueError, match="k_value must be greater than or equal to 1"):
            HypergraphKNNLifting(k_value=-1)

        # Test with non-boolean loop
        with pytest.raises(TypeError, match="loop must be a boolean"):
            HypergraphKNNLifting(k_value=1, loop="True")
