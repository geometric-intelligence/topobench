"""Test the khop lifting."""

import torch

from topobench.transforms.liftings.graph2hypergraph import (
    HypergraphKHopLifting,
)


class TestHypergraphKHopLifting:
    """Test the HypergraphKHopLifting class."""

    def setup_method(self):
        """ Setup the test."""
        # Initialise the HypergraphKHopLifting class
        self.lifting_k1 = HypergraphKHopLifting(k_value=1)
        self.lifting_k2 = HypergraphKHopLifting(k_value=2)
        self.lifting_edge_attr = HypergraphKHopLifting(k_value=1, preserve_edge_attr=True)

    def test_lift_topology(self, simple_graph_2):
        """ Test the lift_topology method.

        Parameters
        ----------
        simple_graph_2 : Data
            A simple graph used for testing.
        """
        # Test the lift_topology method
        self.data = simple_graph_2
        lifted_data_k1 = self.lifting_k1.forward(self.data.clone())

        expected_n_hyperedges = 9

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        assert (
            expected_incidence_1
            == lifted_data_k1.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges (k=1)."
        assert (
            expected_n_hyperedges == lifted_data_k1.num_hyperedges
        ), "Something is wrong with the number of hyperedges (k=1)."

        lifted_data_k2 = self.lifting_k2.forward(self.data.clone())

        expected_n_hyperedges = 9

        expected_incidence_1 = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ]
        )

        assert (
            expected_incidence_1
            == lifted_data_k2.incidence_hyperedges.to_dense()
        ).all(), "Something is wrong with incidence_hyperedges (k=2)."
        assert (
            expected_n_hyperedges == lifted_data_k2.num_hyperedges
        ), "Something is wrong with the number of hyperedges (k=2)."

        self.data_edge_attr = simple_graph_2
        edge_attributes = torch.rand((self.data_edge_attr.edge_index.shape[1], 2))
        self.data_edge_attr.edge_attr = edge_attributes
        lifted_data_edge_attr = self.lifting_edge_attr.forward(self.data_edge_attr.clone())
        assert lifted_data_edge_attr.edge_attr is not None, "Edge attributes are not preserved."
        assert torch.all(edge_attributes == lifted_data_edge_attr.edge_attr), "Edge attributes are not preserved correctly."
