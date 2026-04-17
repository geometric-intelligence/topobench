"""Test the message passing module."""

import torch

from topobench.data.utils.utils import load_manual_graph_second_structure
from topobench.transforms.liftings.graph2combinatorial.graph_induced_cc import (
    GraphTriangleInducedCC,
)


class TestGraphTriangleInducedCC:
    """Test the GraphTriangleInducedCC class."""

    def setup_method(self):
        """Setup the test."""
        # Load the hypergraph
        self.data = load_manual_graph_second_structure()

        # Initialise the GraphTriangleInducedCC class
        self.lifting = GraphTriangleInducedCC(complex_dim=3)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data = self.lifting.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
            [1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.]
            ]
        )

        assert (
            abs(expected_incidence_1) == lifted_data.incidence_1.to_dense()
        ).all(), "Something is wrong with incidence_1 (nodes to edges)."


        expected_incidence_2 = torch.tensor(
            [
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.],
            [1., 1.],
            [1., 1.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
            [1., 0.],
            [1., 0.],
            [1., 0.],
            [1., 0.],
            [0., 1.],
            [0., 0.],
            [0., 0.],
            [1., 0.]
            ]
        )

        assert (
            abs(expected_incidence_2) == lifted_data.incidence_2.to_dense()
        ).all(), "Something is wrong with incidence_2 (edges to faces)."


        expected_incidence_3 = torch.tensor([[1.],[1.]])

        assert (
            abs(expected_incidence_3) == lifted_data.incidence_3.to_dense()
        ).all(), "Something is wrong with incidence_3 (faces to 3-cells)."
