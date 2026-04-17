"""Test the message passing module."""

import torch

from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


class TestSetLifting:
    """Test the SetLifting class."""

    def setup_method(self):
        """Set up the test."""
        # Initialize a lifting class
        self.lifting = SimplicialCliqueLifting(
            feature_lifting="Set", complex_dim=3
        )

    def test_lift_features(self, simple_graph_1):
        """Test the lift_features method.

        Parameters
        ----------
        simple_graph_1 : torch_geometric.data.Data
            A simple graph data object.
        """
        # Test the lift_features method
        data = simple_graph_1
        lifted_data = self.lifting.forward(data.clone())

        expected_x1 = torch.tensor(
            [
                [0, 1],
                [0, 2],
                [0, 4],
                [0, 7],
                [1, 2],
                [1, 4],
                [2, 3],
                [2, 4],
                [2, 5],
                [2, 7],
                [3, 6],
                [5, 6],
                [5, 7],
            ]
        )

        expected_x2 = torch.tensor(
            [[0, 1, 2], [0, 1, 4], [0, 2, 4], [0, 2, 7], [1, 2, 4], [2, 5, 7]]
        )

        expected_x3 = torch.tensor([[0, 1, 2, 4]])

        assert (
            expected_x1 == lifted_data.x_1
        ).all(), "Something is wrong with the lifted features x_1."
        assert (
            expected_x2 == lifted_data.x_2
        ).all(), "Something is wrong with the lifted features x_2."
        assert (
            expected_x3 == lifted_data.x_3
        ).all(), "Something is wrong with the lifted features x_3."
