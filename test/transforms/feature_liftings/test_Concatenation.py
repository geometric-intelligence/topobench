"""Test the message passing module."""

import torch

from topobench.transforms.liftings.graph2simplicial import (
    SimplicialCliqueLifting,
)


class TestConcatention:
    """Test the Concatention feature lifting class."""

    def setup_method(self):
        """Set up the test."""
        # Initialize a lifting class
        self.lifting = SimplicialCliqueLifting(
            feature_lifting="Concatenation", complex_dim=3
        )

    def test_lift_features(self, simple_graph_0, simple_graph_1):
        """Test the lift_features method.

        Parameters
        ----------
        simple_graph_0 : torch_geometric.data.Data
            A simple graph data object.
        simple_graph_1 : torch_geometric.data.Data
            A simple graph data object.
        """

        data = simple_graph_0
        # Test the lift_features method
        lifted_data = self.lifting.forward(data.clone())
        assert lifted_data.x_2.shape == torch.Size([0, 6])

        data = simple_graph_1
        # Test the lift_features method
        lifted_data = self.lifting.forward(data.clone())

        expected_x1 = torch.tensor(
            [
                [1.0, 5.0],
                [1.0, 10.0],
                [1.0, 100.0],
                [1.0, 5000.0],
                [5.0, 10.0],
                [5.0, 100.0],
                [10.0, 50.0],
                [10.0, 100.0],
                [10.0, 500.0],
                [10.0, 5000.0],
                [50.0, 1000.0],
                [500.0, 1000.0],
                [500.0, 5000.0],
            ]
        )

        expected_x2 = torch.tensor(
            [
                [1.0, 5.0, 1.0, 10.0, 5.0, 10.0],
                [1.0, 5.0, 1.0, 100.0, 5.0, 100.0],
                [1.0, 10.0, 1.0, 100.0, 10.0, 100.0],
                [1.0, 10.0, 1.0, 5000.0, 10.0, 5000.0],
                [5.0, 10.0, 5.0, 100.0, 10.0, 100.0],
                [10.0, 500.0, 10.0, 5000.0, 500.0, 5000.0],
            ]
        )

        expected_x3 = torch.tensor(
            [
                [
                    1.0,
                    5.0,
                    1.0,
                    10.0,
                    5.0,
                    10.0,
                    1.0,
                    5.0,
                    1.0,
                    100.0,
                    5.0,
                    100.0,
                    1.0,
                    10.0,
                    1.0,
                    100.0,
                    10.0,
                    100.0,
                    5.0,
                    10.0,
                    5.0,
                    100.0,
                    10.0,
                    100.0,
                ]
            ]
        )

        assert (
            expected_x1 == lifted_data.x_1
        ).all(), "Something is wrong with the lifted features x_1."
        assert (
            expected_x2 == lifted_data.x_2
        ).all(), "Something is wrong with the lifted features x_2."
        assert (
            expected_x3 == lifted_data.x_3
        ).all(), "Something is wrong with the lifted features x_3."
