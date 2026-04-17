"""Test the implicialLineLifting class."""

import torch
import torch_geometric

from topobench.transforms.liftings.graph2simplicial.line_lifting import (
    SimplicialLineLifting,
)


def create_test_graph():
    """Create a simple graph for testing.

    Returns
    -------
    torch_geometric.data
        A simple graph for testing.
    """
    num_nodes = 5
    x = [1] * num_nodes
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 2], [1, 2, 3, 4, 0, 4, 3]], dtype=torch.long
    )  # [[0, 0, 1, 1, 2, 2, 3], [1, 4, 2, 3, 3, 4, 4]]
    y = [0, 0, 1, 1, 0]

    return torch_geometric.data.Data(
        x=torch.tensor(x).float().reshape(-1, 1),
        edge_index=edge_index,  # torch.Tensor(edge_index, dtype=torch.long),
        num_nodes=num_nodes,
        y=torch.tensor(y),
    )


class TestSimplicialLineLifting:
    """Test the SimplicialLineLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data = create_test_graph()  # load_manual_graph()

        # Initialise the SimplicialCliqueLifting class
        self.lifting_signed = SimplicialLineLifting(signed=True)
        self.lifting_unsigned = SimplicialLineLifting(signed=False)
        self.lifting_unsigned_max_simplices = SimplicialLineLifting(
            signed=False, max_simplices=1
        )

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )

        print(lifted_data_signed.incidence_1.to_dense())

        assert (
            abs(expected_incidence_1) == lifted_data_unsigned.incidence_1.to_dense()
        ).all(), "Something is wrong with unsigned incidence_1 (nodes to edges)."
        assert (
            expected_incidence_1 == lifted_data_signed.incidence_1.to_dense()
        ).all(), "Something is wrong with signed incidence_1 (nodes to edges)."

        expected_incidence_2 = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        assert (
            abs(expected_incidence_2) == lifted_data_unsigned.incidence_2.to_dense()
        ).all(), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert (
            expected_incidence_2 == lifted_data_signed.incidence_2.to_dense()
        ).all(), "Something is wrong with signed incidence_2 (edges to triangles)."

        lifted_data = self.lifting_unsigned_max_simplices.forward(self.data.clone())

        expected_incidence = torch.tensor(
            [[1., 1., 1., 1., 0., 0., 0., 0., 0.],
             [1., 0., 0., 0., 1., 1., 0., 0., 0.],
             [0., 1., 0., 0., 1., 0., 1., 0., 0.],
             [0., 0., 1., 0., 0., 0., 0., 1., 1.],
             [0., 0., 0., 0., 0., 1., 0., 1., 0.],
             [0., 0., 0., 1., 0., 0., 1., 0., 1.]]
        )

        assert (
            abs(expected_incidence) == lifted_data.incidence_1.to_dense()
        ).all(), "Something is wrong with incidence (triangles to tetrahedrons)."
