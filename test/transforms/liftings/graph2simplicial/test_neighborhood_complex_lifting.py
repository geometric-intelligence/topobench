"""Test the NeighborhoodComplexLifting class."""
import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2simplicial.neighborhood_complex_lifting import (
    NeighborhoodComplexLifting,
)


class TestNeighborhoodComplexLifting:
    """Test the NeighborhoodComplexLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data = load_manual_graph()

        # Initialize the NeighborhoodComplexLifting class for dim=3
        self.lifting_signed = NeighborhoodComplexLifting(complex_dim=3, signed=True)
        self.lifting_unsigned = NeighborhoodComplexLifting(complex_dim=3, signed=False)
        self.lifting_high = NeighborhoodComplexLifting(complex_dim=7, signed=False)

        # Initialize an empty graph for testing purpouses
        self.empty_graph = nx.empty_graph(10)
        self.empty_data = from_networkx(self.empty_graph)
        self.empty_data["x"] = torch.rand((10, 10))

        # Initialize a start graph for testing
        self.star_graph = nx.star_graph(5)
        self.star_data = from_networkx(self.star_graph)
        self.star_data["x"] = torch.rand((6, 1))

        # Initialize a random graph for testing purpouses
        self.random_graph = nx.fast_gnp_random_graph(5, 0.5)
        self.random_data = from_networkx(self.random_graph)
        self.random_data["x"] = torch.rand((5, 1))


    def has_neighbour(self, simplex_points: list[set]) -> tuple[bool, set[int]]:
        """Verify that the maximal simplices of Data representation of a simplicial complex share a neighbour.

        Parameters
        ----------
        simplex_points : list[set]
            A list of sets representing the maximal simplices of a simplicial complex.

        Returns
        -------
        bool
            True if the simplices share a neighbour, False otherwise.
        """
        for simplex_point_a in simplex_points:
            for simplex_point_b in simplex_points:
                # Same point
                if simplex_point_a == simplex_point_b:
                    continue
                # Search all nodes to check if they are c such that a and b share c as a neighbour
                for node in self.random_graph.nodes:
                    # They share a neighbour
                    if self.random_graph.has_edge(simplex_point_a.item(), node) and self.random_graph.has_edge(simplex_point_b.item(), node):
                        return True
        return False

    def test_lift_topology_random_graph(self):
        """ Verify that the lifting procedure works on a random graph, that is, checks that the simplices generated share a neighbour."""
        lifted_data = self.lifting_high.forward(self.random_data)
        # For each set of simplices
        r = max(int(key.split("_")[-1]) for key in list(lifted_data.keys()) if "x_" in key)
        idx_str = f"x_{r}"

        # Go over each (max_dim)-simplex
        for simplex_points in lifted_data[idx_str]:
            share_neighbour = self.has_neighbour(simplex_points)
            assert share_neighbour, f"The simplex {simplex_points} does not have a common neighbour with all the nodes."

    def test_lift_topology_star_graph(self):
        """ Verify that the lifting procedure works on a small star graph, that is, checks that the simplices generated share a neighbour."""
        lifted_data = self.lifting_high.forward(self.star_data)
        # For each set of simplices
        r = max(int(key.split("_")[-1]) for key in list(lifted_data.keys()) if "x_" in key)
        idx_str = f"x_{r}"

        # Go over each (max_dim)-simplex
        for simplex_points in lifted_data[idx_str]:
            share_neighbour = self.has_neighbour(simplex_points)
            assert share_neighbour, f"The simplex {simplex_points} does not have a common neighbour with all the nodes."



    def test_lift_topology_empty_graph(self):
        """ Test the lift_topology method with an empty graph."""

        lifted_data_signed = self.lifting_signed.forward(self.empty_data)

        assert lifted_data_signed.incidence_1.shape[1] == 0, "Something is wrong with signed incidence_1 (nodes to edges)."

        assert lifted_data_signed.incidence_2.shape[1] == 0, "Something is wrong with signed incidence_2 (edges to triangles)."

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        lifted_data_signed = self.lifting_signed.forward(self.data.clone())
        lifted_data_unsigned = self.lifting_unsigned.forward(self.data.clone())

        expected_incidence_1 = torch.tensor(
            [[-1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
             [ 1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1., -1.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
             [ 0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0., 0.,  0.,  0., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1., 0.,  0.,  0.,  1.,  0.,  0.,  0., -1., -1., -1.,  0.,  0.,  0.],
             [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0., 1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0., -1., -1.,  0.],
             [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0., 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0., -1.],
             [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0., 0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  1.]])

        assert (
            abs(expected_incidence_1) == lifted_data_unsigned.incidence_1.to_dense()
        ).all(), "Something is wrong with unsigned incidence_1 (nodes to edges)."
        assert (
            expected_incidence_1 == lifted_data_signed.incidence_1.to_dense()
        ).all(), "Something is wrong with signed incidence_1 (nodes to edges)."

        expected_incidence_2 = torch.tensor(
            [[ 1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [-1.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0., -1.,  0.,  0., -1.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  0.,  0., -1.,  0., -1.,  0.,  1.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0., -1.,  0.,  0., -1.,  0., -1., -1.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0., 1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0., 0., -1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., 0.,  0., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0., 1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0., -1., -1.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.]])

        assert (
            abs(expected_incidence_2) == lifted_data_unsigned.incidence_2.to_dense()
        ).all(), "Something is wrong with unsigned incidence_2 (edges to triangles)."
        assert (
            expected_incidence_2 == lifted_data_signed.incidence_2.to_dense()
        ).all(), "Something is wrong with signed incidence_2 (edges to triangles)."
