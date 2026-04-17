"""Test the LatentCliqueCoverLifting class."""

import random

import networkx as nx
import torch
from torch_geometric.data import Data

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2simplicial.latentclique_lifting import (
    LatentCliqueLifting,
)


def create_clique_graph(num_nodes=8):
    """
    Create a torch_geometric.data.Data object representing a single clique graph with node features being all ones.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the clique.

    Returns
    -------
    torch_geometric.data.Data
        Torch Geometric data object representing the graph.
    """
    # Create edge index for a complete graph (clique)
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_index.append([i, j])
            edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create node features (all ones)
    x = torch.ones((num_nodes, 1), dtype=torch.float)

    # Create the Data object
    return Data(x=x, edge_index=edge_index)


class TestLatentCliqueCoverLifting:
    """Test the LatentCliqueCoverLifting class."""

    def setup_method(self):
        """Setup the test."""
        # Load the graph
        self.data_test_one = create_clique_graph()
        self.data_test_two = load_manual_graph()
        # Initialise the SimplicialCliqueLifting class
        self.lifting_edge_prob_one = LatentCliqueLifting(edge_prob_mean=1)
        self.lifting_edge_prob_any = LatentCliqueLifting(edge_prob=random.uniform(0, 1))
        self.lifting = LatentCliqueLifting(init="single", do_gibbs=True)

    def test_lift_topology(self):
        """Test the lift_topology method."""

        # Test the lift_topology method
        edge_prob_one_adj = self.lifting_edge_prob_one.forward(
            self.data_test_one.clone()
        ).adjacency_0
        edge_prob_any_adj = self.lifting_edge_prob_any.forward(
            self.data_test_two.clone()
        ).adjacency_0
        edge_prob_single_adj = self.lifting.forward(self.data_test_one.clone()).adjacency_0

        ### TEST #1 ###
        # if edge_prob == 1 and the input graph has a single clique,
        # then the 1-skeleton of the inferred latent SC must have a single clique
        # (or, equivalently, the SC has a simplex in its facets set if complex_dim = |maximal_clique|-1)

        # Convert adjacency matrix to NetworkX graph
        G_from_latent_complex = nx.from_numpy_matrix(
            edge_prob_one_adj.to_dense().numpy()
        )
        G_input = nx.Graph()
        G_input.add_edges_from(self.data_test_one.edge_index.t().tolist())

        # Find all cliques in the graphs
        cliques_latent = list(nx.find_cliques(G_from_latent_complex))

        # Number of cliques
        num_cliques_latent = len(cliques_latent)

        assert (
            num_cliques_latent == num_cliques_latent
        ), "the 1-skeleton of the inferred latent SC does not have a single clique"

        ### TEST #2 ###
        # if edge_prob in (0,1] the set of 0-simplices (edges) of the inferred
        # latent SC must be a superset of the set of edges of the input graph
        # (or, equivalently, there is no subset of the 1-skeleton of the SC isomorphic to the input graph)

        # Convert adjacency matrix to NetworkX graph
        G_from_latent_complex = nx.from_numpy_matrix(
            edge_prob_any_adj.to_dense().numpy()
        )
        G_input = nx.Graph()
        G_input.add_edges_from(self.data_test_two.edge_index.t().tolist())

        # Edges are Undirected
        latent_edge_set = {tuple(sorted(edge)) for edge in G_from_latent_complex.edges}
        input_edge_set = {tuple(sorted(edge)) for edge in G_input.edges}

        assert input_edge_set.issubset(
            latent_edge_set
        ), "the set of 0-simplices of the inferred latent SC is not contained the set of edges of the input graph"
