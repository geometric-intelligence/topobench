"""This module implements the HypergraphKNNLifting class."""

import torch
import torch_geometric
from torch_cluster import knn_graph

from topobench.transforms.liftings.graph2hypergraph import (
    Graph2HypergraphLifting,
)


class HypergraphKNNLifting(Graph2HypergraphLifting):
    r"""Lift graphs to hypergraph domain by considering k-nearest neighbors.

    Parameters
    ----------
    k_value : int, optional
        The number of nearest neighbors to consider. Must be positive. Default is 1.
    loop : bool, optional
        If True the hyperedges will contain the node they were created from.
    **kwargs : optional
        Additional arguments for the class.

    Raises
    ------
    ValueError
        If k_value is less than 1.
    TypeError
        If k_value is not an integer or if loop is not a boolean.
    """

    def __init__(self, k_value=1, loop=True, **kwargs):
        super().__init__(**kwargs)

        # Validate k_value
        if not isinstance(k_value, int):
            raise TypeError("k_value must be an integer")
        if k_value < 1:
            raise ValueError("k_value must be greater than or equal to 1")

        # Validate loop
        if not isinstance(loop, bool):
            raise TypeError("loop must be a boolean")

        self.k = k_value
        self.loop = loop

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift a graph to hypergraph by considering k-nearest neighbors.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        num_nodes = data.x.shape[0]
        num_hyperedges = num_nodes
        incidence_1 = torch.zeros(num_nodes, num_nodes)
        edge_index = knn_graph(data.x, self.k, loop=self.loop)
        incidence_1[edge_index[1], edge_index[0]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
