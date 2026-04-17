"""This module implements the HypergraphFormanRicciCurvatureLifting class."""

import math

import networkx as nx
import numpy as np
import torch
import torch_geometric

from topobench.transforms.liftings.graph2hypergraph.base import (
    Graph2HypergraphLifting,
)


class HypergraphFormanRicciCurvatureLifting(Graph2HypergraphLifting):
    """Lift graphs to hypergraph domain using Forman-Ricci curvature based backbone estimation.

    This lifting identifies a network's structure-preserving, coarse geometry, i.e. its backbones, which lend themselves specifically to model information flows across wide areas of the network via hyperedges. To identify this coarse geometry we apply Forman-Ricci curvature to the original graph. Forman-Ricci curvature defines an edge-based network characteristic that reveals properties of a graph's community structure. In particular high absolute Forman-Ricci curvature exhibits a network's backbone, a coarse, structure preserving graph geometry that forms connections between major communities, most suitable to form hyperedges. In addition, Forman-Ricci curvature was found to be especially useful for network analysis since its intuitive notion allows for efficient computation that scales to large network sizes.

    Parameters
    ----------
    network_type : str, optional
        Network type may be weighted or unweighted. Default is "weighted".
    threshold_type : str, optional
        Type of threshold (either based on quantile or absolute). Default is "quantile".
    threshold_direction : str, optional
        Direction whether to prune values above or below threshold. Default is "above".
    threshold : float, optional
        Absolute value or quantile to estimate cutoff threshold from Forman-Ricci curvature distribution to prune network and reveal backbone. Default is 0.1.
    **kwargs : dict, optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        network_type: str = "weighted",
        threshold_type: str = "quantile",
        threshold_direction: str = "above",
        threshold: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.network_type = network_type
        self.threshold_type = threshold_type
        self.threshold_direction = threshold_direction
        self.threshold = threshold

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a graph to hypergraph domain using Forman-Ricci curvature based backbone estimation.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """

        edge_list = data.edge_index.t().numpy()

        # for unweighted graphs or higher-dimensional edge or node features revert to unweighted network structure
        if (
            data.edge_attr is None
            or self.network_type == "unweighted"
            or data.edge_attr.shape[1] > 1
        ):
            edge_attr = np.ones(shape=(len(edge_list), 1))
        elif isinstance(data.edge_attr, torch.Tensor):
            edge_attr = data.edge_attr.numpy()
        else:
            edge_attr = data.edge_attr

        if (
            data.x is None
            or self.network_type == "unweighted"
            or data.x.shape[1] > 1
        ):
            node_attr = np.ones(shape=(data.num_nodes, 1))
        elif isinstance(data.x, torch.Tensor):
            node_attr = data.x.numpy()
        else:
            node_attr = data.x

        # create undirected networkx graph from pyg data
        G = nx.Graph()
        for v in range(len(node_attr)):
            G.add_node(v)
            G.nodes[v]["w"] = node_attr[v][0]

        for e in range(len(edge_list)):
            v1 = edge_list[e][0]
            v2 = edge_list[e][1]
            G.add_edge(v1, v2, w=edge_attr[e][0])
            G.add_edge(v1, v2, w=edge_attr[e][0])

        # estimate Forman-Ricci curvature as described in:
        # M. Weber, J. Jost, E. Saucan (2018). Detecting the Coarse Geometry of Networks. NeurIPS
        for v1, v2 in G.edges():
            v1_neighbors = set(G.neighbors(v1))
            v1_neighbors.remove(v2)
            v2_neighbors = set(G.neighbors(v2))
            v2_neighbors.remove(v1)

            w_e = G[v1][v2]["w"]
            w_v1 = G.nodes[v1]["w"]
            w_v2 = G.nodes[v2]["w"]
            ev1_sum = sum(
                [w_v1 / math.sqrt(w_e * G[v1][v]["w"]) for v in v1_neighbors]
            )
            ev2_sum = sum(
                [w_v2 / math.sqrt(w_e * G[v2][v]["w"]) for v in v2_neighbors]
            )

            G[v1][v2]["w_frc"] = w_e * (
                w_v1 / w_e + w_v2 / w_e - (ev1_sum + ev2_sum)
            )

        if self.threshold_type == "quantile":
            # estimate threshold from Forman-Ricci curvature distribution to prune network and reveal backbone(s), i.e. hyperedges
            w_frc = list(nx.get_edge_attributes(G, "w_frc").values())
            th_frc = np.quantile(w_frc, self.threshold)
        elif self.threshold_type == "absolute":
            # define absolute Forman-Ricci curvature threshold to prune network and reveal backbone(s), i.e. hyperedges
            th_frc = self.threshold
        else:
            raise NotImplementedError(
                f"threshold type {self.threshold_type} not implemented"
            )

        if self.threshold_direction == "above":
            # select edge with curvature above threshold for pruning
            def select_edge_for_pruning(x):
                return x > th_frc

        elif self.threshold_direction == "below":
            # select edge with curvature below threshold for pruning
            def select_edge_for_pruning(x):
                return x < th_frc

        else:
            raise NotImplementedError(
                f"threshold threshold_direction {self.threshold_direction} not implemented"
            )

        # select all edges that meet criterium for pruning
        edges_to_remove = []
        for v1, v2 in G.edges():
            if select_edge_for_pruning(G[v1][v2]["w_frc"]):
                edges_to_remove.append((v1, v2))

        # remove selected edges from curvature graph
        G.remove_edges_from(edges_to_remove)

        # find connected components in curvature graph (hyperedges)
        hyperedges = [
            c
            for c in sorted(nx.connected_components(G), key=len, reverse=True)
        ]
        shape = (data.num_nodes, len(hyperedges))
        incidence_matrix = np.zeros(shape=shape)

        if len(hyperedges) > 0:
            for i, nodes in enumerate(hyperedges):
                incidence_matrix[list(nodes), i] = 1

        assert len(hyperedges) > 0

        incidences = torch.Tensor(
            incidence_matrix.astype("f4")
        ).to_sparse_coo()

        coo_indices = torch.stack(
            (incidences.indices()[0], incidences.indices()[1])
        )

        coo_values = incidences.values()

        incidence_matrix = torch.sparse_coo_tensor(coo_indices, coo_values)

        return {
            "incidence_hyperedges": incidence_matrix,
            "num_hyperedges": incidence_matrix.size(1),
            "x_0": data.x,
        }
