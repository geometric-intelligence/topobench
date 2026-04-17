"""This module implements the SimplicialLineLifting class, which lifts graphs to simplicial complexes."""

from itertools import combinations
from typing import Any

import networkx as nx
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobench.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)


class SimplicialLineLifting(Graph2SimplicialLifting):
    r"""Lift graphs to a simplicial complex domain by considering line simplicial complex.

    Line simplicial complex is a clique complex of the line graph. Line graph is a graph, in which
    the vertices are the edges in the initial graph, and two vertices are adjacent if the corresponding
    edges are adjacent in the initial graph.

    Parameters
    ----------
    max_simplices : int
        Max simplices to add for each clique given a rank.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, max_simplices=25, **kwargs):
        super().__init__(**kwargs)
        self.max_simplices = max_simplices

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift topology of a graph to simplicial domain via line simplicial complex construction.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """

        graph = self._generate_graph_from_data(data)
        line_graph = nx.line_graph(graph)

        node_features = {
            node: ((data.x[node[0], :] + data.x[node[1], :]) / 2)
            for node in list(line_graph.nodes)
        }

        cliques = nx.find_cliques(line_graph)

        # we need to rename simplices here since now vertices are named as pairs
        self.rename_vertices_dict = {
            node: i for i, node in enumerate(line_graph.nodes)
        }
        self.rename_vertices_dict_inverse = {
            i: node for node, i in self.rename_vertices_dict.items()
        }
        renamed_line_graph = nx.relabel_nodes(
            line_graph, self.rename_vertices_dict
        )
        renamed_cliques = [
            {self.rename_vertices_dict[vertex] for vertex in simplex}
            for simplex in cliques
        ]
        renamed_node_features = {
            self.rename_vertices_dict[node]: value
            for node, value in node_features.items()
        }

        simplicial_complex = SimplicialComplex()

        simplices: list[set[tuple[Any, ...]]] = [
            set() for _ in range(1, self.complex_dim + 1)
        ]
        for clique in renamed_cliques:
            for i in range(1, self.complex_dim + 1):
                for n_simplices, c in enumerate(combinations(clique, i + 1)):
                    simplices[i - 2].add(tuple(c))
                    if n_simplices >= self.max_simplices:
                        break

        for set_k_simplices in simplices:
            simplicial_complex.add_simplices_from(list(set_k_simplices))

        simplicial_complex.set_simplex_attributes(
            renamed_node_features, name="features"
        )

        return self._get_lifted_topology(
            simplicial_complex, renamed_line_graph
        )
