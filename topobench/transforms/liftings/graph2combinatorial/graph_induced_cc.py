"""GraphInducedCC lifting of graphs to combinatorial complexes."""

from collections import defaultdict
from itertools import combinations

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import CombinatorialComplex

from topobench.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
)


class GraphTriangleInducedCC(Graph2CombinatorialLifting):
    r"""Lift graph to combinatorial complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data | dict:
        r"""Lift the topology of a graph to a combinatorial complex.

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
        assert not graph.is_directed(), (
            "Graph supposed to be undirected for this lifting"
        )
        cliques = nx.find_cliques(graph)

        cliques = [sorted(clique) for clique in cliques if len(clique) == 3]

        one_element_paths, two_elements_paths, at_least_one_paths = (
            find_overlapping_paths(cliques)
        )

        combinatorial_complex = CombinatorialComplex(graph)
        two_cells, three_cells = [], []

        # Case when at exactly one element is shared
        for idx_list in one_element_paths:
            temp = []
            for idx in idx_list:
                temp.extend(cliques[idx])
            temp = set(temp)
            two_cells.append(temp)

        # Case when at exactly two elements are shared
        for idx_list in two_elements_paths:
            temp = []
            for idx in idx_list:
                temp.extend(cliques[idx])
            temp = set(temp)
            two_cells.append(temp)

        # Case when at least one element is shared
        for idx_list in at_least_one_paths:
            temp = []
            for idx in idx_list:
                temp.extend(cliques[idx])
            temp = set(temp)
            three_cells.append(temp)

        for cells in two_cells:
            combinatorial_complex.add_cell(cells, 2)

        for cells in three_cells:
            combinatorial_complex.add_cell(cells, 3)

        lifted_topology = self._get_lifted_topology(
            combinatorial_complex, graph
        )

        # Feature liftings
        lifted_topology["x_0"] = data.x

        return lifted_topology

    def _sorted_hyperedge_indices(self, hyperedges):
        """
        Create a list of pairs with the starts and lengths of hyperedges in ascending order of hyperedge size.

        Parameters
        ----------
        hyperedges : torch.tensor
            A tensor with two rows: the first one for hyperedge indices, the second one for node indices.

        Returns
        -------
        list
            A list of pairs (start, length) sorted according to length (ascending).
        """
        # Identify where the changes occur
        changes = torch.cat(
            [torch.tensor([True]), hyperedges[1:] != hyperedges[:-1]]
        )
        change_indices = torch.where(changes)[0]

        # Calculate the size of each hyperedge
        lengths = change_indices[1:] - change_indices[:-1]
        lengths = torch.cat(
            [lengths, torch.tensor([len(hyperedges) - change_indices[-1]])]
        )

        # Sort the list according to the lengths entry
        indices = list(
            zip(change_indices.tolist(), lengths.tolist(), strict=False)
        )
        indices.sort(key=lambda x: x[1])

        return indices


def build_paths(overlap_pairs):
    """Find overlapint sequesnces.

    Parameters
    ----------
    overlap_pairs : list
        List of pairs of overlapping triangles.

    Returns
    -------
    list
        List of sequences of overlapping triangles.
    """
    parent = {}

    def find(x):
        """Find parent of x.

        Parameters
        ----------
        x : int
            Find parent.

        Returns
        -------
        list
            Union of two lists.
        """
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        """Union.

        Parameters
        ----------
        x : list
            List.
        y : list
            List.

        Return
        ------
        list
            Union of two lists.
        """
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for a, b in overlap_pairs:
        if a not in parent:
            parent[a] = a
        if b not in parent:
            parent[b] = b
        union(a, b)

    groups = defaultdict(set)
    for node in parent:
        groups[find(node)].add(node)

    return [tuple(sorted(group)) for group in groups.values()]


def find_overlapping_paths(lists):
    """Find ovelaping triabgles and their seuqences.

    Parameters
    ----------
    lists : list
        List of triangles.

    Returns
    -------
    list
        List of sequences of overlapping triangles.
    """
    one_element_overlap = []
    two_elements_overlap = []
    at_least_one_overlap = []

    sets = [set(lst) for lst in lists]

    for i, j in combinations(range(len(lists)), 2):
        intersection = sets[i] & sets[j]
        overlap_size = len(intersection)

        if overlap_size == 1:
            one_element_overlap.append((i, j))
        if overlap_size == 2:
            two_elements_overlap.append((i, j))
        if overlap_size >= 1:
            at_least_one_overlap.append((i, j))

    return (
        build_paths(one_element_overlap),
        build_paths(two_elements_overlap),
        build_paths(at_least_one_overlap),
    )


# edges = [
#     (0,1),
#     (1,2),
#     (2,3),
#     (3,0),
#     (3,4),
#     (3,6),
#     (3,9),
#     (4,5),
#     (4,6),
#     (4,7),
#     (5,0),
#     (5,7),
#     (5,10),
#     (5,11),
#     (6,9),
#     (7,8),
#     (8,6),
#     (10,11),
#     ]
# nodes = [0,1,2,3,4,5,6,7,8,9,10,11]
# graph = nx.Graph()
# graph.add_nodes_from(nodes)
# graph.add_edges_from(edges)
# graph = graph.to_undirected()

# lists = [[3, 6, 9], [3, 4, 6], [5, 10, 11], [4, 5, 7]]

# one_element_paths, two_elements_paths, at_least_one_paths = find_overlapping_paths(lists)

# print("Paths with exactly one element overlap:", one_element_paths)
# print("Paths with exactly two elements overlap:", two_elements_paths)
# print("Paths with at least one element overlap:", at_least_one_paths)
