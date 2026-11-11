"""Permutation-invariant cell lifting: all 3-cliques as 2-cells."""

import networkx as nx
import torch_geometric
from toponetx.classes import CellComplex

from topobench.transforms.liftings.graph2cell.base import (
    Graph2CellLifting,
)


class CellCliqueLifting(Graph2CellLifting):
    r"""Lift graphs to cell complexes with all triangles as 2-cells.

    Attaches every 3-clique of the graph as a 2-cell. The set of
    triangles of a graph is canonical, so the lifted complex — and
    every connectivity matrix derived from it — is invariant under
    node relabeling. Cycle-basis liftings lack this property: the
    basis returned by ``nx.cycle_basis`` depends on the traversal
    order, so isomorphic graphs presented with different node
    orderings receive differently wired complexes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.complex_dim = 2

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Attach all triangles of the graph as 2-cells.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        G = self._generate_graph_from_data(data)
        cell_complex = CellComplex(G)
        triangles = []
        for clique in nx.enumerate_all_cliques(G):
            if len(clique) == 3:
                triangles.append(sorted(clique))
            elif len(clique) > 3:
                break
        triangles.sort()
        if len(triangles) != 0:
            cell_complex.add_cells_from(triangles, rank=self.complex_dim)
        return self._get_lifted_topology(cell_complex, G)
