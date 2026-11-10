"""Triangle + global-cell lifting of graphs to combinatorial complexes."""

import networkx as nx
import torch_geometric
from toponetx.classes import CombinatorialComplex

from topobench.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
)


class GraphTriangleGlobalCC(Graph2CombinatorialLifting):
    r"""Lift graphs to the combinatorial complexes used by TopoU-Net.

    Constructs the rank structure of Section 4 of "TopoU-Net: A U-Net
    Architecture for Topological Domains" (arXiv:2605.10091): nodes as
    rank-0 cells, edges as rank-1 cells, triangles (maximal 3-cliques) as
    rank-2 cells and, optionally, a single global rank-3 cell incident to
    all rank-2 triangle cells (used by the heterophilic rank path
    :math:`0 \to 1 \to 2 \to 3 \to 2 \to 1 \to 0`, Appendix B.4).

    The ranks that are materialized are controlled by ``complex_dim``:
    triangles are added for ``complex_dim >= 2`` and the global cell for
    ``complex_dim >= 3`` when ``add_global_cell`` is True. The global cell
    is realized as the set of all nodes, so every triangle is contained in
    it and the incidence matrix :math:`B_{2,3}` is a column of ones.

    Parameters
    ----------
    add_global_cell : bool, optional
        Whether to add the single global rank-3 cell (default: False).
    **kwargs : optional
        Additional arguments for the class (e.g. ``complex_dim``).
    """

    def __init__(self, add_global_cell: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.add_global_cell = add_global_cell

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

        combinatorial_complex = CombinatorialComplex(graph)

        triangles = []
        if self.complex_dim >= 2:
            # Triangles as maximal 3-cliques (Section 4 of the paper).
            triangles = [
                sorted(clique)
                for clique in nx.find_cliques(graph)
                if len(clique) == 3
            ]
            for triangle in triangles:
                combinatorial_complex.add_cell(triangle, 2)

        if self.complex_dim >= 3 and self.add_global_cell:
            nodes = sorted(graph.nodes)
            # The global cell is defined as incident to the rank-2 triangle
            # cells (Appendix B.4), so it is only added when triangles exist;
            # this also keeps the active ranks consecutive, as assumed by the
            # connectivity extraction. The size guard avoids the degenerate
            # case where the node set coincides with a lower-rank cell.
            if triangles and len(nodes) > 3:
                combinatorial_complex.add_cell(nodes, 3)

        lifted_topology = self._get_lifted_topology(
            combinatorial_complex, graph
        )
        lifted_topology["x_0"] = data.x

        return lifted_topology
