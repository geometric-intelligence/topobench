"""Exact per-node triangle-membership counts."""

import networkx as nx
import torch
import torch_geometric


class TriangleDegree(torch_geometric.transforms.BaseTransform):
    r"""Attach exact per-node triangle counts as ``data.tri_degree``.

    For each node :math:`v`, ``tri_degree[v]`` is the number of
    triangles of the raw graph containing :math:`v`, computed with
    networkx at preprocessing time (cached with the dataset). The
    graph-level sum equals :math:`3\cdot\#\text{triangles}`, so a
    sum-pooled linear readout can recover the exact count.

    Motivation: cycle liftings attach a cycle *basis* as 2-cells
    (``nx.cycle_basis``), and a basis undercounts triangles on dense
    graphs (a :math:`K_4` has 4 triangles and cycle-space dimension 3).
    Counting signals derived from ``incidence_2`` inherit that bias;
    this transform is lifting-independent and exact.

    Parameters
    ----------
    **kwargs : dict
        Transform configuration (``transform_name``, ``transform_type``).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "triangle_degree"
        self.parameters = kwargs

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        """Compute per-node triangle counts.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph with ``edge_index``.

        Returns
        -------
        torch_geometric.data.Data
            Same data object with ``tri_degree`` of shape (N, 1).
        """
        n = data.num_nodes
        g = nx.Graph()
        g.add_nodes_from(range(n))
        if data.edge_index is not None and data.edge_index.numel() > 0:
            g.add_edges_from(data.edge_index.t().tolist())
        tri = nx.triangles(g)
        data.tri_degree = torch.tensor(
            [tri[i] for i in range(n)], dtype=torch.float32
        ).view(-1, 1)
        return data
