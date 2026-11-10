"""Unit tests for the GraphTriangleGlobalCC lifting."""

import torch
import torch_geometric

from topobench.transforms.liftings.graph2combinatorial.triangle_global_cc import (
    GraphTriangleGlobalCC,
)


def _toy_graph(num_nodes=6, with_triangles=True):
    """Build a small undirected graph.

    With triangles: 6 nodes, two triangles {0, 1, 2} and {2, 3, 4}, plus a
    pendant edge (4, 5). Without triangles: a path graph.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    with_triangles : bool
        Whether the graph contains triangles.

    Returns
    -------
    torch_geometric.data.Data
        The graph data object.
    """
    if with_triangles:
        edges = [
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 3),
            (2, 4),
            (3, 4),
            (4, 5),
        ]
    else:
        edges = [(i, i + 1) for i in range(num_nodes - 1)]
    edge_index = torch.tensor(
        [[u for u, v in edges] + [v for u, v in edges],
         [v for u, v in edges] + [u for u, v in edges]],
        dtype=torch.long,
    )
    return torch_geometric.data.Data(
        x=torch.randn(num_nodes, 4),
        edge_index=edge_index,
        num_nodes=num_nodes,
    )


class TestGraphTriangleGlobalCC:
    """Test the GraphTriangleGlobalCC lifting."""

    def test_triangles_as_rank2_cells(self):
        """Triangles (maximal 3-cliques) must become rank-2 cells."""
        lifting = GraphTriangleGlobalCC(complex_dim=2)
        lifted = lifting.forward(_toy_graph().clone())

        # 6 nodes, 7 edges, 2 triangles.
        assert lifted.incidence_1.shape == (6, 7)
        assert lifted.incidence_2.shape == (7, 2)
        # Each triangle is incident to exactly its 3 boundary edges.
        assert (
            lifted.incidence_2.to_dense().sum(dim=0) == torch.tensor([3.0, 3.0])
        ).all()

    def test_global_cell(self):
        """The global rank-3 cell must be incident to all triangles.

        Appendix B.4 of arXiv:2605.10091: a single global cell incident to
        all rank-2 triangle cells, giving an all-ones incidence B_{2,3}.
        """
        lifting = GraphTriangleGlobalCC(complex_dim=3, add_global_cell=True)
        lifted = lifting.forward(_toy_graph().clone())

        assert lifted.incidence_3.shape == (2, 1)
        assert (lifted.incidence_3.to_dense() == 1.0).all()

    def test_no_global_cell_without_triangles(self):
        """Triangle-free graphs must not receive a global cell."""
        lifting = GraphTriangleGlobalCC(complex_dim=3, add_global_cell=True)
        lifted = lifting.forward(_toy_graph(with_triangles=False).clone())

        assert lifted.incidence_2.shape[1] == 0
        assert lifted.incidence_3.shape[1] == 0

    def test_edge_only_complex(self):
        """With complex_dim=1 only nodes and edges must be materialized."""
        lifting = GraphTriangleGlobalCC(complex_dim=1)
        lifted = lifting.forward(_toy_graph().clone())

        assert lifted.incidence_1.shape == (6, 7)
        assert "incidence_2" not in lifted

    def test_feature_lifting(self):
        """Rank-0 features must be preserved and higher ranks seeded."""
        data = _toy_graph()
        lifting = GraphTriangleGlobalCC(complex_dim=2)
        lifted = lifting.forward(data.clone())

        assert torch.equal(lifted.x_0, data.x)
        assert lifted.x_1.shape[0] == 7
        assert lifted.x_2.shape[0] == 2
