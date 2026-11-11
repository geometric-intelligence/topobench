"""Test the all-3-cliques cell lifting."""

import torch
from torch_geometric.data import Data

from topobench.transforms.liftings.graph2cell.clique_cell_lifting import (
    CellCliqueLifting,
)


def _complete_graph(n):
    """Build K_n as a torch_geometric Data object.

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    torch_geometric.data.Data
        The complete graph with unit node features.
    """
    src, dst = [], []
    for u in range(n):
        for v in range(n):
            if u != v:
                src.append(u)
                dst.append(v)
    return Data(
        x=torch.ones(n, 2),
        edge_index=torch.tensor([src, dst]),
        num_nodes=n,
        y=torch.zeros(n, dtype=torch.long),
    )


class TestCellCliqueLifting:
    """Test the CellCliqueLifting class."""

    def setup_method(self):
        """Initialise the CellCliqueLifting class."""
        self.lifting = CellCliqueLifting()

    def test_k4_attaches_all_triangles(self):
        """K4 has 4 triangles; a cycle basis would only find 3."""
        lifted = self.lifting.forward(_complete_graph(4))
        assert lifted.incidence_2.shape[1] == 4
        # Each triangle has 3 boundary edges
        inc2 = torch.abs(lifted.incidence_2.coalesce())
        col_sums = torch.sparse.sum(inc2, dim=0).to_dense()
        assert torch.all(col_sums == 3)

    def test_triangle_free_graph(self):
        """A path graph lifts with zero 2-cells."""
        data = Data(
            x=torch.ones(4, 2),
            edge_index=torch.tensor([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]]),
            num_nodes=4,
            y=torch.zeros(4, dtype=torch.long),
        )
        lifted = self.lifting.forward(data)
        assert lifted.incidence_2.shape[1] == 0

    def test_permutation_invariance(self):
        """Relabeling the nodes yields an isomorphic complex.

        The count of 2-cells and the multiset of per-node counts
        t_v = |B1||B2|1 must match under any node permutation.
        """
        # Two triangles sharing node 2, plus a pendant node 5.
        edge_list = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5)]

        def build(perm):
            src, dst = [], []
            for u, v in edge_list:
                pu, pv = perm[u], perm[v]
                src += [pu, pv]
                dst += [pv, pu]
            return Data(
                x=torch.ones(6, 2),
                edge_index=torch.tensor([src, dst]),
                num_nodes=6,
                y=torch.zeros(6, dtype=torch.long),
            )

        def t_v(lifted):
            inc1 = torch.abs(lifted.incidence_1.coalesce())
            inc2 = torch.abs(lifted.incidence_2.coalesce())
            ones = torch.ones(inc2.shape[1], 1)
            return torch.sparse.mm(inc1, torch.sparse.mm(inc2, ones))

        identity = list(range(6))
        shuffled = [3, 5, 0, 4, 1, 2]
        lift_a = self.lifting.forward(build(identity))
        lift_b = self.lifting.forward(build(shuffled))

        assert lift_a.incidence_2.shape[1] == 2
        assert lift_b.incidence_2.shape[1] == 2
        counts_a = t_v(lift_a).squeeze(1)
        counts_b = t_v(lift_b).squeeze(1)
        # Node i in graph A corresponds to node shuffled[i] in graph B
        for i in range(6):
            assert counts_a[i] == counts_b[shuffled[i]]
