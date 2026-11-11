"""Test the TriangleDegree transform."""

import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.triangle_degree import (
    TriangleDegree,
)


class TestTriangleDegree:
    """Test the TriangleDegree transform."""

    def setup_method(self):
        """Initialise the TriangleDegree transform."""
        self.transform = TriangleDegree(
            transform_name="TriangleDegree",
            transform_type="data manipulation",
        )

    def test_k4_counts(self):
        """Every node of K4 lies in 3 of its 4 triangles."""
        src, dst = [], []
        for u in range(4):
            for v in range(4):
                if u != v:
                    src.append(u)
                    dst.append(v)
        data = Data(
            x=torch.ones(4, 1),
            edge_index=torch.tensor([src, dst]),
            num_nodes=4,
        )
        out = self.transform(data)
        assert out.tri_degree.shape == (4, 1)
        assert torch.all(out.tri_degree == 3.0)
        # Graph sum is 3x the triangle count
        assert out.tri_degree.sum().item() == 3 * 4

    def test_triangle_free(self):
        """A graph with no edges gets all-zero counts."""
        data = Data(
            x=torch.ones(3, 1),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            num_nodes=3,
        )
        out = self.transform(data)
        assert out.tri_degree.shape == (3, 1)
        assert torch.all(out.tri_degree == 0.0)
