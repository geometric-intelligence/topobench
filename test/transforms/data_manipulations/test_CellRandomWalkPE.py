"""Unit tests for the CellRandomWalkPE transform."""

import torch
import torch_geometric

from topobench.nn.backbones.cell.cellular_transformer import random_walk_pe
from topobench.transforms.data_manipulations import CellRandomWalkPE


class TestCellRandomWalkPE:
    """Test the CellRandomWalkPE transform."""

    def setup_method(self):
        """Set up the test."""
        self.transform = CellRandomWalkPE(pe_steps=4)

    def test_repr(self):
        """Test the string representation."""
        assert "CellRandomWalkPE" in repr(self.transform)
        assert "pe_steps=4" in repr(self.transform)

    def test_attaches_pe_on_cell_complex(self):
        """Test that rwpe_k are attached with correct shapes and values."""
        torch.manual_seed(0)
        idx = torch.tensor([[0, 1], [1, 0]])
        adjacency_0 = torch.sparse_coo_tensor(idx, torch.ones(2), (2, 2))
        coadjacency_1 = (torch.rand(3, 3) < 0.5).float().to_sparse_coo()
        coadjacency_2 = torch.zeros(0, 0).to_sparse_coo()
        data = torch_geometric.data.Data(
            x_0=torch.randn(2, 4),
            adjacency_0=adjacency_0,
            coadjacency_1=coadjacency_1,
            coadjacency_2=coadjacency_2,
        )
        out = self.transform(data)
        assert out.rwpe_0.shape == (2, 4)
        assert out.rwpe_1.shape == (3, 4)
        assert out.rwpe_2.shape == (0, 4)
        # Matches the backbone's own computation exactly.
        assert torch.equal(out.rwpe_0, random_walk_pe(adjacency_0, 4))
        # 2-node path graph: diag(P) = 0, diag(P^2) = 1, ...
        assert torch.allclose(
            out.rwpe_0,
            torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]),
        )

    def test_noop_without_cell_matrices(self):
        """Test that plain graph data passes through unchanged."""
        data = torch_geometric.data.Data(
            x=torch.randn(3, 4),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
        )
        out = self.transform(data)
        assert not hasattr(out, "rwpe_0")
        assert not hasattr(out, "rwpe_1")
