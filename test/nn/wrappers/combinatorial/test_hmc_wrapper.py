"""Unit tests for the HMC wrapper."""

import torch
import torch_geometric

from topobench.nn.backbones.combinatorial.hmc import HMC
from topobench.nn.wrappers.combinatorial import HMCWrapper


def random_sparse(rows, cols, density=0.5, seed=0):
    """Create a random sparse binary matrix.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    density : float, optional
        Density of nonzero entries.
    seed : int, optional
        Random seed.

    Returns
    -------
    torch.sparse.Tensor
        Random sparse matrix in COO format.
    """
    generator = torch.Generator().manual_seed(seed)
    dense = (torch.rand(rows, cols, generator=generator) < density).float()
    return dense.to_sparse_coo()


class TestHMCWrapper:
    """Unit tests for the HMCWrapper."""

    def test_forward(self):
        """Test that the wrapper maps batch fields to the HMC backbone."""
        torch.manual_seed(0)
        n_0, n_1, n_2 = 6, 9, 4
        channels = 8
        model = HMC(channels, channels, n_layers=1)
        batch = torch_geometric.data.Data(
            x_0=torch.randn(n_0, channels),
            x_1=torch.randn(n_1, channels),
            x_2=torch.randn(n_2, channels),
            adjacency_0=random_sparse(n_0, n_0, seed=1),
            adjacency_1=random_sparse(n_1, n_1, seed=2),
            coadjacency_2=random_sparse(n_2, n_2, seed=3),
            incidence_1=random_sparse(n_0, n_1, seed=4),
            incidence_2=random_sparse(n_1, n_2, seed=5),
            y=torch.zeros(1, dtype=torch.long),
            batch_0=torch.zeros(n_0, dtype=torch.long),
        )
        wrapper = HMCWrapper(
            model, out_channels=channels, num_cell_dimensions=3
        )
        model_out = wrapper(batch)
        assert model_out["x_0"].shape == (n_0, channels)
        assert model_out["x_1"].shape == (n_1, channels)
        assert model_out["x_2"].shape == (n_2, channels)
        assert torch.equal(model_out["labels"], batch.y)
        assert torch.equal(model_out["batch_0"], batch.batch_0)
