"""Unit tests for the TopoU-Net wrapper."""

import torch
import torch_geometric

from topobench.nn.backbones.combinatorial.topounet import TopoUNet
from topobench.nn.wrappers.combinatorial import TopoUNetWrapper


def _toy_batch(d_feat=8):
    """Build a single-complex batch with ranks 0-2.

    Parameters
    ----------
    d_feat : int
        Feature dimension of the rank-0 cochain.

    Returns
    -------
    torch_geometric.data.Data
        Batch with rank-0 features, incidence matrices and labels.
    """
    incidence_1 = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ).to_sparse()
    incidence_2 = torch.tensor([[1.0], [1.0], [1.0], [0.0]]).to_sparse()
    return torch_geometric.data.Data(
        x_0=torch.randn(4, d_feat),
        incidence_1=incidence_1,
        incidence_2=incidence_2,
        y=torch.zeros(4, dtype=torch.long),
        batch_0=torch.zeros(4, dtype=torch.long),
    )


def test_topounet_wrapper():
    """The wrapper must expose decoder states of all path ranks."""
    d_feat = 8
    batch = _toy_batch(d_feat)
    backbone = TopoUNet(d_feat, [0, 1, 2])
    wrapper = TopoUNetWrapper(
        backbone,
        out_channels=d_feat,
        num_cell_dimensions=1,
        residual_connections=False,
    )
    _ = wrapper.__repr__()

    model_out = wrapper(batch)
    assert model_out["x_0"].shape == (4, d_feat)
    assert model_out["x_1"].shape == (4, d_feat)
    assert model_out["x_2"].shape == (1, d_feat)
    assert torch.equal(model_out["labels"], batch.y)
    assert torch.equal(model_out["batch_0"], batch.batch_0)


def test_topounet_wrapper_residual():
    """The rank-0 residual connection of AbstractWrapper must apply."""
    d_feat = 8
    batch = _toy_batch(d_feat)
    backbone = TopoUNet(d_feat, [0, 1, 2])
    wrapper = TopoUNetWrapper(
        backbone,
        out_channels=d_feat,
        num_cell_dimensions=1,
        residual_connections=True,
    )
    model_out = wrapper(batch)
    assert model_out["x_0"].shape == (4, d_feat)
