"""Unit tests for the GRIT wrapper."""

import torch
from torch_geometric.data import Batch, Data

from topobench.nn.backbones.graph.grit import GRITBackbone
from topobench.nn.wrappers import GRITWrapper
from topobench.nn.wrappers.graph import WRAPPER_CLASSES
from topobench.transforms.data_manipulations.rrwp_positional_encodings import (
    AddRRWP,
)

HIDDEN_DIM = 16
WALK_LENGTH = 4


def _make_batch(with_rrwp: bool = True) -> Batch:
    """Create a TopoBench-style batch of two small graphs.

    Parameters
    ----------
    with_rrwp : bool, optional
        Whether to attach precomputed RRWP encodings (default: True).

    Returns
    -------
    torch_geometric.data.Batch
        Batch with ``x_0`` and ``batch_0`` attributes.
    """
    transform = AddRRWP(walk_length=WALK_LENGTH)
    data_list = []
    for num_nodes in (5, 4):
        src = torch.arange(num_nodes - 1)
        dst = src + 1
        edge_index = torch.cat(
            [torch.stack([src, dst]), torch.stack([dst, src])], dim=1
        )
        data = Data(
            x=torch.randn(num_nodes, HIDDEN_DIM),
            edge_index=edge_index,
            y=torch.zeros(1, dtype=torch.long),
        )
        if with_rrwp:
            data = transform(data)
        data_list.append(data)

    batch = Batch.from_data_list(data_list)
    batch.x_0 = batch.x
    batch.batch_0 = batch.batch
    return batch


def _make_wrapper() -> GRITWrapper:
    """Instantiate a GRIT wrapper around a small backbone.

    Returns
    -------
    GRITWrapper
        The wrapped backbone.
    """
    backbone = GRITBackbone(
        hidden_dim=HIDDEN_DIM, num_layers=1, walk_length=WALK_LENGTH
    )
    return GRITWrapper(
        backbone,
        out_channels=HIDDEN_DIM,
        num_cell_dimensions=1,
        residual_connections=False,
    )


class TestGRITWrapper:
    """Test the GRIT wrapper."""

    def test_registration(self):
        """Test that the wrapper is auto-discovered by TopoBench."""
        assert "GRITWrapper" in WRAPPER_CLASSES

    def test_forward_with_precomputed_rrwp(self):
        """Test the wrapper forward pass on a batch with RRWP."""
        batch = _make_batch(with_rrwp=True)
        wrapper = _make_wrapper()

        model_out = wrapper(batch)

        assert set(model_out.keys()) == {"labels", "batch_0", "x_0"}
        assert model_out["x_0"].shape == (batch.num_nodes, HIDDEN_DIM)
        assert torch.equal(model_out["labels"], batch.y)
        assert torch.equal(model_out["batch_0"], batch.batch_0)

    def test_forward_without_rrwp(self):
        """Test that the wrapper works when RRWP attributes are absent."""
        batch = _make_batch(with_rrwp=False)
        wrapper = _make_wrapper()

        model_out = wrapper(batch)

        assert model_out["x_0"].shape == (batch.num_nodes, HIDDEN_DIM)
        assert torch.isfinite(model_out["x_0"]).all()
