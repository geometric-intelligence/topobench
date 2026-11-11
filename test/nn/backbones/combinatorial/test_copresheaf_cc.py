"""Tests for the combinatorial-complex copresheaf backbone."""

import torch
from torch_geometric.data import Data

from topobench.nn.backbones.combinatorial.copresheaf_cc import CopresheafCC
from topobench.nn.wrappers.combinatorial.copresheaf_cc_wrapper import (
    CopresheafCCWrapper,
)


def _sparse(indices, size):
    indices = torch.tensor(indices)
    return torch.sparse_coo_tensor(
        indices, torch.ones(indices.size(1)), size
    ).coalesce()


def _complex_data():
    incidence = _sparse([[0, 1, 1, 2], [0, 0, 1, 1]], (3, 2))
    fields = {
        "x_0": torch.randn(3, 4),
        "x_1": torch.randn(2, 4),
        "batch_0": torch.zeros(3, dtype=torch.long),
        "batch_1": torch.zeros(2, dtype=torch.long),
        "up_incidence-0": incidence.t(),
        "down_incidence-1": incidence,
        "y": torch.tensor([1]),
    }
    return Data(**fields)


def test_copresheaf_cc_updates_and_returns_every_rank():
    """The higher-order backbone preserves cell counts at every rank."""
    model = CopresheafCC(
        in_channels=4,
        hidden_channels=8,
        out_channels=5,
        neighborhoods=["up_incidence-0", "down_incidence-1"],
        num_layers=2,
        heads=2,
        stalk_dimension=4,
        map_type={
            "up_incidence-0": "diagonal",
            "down_incidence-1": "full",
        },
    )
    batch = _complex_data()
    output = model(
        {0: batch.x_0, 1: batch.x_1},
        {
            "up_incidence-0": batch["up_incidence-0"],
            "down_incidence-1": batch["down_incidence-1"],
        },
    )

    assert output[0].shape == (3, 5)
    assert output[1].shape == (2, 5)


def test_copresheaf_cc_wrapper_follows_topobench_contract():
    """The wrapper attaches labels and per-rank batch membership."""
    model = CopresheafCC(
        in_channels=4,
        hidden_channels=4,
        out_channels=4,
        neighborhoods=["up_incidence-0", "down_incidence-1"],
        num_layers=1,
        heads=1,
        map_type="identity",
    )
    wrapper = CopresheafCCWrapper(
        model,
        out_channels=4,
        num_cell_dimensions=2,
        residual_connections=False,
    )
    batch = _complex_data()

    output = wrapper(batch)

    assert output["x_0"].shape == batch.x_0.shape
    assert output["x_1"].shape == batch.x_1.shape
    assert torch.equal(output["labels"], batch.y)
    assert torch.equal(output["batch_0"], batch.batch_0)
    assert torch.equal(output["batch_1"], batch.batch_1)
