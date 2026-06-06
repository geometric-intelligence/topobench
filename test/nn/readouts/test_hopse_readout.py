"""Unit tests for ``HOPSEReadout``."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.readouts.hopse import HOPSEReadout


def _make_model_out_and_batch(
    complex_dim: int = 3,
    max_hop: int = 3,
    hidden_dim: int = 8,
    n_per_dim: tuple = (6, 8, 4, 2),
):
    """Build (model_out, batch) mimicking ``HOPSEWrapper`` output."""
    fields = {"y": torch.tensor([0, 1])}
    model_out = {}
    for i in range(complex_dim + 1):
        n = n_per_dim[i]
        # Two graphs in this batch, alternating membership.
        batch_i = (torch.arange(n) % 2).long()
        fields[f"batch_{i}"] = batch_i
        model_out[f"batch_{i}"] = batch_i
        # Cheap incidence matrix linking each rank to the rank below (only
        # used in the node-level readout path). For the graph-level path the
        # readout never touches ``incidence_*`` so a dummy will do.
        if i > 0:
            fields[f"incidence_{i}"] = torch.eye(
                n_per_dim[i - 1], n
            ).to_sparse()
        for j in range(max_hop):
            model_out[f"x{i}_{j}"] = torch.randn(n, hidden_dim)
    batch = Data(**fields)
    return model_out, batch


class TestHOPSEReadoutGraphLevel:
    """Graph-level readout returns per-graph logits."""

    def test_forward_and_call_shapes(self):
        complex_dim, max_hop, hidden_dim, out_channels = 3, 3, 8, 4
        model_out, batch = _make_model_out_and_batch(
            complex_dim=complex_dim,
            max_hop=max_hop,
            hidden_dim=hidden_dim,
        )
        readout = HOPSEReadout(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level="graph",
            pooling_type="mean",
            complex_dim=complex_dim,
            max_hop=max_hop,
        )
        out = readout(model_out, batch)
        n_graphs = int(batch.batch_0.max().item()) + 1
        assert out["logits"].shape == (n_graphs, out_channels)

    @pytest.mark.parametrize("pooling_type", ["sum", "mean", "max"])
    def test_supported_pooling_types(self, pooling_type):
        complex_dim, max_hop, hidden_dim, out_channels = 2, 2, 4, 3
        model_out, batch = _make_model_out_and_batch(
            complex_dim=complex_dim,
            max_hop=max_hop,
            hidden_dim=hidden_dim,
        )
        readout = HOPSEReadout(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level="graph",
            pooling_type=pooling_type,
            complex_dim=complex_dim,
            max_hop=max_hop,
        )
        out = readout(model_out, batch)
        assert out["logits"].shape[-1] == out_channels

    def test_rejects_invalid_pooling(self):
        with pytest.raises(AssertionError):
            HOPSEReadout(
                hidden_dim=4,
                out_channels=2,
                task_level="graph",
                pooling_type="bogus",
                complex_dim=2,
                max_hop=2,
            )


class TestHOPSEReadoutNodeLevel:
    """Node-level readout returns per-node logits."""

    def test_forward_shape(self):
        complex_dim, max_hop, hidden_dim, out_channels = 3, 3, 8, 5
        n_per_dim = (6, 8, 4, 2)
        model_out, batch = _make_model_out_and_batch(
            complex_dim=complex_dim,
            max_hop=max_hop,
            hidden_dim=hidden_dim,
            n_per_dim=n_per_dim,
        )
        readout = HOPSEReadout(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level="node",
            pooling_type="sum",
            complex_dim=complex_dim,
            max_hop=max_hop,
        )
        out = readout(model_out, batch)
        # Node-level: one logit row per 0-cell.
        assert out["logits"].shape == (n_per_dim[0], out_channels)
