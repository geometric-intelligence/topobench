"""Unit tests for the OGFormer readout (topobench/nn/readouts/ogformer_readout.py)."""

import pytest
import torch
import torch_geometric

from topobench.nn.readouts import OGFormerReadOut


@pytest.fixture
def two_graph_batch():
    """Create a deterministic batch of two graphs.

    Returns
    -------
    torch_geometric.data.Data
        Batch with 10 nodes (5 per graph), node features and a batch
        vector.
    """
    torch.manual_seed(42)
    x = torch.randn(10, 12)
    batch_0 = torch.tensor([0] * 5 + [1] * 5)
    return torch_geometric.data.Data(
        x=x, x_0=x, batch_0=batch_0, num_nodes=10
    )


def test_ogformer_readout_node_level(two_graph_batch):
    """Test that the node-level readout exposes unmasked node logits.

    Parameters
    ----------
    two_graph_batch : torch_geometric.data.Data
        Deterministic two-graph batch fixture.
    """
    readout = OGFormerReadOut(
        hidden_dim=12, out_channels=3, task_level="node", pooling_type="sum"
    )
    _ = readout.__repr__()
    model_out = readout({"x_0": two_graph_batch.x_0}, two_graph_batch)
    assert model_out["logits"].shape == (10, 3)
    assert model_out["node_logits"] is model_out["logits"]


def test_ogformer_readout_graph_level(two_graph_batch):
    """Test that the graph-level readout pools the node embeddings.

    Parameters
    ----------
    two_graph_batch : torch_geometric.data.Data
        Deterministic two-graph batch fixture.
    """
    readout = OGFormerReadOut(
        hidden_dim=12, out_channels=3, task_level="graph", pooling_type="sum"
    )
    model_out = readout({"x_0": two_graph_batch.x_0}, two_graph_batch)
    assert model_out["logits"].shape == (2, 3)
