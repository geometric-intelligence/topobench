"""Unit tests for the OGFormer wrapper (topobench/nn/wrappers/graph/ogformer_wrapper.py)."""

import pytest
import torch
import torch_geometric

from topobench.nn.backbones.graph import OGFormer
from topobench.nn.wrappers.graph import OGFormerWrapper


@pytest.fixture
def two_graph_batch():
    """Create a deterministic batch of two graphs with node labels.

    Returns
    -------
    torch_geometric.data.Data
        Batch with 10 nodes (5 per graph), block-structured edges,
        node features, labels and a batch vector.
    """
    torch.manual_seed(42)
    x = torch.randn(10, 12)
    edges_g1 = torch.tensor([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 3]])
    edges_g2 = edges_g1 + 5
    edge_index = torch.cat([edges_g1, edges_g2], dim=1)
    batch_0 = torch.tensor([0] * 5 + [1] * 5)
    y = torch.tensor([0, 0, 1, 1, 2, 0, 1, 1, 2, 2])
    return torch_geometric.data.Data(
        x=x,
        x_0=x,
        edge_index=edge_index,
        batch_0=batch_0,
        y=y,
        num_nodes=10,
    )


def test_ogformer_wrapper(two_graph_batch):
    """Test the wrapper output dictionary in training and eval modes.

    Parameters
    ----------
    two_graph_batch : torch_geometric.data.Data
        Deterministic two-graph batch fixture.
    """
    model = OGFormer(12, 12, n_layers=2)
    wrapper = OGFormerWrapper(
        model,
        out_channels=12,
        num_cell_dimensions=1,
        residual_connections=False,
    )
    _ = wrapper.__repr__()

    wrapper.train()
    model_out = wrapper(two_graph_batch)
    assert model_out["x_0"].shape == (10, 12)
    assert torch.equal(model_out["labels"], two_graph_batch.y)
    assert torch.equal(model_out["batch_0"], two_graph_batch.batch_0)
    assert len(model_out["ogformer_queries"]) == 2
    assert len(model_out["ogformer_attention"]) == 2

    wrapper.eval()
    model_out = wrapper(two_graph_batch)
    assert model_out["x_0"].shape == (10, 12)
    assert "ogformer_queries" not in model_out
    assert "ogformer_attention" not in model_out
