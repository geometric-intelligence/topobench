"""Unit tests for the GPR-GNN backbone."""

import pytest
import torch
import torch_geometric

from topobench.nn.backbones.graph.gprgnn import GPRGNN, GPRProp
from topobench.nn.wrappers.graph import GNNWrapper


def _make_batch(x, edge_index):
    """Build a minimal single-graph batch for the GNN wrapper.

    Parameters
    ----------
    x : torch.Tensor
        Node feature matrix.
    edge_index : torch.Tensor
        Graph connectivity in COO format.

    Returns
    -------
    torch_geometric.data.Data
        A batch consumable by :class:`GNNWrapper`.
    """
    return torch_geometric.data.Data(
        x_0=x,
        x=x,
        y=torch.randint(0, 2, (x.shape[0],)),
        edge_index=edge_index,
        batch_0=torch.zeros(x.shape[0], dtype=torch.long),
    )


@pytest.mark.parametrize("init", ["SGC", "PPR", "NPPR", "Random"])
def test_gprgnn_forward(random_graph_input, init):
    """GPR-GNN returns node embeddings with the hidden dimension.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    init : str
        GPR coefficient initialization scheme under test.
    """
    x, _, _, edges_1, _ = random_graph_input
    hidden = 16
    model = GPRGNN(x.shape[1], hidden, K=4, alpha=0.1, init=init)
    wrapper = GNNWrapper(model, out_channels=hidden, num_cell_dimensions=1)

    _ = wrapper.__repr__()
    _ = model.prop1.__repr__()

    out = wrapper(_make_batch(x, edges_1))
    assert out["x_0"].shape == (x.shape[0], hidden)


def test_gprgnn_warm_start_and_reset(random_graph_input):
    """Warm-start init accepts provided coefficients and survives reset.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, _, _ = random_graph_input
    K = 4
    gamma = torch.ones(K + 1) / (K + 1)
    model = GPRGNN(x.shape[1], 8, K=K, alpha=0.1, init="WS", gamma=gamma)
    torch.testing.assert_close(model.prop1.temp.detach(), gamma)

    model.reset_parameters()
    torch.testing.assert_close(model.prop1.temp.detach(), gamma)


def test_gprgnn_dprate_zero(random_graph_input):
    """The dprate=0 branch runs without dropping the hidden features.

    Parameters
    ----------
    random_graph_input : tuple
        Fixture providing random node features and edge indices.
    """
    x, _, _, edges_1, _ = random_graph_input
    model = GPRGNN(x.shape[1], 8, K=3, dprate=0.0)
    out = model(x, edges_1)
    assert out.shape == (x.shape[0], 8)


def test_gprprop_invalid_init():
    """An unknown init scheme raises a ValueError."""
    with pytest.raises(ValueError):
        GPRProp(K=4, alpha=0.1, init="nope")


def test_gprprop_warm_start_requires_gamma():
    """WS init without coefficients raises a ValueError."""
    with pytest.raises(ValueError):
        GPRProp(K=4, alpha=0.1, init="WS", gamma=None)
