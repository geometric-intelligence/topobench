"""Unit tests for ConfigurableGNN backbone."""

import torch
import torch_geometric

from topobench.nn.backbones.graph import ConfigurableGNN, TunedGNN
from topobench.nn.wrappers.graph import GNNWrapper

_FIXTURE_DOC = """
    Parameters
    ----------
    random_graph_input : tuple
        Pytest fixture providing (x, x_1, x_2, edges_1, edges_2).
"""


def test_configurable_gnn_basic(random_graph_input):
    """Test basic forward pass with default settings.

    Parameters
    ----------
    random_graph_input : tuple
        Pytest fixture providing (x, x_1, x_2, edges_1, edges_2).
    """
    x, _, _, edges_1, _ = random_graph_input
    batch = torch_geometric.data.Data(
        x_0=x,
        y=x,
        x=x,
        edge_index=edges_1,
        batch_0=torch.zeros(x.shape[0], dtype=torch.long),
    )
    model = ConfigurableGNN(
        in_channels=x.shape[1],
        hidden_channels=16,
        num_layers=2,
        preprocess_edges=False,
    )
    wrapper = GNNWrapper(
        model,
        out_channels=16,
        num_cell_dimensions=1,
        residual_connections=False,
    )

    model_out = wrapper(batch)
    assert model_out["x_0"].shape == (x.shape[0], 16)


def test_configurable_gnn_all_features(random_graph_input):
    """Test with all features enabled: res, bn, jk, pre_linear, pre_ln, in_dropout.

    Parameters
    ----------
    random_graph_input : tuple
        Pytest fixture providing (x, x_1, x_2, edges_1, edges_2).
    """
    x, _, _, edges_1, _ = random_graph_input
    model = ConfigurableGNN(
        in_channels=x.shape[1],
        hidden_channels=16,
        num_layers=3,
        dropout=0.1,
        in_dropout=0.1,
        res=True,
        bn=True,
        jk=True,
        pre_linear=True,
        pre_ln=True,
        preprocess_edges=False,
    )
    out = model(x, edges_1)
    assert out.shape == (x.shape[0], 16)


def test_configurable_gnn_gnn_types(random_graph_input):
    """Test GCN, SAGE, and GAT convolution types.

    Parameters
    ----------
    random_graph_input : tuple
        Pytest fixture providing (x, x_1, x_2, edges_1, edges_2).
    """
    x, _, _, edges_1, _ = random_graph_input
    for gnn_type in ["gcn", "sage", "gat"]:
        model = ConfigurableGNN(
            in_channels=x.shape[1],
            hidden_channels=16,
            num_layers=2,
            gnn=gnn_type,
            preprocess_edges=False,
        )
        out = model(x, edges_1)
        assert out.shape == (x.shape[0], 16), f"Failed for gnn={gnn_type}"


def test_configurable_gnn_ln(random_graph_input):
    """Test with LayerNorm instead of BatchNorm.

    Parameters
    ----------
    random_graph_input : tuple
        Pytest fixture providing (x, x_1, x_2, edges_1, edges_2).
    """
    x, _, _, edges_1, _ = random_graph_input
    model = ConfigurableGNN(
        in_channels=x.shape[1],
        hidden_channels=16,
        num_layers=2,
        ln=True,
        bn=False,
        preprocess_edges=False,
    )
    out = model(x, edges_1)
    assert out.shape == (x.shape[0], 16)


def test_configurable_gnn_reset_parameters(random_graph_input):
    """Test that reset_parameters runs without error.

    Parameters
    ----------
    random_graph_input : tuple
        Pytest fixture providing (x, x_1, x_2, edges_1, edges_2).
    """
    x, _, _, edges_1, _ = random_graph_input
    model = ConfigurableGNN(
        in_channels=x.shape[1],
        hidden_channels=16,
        num_layers=2,
        res=True,
        bn=True,
        pre_ln=True,
        preprocess_edges=False,
    )
    out1 = model(x, edges_1)
    model.reset_parameters()
    out2 = model(x, edges_1)
    assert out1.shape == out2.shape


def test_configurable_gnn_preprocess_edges(random_graph_input):
    """Test edge preprocessing (to_undirected + self_loops).

    Parameters
    ----------
    random_graph_input : tuple
        Pytest fixture providing (x, x_1, x_2, edges_1, edges_2).
    """
    x, _, _, edges_1, _ = random_graph_input
    model = ConfigurableGNN(
        in_channels=x.shape[1],
        hidden_channels=16,
        num_layers=2,
        preprocess_edges=True,
    )
    out = model(x, edges_1)
    assert out.shape == (x.shape[0], 16)


def test_tuned_gnn_alias():
    """Verify TunedGNN is an alias for ConfigurableGNN."""
    assert TunedGNN is ConfigurableGNN
