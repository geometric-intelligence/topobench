"""Unit tests for EDGNN."""

import pytest

import torch
from ...._utils.nn_module_auto_test import NNModuleAutoTest
from topobench.nn.backbones.hypergraph.edgnn import (
    EDGNN,
    customMLP as edgnn_MLP,
    PlainMLP,
    EquivSetConv,
    JumpLinkConv,
    MeanDegConv
)

def test_EDGNN(random_graph_input):
    """ Unit test for EDGNN.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing EDGNN.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    auto_test = NNModuleAutoTest([
        {
            "module" : EDGNN,
            "init": (x.shape[1], ),
            "forward": (x, edges_1),
            "assert_shape": x.shape
        },
    ])
    auto_test.run()

    indices = torch.nonzero(edges_1, as_tuple=False).T
    values = edges_1[indices[0], indices[1]]
    sparse_edges_1 = torch.sparse_coo_tensor(indices, values, edges_1.size())

    auto_test2 = NNModuleAutoTest([
        {
            "module" : EDGNN,
            "init": {"num_features": x.shape[1],
                     "edconv_type": "JumpLink"},
            "forward": (x, sparse_edges_1),
            "assert_shape": x.shape
        },
    ])
    auto_test2.run()

    model = EDGNN(x.shape[1], edconv_type="MeanDeg")
    model.reset_parameters()

    with pytest.raises(ValueError):
        model = EDGNN(x.shape[1], edconv_type="Invalid")

def test_edgnn_MLP(random_graph_input):
    """ Unit test for edgnn_MLP.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing edgnn_MLP.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    hid_channels = 4
    out_channels = 10

    for num_layers in [1, 2, 3]:
        for Normalization in ['bn', 'ln', 'None']:
            for InputNorm in [True, False]:
                auto_test = NNModuleAutoTest([
                    {
                        "module" : edgnn_MLP,
                        "init": (x.shape[1], hid_channels, out_channels, num_layers, 0.5, Normalization, InputNorm),
                        "forward": (x, ),
                        "assert_shape": (x.shape[0], out_channels)
                    },
                ])
                auto_test.run()

    model = edgnn_MLP(x.shape[1], hid_channels, out_channels, num_layers, 0.5, 'bn', True)
    model.reset_parameters()
    model.flops(x)

def test_PlainMLP(random_graph_input):
    """ Unit test for PlainMLP.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing PlainMLP.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_nodes = x.shape[0]
    hid_channels = 4
    out_channels = 10
    num_layers = 3

    auto_test = NNModuleAutoTest([
        {
            "module" : PlainMLP,
            "init": (x.shape[1], hid_channels, out_channels, num_layers),
            "forward": (x, ),
            "assert_shape": (num_nodes, out_channels)
        },
    ])
    auto_test.run()

    model = PlainMLP(x.shape[1], hid_channels, out_channels, num_layers)
    model.reset_parameters()

def test_EquivSetConv(random_graph_input):
    """ Unit test for EquivSetConv.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing EquivSetConv.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_nodes = x.shape[0]
    hid_channels = 4
    out_channels = 10
    num_layers = 2

    auto_test = NNModuleAutoTest([
        {
            "module" : EquivSetConv,
            "init": (x.shape[1], x.shape[1]),
            "forward":  (x, edges_1[0], edges_1[1], x),
            "assert_shape": x.shape
        },
    ])
    auto_test.run()


def test_JumpLinkConv(random_graph_input):
    """ Unit test for JumpLinkConv.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing JumpLinkConv.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_nodes = x.shape[0]
    hid_channels = 4
    out_channels = 10
    num_layers = 2

    auto_test = NNModuleAutoTest([
        {
            "module" : JumpLinkConv,
            "init": (x.shape[1], x.shape[1]),
            "forward":  (x, edges_1[0], edges_1[1], x),
            "assert_shape": x.shape
        },
    ])
    auto_test.run()

def test_JumpLinkConv(random_graph_input):
    """ Unit test for JumpLinkConv.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing JumpLinkConv.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    num_nodes = x.shape[0]
    hid_channels = 4
    out_channels = 10
    num_layers = 2

    auto_test = NNModuleAutoTest([
        {
            "module" : JumpLinkConv,
            "init": (x.shape[1], x.shape[1]),
            "forward":  (x, edges_1[0], edges_1[1], x),
            "assert_shape": x.shape
        },
    ])
    auto_test.run()
