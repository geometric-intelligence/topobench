"""Unit tests for GraphMLP."""

import torch
import torch_geometric
from topobench.nn.backbones.graph import GraphMLP
from topobench.nn.wrappers.graph import GraphMLPWrapper
from topobench.loss.model import GraphMLPLoss

def testGraphMLP(random_graph_input):
    """ Unit test for GraphMLP.

    Parameters
    ----------
    random_graph_input : Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        A tuple of input tensors for testing EDGNN.
    """
    x, x_1, x_2, edges_1, edges_2 = random_graph_input
    batch = torch_geometric.data.Data(x_0=x, y=x, x=x, edge_index=edges_1, batch_0=torch.zeros(x.shape[0], dtype=torch.long))
    model = GraphMLP(x.shape[1], x.shape[1])
    wrapper = GraphMLPWrapper(model, **{"out_channels": x.shape[1], "num_cell_dimensions": 1})
    loss_fn = GraphMLPLoss()

    _ = wrapper.__repr__()
    _ = loss_fn.__repr__()

    model_out = wrapper(batch)
    assert model_out["x_0"].shape == x.shape
    assert list(model_out["x_dis"].shape) == [8,8]

    loss = loss_fn(model_out, batch)
    assert loss.item() >= 0

    model_out["x_dis"] = None
    loss = loss_fn(model_out, batch)
    assert loss == torch.tensor(0.0)
