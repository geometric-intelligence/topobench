import pytest
import torch
from torch_geometric.data import Data
from unittest.mock import MagicMock, patch
from topobench.transforms.data_manipulations.add_gpse_information import AddGPSEInformation, interrank_boundary_index

def _base_kwargs(**overrides):
    """Minimum kwargs needed to instantiate ``AddGPSEInformation``."""
    kwargs = {
        "max_rank": 1,
        "copy_initial": True,
        "neighborhoods": ["up_adjacency-0"],
        "device": "cpu",
        "cuda": [0],
        "path_to_pretrained_model": "/tmp/gpse_models",
        "pretrain_model": "molpcba",
    }
    kwargs.update(overrides)
    return kwargs

class MockGPSE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        # GPSE returns (x_out, something_else)
        num_nodes = data.x.shape[0]
        return torch.randn(num_nodes, 512), None

@patch("torch_geometric.nn.GPSE.from_pretrained")
def test_add_gpse_information_init(mock_from_pretrained):
    mock_from_pretrained.return_value = MockGPSE()

    kwargs = _base_kwargs()
    transform = AddGPSEInformation(**kwargs)

    assert transform.type == "add_gpse_information"
    assert transform.max_rank == 1
    assert transform.copy_initial is True
    assert transform.neighborhoods == ["up_adjacency-0"]
    assert transform.device == "cpu"
    assert transform.hidden_dim == 512
    assert transform.dim_in == 20

    mock_from_pretrained.assert_called_once_with("molpcba", root="/tmp/gpse_models")

@patch("torch_geometric.nn.GPSE.from_pretrained")
def test_add_gpse_information_forward(mock_from_pretrained):
    mock_model = MockGPSE()
    mock_from_pretrained.return_value = mock_model

    kwargs = _base_kwargs(neighborhoods=["up_adjacency-0", "adjacency-0"])
    transform = AddGPSEInformation(**kwargs)

    # Create sample data
    # Nodes (rank 0)
    x_0 = torch.randn(5, 3)
    # up_adjacency-0
    up_adjacency_0 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]]),
        values=torch.ones(5),
        size=(5, 5)
    )
    # adjacency-0
    adjacency_0 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]),
        values=torch.ones(5),
        size=(5, 5)
    )

    data = Data(x_0=x_0, **{"up_adjacency-0": up_adjacency_0, "adjacency-0": adjacency_0})

    # We need x_1 even if max_rank is 1, if it's used.
    # The transform iterates up to max_rank + 1
    data.x_1 = torch.randn(0, 3) # Empty rank 1

    out_data = transform(data)

    # Check if x0_0 and x1_0 are set because copy_initial is True
    assert hasattr(out_data, "x0_0")
    assert hasattr(out_data, "x1_0")
    assert torch.equal(out_data.x0_0, x_0)

    # Check if x0_1 and x1_1 are set (hop_num = int(copy_initial) = 1)
    assert hasattr(out_data, "x0_1")
    assert hasattr(out_data, "x1_1")

    # x0_1 should have shape (5, 512 + 512) because it aggregates 2 neighborhoods for rank 0
    assert out_data.x0_1.shape == (5, 1024)
    assert out_data.x1_1.shape == (0, 512)

def test_interrank_boundary_index():
    x_src = torch.tensor([[1.0], [2.0], [3.0]])
    boundary_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    n_dst_nodes = 4

    edge_index, edge_attr = interrank_boundary_index(x_src, boundary_index, n_dst_nodes)

    # Expected: node_ids are [0, 1, 1, 2], shifted edge_ids are [0+4, 0+4, 1+4, 1+4] = [4, 4, 5, 5]
    assert torch.equal(edge_index[0], torch.tensor([0, 1, 1, 2]))
    assert torch.equal(edge_index[1], torch.tensor([4, 4, 5, 5]))
    # edge_attr should be x_src[edge_ids] = [x_src[0], x_src[0], x_src[1], x_src[1]] = [1, 1, 2, 2]
    # In interrank_boundary_index, edge_attr = x_src[edge_ids].squeeze()
    # x_src is (3, 1), x_src[edge_ids] is (4, 1), squeeze() makes it (4,)
    assert torch.equal(edge_attr, torch.tensor([1.0, 1.0, 2.0, 2.0]))

@patch("torch_geometric.nn.GPSE.from_pretrained")
def test_add_gpse_information_forward_interrank(mock_from_pretrained):
    mock_model = MockGPSE()
    mock_from_pretrained.return_value = mock_model

    # Testing interrank (boundary/coboundary)
    # Neighborhood from rank 1 to rank 0 (boundary)
    kwargs = _base_kwargs(neighborhoods=["down_boundary-1"], max_rank=1)
    transform = AddGPSEInformation(**kwargs)

    x_0 = torch.randn(4, 3)
    x_1 = torch.randn(2, 3) # 2 edges
    down_boundary_1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]), # Edge 0 connects nodes 0,1; Edge 1 connects nodes 2,3
        values=torch.ones(4),
        size=(4, 2)
    )

    data = Data(x_0=x_0, x_1=x_1, **{"down_boundary-1": down_boundary_1})

    out_data = transform(data)

    assert hasattr(out_data, "x0_1")
    assert out_data.x0_1.shape == (4, 512)
    assert hasattr(out_data, "x1_1")
    assert out_data.x1_1.shape == (2, 512)

@patch("torch_geometric.nn.GPSE.from_pretrained")
def test_add_gpse_information_forward_edge_cases(mock_from_pretrained):
    mock_model = MockGPSE()
    mock_from_pretrained.return_value = mock_model

    kwargs = _base_kwargs(neighborhoods=["up_adjacency-0"])
    transform = AddGPSEInformation(**kwargs)

    # Case: 0 nodes
    data_0 = Data(x_0=torch.randn(0, 3))
    data_0.x_1 = torch.randn(0, 3)
    out_0 = transform(data_0)
    assert out_0.x0_1.shape == (0, 512)

    # Case: 1 node
    data_1 = Data(x_0=torch.randn(1, 3))
    data_1.x_1 = torch.randn(0, 3)
    out_1 = transform(data_1)
    assert out_1.x0_1.shape == (1, 512)
    assert torch.all(out_1.x0_1 == 0)

@patch("torch_geometric.nn.GPSE.from_pretrained")
def test_add_gpse_information_no_info_for_rank(mock_from_pretrained):
    mock_model = MockGPSE()
    mock_from_pretrained.return_value = mock_model

    # Neighborhoods only for rank 0
    kwargs = _base_kwargs(neighborhoods=["up_adjacency-0"], max_rank=1)
    transform = AddGPSEInformation(**kwargs)

    x_0 = torch.randn(5, 3)
    x_1 = torch.randn(3, 3) # Rank 1 has nodes but no neighborhoods
    data = Data(x_0=x_0, x_1=x_1, **{"up_adjacency-0": torch.sparse_coo_tensor(torch.zeros((2, 0)), torch.zeros(0), (5, 5))})

    out = transform(data)

    # Rank 0 has info
    assert out.x0_1.shape == (5, 512)
    # Rank 1 has NO info passed, should be initialized to zeros
    assert hasattr(out, "x1_1")
    assert out.x1_1.shape == (3, 512)
    assert torch.all(out.x1_1 == 0)
