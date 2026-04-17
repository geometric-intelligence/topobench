"""Unit tests for TopoTune_OneHasse."""

import pytest
import torch
from torch_geometric.data import Data
from test._utils.nn_module_auto_test import NNModuleAutoTest
from topobench.nn.backbones.combinatorial.gccn_onehasse import TopoTune_OneHasse, get_activation
from torch_geometric.nn import GCNConv
from omegaconf import OmegaConf

class MockGNN(torch.nn.Module):
    """Mock GNN module for testing purposes.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
        """Forward pass of the MockGNN.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_index : torch.Tensor
            Edge indices.

        Returns
        -------
        torch.Tensor
            Output of the GCN layer.
        """
        return self.conv(x, edge_index)


class MockGNNWithLinear(MockGNN):
    """
    Mock GNN with Linear layer (ignoring edge_index).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index=None):
        """Forward pass of the MockGNN.

        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_index : torch.Tensor
            Edge indices.

        Returns
        -------
        torch.Tensor
            Output of the GCN layer.
        """
        return self.linear(x)


def create_mock_complex_batch():
    """Create a mock complex batch for testing.

    Returns
    -------
    Data
        A PyTorch Geometric Data object representing a mock complex batch.
    """
    # 3 nodes, 3 edges, 1 face
    x_0 = torch.randn(3, 16)  # 3 nodes
    x_1 = torch.randn(3, 16)  # 3 edges
    x_2 = torch.randn(1, 16)  # 1 face

    batch = Data(x_0=x_0, x_1=x_1, x_2=x_2)

    # Incidence matrices
    incidence_1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 1, 2, 0, 2],  # node indices
                              [0, 0, 1, 1, 2, 2]]),  # edge indices
        values=torch.ones(6),
        size=(3, 3)  # (num_nodes, num_edges)
    ).coalesce()
    batch["down_incidence-1"] = incidence_1

    incidence_2 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2],  # edge indices
                              [0, 0, 0]]),  # face index
        values=torch.ones(3),
        size=(3, 1)  # (num_edges, num_faces)
    ).coalesce()
    batch["down_incidence-2"] = incidence_2

    # Adjacency matrices (remain unchanged)
    adjacency_0 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1, 1, 2, 2],
                              [1, 2, 0, 2, 0, 1]]),
        values=torch.ones(6),
        size=(3, 3)  # (num_nodes, num_nodes)
    ).coalesce()
    batch["up_adjacency-0"] = adjacency_0

    adjacency_1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0, 1, 1, 2, 2],
                              [1, 2, 0, 2, 0, 1]]),
        values=torch.ones(6),
        size=(3, 3)  # (num_edges, num_edges)
    ).coalesce()
    batch["up_adjacency-1"] = adjacency_1

    adjacency_2 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [0]]),
        values=torch.ones(1),
        size=(1, 1)  # (num_faces, num_faces)
    ).coalesce()
    batch["up_adjacency-2"] = adjacency_2

    cell_statistics = torch.tensor([[3, 3, 1]])
    batch["cell_statistics"] = cell_statistics
    return batch

class ModifiedNNModuleAutoTest(NNModuleAutoTest):
    """Modified NNModuleAutoTest class for TopoTune_OneHasse testing."""

    def assert_return_tensor(self, result):
        """Assert that the result contains a dictionary with tensor values.

        Parameters
        ----------
        result : Any
            The result to check.
        """
        assert any(isinstance(r, dict) and any(isinstance(v, torch.Tensor) for v in r.values()) for r in result)

    def assert_equal_output(self, module, result, result_2):
        """Assert that two outputs are equal.

        Parameters
        ----------
        module : torch.nn.Module
            The module being tested.
        result : Any
            The first result to compare.
        result_2 : Any
            The second result to compare.
        """
        assert len(result) == len(result_2)

        for i, r1 in enumerate(result):
            r2 = result_2[i]
            if isinstance(r1, dict) and isinstance(r2, dict):
                assert r1.keys() == r2.keys(), f"Dictionaries have different keys at index {i}"
                for key in r1.keys():
                    assert torch.allclose(r1[key], r2[key], atol=1e-6), f"Tensors not equal for key {key} at index {i}"
            elif isinstance(r1, torch.Tensor):
                assert torch.allclose(r1, r2, atol=1e-6), f"Tensors not equal at index {i}"
            else:
                assert r1 == r2, f"Values not equal at index {i}"

def test_topotune_onehasse():
    """Test the TopoTune_OneHasse module using ModifiedNNModuleAutoTest."""
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    neighborhoods = OmegaConf.create(["up_adjacency-0", "up_adjacency-1", "down_incidence-1", "down_incidence-2"])#[[[0, 0], "adjacency"], [[1, 1], "adjacency"], [[1, 0], "boundary"], [[2, 1], "boundary"]])

    auto_test = ModifiedNNModuleAutoTest([
        {
            "module": TopoTune_OneHasse,
            "init": {
                "GNN": gnn,
                "neighborhoods": neighborhoods,
                "layers": 2,
                "use_edge_attr": False,
                "activation": "relu"
            },
            "forward": (batch,),
        }
    ])
    auto_test.run()

def test_topotune_onehasse_methods():
    """Test individual methods of the TopoTune_OneHasse module."""
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    neighborhoods = OmegaConf.create(["up_adjacency-0", "down_incidence-1"])#[[[0, 0], "adjacency"], [[1, 0], "boundary"]])
    topotune = TopoTune_OneHasse(GNN=gnn, neighborhoods=neighborhoods, layers=2, use_edge_attr=False, activation="relu")

    # Test generate_membership_vectors
    membership = topotune.generate_membership_vectors(batch)
    assert 0 in membership and 1 in membership and 2 in membership
    assert membership[0].shape == (batch.x_0.shape[0],)
    assert membership[1].shape == (batch.x_1.shape[0],)
    assert membership[2].shape == (batch.x_2.shape[0],)

    # Set the membership attribute (simulating the forward method)
    topotune.membership = membership

    # Test all_nbhds_expand
    expanded = topotune.all_nbhds_expand(batch, membership)
    assert isinstance(expanded, Data)
    assert expanded.x.shape == (7, 16)  # (3 nodes + 3 edges + 1 face) * 2 batches
    assert expanded.edge_index.shape[0] == 2
    assert expanded.batch.shape == (7,)

    # Test all_nbhds_gnn_forward
    output = topotune.all_nbhds_gnn_forward(expanded, 0)
    assert output.shape == (7, 16)

    # Test aggregate_inter_nbhd
    x_out = torch.randn(7, 16)
    aggregated = topotune.aggregate_inter_nbhd(x_out)
    assert 0 in aggregated and 1 in aggregated and 2 in aggregated
    assert aggregated[0].shape == (3, 16)  # 3 nodes * 2 batches
    assert aggregated[1].shape == (3, 16)  # 3 edges * 2 batches
    assert aggregated[2].shape == (1, 16)  # 1 face * 2 batches

    # Test forward method
    output = topotune(batch)
    assert isinstance(output, dict)
    assert 0 in output and 1 in output and 2 in output
    assert output[0].shape == (3, 16)  # 3 nodes * 2 batches
    assert output[1].shape == (3, 16)  # 3 edges * 2 batches
    assert output[2].shape == (1, 16)  # 1 face * 2 batches

def test_get_activation():
    """Test the get_activation function."""
    relu_func = get_activation("relu")
    assert callable(relu_func)

    relu_module = get_activation("relu", return_module=True)
    assert issubclass(relu_module, torch.nn.Module)

    with pytest.raises(NotImplementedError):
        get_activation("invalid_activation")


def test_topotune_onehasse_early_return_x2_zero():
    """
    Test the early return path in forward() when batch.x_2.shape[0] == 0.
    """
    batch = create_mock_complex_batch()
    batch.x_2 = torch.zeros((0, 16))  # Force x_2 to have 0 faces
    gnn = MockGNN(16, 32, 16)

    # Define any neighborhoods; they won't matter since x_2=0 triggers early return
    neighborhoods = OmegaConf.create(["up_adjacency-0", "down_incidence-2"])

    model = TopoTune_OneHasse(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=2,
        use_edge_attr=False,
        activation="relu",
    )
    out = model(batch)
    # Model should skip expansions and return {0: x_0, 1: x_1, 2: x_2}.
    assert 0 in out and 1 in out and 2 in out
    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape
    # Verify no changes were made to the features
    assert torch.allclose(out[0], batch.x_0, atol=1e-6)
    assert torch.allclose(out[1], batch.x_1, atol=1e-6)
    assert out[2].numel() == 0  # zero faces indeed


def test_topotune_onehasse_fallback_rank_not_updated():
    """
    Test the fallback in forward() for a rank that is never updated.
    """
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)

    # Suppose we only define neighborhoods that produce a route for rank=0->0,
    # ignoring rank=2 entirely. This means rank=2 won't show up in x_out_per_rank,
    # triggering the fallback assignment in forward().
    neighborhoods = OmegaConf.create(["up_adjacency-0"])

    model = TopoTune_OneHasse(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=False,
        activation="relu",
    )
    out = model(batch)
    # Ranks 0,1,2 should be in final output, even though only rank=0 was updated.
    assert 0 in out and 1 in out and 2 in out
    # rank=2 should remain the same as the input
    assert torch.allclose(out[2], batch.x_2, atol=1e-6)


@pytest.mark.parametrize(
    "bad_neighborhood,expected_errmsg",
    [
        ("up_adjacency-2", "Unsupported src_rank for 'up' neighborhood: 2"),
        ("down_adjacency-0", "Unsupported src_rank for 'down' neighborhood: 0"),
        ("down_incidence-0", "Unsupported src_rank for 'down_incidence' neighborhood: 0"),
        ("up_incidence-2", "Unsupported src_rank for 'up_incidence' neighborhood: 2"),
    ]
)
def test_topotune_onehasse_unsupported_src_rank_raises(bad_neighborhood, expected_errmsg):
    """
    Test that a ValueError is raised if a neighborhood implies an unsupported src_rank.

    Parameters
    ----------
    bad_neighborhood : str
        Unsupported neighborhood name.
    expected_errmsg : str
        Expected error message.
    """
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)

    neighborhoods = OmegaConf.create([bad_neighborhood])
    model = TopoTune_OneHasse(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=False,
        activation="relu",
    )

    with pytest.raises(ValueError, match=expected_errmsg):
        model(batch)


def test_topotune_onehasse_indexerror_in_aggregate_inter_nbhd(mocker):
    """
    Force an IndexError in aggregate_inter_nbhd to cover that branch.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        Mocker object.
    """
    batch = create_mock_complex_batch()
    gnn = MockGNN(16, 32, 16)
    neighborhoods = OmegaConf.create(["up_adjacency-0", "down_incidence-1"])

    model = TopoTune_OneHasse(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=False,
        activation="relu",
    )

    # We'll mock or patch model.membership after it's generated so it claims rank=0 has more elements than it does.
    # For instance, if batch.x_0 has shape [3,16], membership[0].shape is (3,).
    # Let's forcibly set membership[0] to have 10 elements => triggers end_idx out-of-bounds.
    original_generate_membership_vectors = model.generate_membership_vectors

    def fake_generate_membership_vectors(b):
        """
        Fake membership vector generation that inflates membership for rank=0.

        Parameters
        ----------
        b : torch_geometric.data.Data
            The input batch data.

        Returns
        -------
        dict of {int: torch.Tensor}
            The artificially inflated membership dictionary.
        """
        membership = original_generate_membership_vectors(b)
        membership[0] = torch.arange(10)  # artificially claim 10 'nodes' at rank=0
        return membership

    mocker.patch.object(model, 'generate_membership_vectors', side_effect=fake_generate_membership_vectors)

    with pytest.raises(IndexError, match="out of bounds"):
        model(batch)


def create_special_batch():
    """
    Create a batch with shapes adjusted to trigger certain corner cases.

    For instance:
    - 2 faces (x_2 of size [2, *])
    - Non-square adjacency or incidence to see if it leads to certain expansions
      or error-handling in all_nbhds_expand.

    Returns
    -------
    Data
        Bacthed graphs.
    """
    x_0 = torch.randn(4, 8)   # rank 0: 4 nodes
    x_1 = torch.randn(2, 8)   # rank 1: 2 edges
    x_2 = torch.randn(2, 8)   # rank 2: 2 faces
    batch = Data(x_0=x_0, x_1=x_1, x_2=x_2)

    # Minimal adjacency/incidence, possibly non-square
    # to ensure certain expansions or indexing happen
    batch["up_adjacency-0"] = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 2]]),  # slightly "irregular"
        values=torch.ones(2),
        size=(4, 4)
    ).coalesce()
    batch["down_incidence-1"] = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [0, 0]]),
        values=torch.ones(2),
        size=(4, 2)  # node->edge shape or something that might not match typical
    ).coalesce()
    # Possibly no adjacency for rank=2 or something partial

    batch["cell_statistics"] = torch.tensor([[4, 2, 2]])
    return batch


def test_valueerror_in_all_nbhds_expand_missing_neighborhood_key():
    """
    Trigger a ValueError by passing a neighborhood type that leads to an unsupported condition.
    """
    batch = create_special_batch()
    gnn = MockGNNWithLinear(8, 16, 8)
    # Suppose 'down_laplacian-2' is not recognized by the code, or leads to a raise.
    neighborhoods = OmegaConf.create(["down_laplacian-2"])

    model = TopoTune_OneHasse(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=True,
        activation="relu",
    )

    with pytest.raises(AttributeError, match="GlobalStorage' object has no attribute 'down_laplacian-2"):
        model(batch)


def test_aggregate_inter_nbhd_index_error(mocker):
    """
    Force an IndexError in aggregate_inter_nbhd by artificially inflating membership for one of the ranks so end_idx exceeds x_out.shape[0].

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        Mocker object used for patching.
    """
    batch = create_special_batch()
    gnn = MockGNNWithLinear(8, 16, 8)
    neighborhoods = OmegaConf.create(["up_adjacency-0", "down_incidence-1"])

    model = TopoTune_OneHasse(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=False,
        activation="relu",
    )

    # We'll run forward once to set up membership, then patch membership[0].
    model.membership = model.generate_membership_vectors(batch)
    # Suppose membership[0] is an array of length 4, let's artificially set it to 10
    # so the aggregator tries to slice out-of-bounds.
    model.membership[0] = torch.arange(10)

    # We'll call aggregate_inter_nbhd directly, simulating a post-GNN output of size 8
    # but membership says rank 0 alone has 10 elements -> triggers IndexError.
    fake_x_out = torch.randn(8, 8)  # only 8 total features
    with pytest.raises(IndexError, match="out of bounds"):
        model.aggregate_inter_nbhd(fake_x_out)


def test_fallback_for_unupdated_rank():
    """
    Test the scenario where a rank never gets updated because no neighborhoods exist for it.
    """
    batch = create_special_batch()
    # Suppose we define a neighborhood that only touches rank=0 and rank=1
    # but never rank=2. This ensures rank=2 is not updated.
    neighborhoods = OmegaConf.create(["up_adjacency-0"])  # e.g., node->node

    gnn = MockGNNWithLinear(8, 16, 8)
    model = TopoTune_OneHasse(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=2,
        use_edge_attr=False,
        activation="relu",
    )

    out = model(batch)
    # The code's final loop ensures rank=2 is still present even if never updated.
    assert 2 in out
    # rank=2 should remain exactly as input
    assert torch.allclose(out[2], batch.x_2, atol=1e-6)


def test_partial_layer_execution_x2_nonzero():
    """
    Cover the scenario where batch.x_2.shape[0] > 0 but we still have partial execution.
    """
    batch = create_special_batch()
    # We have 2 faces, so x_2.shape[0] != 0 => no early return
    # Let's define neighborhoods that do a partial coverage across layers
    neighborhoods = OmegaConf.create(["up_adjacency-0", "down_incidence-1"])

    gnn = MockGNNWithLinear(8, 16, 8)
    model = TopoTune_OneHasse(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=3,  # multiple layers
        use_edge_attr=False,
        activation="relu",
    )

    out = model(batch)
    # Check that the final dictionary has ranks 0,1,2
    for rank_id in [0, 1, 2]:
        assert rank_id in out
        # Ensure they have the correct shape
        original_feats = getattr(batch, f"x_{rank_id}")
        assert out[rank_id].shape == original_feats.shape
    # This helps cover code inside the for-loop for multiple layers.


def test_activation_id():
    """
    Ensure coverage of the 'id' activation path.
    """
    batch = create_special_batch()
    neighborhoods = OmegaConf.create(["up_adjacency-0"])
    gnn = MockGNNWithLinear(8, 16, 8)

    model = TopoTune_OneHasse(
        GNN=gnn,
        neighborhoods=neighborhoods,
        layers=1,
        use_edge_attr=False,
        activation="id",  # identity activation
    )

    out = model(batch)
    # The identity activation should result in no nonlinearity being applied
    # beyond the raw linear transform.
    for rank_id in [0, 1, 2]:
        assert rank_id in out
