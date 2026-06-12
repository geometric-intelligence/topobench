"""Unit tests for the combinatorial ETNN backbone."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.backbones.combinatorial.etnn import (
    ETNN,
    _neighborhood_to_edge_index,
)
from topobench.nn.wrappers.combinatorial import TuneWrapper


def create_mock_complex_batch():
    """Create a small lifted combinatorial complex batch."""
    # The mock batch mirrors the tensors produced by a graph-to-combinatorial
    # lifting after feature encoding: one feature matrix per cell rank.
    x_0 = torch.randn(4, 16)
    x_1 = torch.randn(4, 16)
    x_2 = torch.randn(2, 16)

    # incidence_1 stores node-edge incidence with rows as rank-0 cells and
    # columns as rank-1 cells. Its transpose is the corresponding up-incidence
    # relation used for node -> edge messages.
    incidence_1 = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 0, 3],
                [0, 0, 1, 1, 2, 2, 3, 3],
            ]
        ),
        values=torch.ones(8),
        size=(4, 4),
    ).coalesce()

    # incidence_2 stores edge-face incidence with rows as rank-1 cells and
    # columns as rank-2 cells. Its transpose gives edge -> face messages.
    incidence_2 = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 1, 2, 1, 2, 3],
                [0, 0, 0, 1, 1, 1],
            ]
        ),
        values=torch.ones(6),
        size=(4, 2),
    ).coalesce()

    # Same-rank adjacency gives ETNN within-rank communication for nodes.
    adjacency_0 = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2],
            ]
        ),
        values=torch.ones(6),
        size=(4, 4),
    ).coalesce()

    # Same-rank adjacency for rank-1 cells protects the edge update path.
    adjacency_1 = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [
                [0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2],
            ]
        ),
        values=torch.ones(6),
        size=(4, 4),
    ).coalesce()

    # Same-rank adjacency for rank-2 cells protects the face update path.
    adjacency_2 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [1, 0]]),
        values=torch.ones(2),
        size=(2, 2),
    ).coalesce()

    return Data(
        x_0=x_0,
        x_1=x_1,
        x_2=x_2,
        y=torch.tensor([1]),
        # Batch vectors are included because the wrapper/readout contract
        # expects them, even though the backbone itself does not use them.
        batch_0=torch.zeros(x_0.shape[0], dtype=torch.long),
        batch_1=torch.zeros(x_1.shape[0], dtype=torch.long),
        batch_2=torch.zeros(x_2.shape[0], dtype=torch.long),
        **{
            "up_adjacency-0": adjacency_0,
            "up_adjacency-1": adjacency_1,
            "up_adjacency-2": adjacency_2,
            "up_incidence-0": incidence_1.T.coalesce(),
            "down_incidence-1": incidence_1,
            "up_incidence-1": incidence_2.T.coalesce(),
            "down_incidence-2": incidence_2,
        },
    )


def create_etnn():
    """Instantiate the Stage 1 ETNN backbone."""
    # Keep this neighborhood list aligned with
    # configs/model/combinatorial/etnn.yaml so unit coverage matches the public
    # model config.
    return ETNN(
        in_channels=16,
        hidden_channels=8,
        out_channels=16,
        neighborhoods=[
            "up_adjacency-0",
            "up_adjacency-1",
            "up_adjacency-2",
            "up_incidence-0",
            "down_incidence-1",
            "up_incidence-1",
            "down_incidence-2",
        ],
        num_layers=2,
    )


def test_etnn_preserves_rankwise_shapes():
    """ETNN returns one tensor per configured cell rank."""
    batch = create_mock_complex_batch()
    model = create_etnn()

    out = model(batch)

    # The backbone should preserve the number of cells per rank so downstream
    # wrappers/readouts can use the original batch vectors.
    assert set(out) == {0, 1, 2}
    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape


def test_etnn_runs_without_positions():
    """The initial TopoBench ETNN core must not require ``data.pos``."""
    batch = create_mock_complex_batch()

    # PyG ``Data`` may answer ``hasattr(data, "pos")`` through dynamic
    # attribute handling, so check the actual stored keys instead.
    assert "pos" not in batch

    out = create_etnn()(batch)

    # This protects the GraphUniverse use case: the coordinate-free core should
    # be runnable before any ETNN-specific coordinate machinery is added.
    assert out[0].shape == batch.x_0.shape


def test_etnn_outputs_fit_tune_wrapper_contract():
    """ETNN can use the existing combinatorial ``TuneWrapper``."""
    batch = create_mock_complex_batch()
    wrapper = TuneWrapper(
        backbone=create_etnn(),
        out_channels=16,
        num_cell_dimensions=3,
        residual_connections=False,
    )

    out = wrapper(batch)

    # TuneWrapper should expose ETNN rank outputs using TopoBench's model_out
    # keys, which keeps the standard readout path available.
    assert torch.equal(out["labels"], batch.y)
    assert torch.equal(out["batch_0"], batch.batch_0)
    assert out["x_0"].shape == batch.x_0.shape
    assert out["x_1"].shape == batch.x_1.shape
    assert out["x_2"].shape == batch.x_2.shape


def test_etnn_handles_empty_rank_with_stored_zero_edges():
    """Empty ranks with explicit sparse zeros should not create messages."""
    batch = create_mock_complex_batch()

    # Real lifted batches can contain no cells for a requested rank while the
    # generated sparse neighborhood still stores zero-valued placeholder
    # entries. The backbone should treat those entries as absent edges.
    batch.x_2 = torch.empty(0, 16)
    batch.batch_2 = torch.empty(0, dtype=torch.long)
    batch["up_adjacency-2"] = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [0, 1]]),
        values=torch.zeros(2),
        size=(2, 2),
    ).coalesce()
    batch["up_incidence-1"] = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(0, 4),
    ).coalesce()
    batch["down_incidence-2"] = torch.sparse_coo_tensor(
        indices=torch.empty(2, 0, dtype=torch.long),
        values=torch.empty(0),
        size=(4, 0),
    ).coalesce()

    out = create_etnn()(batch)

    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape


def test_neighborhood_to_edge_index_uses_columns_as_senders():
    """Sparse neighborhood rows are receivers and columns are senders."""
    # One rank-1 edge e0 is incident to two rank-0 nodes n0 and n1.
    # The up-incidence route is node -> edge, but TopoBench stores the sparse
    # matrix as rows=edge receivers and columns=node senders.
    up_incidence_0 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0], [0, 1]]),
        values=torch.ones(2),
        size=(1, 2),
    ).coalesce()
    batch = Data(**{"up_incidence-0": up_incidence_0})

    edge_index, edge_attr = _neighborhood_to_edge_index(
        batch=batch,
        neighborhood="up_incidence-0",
        src_rank=0,
        dst_rank=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_src_cells=2,
        num_dst_cells=1,
    )

    # After conversion, both node senders should point to edge receiver e0.
    expected_edge_index = torch.tensor([[0, 1], [0, 0]])
    assert torch.equal(edge_index, expected_edge_index)
    assert edge_attr.shape == (2, 1)


def test_neighborhood_to_edge_index_compacts_empty_rank_placeholders():
    """Batched sparse axes may contain placeholders for empty ranks."""
    # Graph 0 has no rank-2 cells, so its lifted rank-2 adjacency is represented
    # by a stored zero placeholder. Graph 1 has two real rank-2 cells. After
    # batching, the sparse matrix axis has three slots, but x_2 has two rows.
    rank_2_adjacency = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2], [0, 2, 1]]),
        values=torch.tensor([0.0, 1.0, 1.0]),
        size=(3, 3),
    ).coalesce()
    batch = Data(
        x_2=torch.randn(2, 16),
        batch_2=torch.tensor([1, 1]),
        **{"up_adjacency-2": rank_2_adjacency},
    )

    edge_index, edge_attr = _neighborhood_to_edge_index(
        batch=batch,
        neighborhood="up_adjacency-2",
        src_rank=2,
        dst_rank=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        num_src_cells=2,
        num_dst_cells=2,
    )

    # Sparse slots 1 and 2 should compact to feature rows 0 and 1, while the
    # placeholder slot 0 should disappear with its zero value.
    expected_edge_index = torch.tensor([[1, 0], [0, 1]])
    assert torch.equal(edge_index, expected_edge_index)
    assert edge_attr.shape == (2, 1)


def test_etnn_requires_neighborhoods():
    """A relation-wise backbone needs at least one cell relation."""
    # An empty relation set would make ETNN's typed message-passing layer
    # ill-defined, so fail early with a clear config error.
    with pytest.raises(ValueError, match="at least one neighborhood"):
        ETNN(
            in_channels=16,
            hidden_channels=8,
            out_channels=16,
            neighborhoods=[],
        )
