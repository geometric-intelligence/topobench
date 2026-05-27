"""Unit tests for the combinatorial TopNets backbone."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.nn.backbones.combinatorial.topnets import (
    CombinatorialTopNetsBackbone,
    _interrank_edge_index,
    _make_activation,
    _node_adjacency_from_incidence,
    _rank_neighborhood_matrix,
)
from topobench.nn.readouts.propagate_signal_down import PropagateSignalDown
from topobench.nn.wrappers.combinatorial import TuneWrapper


def _mock_complex_batch() -> Data:
    x_0 = torch.randn(4, 8)
    x_1 = torch.randn(4, 8)
    x_2 = torch.randn(1, 8)

    batch = Data(
        x_0=x_0,
        x_1=x_1,
        x_2=x_2,
        y=torch.tensor([1]),
        batch_0=torch.zeros(x_0.shape[0], dtype=torch.long),
        batch_1=torch.zeros(x_1.shape[0], dtype=torch.long),
        batch_2=torch.zeros(x_2.shape[0], dtype=torch.long),
        cell_statistics=torch.tensor([[4, 4, 1]]),
    )

    batch["incidence_1"] = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [0, 0, 1, 1, 2, 2, 3, 3]]
        ),
        values=torch.ones(8),
        size=(4, 4),
    ).coalesce()
    batch["incidence_2"] = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]]),
        values=torch.ones(4),
        size=(4, 1),
    ).coalesce()
    batch["down_incidence-1"] = batch["incidence_1"]
    batch["up_incidence-0"] = batch["incidence_1"].t().coalesce()
    batch["down_incidence-2"] = batch["incidence_2"]
    batch["up_incidence-1"] = batch["incidence_2"].t().coalesce()
    batch["up_adjacency-0"] = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
        ),
        values=torch.ones(8),
        size=(4, 4),
    ).coalesce()
    batch["up_adjacency-1"] = torch.sparse_coo_tensor(
        indices=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
        ),
        values=torch.ones(8),
        size=(4, 4),
    ).coalesce()
    return batch


def test_combinatorial_topnets_forward_contract():
    """The backbone returns rank-keyed finite embeddings for TuneWrapper."""
    batch = _mock_complex_batch()
    model = CombinatorialTopNetsBackbone(
        in_channels=8,
        hidden_channels=8,
        neighborhoods=[
            "down_incidence-1",
            "up_incidence-0",
            "down_incidence-2",
            "up_incidence-1",
        ],
        num_layers=1,
        num_steps=2,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )

    out = model(batch)

    assert set(out) == {0, 1, 2}
    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape
    assert all(torch.isfinite(value).all() for value in out.values())


def test_combinatorial_topnets_tune_wrapper_output():
    """TuneWrapper exposes combinatorial TopNets outputs in TopoBench format."""
    batch = _mock_complex_batch()
    model = CombinatorialTopNetsBackbone(
        in_channels=8,
        hidden_channels=8,
        neighborhoods=["down_incidence-1", "down_incidence-2"],
        num_layers=1,
        num_steps=2,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )
    wrapper = TuneWrapper(
        model,
        out_channels=8,
        num_cell_dimensions=3,
        residual_connections=False,
    )

    model_out = wrapper(batch)

    assert model_out["x_0"].shape == batch.x_0.shape
    assert model_out["x_1"].shape == batch.x_1.shape
    assert model_out["x_2"].shape == batch.x_2.shape
    assert torch.equal(model_out["labels"], batch.y)
    assert torch.equal(model_out["batch_0"], batch.batch_0)


def test_combinatorial_topnets_omits_unconfigured_rank_three():
    """Rank-3 cells from complex_dim=3 lifting are not exposed to rank-2 readouts."""
    batch = _mock_complex_batch()
    batch.x_3 = torch.randn(2, 8)
    batch.batch_3 = torch.zeros(2, dtype=torch.long)
    batch["incidence_3"] = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 0], [0, 1]]),
        values=torch.ones(2),
        size=(1, 2),
    ).coalesce()

    model = CombinatorialTopNetsBackbone(
        in_channels=8,
        hidden_channels=8,
        neighborhoods=[
            "down_incidence-1",
            "down_incidence-2",
            "up_incidence-1",
        ],
        num_layers=1,
        num_steps=1,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )

    out = model(batch)

    assert set(out) == {0, 1, 2}

    wrapper = TuneWrapper(
        model,
        out_channels=8,
        num_cell_dimensions=3,
        residual_connections=False,
    )
    model_out = wrapper(batch)
    assert "x_3" not in model_out

    readout = PropagateSignalDown(
        readout_name="PropagateSignalDown",
        num_cell_dimensions=3,
        hidden_dim=8,
        out_channels=2,
        task_level="graph",
        pooling_type="sum",
    )
    readout_out = readout(model_out, batch)
    assert readout_out["logits"].shape == (1, 2)
    assert torch.isfinite(readout_out["logits"]).all()


def test_combinatorial_topnets_updates_without_adjacency_keys():
    """Incidence-only combinatorial neighborhoods are enough to run TopNets."""
    batch = _mock_complex_batch()
    del batch["up_adjacency-0"]
    del batch["up_adjacency-1"]
    model = CombinatorialTopNetsBackbone(
        in_channels=8,
        hidden_channels=8,
        neighborhoods=["down_incidence-1"],
        num_layers=1,
        num_steps=2,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )

    out = model(batch)

    assert out[0].shape == batch.x_0.shape
    assert torch.isfinite(out[0]).all()


def test_combinatorial_topnets_invalid_parameters():
    """Invalid combinatorial TopNets construction raises clear errors."""
    with pytest.raises(ValueError, match="neighborhoods"):
        CombinatorialTopNetsBackbone(
            in_channels=8, hidden_channels=8, neighborhoods=[]
        )

    with pytest.raises(ValueError, match="num_layers"):
        CombinatorialTopNetsBackbone(
            in_channels=8,
            hidden_channels=8,
            neighborhoods=["down_incidence-1"],
            num_layers=0,
        )


def test_rank_fallback_updates_intrarank_routes():
    """Rank fallback API uses up-adjacency neighborhoods."""
    batch = _mock_complex_batch()
    model = CombinatorialTopNetsBackbone(
        in_channels=8,
        hidden_channels=8,
        ranks=(0, 1),
        num_layers=1,
        num_steps=1,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )

    out = model(batch)

    assert set(out) == {0, 1}
    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape


def test_membership_fallbacks():
    """Membership falls back to cell_statistics and single-complex batches."""
    batch = _mock_complex_batch()
    del batch.batch_0
    del batch.batch_1
    del batch.batch_2
    model = CombinatorialTopNetsBackbone(
        in_channels=8,
        hidden_channels=8,
        ranks=(0,),
        num_layers=1,
        num_steps=1,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )

    out_with_stats = model(batch)
    assert out_with_stats[0].shape == batch.x_0.shape

    del batch.cell_statistics
    out_without_stats = model(batch)
    assert out_without_stats[0].shape == batch.x_0.shape


def test_missing_route_rank_is_ignored():
    """Routes whose source rank is absent are skipped safely."""
    batch = _mock_complex_batch()
    model = CombinatorialTopNetsBackbone(
        in_channels=8,
        hidden_channels=8,
        neighborhoods=["down_incidence-3"],
        num_layers=1,
        num_steps=1,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )

    out = model(batch)

    assert out[0].shape == batch.x_0.shape
    assert out[1].shape == batch.x_1.shape
    assert out[2].shape == batch.x_2.shape


def test_helper_edge_cases():
    """Helper functions handle sparse edge cases explicitly."""
    batch = _mock_complex_batch()

    with pytest.raises(KeyError, match="Missing combinatorial neighborhood"):
        _rank_neighborhood_matrix(batch, "missing", rank=1)

    empty_incidence = torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long),
        torch.empty(0),
        (3, 2),
    ).coalesce()
    empty_adj = _node_adjacency_from_incidence(empty_incidence)
    assert empty_adj._nnz() == 0

    singleton_incidence = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [0, 1]]),
        torch.ones(2),
        (3, 2),
    ).coalesce()
    singleton_adj = _node_adjacency_from_incidence(singleton_incidence)
    assert singleton_adj._nnz() == 0

    empty_interrank = _interrank_edge_index(
        empty_incidence, num_dst=3, device=torch.device("cpu")
    )
    assert empty_interrank.shape == (2, 0)


def test_activation_variants_and_invalid_activation():
    """Supported activation names resolve and invalid names fail."""
    for activation in ["elu", "tanh", "sigmoid", "id"]:
        assert isinstance(_make_activation(activation), torch.nn.Module)

    with pytest.raises(ValueError, match="Unsupported activation"):
        _make_activation("bad")

    with pytest.raises(ValueError, match="Unsupported activation"):
        CombinatorialTopNetsBackbone(
            in_channels=8,
            hidden_channels=8,
            ranks=(0,),
            activation="bad",
        )


def test_interrank_route_zero_initializes_destination_features():
    """Inter-rank routes send source-cell signal into zero-initialized destinations."""

    class SpyRoute:
        def __init__(self):
            self.x = None
            self.edge_index = None
            self.batch = None

        def __call__(self, x, edge_index, batch, edge_weight=None):
            self.x = x.clone()
            self.edge_index = edge_index.clone()
            self.batch = batch.clone()
            return torch.arange(
                x.numel(), dtype=x.dtype, device=x.device
            ).view_as(x)

    batch = _mock_complex_batch()
    model = CombinatorialTopNetsBackbone(
        in_channels=8,
        hidden_channels=8,
        neighborhoods=["down_incidence-1"],
        num_layers=1,
        num_steps=1,
        num_filtrations=2,
        filtration_hidden=4,
        coord_fun_count=1,
    )
    spy = SpyRoute()

    out = model._interrank_forward(
        batch=batch,
        route_model=spy,
        neighborhood="down_incidence-1",
        src_x=batch.x_1,
        dst_x=batch.x_0,
        src_batch=batch.batch_1,
        dst_batch=batch.batch_0,
    )

    num_dst = batch.x_0.shape[0]
    expected_route_out = torch.arange(
        spy.x.numel(),
        dtype=spy.x.dtype,
        device=spy.x.device,
    ).view_as(spy.x)
    torch.testing.assert_close(spy.x[:num_dst], torch.zeros_like(batch.x_0))
    torch.testing.assert_close(spy.x[num_dst:], batch.x_1)
    torch.testing.assert_close(out, expected_route_out[:num_dst])
    assert spy.edge_index.shape[0] == 2
    assert spy.edge_index.max() < spy.x.shape[0]
    assert spy.batch.shape[0] == spy.x.shape[0]
