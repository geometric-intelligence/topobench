"""Tests for the TRAWL wrapper: batching, walk sampling and checks."""

import pytest
import torch

from test._utils.trawl import collate, complex_data, network, prepare
from topobench.nn.backbones.combinatorial.trawl import TRAWL
from topobench.nn.wrappers.combinatorial.trawl_wrapper import TRAWLWrapper


@pytest.mark.parametrize("scope", ["union", "separate"])
@pytest.mark.parametrize("fusion", ["mean", "concat", "learned", "attention"])
def test_batch_isolation_and_empty_neighborhood(scope, fusion):
    graphs = [prepare(), prepare(complex_data(empty=True))]
    net = network(graphs, walk_scope=scope, fusion=fusion).eval()
    batched = net(collate(graphs))
    separate = [net(collate([graph])) for graph in graphs]
    torch.testing.assert_close(
        batched["graph_embedding"],
        torch.cat([out["graph_embedding"] for out in separate]),
    )
    torch.testing.assert_close(
        batched["x_0"], torch.cat([out["x_0"] for out in separate])
    )
    assert torch.isfinite(batched["graph_embedding"]).all()


def test_output_layout():
    graphs = [prepare(), prepare()]
    out = network(graphs).eval()(collate(graphs))
    assert out["x_0"].shape == (6, 16) and out["x_2"].shape == (2, 16)
    assert out["batch_1"].tolist() == [0, 0, 1, 1]
    assert out["labels"].tolist() == [1, 1]
    assert out["cell_embeddings"].shape == (12, 16)
    assert out["graph_embedding"].shape == (2, 32)
    assert len(out["walk_batch"]) == len(out["walk_embedding"]) == 6


def test_sampling_state_resume_and_eval_stability():
    graph = prepare()
    a = network([graph])
    a(collate([graph]))
    b = network([graph])
    b.load_state_dict(a.state_dict())
    torch.testing.assert_close(
        a(collate([graph]))["graph_embedding"],
        b(collate([graph]))["graph_embedding"],
    )
    a.eval()
    torch.testing.assert_close(
        a(collate([graph]))["graph_embedding"],
        a(collate([graph]))["graph_embedding"],
    )


def test_host_copies_match_device_fields():
    graphs = [prepare(), prepare()]
    net = network(graphs).eval()
    batch = collate(graphs)
    expected = net(batch.clone())["graph_embedding"]
    batch.host_copies = {
        key: batch[key].numpy() for key in net.host_fields if key in batch
    }
    torch.testing.assert_close(net(batch)["graph_embedding"], expected)


def test_eval_views_add_walks():
    graph = prepare()
    net = network([graph], eval_views=3).eval()
    out = net(collate([graph]))
    assert out["num_views"] == 3
    assert set(out["walk_view"].tolist()) == {0, 1, 2}


def test_rank_and_neighborhood_mismatches_are_reported():
    graph = prepare()
    with pytest.raises(ValueError, match="max_rank=2"):
        network([prepare(max_rank=1)], max_rank=2)
    net = network([graph], max_rank=2)
    lower = prepare(max_rank=1)
    with pytest.raises(ValueError, match="max_rank=2"):
        net(collate([lower, lower]))
    with pytest.raises(ValueError, match="neighborhood counts"):
        network([graph], num_neighborhoods=3)(collate([graph]))


def test_invalid_options_are_rejected():
    with pytest.raises(ValueError, match="walk_refresh"):
        TRAWLWrapper(TRAWL(architecture="mlp"), walk_refresh="never")
    with pytest.raises(ValueError, match="eval_views"):
        TRAWLWrapper(TRAWL(architecture="mlp"), eval_views=0)
