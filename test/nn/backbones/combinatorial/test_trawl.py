"""Tests for the TRAWL backbone."""

import pytest
import torch
from torch_geometric.data import Data

from test._utils.trawl import collate, complex_data, network, prepare
from topobench.nn.backbones.combinatorial.trawl import TRAWL
from topobench.transforms.data_manipulations.trawl import TRAWLTransform


def test_backbone_encodes_given_walks():
    torch.manual_seed(0)
    backbone = TRAWL(hidden_dim=8, depth=1, architecture="mlp")
    backbone.initialize([2, 3, 1], pe_width=4)
    assert backbone.features[1].in_features == 3
    assert backbone.position.in_features == 4
    signals = [torch.randn(3, 2), torch.randn(2, 3), torch.randn(1, 1)]
    ranks = torch.tensor([0, 0, 0, 1, 1, 2])
    paths = torch.tensor([[0, 3, 1, 4], [5, 4, 2, 4]])
    out = backbone(signals, torch.randn(6, 4), ranks, [[(0, paths)]])
    assert out["contextual"].shape == (6, 8)
    assert out["graph_embedding"].shape == (backbone.output_dim,)
    assert [view for view, _ in out["walks"]] == [0]
    assert out["walks"][0][1].shape == (2, backbone.output_dim)
    # Every cell is visited, so none falls back to its raw projection.
    projected = backbone.features[0](signals[0])
    assert not torch.allclose(out["contextual"][:3], projected)


@pytest.mark.parametrize(
    "kind", ["mamba", "sisa", "hybrid", "transformer", "gru", "mlp"]
)
def test_layers_backward(kind):
    graph = prepare()
    net = network([graph], architecture=kind, walks={"k": 2, "length": 4})
    out = net(collate([graph]))
    out["graph_embedding"].square().mean().backward()
    grads = [
        p.grad for p in net.backbone.encoders.parameters() if p.grad is not None
    ]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_empty_graph_relations_fallback_and_gradients():
    graph = prepare(
        Data(x=torch.randn(3, 2), num_nodes=3, y=torch.tensor([1])), max_rank=2
    )
    net = network([graph])
    output = net(collate([graph]))
    assert len(output["walk_embedding"]) == 0
    assert torch.isfinite(output["graph_embedding"]).all()
    output["graph_embedding"].sum().backward()
    assert net.backbone.features[0].weight.grad is not None


def test_attention_occurrences_and_absent_rank_widths():
    first = prepare(complex_data(empty=True))
    second = complex_data()
    second.x_2 = torch.randn(1, 4)
    second = prepare(second)
    TRAWLTransform().finalize_dataset([first, second])
    net = network(
        [first, second], occurrence_pooling="attention", graph_readout="cells"
    )
    result = net(collate([first, second]))
    result["graph_embedding"].square().sum().backward()
    assert net.backbone.occurrence_attention.weight.grad is not None
    assert torch.isfinite(result["graph_embedding"]).all()


@pytest.mark.parametrize("occurrence_pooling", ["mean", "attention"])
@pytest.mark.parametrize("walk_scope", ["union", "separate"])
def test_cpu_autocast_accumulation(occurrence_pooling, walk_scope):
    graphs = [prepare(), prepare()]
    net = network(
        graphs,
        walk_scope=walk_scope,
        occurrence_pooling=occurrence_pooling,
        graph_readout="cells",
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = net(collate(graphs))
        loss = output["graph_embedding"].square().mean()
    assert torch.isfinite(loss)
    assert output["x_0"].dtype == torch.float32


def test_invalid_options_are_rejected():
    with pytest.raises(ValueError, match="walk_scope"):
        TRAWL(walk_scope="both")
    with pytest.raises(ValueError, match="explicit layer list"):
        TRAWL(architecture="custom")
