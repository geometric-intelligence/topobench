"""Behavioral tests for topology, batching, layers, objectives and config."""

import copy
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch_geometric.data import Data

from topobench.data.utils.trawl.encodings import positional_encodings
from topobench.data.utils.trawl.sampling import WalkSampler
from topobench.dataloader import DataloadDataset
from topobench.dataloader.utils import collate_fn
from topobench.evaluator import TBEvaluator
from topobench.loss.dataset.DatasetLoss import DatasetLoss
from topobench.model.trawl_pretraining import TRAWLPretrainer
from topobench.nn.backbones.combinatorial.trawl import (
    TRAWL,
    PureTorchMambaBlock,
)
from topobench.nn.readouts.trawl import TRAWLReadout
from topobench.transforms.data_manipulations.trawl import TRAWLTransform


def complex_data(empty=False):
    incidence = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return Data(
        x_0=torch.randn(3, 2),
        x_1=torch.randn(2, 3),
        x_2=torch.empty(0, 1) if empty else torch.randn(1, 1),
        incidence_1=incidence.to_sparse(),
        incidence_2=(
            torch.empty(2, 0) if empty else torch.ones(2, 1)
        ).to_sparse(),
        num_nodes=3,
        y=torch.tensor([1]),
        train_mask=torch.tensor([True]),
        val_mask=torch.tensor([True]),
        test_mask=torch.tensor([True]),
    )


def prepare(data=None, **kwargs):
    return TRAWLTransform(
        encodings={"local": True, "rw_steps": 2, "rw_samples": 3}, **kwargs
    )(data or complex_data())


def collate(graphs):
    dataset = DataloadDataset(graphs)
    return collate_fn([dataset[i] for i in range(len(dataset))])


def model(graphs, **kwargs):
    backbone = TRAWL(
        hidden_dim=16,
        depth=2,
        architecture="mlp",
        walks={"k": 3, "length": 5},
        **kwargs,
    )
    backbone.initialize(graphs)
    return backbone


def test_hasse_and_overlap_are_different():
    hasse = prepare(graph="hasse")
    overlap = prepare(graph="cell_overlap")
    assert len(hasse.trawl_edges) == 12
    assert hasse.trawl_relations.item() == 2
    assert overlap.trawl_relations.item() == 1
    assert torch.all(overlap.trawl_edges[:, :2] >= 3)
    assert hasse.trawl_counts.tolist() == [[3, 2, 1]]


def test_adjacency_and_multihop():
    graph = prepare(
        neighborhoods=[
            "up_adjacency-0",
            "down_adjacency-1",
            "2-up_incidence-0",
        ]
    )
    edges = graph.trawl_edges
    assert {(0, 1), (1, 0), (1, 2), (2, 1)} == set(
        map(tuple, edges[edges[:, 2] == 0, :2].tolist())
    )
    assert (3, 4) in set(map(tuple, edges[edges[:, 2] == 1, :2].tolist()))
    assert len(edges[edges[:, 2] == 2]) == 6


def test_hypergraph_and_plain_graph():
    hyper = Data(
        x=torch.randn(3, 2),
        incidence_hyperedges=torch.tensor(
            [[1.0, 0], [1, 1], [0, 1]]
        ).to_sparse(),
        num_nodes=3,
        y=torch.tensor([0]),
    )
    adapted = prepare(hyper, max_rank=1)
    assert adapted.trawl_counts.tolist() == [[3, 2]]
    plain = Data(
        x=torch.ones(3, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        num_nodes=3,
        y=torch.tensor([0]),
    )
    adapted = prepare(plain, max_rank=0)
    assert adapted.trawl_edges.shape == (2, 3)


@pytest.mark.parametrize("scope", ["union", "separate"])
@pytest.mark.parametrize("fusion", ["mean", "concat", "learned", "attention"])
def test_batch_isolation_and_empty_neighborhood(scope, fusion):
    graphs = [prepare(), prepare(complex_data(empty=True))]
    net = model(graphs, walk_scope=scope, fusion=fusion).eval()
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


def test_empty_graph_relations_fallback_and_gradients():
    graph = prepare(
        Data(x=torch.randn(3, 2), num_nodes=3, y=torch.tensor([1])), max_rank=2
    )
    net = model([graph])
    output = net(collate([graph]))
    assert len(output["walk_embedding"]) == 0
    assert torch.isfinite(output["graph_embedding"]).all()
    output["graph_embedding"].sum().backward()
    assert net.features[0].weight.grad is not None


@pytest.mark.parametrize(
    "kind", ["mamba", "sisa", "hybrid", "transformer", "gru", "mlp"]
)
def test_layers_backward(kind):
    graph = prepare()
    net = TRAWL(
        hidden_dim=16, depth=2, architecture=kind, walks={"k": 2, "length": 4}
    )
    net.initialize([graph])
    out = net(collate([graph]))
    loss = out["graph_embedding"].square().mean()
    loss.backward()
    grads = [p.grad for p in net.encoders.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_mamba_scan_parity():
    parallel = PureTorchMambaBlock(8, d_state=4)
    sequential = copy.deepcopy(parallel)
    sequential.scan = "sequential"
    x = torch.randn(2, 7, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    first, second = parallel(x), sequential(y)
    torch.testing.assert_close(first, second)
    first.square().sum().backward()
    second.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)


def test_walk_budget_nonbacktracking_and_reversal():
    matrix = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
    paths = WalkSampler(0)(matrix, k=8, length=10, reverse=True)
    assert paths.shape == (16, 10)
    np.testing.assert_array_equal(paths[8:], paths[:8, ::-1])
    assert np.all(paths[:8, 2:] != paths[:8, :-2])
    np.testing.assert_array_equal(
        paths, WalkSampler(0)(matrix, k=8, length=10, reverse=True)
    )


def test_spectral_and_color_refinement():
    matrix = sp.csr_matrix([[0.0, 1], [1, 0]])
    pe = positional_encodings(
        matrix,
        rw_steps=2,
        heat_times=[0.1],
        electrostatic_betas=[0.1],
        laplacian_dim=4,
    )
    assert pe.shape == (2, 10)
    assert np.isfinite(pe).all()
    graph = complex_data()
    a, b = (
        prepare(graph.clone()),
        prepare(graph.clone(), color_refinement=True),
    )
    assert torch.equal(a.trawl_edges, b.trawl_edges)
    assert b.trawl_pe.shape[1] == a.trawl_pe.shape[1] * 6


def test_sampling_state_resume_and_eval_stability():
    graph = prepare()
    a = model([graph])
    a(collate([graph]))
    b = model([graph])
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


@pytest.mark.parametrize(
    "task,loss_type,classes",
    [
        ("classification", "cross_entropy", 2),
        ("classification", "BCE", 1),
        ("regression", "mae", 1),
        ("regression", "mse", 3),
        ("multilabel classification", "BCE", 3),
    ],
)
@pytest.mark.parametrize("level", ["graph", "node"])
def test_tasks(task, loss_type, classes, level):
    graphs = [prepare(), prepare()]
    net = model(graphs)
    batch = collate(graphs)
    output = net(batch)
    readout = TRAWLReadout(16, classes, task_level=level, graph_dim=32)
    output = readout(output, batch)
    n = len(output["logits"])
    target = (
        torch.randint(2, (n,))
        if task == "classification"
        else torch.rand(n, classes)
    )
    if task == "multilabel classification":
        target = target.round()
        target[0, 0] = torch.nan
    loss = DatasetLoss(
        {"task": task, "loss_type": loss_type}
    ).forward_criterion(output["logits"], target)
    assert torch.isfinite(loss)
    loss.backward()
    evaluator = TBEvaluator(
        task,
        num_classes=max(classes, 2),
        metrics=["mae" if task == "regression" else "accuracy"],
    )
    evaluator.update({"logits": output["logits"].detach(), "labels": target})
    assert all(
        torch.isfinite(value).all() for value in evaluator.compute().values()
    )


def test_pretraining_uses_no_labels():
    graphs = [prepare(), prepare(complex_data(empty=True))]
    backbone = model(graphs)
    pretrainer = TRAWLPretrainer(
        backbone, objectives={"features": 1.0, "colors": 1.0, "topology": 1.0}
    )
    batch = collate(graphs)
    del batch.y
    loss = pretrainer._step(batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_hydra_composition():
    config_dir = str(Path(__file__).resolve().parents[3] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="run",
            overrides=[
                "model=combinatorial/trawl",
                "dataset=graph/PROTEINS",
                "logger=[]",
            ],
        )
        assert cfg.transforms.trawl.transform_name == "TRAWLTransform"
        instance = instantiate(
            cfg.model,
            evaluator=cfg.evaluator,
            optimizer=cfg.optimizer,
            loss=cfg.loss,
        )
        assert isinstance(instance.backbone, TRAWL)
