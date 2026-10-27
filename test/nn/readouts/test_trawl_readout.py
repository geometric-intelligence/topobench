"""Tests for the TRAWL readout."""

import pytest
import torch

from test._utils.trawl import collate, network, prepare
from topobench.evaluator import TBEvaluator
from topobench.loss.dataset.DatasetLoss import DatasetLoss
from topobench.nn.readouts.base import AbstractZeroCellReadOut
from topobench.nn.readouts.trawl import TRAWLReadout


def test_is_a_topobench_readout():
    readout = TRAWLReadout(8, 2, task_level="node_inductive")
    assert isinstance(readout, AbstractZeroCellReadOut)
    out = readout({"x_0": torch.randn(5, 8)}, None)
    assert out["logits"].shape == (5, 2)


def test_walk_readout_matches_single_head_call_rng():
    """Walk prediction draws no dropout masks for unused graph logits."""
    readout = TRAWLReadout(
        4, 1, head_hidden=8, dropout=0.5, aggregation="walk_logits"
    )
    walks = torch.randn(6, 4)
    graph = torch.randn(2, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    torch.manual_seed(81)
    expected = readout.linear(walks).reshape(2, 3, 1).mean(1)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(81)
    actual = readout(
        {
            "graph_embedding": graph,
            "walk_embedding": walks,
            "walk_batch": batch,
        },
        None,
    )["logits"]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng)


def test_linear_readout_uses_input_dropout():
    readout = TRAWLReadout(8, 1, dropout=0.5, input_dropout=0.0)
    assert readout.linear[0].p == 0.0


def test_separate_readout_does_not_silently_discard_fusion():
    data = prepare()
    output = network([data], walk_scope="separate", fusion="attention")(
        collate([data])
    )
    with pytest.raises(ValueError, match="fusion"):
        TRAWLReadout(16, 2, graph_dim=32, aggregation="walk_logits")(
            output, None
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
    net = network(graphs)
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
