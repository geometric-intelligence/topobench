"""Unit tests for the OGFormer loss (topobench/loss/model/OGFormerLoss.py)."""

import pytest
import torch
import torch_geometric

from topobench.loss.model import OGFormerLoss
from topobench.nn.backbones.graph import OGFormer
from topobench.nn.readouts import OGFormerReadOut
from topobench.nn.wrappers.graph import OGFormerWrapper


@pytest.fixture
def two_graph_batch():
    """Create a deterministic batch of two graphs with node labels.

    Returns
    -------
    torch_geometric.data.Data
        Batch with 10 nodes (5 per graph), block-structured edges,
        node features, labels and a batch vector.
    """
    torch.manual_seed(42)
    x = torch.randn(10, 12)
    edges_g1 = torch.tensor([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 3]])
    edges_g2 = edges_g1 + 5
    edge_index = torch.cat([edges_g1, edges_g2], dim=1)
    batch_0 = torch.tensor([0] * 5 + [1] * 5)
    y = torch.tensor([0, 0, 1, 1, 2, 0, 1, 1, 2, 2])
    return torch_geometric.data.Data(
        x=x,
        x_0=x,
        edge_index=edge_index,
        batch_0=batch_0,
        y=y,
        num_nodes=10,
    )


def build_model_out(two_graph_batch, training=True):
    """Run OGFormer wrapper and readout to build a model output dict.

    Parameters
    ----------
    two_graph_batch : torch_geometric.data.Data
        Deterministic two-graph batch fixture.
    training : bool, optional
        Whether to run the backbone in training mode (default: True).

    Returns
    -------
    tuple[dict, OGFormer]
        The model output dictionary and the backbone module.
    """
    torch.manual_seed(0)
    model = OGFormer(12, 12, n_layers=2)
    wrapper = OGFormerWrapper(
        model,
        out_channels=12,
        num_cell_dimensions=1,
        residual_connections=False,
    )
    readout = OGFormerReadOut(
        hidden_dim=12, out_channels=3, task_level="node", pooling_type="sum"
    )
    wrapper.train(training)
    model_out = readout(wrapper(two_graph_batch), two_graph_batch)
    return model_out, model


def test_ogformer_loss_training_and_eval(two_graph_batch):
    """Test the NMH loss end-to-end in training and eval modes.

    Parameters
    ----------
    two_graph_batch : torch_geometric.data.Data
        Deterministic two-graph batch fixture.
    """
    loss_fn = OGFormerLoss(lambda_kl=1e-4, lambda_h=1e-4)
    _ = loss_fn.__repr__()

    model_out, model = build_model_out(two_graph_batch, training=True)
    loss = loss_fn(model_out, two_graph_batch)
    assert loss.item() > 0
    loss.backward()  # Loss must be differentiable
    assert model.layers[0].attention.lin_q.weight.grad is not None

    # Validation and test: no auxiliary outputs, loss is zero
    model_out, _ = build_model_out(two_graph_batch, training=False)
    loss = loss_fn(model_out, two_graph_batch)
    assert loss == torch.tensor(0.0)


def test_ogformer_loss_disabled_terms(two_graph_batch):
    """Test that zero lambdas disable the corresponding loss terms.

    Parameters
    ----------
    two_graph_batch : torch_geometric.data.Data
        Deterministic two-graph batch fixture.
    """
    model_out, _ = build_model_out(two_graph_batch, training=True)

    loss = OGFormerLoss(lambda_kl=0.0, lambda_h=0.0)(
        model_out, two_graph_batch
    )
    assert loss.item() == 0.0

    kl_only = OGFormerLoss(lambda_kl=1e-4, lambda_h=0.0)(
        model_out, two_graph_batch
    )
    h_only = OGFormerLoss(lambda_kl=0.0, lambda_h=1e-4)(
        model_out, two_graph_batch
    )
    assert kl_only.item() > 0
    assert h_only.item() > 0


def test_ogformer_loss_kl_matrix_properties():
    """Test that the KL divergence matrix has a zero diagonal and is >= 0."""
    torch.manual_seed(0)
    loss_fn = OGFormerLoss()
    queries = torch.sigmoid(torch.randn(6, 16))
    kl = loss_fn.kl_divergence_matrix(queries)
    assert kl.shape == (6, 6)
    assert torch.allclose(torch.diagonal(kl), torch.zeros(6), atol=1e-6)
    assert (kl > -1e-6).all()  # KL divergence is non-negative


def test_ogformer_loss_weighted_homophily():
    """Test the weighted homophily rate on a perfectly homophilous graph."""
    attention = torch.tensor(
        [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    )
    y_same = torch.tensor([0, 0, 1])
    h = OGFormerLoss.weighted_homophily(attention, y_same)
    assert torch.allclose(h, torch.ones(3))

    y_mixed = torch.tensor([0, 1, 1])
    h = OGFormerLoss.weighted_homophily(attention, y_mixed)
    assert torch.allclose(h, torch.tensor([0.5, 0.5, 1.0]))


def test_ogformer_loss_transductive_pseudo_labels(two_graph_batch):
    """Test pseudo-labels replace ground truth outside the training mask.

    Parameters
    ----------
    two_graph_batch : torch_geometric.data.Data
        Deterministic two-graph batch fixture.
    """
    train_mask = torch.tensor([True] * 5 + [False] * 5)
    two_graph_batch.train_mask = train_mask
    node_logits = torch.zeros(10, 3)
    node_logits[:, 2] = 1.0  # All nodes predicted as class 2
    model_out = {"node_logits": node_logits}

    y = OGFormerLoss._effective_labels(model_out, two_graph_batch)
    assert torch.equal(y[:5], two_graph_batch.y[:5])
    assert (y[5:] == 2).all()
    # The original labels are untouched
    assert not torch.equal(y, two_graph_batch.y)


def test_ogformer_loss_skips_non_node_labels(two_graph_batch):
    """Test that the homophily term is skipped without per-node labels.

    Parameters
    ----------
    two_graph_batch : torch_geometric.data.Data
        Deterministic two-graph batch fixture.
    """
    # Graph-level labels: one per graph
    two_graph_batch.y = torch.tensor([0, 1])
    assert OGFormerLoss._effective_labels({}, two_graph_batch) is None

    # Float (regression) labels
    two_graph_batch.y = torch.rand(10)
    assert OGFormerLoss._effective_labels({}, two_graph_batch) is None
