"""Test the DirichletLoss class."""

import pytest
import torch
import torch_geometric

# Import from the package (populated by the loss-discovery mechanism), not the
# submodule: `from ...model.DirichletLoss import DirichletLoss` would force-import
# the submodule and shadow the class registered on the package, breaking hydra's
# `_target_: topobench.loss.model.DirichletLoss` resolution in later tests.
from topobench.loss.model import DirichletLoss


def _make_inputs(N=3, r=2, d=4, edge_index=None, requires_grad=False):
    """Build a ``(model_out, batch)`` pair for the Dirichlet loss.

    Parameters
    ----------
    N : int, optional
        Number of nodes (default: 3).
    r : int, optional
        Number of frame vectors (default: 2).
    d : int, optional
        Embedding dimension (default: 4).
    edge_index : torch.Tensor or None, optional
        Edge index of shape ``[2, E]``. Defaults to a directed cycle over the
        ``N`` nodes so that every node has exactly one incoming edge.
    requires_grad : bool, optional
        Whether the embedding/frame tensors require gradients (default: False).

    Returns
    -------
    tuple of (dict, torch_geometric.data.Data)
        The mock model output and batch.
    """
    if edge_index is None:
        src = torch.arange(N)
        dst = torch.roll(src, -1)
        edge_index = torch.stack([src, dst], dim=0)

    model_out = {
        "x_0": torch.randn(N, d, requires_grad=requires_grad),
        "z_0": torch.randn(N, d, requires_grad=requires_grad),
        "Q": torch.randn(N, r, d, requires_grad=requires_grad),
    }
    batch = torch_geometric.data.Data(edge_index=edge_index, num_nodes=N)
    return model_out, batch


def test_dirichlet_loss_init():
    """Default hyperparameters are stored as given."""
    loss_fn = DirichletLoss()
    assert loss_fn.lamb == 0.1
    assert loss_fn.reduce == "mean"


def test_dirichlet_loss_init_invalid_reduction():
    """An unsupported reduction raises ``NotImplementedError``."""
    with pytest.raises(NotImplementedError):
        DirichletLoss(reduction="max")


def test_dirichlet_loss_repr():
    """The repr reports the configured hyperparameters."""
    assert repr(DirichletLoss()) == "DirichletLoss(lamb=0.1, reduction=mean)"


def test_dirichlet_loss_forward_is_nonnegative_scalar():
    """The forward pass returns a non-negative scalar tensor."""
    loss_fn = DirichletLoss()
    model_out, batch = _make_inputs()
    loss = loss_fn.forward(model_out, batch)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_dirichlet_loss_reductions_run(reduction):
    """Both supported reductions produce a valid scalar.

    Parameters
    ----------
    reduction : str
        The neighbor aggregation reduction to test.
    """
    loss_fn = DirichletLoss(reduction=reduction)
    model_out, batch = _make_inputs()
    loss = loss_fn.forward(model_out, batch)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_dirichlet_loss_lambda_scales_linearly():
    """The output scales linearly with ``lamb``."""
    model_out, batch = _make_inputs()
    base = DirichletLoss(lamb=0.1).forward(model_out, batch)
    scaled = DirichletLoss(lamb=0.5).forward(model_out, batch)
    assert torch.allclose(scaled, 5.0 * base)


def test_dirichlet_loss_zero_when_frames_and_embeddings_align():
    """Identical embeddings and frames over a cycle give zero loss.

    When every node shares the same embedding and the same frame, each
    projection is identical, so the neighbor-averaged projection of the current
    embedding equals the projection of the (equal) initial embedding, and the
    loss vanishes. The directed cycle guarantees every node has one incoming
    edge, so no node is left with an all-zero aggregate.
    """
    N, r, d = 3, 2, 4
    shared_emb = torch.ones(N, d)
    shared_frame = torch.randn(1, r, d).expand(N, r, d).contiguous()
    model_out = {"x_0": shared_emb, "z_0": shared_emb.clone(), "Q": shared_frame}

    src = torch.arange(N)
    edge_index = torch.stack([src, torch.roll(src, -1)], dim=0)
    batch = torch_geometric.data.Data(edge_index=edge_index, num_nodes=N)

    loss = DirichletLoss().forward(model_out, batch)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_dirichlet_loss_ignores_self_loops():
    """Adding a self-loop on every node leaves the loss unchanged.

    The loss strips self-loops from ``edge_index`` before aggregating over
    neighbors, so an otherwise identical batch augmented with self-loops must
    produce the same value as the original.
    """
    model_out, batch = _make_inputs()
    looped, _ = torch_geometric.utils.add_self_loops(
        batch.edge_index, num_nodes=batch.num_nodes
    )
    batch_looped = torch_geometric.data.Data(
        edge_index=looped, num_nodes=batch.num_nodes
    )

    loss_fn = DirichletLoss()
    base = loss_fn.forward(model_out, batch)
    looped_loss = loss_fn.forward(model_out, batch_looped)
    assert torch.allclose(base, looped_loss, atol=1e-6)


def test_dirichlet_loss_detaches_initial_embedding():
    """Gradients flow to ``x_0`` and ``Q`` but not to the detached ``z_0``."""
    model_out, batch = _make_inputs(requires_grad=True)
    loss = DirichletLoss().forward(model_out, batch)
    loss.backward()

    assert model_out["x_0"].grad is not None
    assert model_out["Q"].grad is not None
    # z_0 is used only as a detached target, so no gradient reaches it.
    assert model_out["z_0"].grad is None


def test_dirichlet_loss_one_isolated_node_finite():
    """A single isolated node among otherwise-connected nodes stays finite."""
    N = 3
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    model_out, batch = _make_inputs(N=N, edge_index=edge_index)
    loss = DirichletLoss().forward(model_out, batch)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_dirichlet_loss_all_nodes_isolated():
    """A zero-edge graph still produces a finite, non-negative loss."""
    N = 3
    edge_index = torch.tensor([[], []], dtype=torch.long)
    model_out, batch = _make_inputs(N=N, edge_index=edge_index)
    loss = DirichletLoss().forward(model_out, batch)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_dirichlet_loss_near_zero_projection_grad_finite():
    """Gradients stay finite when a node's projection is exactly zero."""
    model_out, batch = _make_inputs(requires_grad=True)
    with torch.no_grad():
        model_out["z_0"][0] = 0.0
    loss = DirichletLoss().forward(model_out, batch)
    loss.backward()
    assert torch.isfinite(model_out["Q"].grad).all()
    assert torch.isfinite(model_out["x_0"].grad).all()


def _cycle_edge_index(N, offset=0):
    """Build a directed cycle over ``N`` nodes, indices shifted by ``offset``.

    Parameters
    ----------
    N : int
        Number of nodes in the cycle.
    offset : int, optional
        Amount to shift node indices by, for packing into a larger batch
        (default: 0).

    Returns
    -------
    torch.Tensor
        Edge index of shape ``[2, N]``.
    """
    src = torch.arange(N) + offset
    dst = torch.roll(src, -1)
    return torch.stack([src, dst], dim=0)


def test_dirichlet_loss_no_cross_graph_leakage():
    """Perturbing one graph in a batch must not change another graph's gradient.

    Two independent cycle graphs are packed into a single batch, Since the final reduction
    is a plain mean over all nodes, graph B's gradient should be identical
    regardless of what graph A's embeddings are.
    """
    N_a, N_b, r, d = 3, 3, 2, 4
    N = N_a + N_b
    edge_index = torch.cat(
        [_cycle_edge_index(N_a), _cycle_edge_index(N_b, offset=N_a)], dim=1
    )
    batch = torch_geometric.data.Data(edge_index=edge_index, num_nodes=N)

    x_0 = torch.randn(N, d, requires_grad=True)
    z_0 = torch.randn(N, d)
    Q = torch.randn(N, r, d, requires_grad=True)
    model_out = {"x_0": x_0, "z_0": z_0, "Q": Q}
    loss = DirichletLoss().forward(model_out, batch)
    loss.backward()
    grad_b_before = Q.grad[N_a:].clone()

    # Perturb graph A's embeddings only; graph B's data is untouched.
    x_0b = x_0.detach().clone()
    x_0b[:N_a] = torch.randn(N_a, d)
    x_0b.requires_grad_(True)
    Qb = Q.detach().clone().requires_grad_(True)
    model_out_2 = {"x_0": x_0b, "z_0": z_0, "Q": Qb}
    loss2 = DirichletLoss().forward(model_out_2, batch)
    loss2.backward()
    grad_b_after = Qb.grad[N_a:]

    assert torch.allclose(grad_b_before, grad_b_after)


def test_dirichlet_loss_reductions_actually_differ():
    """``sum`` and ``mean`` reductions must produce different losses."""
    N, r, d = 3, 2, 4
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    model_out, batch = _make_inputs(N=N, r=r, d=d, edge_index=edge_index)

    loss_mean = DirichletLoss(reduction="mean").forward(model_out, batch)
    loss_sum = DirichletLoss(reduction="sum").forward(model_out, batch)

    assert not torch.allclose(loss_mean, loss_sum)
