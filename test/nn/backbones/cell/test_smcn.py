"""Unit tests for the SMCN backbone."""

import pytest
import torch

from topobench.nn.backbones.cell.smcn import SMCN
from topobench.nn.backbones.cell.smcn_utils.layers import (
    BagInit,
    BagPool,
    CINBlock,
    SafeBatchNorm,
    SCLLayer,
    TwoCellInit,
)
from topobench.nn.backbones.cell.smcn_utils.structures import (
    adjacency_from_incidence,
    build_smcn_structures,
    hop_distance_buckets,
)
from topobench.nn.liftings.difflift import (
    CellScorer,
    DiffLift,
    DiffLiftEncoder,
    EdgeSampler,
)


def _two_triangles():
    """Two triangles sharing an edge: 4 nodes, 5 edges, 2 two-cells.

    Returns
    -------
    tuple
        Incidence matrices and batch vectors of a single graph.
    """
    i1 = torch.zeros(4, 5)
    for e, (a, b) in enumerate([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]):
        i1[a, e] = 1
        i1[b, e] = 1
    i2 = torch.zeros(5, 2)
    i2[[0, 1, 2], 0] = 1
    i2[[2, 3, 4], 1] = 1
    b0 = torch.zeros(4, dtype=torch.long)
    b1 = torch.zeros(5, dtype=torch.long)
    b2 = torch.zeros(2, dtype=torch.long)
    return i1, i2, b0, b1, b2


def _structures():
    """Structures of the two-triangle fixture.

    Returns
    -------
    SMCNStructures
        The assembled structures.
    """
    i1, i2, b0, b1, b2 = _two_triangles()
    return build_smcn_structures(
        i1.to_sparse(), i2.to_sparse(), b0, b1, b2
    )


def test_adjacency_pairs():
    """Adjacency pairs and bridges of the fixture are exact."""
    s = _structures()
    assert s.a01_pairs.shape[1] == 10  # 5 edges, both directions
    assert s.a12_pairs.shape[1] == 12  # 2 cells x 6 ordered pairs
    pairs = set(
        map(
            tuple,
            torch.cat([s.a01_pairs, s.a01_bridge.unsqueeze(0)]).t().tolist(),
        )
    )
    assert (0, 1, 0) in pairs and (1, 0, 0) in pairs
    assert (2, 3, 4) in pairs and (3, 2, 4) in pairs


def test_adjacency_invalid_incidence():
    """A node-edge incidence column with one entry is rejected."""
    bad = torch.zeros(3, 2)
    bad[0, 0] = 1
    bad[1, 0] = 1
    bad[2, 1] = 1  # dangling edge
    with pytest.raises(ValueError):
        adjacency_from_incidence(bad, expected_size=2)


def test_bag_layout_and_marking():
    """Bag rows follow row(u, e) = u * n1 + e with min-endpoint marking."""
    s = _structures()
    assert s.bag_low_index.tolist()[:5] == [0, 0, 0, 0, 0]
    assert s.bag_high_index.tolist()[:5] == [0, 1, 2, 3, 4]
    # row(u=3, e0=(0,1)): d(3,0)=2, d(3,1)=1 -> 1
    assert int(s.bag_marking[15]) == 1
    # row(u=0, e0=(0,1)): endpoint -> 0
    assert int(s.bag_marking[0]) == 0
    assert s.bag_low_adj_pairs.shape[1] == 50  # 10 pairs x 5 copies
    assert s.bag_inc_pairs.shape[1] == 50  # 10 incidences x 5 copies


def test_hop_distance_buckets():
    """Distances are exact up to the cutoff, then bucketed."""
    adj = torch.zeros(16, 16, dtype=torch.bool)
    for i in range(14):
        adj[i, i + 1] = adj[i + 1, i] = True
    dist = hop_distance_buckets(adj, max_dist=10)
    assert int(dist[0, 5]) == 5
    assert int(dist[0, 14]) == 10  # connected but farther
    assert int(dist[0, 15]) == 11  # disconnected


def test_batch_offsets():
    """Batched structures equal per-graph structures plus offsets."""
    i1, i2, b0, b1, b2 = _two_triangles()
    single = _structures()
    double = build_smcn_structures(
        torch.block_diag(i1, i1).to_sparse(),
        torch.block_diag(i2, i2).to_sparse(),
        torch.cat([b0, b0 + 1]),
        torch.cat([b1, b1 + 1]),
        torch.cat([b2, b2 + 1]),
    )
    assert torch.equal(double.bag_marking[:20], single.bag_marking)
    assert torch.equal(double.bag_marking[20:], single.bag_marking)
    assert torch.equal(
        double.bag_low_index[20:] - 4, single.bag_low_index
    )
    assert torch.equal(
        double.bag_inc_pairs[:, 50:] - 20, single.bag_inc_pairs
    )


def test_layers_shapes_and_grads():
    """Every layer produces the documented shapes and gradients."""
    s = _structures()
    dim = 16
    x_0 = torch.randn(4, dim, requires_grad=True)
    x_1 = torch.randn(5, dim)
    x_2 = torch.zeros(2, dim)
    x_2 = TwoCellInit(dim)(x_0, x_2, s.inc02_pairs)
    assert x_2.shape == (2, dim)
    x_0b, x_1b, x_2b = CINBlock(dim)(x_0, x_1, x_2, s)
    bag = BagInit(dim)(x_0b, x_1b, s)
    assert bag.shape == (20, dim)
    bag = SCLLayer(dim, 12, edge_dim=dim)(bag, x_1b, s)
    assert bag.shape == (20, 12)
    bag = SCLLayer(12, dim, edge_dim=dim)(bag, x_1b, s)
    p0, p1 = BagPool()(bag, s, 4, 5)
    assert p0.shape == (4, dim) and p1.shape == (5, dim)
    (p0.sum() + p1.sum()).backward()
    assert x_0.grad is not None and x_0.grad.abs().sum() > 0


def test_cin_block_max_rank():
    """The reduced block leaves higher ranks untouched."""
    s = _structures()
    dim = 8
    x = [torch.randn(n, dim) for n in (4, 5, 2)]
    block = CINBlock(dim, max_rank=0)
    y0, y1, y2 = block(*x, s)
    assert torch.equal(y1, x[1]) and torch.equal(y2, x[2])
    with pytest.raises(ValueError):
        CINBlock(dim, max_rank=3)


@pytest.mark.parametrize("learned_lifting", [False, True])
def test_backbone_forward_backward(learned_lifting):
    """The backbone runs and back-propagates in both variants."""
    i1, i2, b0, b1, b2 = _two_triangles()
    dim = 16
    model = SMCN(
        dim,
        sub_channels=12,
        n_scl_layers=2,
        learned_lifting=learned_lifting,
    )
    x_0 = torch.randn(4, dim, requires_grad=True)
    y0, y1, y2 = model(
        x_0,
        torch.randn(5, dim),
        torch.zeros(2, dim),
        i1.to_sparse(),
        i2.to_sparse(),
        b0,
        b1,
        b2,
    )
    assert y0.shape == (4, dim)
    assert y1.shape == (5, dim)
    assert y2.shape == (2, dim)
    (y0.sum() + y1.sum() + y2.sum()).backward()
    assert x_0.grad is not None and x_0.grad.abs().sum() > 0


def test_backbone_no_two_cells():
    """A tree graph (no cycles) runs through both variants."""
    i1 = torch.zeros(4, 3)
    for e, (a, b) in enumerate([(0, 1), (1, 2), (2, 3)]):
        i1[a, e] = 1
        i1[b, e] = 1
    for learned in (False, True):
        model = SMCN(8, n_scl_layers=2, learned_lifting=learned)
        _, _, y2 = model(
            torch.randn(4, 8),
            torch.randn(3, 8),
            torch.zeros(0, 8),
            i1.to_sparse(),
            torch.zeros(3, 0).to_sparse(),
            torch.zeros(4, dtype=torch.long),
            torch.zeros(3, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
        )
        assert y2.shape == (0, 8)


def test_safe_batch_norm_single_row():
    """Single rows pass through in training; larger batches normalize."""
    bn = SafeBatchNorm(8)
    bn.train()
    x1 = torch.randn(1, 8)
    assert torch.equal(bn(x1), x1)
    x4 = torch.randn(4, 8)
    ref = torch.nn.BatchNorm1d(8)
    ref.train()
    assert torch.allclose(bn(x4), ref(x4))


@pytest.mark.parametrize("learned_lifting", [False, True])
def test_backbone_single_two_cell_training(learned_lifting):
    """A batch whose graphs total a single 2-cell trains without error."""
    i1 = torch.zeros(3, 3)
    for e, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
        i1[a, e] = 1
        i1[b, e] = 1
    i2 = torch.ones(3, 1)
    model = SMCN(
        8, sub_channels=6, n_scl_layers=2, learned_lifting=learned_lifting
    )
    model.train()
    x_0 = torch.randn(3, 8, requires_grad=True)
    y0, y1, y2 = model(
        x_0,
        torch.randn(3, 8),
        torch.zeros(1, 8),
        i1.to_sparse(),
        i2.to_sparse(),
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
    )
    assert y2.shape == (1, 8)
    (y0.sum() + y1.sum() + y2.sum()).backward()
    assert x_0.grad is not None


def test_cell_scorer_rescue():
    """The scorer gates cells and the rescue keeps one per graph."""
    i1, i2, b0, b1, b2 = _two_triangles()
    s = build_smcn_structures(i1.to_sparse(), i2.to_sparse(), b0, b1, b2)
    z = torch.randn(4, 8, requires_grad=True)
    scorer = CellScorer(8)
    torch.nn.init.constant_(scorer.score[2].bias, -100.0)
    torch.nn.init.zeros_(scorer.score[2].weight)
    gate = scorer(z, s.inc02_pairs, b2)
    assert gate.shape == (2,)
    assert gate.detach().sum() >= 1.0
    gate.sum().backward()
    assert z.grad is not None


def test_backbone_invalid_scl_layers():
    """Fewer than two SCL layers is rejected."""
    with pytest.raises(ValueError):
        SMCN(8, n_scl_layers=1)


def test_cell_scorer_hypergraph_and_stochastic():
    """The scorer is domain-agnostic and supports Bernoulli sampling."""
    membership = torch.tensor(
        [[0, 1, 2, 1, 2, 3, 4], [0, 0, 0, 1, 1, 2, 2]]
    )
    hyperedge_batch = torch.tensor([0, 0, 1])
    z = torch.randn(5, 8, requires_grad=True)
    gate = CellScorer(8)(z, membership, hyperedge_batch)
    assert gate.shape == (3,)
    assert set(gate.detach().unique().tolist()) <= {0.0, 1.0}
    gate.sum().backward()
    assert z.grad is not None
    stochastic = CellScorer(8, stochastic=True, rescue=False).train()
    gate = stochastic(z.detach(), membership, hyperedge_batch)
    assert set(gate.detach().unique().tolist()) <= {0.0, 1.0}


def test_difflift_encoder_and_edge_sampler():
    """Learned edges are new, deduplicated, and carry gradients."""
    i1, _, b0, _, _ = _two_triangles()
    s = build_smcn_structures(
        i1.to_sparse(), torch.zeros(5, 0).to_sparse(), b0,
        torch.zeros(5, dtype=torch.long), torch.zeros(0, dtype=torch.long),
    )
    x = torch.randn(4, 6, requires_grad=True)
    z = DiffLiftEncoder(6, hidden_channels=16)(x, s.a01_pairs)
    assert z.shape == (4, 16)
    sampler = EdgeSampler(16, k_min=1, k_max=3)
    pairs, gate = sampler(z, s.a01_pairs, b0)
    assert pairs.shape[0] == 2 and gate.shape == (pairs.shape[1],)
    observed = {
        tuple(sorted(p)) for p in s.a01_pairs.t().tolist()
    }
    for a, b in pairs.t().tolist():
        assert (a, b) not in observed  # only new edges
    if gate.numel():
        gate.sum().backward()
        assert x.grad is not None
    with pytest.raises(ValueError):
        EdgeSampler(16, k_min=3, k_max=2)


def test_difflift_full_recipe():
    """The complete lifting returns a consistent gated complex."""
    i1, _, b0, _, _ = _two_triangles()
    s = build_smcn_structures(
        i1.to_sparse(), torch.zeros(5, 0).to_sparse(), b0,
        torch.zeros(5, dtype=torch.long), torch.zeros(0, dtype=torch.long),
    )
    x = torch.randn(4, 6, requires_grad=True)
    lift = DiffLift(6, hidden_channels=16, max_cell_length=6)
    out = lift(x, s.a01_pairs, b0)
    n_cells = out["cell_batch"].numel()
    assert out["cell_gate"].shape == (n_cells,)
    assert out["x_2"].shape == (n_cells, 6)
    assert out["cell_membership"].shape[0] == 2
    total = out["x_2"].sum() + out["cell_gate"].sum()
    if out["new_edge_gate"].numel():
        total = total + out["new_edge_gate"].sum()
    total.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
