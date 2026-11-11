"""Unit tests for the Cellular Transformer backbone."""

import pytest
import torch
import torch_geometric

from topobench.nn.backbones.cell import CellularTransformer
from topobench.nn.backbones.cell.cellular_transformer import (
    CT_ROUTES,
    CTFeedForward,
    CellularTransformerLayer,
    SparseCellAttention,
    random_walk_pe,
    sparse_row_normalize,
)
from topobench.nn.wrappers.cell import CellularTransformerWrapper


def _random_sparse(m, n, density=0.5, seed=0):
    """Create a random binary sparse COO matrix.

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns.
    density : float
        Approximate fraction of nonzero entries.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor of shape [m, n].
    """
    gen = torch.Generator().manual_seed(seed)
    dense = (torch.rand(m, n, generator=gen) < density).float()
    return dense.to_sparse_coo()


def _cell_inputs(n0=6, n1=8, n2=3, dim=16, seed=0):
    """Create random cell-complex tensors for testing.

    Parameters
    ----------
    n0 : int
        Number of rank-0 cells.
    n1 : int
        Number of rank-1 cells.
    n2 : int
        Number of rank-2 cells.
    dim : int
        Feature dimension.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keyword arguments for CellularTransformer.forward.
    """
    torch.manual_seed(seed)
    return {
        "x_0": torch.randn(n0, dim),
        "x_1": torch.randn(n1, dim),
        "x_2": torch.randn(n2, dim),
        "adjacency_0": _random_sparse(n0, n0, seed=seed),
        "coadjacency_1": _random_sparse(n1, n1, seed=seed + 1),
        "coadjacency_2": _random_sparse(n2, n2, seed=seed + 2),
        "incidence_1": _random_sparse(n0, n1, seed=seed + 3),
        "incidence_2": _random_sparse(n1, n2, seed=seed + 4),
    }


def testCellularTransformer():
    """Test the CellularTransformer forward pass output shapes."""
    inputs = _cell_inputs()
    model = CellularTransformer(16, 32, num_layers=2, num_heads=4)
    x_0, x_1, x_2 = model(**inputs)
    assert x_0.shape == (6, 32)
    assert x_1.shape == (8, 32)
    assert x_2.shape == (3, 32)


def testCellularTransformer_zero_pe():
    """Test the zero positional encoding variant."""
    inputs = _cell_inputs()
    model = CellularTransformer(
        16, 32, num_layers=1, num_heads=2, pe_type="zero"
    )
    x_0, x_1, x_2 = model(**inputs)
    assert x_0.shape == (6, 32)


def testCellularTransformer_empty_rank2():
    """Test graceful handling of complexes without rank-2 cells."""
    inputs = _cell_inputs()
    inputs["x_2"] = torch.zeros(0, 16)
    inputs["coadjacency_2"] = torch.zeros(0, 0).to_sparse_coo()
    inputs["incidence_2"] = torch.zeros(8, 0).to_sparse_coo()
    model = CellularTransformer(16, 32, num_layers=2, num_heads=4)
    x_0, x_1, x_2 = model(**inputs)
    assert x_0.shape == (6, 32)
    assert x_2.shape == (0, 32)


def testCellularTransformer_gradients():
    """Test gradient flow through attention, FFN and preprocessing."""
    inputs = _cell_inputs()
    model = CellularTransformer(16, 32, num_layers=1, num_heads=4)
    x_0, x_1, x_2 = model(**inputs)
    (x_0.sum() + x_1.sum() + x_2.sum()).backward()
    for lin in model.preprocess:
        assert lin.weight.grad is not None
    layer = model.layers[0]
    for ks, kt in CT_ROUTES:
        att = layer.attentions[f"{ks}_{kt}"]
        assert att.q.weight.grad is not None


def testCellularTransformer_invalid_args():
    """Test that invalid arguments raise ValueError."""
    with pytest.raises(ValueError):
        CellularTransformer(16, 32, pe_type="unsupported")
    with pytest.raises(ValueError):
        SparseCellAttention(30, num_heads=4)


def testCellularTransformerWrapper():
    """Test the CellularTransformerWrapper end to end on a fake batch."""
    torch.manual_seed(0)
    inputs = _cell_inputs()
    batch = torch_geometric.data.Data(
        x_0=inputs["x_0"],
        x_1=inputs["x_1"],
        x_2=inputs["x_2"],
        y=torch.zeros(1, dtype=torch.long),
        adjacency_0=inputs["adjacency_0"],
        coadjacency_1=inputs["coadjacency_1"],
        coadjacency_2=inputs["coadjacency_2"],
        incidence_1=inputs["incidence_1"],
        incidence_2=inputs["incidence_2"],
        batch_0=torch.zeros(6, dtype=torch.long),
    )
    model = CellularTransformer(16, 16, num_layers=1, num_heads=2)
    wrapper = CellularTransformerWrapper(
        model, **{"out_channels": 16, "num_cell_dimensions": 3}
    )
    _ = wrapper.__repr__()
    model_out = wrapper(batch)
    assert model_out["x_0"].shape == (6, 16)
    assert model_out["x_1"].shape == (8, 16)
    assert model_out["x_2"].shape == (3, 16)


def testSparseCellAttention():
    """Test masked attention: no in-edges yields zero output."""
    torch.manual_seed(0)
    att = SparseCellAttention(8, num_heads=2)
    x_t = torch.randn(3, 8)
    x_s = torch.randn(4, 8)
    # Only targets 0 and 2 have neighbors; target 1 has none.
    idx = torch.tensor([[0, 0, 2], [0, 3, 1]])
    neigh = torch.sparse_coo_tensor(idx, torch.ones(3), (3, 4))
    out = att(x_t, x_s, neigh)
    assert out.shape == (3, 8)
    assert torch.equal(out[1], torch.zeros(8))
    assert out[0].abs().sum() > 0

    # Empty neighborhood -> all-zero output.
    empty = torch.zeros(3, 4).to_sparse_coo()
    assert torch.equal(att(x_t, x_s, empty), torch.zeros(3, 8))


def testSparseCellAttention_convex_combination():
    """Attention outputs are convex combinations of the value vectors.

    With identical source features, every attended target must output
    exactly the (shared) value vector — this pins the masked-softmax
    normalization (weights summing to one per target).
    """
    torch.manual_seed(0)
    att = SparseCellAttention(8, num_heads=2, att_dropout=0.0)
    att.eval()
    x_t = torch.randn(3, 8)
    x_s = torch.randn(1, 8).repeat(4, 1)  # all sources identical
    idx = torch.tensor([[0, 0, 0, 2], [0, 1, 3, 2]])
    neigh = torch.sparse_coo_tensor(idx, torch.ones(4), (3, 4))
    out = att(x_t, x_s, neigh)
    expected = att.v(x_s[:1]).squeeze(0)
    # Target 0 (three neighbors) and target 2 (one neighbor) both
    # receive exactly the shared value vector.
    assert torch.allclose(out[0], expected, atol=1e-5)
    assert torch.allclose(out[2], expected, atol=1e-5)
    assert torch.equal(out[1], torch.zeros(8))


def testCellularTransformer_block_diagonal_batching():
    """Batched block-diagonal complexes equal independent forwards.

    Sparse neighborhood-masked attention must not leak information
    across complexes batched block-diagonally (TopoBench batching).
    """
    torch.manual_seed(0)
    inputs = _cell_inputs()
    model = CellularTransformer(
        16, 32, num_layers=2, num_heads=4, att_dropout=0.0
    )
    model.eval()
    with torch.no_grad():
        single = model(**inputs)

        def block(mat):
            dense = mat.to_dense()
            zeros_tl = torch.zeros_like(dense)
            top = torch.cat([dense, zeros_tl], dim=1)
            bottom = torch.cat([zeros_tl, dense], dim=1)
            return torch.cat([top, bottom], dim=0).to_sparse_coo()

        doubled = model(
            x_0=torch.cat([inputs["x_0"]] * 2),
            x_1=torch.cat([inputs["x_1"]] * 2),
            x_2=torch.cat([inputs["x_2"]] * 2),
            adjacency_0=block(inputs["adjacency_0"]),
            coadjacency_1=block(inputs["coadjacency_1"]),
            coadjacency_2=block(inputs["coadjacency_2"]),
            incidence_1=block(inputs["incidence_1"]),
            incidence_2=block(inputs["incidence_2"]),
        )
    for one, two in zip(single, doubled, strict=True):
        n = one.shape[0]
        assert torch.allclose(one, two[:n], atol=1e-5)
        assert torch.allclose(one, two[n:], atol=1e-5)


def testCellularTransformer_precomputed_pe_equivalence():
    """Precomputed RWPE (fast path) must equal on-the-fly computation.

    The CellRandomWalkPE transform precomputes the encodings at
    preprocessing time; this pins the invariant that both code paths
    produce identical outputs.
    """
    torch.manual_seed(0)
    inputs = _cell_inputs()
    model = CellularTransformer(
        16, 32, num_layers=2, num_heads=4, att_dropout=0.0
    )
    model.eval()
    from topobench.nn.backbones.cell.cellular_transformer import (
        random_walk_pe,
    )

    pes = [
        random_walk_pe(inputs["adjacency_0"], model.pe_steps),
        random_walk_pe(inputs["coadjacency_1"], model.pe_steps),
        random_walk_pe(inputs["coadjacency_2"], model.pe_steps),
    ]
    with torch.no_grad():
        on_the_fly = model(**inputs)
        precomputed = model(
            **inputs, rwpe_0=pes[0], rwpe_1=pes[1], rwpe_2=pes[2]
        )
    for a, b in zip(on_the_fly, precomputed, strict=True):
        assert torch.allclose(a, b, atol=1e-6)


def testCellularTransformerLayer():
    """Test that a single layer preserves shapes across ranks."""
    torch.manual_seed(0)
    inputs = _cell_inputs(dim=16)
    layer = CellularTransformerLayer(16, num_heads=2, dropout=0.1)
    neighborhoods = {
        "0_0": inputs["adjacency_0"],
        "1_1": inputs["coadjacency_1"],
        "2_2": inputs["coadjacency_2"],
        "1_0": inputs["incidence_1"],
        "0_1": inputs["incidence_1"].transpose(0, 1),
        "2_1": inputs["incidence_2"],
        "1_2": inputs["incidence_2"].transpose(0, 1),
    }
    xs = layer(
        [inputs["x_0"], inputs["x_1"], inputs["x_2"]], neighborhoods
    )
    assert [x.shape for x in xs] == [(6, 16), (8, 16), (3, 16)]


def testCTFeedForward():
    """Test the feed-forward block shape and dropout path."""
    torch.manual_seed(0)
    ffn = CTFeedForward(8, 8, dropout=0.5)
    out = ffn(torch.randn(5, 8))
    assert out.shape == (5, 8)


def testRandomWalkPE():
    """Test RWPe return probabilities on a 2-node path graph."""
    # Path graph on 2 nodes: P = [[0, 1], [1, 0]].
    idx = torch.tensor([[0, 1], [1, 0]])
    adj = torch.sparse_coo_tensor(idx, torch.ones(2), (2, 2))
    pe = random_walk_pe(adj, steps=3)
    # diag(P) = 0, diag(P^2) = 1, diag(P^3) = 0.
    expected = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    assert torch.allclose(pe, expected)

    # Empty matrix and empty rank behave gracefully.
    empty = torch.zeros(3, 3).to_sparse_coo()
    assert torch.equal(random_walk_pe(empty, 2), torch.zeros(3, 2))
    none = torch.zeros(0, 0).to_sparse_coo()
    assert random_walk_pe(none, 2).shape == (0, 2)


def testSparseRowNormalize():
    """Test that rows of the random walk operator sum to one."""
    matrix = _random_sparse(5, 5, density=0.6, seed=3)
    walk = sparse_row_normalize(matrix)
    row_sums = torch.zeros(5)
    walk = walk.coalesce()
    row_sums.index_add_(0, walk.indices()[0], walk.values())
    dense_rows = matrix.to_dense().sum(dim=1)
    for i in range(5):
        if dense_rows[i] > 0:
            assert abs(row_sums[i].item() - 1.0) < 1e-6
        else:
            assert row_sums[i].item() == 0.0

    # All-zero matrix passes through unchanged.
    zero = torch.zeros(4, 4).to_sparse_coo()
    assert sparse_row_normalize(zero)._nnz() == 0
