"""Unit tests for SheafTSP.

Tests the SheafTSP backbone, SheafConvLayer, RestrictionMapLearner,
and Sheaf Laplacian builders for correctness and shape consistency.
"""

import pytest
import torch

from topobench.nn.backbones.cell.sheaf_tsp import (
    RestrictionMapLearner,
    SheafConvLayer,
    SheafDirichletLoss,
    SheafTSP,
    build_sheaf_laplacian_sparse,
    build_sheaf_laplacian_torch,
)


@pytest.fixture
def cell_graph_input():
    """Generate a small cell complex test input.

    Returns
    -------
    tuple
        (x_1, edge_index_Ld, edge_index_Lu) representing 1-cell features
        and sparse down/up Laplacians.
    """
    torch.manual_seed(42)
    N = 20  # 1-cells
    C = 16  # feature channels

    x = torch.randn(N, C)

    # Synthetic Laplacians as sparse tensors (symmetric, sparse)
    # Down Laplacian: some connectivity pattern
    edges_d = torch.tensor(
        [
            [0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 0, 1, 2, 3, 6, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    vals_d = torch.ones(edges_d.shape[1])
    Ld = torch.sparse_coo_tensor(edges_d, vals_d, (N, N)).coalesce()

    # Up Laplacian
    edges_u = torch.tensor(
        [
            [0, 2, 4, 6, 8, 10, 12, 14, 2, 4, 6, 8, 10, 12, 14, 16],
            [2, 4, 6, 8, 10, 12, 14, 16, 0, 2, 4, 6, 8, 10, 12, 14],
        ],
        dtype=torch.long,
    )
    vals_u = torch.ones(edges_u.shape[1])
    Lu = torch.sparse_coo_tensor(edges_u, vals_u, (N, N)).coalesce()

    return x, Ld, Lu


def test_restriction_map_learner():
    """Test RestrictionMapLearner produces orthogonal d×d matrices."""
    torch.manual_seed(42)
    in_ch = 16
    d = 2
    N = 10
    E = 15

    learner = RestrictionMapLearner(in_ch, d)
    x = torch.randn(N, in_ch)
    edge_index = torch.randint(0, N, (2, E))

    R = learner(x, edge_index)
    assert R.shape == (E, d, d), f"Expected (E, d, d), got {R.shape}"

    # Check orthogonality: R^T R ≈ I
    RtR = torch.bmm(R.transpose(1, 2), R)
    eye = torch.eye(d).unsqueeze(0).expand(E, -1, -1)
    orth_err = (RtR - eye).abs().max().item()
    assert orth_err < 1e-4, f"Restriction maps not orthogonal: err={orth_err}"


def test_sheaf_laplacian_dense():
    """Test dense Sheaf Laplacian is symmetric PSD."""
    torch.manual_seed(42)
    N = 8
    d = 2
    E = 7
    edge_index = torch.tensor(
        [[i, i + 1] for i in range(E)], dtype=torch.long
    ).T
    R = torch.eye(d).unsqueeze(0).expand(E, -1, -1)  # trivial maps

    L = build_sheaf_laplacian_torch(N, edge_index, R, d)
    assert L.shape == (N * d, N * d)

    # Symmetric
    sym_err = (L - L.T).abs().max().item()
    assert sym_err < 1e-10, f"Not symmetric: err={sym_err}"

    # PSD
    eigs = torch.linalg.eigvalsh(L)
    assert eigs[0] >= -1e-6, f"Not PSD: min eig={eigs[0]}"


def test_sheaf_laplacian_sparse_matches_dense():
    """Test sparse Laplacian matches dense construction."""
    torch.manual_seed(42)
    N = 10
    d = 2
    E = 9
    edge_index = torch.tensor(
        [[i, i + 1] for i in range(E)], dtype=torch.long
    ).T
    # Random orthogonal restriction maps
    learner = RestrictionMapLearner(8, d)
    x = torch.randn(N, 8)
    R = learner(x, edge_index)

    L_dense = build_sheaf_laplacian_torch(N, edge_index, R, d)
    L_sparse = build_sheaf_laplacian_sparse(N, edge_index, R, d)
    L_sparse_dense = L_sparse.to_dense()

    diff = (L_dense - L_sparse_dense).abs().max().item()
    assert diff < 1e-6, f"Sparse/dense mismatch: {diff}"


def test_sheaf_conv_layer():
    """Test SheafConvLayer forward pass shape."""
    torch.manual_seed(42)
    N = 15
    in_ch = 16
    out_ch = 16
    d = 2

    layer = SheafConvLayer(in_ch, out_ch, stalk_dim=d, filter_order=3)
    x = torch.randn(N, in_ch)
    edge_index = torch.tensor(
        [[i, i + 1] for i in range(N - 1)], dtype=torch.long
    ).T

    y = layer(x, edge_index)
    assert y.shape == (N, out_ch), f"Expected ({N}, {out_ch}), got {y.shape}"


def test_sheaf_tsp_backbone(cell_graph_input):
    """Test full SheafTSP backbone forward pass."""
    x, Ld, Lu = cell_graph_input
    N, C = x.shape

    model = SheafTSP(
        in_channels=C, n_layers=2, stalk_dim=2, filter_order=2, dropout=0.0
    )
    y = model(x, Ld, Lu)
    assert y.shape == (N, C), f"Expected ({N}, {C}), got {y.shape}"


def test_sheaf_tsp_backward(cell_graph_input):
    """Test backward pass produces valid gradients."""
    x, Ld, Lu = cell_graph_input
    model = SheafTSP(in_channels=x.shape[1], n_layers=2, stalk_dim=2)

    y = model(x, Ld, Lu)
    loss = y.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_sheaf_tsp_no_crash_single_node():
    """Test edge case: single node graph."""
    x = torch.randn(1, 8)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    Ld = torch.sparse_coo_tensor(edge_index, torch.zeros(0), (1, 1))
    Lu = torch.sparse_coo_tensor(edge_index, torch.zeros(0), (1, 1))

    model = SheafTSP(in_channels=8, n_layers=1, stalk_dim=2)
    y = model(x, Ld, Lu)
    assert y.shape == (1, 8)


def test_weighted_laplacian_sparse_matches_dense():
    """Test kernel-weighted sparse Laplacian matches dense version."""
    torch.manual_seed(42)
    N = 10
    d = 2
    E = 9
    edge_index = torch.tensor(
        [[i, i + 1] for i in range(E)], dtype=torch.long
    ).T
    learner = RestrictionMapLearner(8, d)
    x = torch.randn(N, 8)
    R = learner(x, edge_index)
    k = torch.rand(E) + 0.1  # positive kernel weights

    L_dense = build_sheaf_laplacian_torch(N, edge_index, R, d, edge_weights=k)
    L_sparse = build_sheaf_laplacian_sparse(
        N, edge_index, R, d, edge_weights=k
    ).to_dense()

    diff = (L_dense - L_sparse).abs().max().item()
    assert diff < 1e-6, f"Weighted sparse/dense mismatch: {diff}"

    # Weighted Laplacian must remain symmetric PSD
    sym_err = (L_dense - L_dense.T).abs().max().item()
    assert sym_err < 1e-5, f"Not symmetric: err={sym_err}"
    eigs = torch.linalg.eigvalsh(L_dense)
    assert eigs[0] >= -1e-5, f"Not PSD: min eig={eigs[0]}"


def test_edge_dedup_ignores_self_loops(cell_graph_input):
    """Diagonal (self-loop) Laplacian entries must not change output."""
    x, Ld, Lu = cell_graph_input
    N = x.shape[0]

    # Add explicit diagonal (degree) entries to the down Laplacian,
    # as real Hodge Laplacians have.
    diag = torch.arange(N)
    diag_idx = torch.stack([diag, diag])
    Ld_diag = (
        Ld + torch.sparse_coo_tensor(diag_idx, torch.full((N,), 2.0), (N, N))
    ).coalesce()

    model = SheafTSP(in_channels=x.shape[1], n_layers=2, stalk_dim=2)
    model.eval()
    with torch.no_grad():
        y_plain = model(x, Ld, Lu)
        y_diag = model(x, Ld_diag, Lu)

    diff = (y_plain - y_diag).abs().max().item()
    assert diff < 1e-6, f"Self-loops leaked into the sheaf: {diff}"


def test_dirichlet_regularizer(cell_graph_input):
    """Backbone exposes Dirichlet energy; loss term applies weight."""
    x, Ld, Lu = cell_graph_input
    model = SheafTSP(in_channels=x.shape[1], n_layers=2, stalk_dim=2)
    model.train()
    model(x, Ld, Lu)

    reg = model.dirichlet_energy
    assert reg is not None
    assert reg.dim() == 0, "Dirichlet energy must be a scalar"
    assert torch.isfinite(reg), "Dirichlet energy must be finite"
    assert reg >= -1e-6, "Dirichlet energy of PSD Laplacian must be >= 0"

    class _Batch:
        y = torch.zeros(3)

    loss_fn = SheafDirichletLoss(loss_weight=0.01)
    val = loss_fn({"sheaf_dirichlet": reg}, _Batch())
    assert torch.allclose(val, 0.01 * reg)

    # Missing key (validation/test) → zero loss
    val0 = loss_fn({}, _Batch())
    assert val0.item() == 0.0


def test_chebyshev_filter_basis(cell_graph_input):
    """Chebyshev basis: shape, gradients, init-equivalence, validation."""
    x, Ld, Lu = cell_graph_input
    N, C = x.shape

    # Forward + backward with the Chebyshev basis
    torch.manual_seed(0)
    model_cheb = SheafTSP(
        in_channels=C,
        n_layers=2,
        stalk_dim=2,
        filter_order=3,
        filter_basis="chebyshev",
    )
    y = model_cheb(x, Ld, Lu)
    assert y.shape == (N, C), f"Expected ({N}, {C}), got {y.shape}"
    y.sum().backward()
    for name, param in model_cheb.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Bad gradient in {name}"

    # At init W_{k>0} = 0, so both bases reduce to s @ W_0 and must
    # produce identical outputs given identical initialization.
    torch.manual_seed(0)
    model_mono = SheafTSP(
        in_channels=C,
        n_layers=2,
        stalk_dim=2,
        filter_order=3,
        filter_basis="monomial",
    )
    model_cheb.eval()
    model_mono.eval()
    with torch.no_grad():
        y_c = model_cheb(x, Ld, Lu)
        y_m = model_mono(x, Ld, Lu)
    diff = (y_c - y_m).abs().max().item()
    assert diff < 1e-6, f"Bases differ at init: {diff}"

    # With nonzero higher-order weights the bases must diverge.
    with torch.no_grad():
        for m in (model_cheb, model_mono):
            for layer in m.layers:
                layer.filter_weights[1:].fill_(0.1)
        y_c = model_cheb(x, Ld, Lu)
        y_m = model_mono(x, Ld, Lu)
    assert (y_c - y_m).abs().max().item() > 1e-4, (
        "Chebyshev and monomial bases should differ with nonzero "
        "higher-order filter weights"
    )

    # Invalid basis name must raise
    with pytest.raises(ValueError, match="filter_basis"):
        SheafConvLayer(C, C, stalk_dim=2, filter_basis="fourier")


def test_exp_rotation_param():
    """Exponential-map projection produces proper rotations."""
    torch.manual_seed(0)
    d = 2
    E = 12
    learner = RestrictionMapLearner(8, d, rotation_param="exp")
    x = torch.randn(10, 8)
    edge_index = torch.randint(0, 10, (2, E))

    R = learner(x, edge_index)
    RtR = torch.bmm(R.transpose(1, 2), R)
    eye = torch.eye(d).unsqueeze(0).expand(E, -1, -1)
    assert (RtR - eye).abs().max().item() < 1e-4
    # Proper rotations: det = +1
    dets = torch.linalg.det(R)
    assert (dets - 1.0).abs().max().item() < 1e-4

    with pytest.raises(ValueError, match="rotation_param"):
        RestrictionMapLearner(8, d, rotation_param="householder")


def test_trivial_stalk_dim_one():
    """d = 1 stalks reduce restriction maps to identity scalars."""
    torch.manual_seed(0)
    learner = RestrictionMapLearner(8, 1)
    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    R = learner(x, edge_index)
    assert R.shape == (3, 1, 1)
    assert torch.allclose(R, torch.ones(3, 1, 1))

    layer = SheafConvLayer(8, 8, stalk_dim=1, filter_order=2)
    y = layer(x, edge_index)
    assert y.shape == (6, 8)


def test_submission_configuration(cell_graph_input):
    """The shipped configuration trains end to end.

    PPR filter basis on the sheaf lazy walk, transport-consistency
    kernel, alignment regularizer, and Cayley transports — the exact
    combination in ``configs/model/cell/sheaf_tsp.yaml``.
    """
    x, Ld, Lu = cell_graph_input
    N, C = x.shape
    model = SheafTSP(
        in_channels=C,
        n_layers=3,
        stalk_dim=2,
        filter_order=10,
        mlp_dropout=0.5,
        filter_basis="ppr",
        kernel_distance="transport",
        reg_form="alignment",
        rotation_param="cayley",
    )
    model.train()
    y = model(x, Ld, Lu)
    assert y.shape == (N, C)
    assert torch.isfinite(y).all()

    # Alignment reward is bounded per layer: -mean(exp(-rho^2/4t))
    # lies in [-1, 0], and the backbone sums it over n_layers.
    reg = model.dirichlet_energy
    assert reg is not None and torch.isfinite(reg)
    assert -3.0 - 1e-6 <= reg.item() <= 0.0

    (y.sum() + reg).backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Bad gradient in {name}"


def test_constructor_validation():
    """Invalid layer options raise ValueError."""
    with pytest.raises(ValueError, match="filter_order"):
        SheafConvLayer(8, 8, stalk_dim=2, filter_order=0)
    with pytest.raises(ValueError, match="kernel_distance"):
        SheafConvLayer(8, 8, stalk_dim=2, kernel_distance="cosine")
    with pytest.raises(ValueError, match="reg_form"):
        SheafConvLayer(8, 8, stalk_dim=2, reg_form="ridge")


def test_sparse_autoselect_path():
    """Large complexes route through the sparse Laplacian assembly."""
    torch.manual_seed(0)
    N = 1200  # N * d = 2400 > 2000 → sparse path
    C = 4
    x = torch.randn(N, C)
    edge_index = torch.stack([torch.arange(N - 1), torch.arange(1, N)])
    layer = SheafConvLayer(C, C, stalk_dim=2, filter_order=2)
    y = layer(x, edge_index)
    assert y.shape == (N, C)
    assert torch.isfinite(y).all()

    # Sparse path with the ppr basis and dirichlet regularizer too
    layer_ppr = SheafConvLayer(
        C, C, stalk_dim=2, filter_order=3, filter_basis="ppr"
    )
    layer_ppr.train()
    y = layer_ppr(x, edge_index)
    assert torch.isfinite(y).all()
    assert layer_ppr.last_dirichlet is not None


def test_global_context(cell_graph_input):
    """Zero-init global context is exact identity at initialization."""
    x, Ld, Lu = cell_graph_input
    C = x.shape[1]

    model_ctx = SheafTSP(in_channels=C, n_layers=2, global_context=True)
    model_plain = SheafTSP(in_channels=C, n_layers=2)
    # Same weights on the shared modules; gctx stays zero-init
    model_plain.load_state_dict(
        {
            k: v
            for k, v in model_ctx.state_dict().items()
            if k in model_plain.state_dict()
        }
    )

    model_ctx.eval()
    model_plain.eval()
    with torch.no_grad():
        diff = (
            (model_ctx(x, Ld, Lu) - model_plain(x, Ld, Lu)).abs().max().item()
        )
    assert diff < 1e-6, f"Zero-init context changed the output: {diff}"

    # A nonzero context weight must change the output
    with torch.no_grad():
        model_ctx.gctx[0].weight.fill_(0.1)
        diff = (
            (model_ctx(x, Ld, Lu) - model_plain(x, Ld, Lu)).abs().max().item()
        )
    assert diff > 1e-6, "Global context had no effect"


def test_dirichlet_loss_repr():
    """Loss module repr states the weight."""
    assert "0.01" in repr(SheafDirichletLoss(loss_weight=0.01))
