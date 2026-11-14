"""Tests for the Directional Sheaf Hypergraph Network backbone.

Beyond the usual shape and reproducibility checks, these tests pin down the
mathematics of Mule et al., ICLR 2026 (https://arxiv.org/abs/2510.04727):
Theorems 1-4, Corollary 1, the Dirichlet energy of Theorem 2, and the
Appendix E counterexample that separates this operator from the Sheaf
Hypergraph Laplacian of Duta et al. (2023).
"""

import math

import pytest
import torch

from test._utils.nn_module_auto_test import NNModuleAutoTest
from topobench.nn.backbones.hypergraph.dshn import DSHN
from topobench.nn.backbones.hypergraph.dshn_utils.complex_ops import (
    ComplexLayerNorm,
    RealLinear,
    complex_dropout,
    complex_relu,
    unwind,
)
from topobench.nn.backbones.hypergraph.dshn_utils.laplacian import (
    block_inv_sqrt,
    charge_phase,
    derive_orientation,
    directed_sheaf_laplacian,
    incidence_pairs,
    restriction_degrees,
)
from topobench.nn.backbones.hypergraph.dshn_utils.sheaf_builders import (
    SHEAF_BUILDERS,
    DiagSheafBuilder,
    GeneralSheafBuilder,
    OrthoSheafBuilder,
    _activate,
)

TOL = 1e-9

# V = {0,1,2,3}, E = {e_1 = {0,1,2}, e_2 = {1,2,3}} -- the Appendix E example.
APPENDIX_E = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 1)]


def _edge_index(pairs):
    """Build a [2, nnz] incidence index from (node, hyperedge) pairs."""
    return torch.tensor(pairs, dtype=torch.long).T


def _trivial_sheaf(nnz, d=1):
    """Restriction maps all equal to the identity (F = 1 in the paper)."""
    return torch.eye(d, dtype=torch.float64).expand(nnz, d, d).contiguous()


def _laplacian(pairs, n, m, d=1, f=None, q=0.0, is_head=None, **kwargs):
    """Assemble a dense Laplacian for a small hypergraph, in float64."""
    ei = _edge_index(pairs)
    nnz = ei.size(1)
    blocks = _trivial_sheaf(nnz, d) if f is None else f.double()
    if is_head is None:
        is_head = torch.zeros(nnz, dtype=torch.bool)
    phase = charge_phase(is_head, q, dtype=torch.cdouble)
    lap = directed_sheaf_laplacian(
        ei, blocks, phase, n, m, **kwargs
    )
    return lap.to_dense()


def _diag_blocks(dense, n, d):
    """Extract the n block-diagonal d x d blocks of a dense [nd, nd] matrix."""
    view = dense.view(n, d, n, d).permute(0, 2, 1, 3)
    return torch.diagonal(view, dim1=0, dim2=1).permute(2, 0, 1)


def _random_hypergraph(n=12, m=7, lo=2, hi=6, seed=1):
    """Sample (node, hyperedge) pairs for a random hypergraph.

    Every node is guaranteed to belong to at least one hyperedge. Isolated
    nodes have ``D_u = 0``, and since ``block_inv_sqrt`` uses the
    pseudo-inverse convention they drop out of the normalized operator
    entirely -- which breaks ``I - Q_N = D^{-1/2} L D^{-1/2}``. That behaviour
    is deliberate and is covered by
    :func:`test_isolated_node_drops_out_of_the_operator`; the theorems are
    stated for hypergraphs that actually cover their vertex set.
    """
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for e in range(m):
        size = int(torch.randint(lo, hi, (1,), generator=g))
        for v in torch.randperm(n, generator=g)[:size].tolist():
            pairs.append((v, e))
    covered = {v for v, _ in pairs}
    for v in range(n):
        if v not in covered:
            pairs.append((v, int(torch.randint(m, (1,), generator=g))))
    return pairs, g


def _star_hypergraph(n=12, stride=3):
    """Node-centred hypergraph: hyperedge j is centred on node j."""
    pairs = []
    for v in range(n):
        pairs.append((v, v))
        pairs.extend(
            (u, v) for u in range(n) if u != v and (u + v) % stride == 0
        )
    return pairs


# --------------------------------------------------------------------------
# The operator: theorems and the Appendix E counterexample
# --------------------------------------------------------------------------


def test_appendix_e_counterexample():
    """DSHN is PSD exactly where the Duta et al. operator is not.

    Appendix E builds V = {0,1,2,3}, E = {{0,1,2},{1,2,3}} with a trivial
    sheaf and shows the Laplacian of Duta et al. (2023) has eigenvalues
    {4/3, 1/3, (1 +- sqrt(17))/6}, so it is not positive semidefinite. The two
    operators differ only in the diagonal coefficient: 1 - 1/delta_e here
    against 1/delta_e there.
    """
    lap = _laplacian(APPENDIX_E, 4, 2, normalized=False).real
    assert torch.linalg.eigvalsh(lap).min() >= -TOL

    # Both hyperedges have delta_e = 3 and the sheaf is trivial, so our
    # diagonal is (1 - 1/3) = 2/3 per incident hyperedge.
    assert torch.allclose(
        torch.diagonal(lap),
        torch.tensor([2 / 3, 4 / 3, 4 / 3, 2 / 3], dtype=torch.float64),
    )

    # The Duta et al. operator shares our off-diagonal blocks exactly and
    # differs only by using 1/delta_e on the diagonal.
    off_diagonal = lap - torch.diag(torch.diagonal(lap))
    duta = off_diagonal + torch.diag(
        torch.tensor([1 / 3, 2 / 3, 2 / 3, 1 / 3], dtype=torch.float64)
    )

    expected = torch.tensor(
        [
            [1 / 3, -1 / 3, -1 / 3, 0.0],
            [-1 / 3, 2 / 3, -2 / 3, -1 / 3],
            [-1 / 3, -2 / 3, 2 / 3, -1 / 3],
            [0.0, -1 / 3, -1 / 3, 1 / 3],
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(duta, expected, atol=1e-12)

    eig = torch.linalg.eigvalsh(expected)
    assert eig.min() < -TOL, "the Duta operator must not be PSD here"
    assert math.isclose(
        eig.min().item(), (1 - math.sqrt(17)) / 6, abs_tol=1e-12
    )


def test_diagonal_coefficient():
    """The diagonal is sum_e (1 - 1/delta_e) F^T F (Eq. 3)."""
    pairs, g = _random_hypergraph()
    ei = _edge_index(pairs)
    d = 3
    f = torch.randn(ei.size(1), d, d, generator=g, dtype=torch.float64)
    lap = _laplacian(pairs, 12, 7, d=d, f=f, normalized=False)

    delta = torch.bincount(ei[1], minlength=7)[ei[1]].double()
    contrib = (1.0 - 1.0 / delta).view(-1, 1, 1) * torch.bmm(
        f.transpose(1, 2), f
    )
    expected = torch.zeros(12, d, d, dtype=torch.float64)
    expected.index_add_(0, ei[0], contrib)

    got = _diag_blocks(lap.real, 12, d)
    assert torch.allclose(got, expected, atol=1e-10)


def test_theorem_1_hermitian():
    """L_N^F is Hermitian, hence diagonalizable with real eigenvalues."""
    pairs = _star_hypergraph()
    n = 12
    g = torch.Generator().manual_seed(3)
    ei = _edge_index(pairs)
    f = torch.randn(ei.size(1), 2, 2, generator=g, dtype=torch.float64)
    head = derive_orientation(ei, n, n, "star")
    q_n = _laplacian(pairs, n, n, d=2, f=f, q=0.25, is_head=head)
    lap = torch.eye(n * 2, dtype=torch.cdouble) - q_n
    assert (lap - lap.conj().T).abs().max() < 1e-10


def test_corollary_1_psd_and_theorem_3_upper_bound():
    """spec(L_N^F) is contained in [0, 1] (Corollary 1 and Theorem 3)."""
    for seed in range(5):
        pairs, g = _random_hypergraph(seed=seed)
        ei = _edge_index(pairs)
        f = torch.randn(ei.size(1), 2, 2, generator=g, dtype=torch.float64)
        q_n = _laplacian(pairs, 12, 7, d=2, f=f)
        lap = torch.eye(24, dtype=torch.cdouble) - q_n
        eig = torch.linalg.eigvalsh(lap)
        assert eig.min() >= -1e-9, f"not PSD: {eig.min()}"
        assert eig.max() <= 1.0 + 1e-9, f"lambda_max > 1: {eig.max()}"


def test_theorem_2_dirichlet_energy():
    """x^dagger L_N^F x matches the explicit sum-of-squares of Theorem 2."""
    pairs, g = _random_hypergraph(n=8, m=4, seed=7)
    n, m, d = 8, 4, 2
    ei = _edge_index(pairs)
    f = torch.randn(ei.size(1), d, d, generator=g, dtype=torch.float64)
    head = torch.zeros(ei.size(1), dtype=torch.bool)
    phase = charge_phase(head, 0.0, dtype=torch.cdouble)

    q_n = directed_sheaf_laplacian(ei, f, phase, n, m).to_dense()
    lap = torch.eye(n * d, dtype=torch.cdouble) - q_n
    x = torch.randn(n * d, generator=g, dtype=torch.cdouble)
    energy = (x.conj() @ lap @ x).real

    d_inv = block_inv_sqrt(
        restriction_degrees(ei, f, n), diagonal=False
    ).double()
    xb = x.view(n, d)
    # F_bar_u D_u^{-1/2} x_u for every incidence.
    scaled = torch.stack(
        [
            f[i].to(torch.cdouble)
            @ d_inv[ei[0, i]].to(torch.cdouble)
            @ xb[ei[0, i]]
            for i in range(ei.size(1))
        ]
    )
    delta = torch.bincount(ei[1], minlength=m).double()
    total = 0.0
    for e in range(m):
        idx = (ei[1] == e).nonzero(as_tuple=True)[0]
        for i in idx:
            for j in idx:
                total += (
                    (scaled[i] - scaled[j]).abs().pow(2).sum() / delta[e]
                )
    assert math.isclose(energy.item(), 0.5 * total.item(), rel_tol=1e-8)


def test_theorem_4_recovers_graph_laplacian():
    """A 2-uniform undirected trivial sheaf gives 0.5 * (D - A)."""
    cycle = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (0, 3)]
    lap = _laplacian(cycle, 4, 4, normalized=False).real
    adj = torch.zeros(4, 4, dtype=torch.float64)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        adj[a, b] = adj[b, a] = 1.0
    expected = 0.5 * (torch.diag(adj.sum(1)) - adj)
    assert torch.allclose(lap, expected, atol=1e-12)


def test_undirected_operator_is_real_and_q_invariant():
    """Undirected input makes the operator real and independent of q.

    With H(e) empty every phase product of Eq. 4 is 1, so by Theorem 6 the
    operator is the sheaf-ified undirected hypergraph Laplacian of Zhou et al.
    (2006) and the charge parameter does nothing at all.
    """
    pairs, g = _random_hypergraph(seed=11)
    ei = _edge_index(pairs)
    f = torch.randn(ei.size(1), 2, 2, generator=g, dtype=torch.float64)
    reference = None
    for q in [0.0, 0.05, 0.25, 0.37]:
        q_n = _laplacian(pairs, 12, 7, d=2, f=f, q=q)
        assert q_n.to_dense().imag.abs().max() < 1e-12
        spectrum = torch.linalg.eigvalsh(q_n.to_dense())
        if reference is None:
            reference = spectrum
        else:
            assert torch.allclose(spectrum, reference, atol=1e-12)


def test_star_orientation_activates_the_complex_part():
    """An induced orientation makes the imaginary part non-zero for q != 0."""
    pairs = _star_hypergraph()
    n = 12
    ei = _edge_index(pairs)
    g = torch.Generator().manual_seed(5)
    f = torch.randn(ei.size(1), 2, 2, generator=g, dtype=torch.float64)
    head = derive_orientation(ei, n, n, "star")
    assert head.any() and not head.all()

    at_zero = _laplacian(pairs, n, n, d=2, f=f, q=0.0, is_head=head)
    assert at_zero.to_dense().imag.abs().max() < 1e-12

    charged = _laplacian(pairs, n, n, d=2, f=f, q=0.25, is_head=head)
    assert charged.to_dense().imag.abs().max() > 1e-3


def test_block_diagonal_entries_are_real():
    """The block-diagonal of L^F is real even for a directed hypergraph."""
    pairs = _star_hypergraph()
    n = 12
    ei = _edge_index(pairs)
    g = torch.Generator().manual_seed(9)
    f = torch.randn(ei.size(1), 2, 2, generator=g, dtype=torch.float64)
    head = derive_orientation(ei, n, n, "star")
    lap = _laplacian(
        pairs, n, n, d=2, f=f, q=0.25, is_head=head, normalized=False
    )
    assert _diag_blocks(lap, n, 2).imag.abs().max() < 1e-12


def test_normalization_identity():
    """Q_N^F = I - L_N^F, i.e. normalization order is right."""
    pairs, g = _random_hypergraph(seed=13)
    ei = _edge_index(pairs)
    f = torch.randn(ei.size(1), 1, 1, generator=g, dtype=torch.float64)
    q_n = _laplacian(pairs, 12, 7, f=f).to_dense()
    lap = _laplacian(pairs, 12, 7, f=f, normalized=False).to_dense()
    d_inv = block_inv_sqrt(restriction_degrees(ei, f, 12), diagonal=True)
    scale = torch.diag(d_inv.view(-1).to(torch.cdouble))
    assert torch.allclose(
        torch.eye(12, dtype=torch.cdouble) - q_n,
        scale @ lap @ scale,
        atol=1e-10,
    )


def test_isolated_node_drops_out_of_the_operator():
    """An isolated node contributes nothing rather than producing NaNs.

    With ``D_u = 0`` the pseudo-inverse convention of :func:`block_inv_sqrt`
    zeroes that node's row and column of the normalized operator, so the rest
    of the hypergraph is unaffected and nothing blows up.
    """
    # Node 3 belongs to no hyperedge.
    pairs = [(0, 0), (1, 0), (2, 0)]
    q_n = _laplacian(pairs, 4, 1).to_dense()
    assert torch.isfinite(q_n.abs()).all()
    assert q_n[3].abs().max() == 0.0
    assert q_n[:, 3].abs().max() == 0.0


@pytest.mark.parametrize("size", [1, 2, 3, 6])
def test_hyperedge_size_edge_cases(size):
    """Hyperedges of size 1, 2 and larger all stay well behaved.

    A size-1 hyperedge contributes 1 - 1/1 = 0 to the diagonal and nothing
    off-diagonal, so it must not perturb the operator at all.
    """
    pairs = [(v, 0) for v in range(size)]
    lap = _laplacian(pairs, size, 1, normalized=False).real
    eig = torch.linalg.eigvalsh(lap)
    assert eig.min() >= -TOL
    assert torch.isfinite(lap).all()
    if size == 1:
        assert lap.abs().max() < TOL


def test_incidence_pairs_counts():
    """The pair enumeration yields exactly sum_e delta_e^2 pairs."""
    pairs, _ = _random_hypergraph(seed=17)
    ei = _edge_index(pairs)
    order, left, right = incidence_pairs(ei, 7)
    delta = torch.bincount(ei[1], minlength=7)
    assert left.numel() == int((delta**2).sum())
    assert right.numel() == left.numel()
    assert order.numel() == ei.size(1)
    # Every pair must live inside a single hyperedge.
    hedges = ei[1][order]
    assert torch.equal(hedges[left], hedges[right])


def test_restriction_degrees_matches_loop():
    """D_u equals the explicit sum of F^T F over incident hyperedges."""
    pairs, g = _random_hypergraph(seed=19)
    ei = _edge_index(pairs)
    f = torch.randn(ei.size(1), 3, 3, generator=g, dtype=torch.float64)
    got = restriction_degrees(ei, f, 12)
    expected = torch.zeros(12, 3, 3, dtype=torch.float64)
    for i in range(ei.size(1)):
        expected[ei[0, i]] += f[i].T @ f[i]
    assert torch.allclose(got, expected, atol=1e-10)


def test_block_inv_sqrt_degenerate_and_singular():
    """Degenerate and singular degree blocks stay finite forwards and back.

    Orthogonal restriction maps give D_u = deg(u) I, whose repeated
    eigenvalues make the eigendecomposition backward pass return NaN. The
    diagonal path exists for exactly this case.
    """
    degenerate = (torch.eye(3) * 2.0).unsqueeze(0).requires_grad_(True)
    out = block_inv_sqrt(degenerate, diagonal=True)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(degenerate.grad).all()
    assert torch.allclose(out, torch.eye(3).unsqueeze(0) / math.sqrt(2.0))

    # An isolated node has D_u = 0; the pseudo-inverse convention gives zero.
    singular = torch.zeros(1, 2, 2)
    assert block_inv_sqrt(singular, diagonal=True).abs().max() == 0.0
    assert block_inv_sqrt(singular, diagonal=False).abs().max() == 0.0

    dense = torch.tensor([[[4.0, 1.0], [1.0, 3.0]]], requires_grad=True)
    out = block_inv_sqrt(dense)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(dense.grad).all()
    # D^{-1/2} D^{-1/2} D = I
    prod = out[0] @ out[0] @ dense[0].detach()
    assert torch.allclose(prod, torch.eye(2), atol=1e-5)


def test_derive_orientation():
    """Orientation modes, and the errors they raise."""
    ei = _edge_index(_star_hypergraph(n=6))
    none = derive_orientation(ei, 6, 6, "none")
    assert none.dtype == torch.bool and not none.any()

    star = derive_orientation(ei, 6, 6, "star")
    # The tail of hyperedge j is exactly node j.
    assert torch.equal(star, ei[0] != ei[1])

    with pytest.raises(ValueError, match="Unknown orientation"):
        derive_orientation(ei, 6, 6, "nope")
    with pytest.raises(ValueError, match="node-centred"):
        derive_orientation(ei, 6, 5, "star")


def test_charge_phase():
    """The charge has unit modulus and collapses to 1 at q = 0."""
    head = torch.tensor([True, False, True, False])
    at_zero = charge_phase(head, 0.0, dtype=torch.cdouble)
    assert torch.allclose(at_zero, torch.ones(4, dtype=torch.cdouble))

    charged = charge_phase(head, 0.25, dtype=torch.cdouble)
    assert torch.allclose(charged.abs(), torch.ones(4, dtype=torch.double))
    # Paper convention: exp(-2 pi i q) on tails, so q = 1/4 gives -i.
    assert torch.allclose(charged[1], torch.tensor(-1j, dtype=torch.cdouble))
    assert torch.allclose(charged[0], torch.tensor(1 + 0j, dtype=torch.cdouble))


# --------------------------------------------------------------------------
# Complex primitives
# --------------------------------------------------------------------------


def test_unwind_and_complex_relu():
    """unwind concatenates components; complex ReLU gates on the real part."""
    x = torch.complex(
        torch.tensor([[1.0, -2.0]]), torch.tensor([[3.0, 4.0]])
    )
    assert torch.equal(
        unwind(x), torch.tensor([[1.0, -2.0, 3.0, 4.0]])
    )

    gated = complex_relu(x)
    assert gated[0, 0] == x[0, 0]  # Re > 0 is kept whole
    assert gated[0, 1] == 0  # Re < 0 zeroes both components


def test_complex_dropout():
    """Dropout zeroes whole complex entries, and is a no-op in eval."""
    x = torch.complex(torch.ones(2000), torch.ones(2000))
    assert torch.equal(complex_dropout(x, 0.5, training=False), x)
    assert torch.equal(complex_dropout(x, 0.0, training=True), x)

    out = complex_dropout(x, 0.5, training=True)
    kept = out != 0
    # Whichever entries survive, both components survive together.
    assert torch.equal((out.real != 0), (out.imag != 0))
    assert 0.3 < kept.float().mean() < 0.7


def test_real_linear_shares_weights():
    """RealLinear applies one real matrix to both components."""
    layer = RealLinear(3, 2, bias=False)
    x = torch.complex(torch.randn(4, 3), torch.randn(4, 3))
    out = layer(x)
    assert out.shape == (4, 2) and out.is_complex()
    assert torch.allclose(out.real, layer.lin(x.real), atol=1e-6)
    assert torch.allclose(out.imag, layer.lin(x.imag), atol=1e-6)
    layer.reset_parameters()


def test_complex_layer_norm():
    """The whitener is identity-preserving at init and produces finite output."""
    norm = ComplexLayerNorm(16)
    assert torch.allclose(norm.gamma, torch.eye(2) / math.sqrt(2.0))
    assert torch.allclose(norm.beta, torch.zeros(2))

    x = torch.complex(torch.randn(5, 16) * 7, torch.randn(5, 16) * 3)
    out = norm(x)
    assert out.shape == x.shape and torch.isfinite(out.abs()).all()

    plain = ComplexLayerNorm(16, elementwise_affine=False)
    assert plain.gamma is None
    assert torch.isfinite(plain(x).abs()).all()


# --------------------------------------------------------------------------
# Restriction-map builders
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SHEAF_BUILDERS))
def test_builder_shapes(name):
    """Every builder emits one d x d real block per incidence."""
    d, f, n, m = 3, 8, 6, 6
    ei = _edge_index(_star_hypergraph(n=n))
    builder = SHEAF_BUILDERS[name](d, f)
    x = torch.complex(torch.randn(n * d, f), torch.randn(n * d, f))
    e = torch.complex(torch.randn(m * d, f), torch.randn(m * d, f))
    blocks = builder(x, e, ei, n, m)
    assert blocks.shape == (ei.size(1), d, d)
    assert not blocks.is_complex()
    builder.reset_parameters()


def test_base_builder_requires_to_blocks():
    """The base class refuses to guess how to shape a restriction map."""
    from topobench.nn.backbones.hypergraph.dshn_utils.sheaf_builders import (
        SheafBuilder,
    )

    with pytest.raises(NotImplementedError):
        SheafBuilder(2, 4, out_dim=2).to_blocks(torch.randn(3, 2))


def test_diag_builder_is_diagonal():
    """The diagonal family really does produce diagonal maps."""
    builder = DiagSheafBuilder(3, 4)
    blocks = builder.to_blocks(torch.randn(5, 3))
    off = blocks - torch.diag_embed(
        torch.diagonal(blocks, dim1=-2, dim2=-1)
    )
    assert off.abs().max() == 0.0


def test_general_builder_reshapes():
    """The general family fills the whole d x d block."""
    builder = GeneralSheafBuilder(2, 4)
    raw = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(
        builder.to_blocks(raw), torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    )


@pytest.mark.parametrize("d", [2, 3, 4])
@pytest.mark.parametrize("omap", ["cayley", "matrix_exp"])
def test_ortho_builder_is_orthogonal(d, omap):
    """Orthogonal maps satisfy Q^T Q = I, and reach SO(d) only."""
    builder = OrthoSheafBuilder(d, 4, orthogonal_map=omap)
    raw = torch.randn(7, d * (d - 1) // 2)
    q = builder.to_blocks(raw)
    eye = torch.eye(d).expand_as(q)
    assert (q.transpose(-2, -1) @ q - eye).abs().max() < 1e-5
    # Cayley and matrix_exp both exponentiate a skew-symmetric matrix, so the
    # reflection component of O(d) is out of reach.
    assert torch.allclose(
        torch.linalg.det(q), torch.ones(7), atol=1e-5
    )


@pytest.mark.parametrize("d", [2, 3, 4])
def test_ortho_builder_has_no_dead_parameters(d):
    """Every predicted orthogonal parameter influences the output.

    The shared ``Orthogonal`` module takes d(d+1)/2 parameters but forms
    A = P - P^T, which annihilates the diagonal, so d of them would have
    identically zero gradient. We predict only the d(d-1)/2 live ones.
    """
    builder = OrthoSheafBuilder(d, 4)
    raw = torch.randn(5, d * (d - 1) // 2, requires_grad=True)
    builder.to_blocks(raw).sum().backward()
    per_param = raw.grad.abs().sum(dim=0)
    assert (per_param > 0).all(), f"dead parameters: {per_param}"


def test_ortho_builder_rejects_d_1():
    """A 1x1 orthogonal map has no free parameters."""
    with pytest.raises(ValueError, match="d > 1"):
        OrthoSheafBuilder(1, 4)


@pytest.mark.parametrize("act", ["sigmoid", "tanh", "relu", "none"])
def test_activations(act):
    """All documented sheaf activations run and stay finite."""
    out = _activate(torch.randn(10), act)
    assert out.shape == (10,) and torch.isfinite(out).all()


def test_unknown_activation_raises():
    """An unknown activation is rejected rather than silently ignored."""
    with pytest.raises(ValueError, match="Unknown sheaf activation"):
        _activate(torch.randn(3), "softmax")


# --------------------------------------------------------------------------
# The backbone
# --------------------------------------------------------------------------


def _toy_inputs(n=10, f_in=5):
    """A node-centred hypergraph plus features, valid for any orientation."""
    ei = _edge_index(_star_hypergraph(n=n, stride=3))
    inc = torch.sparse_coo_tensor(
        ei, torch.ones(ei.size(1)), (n, n)
    ).coalesce()
    return torch.randn(n, f_in), inc, ei


def test_auto_module():
    """Shape and reproducibility checks, per the repo's test convention."""
    x, inc, _ = _toy_inputs()
    params = [
        {
            "module": DSHN,
            "init": (5, 8, 4),
            "forward": (x, inc),
            "assert_shape": [(10, 4), (10, 4)],
        }
    ]
    NNModuleAutoTest(params).run()


@pytest.mark.parametrize("sheaf_type", list(SHEAF_BUILDERS))
@pytest.mark.parametrize("orientation", ["none", "star"])
def test_forward_backward(sheaf_type, orientation):
    """Every family and orientation produces finite outputs and gradients.

    The backward pass matters: with orthogonal maps the degree blocks are
    degenerate, which is where a naive eigendecomposition returns NaN.
    """
    torch.manual_seed(0)
    x, inc, _ = _toy_inputs()
    model = DSHN(
        5,
        8,
        4,
        n_layers=2,
        d=3,
        sheaf_type=sheaf_type,
        orientation=orientation,
        q=0.25,
    )
    x_0, x_1 = model(x, inc)
    assert x_0.shape == (10, 4) and x_1.shape == (10, 4)
    assert torch.isfinite(x_0).all() and torch.isfinite(x_1).all()

    x_0.pow(2).mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


def test_dense_incidence_input():
    """A [2, nnz] index is accepted as well as a sparse incidence matrix."""
    torch.manual_seed(0)
    x, inc, ei = _toy_inputs()
    model = DSHN(5, 8, 4, n_layers=1, d=2)
    from_sparse, _ = model(x, inc)
    from_dense, _ = model(x, ei)
    assert torch.allclose(from_sparse, from_dense, atol=1e-6)


def test_dshn_light_freezes_phi():
    """DSHNLight detaches the Laplacian but still trains the projection.

    "the parameters of the MLP responsible for predicting the restriction maps
    ... remain fixed throughout the training process. The model's adaptability
    arises from the initial projection layer" (p. 8).
    """
    x, inc, _ = _toy_inputs()
    for light in [False, True]:
        torch.manual_seed(0)
        model = DSHN(5, 8, 4, n_layers=2, d=2, light=light)
        model(x, inc)[0].pow(2).mean().backward()
        phi_grad = model.sheaf_builder[0].lin.weight.grad
        assert model.lin.weight.grad.abs().sum() > 0
        if light:
            assert phi_grad is None
        else:
            assert phi_grad is not None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dynamic_sheaf": True},
        {"residual": True},
        {"layer_norm": True},
        {"sheaf_left_proj": True},
        {"sheaf_dropout": True, "dropout": 0.3},
        {"init_hedge": "rand"},
        {"add_identity": True},
        {"input_norm": False},
        {"sheaf_act": "tanh"},
        {"d": 1},
    ],
)
def test_optional_components(kwargs):
    """Each optional switch runs end to end and stays finite."""
    torch.manual_seed(0)
    x, inc, _ = _toy_inputs()
    model = DSHN(5, 8, 4, n_layers=2, orientation="star", **kwargs)
    out, _ = model(x, inc)
    assert out.shape == (10, 4) and torch.isfinite(out).all()
    out.pow(2).mean().backward()


def test_reset_parameters_changes_weights():
    """reset_parameters actually reinitializes every submodule."""
    model = DSHN(
        5, 8, 4, n_layers=2, d=2, sheaf_left_proj=True, layer_norm=True
    )
    before = model.lin.weight.clone()
    torch.manual_seed(123)
    model.reset_parameters()
    assert not torch.equal(before, model.lin.weight)


def test_invalid_sheaf_type():
    """An unknown restriction-map family is rejected at construction."""
    with pytest.raises(ValueError, match="Unknown sheaf_type"):
        DSHN(5, 8, 4, sheaf_type="TriangularSheafs")


@pytest.mark.parametrize("orientation", ["none", "star"])
def test_overfits_toy_task(orientation):
    """The backbone can drive a small node-classification loss down.

    A cheap separation of "wrong hyperparameters" from "broken model".
    """
    torch.manual_seed(0)
    n, f_in, classes = 24, 6, 3
    ei = _edge_index(
        [(v, v) for v in range(n)]
        + [(u, v) for v in range(n) for u in range(n) if 0 < abs(u - v) <= 2]
    )
    inc = torch.sparse_coo_tensor(
        ei, torch.ones(ei.size(1)), (n, n)
    ).coalesce()
    x = torch.randn(n, f_in)
    y = torch.randint(0, classes, (n,))

    model = DSHN(
        f_in,
        32,
        classes,
        n_layers=2,
        d=2,
        orientation=orientation,
        dropout=0.0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    first = None
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x, inc)[0], y)
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.5 * first
