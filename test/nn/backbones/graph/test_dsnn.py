"""Tests for the Directed Sheaf Neural Network backbone.

These tests pin down the mathematics of "Sheaves Reloaded: A Directional
Awakening" (ICLR 2026, https://arxiv.org/abs/2506.02842) rather than tensor
shapes: Theorem 1 (Hermitian, positive semidefinite), Theorem 2
(spectrum in [0, 2]), Theorem 3 (collapse to the real sheaf Laplacian on
undirected input, for every q), Theorem 4 (the Magnetic and Sign-Magnetic
Laplacians as special cases) and Theorem 5 (the B B* factorization), plus the
real lifting of Appendix D.

Exactness claims are measured in float64. References -- the Magnetic
Laplacian, the Sign-Magnetic Laplacian, the complex incidence matrix and a
dense delta* delta -- are written here independently of the implementation.
"""

import pytest
import torch

from test._utils.nn_module_auto_test import NNModuleAutoTest
from topobench.nn.backbones.graph.dsnn import DSNNEncoder
from topobench.nn.backbones.graph.dsnn_utils import (
    complex_ops,
    discrete_models,
    laplace,
    laplacian_builders,
    orthogonal,
    phase,
    sheaf_models,
)
from topobench.nn.backbones.graph.nsd_utils import (
    laplacian_builders as nsd_builders,
)

D = torch.float64
CD = torch.complex128

Q_GRID = (0.0, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0)
FAMILIES = ("diag", "bundle", "general")

# A graph with digons, one-way arcs, a triangle (so some cycle carries flux)
# and an isolated node.
MIXED = torch.tensor(
    [[0, 1, 1, 2, 2, 0, 3, 4, 5], [1, 0, 2, 1, 0, 2, 4, 3, 3]]
)
MIXED_N = 7

# Fully reciprocal, i.e. undirected.
UNDIRECTED = torch.tensor(
    [[0, 1, 1, 2, 2, 0, 3, 4], [1, 0, 2, 1, 0, 2, 4, 3]]
)

# Directed with no reciprocal pair, so A_s = 1/2 off the diagonal.
DIGON_FREE = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
ALL_DIGON = torch.tensor(
    [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]
)
SMALL_N = 5

# Degree-heterogeneous, so an induced degree orientation is non-trivial.
HETEROGENEOUS = torch.tensor(
    [
        [0, 1, 0, 2, 0, 3, 1, 2, 3, 4, 0, 5, 5, 6],
        [1, 0, 2, 0, 3, 0, 2, 1, 4, 3, 5, 0, 6, 5],
    ]
)
HETEROGENEOUS_N = 7

# Degree-regular, so every invariant key ties.
REGULAR = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def structure(edge_index, num_nodes, q, orientation="none", phase_sign=1):
    """Return the support, pairing, sign and phase for a graph."""
    support = laplace.symmetrize_support(edge_index, num_nodes)
    left_right, pair = laplace.compute_left_right_map_index(
        support, num_nodes
    )
    sign = phase.pair_sign(
        edge_index, support, pair, num_nodes, orientation
    )
    cos, sin = phase.phase_from_sign(sign, q, phase_sign, dtype=D)
    return support, left_right, pair, sign, cos, sin


def make_maps(family, num_arcs, d, seed=0):
    """Random restriction-map parameters for a family."""
    generator = torch.Generator().manual_seed(seed)
    if family == "diag":
        return torch.randn(num_arcs, d, generator=generator, dtype=D)
    if family == "bundle":
        return torch.randn(
            num_arcs,
            orthogonal.num_orthogonal_params(d),
            generator=generator,
            dtype=D,
        )
    return torch.randn(num_arcs, d, d, generator=generator, dtype=D)


def build_dense(family, maps, num_nodes, d, struct, **kwargs):
    """Assemble the operator densely through the builder under test."""
    support, left_right, pair, _, cos, sin = struct
    builder = laplacian_builders.LAPLACIAN_BUILDERS[family](
        num_nodes, support, d, left_right, pair, cos, sin, **kwargs
    )
    real_i, real_v, imag_i, imag_v, _ = builder.hermitian_parts(maps)
    dense = complex_ops.hermitian_to_dense(
        real_i, real_v, imag_i, imag_v, num_nodes * d
    )
    return builder, dense


def dense_blocks(family, maps, d):
    """The dense d x d real base map of each support arc."""
    if family == "diag":
        return torch.diag_embed(maps)
    if family == "bundle":
        return orthogonal.Orthogonal(d, "cayley")(maps)
    return maps


def reference_delta_star_delta(family, maps, d, num_nodes, struct):
    """Dense delta~* delta~, straight from Definition 1 and Eq. 1."""
    _, left_right, pair, _, cos, sin = struct
    blocks = dense_blocks(family, maps, d)
    num_pairs = pair.size(1)
    delta = torch.zeros(num_pairs * d, num_nodes * d, dtype=CD)
    edge_phase = torch.complex(cos, sin)
    for k in range(num_pairs):
        head, tail = int(pair[0, k]), int(pair[1, k])
        map_head = blocks[int(left_right[0, k])].to(CD)
        map_tail = blocks[int(left_right[1, k])].to(CD)
        rows = slice(k * d, (k + 1) * d)
        delta[rows, head * d : (head + 1) * d] = map_head
        delta[rows, tail * d : (tail + 1) * d] = -map_tail * edge_phase[k]
    return delta.conj().T @ delta


def binary_adjacency(edge_index, num_nodes):
    """Dense binary adjacency of the raw directed graph."""
    adjacency = torch.zeros(num_nodes, num_nodes, dtype=D)
    adjacency[edge_index[0], edge_index[1]] = 1.0
    return adjacency


def magnetic_laplacian(edge_index, num_nodes, q, sign=1):
    """MagNet's L^(q) = D_s - A_s (*) exp(i 2 pi q (A - A^T))."""
    adjacency = binary_adjacency(edge_index, num_nodes)
    symmetric = (adjacency + adjacency.T) / 2
    theta = sign * 2 * torch.pi * q * (adjacency - adjacency.T)
    hermitian = symmetric.to(CD) * torch.complex(
        torch.cos(theta), torch.sin(theta)
    )
    return torch.diag(symmetric.sum(1)).to(CD) - hermitian


def sign_magnetic_laplacian(edge_index, num_nodes):
    """L^sigma, written from its own definition rather than through exp()."""
    adjacency = binary_adjacency(edge_index, num_nodes)
    symmetric = (adjacency + adjacency.T) / 2
    ones = torch.ones(num_nodes, num_nodes, dtype=D)
    real = ones - torch.sign((adjacency - adjacency.T).abs())
    imag = torch.sign(adjacency.abs() - adjacency.T.abs())
    hermitian = symmetric.to(CD) * torch.complex(real, imag)
    return torch.diag(symmetric.abs().sum(1)).to(CD) - hermitian


def complex_incidence(edge_index, num_nodes, q, sign=1):
    """B_hat of Theorem 5: +1 at the representative tail, -T at the head."""
    adjacency = binary_adjacency(edge_index, num_nodes)
    representatives = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            forward = adjacency[u, v] > 0
            backward = adjacency[v, u] > 0
            if not (forward or backward):
                continue
            if backward and not forward:
                representatives.append((v, u))
            else:
                representatives.append((u, v))
    incidence = torch.zeros(num_nodes, len(representatives), dtype=CD)
    for column, (tail, head) in enumerate(representatives):
        # "-T_uv if e = (v,u)": for the arc tail -> head the entry at the head
        # is indexed T_{head tail}, the conjugate of T_{tail head}.
        theta = torch.tensor(
            sign
            * 2
            * torch.pi
            * q
            * float(adjacency[head, tail] - adjacency[tail, head])
        )
        incidence[tail, column] = 1.0
        incidence[head, column] = -complex(
            torch.cos(theta), torch.sin(theta)
        )
    return incidence


def trivial_sheaf_laplacian(edge_index, num_nodes, q, phase_sign=1):
    """Our operator for a Trivial Directed Cellular Sheaf: d = 1, maps = 1."""
    struct = structure(edge_index, num_nodes, q, phase_sign=phase_sign)
    maps = torch.ones(struct[0].size(1), 1, dtype=D)
    _, dense = build_dense(
        "diag", maps, num_nodes, 1, struct, normalised=False
    )
    return dense


def encoder(**kwargs):
    """A small DSNNEncoder with dropout switched off by default."""
    config = {
        "input_dim": 12,
        "hidden_dim": 24,
        "num_layers": 2,
        "dropout": 0.0,
        "input_dropout": 0.0,
    }
    config.update(kwargs)
    return DSNNEncoder(**config)


# --------------------------------------------------------------------------- #
# Theorem 1 -- Hermitian, real diagonal, positive semidefinite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("d", [2, 3])
def test_theorem_1_matches_delta_star_delta(family, d):
    """The builder reproduces a dense delta~* delta~ exactly."""
    struct = structure(MIXED, MIXED_N, 0.31)
    maps = make_maps(family, struct[0].size(1), d)
    _, dense = build_dense(
        family, maps, MIXED_N, d, struct, normalised=False
    )
    reference = reference_delta_star_delta(family, maps, d, MIXED_N, struct)
    assert torch.allclose(dense, reference, atol=1e-11)


@pytest.mark.parametrize("family", FAMILIES)
def test_theorem_1_hermitian(family):
    """L equals its own conjugate transpose."""
    struct = structure(MIXED, MIXED_N, 0.31)
    maps = make_maps(family, struct[0].size(1), 3)
    _, dense = build_dense(family, maps, MIXED_N, 3, struct)
    assert torch.allclose(dense, dense.conj().T, atol=1e-11)


@pytest.mark.parametrize("family", FAMILIES)
def test_theorem_1_diagonal_is_exactly_real(family):
    """The diagonal has no imaginary part at all, not merely a small one.

    Catches a conjugation applied to the whole matrix instead of only to the
    off-diagonal blocks.
    """
    struct = structure(MIXED, MIXED_N, 0.31)
    maps = make_maps(family, struct[0].size(1), 3)
    _, dense = build_dense(family, maps, MIXED_N, 3, struct)
    assert torch.equal(
        dense.diagonal().imag, torch.zeros(MIXED_N * 3, dtype=D)
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_theorem_1_psd_quadratic_form(family):
    """x* L x is real and non-negative, without an eigensolver."""
    struct = structure(MIXED, MIXED_N, 0.2)
    maps = make_maps(family, struct[0].size(1), 2)
    _, dense = build_dense(family, maps, MIXED_N, 2, struct)
    generator = torch.Generator().manual_seed(7)
    for _ in range(16):
        vector = torch.complex(
            torch.randn(MIXED_N * 2, generator=generator, dtype=D),
            torch.randn(MIXED_N * 2, generator=generator, dtype=D),
        )
        value = vector.conj() @ (dense @ vector)
        assert value.imag.abs() < 1e-10
        assert value.real > -1e-10


@pytest.mark.parametrize("family", FAMILIES)
def test_theorem_1_psd_eigenvalues(family):
    """The spectrum is non-negative."""
    struct = structure(MIXED, MIXED_N, 0.2)
    maps = make_maps(family, struct[0].size(1), 3)
    _, dense = build_dense(family, maps, MIXED_N, 3, struct)
    assert torch.linalg.eigvalsh(dense).min() > -1e-9


@pytest.mark.parametrize("family", FAMILIES)
def test_off_diagonal_blocks_are_conjugate_transposes(family):
    """Block (b, a) is the conjugate transpose of block (a, b).

    The Neural Sheaf Diffusion port mirrors its lower triangle with identical
    values, which is right for a symmetric operator; a Hermitian one needs the
    values conjugated as well.
    """
    d = 3
    struct = structure(MIXED, MIXED_N, 0.31)
    maps = make_maps(family, struct[0].size(1), d)
    _, dense = build_dense(family, maps, MIXED_N, d, struct)
    pair = struct[2]
    for k in range(pair.size(1)):
        a, b = int(pair[0, k]), int(pair[1, k])
        upper = dense[a * d : (a + 1) * d, b * d : (b + 1) * d]
        lower = dense[b * d : (b + 1) * d, a * d : (a + 1) * d]
        assert torch.allclose(lower, upper.conj().T, atol=1e-12)


# --------------------------------------------------------------------------- #
# Theorem 2 -- spectrum of the normalized operator lies in [0, 2]
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("q", Q_GRID)
def test_theorem_2_spectrum_bound(family, q):
    """spec(L_N) is contained in [0, 2] across the paper's whole q grid."""
    struct = structure(MIXED, MIXED_N, q, orientation="index")
    maps = make_maps(family, struct[0].size(1), 2)
    _, dense = build_dense(family, maps, MIXED_N, 2, struct)
    eigenvalues = torch.linalg.eigvalsh(dense)
    assert eigenvalues.min() > -1e-9
    assert eigenvalues.max() < 2 + 1e-9


@pytest.mark.parametrize("family", FAMILIES)
def test_theorem_2_bound_is_attained(family):
    """The upper bound is approached, so it is not a vacuous statement."""
    struct = structure(MIXED, MIXED_N, 0.25, orientation="index")
    maps = make_maps(family, struct[0].size(1), 3)
    _, dense = build_dense(family, maps, MIXED_N, 3, struct)
    assert torch.linalg.eigvalsh(dense).max() > 1.5


def test_jacobi_normalization_breaks_the_theorem_2_bound():
    """The cheap diagonal normalization is PSD but exceeds 2.

    Only the general family has full degree blocks. Approximating Eq. 5 by
    their diagonal keeps the operator a congruence, hence Hermitian and
    positive semidefinite, but loses the bound of Theorem 2 -- which is why
    ``block_norm`` defaults to the exact computation.
    """
    struct = structure(MIXED, MIXED_N, 0.25, orientation="index")
    maps = make_maps("general", struct[0].size(1), 3)
    _, exact = build_dense(
        "general", maps, MIXED_N, 3, struct, block_norm=True
    )
    _, jacobi = build_dense(
        "general", maps, MIXED_N, 3, struct, block_norm=False
    )
    assert torch.linalg.eigvalsh(exact).max() < 2 + 1e-9
    assert torch.linalg.eigvalsh(jacobi).max() > 2 + 1e-3
    assert torch.linalg.eigvalsh(jacobi).min() > -1e-9


def test_block_normalization_survives_repeated_eigenvalues():
    """Degenerate degree blocks must not produce non-finite gradients.

    On a degree-regular graph every degree block is :math:`\\deg(u) I_d`, so
    its eigenvalues are exactly repeated and the backward pass of ``eigh``
    would divide by a zero gap. Following the reference implementation,
    ``normalise`` detaches the inverse square root and jitters the blocks
    while training, which keeps that unreachable.
    """
    torch.manual_seed(0)
    model = (
        encoder(sheaf_type="general", d=3, hidden_dim=12, input_dim=6)
        .double()
        .train()
    )
    out = model(torch.randn(3, 6, dtype=D), REGULAR)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(bool(g.isfinite().all()) for g in grads)


def test_degree_block_jitter_is_training_only():
    """The jitter perturbs the operator while training and never in eval."""
    struct = structure(MIXED, MIXED_N, 0.31)
    maps = make_maps("general", struct[0].size(1), 3)
    _, first = build_dense("general", maps, MIXED_N, 3, struct, training=False)
    _, again = build_dense("general", maps, MIXED_N, 3, struct, training=False)
    assert torch.equal(first, again)

    torch.manual_seed(0)
    _, noisy = build_dense("general", maps, MIXED_N, 3, struct, training=True)
    assert not torch.equal(first, noisy)
    assert torch.allclose(first, noisy, atol=1e-2)


@pytest.mark.parametrize("family", FAMILIES)
def test_normalization_makes_diagonal_blocks_identity(family):
    """Eq. 5 sends every non-isolated node's degree block to the identity."""
    d = 3
    struct = structure(MIXED, MIXED_N, 0.25, orientation="index")
    maps = make_maps(family, struct[0].size(1), d)
    _, dense = build_dense(family, maps, MIXED_N, d, struct)
    degree = laplace.node_degree(struct[0], MIXED_N)
    identity = torch.eye(d, dtype=CD)
    for u in range(MIXED_N):
        block = dense[u * d : (u + 1) * d, u * d : (u + 1) * d]
        target = identity if degree[u] > 0 else torch.zeros_like(identity)
        assert torch.allclose(block, target, atol=1e-9)


def test_unnormalised_flag_skips_equation_5():
    """``normalised=False`` returns the raw operator."""
    struct = structure(MIXED, MIXED_N, 0.25)
    maps = make_maps("diag", struct[0].size(1), 2)
    _, raw = build_dense(
        "diag", maps, MIXED_N, 2, struct, normalised=False
    )
    _, normalised = build_dense("diag", maps, MIXED_N, 2, struct)
    assert not torch.allclose(raw, normalised, atol=1e-6)


def test_degree_shift_reproduces_the_reference_normalization():
    """``degree_shift=1.0`` gives the (D + I)^-1/2 of the reference code."""
    d = 2
    struct = structure(UNDIRECTED, MIXED_N, 0.0)
    support, left_right, pair, _, cos, sin = struct
    maps = make_maps("bundle", support.size(1), d)
    _, ours = build_dense(
        "bundle", maps, MIXED_N, d, struct, degree_shift=1.0
    )
    degree = laplace.node_degree(support, MIXED_N).to(D)
    scale = (degree + 1.0).pow(-0.5)
    blocks = orthogonal.Orthogonal(d, "cayley")(maps)
    expected = torch.zeros(MIXED_N * d, MIXED_N * d, dtype=D)
    for k in range(pair.size(1)):
        a, b = int(pair[0, k]), int(pair[1, k])
        block = -(
            blocks[int(left_right[0, k])].T @ blocks[int(left_right[1, k])]
        )
        block = block * scale[a] * scale[b]
        expected[a * d : (a + 1) * d, b * d : (b + 1) * d] = block
        expected[b * d : (b + 1) * d, a * d : (a + 1) * d] = block.T
    for u in range(MIXED_N):
        expected[u * d : (u + 1) * d, u * d : (u + 1) * d] = (
            torch.eye(d, dtype=D) * degree[u] * scale[u] ** 2
        )
    assert torch.allclose(ours.real, expected, atol=1e-10)


# --------------------------------------------------------------------------- #
# Theorem 3 -- collapse to the real sheaf Laplacian
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("family", FAMILIES)
def test_theorem_3_undirected_input_is_exactly_real(family):
    """On undirected input the imaginary part is identically zero, for all q."""
    for q in Q_GRID:
        struct = structure(UNDIRECTED, MIXED_N, q)
        maps = make_maps(family, struct[0].size(1), 3)
        _, dense = build_dense(family, maps, MIXED_N, 3, struct)
        assert torch.equal(dense.imag, torch.zeros_like(dense.imag))


@pytest.mark.parametrize("family", FAMILIES)
def test_theorem_3_operator_does_not_move_with_q(family):
    """On undirected input the operator is bitwise identical for every q."""
    struct = structure(UNDIRECTED, MIXED_N, 0.0)
    maps = make_maps(family, struct[0].size(1), 3)
    _, base = build_dense(family, maps, MIXED_N, 3, struct)
    for q in Q_GRID:
        struct_q = structure(UNDIRECTED, MIXED_N, q)
        _, dense = build_dense(family, maps, MIXED_N, 3, struct_q)
        assert torch.equal(dense, base)


@pytest.mark.parametrize("family", ["diag", "general"])
def test_theorem_3_equals_the_neural_sheaf_diffusion_builder(family):
    """The unphased operator is bit-equal to the real sheaf Laplacian.

    Compared against the Neural Sheaf Diffusion builders already in TopoBench,
    fed the same restriction-map parameters. Only the diagonal and general
    families are compared: our orthogonal parameterization takes
    ``d(d-1)/2`` parameters instead of ``d(d+1)/2``, so the two cannot be
    given identical inputs. That family is covered instead by
    ``test_theorem_1_matches_delta_star_delta``.

    Note the comparison uses ``normalised=False``, because those two Neural
    Sheaf Diffusion builders return the unnormalized operator.
    """
    d = 3
    struct = structure(UNDIRECTED, MIXED_N, 0.37)
    support = struct[0]
    maps = make_maps(family, support.size(1), d)
    _, ours = build_dense(
        family, maps, MIXED_N, d, struct, normalised=False
    )

    reference_class = {
        "diag": nsd_builders.DiagLaplacianBuilder,
        "general": nsd_builders.GeneralLaplacianBuilder,
    }[family]
    reference = reference_class(MIXED_N, support, d=d)
    (index, value), _ = reference(maps)
    dense = torch.zeros(MIXED_N * d, MIXED_N * d, dtype=D)
    dense.index_put_((index[0], index[1]), value, accumulate=True)

    assert torch.allclose(ours.real, dense, atol=1e-12)
    assert torch.equal(ours.imag, torch.zeros_like(ours.imag))


@pytest.mark.parametrize("family", FAMILIES)
def test_theorem_3_end_to_end_q_is_inert(family):
    """The whole encoder is bitwise invariant to q on undirected input."""
    torch.manual_seed(1)
    model = encoder(sheaf_type=family, d=3).eval()
    features = torch.randn(MIXED_N + 1, 12)
    base = model(features, UNDIRECTED)
    for q in Q_GRID:
        model.sheaf_model.q = q
        assert torch.equal(model(features, UNDIRECTED), base)


def test_q_is_a_plain_float_not_a_parameter():
    """q is searched over in the paper, never learned."""
    model = encoder()
    assert "q" not in dict(model.named_parameters())
    assert isinstance(model.sheaf_model.q, float)


@pytest.mark.parametrize("family", FAMILIES)
def test_conjugate_operators_for_negated_charge(family):
    """L(-q) is the conjugate of L(q), so q and 1 - q are equivalent."""
    struct_plus = structure(MIXED, MIXED_N, 0.23)
    struct_minus = structure(MIXED, MIXED_N, -0.23)
    maps = make_maps(family, struct_plus[0].size(1), 2)
    _, plus = build_dense(family, maps, MIXED_N, 2, struct_plus)
    _, minus = build_dense(family, maps, MIXED_N, 2, struct_minus)
    assert torch.allclose(minus, plus.conj(), atol=1e-12)


# --------------------------------------------------------------------------- #
# Theorem 4 -- Magnetic and Sign-Magnetic Laplacians
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", Q_GRID)
def test_theorem_4_magnetic_laplacian_digon_free(q):
    """A trivial sheaf on a digon-free graph gives 2 L^(q).

    The factor 2 is the paper's own remark: a one-way arc has
    ``A_s = 1/2``, and the factor is absorbed by W_1 and W_2.
    """
    ours = trivial_sheaf_laplacian(DIGON_FREE, SMALL_N, q)
    reference = magnetic_laplacian(DIGON_FREE, SMALL_N, q)
    assert torch.allclose(ours, 2 * reference, atol=1e-12)


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5])
def test_theorem_4_factor_two_disappears_on_a_reciprocal_graph(q):
    """On an all-digon graph A_s = 1, so the operator equals L^(q) exactly."""
    ours = trivial_sheaf_laplacian(ALL_DIGON, SMALL_N, q)
    reference = magnetic_laplacian(ALL_DIGON, SMALL_N, q)
    assert torch.allclose(ours, reference, atol=1e-12)


def test_theorem_4_sign_magnetic_laplacian_at_quarter_charge():
    """At q = 1/4 the Magnetic Laplacian is the Sign-Magnetic Laplacian."""
    sign_magnetic = sign_magnetic_laplacian(DIGON_FREE, SMALL_N)
    magnetic = magnetic_laplacian(DIGON_FREE, SMALL_N, 0.25)
    ours = trivial_sheaf_laplacian(DIGON_FREE, SMALL_N, 0.25)
    assert torch.allclose(magnetic, sign_magnetic, atol=1e-12)
    assert torch.allclose(ours, 2 * sign_magnetic, atol=1e-12)


@pytest.mark.parametrize("q", [0.1, 0.25, 0.4])
def test_both_sign_conventions_satisfy_theorem_4(q):
    """Definition 1's printed sign and its worked example agree in substance.

    Each matches the Magnetic Laplacian built with the corresponding sign, and
    the two operators are conjugate and isospectral. Theorem 4 therefore does
    not single out one reading, and the discrepancy in the paper has no
    consequence for the model.
    """
    plus = trivial_sheaf_laplacian(DIGON_FREE, SMALL_N, q, phase_sign=1)
    minus = trivial_sheaf_laplacian(DIGON_FREE, SMALL_N, q, phase_sign=-1)
    assert torch.allclose(
        plus, 2 * magnetic_laplacian(DIGON_FREE, SMALL_N, q, sign=1), atol=1e-12
    )
    assert torch.allclose(
        minus,
        2 * magnetic_laplacian(DIGON_FREE, SMALL_N, q, sign=-1),
        atol=1e-12,
    )
    assert torch.allclose(minus, plus.conj(), atol=1e-12)
    assert torch.allclose(
        torch.linalg.eigvalsh(plus),
        torch.linalg.eigvalsh(minus),
        atol=1e-10,
    )


@pytest.mark.parametrize("q", [0.0, 0.5])
def test_sign_conventions_coincide_at_real_charges(q):
    """At q = 0 and q = 1/2 the phase is real, so the conventions agree."""
    plus = trivial_sheaf_laplacian(DIGON_FREE, SMALL_N, q, phase_sign=1)
    minus = trivial_sheaf_laplacian(DIGON_FREE, SMALL_N, q, phase_sign=-1)
    assert torch.allclose(plus, minus, atol=1e-12)


# --------------------------------------------------------------------------- #
# Theorem 5 -- the complex incidence factorization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [0.0, 0.25, 0.31, 0.5])
def test_theorem_5_incidence_factorization(q):
    """B B* reproduces the operator, hence 2 L^(q)."""
    incidence = complex_incidence(DIGON_FREE, SMALL_N, q)
    product = incidence @ incidence.conj().T
    ours = trivial_sheaf_laplacian(DIGON_FREE, SMALL_N, q)
    assert torch.allclose(product, ours, atol=1e-12)
    assert torch.allclose(
        product, 2 * magnetic_laplacian(DIGON_FREE, SMALL_N, q), atol=1e-12
    )


def test_theorem_5_incidence_structure():
    """Each column of B has exactly two non-zeros, of unit modulus."""
    incidence = complex_incidence(DIGON_FREE, SMALL_N, 0.25)
    magnitudes = incidence.abs()
    assert torch.equal(
        (magnitudes > 0).sum(0),
        torch.full((incidence.size(1),), 2),
    )
    nonzero = magnitudes[magnitudes > 0]
    assert torch.allclose(nonzero, torch.ones_like(nonzero), atol=1e-12)


def test_theorem_5_kernel_at_zero_charge():
    """At q = 0 the all-ones vector lies in the kernel of B*."""
    incidence = complex_incidence(DIGON_FREE, SMALL_N, 0.0)
    image = incidence.conj().T @ torch.ones(SMALL_N, dtype=CD)
    assert torch.allclose(image, torch.zeros_like(image), atol=1e-12)


# --------------------------------------------------------------------------- #
# Appendix D -- the real lifting
# --------------------------------------------------------------------------- #
def test_real_lifting_matches_the_complex_product():
    """The lifted real product equals the complex one."""
    torch.manual_seed(0)
    size, channels = 6, 4
    raw = torch.randn(size, size, dtype=D) + 1j * torch.randn(
        size, size, dtype=D
    )
    operator = (raw + raw.conj().T) / 2
    real_i = operator.real.nonzero().t()
    real_v = operator.real[real_i[0], real_i[1]]
    imag_i = operator.imag.nonzero().t()
    imag_v = operator.imag[imag_i[0], imag_i[1]]
    index = complex_ops.lifted_index(real_i, imag_i, size)
    value = complex_ops.lifted_value(real_v, imag_v)

    features = torch.randn(size, channels, dtype=D) + 1j * torch.randn(
        size, channels, dtype=D
    )
    stacked = torch.cat([features.real, features.imag], dim=0)
    lifted = laplace.spmm(index, value, 2 * size, stacked)
    expected = operator @ features
    assert torch.allclose(lifted[:size], expected.real, atol=1e-12)
    assert torch.allclose(lifted[size:], expected.imag, atol=1e-12)


def test_real_lifting_is_symmetric_with_doubled_spectrum():
    """A Hermitian operator lifts to a symmetric one with each eigenvalue twice.

    Catches a swap of the ``+A_I`` and ``-A_I`` blocks, which would leave the
    product correct for real inputs only.
    """
    struct = structure(MIXED, MIXED_N, 0.25, orientation="index")
    maps = make_maps("diag", struct[0].size(1), 2)
    builder, dense = build_dense("diag", maps, MIXED_N, 2, struct)
    (index, value), _ = builder(maps)
    size = 2 * MIXED_N * 2
    lifted = torch.zeros(size, size, dtype=D)
    lifted.index_put_((index[0], index[1]), value, accumulate=True)
    assert torch.allclose(lifted, lifted.T, atol=1e-12)
    single = torch.linalg.eigvalsh(dense)
    assert torch.allclose(
        torch.sort(torch.cat([single, single])).values,
        torch.linalg.eigvalsh(lifted),
        atol=1e-9,
    )


def test_stack_and_unwind_round_trip():
    """``stack_real`` and ``split_parts`` invert one another."""
    values = torch.randn(10, 3, dtype=D)
    stacked = complex_ops.stack_real(values)
    real, imag = complex_ops.split_parts(stacked, 10)
    assert torch.equal(real, values)
    assert torch.equal(imag, torch.zeros_like(values))


def test_unwind_puts_the_real_part_first():
    """``unwind`` emits (real || imaginary), in that order."""
    num_nodes, d, channels = 3, 2, 2
    size = num_nodes * d
    stacked = torch.arange(2 * size * channels, dtype=D).reshape(
        2 * size, channels
    )
    unwound = complex_ops.unwind_split(stacked, num_nodes)
    assert unwound.shape == (num_nodes, 2 * d * channels)
    assert torch.equal(
        unwound[:, : d * channels], stacked[:size].reshape(num_nodes, -1)
    )


def test_complex_relu_gates_on_the_real_part():
    """sigma zeroes both components when the real part is negative."""
    real = torch.tensor([[-1.0], [2.0], [0.0]], dtype=D)
    imag = torch.tensor([[2.0], [3.0], [5.0]], dtype=D)
    out = complex_ops.complex_relu_split(torch.cat([real, imag], 0), 3)
    assert out[0].item() == 0.0
    assert out[3].item() == 0.0
    assert out[1].item() == 2.0
    assert out[4].item() == 3.0
    assert out[2].item() == 0.0
    assert out[5].item() == 5.0


def test_complex_relu_is_not_componentwise():
    """sigma(-1 + 2i) is 0, not 2i."""
    stacked = torch.tensor([[-1.0], [2.0]], dtype=D)
    out = complex_ops.complex_relu_split(stacked, 1)
    assert torch.equal(out, torch.zeros_like(out))


def test_complex_relu_matches_relu_on_real_input():
    """With a zero imaginary part sigma reduces to the usual ReLU."""
    values = torch.randn(5, 3, dtype=D)
    out = complex_ops.complex_relu_split(complex_ops.stack_real(values), 5)
    assert torch.allclose(out[:5], torch.relu(values))


def test_complex_dropout_drops_whole_entries():
    """Real and imaginary parts of an entry are dropped together."""
    torch.manual_seed(0)
    size = 60
    real = torch.randn(size, 3, dtype=D).abs() + 0.5
    imag = torch.randn(size, 3, dtype=D).abs() + 0.5
    stacked = torch.cat([real, imag], 0)
    out = complex_ops.complex_dropout_split(stacked, size, 0.5, True)
    assert torch.equal(out[:size] == 0, out[size:] == 0)
    kept = out[:size] != 0
    assert torch.allclose(out[:size][kept], real[kept] * 2.0, atol=1e-12)


def test_complex_dropout_is_identity_outside_training():
    """Evaluation mode and p = 0 both leave the features untouched."""
    stacked = complex_ops.stack_real(torch.randn(8, 3, dtype=D))
    assert torch.equal(
        complex_ops.complex_dropout_split(stacked, 8, 0.5, False), stacked
    )
    assert torch.equal(
        complex_ops.complex_dropout_split(stacked, 8, 0.0, True), stacked
    )


def test_hermitian_to_dense_round_trip():
    """The dense helper reproduces the matrix it was built from."""
    torch.manual_seed(0)
    size = 5
    raw = torch.randn(size, size, dtype=D) + 1j * torch.randn(
        size, size, dtype=D
    )
    operator = (raw + raw.conj().T) / 2
    real_i = operator.real.nonzero().t()
    imag_i = operator.imag.nonzero().t()
    rebuilt = complex_ops.hermitian_to_dense(
        real_i,
        operator.real[real_i[0], real_i[1]],
        imag_i,
        operator.imag[imag_i[0], imag_i[1]],
        size,
    )
    assert torch.allclose(rebuilt, operator, atol=1e-14)


def test_hermitian_to_dense_with_no_imaginary_entries():
    """An empty imaginary pattern yields a real matrix."""
    index = torch.tensor([[0, 1], [1, 0]])
    empty = torch.empty((2, 0), dtype=torch.long)
    dense = complex_ops.hermitian_to_dense(
        index, torch.ones(2, dtype=D), empty, torch.empty(0, dtype=D), 2
    )
    assert torch.equal(dense.imag, torch.zeros(2, 2, dtype=D))


# --------------------------------------------------------------------------- #
# structural utilities
# --------------------------------------------------------------------------- #
def test_symmetrize_support_sanitizes_the_graph():
    """Self-loops go, duplicates collapse, and every arc gains its reverse."""
    raw = torch.tensor([[0, 0, 1, 1, 1, 2], [0, 1, 1, 2, 2, 1]])
    support = laplace.symmetrize_support(raw, 4)
    arcs = {tuple(a) for a in support.t().tolist()}
    assert arcs == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert len(arcs) == support.size(1)


def test_pair_index_matches_a_brute_force_reference():
    """The sorted pairing agrees with an explicit dictionary lookup."""
    support = laplace.symmetrize_support(MIXED, MIXED_N)
    left_right, pair = laplace.compute_left_right_map_index(
        support, MIXED_N
    )
    lookup = {
        (u, v): k for k, (u, v) in enumerate(support.t().tolist())
    }
    for k in range(pair.size(1)):
        a, b = int(pair[0, k]), int(pair[1, k])
        assert a < b
        assert int(left_right[0, k]) == lookup[(a, b)]
        assert int(left_right[1, k]) == lookup[(b, a)]


def test_pair_index_requires_a_symmetric_support():
    """A one-sided arc is rejected with a clear message."""
    with pytest.raises(ValueError, match="reverse"):
        laplace.compute_left_right_map_index(
            torch.tensor([[0, 1], [1, 2]]), 3
        )


def test_pair_index_handles_an_empty_graph():
    """No edges gives empty pairings rather than an error."""
    empty = torch.empty((2, 0), dtype=torch.long)
    left_right, pair = laplace.compute_left_right_map_index(empty, 3)
    assert left_right.shape == (2, 0)
    assert pair.shape == (2, 0)


def test_pair_index_is_not_quadratic_in_memory():
    """A graph far larger than a quadratic pairing could hold still builds."""
    num_nodes = 20_000
    generator = torch.Generator().manual_seed(0)
    src = torch.randint(0, num_nodes, (100_000,), generator=generator)
    dst = torch.randint(0, num_nodes, (100_000,), generator=generator)
    support = laplace.symmetrize_support(
        torch.stack([src, dst]), num_nodes
    )
    left_right, pair = laplace.compute_left_right_map_index(
        support, num_nodes
    )
    assert pair.size(1) * 2 == support.size(1)
    assert left_right.size(1) == pair.size(1)


def test_flip_index_transposes_each_block():
    """Swapping element indices places the block transpose opposite."""
    num_nodes, d = 4, 3
    pair = torch.tensor([[0, 1], [2, 3]])
    _, off = laplace.block_indices(num_nodes, pair, d)
    blocks = torch.randn(pair.size(1), d, d, dtype=D)
    direct = torch.zeros(num_nodes * d, num_nodes * d, dtype=D)
    direct[off[0], off[1]] = blocks.reshape(-1)
    flipped_index = laplace.flip_index(off)
    flipped = torch.zeros_like(direct)
    flipped[flipped_index[0], flipped_index[1]] = blocks.reshape(-1)
    for k, (a, b) in enumerate(pair.t().tolist()):
        assert torch.equal(
            direct[a * d : (a + 1) * d, b * d : (b + 1) * d], blocks[k]
        )
        assert torch.equal(
            flipped[b * d : (b + 1) * d, a * d : (a + 1) * d], blocks[k].T
        )


def test_block_indices_rejects_unordered_pairs():
    """Pairs must be given with the smaller node first."""
    with pytest.raises(ValueError, match="pair_index"):
        laplace.block_indices(4, torch.tensor([[3], [1]]), 2)


def test_block_indices_diagonal_variant_is_narrower():
    """The diagonal layout stores d entries per block instead of d squared."""
    pair = torch.tensor([[0], [1]])
    _, full = laplace.block_indices(3, pair, 3, diagonal=False)
    _, thin = laplace.block_indices(3, pair, 3, diagonal=True)
    assert full.size(1) == 9
    assert thin.size(1) == 3


def test_spmm_matches_dense_matmul_and_backpropagates():
    """The sparse product is correct and differentiable in its values."""
    torch.manual_seed(0)
    dense = torch.randn(7, 7, dtype=D)
    dense[dense.abs() < 0.8] = 0
    index = dense.nonzero().t()
    value = dense[index[0], index[1]].clone().requires_grad_(True)
    features = torch.randn(7, 4, dtype=D)
    out = laplace.spmm(index, value, 7, features)
    assert torch.allclose(out, dense @ features, atol=1e-14)
    out.sum().backward()
    assert value.grad is not None
    assert bool(value.grad.isfinite().all())


def test_edge_keys_are_injective():
    """The integer encoding separates distinct arcs."""
    arcs = torch.tensor([[0, 1, 2, 2], [1, 0, 3, 4]])
    keys = laplace.edge_keys(arcs, 5)
    assert len(set(keys.tolist())) == 4


def test_remove_self_loops_and_node_degree():
    """Self-loops are dropped and degrees count outgoing arcs."""
    arcs = torch.tensor([[0, 1, 1, 2], [0, 0, 2, 1]])
    assert laplace.remove_self_loops(arcs).size(1) == 3
    assert laplace.node_degree(arcs, 3).tolist() == [1.0, 2.0, 1.0]


def test_mergesp_concatenates_patterns():
    """Merging two sparse patterns concatenates indices and values."""
    index, value = laplace.mergesp(
        torch.tensor([[0], [1]]),
        torch.tensor([1.0], dtype=D),
        torch.tensor([[1], [0]]),
        torch.tensor([2.0], dtype=D),
    )
    assert index.shape == (2, 2)
    assert value.tolist() == [1.0, 2.0]


def test_has_directed_edge_uses_the_binary_adjacency():
    """A repeated arc counts once, as Definition 1 requires."""
    once = torch.tensor([[0], [1]])
    thrice = torch.tensor([[0, 0, 0], [1, 1, 1]])
    query = torch.tensor([[0, 1], [1, 0]])
    assert laplace.has_directed_edge(once, 2, query).tolist() == [
        True,
        False,
    ]
    assert torch.equal(
        laplace.has_directed_edge(once, 2, query),
        laplace.has_directed_edge(thrice, 2, query),
    )


def test_has_directed_edge_on_empty_inputs():
    """Empty adjacencies and empty queries return empty or false results."""
    empty = torch.empty((2, 0), dtype=torch.long)
    query = torch.tensor([[0], [1]])
    assert laplace.has_directed_edge(empty, 2, query).tolist() == [False]
    assert laplace.has_directed_edge(query, 2, empty).numel() == 0


def test_phase_is_built_from_the_binary_adjacency():
    """Tripling an arc does not triple its phase angle."""
    once = torch.tensor([[0, 1, 1], [1, 0, 2]])
    thrice = torch.tensor([[0, 1, 1, 1, 1], [1, 0, 2, 2, 2]])
    struct_once = structure(once, 3, 0.25)
    struct_thrice = structure(thrice, 3, 0.25)
    assert torch.equal(struct_once[3], struct_thrice[3])
    assert torch.allclose(struct_once[4], struct_thrice[4])


# --------------------------------------------------------------------------- #
# orientation
# --------------------------------------------------------------------------- #
def test_orientation_none_reads_directions_from_the_data():
    """The faithful path gives 0 for reciprocal pairs and +/-1 for one-way."""
    directed = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 3]])
    _, _, pair, sign, _, _ = structure(directed, 4, 0.25)
    assert pair.t().tolist() == [[0, 1], [1, 2], [2, 3]]
    assert sign.tolist() == [0, 1, 1]


def test_orientation_none_is_all_zero_on_undirected_input():
    """Undirected input carries no direction to read."""
    _, _, _, sign, cos, sin = structure(UNDIRECTED, MIXED_N, 0.25)
    assert torch.equal(sign, torch.zeros_like(sign))
    assert torch.equal(sin, torch.zeros_like(sin))
    assert torch.allclose(cos, torch.ones_like(cos))


def test_degree_orientation_activates_the_imaginary_part():
    """An induced orientation makes the phase non-trivial."""
    struct_none = structure(HETEROGENEOUS, HETEROGENEOUS_N, 0.25)
    struct_degree = structure(
        HETEROGENEOUS, HETEROGENEOUS_N, 0.25, orientation="degree"
    )
    maps = make_maps("diag", struct_none[0].size(1), 2)
    _, plain = build_dense(
        "diag", maps, HETEROGENEOUS_N, 2, struct_none
    )
    _, oriented = build_dense(
        "diag", maps, HETEROGENEOUS_N, 2, struct_degree
    )
    assert torch.equal(plain.imag, torch.zeros_like(plain.imag))
    assert oriented.imag.abs().max() > 0


def test_degree_orientation_ties_leave_edges_undirected():
    """On a degree-regular graph no invariant key separates the endpoints.

    This is the cost of permutation equivariance: a tie-break on node index
    would orient every edge but would make the operator depend on the node
    ordering.
    """
    _, _, _, sign, _, _ = structure(
        REGULAR, 3, 0.25, orientation="degree"
    )
    assert torch.equal(sign, torch.zeros_like(sign))


def test_degree_orientation_is_permutation_equivariant():
    """Relabelling the nodes relabels the induced orientation."""
    num_nodes = HETEROGENEOUS_N
    generator = torch.Generator().manual_seed(3)
    permutation = torch.randperm(num_nodes, generator=generator)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(num_nodes)
    relabelled = inverse[HETEROGENEOUS]

    def arcs(edge_index, relabel=None):
        _, _, pair, sign, _, _ = structure(
            edge_index, num_nodes, 0.25, orientation="degree"
        )
        out = set()
        for (a, b), s in zip(pair.t().tolist(), sign.tolist(), strict=True):
            if relabel is not None:
                a, b = int(relabel[a]), int(relabel[b])
            if s > 0:
                out.add((a, b))
            elif s < 0:
                out.add((b, a))
        return out

    assert arcs(relabelled) == arcs(HETEROGENEOUS, relabel=inverse)


def test_index_orientation_orients_every_edge():
    """The node-index potential never ties."""
    _, _, _, sign, _, _ = structure(
        UNDIRECTED, MIXED_N, 0.25, orientation="index"
    )
    assert bool((sign != 0).all())


def test_index_orientation_follows_node_order():
    """u -> v exactly when u < v."""
    _, _, pair, sign, _, _ = structure(
        UNDIRECTED, MIXED_N, 0.25, orientation="index"
    )
    for (a, b), s in zip(pair.t().tolist(), sign.tolist(), strict=True):
        assert a < b
        assert s == 1


def test_unknown_orientation_raises():
    """An unrecognised orientation is rejected."""
    support = laplace.symmetrize_support(UNDIRECTED, MIXED_N)
    _, pair = laplace.compute_left_right_map_index(support, MIXED_N)
    with pytest.raises(ValueError, match="orientation"):
        phase.pair_sign(UNDIRECTED, support, pair, MIXED_N, "bogus")


def test_unknown_node_potential_raises():
    """``node_potential`` rejects a mode it cannot build."""
    support = laplace.symmetrize_support(UNDIRECTED, MIXED_N)
    with pytest.raises(ValueError, match="induced orientation"):
        phase.node_potential(support, MIXED_N, "bogus")


def test_induced_sign_on_an_empty_graph():
    """No pairs gives no signs."""
    empty = torch.empty((2, 0), dtype=torch.long)
    sign = phase.induced_pair_sign(empty, empty, 3, "degree")
    assert sign.numel() == 0


def test_phase_sign_must_be_plus_or_minus_one():
    """Only the two conventions of Definition 1 are accepted."""
    with pytest.raises(ValueError, match="phase_sign"):
        phase.phase_from_sign(torch.zeros(1, dtype=torch.long), 0.25, 2)


# --------------------------------------------------------------------------- #
# orthogonal parameterization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("orth_map", ["cayley", "matrix_exp"])
@pytest.mark.parametrize("d", [2, 3, 4])
def test_orthogonal_maps_land_in_special_orthogonal_group(orth_map, d):
    """Both retractions produce rotations."""
    torch.manual_seed(0)
    params = torch.randn(6, orthogonal.num_orthogonal_params(d), dtype=D)
    blocks = orthogonal.Orthogonal(d, orth_map)(params)
    identity = torch.eye(d, dtype=D).expand(6, d, d)
    assert torch.allclose(
        blocks @ blocks.transpose(-1, -2), identity, atol=1e-10
    )
    assert torch.allclose(
        blocks.det(), torch.ones(6, dtype=D), atol=1e-10
    )


@pytest.mark.parametrize("d", [2, 3, 4])
def test_orthogonal_parameterization_has_no_dead_parameters(d):
    """Every predicted parameter influences the output.

    The skew-symmetrization erases the diagonal of the generator, so feeding
    ``d(d+1)/2`` parameters would leave ``d`` of them with zero gradient. We
    feed the strictly lower triangle instead.
    """
    params = torch.randn(
        4, orthogonal.num_orthogonal_params(d), dtype=D, requires_grad=True
    )
    orthogonal.Orthogonal(d, "cayley")(params).square().sum().backward()
    assert bool((params.grad.abs().sum(0) > 0).all())


def test_orthogonal_rejects_an_unknown_retraction():
    """Only the two supported retractions are accepted."""
    with pytest.raises(ValueError, match="orthogonal_map"):
        orthogonal.Orthogonal(3, "householder")


def test_orthogonal_rejects_the_wrong_parameter_count():
    """The parameter width is checked against d."""
    with pytest.raises(ValueError, match="orthogonal parameters"):
        orthogonal.Orthogonal(3, "cayley")(torch.randn(2, 6, dtype=D))


def test_orthogonal_repr_mentions_the_retraction():
    """The representation is informative."""
    assert "cayley" in repr(orthogonal.Orthogonal(3, "cayley"))


def test_block_inv_sqrt_inverts_and_pseudo_inverts():
    """Non-singular blocks invert; an all-zero block maps to zero."""
    torch.manual_seed(0)
    factor = torch.randn(3, 3, 3, dtype=D)
    blocks = factor @ factor.transpose(-1, -2) + 3 * torch.eye(3, dtype=D)
    blocks = torch.cat([blocks, torch.zeros(1, 3, 3, dtype=D)])
    inverse = laplacian_builders.block_inv_sqrt(blocks)
    reconstructed = inverse[:3] @ blocks[:3] @ inverse[:3]
    assert torch.allclose(
        reconstructed, torch.eye(3, dtype=D).expand(3, 3, 3), atol=1e-9
    )
    assert torch.equal(inverse[3], torch.zeros(3, 3, dtype=D))


def test_block_inv_sqrt_is_differentiable_through_a_zero_block():
    """An isolated node must not poison the gradient with a NaN."""
    torch.manual_seed(0)
    factor = torch.randn(2, 3, 3, dtype=D)
    good = factor @ factor.transpose(-1, -2) + 3 * torch.eye(3, dtype=D)
    blocks = torch.cat([good, torch.zeros(1, 3, 3, dtype=D)])
    blocks = blocks.clone().requires_grad_(True)
    laplacian_builders.block_inv_sqrt(blocks).square().sum().backward()
    assert bool(blocks.grad.isfinite().all())


# --------------------------------------------------------------------------- #
# sheaf learner
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("act", list(sheaf_models.SHEAF_ACTIVATIONS))
def test_sheaf_learner_supports_every_activation(act):
    """All activations of the paper's grid are reachable, ``relu`` included."""
    learner = sheaf_models.LocalConcatSheafLearner(4, (3,), sheaf_act=act)
    out = learner(torch.randn(5, 4), torch.tensor([[0, 1], [1, 2]]))
    assert out.shape == (2, 3)


def test_sheaf_learner_produces_matrix_shaped_output():
    """A two-dimensional ``out_shape`` yields per-edge matrices."""
    learner = sheaf_models.LocalConcatSheafLearner(4, (3, 3))
    out = learner(torch.randn(5, 4), torch.tensor([[0, 1], [1, 2]]))
    assert out.shape == (2, 3, 3)


def test_sheaf_learner_is_asymmetric_in_its_endpoints():
    """Phi(x_u || x_v) differs from Phi(x_v || x_u), so the maps differ."""
    torch.manual_seed(0)
    learner = sheaf_models.LocalConcatSheafLearner(4, (3,), sheaf_act="id")
    features = torch.randn(2, 4)
    forward = learner(features, torch.tensor([[0], [1]]))
    backward = learner(features, torch.tensor([[1], [0]]))
    assert not torch.allclose(forward, backward)


def test_sheaf_learner_rejects_a_bad_activation():
    """An unsupported activation name is refused."""
    with pytest.raises(ValueError, match="Unsupported act"):
        sheaf_models.LocalConcatSheafLearner(4, (3,), sheaf_act="bogus")


def test_sheaf_learner_rejects_a_bad_output_shape():
    """``out_shape`` must be one- or two-dimensional."""
    with pytest.raises(ValueError, match="out_shape"):
        sheaf_models.LocalConcatSheafLearner(4, (2, 2, 2))


def test_sheaf_learner_records_the_transport_maps():
    """``set_L`` stores a detached copy."""
    learner = sheaf_models.LocalConcatSheafLearner(4, (3,))
    weights = torch.randn(2, 3, requires_grad=True)
    learner.set_L(weights)
    assert learner.L is not None
    assert not learner.L.requires_grad


def test_sheaf_learner_base_class_is_abstract():
    """The base learner refuses to run."""
    with pytest.raises(NotImplementedError):
        sheaf_models.SheafLearner().forward(
            torch.zeros(1, 1), torch.zeros(2, 1, dtype=torch.long)
        )


def test_sheaf_learner_repr_mentions_its_shape():
    """The representation is informative."""
    text = repr(sheaf_models.LocalConcatSheafLearner(4, (3,)))
    assert "out_shape=(3,)" in text


def test_builder_base_class_is_abstract():
    """The base builder has no restriction-map algebra of its own."""
    struct = structure(UNDIRECTED, MIXED_N, 0.0)
    support, left_right, pair, _, cos, sin = struct
    builder = laplacian_builders.DirectedLaplacianBuilder(
        MIXED_N, support, 2, left_right, pair, cos, sin
    )
    with pytest.raises(NotImplementedError):
        builder.restriction_blocks(torch.zeros(support.size(1), 2, dtype=D))


def test_builder_repr_mentions_its_size():
    """The representation is informative."""
    struct = structure(UNDIRECTED, MIXED_N, 0.0)
    builder = laplacian_builders.DirectedDiagLaplacianBuilder(
        MIXED_N, struct[0], 2, struct[1], struct[2], struct[4], struct[5]
    )
    assert f"size={MIXED_N}" in repr(builder)


@pytest.mark.parametrize(
    ("family", "bad"),
    [
        ("diag", torch.zeros(2, 5, dtype=D)),
        ("bundle", torch.zeros(2, 5, dtype=D)),
        ("general", torch.zeros(2, 3, dtype=D)),
    ],
)
def test_builders_reject_wrong_map_shapes(family, bad):
    """Each family validates the width of the predicted parameters."""
    struct = structure(UNDIRECTED, MIXED_N, 0.0)
    support, left_right, pair, _, cos, sin = struct
    builder = laplacian_builders.LAPLACIAN_BUILDERS[family](
        MIXED_N, support, 3, left_right, pair, cos, sin
    )
    with pytest.raises(ValueError, match="maps"):
        builder.restriction_blocks(bad)


# --------------------------------------------------------------------------- #
# Eq. 8 -- the diffusion update rule
# --------------------------------------------------------------------------- #
def reference_equation_8(model, x, edge_index, unit_coeff=False):
    r"""One layer of Eq. 8, in dense complex arithmetic.

    Written against the equation rather than against the implementation: the
    operator is materialized densely, the stalk mixing is an explicit
    :math:`I_n \otimes W_1`, and every product is a complex matmul, so none of
    the real lifting of Appendix D is reused. What comes back is

    .. math::

        X^{(1)} = \mathrm{diag}(1 + \varepsilon) X^{(0)}
                  - \sigma\!\big(L^{\tilde{\mathcal{F}}}_N
                    (I_n \otimes W_1) X^{(0)} W_2\big),

    read out through :math:`(\Re \Vert \Im)`.

    Parameters
    ----------
    model : DSNNEncoder
        A one-layer encoder in evaluation mode and float64.
    x : torch.Tensor
        Node features of shape ``[num_nodes, input_dim]``.
    edge_index : torch.Tensor
        Raw arc indices of shape ``[2, num_edges]``.
    unit_coeff : bool, optional
        Force the residual coefficient to 1, giving the Eq. 7 update
        :math:`X^{(1)} = X^{(0)} - \sigma(\cdot)`. Default is False.

    Returns
    -------
    torch.Tensor
        Node features of shape ``[num_nodes, output_dim]``.
    """
    stack = model.sheaf_model
    num_nodes = x.size(0)
    size = num_nodes * stack.d
    builder, support = stack.build_builder(num_nodes, edge_index, x.dtype)

    hidden = torch.nn.functional.elu(stack.lin1(x)).view(size, -1)
    maps = stack.sheaf_learners[0](
        stack.sheaf_input(complex_ops.stack_real(hidden), num_nodes), support
    )
    real_i, real_v, imag_i, imag_v, _ = builder.hermitian_parts(maps)
    operator = complex_ops.hermitian_to_dense(
        real_i, real_v, imag_i, imag_v, size
    )

    x0 = hidden.to(CD)
    stalk_mix = torch.kron(
        torch.eye(num_nodes, dtype=CD),
        stack.lin_left_weights[0].weight.to(CD),
    )
    channel_mix = stack.lin_right_weights[0].weight.to(CD).T
    diffused = operator @ (stalk_mix @ x0 @ channel_mix)
    # sigma gates on the real part and zeroes both components (Sec. 3).
    gated = torch.where(
        diffused.real >= 0, diffused, torch.zeros_like(diffused)
    )

    coeff = (
        torch.ones(size, 1, dtype=CD)
        if unit_coeff
        else (1 + torch.tanh(stack.epsilons[0])).to(CD).tile(num_nodes, 1)
    )
    x1 = coeff * x0 - gated
    return stack.lin2(
        torch.cat(
            [
                x1.real.reshape(num_nodes, -1),
                x1.imag.reshape(num_nodes, -1),
            ],
            dim=1,
        )
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("q", [0.0, 0.25])
def test_equation_8_matches_a_dense_complex_reference(family, q):
    """The lifted, sparse layer computes exactly the update of Eq. 8."""
    torch.manual_seed(0)
    model = encoder(sheaf_type=family, num_layers=1, q=q).double().eval()
    with torch.no_grad():
        # A non-zero epsilon, so the residual term is actually exercised.
        model.sheaf_model.epsilons[0].uniform_(-0.5, 0.5)
    features = torch.randn(MIXED_N + 1, 12, dtype=D)
    assert torch.allclose(
        model(features, MIXED),
        reference_equation_8(model, features, MIXED),
        atol=1e-11,
    )


def test_equation_8_recovers_equation_7_at_zero_epsilon():
    """At ``epsilon = 0`` the residual collapses to the Eq. 7 update."""
    torch.manual_seed(0)
    model = encoder(num_layers=1).double().eval()
    assert bool((model.sheaf_model.epsilons[0] == 0).all())
    features = torch.randn(MIXED_N + 1, 12, dtype=D)
    assert torch.allclose(
        model(features, MIXED),
        reference_equation_8(model, features, MIXED, unit_coeff=True),
        atol=1e-11,
    )


@pytest.mark.parametrize("orientation", ["none", "degree"])
def test_encoder_is_permutation_equivariant(orientation):
    """Relabelling the nodes permutes the output and changes nothing else.

    ``index`` is deliberately excluded: it breaks ties using node identity, so
    it is documented as not equivariant.
    """
    torch.manual_seed(0)
    model = encoder(orientation=orientation).double().eval()
    features = torch.randn(HETEROGENEOUS_N, 12, dtype=D)
    perm = torch.randperm(
        HETEROGENEOUS_N, generator=torch.Generator().manual_seed(3)
    )
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(HETEROGENEOUS_N)
    relabelled = model(features[perm], inverse[HETEROGENEOUS])
    assert torch.allclose(
        model(features, HETEROGENEOUS)[perm], relabelled, atol=1e-11
    )


# --------------------------------------------------------------------------- #
# DSNNEncoder
# --------------------------------------------------------------------------- #
def test_encoder_default_attributes():
    """The constructor records its configuration."""
    model = encoder()
    assert model.sheaf_type == "diag"
    assert model.d == 2
    assert model.q == 0.25
    assert model.orientation == "none"
    assert model.num_layers == 2


def test_encoder_custom_attributes():
    """Non-default arguments reach the diffusion configuration."""
    model = encoder(
        sheaf_type="general",
        d=3,
        q=0.1,
        orientation="index",
        sheaf_features="real",
        num_layers=3,
    )
    assert model.sheaf_config["sheaf_type"] == "general"
    assert model.sheaf_config["hidden_channels"] == 8
    assert model.sheaf_config["layers"] == 3
    assert model.sheaf_config["sheaf_features"] == "real"


def test_encoder_repr_mentions_the_family():
    """The representation is informative."""
    assert "sheaf_type='diag'" in repr(encoder())


def test_encoder_exposes_the_diffusion_module():
    """``get_sheaf_model`` returns the inner stack."""
    model = encoder()
    assert model.get_sheaf_model() is model.sheaf_model
    assert "sheaf_type='diag'" in repr(model.sheaf_model)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("d", [2, 3])
def test_encoder_forward_shape(family, d):
    """The encoder maps node features to ``hidden_dim`` channels."""
    torch.manual_seed(0)
    model = encoder(sheaf_type=family, d=d).eval()
    features = torch.randn(MIXED_N + 1, 12)
    out = model(features, MIXED)
    assert out.shape == (MIXED_N + 1, 24)
    assert out.dtype == torch.float32
    assert bool(out.isfinite().all())


@pytest.mark.parametrize("family", ["diag", "general"])
def test_encoder_supports_unit_stalk_dimension(family):
    """A one-dimensional stalk is allowed outside the orthogonal family."""
    model = encoder(sheaf_type=family, d=1).eval()
    out = model(torch.randn(MIXED_N + 1, 12), MIXED)
    assert out.shape == (MIXED_N + 1, 24)


def test_encoder_ignores_edge_attributes_and_weights():
    """``edge_attr`` and ``edge_weight`` are accepted and have no effect."""
    torch.manual_seed(0)
    model = encoder().eval()
    features = torch.randn(MIXED_N + 1, 12)
    plain = model(features, MIXED)
    extra = model(
        features,
        MIXED,
        edge_attr=torch.randn(MIXED.size(1), 3),
        edge_weight=torch.rand(MIXED.size(1)),
        batch=torch.zeros(MIXED_N + 1, dtype=torch.long),
        unexpected=object(),
    )
    assert torch.equal(plain, extra)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("orientation", ["none", "degree", "index"])
def test_encoder_treats_a_batch_as_a_disjoint_union(family, orientation):
    """A batched graph gives each component the same output as on its own.

    The operator is block diagonal across components, so this holds exactly in
    exact arithmetic. It also guards against any structural computation --
    degrees or the induced orientation -- leaking across graphs in a batch.

    Run in float64. The claim is about block structure, not about a float32
    error budget, and the ``general`` family reaches ``block_inv_sqrt``, whose
    ``eigh`` returns an arbitrary basis on a degenerate degree block. The
    reassembled matrix function is basis independent, but the route to it is
    not, so the residual is LAPACK dependent: ~1e-6 locally against ~1e-5 on
    CI, either side of a float32 tolerance. In float64 it drops to ~1e-13 for
    every family, which tests the structure rather than the arithmetic.
    """
    torch.manual_seed(2)
    first = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    second = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    n_first, n_second = 3, 4
    x_first = torch.randn(n_first, 12, dtype=torch.float64)
    x_second = torch.randn(n_second, 12, dtype=torch.float64)
    model = (
        encoder(sheaf_type=family, d=2, orientation=orientation)
        .double()
        .eval()
    )
    joint = model(
        torch.cat([x_first, x_second]),
        torch.cat([first, second + n_first], dim=1),
    )
    assert torch.allclose(
        joint[:n_first], model(x_first, first), atol=1e-9
    )
    assert torch.allclose(
        joint[n_first:], model(x_second, second), atol=1e-9
    )


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("act", ["tanh", "relu"])
def test_encoder_gradients_reach_every_parameter(family, act):
    """Every parameter receives a finite, non-zero gradient.

    ``relu`` is included on purpose: it drives predicted map entries to exactly
    zero, which would silently break the gradient if the sparse pattern of the
    imaginary part were chosen from the values rather than from the phase.
    """
    torch.manual_seed(3)
    model = encoder(
        sheaf_type=family, d=2, sheaf_act=act, orientation="degree"
    )
    features = torch.randn(HETEROGENEOUS_N, 12, requires_grad=True)
    model(features, HETEROGENEOUS).square().sum().backward()
    assert bool(features.grad.isfinite().all())
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.isfinite().all()), name
        assert bool((parameter.grad != 0).any()), name


def test_encoder_handles_degenerate_graphs():
    """Empty edge sets, single nodes, self-loops and isolated nodes are fine."""
    model = encoder().eval()
    empty = torch.empty((2, 0), dtype=torch.long)
    features = torch.randn(MIXED_N + 1, 12)
    assert model(features, empty).shape == (MIXED_N + 1, 24)
    assert model(torch.randn(1, 12), empty).shape == (1, 24)
    loops = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 0, 2]])
    out = model(torch.randn(3, 12), loops)
    assert bool(out.isfinite().all())
    isolated = torch.tensor([[0, 1], [1, 0]])
    assert bool(model(features, isolated).isfinite().all())


def test_encoder_survives_heavy_dropout_in_training():
    """A high dropout rate must not produce non-finite activations."""
    torch.manual_seed(0)
    model = encoder(dropout=0.9, input_dropout=0.9).train()
    out = model(torch.randn(MIXED_N + 1, 12), MIXED)
    assert bool(out.isfinite().all())


def test_encoder_is_deterministic_in_evaluation():
    """Two forward passes agree once dropout is disabled."""
    torch.manual_seed(0)
    model = encoder().eval()
    features = torch.randn(MIXED_N + 1, 12)
    assert torch.equal(model(features, MIXED), model(features, MIXED))


def test_encoder_real_sheaf_features_setting():
    """``sheaf_features="real"`` halves the learner's input width."""
    model = encoder(sheaf_features="real", d=2).eval()
    learner = model.sheaf_model.sheaf_learners[0]
    assert learner.in_channels == model.sheaf_model.hidden_dim
    out = model(torch.randn(MIXED_N + 1, 12), MIXED)
    assert out.shape == (MIXED_N + 1, 24)


def test_encoder_matrix_exponential_retraction():
    """The alternative retraction runs end to end."""
    model = encoder(sheaf_type="bundle", d=3, orth="matrix_exp").eval()
    assert model(torch.randn(MIXED_N + 1, 12), MIXED).shape == (
        MIXED_N + 1,
        24,
    )


def test_encoder_unnormalised_and_shifted_variants_run():
    """The normalization flags are wired through to the builder."""
    for kwargs in (
        {"normalised": False},
        {"degree_shift": 1.0},
        {"block_norm": False, "sheaf_type": "general"},
        {"phase_sign": -1},
    ):
        model = encoder(**kwargs).eval()
        out = model(torch.randn(MIXED_N + 1, 12), MIXED)
        assert bool(out.isfinite().all())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sheaf_type": "bogus"}, "sheaf type"),
        ({"orientation": "bogus"}, "orientation"),
        ({"sheaf_features": "bogus"}, "sheaf_features"),
        ({"sheaf_type": "bundle", "d": 1}, "orthogonal family"),
        ({"hidden_dim": 1, "d": 4}, "too small"),
        ({"sheaf_act": "bogus"}, "Unsupported act"),
        ({"orth": "householder"}, "orth"),
        ({"num_layers": 0}, "layers"),
        ({"d": 0}, "d must be at least"),
    ],
)
def test_encoder_rejects_invalid_configuration(kwargs, match):
    """Every guarded constructor argument raises with a clear message."""
    with pytest.raises(ValueError, match=match):
        encoder(**kwargs)


def diffusion_config(**overrides):
    """A minimal valid configuration for the diffusion stack."""
    config = {
        "d": 2,
        "layers": 1,
        "hidden_channels": 4,
        "input_dim": 6,
        "output_dim": 8,
        "input_dropout": 0.0,
        "dropout": 0.0,
        "sheaf_act": "tanh",
        "sheaf_type": "diag",
        "sheaf_features": "unwind",
        "q": 0.25,
        "orientation": "none",
        "phase_sign": 1,
        "normalised": True,
        "degree_shift": 0.0,
        "block_norm": True,
        "orth": "cayley",
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"sheaf_type": "bogus"}, "sheaf type"),
        ({"sheaf_features": "bogus"}, "sheaf_features"),
        ({"orientation": "bogus"}, "orientation"),
        ({"orth": "householder"}, "orth"),
        ({"d": 0}, "d must be at least"),
        ({"sheaf_type": "bundle", "d": 1}, "orthogonal family"),
        ({"layers": 0}, "layers"),
    ],
)
def test_diffusion_stack_validates_its_own_configuration(overrides, match):
    """The diffusion stack checks its configuration independently.

    ``DSNNEncoder`` validates first, so these guards are unreachable through
    it; they matter because the stack is also usable on its own.
    """
    with pytest.raises(ValueError, match=match):
        discrete_models.DirectedSheafDiffusion(diffusion_config(**overrides))


def test_encoder_passes_the_repository_module_check():
    """The shared automatic module test accepts the encoder."""
    torch.manual_seed(0)
    features = torch.randn(8, 12)
    edges = torch.randint(0, 8, (2, 16))
    NNModuleAutoTest(
        [
            {
                "module": DSNNEncoder,
                "init": {
                    "input_dim": 12,
                    "hidden_dim": 24,
                    "num_layers": 2,
                    "d": 2,
                    "dropout": 0.0,
                    "input_dropout": 0.0,
                },
                "forward": (features, edges),
                "assert_shape": (8, 24),
            }
        ]
    ).run()


def test_encoder_learns_a_task_that_only_directions_reveal():
    """A charge helps when the label lives in the edge directions.

    A miniature of the paper's Fig. 2: node features carry no class signal, so
    only the orientation of the edges can separate the two groups. With the
    phase switched off (q = 0) the operator is real and the task is much
    harder.
    """
    torch.manual_seed(0)
    num_nodes = 60
    labels = (torch.arange(num_nodes) % 2).long()
    src, dst = [], []
    for u in range(num_nodes):
        for v in range(num_nodes):
            if u == v:
                continue
            if labels[u] == 0 and labels[v] == 1 and (u + v) % 5 == 0:
                src.append(u)
                dst.append(v)
    edge_index = torch.tensor([src, dst])
    features = torch.ones(num_nodes, 4)

    def accuracy(q):
        torch.manual_seed(0)
        model = DSNNEncoder(
            input_dim=4,
            hidden_dim=16,
            num_layers=2,
            d=2,
            q=q,
            dropout=0.0,
            input_dropout=0.0,
        )
        head = torch.nn.Linear(16, 2)
        optimiser = torch.optim.Adam(
            list(model.parameters()) + list(head.parameters()), lr=0.02
        )
        for _ in range(120):
            optimiser.zero_grad()
            logits = head(model(features, edge_index))
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            predicted = head(model(features, edge_index)).argmax(1)
        return (predicted == labels).double().mean().item()

    assert accuracy(0.25) >= accuracy(0.0)
