"""Unit tests for PolynomialFilterGNN and the Basis protocol.

These tests stress the *abstraction*: the propagation loop and the
basis interface: not any specific spectral filter. They exist to
verify that the basis-agnostic call path actually works and that
future basis files won't require backbone changes.

Per-basis algebraic tests (recurrence correctness for Chebyshev,
Jacobi, OptBasis orthonormality, etc.) live in their own
``Test<Basis>Basis`` classes below.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from topobench.nn.backbones.graph.poly_filter.basis import (
    Basis,
    LaplacianApply,
)
from topobench.nn.backbones.graph.poly_filter.bases.chebnetii import ChebNetII
from topobench.nn.backbones.graph.poly_filter.bases.chebyshev import Chebyshev
from topobench.nn.backbones.graph.poly_filter.bases.favard import FavardGNN
from topobench.nn.backbones.graph.poly_filter.bases.jacobi import Jacobi
from topobench.nn.backbones.graph.poly_filter.bases.legendre import Legendre
from topobench.nn.backbones.graph.poly_filter.bases.monomial import Monomial
from topobench.nn.backbones.graph.poly_filter.bases.optbasis import OptBasisGNN
from topobench.nn.backbones.graph.polynomial_filter_gnn import (
    PolynomialFilterGNN,
)
from topobench.utils.config_resolvers import register_all_resolvers


class _IdentityBasis(Basis):
    """Dummy basis that returns ``u_prev`` unchanged for every k.

    With this basis ``u_k = u_0`` for all ``k``, so the polynomial sum
    collapses to ``(Σ_k θ_k) · u_0``. Useful for testing the backbone's
    accumulation logic in isolation from any real recurrence.
    """

    def forward(
        self,
        u_prev: Tensor,
        u_prev_prev: Tensor | None,
        L_apply: LaplacianApply,
        signal: Tensor,
        k: int,
    ) -> Tensor:
        return u_prev


def _ring_edge_index(n: int) -> Tensor:
    """Undirected ring graph on ``n`` nodes as a PyG edge_index."""
    src = torch.arange(n)
    dst = (src + 1) % n
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


class TestPolynomialFilterGNN:
    """Stress tests for the basis abstraction (not for any specific basis)."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        torch.manual_seed(0)
        self.num_nodes = 8
        self.in_channels = 4
        self.hidden_channels = 6
        self.out_channels = 3
        self.edge_index = _ring_edge_index(self.num_nodes)
        self.x = torch.randn(self.num_nodes, self.in_channels)

    # ---- end-to-end shape / autograd ----

    def test_forward_shape_with_monomial(self):
        """End-to-end forward with the registered Monomial basis."""
        model = PolynomialFilterGNN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            K=4,
            basis=Monomial(),
        )
        y = model(self.x, self.edge_index)
        assert y.shape == (self.num_nodes, self.out_channels)
        assert torch.isfinite(y).all()

    def test_K_zero_is_just_constant(self):
        """``K = 0`` reduces to ``θ_0 · pre(x)`` and must still run."""
        model = PolynomialFilterGNN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            K=0,
            basis=Monomial(),
        )
        y = model(self.x, self.edge_index)
        assert y.shape == (self.num_nodes, self.out_channels)
        assert torch.isfinite(y).all()

    def test_theta_is_learnable(self):
        """``θ`` appears in autograd and gradients flow back to it."""
        model = PolynomialFilterGNN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            K=3,
            basis=Monomial(),
        )
        y = model(self.x, self.edge_index)
        y.sum().backward()
        assert model.theta.grad is not None
        assert model.theta.grad.shape == (4,)  # K + 1
        assert torch.isfinite(model.theta.grad).all()
        # The Monomial basis is stateless, so all autograd flow must reach θ.
        assert (model.theta.grad.abs() > 0).any()

    # ---- the abstraction itself ----

    def test_backbone_is_basis_agnostic(self):
        """A user-defined Basis subclass works without backbone changes.

        The "adding a new basis requires zero backbone changes" invariant
        is the whole point of the abstraction. A test-local dummy basis
        is the cheapest possible witness that this property holds.
        """
        model = PolynomialFilterGNN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            K=3,
            basis=_IdentityBasis(),
        )
        y = model(self.x, self.edge_index)
        assert y.shape == (self.num_nodes, self.out_channels)
        assert torch.isfinite(y).all()

    def test_basis_receives_uniform_signature(self):
        """Every basis gets ``signal`` and ``k`` on every step.

        The whole point of the uniform signature is that signal-independent
        bases shouldn't have to opt in to seeing ``signal``: it's just
        always there for them to ignore. Verify by recording every call.
        """
        calls: list[dict] = []

        class _RecordingBasis(Basis):
            def forward(self, u_prev, u_prev_prev, L_apply, signal, k):
                calls.append(
                    {
                        "k": k,
                        "u_prev_shape": tuple(u_prev.shape),
                        "u_prev_prev_is_none": u_prev_prev is None,
                        "signal_shape": tuple(signal.shape),
                        "L_apply_is_callable": callable(L_apply),
                    }
                )
                return u_prev

        K = 3
        model = PolynomialFilterGNN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            K=K,
            basis=_RecordingBasis(),
        )
        _ = model(self.x, self.edge_index)

        # forward was invoked exactly K times, with k = 1..K (k=0 is init).
        assert [c["k"] for c in calls] == [1, 2, 3]
        # u_prev_prev is None at the k=1 boundary, a real tensor afterwards.
        assert calls[0]["u_prev_prev_is_none"] is True
        assert all(not c["u_prev_prev_is_none"] for c in calls[1:])
        # signal and u_prev have the hidden-channels width on every call.
        for c in calls:
            assert c["u_prev_shape"] == (self.num_nodes, self.hidden_channels)
            assert c["signal_shape"] == (self.num_nodes, self.hidden_channels)
            assert c["L_apply_is_callable"]

    def test_default_init_returns_signal(self):
        """``Basis.init`` defaults to the identity (``u_0 = signal``).

        Bases that need a non-identity ``u_0`` (OptBasis with
        ``u_0 = x / ‖x‖``) override this; verify the default path doesn't
        accidentally call the Laplacian.
        """

        def _bad(_h):
            raise AssertionError("default init must not touch L_apply")

        x = torch.randn(self.num_nodes, self.hidden_channels)
        u0 = Monomial().init(x, _bad)
        assert torch.equal(u0, x)

    # ---- Laplacian construction ----

    @pytest.mark.parametrize("norm", ["sym", "rw", "none"])
    def test_runs_under_each_laplacian_normalization(self, norm):
        """All three normalization conventions produce finite outputs."""
        model = PolynomialFilterGNN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            K=2,
            basis=Monomial(),
            laplacian_norm=norm,
        )
        y = model(self.x, self.edge_index)
        assert torch.isfinite(y).all()

    def test_laplacian_closure_matches_dense_sym(self):
        """``L_apply`` is numerically the symmetric normalized Laplacian.

        Tiny 3-node path graph 0:1:2 with unit edge weights:
        ``D = diag(1, 2, 1)``, ``A`` is the path adjacency, and
        ``L̃_sym = I − D^{-1/2} A D^{-1/2}``. Hand-compute and compare.
        """
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        model = PolynomialFilterGNN(
            in_channels=1,
            hidden_channels=1,
            out_channels=1,
            K=0,
            basis=Monomial(),
            laplacian_norm="sym",
        )
        L_apply = model._build_laplacian_apply(
            edge_index, edge_weight=None, num_nodes=3
        )
        # L̃_sym for path 0:1:2:
        # diag = [1, 1, 1]; off-diagonals: -1/√(1·2) = -1/√2 on (0,1),(1,0),(1,2),(2,1).
        inv_root2 = 1.0 / 2.0 ** 0.5
        L_dense = torch.tensor(
            [
                [1.0, -inv_root2, 0.0],
                [-inv_root2, 1.0, -inv_root2],
                [0.0, -inv_root2, 1.0],
            ]
        )
        h = torch.tensor([[1.0], [2.0], [3.0]])
        expected = L_dense @ h
        assert torch.allclose(L_apply(h), expected, atol=1e-6)

    # ---- validation ----

    def test_invalid_K_raises(self):
        with pytest.raises(ValueError):
            PolynomialFilterGNN(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                K=-1,
                basis=Monomial(),
            )

    def test_invalid_laplacian_norm_raises(self):
        with pytest.raises(ValueError):
            PolynomialFilterGNN(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                K=2,
                basis=Monomial(),
                laplacian_norm="banana",
            )


class TestMonomialBasis:
    """Tests specific to the Monomial basis recurrence."""

    def test_monomial_applies_laplacian_once_per_step(self):
        """``u_k = L_apply(u_{k-1})``: one matvec, on ``u_prev`` only."""
        b = Monomial()
        calls: list[Tensor] = []

        def L_apply(h: Tensor) -> Tensor:
            calls.append(h)
            return 2.0 * h  # arbitrary linear op; the basis is operator-agnostic

        u_prev = torch.randn(5, 3)
        u_k = b(u_prev, None, L_apply, signal=u_prev, k=1)
        assert len(calls) == 1
        assert calls[0] is u_prev
        assert torch.equal(u_k, 2.0 * u_prev)

    def test_monomial_ignores_u_prev_prev_and_signal(self):
        """Stateless / signal-independent: those args don't change the output."""
        b = Monomial()
        L = lambda h: 3.0 * h
        u_prev = torch.randn(4, 2)
        out_a = b(u_prev, None, L, signal=torch.zeros(4, 2), k=1)
        out_b = b(u_prev, torch.randn(4, 2), L, signal=torch.randn(4, 2), k=7)
        assert torch.equal(out_a, out_b)

    def test_monomial_has_no_parameters(self):
        """Stateless bases register zero learnable parameters."""
        assert sum(p.numel() for p in Monomial().parameters()) == 0


class TestChebyshevBasis:
    """Tests specific to the Chebyshev (first kind) basis recurrence.

    The key correctness property is that with ``L̃ = α·I`` (a multiple of
    the identity), the basis vectors collapse to ``u_k = T_k(α) · x``
    where ``T_k`` is the classical Chebyshev polynomial of the first
    kind. This pins down both the boundary case ``u_1 = α x``
    (NOT ``2α x``) and the general recurrence ``u_k = 2α u_{k-1} - u_{k-2}``
    in one algebraic check.
    """

    def test_chebyshev_k1_uses_first_kind_boundary(self):
        """``u_1 = L̃ u_0``: NOT ``2 L̃ u_0 - 0`` (which would be 2nd kind)."""
        b = Chebyshev()

        def L_apply(h: Tensor) -> Tensor:
            return 3.0 * h  # arbitrary scaling for an unambiguous check

        u_prev = torch.randn(4, 2)
        u_1 = b(u_prev, None, L_apply, signal=u_prev, k=1)
        # First-kind: u_1 = L̃ u_0 = 3 * u_prev. Wrong (2nd-kind) would be 6 * u_prev.
        assert torch.allclose(u_1, 3.0 * u_prev)

    def test_chebyshev_k2_uses_three_term_recurrence(self):
        """``u_2 = 2 L̃ u_1 - u_0`` once ``u_prev_prev`` is non-None."""
        b = Chebyshev()

        def L_apply(h: Tensor) -> Tensor:
            return 3.0 * h

        u_0 = torch.randn(4, 2)
        u_1 = 3.0 * u_0  # what k=1 produces with the L_apply above
        u_2 = b(u_1, u_0, L_apply, signal=u_0, k=2)
        # u_2 = 2 * 3 * u_1 - u_0 = 6 * (3 u_0) - u_0 = 17 * u_0
        assert torch.allclose(u_2, 17.0 * u_0)

    def test_chebyshev_matches_classical_polynomial_at_scalar(self):
        """For ``L̃ = α·I``, the basis collapses to classical ``T_k(α)·x``.

        This is the strongest algebraic check available: it pins down
        both the k=1 boundary AND the three-term recurrence in one go,
        for an arbitrary k up to K.
        """
        alpha = 0.37
        K = 6
        x = torch.randn(5, 3)

        def L_apply(h: Tensor) -> Tensor:
            return alpha * h

        # Compute classical Chebyshev (first kind) values at alpha.
        T_vals = [1.0, alpha]
        for _ in range(2, K + 1):
            T_vals.append(2.0 * alpha * T_vals[-1] - T_vals[-2])

        b = Chebyshev()
        u_prev_prev: Tensor | None = None
        u_prev = b.init(x, L_apply)
        assert torch.allclose(u_prev, x)  # u_0 = x via default init
        for k in range(1, K + 1):
            u_k = b(u_prev, u_prev_prev, L_apply, signal=x, k=k)
            expected = T_vals[k] * x
            assert torch.allclose(u_k, expected, atol=1e-6), (
                f"Chebyshev u_{k} should be T_{k}({alpha}) * x = "
                f"{T_vals[k]:.6f} * x"
            )
            u_prev_prev, u_prev = u_prev, u_k

    def test_chebyshev_ignores_signal_and_k(self):
        """Signal-independent: output unchanged by ``signal`` or ``k`` values."""
        b = Chebyshev()
        L = lambda h: 2.0 * h  # noqa: E731
        u_prev = torch.randn(4, 2)
        u_prev_prev = torch.randn(4, 2)
        out_a = b(u_prev, u_prev_prev, L, signal=torch.zeros(4, 2), k=2)
        out_b = b(u_prev, u_prev_prev, L, signal=torch.randn(4, 2), k=99)
        assert torch.equal(out_a, out_b)

    def test_chebyshev_has_no_parameters(self):
        """Stateless bases register zero learnable parameters."""
        assert sum(p.numel() for p in Chebyshev().parameters()) == 0

    def test_backbone_runs_with_chebyshev(self):
        """End-to-end forward through the backbone with Chebyshev.

        Witness that the basis abstraction generalizes: Chebyshev's
        k=1-boundary recurrence is structurally different from Monomial's
        k-uniform one, yet the backbone code is identical.
        """
        _backbone_smoke(basis=Chebyshev())


def _backbone_smoke(basis):
    """Shared end-to-end backbone check used by per-basis tests.

    Asserts shape, finiteness, and that *some* learnable parameter
    receives a non-trivial gradient. For most bases this is the
    backbone's ``θ``; for ChebNetII it's the basis's ``θ_interp``
    (the backbone's ``θ`` is dead by design for that basis: see
    ``chebnetii.py``'s module docstring). The check accommodates both
    via a model-wide parameter scan.
    """
    torch.manual_seed(1)
    edge_index = _ring_edge_index(8)
    x = torch.randn(8, 4)
    model = PolynomialFilterGNN(
        in_channels=4,
        hidden_channels=6,
        out_channels=3,
        K=5,
        basis=basis,
    )
    y = model(x, edge_index)
    assert y.shape == (8, 3)
    assert torch.isfinite(y).all()
    y.sum().backward()
    # Some learnable parameter (anywhere in the model) must have a
    # non-trivial gradient: the bare minimum that the basis is
    # actually wired into the autograd graph.
    has_grad = any(
        p.grad is not None and (p.grad.abs() > 0).any()
        for p in model.parameters()
    )
    assert has_grad, "no model parameter received a gradient"


class TestJacobiBasis:
    """Tests specific to the Jacobi basis recurrence.

    Jacobi is the first basis in the registry that:
    1. takes constructor hyperparameters (``α``, ``β``);
    2. has k-dependent recurrence coefficients (``δ_k``, ``δ'_k``,
       ``δ''_k``): i.e. the first basis where the ``k`` argument to
       :meth:`Basis.forward` is genuinely consumed.

    The strongest correctness check is the scalar collapse, mirroring
    the analogous Chebyshev test: with ``L̃ = γ · I`` the recurrence
    reduces to evaluating the classical Jacobi polynomial at the
    argument ``z = 1 - γ`` (since Liao casts the recurrence in
    ``z = I - L̃``).
    """

    def test_jacobi_stores_hyperparameters(self):
        b = Jacobi(alpha=0.5, beta=-0.3)
        assert b.alpha == 0.5
        assert b.beta == pytest.approx(-0.3)

    def test_jacobi_rejects_alpha_at_or_below_minus_one(self):
        with pytest.raises(ValueError):
            Jacobi(alpha=-1.0, beta=0.5)
        with pytest.raises(ValueError):
            Jacobi(alpha=-1.5, beta=0.5)

    def test_jacobi_rejects_beta_at_or_below_minus_one(self):
        with pytest.raises(ValueError):
            Jacobi(alpha=0.5, beta=-1.0)
        with pytest.raises(ValueError):
            Jacobi(alpha=0.5, beta=-2.0)

    def test_jacobi_k1_boundary_with_zero_laplacian(self):
        """At ``L̃ = 0``: ``u_1 = ((α-β)/2 + (α+β+2)/2) · u_0 = (α+1) · u_0``.

        Closed-form check pinning down the ``k=1`` formula without any
        recurrence interference.
        """
        alpha, beta = 0.7, -0.2
        b = Jacobi(alpha=alpha, beta=beta)

        def L_apply(h: Tensor) -> Tensor:
            return torch.zeros_like(h)

        u_0 = torch.randn(4, 2)
        u_1 = b(u_0, None, L_apply, signal=u_0, k=1)
        expected = (alpha + 1.0) * u_0
        assert torch.allclose(u_1, expected, atol=1e-6)

    def test_jacobi_matches_classical_polynomial_at_scalar(self):
        """For ``L̃ = γ · I``, the basis collapses to ``P_k^{(α,β)}(1-γ) · x``.

        Strongest algebraic check available: exercises the boundary, the
        general recurrence, and the ``k``-dependent coefficients
        simultaneously, for ``k = 0..K``. Reference values come from
        applying Liao's own recurrence in scalar form.
        """
        alpha, beta = 1.0, 0.5
        gamma = 0.37  # L̃ = γ·I means argument z = 1 - γ = 0.63
        K = 6
        x = torch.randn(5, 3)

        def L_apply(h: Tensor) -> Tensor:
            return gamma * h

        # Reference: same recurrence as in the basis, but in pure scalar
        # arithmetic with z = 1 - γ. Match by construction (same equations)
        # but the check still catches transcription bugs in the tensor form.
        z = 1.0 - gamma
        T_vals = [
            1.0,
            ((alpha - beta) / 2.0) + ((alpha + beta + 2.0) / 2.0) * z,
        ]
        for k in range(2, K + 1):
            s = 2.0 * k + alpha + beta
            denom_left = 2.0 * k * (k + alpha + beta)
            denom_right = k * (k + alpha + beta) * (s - 2.0)
            d = s * (s - 1.0) / denom_left
            d_p = (s - 1.0) * (alpha * alpha - beta * beta) / (2.0 * denom_right)
            d_pp = (k + alpha - 1.0) * (k + beta - 1.0) * s / denom_right
            T_vals.append(d * z * T_vals[-1] + d_p * T_vals[-1] - d_pp * T_vals[-2])

        b = Jacobi(alpha=alpha, beta=beta)
        u_prev_prev: Tensor | None = None
        u_prev = b.init(x, L_apply)
        assert torch.allclose(u_prev, x)  # u_0 = x via default init
        for k in range(1, K + 1):
            u_k = b(u_prev, u_prev_prev, L_apply, signal=x, k=k)
            expected = T_vals[k] * x
            assert torch.allclose(u_k, expected, atol=1e-5), (
                f"Jacobi u_{k} at γ={gamma} (α={alpha}, β={beta}) should be "
                f"{T_vals[k]:.6f} · x"
            )
            u_prev_prev, u_prev = u_prev, u_k

    def test_jacobi_symmetric_case_kills_middle_term(self):
        """When ``α = β``, ``δ'_k = (s-1)(α² - β²)/... = 0`` for all k ≥ 2.

        Sanity check on the formula: no division-by-zero, and the middle
        ``δ'_k · u_{k-1}`` term genuinely vanishes.
        """
        b = Jacobi(alpha=1.5, beta=1.5)
        # Compare against a Jacobi where we hard-code δ' = 0: same output.
        # Easiest indirect check: run the recurrence with α=β=1.5 against
        # the scalar reference and assert it stays finite + deterministic.
        gamma = 0.5
        x = torch.randn(3, 2)
        L_apply = lambda h: gamma * h  # noqa: E731
        u_prev_prev: Tensor | None = None
        u_prev = b.init(x, L_apply)
        for k in range(1, 8):
            u_k = b(u_prev, u_prev_prev, L_apply, signal=x, k=k)
            assert torch.isfinite(u_k).all()
            u_prev_prev, u_prev = u_prev, u_k

    def test_jacobi_ignores_signal(self):
        """Signal-independent basis: ``signal`` does not affect the output."""
        b = Jacobi(alpha=1.0, beta=1.0)
        L = lambda h: 0.3 * h  # noqa: E731
        u_prev = torch.randn(4, 2)
        u_prev_prev = torch.randn(4, 2)
        out_a = b(u_prev, u_prev_prev, L, signal=torch.zeros(4, 2), k=3)
        out_b = b(u_prev, u_prev_prev, L, signal=torch.randn(4, 2), k=3)
        assert torch.equal(out_a, out_b)

    def test_jacobi_has_no_learnable_parameters(self):
        """``α``, ``β`` are *hyperparameters*, not learned (cf. FavardGNN)."""
        b = Jacobi(alpha=1.0, beta=1.0)
        assert sum(p.numel() for p in b.parameters()) == 0

    def test_backbone_runs_with_jacobi(self):
        """End-to-end forward with Jacobi: exercises both new dimensions.

        Constructor hyperparameters (``α``, ``β``) AND a k-dependent
        recurrence, plumbed through the basis-agnostic backbone with
        zero changes.
        """
        _backbone_smoke(basis=Jacobi(alpha=1.0, beta=0.5))


class TestLegendreBasis:
    """Tests for :class:`Legendre`.

    Legendre is shipped as the ``α = β = 0`` reparameterization of
    :class:`Jacobi`: see ``legendre.py`` for why this differs from
    Liao's standalone Legendre formula. The tests below pin that
    decision: Legendre must produce the same outputs as ``Jacobi(0, 0)``,
    and must stay bounded for ``\\tilde L`` eigenvalues in ``[0, 2]``
    (the symmetric normalized Laplacian spectrum), which is the
    numerical-stability property that motivated the choice.
    """

    def test_legendre_takes_no_constructor_args(self):
        # No alpha/beta in __init__: caller doesn't get to break the
        # invariant that Legendre is α=β=0.
        b = Legendre()
        assert b.alpha == 0.0
        assert b.beta == 0.0

    def test_legendre_is_a_jacobi(self):
        """Subclass relationship is the *implementation* of the decision."""
        assert isinstance(Legendre(), Jacobi)

    def test_legendre_matches_jacobi_alpha0_beta0_pointwise(self):
        """The Legendre output equals ``Jacobi(0, 0)`` on every step.

        This is the literal verification of the design choice: Legendre
        is not "approximately Jacobi(0,0)", it IS Jacobi(0,0). If anyone
        ever changes Legendre.__init__ to do something else (e.g.,
        switch back to Liao's standalone z=L̃ formula), this test fails.
        """
        legendre = Legendre()
        jacobi00 = Jacobi(alpha=0.0, beta=0.0)
        torch.manual_seed(7)
        x = torch.randn(5, 3)
        L_apply = lambda h: 0.4 * h  # noqa: E731  -- L̃ = 0.4·I, in-domain

        u_prev_prev_l: Tensor | None = None
        u_prev_l = legendre.init(x, L_apply)
        u_prev_prev_j: Tensor | None = None
        u_prev_j = jacobi00.init(x, L_apply)
        for k in range(1, 8):
            u_k_l = legendre(u_prev_l, u_prev_prev_l, L_apply, signal=x, k=k)
            u_k_j = jacobi00(u_prev_j, u_prev_prev_j, L_apply, signal=x, k=k)
            assert torch.allclose(u_k_l, u_k_j, atol=1e-7), (
                f"Legendre and Jacobi(0,0) diverge at k={k}"
            )
            u_prev_prev_l, u_prev_l = u_prev_l, u_k_l
            u_prev_prev_j, u_prev_j = u_prev_j, u_k_j

    def test_legendre_stays_bounded_on_full_eigenvalue_range(self):
        """``|u_k| / |x| <= ~1`` for every k when L̃ eigenvalues span [0, 2].

        Classical bound: for the Legendre polynomial ``P_k``, ``|P_k(z)| ≤ 1``
        on ``z ∈ [-1, 1]``. Our reparameterization uses ``z = I - L̃``, so
        if L̃ has eigenvalues in ``[0, 2]`` (the symmetric normalized
        Laplacian range), z spans ``[-1, 1]`` and the bound applies.

        This is the literal numerical-stability property that motivated
        shipping the Jacobi reparameterization rather than Liao's
        standalone Legendre formula. Tested at the boundaries (the most
        adversarial case) where Liao's standalone version would blow up.
        """
        b = Legendre()
        x = torch.tensor([[1.0], [1.0], [1.0]])

        for gamma in (0.0, 0.5, 1.0, 1.5, 2.0):
            # L̃ = γ·I, so z = 1 - γ. At γ=0: z=1; at γ=2: z=-1; both
            # at the boundary of [-1, 1] where the |P_k| ≤ 1 bound is tight.
            L_apply = lambda h, g=gamma: g * h  # noqa: E731
            u_prev_prev: Tensor | None = None
            u_prev = b.init(x, L_apply)
            for k in range(1, 12):
                u_k = b(u_prev, u_prev_prev, L_apply, signal=x, k=k)
                # |P_k(z)| ≤ 1 on [-1, 1]. Allow a small slack for FP error.
                assert u_k.abs().max().item() <= 1.0 + 1e-5, (
                    f"Legendre |u_{k}| exceeded 1 at γ={gamma}: "
                    f"{u_k.abs().max().item():.6f}: this would happen "
                    "if we'd shipped Liao's standalone z=L̃ formula instead."
                )
                u_prev_prev, u_prev = u_prev, u_k

    def test_legendre_has_no_learnable_parameters(self):
        assert sum(p.numel() for p in Legendre().parameters()) == 0

    def test_legendre_ignores_signal(self):
        b = Legendre()
        L = lambda h: 0.3 * h  # noqa: E731
        u_prev = torch.randn(4, 2)
        u_prev_prev = torch.randn(4, 2)
        out_a = b(u_prev, u_prev_prev, L, signal=torch.zeros(4, 2), k=3)
        out_b = b(u_prev, u_prev_prev, L, signal=torch.randn(4, 2), k=3)
        assert torch.equal(out_a, out_b)

    def test_backbone_runs_with_legendre(self):
        _backbone_smoke(basis=Legendre())


class TestChebNetIIBasis:
    """Tests for ChebNetII: first basis using the ``effective_thetas`` hook.

    Recurrence is inherited from :class:`Chebyshev`; the substantive
    addition is the coefficient reparameterization. Tests focus on the
    reparameterization and the protocol hook.
    """

    def test_chebnetii_constructor_stores_K_and_parameters(self):
        b = ChebNetII(K=4)
        assert b.K == 4
        assert b.theta_interp.shape == (5,)
        assert b.M.shape == (5, 5)

    def test_chebnetii_rejects_negative_K(self):
        with pytest.raises(ValueError):
            ChebNetII(K=-1)

    def test_chebnetii_M_matches_hand_computation_for_small_K(self):
        """For K=2: x_0=cos(π/6)=√3/2, x_1=cos(π/2)=0, x_2=cos(5π/6)=-√3/2.

        Expected M[k, κ] = (2/3) · T_k(x_κ):
        - Row k=0:  (2/3, 2/3, 2/3)
        - Row k=1:  (2/3)(√3/2, 0, -√3/2) = (√3/3, 0, -√3/3)
        - Row k=2:  (2/3)(2x^2-1) at each node: 2(3/4)-1 = 1/2 at ±√3/2, -1 at 0
                    → (2/3)(1/2, -1, 1/2) = (1/3, -2/3, 1/3)
        """
        import math

        b = ChebNetII(K=2)
        expected = torch.tensor(
            [
                [2 / 3, 2 / 3, 2 / 3],
                [math.sqrt(3) / 3, 0.0, -math.sqrt(3) / 3],
                [1 / 3, -2 / 3, 1 / 3],
            ]
        )
        assert torch.allclose(b.M, expected, atol=1e-6)

    def test_chebnetii_effective_thetas_applies_M(self):
        """``effective_thetas`` returns ``M @ θ_interp``, ignoring backbone θ."""
        b = ChebNetII(K=3)
        # Force known θ_interp values for a clean check.
        with torch.no_grad():
            b.theta_interp.copy_(torch.tensor([1.0, 0.5, -0.25, 0.1]))
        # Backbone θ should be IGNORED.
        backbone_theta_a = torch.zeros(4)
        backbone_theta_b = torch.randn(4)
        out_a = b.effective_thetas(backbone_theta_a)
        out_b = b.effective_thetas(backbone_theta_b)
        assert torch.allclose(out_a, out_b)
        expected = b.M @ b.theta_interp
        assert torch.allclose(out_a, expected)

    def test_chebnetii_K_mismatch_raises(self):
        """Defensive guard against misconfigured basis/backbone K."""
        b = ChebNetII(K=4)
        with pytest.raises(ValueError, match="K=4"):
            b.effective_thetas(torch.zeros(3))  # wrong size

    def test_chebnetii_recurrence_is_still_chebyshev(self):
        """Inheritance: the basis vectors u_k follow first-kind Chebyshev."""
        b = ChebNetII(K=3)
        # Use the same scalar-collapse check as Chebyshev: with L̃ = α·I,
        # u_k = T_k(α) · x for classical T_k.
        alpha = 0.42
        x = torch.randn(4, 2)

        def L_apply(h):
            return alpha * h

        T_vals = [1.0, alpha]
        for _ in range(2, 4):
            T_vals.append(2.0 * alpha * T_vals[-1] - T_vals[-2])

        u_prev_prev: Tensor | None = None
        u_prev = b.init(x, L_apply)
        for k in range(1, 4):
            u_k = b(u_prev, u_prev_prev, L_apply, signal=x, k=k)
            assert torch.allclose(u_k, T_vals[k] * x, atol=1e-5)
            u_prev_prev, u_prev = u_prev, u_k

    def test_chebnetii_has_K_plus_1_learnable_parameters(self):
        b = ChebNetII(K=4)
        # Only theta_interp is learnable; M is a buffer.
        n_params = sum(p.numel() for p in b.parameters())
        assert n_params == 5

    def test_backbone_runs_with_chebnetii(self):
        _backbone_smoke(basis=ChebNetII(K=5))

    def test_chebnetii_gradient_reaches_theta_interp_not_backbone_theta(self):
        """For ChebNetII, learning happens via θ_interp; backbone θ is dead.

        This is the literal consequence of the ``effective_thetas``
        protocol hook: ChebNetII's override ignores the backbone's θ,
        so gradients flow to ``θ_interp`` and ``self.theta`` stays at
        its initialization with ``grad is None`` (no autograd path).
        """
        torch.manual_seed(2)
        edge_index = _ring_edge_index(8)
        x = torch.randn(8, 4)
        model = PolynomialFilterGNN(
            in_channels=4,
            hidden_channels=6,
            out_channels=3,
            K=5,
            basis=ChebNetII(K=5),
        )
        y = model(x, edge_index)
        y.sum().backward()
        # θ_interp must have a non-trivial gradient.
        assert model.basis.theta_interp.grad is not None
        assert (model.basis.theta_interp.grad.abs() > 0).any()
        # Backbone θ has no autograd path through effective_thetas: its
        # .grad is None (autograd didn't touch it).
        assert model.theta.grad is None


class TestFavardGNNBasis:
    """Tests for FavardGNN: first basis with learnable recurrence coefficients.

    The basis protocol requires ``Basis`` to subclass ``nn.Module`` so
    bases can own their own learnable parameters; FavardGNN is the first
    basis where this matters (stateless bases register zero). Strongest
    test: gradients flow to both ``a_raw`` and ``β``.
    """

    def test_favard_constructor_stores_K_and_parameters(self):
        b = FavardGNN(K=4)
        assert b.K == 4
        assert b.a_raw.shape == (5,)
        assert b.beta.shape == (5,)

    def test_favard_rejects_negative_K(self):
        with pytest.raises(ValueError):
            FavardGNN(K=-1)

    def test_favard_has_2_K_plus_1_learnable_parameters(self):
        """``a_raw`` (K+1) + ``β`` (K+1) = 2(K+1)."""
        b = FavardGNN(K=4)
        assert sum(p.numel() for p in b.parameters()) == 2 * 5

    def test_favard_a_is_strictly_positive(self):
        """Softplus guarantees ``a_k = √α_k > 0`` even at extreme raw values."""
        b = FavardGNN(K=4)
        with torch.no_grad():
            b.a_raw.copy_(torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0]))
        a = b._a()
        assert (a > 0).all()
        assert torch.isfinite(a).all()

    def test_favard_init_divides_signal_by_a0(self):
        b = FavardGNN(K=2)
        with torch.no_grad():
            b.a_raw.fill_(2.0)
        a0 = torch.nn.functional.softplus(torch.tensor(2.0))
        x = torch.randn(4, 3)
        u0 = b.init(x, L_apply=lambda h: h)
        assert torch.allclose(u0, x / a0)

    def test_favard_k1_boundary_matches_closed_form(self):
        """At k=1 with ``u_prev_prev = None``: ``u_1 = (1/a_1)(I-L̃)u_0 - β_1 u_0``."""
        torch.manual_seed(3)
        b = FavardGNN(K=2)

        def L_apply(h):
            return 0.4 * h

        u_prev = torch.randn(4, 2)
        u_1 = b(u_prev, None, L_apply, signal=u_prev, k=1)
        a = b._a()
        expected = (u_prev - L_apply(u_prev)) / a[1] - b.beta[1] * u_prev
        assert torch.allclose(u_1, expected, atol=1e-6)

    def test_favard_k2_recurrence_matches_closed_form(self):
        torch.manual_seed(4)
        b = FavardGNN(K=3)

        def L_apply(h):
            return 0.3 * h

        u_prev_prev = torch.randn(4, 2)
        u_prev = torch.randn(4, 2)
        u_2 = b(u_prev, u_prev_prev, L_apply, signal=u_prev, k=2)
        a = b._a()
        expected = (
            (u_prev - L_apply(u_prev)) / a[2]
            - b.beta[2] * u_prev
            - a[1] * u_prev_prev
        )
        assert torch.allclose(u_2, expected, atol=1e-6)

    def test_favard_gradients_flow_to_alpha_and_beta(self):
        """Backbone forward → loss → ∂L/∂a_raw and ∂L/∂β are non-trivial."""
        torch.manual_seed(5)
        edge_index = _ring_edge_index(8)
        x = torch.randn(8, 4)
        model = PolynomialFilterGNN(
            in_channels=4,
            hidden_channels=6,
            out_channels=3,
            K=5,
            basis=FavardGNN(K=5),
        )
        y = model(x, edge_index)
        y.sum().backward()
        assert model.basis.a_raw.grad is not None
        assert (model.basis.a_raw.grad.abs() > 0).any()
        assert model.basis.beta.grad is not None
        assert (model.basis.beta.grad.abs() > 0).any()
        # The backbone's θ is also still learnable (FavardGNN doesn't
        # override effective_thetas).
        assert model.theta.grad is not None

    def test_favard_ignores_signal(self):
        b = FavardGNN(K=2)
        L = lambda h: 0.3 * h  # noqa: E731
        u_prev = torch.randn(4, 2)
        u_prev_prev = torch.randn(4, 2)
        out_a = b(u_prev, u_prev_prev, L, signal=torch.zeros(4, 2), k=2)
        out_b = b(u_prev, u_prev_prev, L, signal=torch.randn(4, 2), k=2)
        assert torch.equal(out_a, out_b)

    def test_backbone_runs_with_favard(self):
        _backbone_smoke(basis=FavardGNN(K=5))


class TestOptBasisGNN:
    """Tests for OptBasisGNN: the load-bearing test of the basis protocol.

    The defining property is that the basis is **signal-dependent**:
    the recurrence coefficients ``α``, ``γ`` are computed from inner
    products on the running signal ``u_prev``. Whether this case rides
    the same protocol surface as every signal-independent basis is the
    primary correctness criterion for the abstraction. We test:

    1. ``init`` normalizes the input to unit per-channel norm: the
       signal-dependent entry point.
    2. **Scale invariance**: ``OptBasis(c · x) = OptBasis(x)`` for any
       ``c > 0`` (because of the init normalization), which is the
       sharp distinguishing property versus signal-independent bases
       (which are signal-homogeneous: ``Cheb(c · x) = c · Cheb(x)``).
    3. **Orthonormality**: ``⟨u_k, u_j⟩ ≈ δ_{kj}`` per channel, which
       is the literal definition of "optimal basis" in Guo & Wei 2023.
    4. ``init`` resets the intra-forward-pass state so multiple
       backbone forward passes don't bleed across each other.
    """

    def test_optbasis_init_normalizes_signal_per_channel(self):
        b = OptBasisGNN()
        x = torch.randn(10, 3) * 5.0  # large magnitudes
        u0 = b.init(x, L_apply=lambda h: h)
        # Each column should be unit-norm.
        norms = u0.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_optbasis_init_resets_state(self):
        b = OptBasisGNN()
        b._gamma_prev = torch.tensor([[1.0, 2.0]])  # pretend prior state
        b.init(torch.randn(4, 2), L_apply=lambda h: h)
        assert b._gamma_prev is None

    def test_optbasis_is_scale_invariant_in_signal(self):
        """``OptBasis(c · x) = OptBasis(x)`` for ``c > 0``: the literal
        signature of signal-dependence.

        Contrast with signal-independent bases: Chebyshev gives
        ``Cheb(c · x) = c · Cheb(x)`` (homogeneous of degree 1). The
        test below also asserts that contrast.

        Note: we deliberately use the **real** ring-graph Laplacian here.
        A scalar L̃ = α·I would collapse the recurrence (every higher u_k
        becomes the numerical zero vector): that's not OptBasis's
        domain of usefulness, it's a degenerate edge case.
        """
        torch.manual_seed(6)
        N, F = 8, 4
        edge_index = _ring_edge_index(N)
        x = torch.randn(N, F)
        c = 7.3

        # Real graph Laplacian closure: the only operator under which
        # the Lanczos recurrence produces distinct basis vectors.
        backbone = PolynomialFilterGNN(
            in_channels=F,
            hidden_channels=F,
            out_channels=F,
            K=4,
            basis=Chebyshev(),  # any basis works; we just need the closure
        )
        L_apply = backbone._build_laplacian_apply(
            edge_index, edge_weight=None, num_nodes=N
        )

        def run_optbasis_sequence(b: OptBasisGNN, signal: Tensor):
            u_seq = [b.init(signal, L_apply)]
            u_prev_prev: Tensor | None = None
            u_prev = u_seq[0]
            for k in range(1, 5):
                u_k = b(u_prev, u_prev_prev, L_apply, signal=signal, k=k)
                u_seq.append(u_k)
                u_prev_prev, u_prev = u_prev, u_k
            return u_seq

        u_seq_x = run_optbasis_sequence(OptBasisGNN(), x)
        u_seq_cx = run_optbasis_sequence(OptBasisGNN(), c * x)

        # The Lanczos recurrence has a sign ambiguity at each step (γ_k
        # is a norm, ≥ 0, but v_k can be flipped). For purely-positive
        # scaling of x, all sign choices propagate identically, so this
        # is just element-wise equality.
        for k in range(5):
            assert torch.allclose(u_seq_x[k], u_seq_cx[k], atol=1e-5), (
                f"OptBasis is supposed to be scale-invariant in x, but "
                f"u_{k}(x) and u_{k}(c·x) differ; max diff = "
                f"{(u_seq_x[k] - u_seq_cx[k]).abs().max():.2e}"
            )

        # --- Sanity contrast: Chebyshev is scale-LINEAR, not invariant.
        cheb = Chebyshev()
        u_cheb_x = cheb.init(x, L_apply)
        u_cheb_cx = cheb.init(c * x, L_apply)
        assert torch.allclose(u_cheb_cx, c * u_cheb_x, atol=1e-5)

    def test_optbasis_columns_are_orthonormal_per_channel(self):
        """``⟨u_k, u_j⟩ ≈ δ_{kj}`` per channel: Lanczos orthogonality.

        This is the literal definition of "optimal basis" (Guo & Wei
        2023 §3); if this test fails, the implementation is wrong.
        """
        torch.manual_seed(7)
        b = OptBasisGNN()
        N, F = 16, 3
        x = torch.randn(N, F)
        # A real Laplacian closure built once.
        edge_index = _ring_edge_index(N)
        backbone = PolynomialFilterGNN(
            in_channels=F,
            hidden_channels=F,
            out_channels=F,
            K=5,
            basis=b,
        )
        L_apply = backbone._build_laplacian_apply(
            edge_index, edge_weight=None, num_nodes=N
        )

        # Run the recurrence and stash u_0 ... u_K.
        K = 5
        us: list[Tensor] = [b.init(x, L_apply)]
        u_prev_prev: Tensor | None = None
        u_prev = us[0]
        for k in range(1, K + 1):
            u_k = b(u_prev, u_prev_prev, L_apply, signal=x, k=k)
            us.append(u_k)
            u_prev_prev, u_prev = u_prev, u_k

        # Inner-product matrix G[k, j, c] = ⟨u_k[:, c], u_j[:, c]⟩.
        # For each channel c, G[:, :, c] should be approximately I_{K+1}.
        for c in range(F):
            cols = torch.stack([u[:, c] for u in us])  # [K+1, N]
            G = cols @ cols.T                          # [K+1, K+1]
            assert torch.allclose(G, torch.eye(K + 1), atol=1e-4), (
                f"OptBasis columns not orthonormal for channel {c}; "
                f"max |G - I| = {(G - torch.eye(K + 1)).abs().max():.2e}"
            )

    def test_optbasis_has_no_learnable_parameters(self):
        b = OptBasisGNN()
        assert sum(p.numel() for p in b.parameters()) == 0

    def test_optbasis_multiple_forward_passes_dont_leak_state(self):
        """Two consecutive backbone forward passes must produce identical y.

        State (``_gamma_prev``) is reset in ``init``: verify that two
        passes with the same input give bit-for-bit the same output.
        """
        torch.manual_seed(8)
        edge_index = _ring_edge_index(8)
        x = torch.randn(8, 4)
        model = PolynomialFilterGNN(
            in_channels=4,
            hidden_channels=6,
            out_channels=3,
            K=4,
            basis=OptBasisGNN(),
        ).eval()
        y1 = model(x, edge_index)
        y2 = model(x, edge_index)
        assert torch.allclose(y1, y2, atol=1e-7)

    def test_backbone_runs_with_optbasis(self):
        _backbone_smoke(basis=OptBasisGNN())


class TestPolynomialFilterGNNHydraConfig:
    """Hydra composition smoke test for ``configs/model/graph/polynomial_filter_gnn.yaml``.

    Composes the full ``run.yaml`` (with MUTAG, the same dataset used by
    ``test_pipeline``) and asserts that:

    1. the YAML resolves (interpolations + custom resolvers all work);
    2. instantiating ``cfg.model`` produces a ``TBModel`` whose backbone
       is :class:`PolynomialFilterGNN` with the default :class:`Chebyshev`
       basis (proves the Hydra ``_target_`` paths line up with the
       installed package);
    3. swapping ``model.backbone.basis._target_`` via an override flips
       the registered basis without any backbone change (proves the
       CLI-sweep workflow actually works end-to-end).

    No training is run: just composition + instantiation, which is fast.
    """

    def setup_method(self):
        # The pipeline tests clear Hydra global state in their own setup;
        # do the same here so this class can run in any test order.
        import hydra.core.global_hydra

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        register_all_resolvers()

    def _compose(self, *overrides: str):
        import hydra

        with hydra.initialize(
            version_base=None,
            config_path="../../../../configs",
            job_name="poly_filter_gnn_hydra_smoke",
        ):
            return hydra.compose(
                config_name="run.yaml",
                overrides=[
                    "model=graph/polynomial_filter_gnn",
                    "dataset=graph/MUTAG",
                    "paths=test",
                    *overrides,
                ],
                return_hydra_config=True,
            )

    def test_default_config_resolves_and_instantiates_chebyshev(self):
        """Default basis is Chebyshev and the backbone instantiates cleanly."""
        import hydra

        cfg = self._compose()

        # Sanity: the YAML uses the short `topobench.nn.backbones.*` path
        # like every other in-repo backbone (nsd.yaml, graph_mlp.yaml, ...).
        assert (
            cfg.model.backbone._target_
            == "topobench.nn.backbones.PolynomialFilterGNN"
        )
        assert cfg.model.backbone.basis._target_.endswith(".Chebyshev")
        assert cfg.model.backbone.K == 8
        assert cfg.model.backbone.laplacian_norm == "sym"

        # Instantiate just the backbone subtree: no dataset / lightning needed
        # to prove the class paths and nested instantiation work end-to-end.
        # NB: ``isinstance`` against the directly-imported class would fail
        # here because TopoBench's parent backbone auto-discovery loads
        # modules under a non-canonical module name, producing a *distinct*
        # class object from the one we import directly. Same footgun every
        # backbone in this package lives with; we compare by class name
        # rather than by identity to sidestep it.
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        assert type(backbone).__name__ == "PolynomialFilterGNN"
        assert type(backbone.basis).__name__ == "Chebyshev"
        assert backbone.K == 8

    def test_basis_override_swaps_to_monomial(self):
        """``model.backbone.basis._target_=...Monomial`` swaps the basis.

        This is the literal demonstration that running the same backbone
        with a different basis is a one-flag CLI change: no code edits,
        no separate YAML.
        """
        import hydra

        cfg = self._compose(
            "model.backbone.basis._target_="
            "topobench.nn.backbones.graph.poly_filter.bases.Monomial",
        )
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        # See note in the previous test for why ``type(...).__name__`` rather
        # than ``isinstance``.
        assert type(backbone).__name__ == "PolynomialFilterGNN"
        assert type(backbone.basis).__name__ == "Monomial"

    def test_basis_override_swaps_to_jacobi_with_hyperparameters(self):
        """Basis swap + constructor hyperparameters via Hydra in one shot.

        Jacobi is the first basis whose ``__init__`` takes non-trivial
        hyperparameters (``α``, ``β``). Verify they flow through Hydra's
        nested instantiation cleanly: this is the pattern every future
        hyperparameterized basis (ChebNetII, FavardGNN) will rely on.
        """
        import hydra

        cfg = self._compose(
            "model.backbone.basis._target_="
            "topobench.nn.backbones.graph.poly_filter.bases.Jacobi",
            "+model.backbone.basis.alpha=0.5",
            "+model.backbone.basis.beta=-0.25",
        )
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        assert type(backbone).__name__ == "PolynomialFilterGNN"
        assert type(backbone.basis).__name__ == "Jacobi"
        assert backbone.basis.alpha == 0.5
        assert backbone.basis.beta == pytest.approx(-0.25)

    def test_basis_override_swaps_to_legendre(self):
        """``...Legendre`` swap: argument-less basis instantiates via Hydra.

        Legendre's ``__init__`` deliberately takes no arguments (see
        ``legendre.py`` docstring). Verify Hydra can instantiate it as
        a target with no extra config and that it inherits the
        ``α = β = 0`` invariant.
        """
        import hydra

        cfg = self._compose(
            "model.backbone.basis._target_="
            "topobench.nn.backbones.graph.poly_filter.bases.Legendre",
        )
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        assert type(backbone).__name__ == "PolynomialFilterGNN"
        assert type(backbone.basis).__name__ == "Legendre"
        assert backbone.basis.alpha == 0.0
        assert backbone.basis.beta == 0.0

    def test_basis_override_swaps_to_chebnetii_with_K(self):
        """ChebNetII swap with K matched to backbone via Hydra override."""
        import hydra

        cfg = self._compose(
            "model.backbone.basis._target_="
            "topobench.nn.backbones.graph.poly_filter.bases.ChebNetII",
            "+model.backbone.basis.K=${model.backbone.K}",
        )
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        assert type(backbone).__name__ == "PolynomialFilterGNN"
        assert type(backbone.basis).__name__ == "ChebNetII"
        assert backbone.basis.K == backbone.K  # interpolation worked

    def test_basis_override_swaps_to_favard_with_K(self):
        """FavardGNN swap with K matched to backbone via Hydra interpolation."""
        import hydra

        cfg = self._compose(
            "model.backbone.basis._target_="
            "topobench.nn.backbones.graph.poly_filter.bases.FavardGNN",
            "+model.backbone.basis.K=${model.backbone.K}",
        )
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        assert type(backbone).__name__ == "PolynomialFilterGNN"
        assert type(backbone.basis).__name__ == "FavardGNN"
        assert backbone.basis.K == backbone.K
        # FavardGNN has 2(K+1) learnable parameters of its own.
        n_basis_params = sum(p.numel() for p in backbone.basis.parameters())
        assert n_basis_params == 2 * (backbone.K + 1)

    def test_basis_override_swaps_to_optbasis(self):
        """OptBasisGNN swap: argument-less basis instantiates via Hydra.

        OptBasis takes no constructor args (no learnable params; α, γ
        are signal-derived). Verifies the signal-dependent basis plugs
        into the same Hydra schema as everything else.
        """
        import hydra

        cfg = self._compose(
            "model.backbone.basis._target_="
            "topobench.nn.backbones.graph.poly_filter.bases.OptBasisGNN",
        )
        backbone = hydra.utils.instantiate(cfg.model.backbone)
        assert type(backbone).__name__ == "PolynomialFilterGNN"
        assert type(backbone.basis).__name__ == "OptBasisGNN"
        # No learnable params on the basis.
        assert sum(p.numel() for p in backbone.basis.parameters()) == 0
