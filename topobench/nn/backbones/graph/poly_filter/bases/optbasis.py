"""OptBasisGNN polynomial basis.

The "Optimal Basis" — three-term recurrence where the coefficients
``α_k, β_k`` are derived from the **running signal** ``h^{(k)}`` (the
current basis vector) via inner products, not learned. This is the
classical Lanczos algorithm applied to the operator ``(I - \\tilde L)``
with starting vector ``x``: the resulting ``{h^{(k)}}`` are
orthonormal under the Euclidean inner product, and the spectral filter
``g(\\tilde L; θ) = Σ_k θ_k h^{(k)}`` achieves the fastest convergence
rate among orthogonal-polynomial filters for the graph-signal-denoising
problem (Guo & Wei 2023, Theorem 4.1).

Liao Appendix B formulation (with ``z = I - \\tilde L``):

.. math::

    T^{(-1)} = O, \\qquad T^{(0)} = \\frac{1}{\\|x\\|}\\, I,

    T^{(k)} = \\frac{1}{\\gamma_{k-1}}
              \\big(\\, z\\, T^{(k-1)} \\;-\\; \\alpha_{k-1}\\, T^{(k-1)}
                       \\;-\\; \\gamma_{k-2}\\, T^{(k-2)} \\,\\big),

with

.. math::

    \\alpha_{k-1} = \\langle z\\, h^{(k-1)}, h^{(k-1)} \\rangle ,
    \\qquad
    \\gamma_{k-1} = \\big\\| z\\, h^{(k-1)} - \\alpha_{k-1} h^{(k-1)}
                          - \\gamma_{k-2} h^{(k-2)} \\big\\| ,

and ``γ_{-1} = 0``.

**Why this is the load-bearing test of the basis protocol.**
Every other basis in the registry is signal-independent — Monomial,
Chebyshev, Jacobi, Legendre, ChebNetII, and FavardGNN all produce
``u_k`` from a recurrence whose coefficients are fixed (or learned,
but independent of ``x``). OptBasis is the only basis whose
recurrence coefficients are functions of the input signal. This is
the literal test of whether the basis protocol survives: the design
passes ``signal`` to every basis on every step (uniform interface),
with stateless bases ignoring it. OptBasis is the single basis in
this registry that *uses* the signal-dependence — through the
``u_prev`` argument, which evolves from ``u_0 = x / \\|x\\|`` and
feeds the inner-product computations.

**State across steps.** The recurrence needs ``γ_{k-1}`` (computed
at step ``k - 1``) to construct ``u_k``. The protocol passes
``u_prev`` and ``u_prev_prev`` but not the previous ``γ``; we
therefore store ``γ`` as an instance attribute, reset in
:meth:`init` and updated in :meth:`forward`.

**Concurrency caveat.** This basis is not thread-safe. The
backbone's single ``init → forward → … → forward`` sequence per
forward pass is safe, which is the only pattern Lightning/PyG
actually exercise. Two interleaved forward passes on the same
instance would corrupt the state.

**No learnable parameters.** ``α``, ``γ`` are derived from the
signal — nothing is learned at the basis level. Compare FavardGNN,
which learns its ``α``, ``β``. The backbone's ``θ`` is still
learnable.

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B, "OptBasis" entry.

Guo & Wei (2023) *Graph Neural Networks with Learnable and Optimal
Polynomial Bases* (ICML, arXiv:2302.12432) — primary reference for
OptBasisGNN. Their Theorem 4.1 proves the optimal convergence rate.
"""

from __future__ import annotations

from torch import Tensor

from topobench.nn.backbones.graph.poly_filter.basis import (
    Basis,
    LaplacianApply,
)

_EPS = 1e-12


class OptBasisGNN(Basis):
    """Lanczos-style orthonormal basis with signal-derived coefficients.

    Stateless w.r.t. learnable parameters; **stateful** w.r.t. the
    intermediate ``γ_{k-1}`` value carried across recurrence steps
    within a single forward pass. See module docstring for details
    and the concurrency caveat.
    """

    def __init__(self):
        super().__init__()
        # γ_{k-1} from the previous recurrence step. Set to None
        # at __init__ time; the proper reset happens in init() which
        # the backbone always calls at the start of every forward pass.
        self._gamma_prev: Tensor | None = None

    def init(self, x: Tensor, L_apply: LaplacianApply) -> Tensor:
        """Reset intra-pass state and return ``u_0 = x / ‖x‖`` per-channel.

        This is the only point at which the original input signal
        enters the recurrence; everything downstream flows through
        ``u_prev``.

        Parameters
        ----------
        x : Tensor, shape ``[N, F]``
            Input features.
        L_apply : LaplacianApply
            Unused for OptBasis's ``init``; only the recurrence
            consumes it.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_0`` with unit per-channel norm.
        """
        norm = x.norm(dim=0, keepdim=True).clamp_min(_EPS)
        # Reset the intra-forward-pass state.
        self._gamma_prev = None
        return x / norm

    def forward(
        self,
        u_prev: Tensor,
        u_prev_prev: Tensor | None,
        L_apply: LaplacianApply,
        signal: Tensor,  # unused — signal-dependence flows via u_prev
        k: int,
    ) -> Tensor:
        """Run one Lanczos step against ``(I - L̃)`` on the running signal.

        Computes ``α`` from an inner product on ``u_prev``, forms the
        unnormalized next vector, updates ``γ`` state, and returns the
        normalized ``u_k``.

        Parameters
        ----------
        u_prev : Tensor, shape ``[N, F]``
            ``u_{k-1}``, the previous orthonormal basis vector.
        u_prev_prev : Tensor or None, shape ``[N, F]``
            ``u_{k-2}``. ``None`` at ``k = 1`` (encodes
            ``γ_{-1} = 0``).
        L_apply : LaplacianApply
            Closure ``h ↦ L̃ @ h``.
        signal : Tensor
            Unused for OptBasis (signal-dependence already flows via
            ``u_prev``, seeded from ``signal`` in :meth:`init`).
        k : int
            Step index (unused; kept for protocol uniformity).

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_k``, with unit per-channel norm.
        """
        # z u_prev = (I - L̃) u_prev
        z_u = u_prev - L_apply(u_prev)

        # α_{k-1} = ⟨z u_prev, u_prev⟩, per-channel scalar.
        alpha = (z_u * u_prev).sum(dim=0, keepdim=True)

        # Unnormalized v_k = z u_prev - α u_prev - γ_{k-2} u_prev_prev.
        v = z_u - alpha * u_prev
        if u_prev_prev is not None:
            # At k = 1 this branch is skipped (u_prev_prev is None), which
            # encodes the γ_{-1} = 0 boundary condition naturally.
            assert self._gamma_prev is not None, (
                "OptBasisGNN: u_prev_prev is not None but γ_prev state is "
                "missing. The backbone must call init() at the start of "
                "every forward pass — see module docstring."
            )
            v = v - self._gamma_prev * u_prev_prev

        # γ_{k-1} = ‖v_k‖, per-channel. Stored for the next step.
        gamma = v.norm(dim=0, keepdim=True).clamp_min(_EPS)
        self._gamma_prev = gamma

        return v / gamma
