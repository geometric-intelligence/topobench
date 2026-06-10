r"""Jacobi polynomial basis.

Three-term recurrence with two real hyperparameters ``α, β > -1``.
Chebyshev (with the appropriate ``α, β``) and Legendre (``α = β = 0``,
modulo a normalization shift) are special cases of this family.

Recurrence (Liao Appendix B; equivalent to Wang & Zhang 2022 Eq. (5)):

.. math::

    T^{(0)}(\tilde L) = I,
    \qquad
    T^{(1)}(\tilde L) = \tfrac{\alpha - \beta}{2}\, I
                          + \tfrac{\alpha + \beta + 2}{2}\,(I - \tilde L),

    T^{(k)}(\tilde L) = \delta_k (I - \tilde L)\, T^{(k-1)}(\tilde L)
                          + \delta'_k\, T^{(k-1)}(\tilde L)
                          - \delta''_k\, T^{(k-2)}(\tilde L), \quad k \ge 2,

with

.. math::

    \delta_k       = \frac{(2k+\alpha+\beta)(2k+\alpha+\beta-1)}
                            {2k(k+\alpha+\beta)} ,

    \delta'_k      = \frac{(2k+\alpha+\beta-1)(\alpha^2-\beta^2)}
                            {2k(k+\alpha+\beta)(2k+\alpha+\beta-2)} ,

    \delta''_k     = \frac{(k+\alpha-1)(k+\beta-1)(2k+\alpha+\beta)}
                            {k(k+\alpha+\beta)(2k+\alpha+\beta-2)} .

The constraint ``α > -1, β > -1`` (giving ``α + β > -2``) is enforced in
``__init__``: it is the standard integrability condition for the
Jacobi weight ``(1-z)^α (1+z)^β`` and also makes every denominator in
``δ_k, δ'_k, δ''_k`` strictly positive for ``k >= 2``.

The ``k = 1`` boundary is handled inside this basis via the
``u_prev_prev is None`` check; the ``k`` step index is used to evaluate
the per-``k`` recurrence coefficients (the first basis in the registry
where the ``k`` argument to :meth:`forward` is genuinely consumed,
which is why the protocol passes it explicitly rather than asking each
basis to maintain its own step counter as mutable state).

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B, "Jacobi" subsection
(Variable Basis block), Eqs. for ``T^{(0)}, T^{(1)}, T^{(k)}``.

Wang & Zhang (2022) *How Powerful are Spectral Graph Neural Networks*
(ICML, arXiv:2205.11172): primary reference for JacobiConv. The
spectral filter ``g(L̃; θ) = Σ_k θ_k T^{(k)}(L̃)`` with the above
recurrence is their Eq. (5). They prove (Theorem 3.1) that any
polynomial filter can be expressed in this form and that the choice
of ``α, β`` controls the basis's adaptivity to different graph signal
distributions.
"""

from __future__ import annotations

from torch import Tensor

from topobench.nn.backbones.graph.poly_filter.basis import (
    Basis,
    LaplacianApply,
)


class Jacobi(Basis):
    r"""Jacobi basis ``T_k^{(α, β)}(L̃)``.

    Parameters
    ----------
    alpha : float, optional
        Jacobi hyperparameter ``α``. Must satisfy ``α > -1``.
        Defaults to ``1.0`` (the JacobiConv paper's recommended setting).
    beta : float, optional
        Jacobi hyperparameter ``β``. Must satisfy ``β > -1``.
        Defaults to ``1.0``.

    Notes
    -----
    Stateless w.r.t. the input signal: ``signal`` is ignored. ``α`` and
    ``β`` are stored as plain Python floats, not ``nn.Parameter``
    instances: in JacobiConv they are *hyperparameters*, not learned.
    The basis with learnable recurrence coefficients is FavardGNN
    (separate file).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        if alpha <= -1.0:
            raise ValueError(f"Jacobi requires alpha > -1; got alpha={alpha}")
        if beta <= -1.0:
            raise ValueError(f"Jacobi requires beta > -1; got beta={beta}")
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(
        self,
        u_prev: Tensor,
        u_prev_prev: Tensor | None,
        L_apply: LaplacianApply,
        signal: Tensor,  # unused: basis is signal-independent
        k: int,
    ) -> Tensor:
        r"""Apply the Jacobi three-term recurrence with k-dependent coefficients.

        Parameters
        ----------
        u_prev : Tensor, shape ``[N, F]``
            ``u_{k-1}``, the previous basis vector.
        u_prev_prev : Tensor or None, shape ``[N, F]``
            ``u_{k-2}``. ``None`` at ``k = 1``.
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h``.
        signal : Tensor
            Unused for Jacobi (signal-independent basis).
        k : int
            Step index, used to evaluate the per-``k`` coefficients
            ``δ_k, δ'_k, δ''_k``.

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_k``.
        """
        a, b = self.alpha, self.beta

        if u_prev_prev is None:
            # k == 1:
            #   T_1 = (α-β)/2 · I + (α+β+2)/2 · (I - L̃)
            #   u_1 = (α-β)/2 · u_0 + (α+β+2)/2 · (u_0 - L̃ u_0)
            return ((a - b) / 2.0) * u_prev + ((a + b + 2.0) / 2.0) * (
                u_prev - L_apply(u_prev)
            )

        # k >= 2: three-term recurrence with k-dependent coefficients.
        # All three denominators are strictly positive given α, β > -1
        # (so α + β > -2), as enforced in __init__.
        s = 2.0 * k + a + b  # 2k + α + β
        denom_left = 2.0 * k * (k + a + b)  # 2k(k+α+β)
        denom_right = k * (k + a + b) * (s - 2.0)  # k(k+α+β)(2k+α+β-2)

        delta = s * (s - 1.0) / denom_left
        delta_prime = (s - 1.0) * (a * a - b * b) / (2.0 * denom_right)
        delta_double = (k + a - 1.0) * (k + b - 1.0) * s / denom_right

        # u_k = δ_k (u_{k-1} - L̃ u_{k-1}) + δ'_k u_{k-1} - δ''_k u_{k-2}
        return (
            delta * (u_prev - L_apply(u_prev))
            + delta_prime * u_prev
            - delta_double * u_prev_prev
        )
