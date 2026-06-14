r"""Chebyshev polynomial basis (first kind).

Three-term recurrence

.. math::

    T^{(0)}(\tilde L) = I, \qquad T^{(1)}(\tilde L) = \tilde L,
    \qquad T^{(k)}(\tilde L) = 2 \tilde L \, T^{(k-1)}(\tilde L)
                                  - T^{(k-2)}(\tilde L) \quad (k \ge 2).

Equivalently, with ``u_k := T^{(k)}(L̃) x``,

.. math::

    u_0 = x, \quad u_1 = \tilde L \, u_0, \quad
    u_k = 2 \tilde L \, u_{k-1} - u_{k-2} \quad (k \ge 2).

The ``k = 1`` boundary, where ``u_1 = L̃ u_0`` rather than the generic
``2 L̃ u_{k-1} - u_{k-2}``, is handled **inside this basis** via the
``u_prev_prev is None`` check. The backbone never has to know that
Chebyshev's first step differs from its general recurrence; the
boundary-vs-bulk dispatch stays a basis-local concern.

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B, "Chebyshev" subsection
(Variable Basis block):

.. math::

    g(\tilde L; \theta) = \sum_{k=0}^{K} \theta_k T^{(k)}(\tilde L),
    \quad T^{(k)}(\tilde L) = 2 \tilde L T^{(k-1)}(\tilde L)
                                  - T^{(k-2)}(\tilde L),
    \quad T^{(1)}(\tilde L) = \tilde L,\ T^{(0)}(\tilde L) = I.

Time complexity ``O(K m F)``.

Defferrard, Bresson & Vandergheynst (2016) *Convolutional Neural
Networks on Graphs with Fast Localized Spectral Filtering* (NeurIPS,
arXiv:1606.09375): primary reference for ChebNet, which uses this
basis in its iterative form.

He, Wei & Wen (2022) *Convolutional Neural Networks on Graphs with
Chebyshev Approximation, Revisited* (NeurIPS, arXiv:2202.03580):
introduces ChebBase, the decoupled-propagation variant with learnable
per-order coefficients ``θ_k`` (same recurrence, different
parameterization of the accumulator). The same paper also introduces
ChebNetII, registered separately as :class:`.chebnetii.ChebNetII`.

Not to be confused with **Chebyshev of the second kind**, which has
``T^{(1)} = 2 L̃`` (same recurrence, different boundary) and is the basis
underlying ClenshawGCN: out of scope for this initial registry; a
deferred follow-up.
"""

from __future__ import annotations

from torch import Tensor

from topobench.nn.backbones.graph.poly_filter.basis import (
    Basis,
    LaplacianApply,
)


class Chebyshev(Basis):
    """Chebyshev (first kind) basis ``T_k(L̃)``.

    Stateless: no learnable parameters of its own; the backbone owns
    ``θ_k``. ``signal`` is ignored (signal-independent basis).
    """

    def forward(
        self,
        u_prev: Tensor,
        u_prev_prev: Tensor | None,
        L_apply: LaplacianApply,
        signal: Tensor,  # unused: signal-independent
        k: int,  # unused: boundary handled via `u_prev_prev is None`
    ) -> Tensor:
        """Apply the Chebyshev (first kind) three-term recurrence.

        The ``k = 1`` boundary returns ``L̃ u_0`` (rather than the
        generic ``2 L̃ u_{k-1} - u_{k-2}``) and is encoded by the
        ``u_prev_prev is None`` check.

        Parameters
        ----------
        u_prev : Tensor, shape ``[N, F]``
            ``u_{k-1}``, the previous basis vector.
        u_prev_prev : Tensor or None, shape ``[N, F]``
            ``u_{k-2}``. ``None`` at ``k = 1``.
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h``.
        signal : Tensor
            Unused for Chebyshev (signal-independent basis).
        k : int
            Unused for Chebyshev (boundary is encoded via
            ``u_prev_prev is None``).

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_k``: either ``L̃ u_0`` at the boundary or
            ``2 L̃ u_{k-1} - u_{k-2}`` for ``k >= 2``.
        """
        Lu = L_apply(u_prev)
        if u_prev_prev is None:
            # k == 1 boundary: u_1 = L̃ u_0
            return Lu
        # k >= 2: u_k = 2 L̃ u_{k-1} - u_{k-2}
        return 2.0 * Lu - u_prev_prev
