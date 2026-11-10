r"""Chebyshev polynomial basis (first kind).

The Chebyshev polynomials ``T_k`` are bounded (``|T_k(z)| <= 1``) only on
their orthogonality interval ``z in [-1, 1]``, while the symmetric
normalized Laplacian ``L̃`` has eigenvalues in ``[0, 2]``. We therefore
evaluate the recurrence at the **rescaled** argument ``z = L̃ - I``
(eigenvalues in ``[-1, 1]``), i.e. ``-Â`` for the normalized adjacency
``Â = I - L̃``. This is the standard ChebNet domain shift
``2L̃/λ_max - I`` with ``λ_max = 2``, and matches Liao's benchmark, whose
``ChebConv`` propagates ``-Â`` (the ``'L-I'`` scheme). Without it the
high-order terms blow up at the high-frequency end of the spectrum
(``T_8(2) ≈ 1.9e4``).

Three-term recurrence in ``z = L̃ - I``:

.. math::

    T^{(0)} = I, \qquad T^{(1)} = \tilde L - I,
    \qquad T^{(k)} = 2 (\tilde L - I) \, T^{(k-1)} - T^{(k-2)}
    \quad (k \ge 2).

Equivalently, with ``u_k := T^{(k)} x``,

.. math::

    u_0 = x, \quad u_1 = (\tilde L - I) \, u_0, \quad
    u_k = 2 (\tilde L - I) \, u_{k-1} - u_{k-2} \quad (k \ge 2).

The ``k = 1`` boundary, where ``u_1 = (L̃ - I) u_0`` rather than the
generic ``2 (L̃ - I) u_{k-1} - u_{k-2}``, is handled **inside this basis**
via the ``u_prev_prev is None`` check. The backbone never has to know
that Chebyshev's first step differs from its general recurrence; the
boundary-vs-bulk dispatch stays a basis-local concern.

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B, "Chebyshev" subsection
(Variable Basis block). Liao's ``ChebConv`` runs the recurrence on the
``'L-I'`` propagation matrix ``= -Â = L̃ - I``, the rescaled argument
used here:

.. math::

    g(\tilde L; \theta) = \sum_{k=0}^{K} \theta_k T^{(k)},
    \quad T^{(k)} = 2 (\tilde L - I) T^{(k-1)} - T^{(k-2)},
    \quad T^{(1)} = \tilde L - I,\ T^{(0)} = I.

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

        Runs on the rescaled argument ``(L̃ - I)`` (eigenvalues in
        ``[-1, 1]``), computed as ``L_apply(u) - u``. The ``k = 1``
        boundary returns ``(L̃ - I) u_0`` (rather than the generic
        ``2 (L̃ - I) u_{k-1} - u_{k-2}``) and is encoded by the
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
            ``u_k``: either ``(L̃ - I) u_0`` at the boundary or
            ``2 (L̃ - I) u_{k-1} - u_{k-2}`` for ``k >= 2``.
        """
        # Rescaled argument z = L̃ - I = -Â, eigenvalues in [-1, 1].
        z_u = L_apply(u_prev) - u_prev
        if u_prev_prev is None:
            # k == 1 boundary: u_1 = (L̃ - I) u_0
            return z_u
        # k >= 2: u_k = 2 (L̃ - I) u_{k-1} - u_{k-2}
        return 2.0 * z_u - u_prev_prev
