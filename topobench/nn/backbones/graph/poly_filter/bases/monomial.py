r"""Monomial polynomial basis.

Implements the trivial recurrence

.. math::

    u_k = \tilde L \, u_{k-1}, \qquad u_0 = x,

so that ``u_k = L̃^k x`` and the backbone's accumulation is

.. math::

    y = \sum_{k=0}^{K} \theta_k \, \tilde L^k \, x .

This is the simplest basis in the registry: primarily a baseline and a
**stress test for the basis protocol itself**. If
:class:`PolynomialFilterGNN` works with :class:`Monomial`, it works with
every other basis whose recurrence is a strict refinement of this one
(Chebyshev, Jacobi, Legendre, Favard).

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B, "Monomial" entry of the
Variable Basis table:

.. math::

    g(\tilde L; \theta) = \sum_{k=0}^{K} \theta_k \, \tilde L^k ,
    \qquad T^{(k)}(\tilde L) = \tilde L^k .

Time complexity ``O(K m F)``.

Chien, Peng, Li & Milenkovic (2021) *Adaptive Universal Generalized
PageRank Graph Neural Network* (ICLR, arXiv:2006.07988): primary
reference for GPR-GNN, the family this basis covers (the spectral form
of GPR-GNN with variable ``θ_k`` is exactly this monomial expansion in
``L̃``).
"""

from __future__ import annotations

from torch import Tensor

from topobench.nn.backbones.graph.poly_filter.basis import (
    Basis,
    LaplacianApply,
)


class Monomial(Basis):
    """Monomial basis ``T_k(L̃) = L̃^k``.

    Stateless: no learnable parameters of its own; the backbone owns
    ``θ_k``. ``signal`` and ``u_prev_prev`` are ignored.
    """

    def forward(
        self,
        u_prev: Tensor,
        u_prev_prev: Tensor | None,  # unused: recurrence is single-step
        L_apply: LaplacianApply,
        signal: Tensor,  # unused: basis is signal-independent
        k: int,  # unused: recurrence is k-uniform
    ) -> Tensor:
        """Apply the single-step recurrence ``u_k = L̃ * u_{k-1}``.

        Parameters
        ----------
        u_prev : Tensor, shape ``[N, F]``
            ``u_{k-1}``, the previous basis vector.
        u_prev_prev : Tensor or None
            Unused for Monomial (recurrence is single-step).
        L_apply : LaplacianApply
            Closure ``h -> L̃ @ h``.
        signal : Tensor
            Unused for Monomial (signal-independent basis).
        k : int
            Unused for Monomial (recurrence is k-uniform).

        Returns
        -------
        Tensor, shape ``[N, F]``
            ``u_k = L̃ * u_{k-1}``.
        """
        return L_apply(u_prev)
