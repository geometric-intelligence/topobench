"""Legendre polynomial basis.

Implemented as the ``α = β = 0`` special case of the Jacobi family. With
those hyperparameters Liao's Jacobi recurrence (Appendix B, "Jacobi"
subsection) reduces to

.. math::

    T^{(0)}(\\tilde L) = I,
    \\qquad
    T^{(1)}(\\tilde L) = I - \\tilde L,

    T^{(k)}(\\tilde L) = \\frac{2k-1}{k}\\, (I - \\tilde L)\\, T^{(k-1)}(\\tilde L)
                          - \\frac{k-1}{k}\\, T^{(k-2)}(\\tilde L),
    \\quad k \\ge 2,

i.e. the standard three-term recurrence for the Legendre polynomial
``P_k`` applied to the argument ``z = I - \\tilde L``.

**Why this reparameterization rather than Liao's standalone formula.**
Liao Appendix B lists Legendre as a *separate* entry, using ``z = \\tilde L``
as the polynomial argument:

.. math::

    T^{(1)}_{\\text{Liao-Legendre}}(\\tilde L) = \\tilde L, \\quad
    T^{(k)}_{\\text{Liao-Legendre}}(\\tilde L) = \\frac{2k-1}{k}\\, \\tilde L \\,
        T^{(k-1)}(\\tilde L) - \\frac{k-1}{k}\\, T^{(k-2)}(\\tilde L).

That recurrence is the classical Legendre polynomial ``P_k`` applied to
``\\tilde L`` directly. **The orthogonality interval of ``P_k`` is
``[-1, 1]``, but the symmetric normalized graph Laplacian ``\\tilde L``
has eigenvalues in ``[0, 2]``.** Evaluating ``P_k(z)`` at ``z \\in [0, 2]``
means evaluating it *outside* its orthogonality interval, where Legendre
polynomials grow rapidly in ``k`` (``|P_k(2)|`` grows roughly like
``\\Theta(3^k / \\sqrt{k})``). The basis is mathematically defined but
numerically unstable as a graph filter — the high-order channels
amplify by exponentially growing factors and dominate the accumulator.

The ``α = β = 0`` Jacobi reparameterization shifts the argument to
``z = I - \\tilde L``, which has eigenvalues in ``[-1, 1]`` — back inside
Legendre's orthogonality interval. There the classical bound
``|P_k(z)| \\le 1`` holds for all ``k``, so the basis is uniformly
bounded relative to the input ``x``. This is the same domain-shift trick
ChebNet uses for the Chebyshev basis: rescale the operator so the
polynomial argument lies in ``[-1, 1]``.

We ship the well-conditioned form. The deviation is deliberate; if a
reviewer specifically requests Liao's literal Legendre formula it would
land as a separate ``LegendreLiao`` class — the unstable form is not a
useful default for graph spectral filters.

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B — the "Jacobi" subsection,
``α = β = 0`` instance of the recurrence cited there.

Chen & Xu (2023) *Improved Modeling and Generalization Capabilities of
Graph Neural Networks With Legendre Polynomials* (IEEE Access; Liao
ref [14]) — primary reference for LegendreNet, the family this basis
covers in the registry.
"""

from __future__ import annotations

from topobench.nn.backbones.graph.poly_filter.bases.jacobi import Jacobi


class Legendre(Jacobi):
    """Legendre basis (``α = β = 0`` reparameterization of :class:`Jacobi`).

    Stateless, signal-independent, takes no constructor arguments. See
    the module docstring for the rationale behind shipping this form
    rather than Liao's standalone Legendre recurrence (numerical
    stability — eigenvalue interval alignment with the orthogonality
    interval of ``P_k``).
    """

    def __init__(self):
        """Initialize as ``Jacobi(alpha=0.0, beta=0.0)``."""
        super().__init__(alpha=0.0, beta=0.0)
