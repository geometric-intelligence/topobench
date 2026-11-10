r"""Legendre polynomial basis.

Implemented as the ``α = β = 0`` special case of the Jacobi family. With
those hyperparameters Liao's Jacobi recurrence (Appendix B, "Jacobi"
subsection) reduces to

.. math::

    T^{(0)}(\tilde L) = I,
    \qquad
    T^{(1)}(\tilde L) = I - \tilde L,

    T^{(k)}(\tilde L) = \frac{2k-1}{k}\, (I - \tilde L)\, T^{(k-1)}(\tilde L)
                          - \frac{k-1}{k}\, T^{(k-2)}(\tilde L),
    \quad k \ge 2,

i.e. the standard three-term recurrence for the Legendre polynomial
``P_k`` applied to the argument ``z = I - L̃``.

**Argument domain: matching Liao, avoiding the unstable naive form.**
The Legendre polynomials ``P_k`` are bounded (``|P_k(z)| <= 1``) only on
their orthogonality interval ``z in [-1, 1]``. A naive reading of the
classical recurrence with ``z = L̃`` would evaluate ``P_k`` on the
Laplacian spectrum ``[0, 2]`` -- outside ``[-1, 1]``, where Legendre
polynomials grow rapidly in ``k`` (``|P_k(2)|`` grows roughly like
``Theta(3^k / sqrt(k))``), so the high-order channels would amplify by
exponentially growing factors and dominate the accumulator.

Neither we nor Liao use that naive form. The ``α = β = 0`` Jacobi
instance evaluates the recurrence at ``z = I - L̃ = Â`` (the normalized
adjacency), whose eigenvalues lie in ``[-1, 1]`` -- inside the
orthogonality interval, where the classical bound ``|P_k(z)| <= 1``
holds for all ``k``. This **agrees with** Liao's ``LegendreConv``, which
propagates ``Â`` directly (the ``'A'`` scheme): same polynomial, same
well-conditioned argument. It is the same domain alignment ChebNet uses
for the Chebyshev basis.

Architecturally this makes ``legendre.py`` a thin subclass of
:class:`Jacobi`: the recurrence and the ``z = Â`` argument come for free
from ``Jacobi(α=0, β=0)``, whose recurrence already enters ``L̃`` only
through ``I - L̃ = Â``.

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Appendix B: the "Jacobi" subsection,
``α = β = 0`` instance of the recurrence cited there.

Chen & Xu (2023) *Improved Modeling and Generalization Capabilities of
Graph Neural Networks With Legendre Polynomials* (IEEE Access; Liao
ref [14]): primary reference for LegendreNet, the family this basis
covers in the registry.
"""

from __future__ import annotations

from topobench.nn.backbones.graph.poly_filter.bases.jacobi import Jacobi


class Legendre(Jacobi):
    """Legendre basis (``α = β = 0`` reparameterization of :class:`Jacobi`).

    Stateless, signal-independent, takes no constructor arguments. See
    the module docstring for why the recurrence is evaluated at
    ``z = I - L̃ = Â`` (eigenvalue interval aligned with the
    orthogonality interval of ``P_k``), which agrees with Liao's
    ``LegendreConv``.
    """

    def __init__(self):
        """Initialize as ``Jacobi(alpha=0.0, beta=0.0)``."""
        super().__init__(alpha=0.0, beta=0.0)
