"""Concrete polynomial bases for :class:`PolynomialFilterGNN`.

Each basis lives in its own file: a reader who knows the original
paper should recognize the recurrence immediately, with docstrings
citing the primary reference and Liao et al. (2024) Appendix B as the
unified formulation. Each basis implements the
:class:`~topobench.nn.backbones.graph.poly_filter.basis.Basis` protocol;
no backbone changes are required to add or swap one.

Registry:

- :class:`~.monomial.Monomial`
- :class:`~.chebyshev.Chebyshev`
- :class:`~.jacobi.Jacobi`
- :class:`~.legendre.Legendre` (``α = β = 0`` reparameterization of
  Jacobi; see ``legendre.py`` docstring for why we ship this form
  rather than Liao's standalone Legendre recurrence)
- :class:`~.chebnetii.ChebNetII` (Chebyshev recurrence with
  He-Wei-Wen 2022 interpolation reparameterization of the
  coefficients; the only basis here that uses the
  ``effective_thetas`` protocol hook)
- :class:`~.favard.FavardGNN` (three-term recurrence with learnable α, β)
- :class:`~.optbasis.OptBasisGNN` (Lanczos-style recurrence with
  signal-derived coefficients; the basis whose existence stress-tests
  the uniform protocol against the signal-dependence dimension)

Bernstein is deliberately omitted: Liao Appendix B presents it in
closed form per ``k`` (the only variable basis with ``O(K^2 m F)``
complexity instead of ``O(K m F)``), so it does not fit the three-term
recurrence protocol without stretching the abstraction. Deferred.
"""

from topobench.nn.backbones.graph.poly_filter.bases.chebnetii import ChebNetII
from topobench.nn.backbones.graph.poly_filter.bases.chebyshev import Chebyshev
from topobench.nn.backbones.graph.poly_filter.bases.favard import FavardGNN
from topobench.nn.backbones.graph.poly_filter.bases.jacobi import Jacobi
from topobench.nn.backbones.graph.poly_filter.bases.legendre import Legendre
from topobench.nn.backbones.graph.poly_filter.bases.monomial import Monomial
from topobench.nn.backbones.graph.poly_filter.bases.optbasis import OptBasisGNN

__all__ = [
    "ChebNetII",
    "Chebyshev",
    "FavardGNN",
    "Jacobi",
    "Legendre",
    "Monomial",
    "OptBasisGNN",
]
