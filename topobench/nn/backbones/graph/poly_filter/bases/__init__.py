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
  Jacobi, evaluated at ``Â ∈ [-1, 1]``; see ``legendre.py`` docstring)
- :class:`~.chebnetii.ChebNetII` (Chebyshev recurrence with
  He-Wei-Wen 2022 interpolation reparameterization of the
  coefficients; the only basis here that uses the
  ``effective_thetas`` protocol hook)
- :class:`~.favard.FavardGNN` (three-term recurrence with learnable α, β)
- :class:`~.optbasis.OptBasisGNN` (Lanczos-style recurrence with
  signal-derived coefficients; the only signal-dependent basis in the
  registry)
- :class:`~.bernstein.Bernstein` (closed-form Bezier basis; the one
  non-orthogonal, ``O(K^2 m F)`` member -- see ``bernstein.py`` docstring)

The first seven are three-term recurrences (``O(K m F)``); Bernstein is
the lone closed-form, ``O(K^2 m F)`` member. It fits the same protocol by
ignoring the recurrence arguments (the way signal-independent bases ignore
``signal``), so the registry covers *every* variable basis in Liao
Appendix B.
"""

from topobench.nn.backbones.graph.poly_filter.bases.bernstein import Bernstein
from topobench.nn.backbones.graph.poly_filter.bases.chebnetii import ChebNetII
from topobench.nn.backbones.graph.poly_filter.bases.chebyshev import Chebyshev
from topobench.nn.backbones.graph.poly_filter.bases.favard import FavardGNN
from topobench.nn.backbones.graph.poly_filter.bases.jacobi import Jacobi
from topobench.nn.backbones.graph.poly_filter.bases.legendre import Legendre
from topobench.nn.backbones.graph.poly_filter.bases.monomial import Monomial
from topobench.nn.backbones.graph.poly_filter.bases.optbasis import OptBasisGNN

__all__ = [
    "Bernstein",
    "ChebNetII",
    "Chebyshev",
    "FavardGNN",
    "Jacobi",
    "Legendre",
    "Monomial",
    "OptBasisGNN",
]
