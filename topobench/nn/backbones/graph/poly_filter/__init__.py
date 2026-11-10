"""Polynomial filter machinery for the graph backbone.

This subpackage is **not** scanned by the parent ``graph/__init__.py``
auto-discovery (which globs ``*.py`` non-recursively). That's
deliberate: the ``Basis`` protocol and concrete basis implementations
are not backbones in their own right; they are internal components of
``PolynomialFilterGNN``.

Layout: ``basis.py`` holds the ``Basis`` protocol (ABC plus the
``LaplacianApply`` type), and ``bases/`` holds the concrete basis
implementations, one file per family. The ``PolynomialFilterGNN``
backbone itself lives one level up at
``topobench/nn/backbones/graph/polynomial_filter_gnn.py`` so the
parent auto-discovery picks it up and re-exports it as
``topobench.nn.backbones.PolynomialFilterGNN`` for Hydra ``_target_``.
"""

from topobench.nn.backbones.graph.poly_filter.basis import (
    Basis,
    LaplacianApply,
)

__all__ = ["Basis", "LaplacianApply"]
