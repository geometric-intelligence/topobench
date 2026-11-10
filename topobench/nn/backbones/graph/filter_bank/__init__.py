"""Filter-bank machinery for the graph backbone.

A filter bank is Q parallel polynomial filters fused into one signal
(Liao et al. 2024, Eq. (3)):

    g(L̃; γ, θ) = fuse_q( γ_q · g_q(L̃) ),   g_q(L̃) = Σ_k θ_{q,k} T_q^(k)(L̃).

Each channel ``g_q`` is a single polynomial filter and **reuses the
basis registry** (``poly_filter.bases``); the new pieces here are the
per-channel wrapper and the fusion step. Like ``poly_filter``, this
subpackage is not scanned by the parent ``graph/__init__.py``
auto-discovery: it holds internal components, not backbones. The
``FilterBankGNN`` backbone and its variants live one level up at
``topobench/nn/backbones/graph/filter_bank_gnn.py`` so the parent
auto-discovery re-exports them for Hydra ``_target_``.

Layout: ``fusion.py`` holds the ``Fusion`` protocol and ``SumFusion``;
``channel.py`` holds the ``Channel`` wrapper and the shared
``apply_polynomial_filter`` / ``build_laplacian_apply`` helpers.

References
----------
Liao et al. (2024) *A Comprehensive Benchmark on Spectral GNNs*
(SIGMOD '26, arXiv:2406.09675), Section 3.3 and Table 1 (Filter Bank).
"""

from topobench.nn.backbones.graph.filter_bank.channel import (
    Channel,
    GaussianChannel,
    PPRChannel,
    apply_polynomial_filter,
    build_laplacian_apply,
)
from topobench.nn.backbones.graph.filter_bank.fusion import Fusion, SumFusion

__all__ = [
    "Channel",
    "Fusion",
    "GaussianChannel",
    "PPRChannel",
    "SumFusion",
    "apply_polynomial_filter",
    "build_laplacian_apply",
]
