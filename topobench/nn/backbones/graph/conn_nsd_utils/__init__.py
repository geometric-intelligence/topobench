"""Connection-Laplacian sheaf utilities.

This sub-package implements the *deterministic* construction of orthogonal
restriction maps for a Sheaf Neural Network, following Algorithm 1 of:

    Barbero, Bodnar, Sáez de Ocáriz Borde, Bronstein, Veličković, Liò.
    "Sheaf Neural Networks with Connection Laplacians."
    ICML 2022 TAG-ML Workshop. arXiv:2206.08702.

The construction is purely a function of the node features and the edge set:
no gradients flow through it. It mirrors Singer & Wu's vector diffusion maps
(2012) adapted to a graph rather than a point cloud.
"""

from topobench.nn.backbones.graph.conn_nsd_utils.connection import (
    build_connection,
    local_tangent_basis,
    optimal_alignment,
)

__all__ = [
    "build_connection",
    "local_tangent_basis",
    "optimal_alignment",
]
