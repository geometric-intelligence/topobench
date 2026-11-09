"""Sheaf Laplacian builder for *fixed*, externally supplied orthogonal maps.

Mirrors :class:`topobench.nn.backbones.graph.nsd_utils.laplacian_builders.NormConnectionLaplacianBuilder`
in formula but bypasses the lower-triangular-parameter → Cayley → orthogonal
pipeline. Conn-NSD computes its restriction maps deterministically (via
Algorithm 1 of Barbero et al., 2022) and they arrive at this builder
already in ``O(d)``; trying to re-route them through the Bodnar
parametrisation would be both wasteful and misleading about what the model
is doing.

The math is the *normalised* sheaf Laplacian of an ``O(d)``-bundle:

    Δ_F  =  D^{-1/2}  ( δ^⊤ δ )  D^{-1/2}

where ``δ`` is the coboundary acting on 0-cochains, the diagonal block
``D_{vv}`` equals ``deg(v) · I_d`` (every restriction map is orthogonal, so
``F^⊤ F = I`` and each edge contributes ``I`` to the diagonal accumulator),
and the off-diagonal block is the negative parallel transport supplied by
Algorithm 1 for that oriented edge.
"""

from __future__ import annotations

from torch import Tensor

from topobench.nn.backbones.graph.nsd_utils.laplace import (
    compute_learnable_diag_laplacian_indices,
    compute_learnable_laplacian_indices,
    mergesp,
)
from topobench.nn.backbones.graph.nsd_utils.laplacian_builders import (
    LaplacianBuilder,
)


class FixedConnectionLaplacianBuilder(LaplacianBuilder):
    """Normalised ``O(d)``-bundle sheaf Laplacian from pre-computed maps.

    Parameters
    ----------
    size : int
        Number of nodes ``N`` in the graph.
    edge_index : torch.Tensor, shape ``[2, E]``
        Directed edge index in PyG convention (bidirectional pairs expected).
    d : int
        Stalk dimension. Must equal the last two dims of the maps we receive
        in :meth:`forward`.

    Notes
    -----
    Unlike a pair of independently learned endpoint restriction maps, the
    supplied matrices are already the edge transports produced by Algorithm
    1 of Barbero et al. They must therefore be inserted directly rather than
    composed with their reverse orientation.
    """

    def __init__(self, size: int, edge_index: Tensor, d: int):
        super().__init__(size, edge_index, d, normalised=True)

        # Same sparsity-index plumbing as the learnable bundle builder.
        _, self.tril_indices = compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.d
        )
        self.diag_indices, _ = compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.d
        )

    def forward(self, restriction_maps: Tensor):
        """Assemble the normalised bundle Laplacian.

        Parameters
        ----------
        restriction_maps : torch.Tensor, shape ``[E, d, d]``
            Pre-computed orthogonal restriction maps, one per directed edge.
            Must satisfy ``F^⊤ F = I`` per row; we do not re-verify at every
            forward call (cheap, but trips the data-flow purity principle).

        Returns
        -------
        sparse_laplacian : tuple[torch.Tensor, torch.Tensor]
            ``(indices, values)`` representation of the ``Nd × Nd`` block
            sparse Laplacian, ready for ``torch_sparse.spmm``.
        saved_tril_maps : torch.Tensor
            Lower-triangular negative transport maps (for
            diagnostics; matches the contract of the learnable builder).
        """
        assert restriction_maps.dim() == 3, (
            f"restriction_maps must be [E, d, d], got "
            f"{tuple(restriction_maps.shape)}"
        )
        assert restriction_maps.size(-1) == self.d, (
            f"restriction_maps last dim must equal stalk dim d={self.d}, "
            f"got {restriction_maps.size(-1)}"
        )
        assert restriction_maps.size(-2) == self.d, (
            f"restriction_maps second-to-last dim must equal stalk dim "
            f"d={self.d}, got {restriction_maps.size(-2)}"
        )

        left_idx, _ = self.left_right_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices

        # Diagonal:  D_{vv} = deg(v) · I_d  for orthogonal F. No SVD or
        # accumulation needed — see the discussion above.
        diag_maps = self.deg.to(dtype=restriction_maps.dtype).unsqueeze(
            -1
        )  # [N, 1]

        # Algorithm 1 already returns the oriented parallel transport. Using
        # both directions here would compose O_ij with O_ji and erase/square
        # the connection information.
        tril_maps = -restriction_maps.index_select(0, left_idx)
        saved_tril_maps = tril_maps.detach().clone()

        # Paper Definition 2.5: Δ_F = D^{-1/2} L_F D^{-1/2}, where D
        # is exactly the block diagonal of L_F. The generic NSD helper uses
        # ``diag + 1`` (an unreported self-loop renormalisation), so Conn-NSD
        # performs the paper's normalization explicitly here.
        row, col = self.vertex_tril_idx
        diag_sqrt_inv = diag_maps.rsqrt()
        diag_sqrt_inv = diag_sqrt_inv.masked_fill(
            ~diag_sqrt_inv.isfinite(), 0.0
        )
        tril_maps = (
            diag_sqrt_inv[row].unsqueeze(-1)
            * tril_maps
            * diag_sqrt_inv[col].unsqueeze(-1)
        )
        diag_maps = diag_sqrt_inv.square() * diag_maps
        tril_flat = tril_maps.reshape(-1)
        diag_flat = diag_maps.expand(-1, self.d).reshape(-1)

        # Mirror the lower triangle to the upper triangle and merge with the
        # diagonal — identical assembly path to the learnable builder.
        triu_indices = tril_indices.flip(0)
        non_diag_indices, non_diag_values = mergesp(
            tril_indices, tril_flat, triu_indices, tril_flat
        )
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_flat
        )
        return (edge_index, weights), saved_tril_maps
