"""Orthogonal parameterizations for directed sheaf restriction maps.

Adapted from the Neural Sheaf Diffusion reference implementation of Bodnar
et al. (https://github.com/twitter-research/neural-sheaf-diffusion), which in
turn follows PyTorch's ``torch.nn.utils.parametrizations.orthogonal``.

The one deviation from both is the parameter count. The skew-symmetrization
``A = P - P.T`` annihilates the diagonal of ``P``, so a parameterization that
feeds ``d(d + 1) / 2`` lower-triangular entries (including the diagonal) leaves
``d`` of them with identically zero gradient. We take the strictly
lower-triangular ``d(d - 1) / 2`` entries instead: the reachable set of
``A`` -- and hence of ``Q`` -- is unchanged, and every predicted parameter has
a non-zero gradient.
"""

# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

ORTHOGONAL_MAPS = ("cayley", "matrix_exp")


def num_orthogonal_params(d: int) -> int:
    """Return the number of parameters a ``d x d`` orthogonal map consumes.

    Parameters
    ----------
    d : int
        Dimension of the square orthogonal matrices.

    Returns
    -------
    int
        The count ``d * (d - 1) // 2`` of strictly lower-triangular entries.
    """
    return d * (d - 1) // 2


class Orthogonal(nn.Module):
    r"""Map free parameters to special-orthogonal ``d x d`` matrices.

    Builds a skew-symmetric ``A`` from strictly lower-triangular parameters and
    retracts it onto :math:`SO(d)` with either the Cayley transform
    :math:`(I + A/2)(I - A/2)^{-1}` or the matrix exponential
    :math:`\exp(A)`.

    Both retractions land in :math:`SO(d)`, so the reflection component of
    :math:`O(d)` is unreachable. This matches the Neural Sheaf Diffusion port
    already in TopoBench; the reference implementation's Householder option
    would reach all of :math:`O(d)` but needs an external dependency.

    Parameters
    ----------
    d : int
        Dimension of the square orthogonal matrices to generate.
    orthogonal_map : str
        Retraction to use, either ``"cayley"`` or ``"matrix_exp"``.

    Raises
    ------
    ValueError
        If ``orthogonal_map`` is not one of ``ORTHOGONAL_MAPS``.
    """

    def __init__(self, d: int, orthogonal_map: str) -> None:
        super().__init__()
        if orthogonal_map not in ORTHOGONAL_MAPS:
            raise ValueError(
                f"Unsupported orthogonal_map {orthogonal_map!r}; "
                f"expected one of {ORTHOGONAL_MAPS}"
            )
        self.d = d
        self.orthogonal_map = orthogonal_map

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(d={self.d}, "
            f"orthogonal_map={self.orthogonal_map!r})"
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Convert free parameters into special-orthogonal matrices.

        Parameters
        ----------
        params : torch.Tensor
            Parameters of shape ``[batch_size, d * (d - 1) // 2]`` holding the
            strictly lower-triangular entries of the generator.

        Returns
        -------
        torch.Tensor
            Orthogonal matrices of shape ``[batch_size, d, d]``.

        Raises
        ------
        ValueError
            If ``params`` does not have ``d * (d - 1) // 2`` columns.
        """
        expected = num_orthogonal_params(self.d)
        if params.size(-1) != expected:
            raise ValueError(
                f"Expected {expected} orthogonal parameters for d={self.d}, "
                f"got {params.size(-1)}"
            )

        # offset=-1 excludes the diagonal, which A = P - P.T would erase.
        tril_indices = torch.tril_indices(
            row=self.d, col=self.d, offset=-1, device=params.device
        )
        generator = torch.zeros(
            (params.size(0), self.d, self.d),
            dtype=params.dtype,
            device=params.device,
        )
        generator[:, tril_indices[0], tril_indices[1]] = params

        skew = generator - generator.transpose(-2, -1)
        if self.orthogonal_map == "matrix_exp":
            return torch.matrix_exp(skew)

        identity = torch.eye(self.d, dtype=skew.dtype, device=skew.device)
        return torch.linalg.solve(
            torch.add(identity, skew, alpha=-0.5),
            torch.add(identity, skew, alpha=0.5),
        )
