"""Orthogonal transformations for sheaf diffusion."""
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class Orthogonal(nn.Module):
    """
    Orthogonal transformation module for sheaf restriction maps.

    Converts lower-triangular parameters into orthogonal matrices using either
    the matrix exponential or Cayley transform. Based on PyTorch's parametrization
    utilities.

    Reference: https://pytorch.org/docs/stable/_modules/torch/nn/utils/parametrizations.html#orthogonal

    Parameters
    ----------
    d : int
        Dimension of the square orthogonal matrices to generate.
    orthogonal_map : str
        Method for generating orthogonal matrices. Options are 'matrix_exp' or 'cayley'.

    Raises
    ------
    AssertionError
        If orthogonal_map is not 'matrix_exp' or 'cayley'.
    """

    def __init__(self, d, orthogonal_map):
        super().__init__()
        assert orthogonal_map in ["matrix_exp", "cayley"]
        self.d = d
        self.orthogonal_map = orthogonal_map

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert parameters to orthogonal matrices.

        Parameters
        ----------
        params : torch.Tensor
            Lower-triangular parameters of shape [batch_size, d*(d+1)/2].

        Returns
        -------
        torch.Tensor
            Orthogonal matrices of shape [batch_size, d, d].

        Raises
        ------
        ValueError
            If an unsupported orthogonal_map method is specified.
        """

        offset = 0
        tril_indices = torch.tril_indices(
            row=self.d, col=self.d, offset=offset, device=params.device
        )
        new_params = torch.zeros(
            (params.size(0), self.d, self.d),
            dtype=params.dtype,
            device=params.device,
        )
        new_params[:, tril_indices[0], tril_indices[1]] = params
        params = new_params

        params = params.tril()
        A = params - params.transpose(-2, -1)
        # A is skew-symmetric (or skew-hermitian)
        if self.orthogonal_map == "matrix_exp":
            Q = torch.matrix_exp(A)
        elif self.orthogonal_map == "cayley":
            # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
            Id = torch.eye(self.d, dtype=A.dtype, device=A.device)
            Q = torch.linalg.solve(
                torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5)
            )
        else:
            raise ValueError(
                f"Unsupported transformations {self.orthogonal_map}"
            )

        return Q
