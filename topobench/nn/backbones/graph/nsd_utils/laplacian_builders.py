# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Laplacian builders for Neural Sheaf Diffusion.

This module provides builders for constructing different types of sheaf Laplacians:
diagonal, bundle (with orthogonal maps), and general (full matrices).
"""

import os
import sys

import torch
from torch import nn
from torch_geometric.utils import degree
from torch_scatter import scatter_add

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .laplace import (
    compute_learnable_diag_laplacian_indices,
    compute_learnable_laplacian_indices,
    compute_left_right_map_index,
    mergesp,
)
from .orthogonal import Orthogonal


class LaplacianBuilder(nn.Module):
    """
    Base class for building sheaf Laplacians.

    This class provides common functionality for all Laplacian builders,
    including preprocessing edge indices and computing normalization.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    d : int
        Dimension of the stalk space.
    normalised : bool, optional
        Whether to use normalized Laplacian. Default is False.
    deg_normalised : bool, optional
        Whether to use degree normalization (not used). Default is False.
    """

    def __init__(
        self,
        size,
        edge_index,
        d,
        normalised=False,
        deg_normalised=False,
    ):
        super().__init__()

        self.d = d
        self.size = size
        self.edges = edge_index.size(1) // 2
        self.edge_index = edge_index
        self.normalised = normalised
        self.device = edge_index.device

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.full_left_right_idx, _ = compute_left_right_map_index(
            edge_index, full_matrix=True
        )
        self.left_right_idx, self.vertex_tril_idx = (
            compute_left_right_map_index(edge_index)
        )
        self.deg = degree(self.edge_index[0], num_nodes=self.size)

    def scalar_normalise(self, diag, tril, row, col):
        """
        Apply scalar normalization to Laplacian entries.

        Normalizes diagonal and off-diagonal entries by node degrees,
        similar to symmetric normalization in standard graph Laplacians.

        Parameters
        ----------
        diag : torch.Tensor
            Diagonal block values.
        tril : torch.Tensor
            Lower triangular block values.
        row : torch.Tensor
            Row indices of edges.
        col : torch.Tensor
            Column indices of edges.

        Returns
        -------
        diag_maps : torch.Tensor
            Normalized diagonal block values.
        non_diag_maps : torch.Tensor
            Normalized off-diagonal block values.
        """
        if tril.dim() > 2:
            assert tril.size(-1) == tril.size(-2)
            assert diag.dim() == 2
        d = diag.size(-1)
        diag_sqrt_inv = (diag + 1).pow(-0.5)

        diag_sqrt_inv = (
            diag_sqrt_inv.view(-1, 1, 1)
            if tril.dim() > 2
            else diag_sqrt_inv.view(-1, d)
        )
        left_norm = diag_sqrt_inv[row]
        right_norm = diag_sqrt_inv[col]
        non_diag_maps = left_norm * tril * right_norm

        diag_sqrt_inv = (
            diag_sqrt_inv.view(-1, 1, 1)
            if diag.dim() > 2
            else diag_sqrt_inv.view(-1, d)
        )
        diag_maps = diag_sqrt_inv**2 * diag

        return diag_maps, non_diag_maps


class DiagLaplacianBuilder(LaplacianBuilder):
    """
    Builder for sheaf Laplacian with diagonal restriction maps.

    This builder constructs a sheaf Laplacian where the restriction maps
    are diagonal matrices, parameterized by d values per edge.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    d : int
        Dimension of the diagonal stalk space.
    """

    def __init__(
        self,
        size,
        edge_index,
        d,
    ):
        super().__init__(size, edge_index, d)

        self.diag_indices, self.tril_indices = (
            compute_learnable_diag_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.d
            )
        )

    def forward(self, maps):
        """
        Build the sheaf Laplacian from diagonal restriction maps.

        Parameters
        ----------
        maps : torch.Tensor
            Diagonal restriction map parameters of shape [num_edges, d].

        Returns
        -------
        L : tuple of torch.Tensor
            Sparse Laplacian representation as (indices, values).
        saved_tril_maps : torch.Tensor
            Saved lower triangular restriction maps for analysis.
        """
        assert len(maps.size()) == 2
        assert maps.size(1) == self.d
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        row, _ = self.edge_index

        # Compute the un-normalised Laplacian entries.
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -left_maps * right_maps
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = scatter_add(maps**2, row, dim=0, dim_size=self.size)

        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        tril_maps, diag_maps = tril_maps.view(-1), diag_maps.view(-1)

        # Add the upper triangular part
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = mergesp(
            tril_indices, tril_maps, triu_indices, tril_maps
        )

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_maps
        )

        return (edge_index, weights), saved_tril_maps


class NormConnectionLaplacianBuilder(LaplacianBuilder):
    """
    Builder for normalized bundle sheaf Laplacian with orthogonal restriction maps.

    This builder constructs a normalized sheaf Laplacian where the restriction maps
    are orthogonal matrices parameterized via Cayley transform or matrix exponential.
    Used for bundle sheaf models.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    d : int
        Dimension of the stalk space.
    orth_map : str or None, optional
        Method for orthogonalization ('cayley' or 'matrix_exp'). Default is None.
    """

    def __init__(self, size, edge_index, d, orth_map=None):
        super().__init__(
            size,
            edge_index,
            d,
            normalised=True,
        )
        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=orth_map)
        self.orth_map = orth_map

        _, self.tril_indices = compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.d
        )
        self.diag_indices, _ = compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.d
        )

    def forward(self, map_params):
        """
        Build the normalized sheaf Laplacian from orthogonal restriction maps.

        Parameters
        ----------
        map_params : torch.Tensor
            Orthogonal map parameters of shape [num_edges, d*(d+1)/2].

        Returns
        -------
        L : tuple of torch.Tensor
            Sparse normalized Laplacian representation as (indices, values).
        saved_tril_maps : torch.Tensor
            Saved lower triangular transport maps for analysis.
        """
        assert len(map_params.size()) == 2
        assert map_params.size(1) == self.d * (self.d + 1) // 2

        _, full_right_idx = self.full_left_right_idx
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # Convert the parameters to orthogonal matrices.
        maps = self.orth_transform(map_params)
        diag_maps = self.deg.unsqueeze(-1)

        # Compute the transport maps.
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(torch.transpose(left_maps, -1, -2), right_maps)
        saved_tril_maps = tril_maps.detach().clone()

        # Normalise the entries if the normalised Laplacian is used.
        diag_maps, tril_maps = self.scalar_normalise(
            diag_maps, tril_maps, tril_row, tril_col
        )
        tril_maps, diag_maps = (
            tril_maps.view(-1),
            diag_maps.expand(-1, self.d).reshape(-1),
        )

        # Add the upper triangular part
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = mergesp(
            tril_indices, tril_maps, triu_indices, tril_maps
        )

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_maps
        )

        return (edge_index, weights), saved_tril_maps


class GeneralLaplacianBuilder(LaplacianBuilder):
    """
    Builder for general sheaf Laplacian with full matrix restriction maps.

    This builder constructs a sheaf Laplacian where the restriction maps
    are arbitrary d x d matrices learned from data.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    d : int
        Dimension of the stalk space.
    augmented : bool, optional
        Whether to use augmented representation (not currently used). Default is True.
    """

    def __init__(
        self,
        size,
        edge_index,
        d,
        augmented=True,
    ):
        super().__init__(
            size,
            edge_index,
            d,
        )

        # Preprocess the sparse indices required to compute the Sheaf Laplacian.
        self.diag_indices, self.tril_indices = (
            compute_learnable_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.d
            )
        )

    def forward(self, maps):
        """
        Build the sheaf Laplacian from general restriction maps.

        Parameters
        ----------
        maps : torch.Tensor
            General restriction map matrices of shape [num_edges, d, d].

        Returns
        -------
        L : tuple of torch.Tensor
            Sparse Laplacian representation as (indices, values).
        saved_tril_maps : torch.Tensor
            Saved lower triangular transport maps for analysis.
        """
        left_idx, right_idx = self.left_right_idx
        tril_row, tril_col = self.vertex_tril_idx
        tril_indices, diag_indices = self.tril_indices, self.diag_indices
        row, _ = self.edge_index

        # Compute transport maps.
        assert torch.all(torch.isfinite(maps))
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = -torch.bmm(
            torch.transpose(left_maps, dim0=-1, dim1=-2), right_maps
        )
        saved_tril_maps = tril_maps.detach().clone()
        diag_maps = torch.bmm(torch.transpose(maps, dim0=-1, dim1=-2), maps)
        diag_maps = scatter_add(diag_maps, row, dim=0, dim_size=self.size)
        diag_maps, tril_maps = diag_maps.view(-1), tril_maps.view(-1)

        # Add the upper triangular part.
        triu_indices = torch.empty_like(tril_indices)
        triu_indices[0], triu_indices[1] = tril_indices[1], tril_indices[0]
        non_diag_indices, non_diag_values = mergesp(
            tril_indices, tril_maps, triu_indices, tril_maps
        )

        # Merge diagonal and non-diagonal
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, diag_indices, diag_maps
        )

        return (edge_index, weights), saved_tril_maps
