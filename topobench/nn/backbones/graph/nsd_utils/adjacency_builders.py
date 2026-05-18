"""
Sheaf adjacency builders for Sheaf Attention Networks.

These builders mirror the LaplacianBuilder family but assemble the
attention-weighted sheaf adjacency matrix used by SheafAN
(Barbero et al., 2022), rather than the sheaf Laplacian used by NSD.

For an ordered tril edge (i, j) with i < j, the (i, j) and (j, i) blocks
of the sheaf adjacency are P_ij = F_i^T F_j and P_ji = P_ij^T. SheafAN
takes the Hadamard product of A_F with a Kronecker-broadcast attention
matrix, so each block is rescaled by a scalar attention coefficient that
depends on direction (alpha_ij for the (i, j) block, alpha_ji for the
(j, i) block). Diagonal blocks are identity matrices scaled by the
self-loop attention alpha_ii.
"""

import os
import sys

import torch
from torch import nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .laplace import (
    compute_learnable_diag_laplacian_indices,
    compute_learnable_laplacian_indices,
    compute_left_right_map_index,
    mergesp,
)
from .orthogonal import Orthogonal


class _SheafAdjacencyBuilder(nn.Module):
    """
    Base class for sheaf adjacency builders.

    Stores the sparse index plumbing needed to assemble the attention
    weighted sheaf adjacency matrix from learned restriction maps and a
    vector of attention coefficients over an augmented edge index
    (original edges followed by N self-loops).

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Directed edge indices of shape [2, num_edges] (no self-loops;
        bidirectional pairs expected).
    d : int
        Dimension of the stalk space.
    """

    def __init__(self, size, edge_index, d):
        super().__init__()
        self.d = d
        self.size = size
        self.edge_index = edge_index
        self.num_edges = edge_index.size(1)

        self.left_right_idx, self.vertex_tril_idx = (
            compute_left_right_map_index(edge_index)
        )

    def _split_alpha(self, alpha):
        """
        Slice the augmented attention vector into tril/triu/self parts.

        Parameters
        ----------
        alpha : torch.Tensor
            Attention coefficients of shape [num_edges + size]. The
            first num_edges entries align with the original directed
            edge index; the trailing size entries are self-loop scores
            in node order.

        Returns
        -------
        alpha_tril : torch.Tensor
            Attention values for tril edges (i, j) with i < j.
        alpha_triu : torch.Tensor
            Attention values for the reverse edges (j, i).
        alpha_self : torch.Tensor
            Self-loop attention values, one per node.
        """
        left_idx, right_idx = self.left_right_idx
        alpha_tril = alpha[left_idx]
        alpha_triu = alpha[right_idx]
        alpha_self = alpha[self.num_edges :]
        return alpha_tril, alpha_triu, alpha_self


class DiagSheafAdjacencyBuilder(_SheafAdjacencyBuilder):
    """
    Sheaf adjacency builder with diagonal restriction maps.

    Each restriction map is parameterized by d scalar values. The
    resulting adjacency blocks P_ij are diagonal and are the elementwise
    product of the two endpoint maps.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Directed edge indices of shape [2, num_edges].
    d : int
        Dimension of the diagonal stalk space.
    """

    def __init__(self, size, edge_index, d):
        super().__init__(size, edge_index, d)
        self.diag_indices, self.tril_indices = (
            compute_learnable_diag_laplacian_indices(
                size, self.vertex_tril_idx, self.d, self.d
            )
        )

    def forward(self, maps, alpha):
        """
        Build the attention-weighted diagonal sheaf adjacency.

        Parameters
        ----------
        maps : torch.Tensor
            Diagonal restriction map parameters of shape [num_edges, d].
        alpha : torch.Tensor
            Attention coefficients of shape [num_edges + size].

        Returns
        -------
        A : tuple of torch.Tensor
            Sparse adjacency as (indices, values).
        saved_tril_maps : torch.Tensor
            Unscaled tril transport maps, for analysis.
        """
        assert maps.dim() == 2 and maps.size(1) == self.d
        left_idx, right_idx = self.left_right_idx
        alpha_tril, alpha_triu, alpha_self = self._split_alpha(alpha)

        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = left_maps * right_maps
        saved_tril_maps = tril_maps.detach().clone()

        tril_vals = (tril_maps * alpha_tril.unsqueeze(-1)).view(-1)
        triu_vals = (tril_maps * alpha_triu.unsqueeze(-1)).view(-1)
        diag_vals = alpha_self.unsqueeze(-1).expand(-1, self.d).reshape(-1)

        triu_indices = torch.empty_like(self.tril_indices)
        triu_indices[0], triu_indices[1] = (
            self.tril_indices[1],
            self.tril_indices[0],
        )
        non_diag_indices, non_diag_values = mergesp(
            self.tril_indices, tril_vals, triu_indices, triu_vals
        )
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, self.diag_indices, diag_vals
        )
        return (edge_index, weights), saved_tril_maps


class NormConnectionSheafAdjacencyBuilder(_SheafAdjacencyBuilder):
    """
    Sheaf adjacency builder with orthogonal restriction maps.

    Mirrors the paper-faithful setting of SheafAN, where restriction
    maps live in O(d). The lower-triangular Cayley or matrix-exponential
    parameterization is reused from NSD's bundle builder. No degree
    normalization is applied, in keeping with equation (5) of the SAN
    paper.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Directed edge indices of shape [2, num_edges].
    d : int
        Dimension of the stalk space.
    orth_map : str or None, optional
        Orthogonalization method, 'cayley' or 'matrix_exp'. Default is
        None (which the Orthogonal layer will reject).
    """

    def __init__(self, size, edge_index, d, orth_map=None):
        super().__init__(size, edge_index, d)
        self.orth_transform = Orthogonal(d=self.d, orthogonal_map=orth_map)
        self.orth_map = orth_map

        _, self.tril_indices = compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.d
        )
        self.diag_indices, _ = compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.d
        )

    def forward(self, map_params, alpha):
        """
        Build the attention-weighted orthogonal sheaf adjacency.

        Parameters
        ----------
        map_params : torch.Tensor
            Orthogonal map parameters of shape
            [num_edges, d * (d + 1) / 2].
        alpha : torch.Tensor
            Attention coefficients of shape [num_edges + size].

        Returns
        -------
        A : tuple of torch.Tensor
            Sparse adjacency as (indices, values).
        saved_tril_maps : torch.Tensor
            Unscaled tril transport maps, for analysis.
        """
        assert map_params.dim() == 2
        assert map_params.size(1) == self.d * (self.d + 1) // 2
        left_idx, right_idx = self.left_right_idx
        alpha_tril, alpha_triu, alpha_self = self._split_alpha(alpha)

        maps = self.orth_transform(map_params)
        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        # P_ij = F_i^T F_j -- no negation, unlike the Laplacian.
        tril_maps = torch.bmm(torch.transpose(left_maps, -1, -2), right_maps)
        saved_tril_maps = tril_maps.detach().clone()

        tril_vals = (tril_maps * alpha_tril.view(-1, 1, 1)).view(-1)
        triu_vals = (tril_maps * alpha_triu.view(-1, 1, 1)).view(-1)
        diag_vals = alpha_self.unsqueeze(-1).expand(-1, self.d).reshape(-1)

        triu_indices = torch.empty_like(self.tril_indices)
        triu_indices[0], triu_indices[1] = (
            self.tril_indices[1],
            self.tril_indices[0],
        )
        non_diag_indices, non_diag_values = mergesp(
            self.tril_indices, tril_vals, triu_indices, triu_vals
        )
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, self.diag_indices, diag_vals
        )
        return (edge_index, weights), saved_tril_maps


class GeneralSheafAdjacencyBuilder(_SheafAdjacencyBuilder):
    """
    Sheaf adjacency builder with general (unrestricted) restriction maps.

    Each restriction map is a full d x d matrix.

    Parameters
    ----------
    size : int
        Number of nodes in the graph.
    edge_index : torch.Tensor
        Directed edge indices of shape [2, num_edges].
    d : int
        Dimension of the stalk space.
    """

    def __init__(self, size, edge_index, d):
        super().__init__(size, edge_index, d)
        _, self.tril_indices = compute_learnable_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.d
        )
        self.diag_indices, _ = compute_learnable_diag_laplacian_indices(
            size, self.vertex_tril_idx, self.d, self.d
        )

    def forward(self, maps, alpha):
        """
        Build the attention-weighted general sheaf adjacency.

        Parameters
        ----------
        maps : torch.Tensor
            Full restriction map matrices of shape [num_edges, d, d].
        alpha : torch.Tensor
            Attention coefficients of shape [num_edges + size].

        Returns
        -------
        A : tuple of torch.Tensor
            Sparse adjacency as (indices, values).
        saved_tril_maps : torch.Tensor
            Unscaled tril transport maps, for analysis.
        """
        assert maps.dim() == 3
        assert maps.size(1) == self.d and maps.size(2) == self.d
        assert torch.all(torch.isfinite(maps))
        left_idx, right_idx = self.left_right_idx
        alpha_tril, alpha_triu, alpha_self = self._split_alpha(alpha)

        left_maps = torch.index_select(maps, index=left_idx, dim=0)
        right_maps = torch.index_select(maps, index=right_idx, dim=0)
        tril_maps = torch.bmm(
            torch.transpose(left_maps, dim0=-1, dim1=-2), right_maps
        )
        saved_tril_maps = tril_maps.detach().clone()

        tril_vals = (tril_maps * alpha_tril.view(-1, 1, 1)).view(-1)
        triu_vals = (tril_maps * alpha_triu.view(-1, 1, 1)).view(-1)
        diag_vals = alpha_self.unsqueeze(-1).expand(-1, self.d).reshape(-1)

        triu_indices = torch.empty_like(self.tril_indices)
        triu_indices[0], triu_indices[1] = (
            self.tril_indices[1],
            self.tril_indices[0],
        )
        non_diag_indices, non_diag_values = mergesp(
            self.tril_indices, tril_vals, triu_indices, triu_vals
        )
        edge_index, weights = mergesp(
            non_diag_indices, non_diag_values, self.diag_indices, diag_vals
        )
        return (edge_index, weights), saved_tril_maps
