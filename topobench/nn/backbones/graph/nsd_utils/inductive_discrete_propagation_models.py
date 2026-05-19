# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Inductive Neural Sheaf Propagation models.

This module implements three variants of inductive sheaf propagation:
- Diagonal: Diagonal restriction maps
- Bundle: Orthogonal restriction maps with normalization
- General: Full matrix restriction maps
"""

import torch
import torch.nn.functional as F
import torch_sparse
from torch import nn

from .laplacian_builders import (
    DiagLaplacianBuilder,
    GeneralLaplacianBuilder,
    NormConnectionLaplacianBuilder,
)
from .sheaf_base import SheafPropagation
from .sheaf_models import LocalConcatSheafLearner


class InductiveDiscreteDiagSheafPropagation(SheafPropagation):
    """
    Inductive sheaf propagation with diagonal restriction maps.

    This model learns diagonal d x d restriction maps for each edge,
    parameterized by d scalar values. Suitable for problems where
    feature channels can be processed independently.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - d (int): Dimension of stalk space (must be > 0).
        - layers (int): Number of propagation layers.
        - hidden_channels (int): Hidden channels per stalk dimension.
        - input_dim (int): Input feature dimension.
        - output_dim (int): Output feature dimension.
        - device (str): Device to run on.
        - input_dropout (float): Input layer dropout rate.
        - dropout (float): Hidden layer dropout rate.
        - sheaf_act (str): Activation for sheaf learning.
    """

    def __init__(self, config):
        super().__init__(None, config)
        assert config["d"] > 0

        self.config = config
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        for _i in range(self.layers):
            self.lin_right_weights.append(
                nn.Linear(
                    self.hidden_channels, self.hidden_channels, bias=False
                )
            )
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for _i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers)
        for _i in range(num_sheaf_learners):
            self.sheaf_learners.append(
                LocalConcatSheafLearner(
                    self.hidden_dim,
                    out_shape=(self.d,),
                    sheaf_act=self.sheaf_act,
                )
            )

        self.epsilons = nn.ParameterList()
        for _i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of diagonal sheaf propagation.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Output node features of shape [num_nodes, output_dim].
        """
        # Get actual number of nodes dynamically
        actual_num_nodes = x.size(0)

        # Create laplacian builder for this specific graph
        laplacian_builder = DiagLaplacianBuilder(
            actual_num_nodes,
            edge_index,
            d=self.d,
        )

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Use actual number of nodes
        x = x.view(actual_num_nodes * self.d, -1)

        x_prev = x
        x_curr = x
        for layer in range(self.layers):
            x_maps = F.dropout(
                x_curr,
                p=self.dropout if layer > 0 else 0.0,
                training=self.training,
            )
            # Reshape using actual number of nodes
            maps = self.sheaf_learners[layer](
                x_maps.reshape(actual_num_nodes, -1), edge_index
            )
            L, trans_maps = laplacian_builder(maps)
            self.sheaf_learners[layer].set_L(trans_maps)

            x_layer = F.dropout(x_curr, p=self.dropout, training=self.training)

            x_layer = x_layer.t().reshape(-1, self.d)
            x_layer = self.lin_left_weights[layer](x_layer)
            x_layer = x_layer.reshape(-1, actual_num_nodes * self.d).t()
            x_layer = self.lin_right_weights[layer](x_layer)

            x_layer = torch_sparse.spmm(L[0], L[1], x_layer.size(0), x_layer.size(0), x_layer)
            x_layer = F.elu(x_layer)

            x_new = 2 * x_curr - x_prev - x_layer
            x_prev = x_curr
            x_curr = x_new
        
        x = x_curr

        # Reshape using actual number of nodes
        x = x.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveDiscreteBundleSheafPropagation(SheafPropagation):
    """
    Inductive sheaf propagation with orthogonal bundle restriction maps.

    This model learns orthogonal d x d restriction maps for each edge,
    ensuring isometric transport between stalks. Uses normalized Laplacian
    and Cayley/matrix exponential parameterization for orthogonality.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - d (int): Dimension of stalk space (must be > 1).
        - layers (int): Number of propagation layers.
        - hidden_channels (int): Hidden channels per stalk dimension.
        - input_dim (int): Input feature dimension.
        - output_dim (int): Output feature dimension.
        - device (str): Device to run on.
        - input_dropout (float): Input layer dropout rate.
        - dropout (float): Hidden layer dropout rate.
        - sheaf_act (str): Activation for sheaf learning.
        - orth (str): Orthogonalization method ('cayley' or 'matrix_exp').

    Raises
    ------
    AssertionError
        If d is not greater than 1 or hidden_dim is not divisible by d.
    """

    def __init__(self, config):
        super().__init__(None, config)
        assert config["d"] > 1
        assert self.hidden_dim % self.d == 0

        self.config = config
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        self.batch_norms = nn.ModuleList()
        for _i in range(self.layers):
            self.lin_right_weights.append(
                nn.Linear(
                    self.hidden_channels, self.hidden_channels, bias=False
                )
            )
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for _i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()
        self.weight_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers)
        for _i in range(num_sheaf_learners):
            self.sheaf_learners.append(
                LocalConcatSheafLearner(
                    self.hidden_dim,
                    out_shape=(self.get_param_size(),),
                    sheaf_act=self.sheaf_act,
                )
            )

        self.epsilons = nn.ParameterList()
        for _i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def get_param_size(self):
        """
        Get the number of parameters needed for orthogonal maps.

        Returns
        -------
        int
            Number of parameters (d*(d+1)/2 for lower triangular parameterization).
        """
        return self.d * (self.d + 1) // 2

    def left_right_linear(self, x, left, right, actual_num_nodes):
        """
        Apply left and right linear transformations to stalk vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [num_nodes * d, hidden_channels].
        left : nn.Linear
            Left linear transformation (acts on stalk dimension).
        right : nn.Linear
            Right linear transformation (acts on hidden channels).
        actual_num_nodes : int
            Number of nodes in the current graph.

        Returns
        -------
        torch.Tensor
            Transformed tensor of shape [num_nodes * d, hidden_channels].
        """
        x = x.t().reshape(-1, self.d)
        x = left(x)
        x = x.reshape(-1, actual_num_nodes * self.d).t()
        x = right(x)

        return x

    def forward(self, x, edge_index):
        """
        Forward pass of bundle sheaf propagation.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Output node features of shape [num_nodes, output_dim].
        """
        # Get actual number of nodes dynamically
        actual_num_nodes = x.size(0)

        # Create laplacian builder for this specific graph
        laplacian_builder = NormConnectionLaplacianBuilder(
            actual_num_nodes,
            edge_index,
            d=self.d,
            orth_map=self.orth_trans,
        )

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Use actual number of nodes
        x = x.view(
            actual_num_nodes * self.d, -1
        )  # So for each node, we put reshape the output of the lin1 to a tensor of size (final_d, hidden_dim // final_d)
        # This means that if we set "hidden_dim" to 64 and "final_d" to 2, then we have that for each node, we have a tensor of size (2, 32)

        x_prev, x_curr, L = x, x, None
        for layer in range(self.layers):
            # Time each component of the forward pass
            x_maps = F.dropout(
                x_curr,
                p=self.dropout if layer > 0 else 0.0,
                training=self.training,
            )
            x_maps = x_maps.reshape(
                actual_num_nodes, -1
            )  # Reshape using actual number of nodes (so back to the original shape)
            maps = self.sheaf_learners[layer](x_maps, edge_index)
            L, trans_maps = laplacian_builder(maps)
            self.sheaf_learners[layer].set_L(trans_maps)

            x_layer = F.dropout(x_curr, p=self.dropout, training=self.training)

            # Pass actual_num_nodes to left_right_linear
            x_layer = self.left_right_linear(
                x_layer,
                self.lin_left_weights[layer],
                self.lin_right_weights[layer],
                actual_num_nodes,
            )

            # Use the adjacency matrix rather than the diagonal
            x_layer = torch_sparse.spmm(L[0], L[1], x_layer.size(0), x_layer.size(0), x_layer)

            x_layer = F.elu(x_layer)

            x_new = 2 * x_curr - x_prev - x_layer
            x_prev = x_curr
            x_curr = x_new
        
        x = x_curr

        # Reshape using actual number of nodes
        x = x.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x


class InductiveDiscreteGeneralSheafPropagation(SheafPropagation):
    """
    Inductive sheaf propagation with general (unrestricted) restriction maps.

    This model learns arbitrary d x d restriction maps for each edge,
    providing maximum expressiveness but requiring more parameters.
    Each restriction map is a full d x d matrix.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - d (int): Dimension of stalk space (must be > 1).
        - layers (int): Number of propagation layers.
        - hidden_channels (int): Hidden channels per stalk dimension.
        - input_dim (int): Input feature dimension.
        - output_dim (int): Output feature dimension.
        - device (str): Device to run on.
        - input_dropout (float): Input layer dropout rate.
        - dropout (float): Hidden layer dropout rate.
        - sheaf_act (str): Activation for sheaf learning.

    Raises
    ------
    AssertionError
        If d is not greater than 1.
    """

    def __init__(self, config):
        super().__init__(None, config)
        assert config["d"] > 1

        self.config = config
        self.lin_right_weights = nn.ModuleList()
        self.lin_left_weights = nn.ModuleList()

        for _i in range(self.layers):
            self.lin_right_weights.append(
                nn.Linear(
                    self.hidden_channels, self.hidden_channels, bias=False
                )
            )
            nn.init.orthogonal_(self.lin_right_weights[-1].weight.data)
        for _i in range(self.layers):
            self.lin_left_weights.append(nn.Linear(self.d, self.d, bias=False))
            nn.init.eye_(self.lin_left_weights[-1].weight.data)

        self.sheaf_learners = nn.ModuleList()

        num_sheaf_learners = min(self.layers, self.layers)
        for _i in range(num_sheaf_learners):
            self.sheaf_learners.append(
                LocalConcatSheafLearner(
                    self.hidden_dim,
                    out_shape=(self.d, self.d),
                    sheaf_act=self.sheaf_act,
                )
            )

        self.epsilons = nn.ParameterList()
        for _i in range(self.layers):
            self.epsilons.append(nn.Parameter(torch.zeros((self.d, 1))))

        self.lin1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.lin2 = nn.Linear(self.hidden_dim, self.output_dim)

    def left_right_linear(self, x, left, right, actual_num_nodes):
        """
        Apply left and right linear transformations to stalk vectors.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [num_nodes * d, hidden_channels].
        left : nn.Linear
            Left linear transformation (acts on stalk dimension).
        right : nn.Linear
            Right linear transformation (acts on hidden channels).
        actual_num_nodes : int
            Number of nodes in the current graph.

        Returns
        -------
        torch.Tensor
            Transformed tensor of shape [num_nodes * d, hidden_channels].
        """
        x = x.t().reshape(-1, self.d)
        x = left(x)
        x = x.reshape(-1, actual_num_nodes * self.d).t()
        x = right(x)
        return x

    def forward(self, x, edge_index):
        """
        Forward pass of general sheaf propagation.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, input_dim].
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Output node features of shape [num_nodes, output_dim].
        """
        # Get actual number of nodes dynamically
        actual_num_nodes = x.size(0)

        # Create laplacian builder for this specific graph
        laplacian_builder = GeneralLaplacianBuilder(
            actual_num_nodes,
            edge_index,
            d=self.d,
        )

        x = F.dropout(x, p=self.input_dropout, training=self.training)
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Use actual number of nodes
        x = x.view(actual_num_nodes * self.d, -1)

        x_prev, x_curr, L = x, x, None
        for layer in range(self.layers):
            x_maps = F.dropout(
                x_curr,
                p=self.dropout if layer > 0 else 0.0,
                training=self.training,
            )
            # Reshape using actual number of nodes
            maps = self.sheaf_learners[layer](
                x_maps.reshape(actual_num_nodes, -1), edge_index
            )
            L, trans_maps = laplacian_builder(maps)
            self.sheaf_learners[layer].set_L(trans_maps)

            x_layer = F.dropout(x_curr, p=self.dropout, training=self.training)

            # Pass actual_num_nodes to left_right_linear
            x_layer = self.left_right_linear(
                x_layer,
                self.lin_left_weights[layer],
                self.lin_right_weights[layer],
                actual_num_nodes,
            )

            # Use the adjacency matrix rather than the diagonal
            x_layer = torch_sparse.spmm(L[0], L[1], x_layer.size(0), x_layer.size(0), x_layer)

            x_layer = F.elu(x_layer)

            x_new = 2 * x_curr - x_prev - x_layer
            x_prev = x_curr
            x_curr = x_new
        
        x = x_curr

        # To detect the numerical instabilities of SVD.
        assert torch.all(torch.isfinite(x))

        # Reshape using actual number of nodes
        x = x.reshape(actual_num_nodes, -1)
        x = self.lin2(x)
        return x
