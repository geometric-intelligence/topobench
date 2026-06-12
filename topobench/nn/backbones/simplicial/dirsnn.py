"""Directed Simplicial Neural Network Backbone.

This module implements the DirSNN architecture for processing directed
simplicial complexes within the TopoBench framework.
"""

import torch
import torch.nn as nn


class DirSNNLayer(nn.Module):
    """Directed Simplicial Neural Network Layer.

    Computes message passing over directed simplices using oriented boundary matrices.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    """

    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)

        # Learnable weight matrices for lower and upper directed neighborhoods
        self.W_lower = nn.Linear(in_features, out_features, bias=False)
        self.W_upper = nn.Linear(in_features, out_features, bias=False)
        self.W_self = nn.Linear(in_features, out_features, bias=True)

        self.activation = nn.ReLU()

    def forward(self, x, boundary_lower, boundary_upper):
        """Forward pass for the directed layer.

        Parameters
        ----------
        x : torch.Tensor
            Simplex features of shape [num_simplices, in_features].
        boundary_lower : torch.Tensor
            Directed lower incidence matrix.
        boundary_upper : torch.Tensor
            Directed upper incidence matrix.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [num_simplices, out_features].
        """
        # Dropout for regularization
        x = self.dropout(x)

        # Compute message from lower adjacent simplices
        msg_lower = torch.sparse.mm(boundary_lower, x)
        out_lower = self.W_lower(msg_lower)

        # Compute message from upper adjacent simplices
        msg_upper = torch.sparse.mm(boundary_upper, x)
        out_upper = self.W_upper(msg_upper)

        # Self-loop update
        out_self = self.W_self(x)

        # Aggregate directed signals
        out = self.activation(out_lower + out_upper + out_self)
        return out


class DirSNN(nn.Module):
    """The main DirSNN Backbone that integrates into TopoBench.

    Parameters
    ----------
    in_channels : int
        Input feature dimensions.
    hidden_channels : int
        Hidden feature dimensions.
    out_channels : int
        Output feature dimensions.
    num_layers : int, optional
        Number of message passing layers. Default is 2.
    dropout : float, optional
        Dropout probability. Default is 0.5.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        dropout=0.5,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(DirSNNLayer(in_channels, hidden_channels, dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                DirSNNLayer(hidden_channels, hidden_channels, dropout)
            )

        # Output layer
        self.layers.append(DirSNNLayer(hidden_channels, out_channels, dropout))

    def forward(self, batch):
        """Forward pass through the backbone.

        Parameters
        ----------
        batch : Any
            Expected batch inputs from TopoBench's Simplicial wrapper containing edge_x, B1, B2.

        Returns
        -------
        torch.Tensor
            Final node representations.
        """
        # Extract edge features (1-simplices) as the primary representation
        x = batch.edge_x

        # Extract incidence matrices for the directed flow
        B1 = batch.B1
        B2 = batch.B2

        # Compute the directed Hodge Laplacians
        L_down = torch.sparse.mm(B1.t(), B1)
        L_up = torch.sparse.mm(B2, B2.t())

        # Pass messages through layers
        for layer in self.layers:
            x = layer(x, L_down, L_up)

        return x
        