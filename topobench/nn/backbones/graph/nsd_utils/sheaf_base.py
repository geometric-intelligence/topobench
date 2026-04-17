"""Base classes for sheaf neural network layers."""
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

from torch import nn


class SheafDiffusion(nn.Module):
    """
    Base class for sheaf diffusion models.

    This class provides the foundational structure for all sheaf diffusion variants,
    storing common parameters and configurations.

    Parameters
    ----------
    edge_index : torch.Tensor or None
        Edge indices of shape [2, num_edges]. Can be None for inductive models.
    args : dict
        Configuration dictionary containing:
        - d (int): Dimension of the stalk space (must be > 0).
        - hidden_channels (int): Number of hidden channels per stalk dimension.
        - device (str): Device to run the model on.
        - layers (int): Number of diffusion layers.
        - input_dropout (float): Dropout rate for input layer.
        - dropout (float): Dropout rate for hidden layers.
        - input_dim (int): Dimension of input features.
        - output_dim (int): Dimension of output features.
        - sheaf_act (str): Activation function for sheaf learning.
        - orth (str): Orthogonalization method.
    """

    def __init__(self, edge_index, args):
        super().__init__()

        assert args["d"] > 0
        self.d = args["d"]
        self.edge_index = edge_index
        self.hidden_dim = args["hidden_channels"] * self.d
        self.device = args["device"]
        self.layers = args["layers"]
        self.input_dropout = args["input_dropout"]
        self.dropout = args["dropout"]
        self.input_dim = args["input_dim"]
        self.hidden_channels = args["hidden_channels"]
        self.output_dim = args["output_dim"]
        self.sheaf_act = args["sheaf_act"]
        self.orth_trans = args["orth"]
        self.laplacian_builder = None
