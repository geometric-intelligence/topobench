"""Implementation of the Simplicial Complex Convolutional Neural Network (SCCNN) for complex classification."""

import torch
from torch.nn.parameter import Parameter


class SCCNNCustom(torch.nn.Module):
    """SCCNN implementation for complex classification.

    Note: In this task, we can consider the output on any order of simplices for the
    classification task, which of course can be amended by a readout layer.

    Parameters
    ----------
    in_channels_all : tuple of int
        Dimension of input features on (nodes, edges, faces).
    hidden_channels_all : tuple of int
        Dimension of features of hidden layers on (nodes, edges, faces).
    conv_order : int
        Order of convolutions, we consider the same order for all convolutions.
    sc_order : int
        Order of simplicial complex.
    aggr_norm : bool, optional
        Whether to normalize the aggregation (default: False).
    update_func : str, optional
        Update function for the simplicial complex convolution (default: None).
    n_layers : int, optional
        Number of layers (default: 2).
    """

    def __init__(
        self,
        in_channels_all,
        hidden_channels_all,
        conv_order,
        sc_order,
        aggr_norm=False,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()
        # first layer
        # we use an MLP to map the features on simplices of different dimensions to the same dimension
        self.in_linear_0 = torch.nn.Linear(
            in_channels_all[0], hidden_channels_all[0]
        )
        self.in_linear_1 = torch.nn.Linear(
            in_channels_all[1], hidden_channels_all[1]
        )
        self.in_linear_2 = torch.nn.Linear(
            in_channels_all[2], hidden_channels_all[2]
        )

        self.layers = torch.nn.ModuleList(
            SCCNNLayer(
                in_channels=hidden_channels_all,
                out_channels=hidden_channels_all,
                conv_order=conv_order,
                sc_order=sc_order,
                aggr_norm=aggr_norm,
                update_func=update_func,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_all, laplacian_all, incidence_all):
        """Forward computation.

        Parameters
        ----------
        x_all : tuple(tensors)
            Tuple of feature tensors (node, edge, face).
        laplacian_all : tuple(tensors)
            Tuple of Laplacian tensors (graph laplacian L0, down edge laplacian L1_d, upper edge laplacian L1_u, face laplacian L2).
        incidence_all : tuple(tensors)
            Tuple of order 1 and 2 incidence matrices.

        Returns
        -------
        tuple(tensors)
            Tuple of final hidden state tensors (node, edge, face).
        """
        x_0, x_1, x_2 = x_all
        in_x_0 = self.in_linear_0(x_0)
        in_x_1 = self.in_linear_1(x_1)
        in_x_2 = self.in_linear_2(x_2)

        # Forward through SCCNN
        x_all = (in_x_0, in_x_1, in_x_2)
        for layer in self.layers:
            x_all = layer(x_all, laplacian_all, incidence_all)

        return x_all


class SCCNNLayer(torch.nn.Module):
    r"""Layer of a Simplicial Complex Convolutional Neural Network.

    Parameters
    ----------
    in_channels : tuple of int
        Dimensions of input features on nodes, edges, and faces.
    out_channels : tuple of int
        Dimensions of output features on nodes, edges, and faces.
    conv_order : int
        Convolution order of the simplicial filters.
    sc_order : int
        SC order.
    aggr_norm : bool, optional
        Whether to normalize the aggregated message by the neighborhood size (default: False).
    update_func : str, optional
        Activation function used in aggregation layers (default: None).
    initialization : str, optional
        Initialization method for the weights (default: "xavier_normal").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order,
        sc_order,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_normal",
    ) -> None:
        super().__init__()

        in_channels_0, in_channels_1, in_channels_2 = in_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels

        self.in_channels_0 = in_channels_0
        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2
        self.out_channels_0 = out_channels_0
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2

        self.conv_order = conv_order
        self.sc_order = sc_order

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.conv_order > 0

        self.weight_0 = Parameter(
            torch.Tensor(
                self.in_channels_0,
                self.out_channels_0,
                1 + conv_order + 1 + conv_order,
            )
        )

        self.weight_1 = Parameter(
            torch.Tensor(
                self.in_channels_1,
                self.out_channels_1,
                6 * conv_order + 3,
            )
        )

        # determine the third dimensions of the weights
        # because when SC order is larger than 2, there are lower and upper
        # parts for L_2; otherwise, L_2 contains only the lower part

        if sc_order > 2:
            self.weight_2 = Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    4 * conv_order
                    + 2,  # in the future for arbitrary sc_order we should have this 6*conv_order + 3,
                )
            )

        elif sc_order == 2:
            self.weight_2 = Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    4 * conv_order + 2,
                )
            )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight_0, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_2, gain=gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight_0, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_1, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_2, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def aggr_norm_func(self, conv_operator, x):
        r"""Perform aggregation normalization.

        Parameters
        ----------
        conv_operator : torch.sparse
            Convolution operator.
        x : torch.Tensor
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Normalized feature tensor.
        """
        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Updated tensor.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        if self.update_func == "leaky_relu":
            return torch.nn.functional.leaky_relu(x)
        return None

    def chebyshev_conv(self, conv_operator, conv_order, x):
        r"""Perform Chebyshev convolution.

        Parameters
        ----------
        conv_operator : torch.sparse
            Convolution operator.
        conv_order : int
            Order of the convolution.
        x : torch.Tensor
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        num_simplices, num_channels = x.shape
        X = torch.empty(size=(num_simplices, num_channels, conv_order)).to(
            x.device
        )

        if self.aggr_norm:
            X[:, :, 0] = torch.mm(conv_operator, x)
            X[:, :, 0] = self.aggr_norm_func(conv_operator, X[:, :, 0])
            for k in range(1, conv_order):
                X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
                X[:, :, k] = self.aggr_norm_func(conv_operator, X[:, :, k])
        else:
            X[:, :, 0] = torch.mm(conv_operator, x)
            for k in range(1, conv_order):
                X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
        return X

    def forward(self, x_all, laplacian_all, incidence_all):
        r"""Forward computation.

        Parameters
        ----------
        x_all : tuple of tensors
            Tuple of input feature tensors (node, edge, face).
        laplacian_all : tuple of tensors
            Tuple of Laplacian tensors (graph laplacian L0, down edge laplacian L1_d, upper edge laplacian L1_u, face laplacian L2).
        incidence_all : tuple of tensors
            Tuple of order 1 and 2 incidence matrices.

        Returns
        -------
        torch.Tensor
            Output tensor for each 0-cell.
        torch.Tensor
            Output tensor for each 1-cell.
        torch.Tensor
            Output tensor for each 2-cell.
        """
        x_0, x_1, x_2 = x_all

        if self.sc_order == 2:
            laplacian_0, laplacian_down_1, laplacian_up_1, laplacian_2 = (
                laplacian_all
            )
        elif self.sc_order > 2:
            (
                laplacian_0,
                laplacian_down_1,
                laplacian_up_1,
                laplacian_down_2,
                laplacian_up_2,
            ) = laplacian_all

        # num_nodes, num_edges, num_triangles = x_0.shape[0], x_1.shape[0], x_2.shape[0]

        b1, b2 = incidence_all

        # identity_0, identity_1, identity_2 = (
        #     torch.eye(num_nodes).to(x_0.device),
        #     torch.eye(num_edges).to(x_0.device),
        #     torch.eye(num_triangles).to(x_0.device),
        # )
        """
        Convolution in the node space
        """
        # -----------Logic to obtain update for 0-cells --------
        # x_identity_0 = torch.unsqueeze(identity_0 @ x_0, 2)
        # x_0_to_0 = self.chebyshev_conv(laplacian_0, self.conv_order, x_0)
        # x_0_to_0 = torch.cat((x_identity_0, x_0_to_0), 2)

        x_0_laplacian = self.chebyshev_conv(laplacian_0, self.conv_order, x_0)
        x_0_to_0 = torch.cat([x_0.unsqueeze(2), x_0_laplacian], dim=2)
        # -------------------

        # x_1_to_0 = torch.mm(b1, x_1)
        # x_1_to_0_identity = torch.unsqueeze(identity_0 @ x_1_to_0, 2)
        # x_1_to_0 = self.chebyshev_conv(laplacian_0, self.conv_order, x_1_to_0)
        # x_1_to_0 = torch.cat((x_1_to_0_identity, x_1_to_0), 2)

        x_1_to_0_upper = torch.mm(b1, x_1)
        x_1_to_0_laplacian = self.chebyshev_conv(
            laplacian_0, self.conv_order, x_1_to_0_upper
        )
        x_1_to_0 = torch.cat(
            [x_1_to_0_upper.unsqueeze(2), x_1_to_0_laplacian], dim=2
        )
        # -------------------

        x_0_all = torch.cat((x_0_to_0, x_1_to_0), 2)

        # -------------------
        """
        Convolution in the edge space
        """

        # -----------Logic to obtain update for 1-cells --------
        # x_identity_1 = torch.unsqueeze(identity_1 @ x_1, 2)
        # x_1_down = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        # x_1_up = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_1)
        # x_1_to_1 = torch.cat((x_identity_1, x_1_down, x_1_up), 2)

        x_1_down = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        x_1_up = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        x_1_to_1 = torch.cat((x_1.unsqueeze(2), x_1_down, x_1_up), 2)

        # -------------------

        # x_0_to_1 = torch.mm(b1.T, x_0)
        # x_0_to_1_identity = torch.unsqueeze(identity_1 @ x_0_to_1, 2)
        # x_0_to_1 = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_0_to_1)
        # x_0_to_1 = torch.cat((x_0_to_1_identity, x_0_to_1), 2)

        # Lower projection
        x_0_1_lower = torch.mm(b1.T, x_0)

        # Calculate lowwer chebyshev_conv
        x_0_1_down = self.chebyshev_conv(
            laplacian_down_1, self.conv_order, x_0_1_lower
        )

        # Calculate upper chebyshev_conv (Note: in case of signed incidence should be always zero)
        x_0_1_up = self.chebyshev_conv(
            laplacian_up_1, self.conv_order, x_0_1_lower
        )

        # Concatenate output of filters
        x_0_to_1 = torch.cat(
            [x_0_1_lower.unsqueeze(2), x_0_1_down, x_0_1_up], dim=2
        )
        # -------------------

        # x_2_to_1 = torch.mm(b2, x_2)
        # x_2_to_1_identity = torch.unsqueeze(identity_1 @ x_2_to_1, 2)
        # x_2_to_1 = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_2_to_1)
        # x_2_to_1 = torch.cat((x_2_to_1_identity, x_2_to_1), 2)

        x_2_1_upper = torch.mm(b2, x_2)

        # Calculate lowwer chebyshev_conv (Note: In case of signed incidence should be always zero)
        x_2_1_down = self.chebyshev_conv(
            laplacian_down_1, self.conv_order, x_2_1_upper
        )

        # Calculate upper chebyshev_conv
        x_2_1_up = self.chebyshev_conv(
            laplacian_up_1, self.conv_order, x_2_1_upper
        )

        x_2_to_1 = torch.cat(
            [x_2_1_upper.unsqueeze(2), x_2_1_down, x_2_1_up], dim=2
        )

        # -------------------
        x_1_all = torch.cat((x_0_to_1, x_1_to_1, x_2_to_1), 2)
        """Convolution in the face (triangle) space, depending on the SC order,
        the exact form maybe a little different."""
        # -------------------Logic to obtain update for 2-cells --------
        # x_identity_2 = torch.unsqueeze(identity_2 @ x_2, 2)

        # if self.sc_order == 2:
        #     x_2 = self.chebyshev_conv(laplacian_2, self.conv_order, x_2)
        #     x_2_to_2 = torch.cat((x_identity_2, x_2), 2)
        # elif self.sc_order > 2:
        #     x_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_2)
        #     x_2_up = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_2)
        #     x_2_to_2 = torch.cat((x_identity_2, x_2_down, x_2_up), 2)
        x_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_2)
        x_2_up = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_2)
        x_2_to_2 = torch.cat((x_2.unsqueeze(2), x_2_down, x_2_up), 2)

        # -------------------

        # x_1_to_2 = torch.mm(b2.T, x_1)
        # x_1_to_2_identity = torch.unsqueeze(identity_2 @ x_1_to_2, 2)
        # if self.sc_order == 2:
        #     x_1_to_2 = self.chebyshev_conv(laplacian_2, self.conv_order, x_1_to_2)
        # elif self.sc_order > 2:
        #     x_1_to_2 = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_1_to_2)
        # x_1_to_2 = torch.cat((x_1_to_2_identity, x_1_to_2), 2)

        x_1_2_lower = torch.mm(b2.T, x_1)
        x_1_2_down = self.chebyshev_conv(
            laplacian_down_2, self.conv_order, x_1_2_lower
        )
        x_1_2_down = self.chebyshev_conv(
            laplacian_up_2, self.conv_order, x_1_2_lower
        )

        x_1_to_2 = torch.cat(
            [x_1_2_lower.unsqueeze(2), x_1_2_down, x_1_2_down], dim=2
        )

        # That is my code, but to execute this part we need to have simplices order of k+1 in this case order of 3
        # x_3_2_upper = x_1_to_2 = torch.mm(b2, x_3)
        # x_3_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_3_2_upper)
        # x_3_2_up = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_3_2_upper)

        # x_3_to_2 = torch.cat([x_3_2_upper.unsueeze(2), x_3_2_down, x_3_2_up], dim=2)

        # -------------------

        x_2_all = torch.cat([x_1_to_2, x_2_to_2], dim=2)
        # The final version of this model should have the following line
        # x_2_all = torch.cat([x_1_to_2, x_2_to_2, x_3_to_2], dim=2)

        # -------------------

        # Need to check that this einsums are correct
        y_0 = torch.einsum("nik,iok->no", x_0_all, self.weight_0)
        y_1 = torch.einsum("nik,iok->no", x_1_all, self.weight_1)
        y_2 = torch.einsum("nik,iok->no", x_2_all, self.weight_2)

        if self.update_func is None:
            return y_0, y_1, y_2

        return self.update(y_0), self.update(y_1), self.update(y_2)
