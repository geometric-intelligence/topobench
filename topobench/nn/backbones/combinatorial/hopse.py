"""HOPSE model."""

import torch
import torch.nn.functional


class HOPSE(torch.nn.Module):
    r"""HOPSE model.

    Parameters
    ----------
    in_channels : tuple of int or int
        Dimension of the hidden layers.
    hidden_channels : int
        Dimension of the output layer.
    update_func : str
        Update function.
    complex_dim : int
        Dimension of the complex.
    max_hop : int
        Number of hops.
    n_layers : int
        Number of layers.
    layer_norm : bool, optional
        Wether to perform layer normalization.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        update_func=None,
        complex_dim=3,
        max_hop=3,
        n_layers=2,
        layer_norm=True,
    ):
        super().__init__()
        self.complex_dim = complex_dim
        self.max_hop = max_hop
        self.layer_norm = layer_norm

        assert n_layers >= 1

        if isinstance(in_channels, int):  # If only one value is passed
            in_channels = [in_channels] * self.max_hop

        self.layers = torch.nn.ModuleList()

        # Set of simplices layers
        self.layers_0 = torch.nn.ModuleList(
            HOPSELayer(
                [in_channels[i] for i in range(max_hop)],
                [hidden_channels] * max_hop,
                update_func=update_func,
                max_hop=max_hop,
            )
            for i in range(complex_dim)
        )
        self.layers.append(self.layers_0)

        # From layer 1 to n_layers
        for i in range(1, n_layers):
            self.layers.append(
                torch.nn.ModuleList(
                    HOPSELayer(
                        [hidden_channels] * max_hop,
                        [hidden_channels] * max_hop,
                        update_func=update_func,
                        max_hop=max_hop,
                    )
                    for i in range(complex_dim)
                )
            )

    def forward(self, x):
        r"""Forward pass of the model.

        Parameters
        ----------
        x : tuple(tuple(torch.Tensor))
            Tuple of tuple containing the input tensors for each simplex.

        Returns
        -------
        tuple(tuple(torch.Tensor))
            Tuple of tuples of final hidden state tensors.
        """

        # The follwing line will mean the same as:
        # # For each k: 0 to k (k=0,1,2)
        # x_0_tup = tuple(self.in_linear_0[i](x[0][i]) for i in range(3))
        # # For each k: 1 to k (k=0,1,2)
        # x_1_tup = tuple(self.in_linear_1[i](x[1][i]) for i in range(3))
        # # For each k: 2 to k (k=0,1,2)
        # x_2_tup = tuple(self.in_linear_2[i](x[2][i]) for i in range(3))

        # For each layer in the network
        for layer in self.layers:
            # For each simplex dimension (0, 1, 2)
            x = tuple(layer[i](x[i]) for i in range(self.complex_dim))
        return x


class HOPSELayer(torch.nn.Module):
    r"""One layer in the HOPSE model.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    max_hop : int
        Number of hop representations to consider.
    aggr_norm : bool
        Whether to perform aggregation normalization.
    update_func : str
        Update function.
    initialization : str
        Initialization method.
    layer_norm : bool, optional
        Whether to apply layer normalization (default: True).

    Returns
    -------
    torch.Tensor
        Output
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        max_hop,
        aggr_norm: bool = True,
        update_func=None,
        initialization: str = "xavier_uniform",
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        assert max_hop == len(in_channels), (
            "Number of hops must be equal to the number of input channels."
        )
        assert max_hop == len(out_channels), (
            "Number of hops must be equal to the number of output channels."
        )

        self.max_hop = max_hop
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization

        self.layer_norm = layer_norm

        assert initialization in ["xavier_uniform", "xavier_normal"]

        self.list_linear = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    in_features=self.in_channels[i],
                    out_features=self.out_channels[i],
                )
                for i in range(max_hop)
            ]
        )

        if self.layer_norm:
            # self.LN = torch.nn.ModuleList(
            #     torch.nn.BatchNorm1d(self.out_channels[i])
            #     for i in range(max_hop)
            # )
            self.LN = torch.nn.ModuleList(
                torch.nn.LayerNorm(self.out_channels[i])
                for i in range(max_hop)
            )
        else:
            self.LN = torch.nn.ModuleList(
                torch.nn.Identity() for i in range(max_hop)
            )

    def update(self, x: torch.Tensor):
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
        if self.update_func == "gelu":
            return torch.nn.functional.gelu(x)
        if self.update_func == "silu":
            return torch.nn.functional.silu(x)
        return None

    def forward(self, x_all: dict[int, torch.Tensor]):
        r"""Forward computation.

        Parameters
        ----------
        x_all : Dict[Int, torch.Tensor]
            Dictionary of tensors where each simplex dimension (node, edge, face) represents a key, 0-indexed.

        Returns
        -------
        torch.Tensor
            Output tensors for each 0-cell.
        torch.Tensor
            Output tensors for each 1-cell.
        torch.Tensor
            Output tensors for each 2-cell.
        """
        y_k_t = [
            linear_layer(x)
            for x, linear_layer in zip(x_all, self.list_linear, strict=False)
        ]

        if self.update_func is None:
            return tuple(y_k_t.values())

        # Maybe add skip-connections here: x = LN(x + x_0)
        # x_all
        # x_all = tuple([self.update(y_t) for y_t in y_k_t.values()])

        x_out = []
        for ln, y, x in zip(self.LN, y_k_t, x_all, strict=False):
            y_t = self.update(y + x)
            y_t = ln(y_t)
            x_out.append(y_t)

        return tuple(x_out)
