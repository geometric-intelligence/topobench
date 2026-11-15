"""Class to apply BaseEncoder to the features of higher order structures."""

import sklearn
import torch
import torch_geometric

from topobench.nn.encoders.base import AbstractFeatureEncoder


class AllPCAFeatureEncoder(AbstractFeatureEncoder):
    r"""Encoder class to apply BaseEncoder.

    The BaseEncoder is applied to the features of higher order
    structures. The class creates a BaseEncoder for each dimension specified in
    selected_dimensions. Then during the forward pass, the BaseEncoders are
    applied to the features of the corresponding dimensions.

    Parameters
    ----------
    in_channels : list[int]
        Input dimensions for the features.
    out_channels : list[int]
        Output dimensions for the features.
    proj_dropout : float, optional
        Dropout for the BaseEncoders (default: 0).
    selected_dimensions : list[int], optional
        List of indexes to apply the BaseEncoders to (default: None).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = (
            selected_dimensions
            if (
                selected_dimensions is not None
            )  # and len(selected_dimensions) <= len(self.in_channels))
            else range(len(self.in_channels))
        )
        for i in self.dimensions:
            setattr(
                self,
                f"encoder_{i}",
                BasePCA(
                    self.in_channels[i],
                    self.out_channels,
                ),
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, dimensions={self.dimensions})"

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass.

        The method applies the BaseEncoders to the features of the selected_dimensions.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain x_{i} features for each i in the selected_dimensions.

        Returns
        -------
        torch_geometric.data.Data
            Output data object with updated x_{i} features.
        """
        if not hasattr(data, "x_0"):
            data.x_0 = data.x

        for i in self.dimensions:
            if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                batch = getattr(data, f"batch_{i}")
                data["x_0"] = getattr(self, f"encoder_{i}")(data["x_0"], batch)
        return data


class BasePCA(torch.nn.Module):
    r"""Base encoder class used by AllCellFeatureEncoder.

    This class uses two linear layers with GraphNorm, Relu activation function, and dropout between the two layers.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimensions of output features.
    dropout : float, optional
        Percentage of channels to discard between the two linear layers (default: 0).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pca = sklearn.decomposition.PCA(n_components=out_channels)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.linear1.in_features}, out_channels={self.linear1.out_features})"

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the encoder.

        It applies two linear layers with GraphNorm, Relu activation function, and dropout between the two layers.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of dimensions [N, in_channels].
        batch : torch.Tensor
            The batch vector which assigns each element to a specific example.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [N, out_channels].
        """
        device = x.device

        if self.pca.n_components >= x.shape[1]:
            print(
                f"n_components ({self.pca.n_components}) must be less than the number of features ({x.shape[1]}). Ignoring PCA"
            )
            return x

        x = (
            torch.Tensor(self.pca.fit_transform(x.cpu().numpy()))
            .to(device)
            .float()
        )
        return x
