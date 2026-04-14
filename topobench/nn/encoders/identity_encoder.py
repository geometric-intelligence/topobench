"""Identity feature encoder that passes through raw features unchanged."""

import torch_geometric

from topobench.nn.encoders.base import AbstractFeatureEncoder


class IdentityFeatureEncoder(AbstractFeatureEncoder):
    r"""Feature encoder that passes through raw features unchanged.

    Useful when the backbone handles its own input projection (e.g.
    ConfigurableGNN with pre_linear=True) or when raw features should
    be fed directly to the backbone.

    Parameters
    ----------
    in_channels : list[int]
        Input dimensions for the features (unused, kept for interface compat).
    out_channels : int
        Output dimension (must match in_channels[0] for true identity).
    selected_dimensions : list[int], optional
        Dimensions to process (default: None = all).
    **kwargs : dict, optional
        Additional keyword arguments (ignored).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        selected_dimensions=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimensions = (
            selected_dimensions
            if selected_dimensions is not None
            else range(len(self.in_channels))
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels})"

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Forward pass — identity (no-op).

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object.

        Returns
        -------
        torch_geometric.data.Data
            Same data object, unmodified.
        """
        if not hasattr(data, "x_0"):
            data.x_0 = data.x
        return data
