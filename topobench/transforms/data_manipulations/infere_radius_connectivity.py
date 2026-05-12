"""InfereRadiusConnectivity class definition."""

import torch_geometric
from torch_cluster import radius_graph


class InfereRadiusConnectivity(torch_geometric.transforms.BaseTransform):
    r"""Class to infer point cloud connectivity.

    The transform generates the radius connectivity of the input point cloud.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "infer_radius_connectivity"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        data.edge_index = radius_graph(data.x, **self.parameters["args"])
        return data
