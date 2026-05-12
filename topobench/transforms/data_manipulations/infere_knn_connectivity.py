"""InfereKNNConnectivity class definition."""

import torch_geometric
from torch_cluster import knn_graph


class InfereKNNConnectivity(torch_geometric.transforms.BaseTransform):
    r"""Transform to infer point cloud connectivity.

    The transform generates the k-nearest neighbor connectivity of the input point cloud.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "infer_knn_connectivity"
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
        edge_index = knn_graph(data.x, **self.parameters["args"])

        # Remove duplicates
        data.edge_index = edge_index
        return data
