"""Lifting pointcloud to Voronoi graph induced by Farthest Point Sampling (FPS) support set."""

import torch
import torch_geometric

from topobench.transforms.liftings.pointcloud2hypergraph.base import (
    PointCloud2HypergraphLifting,
)


class VoronoiLifting(PointCloud2HypergraphLifting):
    r"""Lifts pointcloud to Farthest-point Voronoi graph.

    Parameters
    ----------
    support_ratio : float
        Ratio of points to sample with FPS to form voronoi support set.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, support_ratio: float, **kwargs):
        super().__init__(**kwargs)
        self.support_ratio = support_ratio

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift pointcloud to voronoi graph induced by Farthest Point Sampling (FPS) support set.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        from torch_cluster import fps, knn

        # Sample FPS induced Voronoi graph
        support_idcs = fps(data.x, ratio=self.support_ratio)
        target_idcs, source_idcs = knn(data.x[support_idcs], data.x, k=1)

        # Construct incidence matrix
        incidence_matrix = torch.sparse_coo_tensor(
            torch.stack((target_idcs, source_idcs)),
            torch.ones(source_idcs.numel()),
            size=(data.num_nodes, support_idcs.numel()),
        )

        return {
            "incidence_hyperedges": incidence_matrix,
            "num_hyperedges": incidence_matrix.size(1),
            "x_0": data.x,
        }
