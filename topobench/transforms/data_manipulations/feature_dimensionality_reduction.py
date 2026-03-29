"""This module contains a transform that reduces feature dimensionality for the input graph."""

import numpy as np
import torch
import torch_geometric
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD


class FeatureDimensionalityReduction(torch_geometric.transforms.BaseTransform):
    r"""A transform that reduces dimensionality of node features.
    Parameters
    ----------
    reduced_dim : int
        The retained number of components after dimensionality reduction.
    svd_iter : int
        The number of iterations for randomized SVD.
    svd_seed : int
        The field containing the node features.
    **kwargs : optional
        Additional arguments for the class.
    """
    def __init__(
        self,
        reduced_dim: int,
        svd_iter: int,
        svd_seed: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.type = "feature_dim_reduction"
        self.reduced_dim = reduced_dim
        self.svd_iter = svd_iter
        self.svd_seed = svd_seed
        self.svd = TruncatedSVD(
            n_components = self.reduced_dim,
            n_iter = self.svd_iter,
            random_state = self.svd_seed,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, reduced_dim={self.reduced_dim}, svd_iter={self.svd_iter}, svd_seed={self.svd_seed}), svd_red={self.svd}"

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
        if not hasattr(data, "x") or data.x is None:
            return data
        x_sparse = coo_matrix((data.x.coalesce().values().numpy(), data.x.coalesce().indices().numpy()), tuple(data.x.shape))
        x = self.svd.fit_transform(x_sparse).astype(np.float32)
        data.x = torch.from_numpy(x)
        return data
