"""This module contains a transform that generates equal Gaussian features for all nodes in the input graph."""

import torch
import torch_geometric


class EqualGausFeatures(torch_geometric.transforms.BaseTransform):
    r"""A transform that generates equal Gaussian features for all nodes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class. It should contain the following keys:
        - mean (float): The mean of the Gaussian distribution.
        - std (float): The standard deviation of the Gaussian distribution.
        - num_features (int): The number of features to generate.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "generate_non_informative_features"

        # Torch generate feature vector from Gaussian distribution
        self.mean = kwargs["mean"]
        self.std = kwargs["std"]
        self.feature_vector = kwargs["num_features"]
        self.feature_vector = torch.normal(
            mean=self.mean, std=self.std, size=(1, self.feature_vector)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, mean={self.mean!r}, std={self.std!r}, feature_vector={self.feature_vector!r})"

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
        data.x = self.feature_vector.expand(data.num_nodes, -1)
        return data
