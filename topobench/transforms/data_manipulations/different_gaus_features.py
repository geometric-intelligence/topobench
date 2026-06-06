"""This module contains a transform that generates equal Gaussian features for all nodes in the input graph."""

import torch
import torch_geometric


class DifferentGausFeatures(torch_geometric.transforms.BaseTransform):
    r"""A transform that generates different Gaussian features for all nodes.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class. It should contain the following keys:
        - mean (float): The mean of the Gaussian distribution.
        - std (float): The standard deviation of the Gaussian distribution.
        - num_features (int): The number of features to generate, defaults to -1 where the intial feature vector shape is taken.
        - dimensions (list): The dimension numbers to generate features for.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "generate_non_informative_features"

        # Torch generate feature vector from gaus distribution
        self.mean = kwargs["mean"]
        self.std = kwargs["std"]
        self.feature_vector_size = kwargs.get("num_features", -1)
        self.dimensions = kwargs["dimensions"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r},dimensions={list(self.dimensions)} mean={self.mean!r}, std={self.std!r}, feature_vector={self.feature_vector_size!r})"

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
        for dim in self.dimensions:
            data[f"x_{dim}"] = torch.normal(
                mean=self.mean,
                std=self.std,
                size=(
                    data[f"incidence_{dim}"].size()[1],
                    self.feature_vector_size,
                ),
            )
        return data
