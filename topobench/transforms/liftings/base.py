"""Abstract class for topological liftings."""

from abc import abstractmethod

import torch_geometric

from topobench.transforms.feature_liftings import FEATURE_LIFTINGS


class AbstractLifting(torch_geometric.transforms.BaseTransform):
    r"""Abstract class for topological liftings.

    Parameters
    ----------
    feature_lifting : str, optional
        The feature lifting method to be used. Default is 'ProjectionSum'.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, feature_lifting=None, **kwargs):
        super().__init__()
        self.feature_lifting = FEATURE_LIFTINGS[feature_lifting](**kwargs)
        self.neighborhoods = kwargs.get("neighborhoods")
        self.complex_dim = kwargs.get("complex_dim")

    @abstractmethod
    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a graph to higher-order topological domains.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        raise NotImplementedError

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Apply the full lifting (topology + features) to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The lifted data.
        """
        initial_data = data.to_dict()
        lifted_topology = self.lift_topology(data)
        lifted_topology = self.feature_lifting(lifted_topology)
        # Remove the original features to avoid conflict with the feature lifting
        if "x_0" in initial_data:
            del initial_data["x_0"]
        return torch_geometric.data.Data(**initial_data, **lifted_topology)
