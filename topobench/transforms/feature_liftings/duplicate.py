"""Duplicate input data and possible apply a static transform."""

import torch
import torch_geometric


class Duplicate(torch_geometric.transforms.BaseTransform):
    r"""A static copy of input data that also applies a set transform.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "domain2domain"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def lift_features(
        self, data: torch_geometric.data.Data | dict
    ) -> torch_geometric.data.Data | dict:
        r"""Duplicate r-cell features of a graph to r+1-cell structures and apply an optional transform.

        Parameters
        ----------
        data : torch_geometric.data.Data | dict
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data | dict
            The data with the lifted features.
        """

        keys = sorted(
            [
                key.split("_")[1]
                for key in data
                if "incidence" in key and "-" not in key
            ]
        )

        for elem in keys:
            if f"x_{elem}" not in data:
                first_dim = data[f"incidence_{elem}"].size()[1]
                data["x_" + elem] = data["x"] if "x" in data else data["x_0"]
                # if "all_ones" in self.parameters:
                #     data["x_" + elem] = torch.ones(
                #         (first_dim, data["x_" + elem].size()[1])
                #     )
                if "all_zeros" in self.parameters:
                    data["x_" + elem] = torch.zeros_like(data["x_" + elem])
                elif "absolute_value" in self.parameters:
                    data["x_" + elem] = torch.abs(data["x_" + elem])
                else:
                    # By default "all_ones" is used
                    data["x_" + elem] = torch.ones(
                        (first_dim, data["x_" + elem].size()[1])
                    )
                    # raise Exception(
                    #     "Invalid parameter for the Duplicate transform."
                    # )
        return data

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The same data.
        """
        data = self.lift_features(data)
        return data
