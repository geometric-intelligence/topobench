"""A transform that keeps only particular target field indices."""

import torch_geometric


class KeepSelectedTargetIndices(torch_geometric.transforms.BaseTransform):
    r"""A transform that keeps only the selected fields of the input data.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "keep_selected_target_indices"
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
        data.y = data.y[:, self.parameters["target_indices"]].squeeze(0)

        return data
