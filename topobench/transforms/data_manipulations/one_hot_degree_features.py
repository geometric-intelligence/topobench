"""One hot degree features transform."""

import torch
import torch_geometric
from torch_geometric.utils import one_hot


class OneHotDegreeFeatures(torch_geometric.transforms.BaseTransform):
    r"""Class for one hot degree features transform.

    A transform that adds the node degree as one hot encodings to the node features.

    Parameters
    ----------
    max_degree : int
        The maximum degree of the graph.
    degrees_field : str
        The field containing the node degrees.
    features_field : str
        The field containing the node features.
    cat : bool, optional
        If set to `True`, the one hot encodings are concatenated to the node
        features (default: False).
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        max_degree: int,
        degrees_field: str,
        features_field: str,
        cat: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.type = "one_hot_degree_features"
        self.max_degree = max_degree
        self.degrees_field = degrees_field
        self.features_field = features_field
        self.cat = cat
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, max_degree={self.max_degree}, degrees_field={self.deg_field!r}, features_field={self.features_fields!r})"

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

        deg = data[self.degrees_field].to(torch.long)

        if len(deg.shape) == 2:
            deg = deg.squeeze(1)

        deg = one_hot(deg, num_classes=self.max_degree + 1)

        if self.cat:
            x = data[self.features_field]
            x = x.view(-1, 1) if x.dim() == 1 else x
            data[self.features_field] = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data[self.features_field] = deg

        if not self.kwargs["keep_degree_field"]:
            delattr(data, self.degrees_field)  # Remove the old deg attribute

        return data
