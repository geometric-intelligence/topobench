"""RenameFields transform that does nothing to the input data."""

import warnings

import torch_geometric


class RenameFields(torch_geometric.transforms.BaseTransform):
    r"""A transform that renames specified fields in a `torch_geometric.data.Data` object.

    Parameters
    ----------
    init_field_name : list of str
        List of original field names to be renamed.
    new_field_name : list of str
        List of new field names corresponding to `init_field_name`.
    **kwargs : dict, optional
        Additional keyword arguments stored on the transform as
        ``self.parameters``.
    """

    def __init__(self, init_field_name, new_field_name, **kwargs):
        super().__init__()
        assert len(init_field_name) == len(new_field_name), (
            "init_field_name and new_field_name must have the same length."
        )
        self.type = "data manipulation"
        self.parameters = kwargs
        self.init_field_name = init_field_name
        self.new_field_name = new_field_name

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(type={self.type!r}, "
            f"init_field_name={self.init_field_name!r}, "
            f"new_field_name={self.new_field_name!r}, "
            f"parameters={self.parameters!r})"
        )

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to rename fields in the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The modified data with renamed fields.
        """
        for old_name, new_name in zip(
            self.init_field_name, self.new_field_name, strict=False
        ):
            if hasattr(data, old_name):
                if hasattr(data, new_name):
                    warnings.warn(
                        f"Attribute '{new_name}' already exists and will be overwritten.",
                        UserWarning,
                        stacklevel=2,
                    )
                    delattr(
                        data, new_name
                    )  # Delete the existing attribute before renaming

                setattr(data, new_name, getattr(data, old_name))
                delattr(data, old_name)  # Remove the old attribute

        return data
