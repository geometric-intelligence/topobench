"""Dataloader utilities."""

from collections import defaultdict
from typing import Any

import torch
import torch_geometric
from torch_sparse import SparseTensor


class DomainData(torch_geometric.data.Data):
    r"""Helper Data class so that not only sparse matrices with adj in the name can work with PyG dataloaders.

    It overwrites some methods from `torch_geometric.data.Data`
    """

    def is_valid(self, string):
        r"""Check if the string contains any of the valid names.

        Parameters
        ----------
        string : str
            String to check.

        Returns
        -------
        bool
            Whether the string contains any of the valid names.
        """
        valid_names = ["adj", "incidence", "laplacian"]
        return any(name in string for name in valid_names)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        r"""Overwrite the `__cat_dim__` method to handle sparse matrices to handle the names specified in `is_valid`.

        Parameters
        ----------
        key : str
            Key of the data.
        value : Any
            Value of the data.
        *args : Any
            Additional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The concatenation dimension.
        """
        if torch_geometric.utils.is_sparse(value) and self.is_valid(key):
            return (0, 1)
        elif "index" in key or key == "face":
            return -1
        else:
            return 0


def to_data_list(batch):
    """Workaround needed since `torch_geometric` doesn't work when using `torch.sparse` instead of `torch_sparse`.

    Parameters
    ----------
    batch : torch_geometric.data.Batch
        The batch of data.

    Returns
    -------
    list
        List of data objects.
    """
    for key, _ in batch:
        if batch[key].is_sparse:
            sparse_data = batch[key].coalesce()
            batch[key] = SparseTensor.from_torch_sparse_coo_tensor(sparse_data)
    data_list = batch.to_data_list()
    for i, data in enumerate(data_list):
        for key, d in data:
            if isinstance(data[key], SparseTensor):
                data_list[i][key] = d.to_torch_sparse_coo_tensor()
    return data_list


def collate_fn(batch):
    r"""Overwrite `torch_geometric.data.DataLoader` collate function to use the `DomainData` class.

    This ensures that the `torch_geometric` dataloaders work with sparse matrices that are not necessarily named `adj`. The function also generates the batch slices for the different cell dimensions.

    Parameters
    ----------
    batch : list
        List of data objects (e.g., `torch_geometric.data.Data`).

    Returns
    -------
    torch_geometric.data.Batch
        A `torch_geometric.data.Batch` object.
    """
    data_list = []
    batch_idx_dict = defaultdict(list)

    # Keep track of the running index for each cell dimension
    running_idx = {}

    for batch_idx, b in enumerate(batch):
        values, keys = b[0], b[1]
        data = DomainData()
        for key, value in zip(keys, values, strict=False):
            if torch_geometric.utils.is_sparse(value):
                value = value.coalesce()
            data[key] = value

        # Generate batch_slice values for x_1, x_2, x_3, ...
        x_keys = [el for el in keys if ("x_" in el)]
        for x_key in x_keys:
            # if x_key != "x_0":
            if x_key != "x_hyperedges":
                cell_dim = int(x_key.split("_")[1])
            else:
                cell_dim = x_key.split("_")[1]

            current_number_of_cells = data[x_key].shape[0]

            batch_idx_dict[f"batch_{cell_dim}"].append(
                torch.tensor([[batch_idx] * current_number_of_cells])
            )

            if running_idx.get(f"cell_running_idx_number_{cell_dim}") is None:
                running_idx[f"cell_running_idx_number_{cell_dim}"] = (
                    current_number_of_cells
                )

            else:
                running_idx[f"cell_running_idx_number_{cell_dim}"] += (
                    current_number_of_cells
                )

        data_list.append(data)

    batch = torch_geometric.data.Batch.from_data_list(data_list)

    # Add batch slices to batch
    for key, value in batch_idx_dict.items():
        batch[key] = torch.cat(value, dim=1).squeeze(0).long()

    # Rename batch.batch to batch.batch_0 for consistency
    if (batch.get("batch") is not None) and (batch.get("batch_0") is not None):
        # Back compatiility check
        assert torch.all(batch["batch_0"].cpu() == batch.get("batch").cpu()), (
            "batch['batch_0'] and batch['batch'] should match in the number of nodes"
        )
        batch.pop("batch")

    if (batch.get("batch") is not None) and (batch.get("batch_0") is None):
        # Back compatiility check
        # Instead of batch.batch as in the original code, we use batch.batch_0 always to refer to nodes
        batch["batch_0"] = batch.pop("batch")

    # Ensure shape is torch.Tensor
    # "shape" describes the number of n_cells in each graph
    if batch.get("shape") is not None:
        cell_statistics = batch.pop("shape")
        batch["cell_statistics"] = torch.Tensor(cell_statistics).long()

    return batch
