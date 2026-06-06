"""Preprocessor for datasets."""

import json
import os
import time

import torch
import torch_geometric
from filelock import FileLock
from torch_geometric.io import fs
from tqdm import tqdm

from topobench.data.utils import (
    ensure_serializable,
    load_inductive_splits,
    load_transductive_splits,
    make_hash,
)
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


class PreProcessor(torch_geometric.data.InMemoryDataset):
    """Preprocessor for datasets.

    Parameters
    ----------
    dataset : list
        List of data objects.
    data_dir : str
        Path to the directory containing the data.
    transforms_config : DictConfig, optional
        Configuration parameters for the transforms (default: None).
    **kwargs : optional
        Optional additional arguments.
    """

    def __init__(self, dataset, data_dir, transforms_config=None, **kwargs):
        self.dataset = dataset
        self.preprocessing_time = 0
        if transforms_config is not None:
            self.transforms_applied = True
            pre_transform = self.instantiate_pre_transform(
                data_dir, transforms_config
            )

            # 1. Ensure the target directory exists so we can place a lock file in it
            os.makedirs(self.processed_data_dir, exist_ok=True)
            lock_path = os.path.join(
                self.processed_data_dir, "preprocessing.lock"
            )

            start_time = time.time()

            with FileLock(lock_path):
                # When Process 1 finishes, Process 2 checks, sees data.pt, and skips.
                super().__init__(
                    self.processed_data_dir, None, pre_transform, **kwargs
                )
                self.save_transform_parameters()

            end_time = time.time()
            self.preprocessing_time = end_time - start_time

            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.load(self.processed_paths[0])
            self.data_list = [data for data in self]
        else:
            self.transforms_applied = False
            super().__init__(data_dir, None, None, **kwargs)
            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.data, self.slices = dataset._data, dataset.slices
            self.data_list = [data for data in dataset]

        # Some datasets have fixed splits, and those are stored as split_idx during loading
        # We need to store this information to be able to reproduce the splits afterwards
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx
        if hasattr(dataset, "split_idx_list"):
            self.split_idx_list = dataset.split_idx_list

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        return self.root

    @property
    def processed_file_names(self) -> str:
        """Return the name of the processed file.

        Returns
        -------
        str
            Name of the processed file.
        """
        return "data.pt"

    def instantiate_pre_transform(
        self, data_dir, transforms_config
    ) -> torch_geometric.transforms.Compose:
        """Instantiate the pre-transforms.

        Parameters
        ----------
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.

        Returns
        -------
        torch_geometric.transforms.Compose
            Pre-transform object.
        """
        from torch_geometric.transforms import ToDevice

        if transforms_config.keys() == {"liftings"}:
            transforms_config = transforms_config.liftings

        if "transform_name" in transforms_config:
            config_items = [
                (transforms_config.transform_name, transforms_config)
            ]
        else:
            config_items = transforms_config.items()

        pre_transforms_list = []
        pre_transforms_dict = {}

        # Track where the graph currently lives in the pipeline
        current_device = "cpu"

        for key, value in config_items:
            kwargs = dict(value)

            requested_device = kwargs.pop("preprocessor_device", "cpu")

            target_device = (
                "cuda"
                if requested_device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )

            transform_instance = DataTransform(**kwargs)
            pre_transforms_dict[key] = transform_instance

            if target_device != current_device:
                pre_transforms_list.append(ToDevice(target_device))
                current_device = target_device

            pre_transforms_list.append(transform_instance)

        # If the pipeline ends while the graph is still on the GPU,
        # we MUST pull it back to the CPU before PyTorch Geometric saves it to disk.
        if current_device == "cuda":
            pre_transforms_list.append(ToDevice("cpu"))

        pre_transforms = torch_geometric.transforms.Compose(
            pre_transforms_list
        )

        self.set_processed_data_dir(
            pre_transforms_dict, data_dir, transforms_config
        )
        return pre_transforms

    def set_processed_data_dir(
        self, pre_transforms_dict, data_dir, transforms_config
    ) -> None:
        """Set the processed data directory.

        Parameters
        ----------
        pre_transforms_dict : dict
            Dictionary containing the pre-transforms.
        data_dir : str
            Path to the directory containing the data.
        transforms_config : DictConfig
            Configuration parameters for the transforms.
        """
        # Use self.transform_parameters to define unique save/load path for each transform parameters
        repo_name = "_".join(list(transforms_config.keys()))
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(
            *[data_dir, repo_name, f"{params_hash}"]
        )

    def save_transform_parameters(self) -> None:
        """Save the transform parameters."""
        # Check if root/params_dict.json exists, if not, save it
        path_transform_parameters = os.path.join(
            self.processed_data_dir, "path_transform_parameters_dict.json"
        )
        if not os.path.exists(path_transform_parameters):
            with open(path_transform_parameters, "w") as f:
                json.dump(self.transforms_parameters, f, indent=4)
        else:
            # If path_transform_parameters exists, check if the transform_parameters are the same
            with open(path_transform_parameters) as f:
                saved_transform_parameters = json.load(f)

            if saved_transform_parameters != self.transforms_parameters:
                raise ValueError(
                    "Different transform parameters for the same data_dir"
                )

            print(
                f"Transform parameters are the same, using existing data_dir: {self.processed_data_dir}"
            )

    def process(self) -> None:
        """Method that processes the data."""
        if isinstance(
            self.dataset,
            (torch_geometric.data.Dataset, torch.utils.data.Dataset),
        ):
            data_list = [data for data in self.dataset]
        elif isinstance(self.dataset, torch_geometric.data.Data):
            data_list = [self.dataset]

        if self.pre_transform is not None:
            print(f"\nApplying transforms to {len(data_list)} graphs...")
            self.data_list = [
                self.pre_transform(d)
                for d in tqdm(
                    data_list, desc="Processing graphs", unit="graph"
                )
            ]
        else:
            self.data_list = data_list

        self._data, self.slices = self.collate(self.data_list)
        self._data_list = None  # Reset cache.

        assert isinstance(self._data, torch_geometric.data.Data)
        self.save(self.data_list, self.processed_paths[0])

    def load(self, path: str) -> None:
        r"""Load the dataset from the file path `path`.

        Parameters
        ----------
        path : str
            The path to the processed data.
        """
        out = fs.torch_load(path)
        assert isinstance(out, tuple)
        assert len(out) >= 2 and len(out) <= 4
        if len(out) == 2:  # Backward compatibility (1).
            data, self.slices = out
        elif len(out) == 3:  # Backward compatibility (2).
            data, self.slices, data_cls = out
        else:  # TU Datasets store additional element (__class__) in the processed file
            data, self.slices, sizes, data_cls = out

        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            self.data = data_cls.from_dict(data)

    def load_dataset_splits(
        self, split_params
    ) -> tuple[
        DataloadDataset, DataloadDataset | None, DataloadDataset | None
    ]:
        """Load the dataset splits.

        Parameters
        ----------
        split_params : dict
            Parameters for loading the dataset splits.

        Returns
        -------
        tuple
            A tuple containing the train, validation, and test datasets.
        """
        if not split_params.get("learning_setting", False):
            raise ValueError("No learning setting specified in split_params")

        if split_params.learning_setting == "inductive":
            return load_inductive_splits(self, split_params)
        elif split_params.learning_setting == "transductive":
            return load_transductive_splits(self, split_params)
        else:
            raise ValueError(
                f"Invalid '{split_params.learning_setting}' learning setting.\
                Please define either 'inductive' or 'transductive'."
            )
