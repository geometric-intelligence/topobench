"""On-disk preprocessor for large-scale datasets."""

import json
import os

import torch
import torch_geometric
from tqdm import tqdm

from topobench.data.utils import (
    ensure_serializable,
    load_inductive_splits,
    load_transductive_splits,
    make_hash,
)
from topobench.dataloader import DataloadDataset
from topobench.transforms.data_transform import DataTransform


class OnDiskPreProcessor(torch_geometric.data.OnDiskDataset):
    """On-disk preprocessor for large-scale datasets.

    This class processes datasets one sample at a time, saving each processed
    sample to disk immediately to avoid memory bottlenecks.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        The raw dataset to be processed.
    data_dir : str
        Path to the directory for saving processed data.
    transforms_config : DictConfig, optional
        Configuration for the transforms (default: None).
    **kwargs : optional
        Optional additional arguments.
    """

    def __init__(self, dataset, data_dir, transforms_config=None, **kwargs):
        self.dataset = dataset
        if transforms_config is not None:
            self.transforms_applied = True
            pre_transform = self.instantiate_pre_transform(
                data_dir, transforms_config
            )
            super().__init__(self.processed_data_dir, **kwargs)
            # Store pre_transform after parent init since OnDiskDataset may overwrite it
            self.pre_transform = pre_transform
            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.save_transform_parameters()
            self.data_list = [data for data in self]
        else:
            self.transforms_applied = False
            super().__init__(data_dir, **kwargs)
            self.pre_transform = None
            self.transform = (
                dataset.transform if hasattr(dataset, "transform") else None
            )
            self.data_list = [data for data in dataset]

        # Some datasets have fixed splits, and those are stored as split_idx during loading
        # We need to store this information to be able to reproduce the splits afterwards
        if hasattr(dataset, "split_idx"):
            self.split_idx = dataset.split_idx

    @property
    def processed_dir(self) -> str:
        """Return the path to the processed directory.

        Returns
        -------
        str
            Path to the processed directory.
        """
        if not self.transforms_applied:
            return self.root
        else:
            return self.root + "/processed"

    @property
    def processed_file_names(self) -> list:
        """Return list of processed file names.

        Returns
        -------
        list
            List of processed file names.
        """
        num_samples = len(self.dataset)
        return [f"data_{i}.pt" for i in range(num_samples)]

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
        if transforms_config.keys() == {"liftings"}:
            transforms_config = transforms_config.liftings

        # Check configuration structure
        if "transform_name" in transforms_config:
            # Single transform configuration
            pre_transforms_dict = {
                transforms_config.transform_name: DataTransform(
                    **transforms_config
                )
            }
        else:
            # Multiple transforms configuration
            pre_transforms_dict = {
                key: DataTransform(**value)
                for key, value in transforms_config.items()
            }

        pre_transforms = torch_geometric.transforms.Compose(
            list(pre_transforms_dict.values())
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
        repo_name = "_".join(list(transforms_config.keys()))
        transforms_parameters = {
            transform_name: transform.parameters
            for transform_name, transform in pre_transforms_dict.items()
        }
        params_hash = make_hash(transforms_parameters)
        self.transforms_parameters = ensure_serializable(transforms_parameters)
        self.processed_data_dir = os.path.join(
            data_dir, repo_name, f"{params_hash}"
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
        """Method that processes the dataset one sample at a time, saving to disk."""
        print(f"Processing {len(self.dataset)} samples to disk...")

        for idx in tqdm(range(len(self.dataset)), desc="Processing samples"):
            processed_path = os.path.join(self.processed_dir, f"data_{idx}.pt")

            if os.path.exists(processed_path):
                continue

            data = self.dataset[idx]

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, processed_path)

        print(f"Processing complete. Saved to {self.processed_dir}")

    def len(self) -> int:
        """Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.dataset)

    def get(self, idx: int) -> torch_geometric.data.Data:
        """Load and return a single processed sample from disk.

        Parameters
        ----------
        idx : int
            Index of the sample to load.

        Returns
        -------
        torch_geometric.data.Data
            The loaded processed sample.
        """
        processed_path = os.path.join(self.processed_dir, f"data_{idx}.pt")
        data = torch.load(processed_path)

        # Apply runtime transforms
        if self.transform is not None:
            data = self.transform(data)

        return data

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
