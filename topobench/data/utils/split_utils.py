"""Split utilities."""

import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from topobench.dataloader import DataloadDataset


# Generate splits in different fasions
def k_fold_split(labels, parameters, root=None):
    """Return train and valid indices as in K-Fold Cross-Validation.

    If the split already exists it loads it automatically, otherwise it creates the
    split file for the subsequent runs.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameters.
    root : str, optional
        Root directory for data splits. Overwrite the default directory.

    Returns
    -------
    dict
        Dictionary containing the train, validation and test indices, with keys "train", "valid", and "test".
    """

    data_dir = (
        parameters["data_split_dir"]
        if root is None
        else os.path.join(root, "data_splits")
    )
    k = parameters.k
    fold = parameters.data_seed
    assert fold < k, "data_seed needs to be less than k"

    torch.manual_seed(0)
    np.random.seed(0)

    split_dir = os.path.join(data_dir, f"{k}-fold")

    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)

    split_path = os.path.join(split_dir, f"{fold}.npz")
    if not os.path.isfile(split_path):
        n = len(labels)
        x_idx = np.arange(n)
        x_idx = np.random.permutation(x_idx)
        labels = labels[x_idx]

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        for fold_n, (train_idx, valid_idx) in enumerate(
            skf.split(x_idx, labels)
        ):
            split_idx = {
                "train": train_idx,
                "valid": valid_idx,
                "test": valid_idx,
            }

            # Check that all nodes/graph have been assigned to some split
            assert np.all(
                np.sort(
                    np.array(
                        split_idx["train"].tolist()
                        + split_idx["valid"].tolist()
                    )
                )
                == np.sort(np.arange(len(labels)))
            ), "Not every sample has been loaded."
            split_path = os.path.join(split_dir, f"{fold_n}.npz")

            np.savez(split_path, **split_idx)

    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    assert np.unique(
        np.array(
            split_idx["train"].tolist()
            + split_idx["valid"].tolist()
            + split_idx["test"].tolist()
        )
    ).shape[0] == len(labels), "Not all nodes within splits"

    return split_idx


def random_splitting(labels, parameters, root=None, global_data_seed=42):
    r"""Randomly splits label into train/valid/test splits.

    Adapted from https://github.com/CUAI/Non-Homophily-Benchmarks.

    Parameters
    ----------
    labels : torch.Tensor
        Label tensor.
    parameters : DictConfig
        Configuration parameter.
    root : str, optional
        Root directory for data splits. Overwrite the default directory.
    global_data_seed : int
        Seed for the random number generator.

    Returns
    -------
    dict:
        Dictionary containing the train, validation and test indices with keys "train", "valid", and "test".
    """
    fold = (
        parameters["data_seed"] % 10
    )  # Ensure fold is between 0 and 9, TODO: Modify hardcoded 10 split number
    data_dir = (
        parameters["data_split_dir"]
        if root is None
        else os.path.join(root, "data_splits")
    )
    train_prop = parameters["train_prop"]
    valid_prop = (1 - train_prop) / 2

    # Create split directory if it does not exist
    split_dir = os.path.join(
        data_dir, f"train_prop={train_prop}_global_seed={global_data_seed}"
    )
    generate_splits = False
    if not os.path.isdir(split_dir):
        os.makedirs(split_dir)
        generate_splits = True

    # Generate splits if they do not exist
    if generate_splits:
        # Set initial seed
        torch.manual_seed(global_data_seed)
        np.random.seed(global_data_seed)
        # Generate a split
        n = len(labels)
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)

        # Generate 10 splits
        for fold_n in range(10):
            # Permute indices
            perm = torch.as_tensor(np.random.permutation(n))

            train_indices = perm[:train_num]
            val_indices = perm[train_num : train_num + valid_num]
            test_indices = perm[train_num + valid_num :]
            split_idx = {
                "train": train_indices,
                "valid": val_indices,
                "test": test_indices,
            }

            # Save generated split
            split_path = os.path.join(split_dir, f"{fold_n}.npz")
            np.savez(split_path, **split_idx)

    # Load the split
    split_path = os.path.join(split_dir, f"{fold}.npz")
    split_idx = np.load(split_path)

    # Check that all nodes/graph have been assigned to some split
    assert np.unique(
        np.array(
            split_idx["train"].tolist()
            + split_idx["valid"].tolist()
            + split_idx["test"].tolist()
        )
    ).shape[0] == len(labels), "Not all nodes within splits"

    return split_idx


def assign_train_val_test_mask_to_graphs(dataset, split_idx, use_lazy=False):
    """Split the graph dataset into train, validation, and test datasets.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Considered dataset.
    split_idx : dict
        Dictionary containing the train, validation, and test indices.
    use_lazy : bool, optional
        Use lazy subsets for O(1) memory (default: False for backward compatibility).
        When True, returns LazyDataloadDataset objects compatible with TBDataloader.

    Returns
    -------
    tuple:
        Tuple containing the train, validation, and test datasets.
        When use_lazy=True, returns LazyDataloadDataset objects.
        When use_lazy=False, returns DataloadDataset objects (in-memory).
    """
    if use_lazy:
        # Use lazy splits for O(1) memory usage with TBDataloader compatibility
        # Import here to avoid circular dependency
        from topobench.data.datasets import LazyDataloadDataset

        return (
            LazyDataloadDataset(dataset, split_idx["train"]),
            LazyDataloadDataset(dataset, split_idx["valid"]),
            LazyDataloadDataset(dataset, split_idx["test"]),
        )

    # Traditional approach: load all samples and assign masks
    data_train_lst, data_val_lst, data_test_lst = [], [], []

    # Assign masks directly by iterating over pre-split indices
    for i in split_idx["train"]:
        graph = dataset[i]
        graph.train_mask = torch.tensor([1], dtype=torch.long)
        graph.val_mask = torch.tensor([0], dtype=torch.long)
        graph.test_mask = torch.tensor([0], dtype=torch.long)
        data_train_lst.append(graph)

    for i in split_idx["valid"]:
        graph = dataset[i]
        graph.train_mask = torch.tensor([0], dtype=torch.long)
        graph.val_mask = torch.tensor([1], dtype=torch.long)
        graph.test_mask = torch.tensor([0], dtype=torch.long)
        data_val_lst.append(graph)

    for i in split_idx["test"]:
        graph = dataset[i]
        graph.train_mask = torch.tensor([0], dtype=torch.long)
        graph.val_mask = torch.tensor([0], dtype=torch.long)
        graph.test_mask = torch.tensor([1], dtype=torch.long)
        data_test_lst.append(graph)

    return (
        DataloadDataset(data_train_lst),
        DataloadDataset(data_val_lst),
        DataloadDataset(data_test_lst),
    )


def load_inductive_splits(dataset, parameters, use_lazy=False):
    r"""Load multiple-graph datasets with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.
    use_lazy : bool, optional
        Use lazy subsets for O(1) memory usage (default: False for backward compatibility).
        Recommended for large on-disk datasets.

    Returns
    -------
    list:
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert len(dataset) > 1, (
        "Datasets should have more than one graph in an inductive setting."
    )
    # Check if labels are ragged (different sizes across graphs)
    label_list = [data.y.squeeze(0).numpy() for data in dataset]
    label_shapes = [label.shape for label in label_list]
    # Use dtype=object only if labels have different shapes (ragged)
    labels = (
        np.array(label_list, dtype=object)
        if len(set(label_shapes)) > 1
        else np.array(label_list)
    )

    root = (
        dataset.dataset.get_data_dir()
        if hasattr(dataset.dataset, "get_data_dir")
        else None
    )

    if parameters.split_type == "random":
        split_idx = random_splitting(labels, parameters, root=root)

    elif parameters.split_type == "k-fold":
        assert type(labels) is not object, (
            "K-Fold splitting not supported for ragged labels."
        )
        split_idx = k_fold_split(labels, parameters, root=root)

    elif parameters.split_type == "fixed" and hasattr(dataset, "split_idx"):
        split_idx = dataset.split_idx

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random', 'k-fold' or 'fixed'.\
            If 'fixed' is chosen, the dataset should have the attribute split_idx"
        )

    train_dataset, val_dataset, test_dataset = (
        assign_train_val_test_mask_to_graphs(
            dataset, split_idx, use_lazy=use_lazy
        )
    )

    return train_dataset, val_dataset, test_dataset


def load_transductive_splits(preprocessor, split_config):
    r"""Load transductive splits with pre-batched loaders.

    This function creates TransductiveSplitDataset wrappers for transductive
    learning, which handle batching internally. Unlike inductive splits,
    transductive splits work on a single graph with train/val/test masks.

    Parameters
    ----------
    preprocessor : OnDiskTransductivePreprocessor
        Preprocessor with built index and graph data.
    split_config : DictConfig
        Configuration for batching strategy, including:
        - strategy : str ("structure_centric" or "extended_context")
        - Strategy-specific parameters (e.g., structures_per_batch, node_budget)

    Returns
    -------
    tuple
        A tuple containing (train_dataset, val_dataset, test_dataset).
        Each is a TransductiveSplitDataset instance.
    """
    # Import here to avoid circular dependency
    from topobench.data.datasets.transductive_split import (
        TransductiveSplitDataset,
    )

    # Build index if not already built
    if not preprocessor._index_built:
        print("Building structure index...")
        preprocessor.build_index()
        print(f"✓ Index built: {preprocessor.num_structures:,} structures")

    # Create train/val/test splits using masks from graph_data
    train_dataset = TransductiveSplitDataset(
        preprocessor=preprocessor,
        split_config=split_config,
        mask=preprocessor.graph_data.train_mask
        if hasattr(preprocessor.graph_data, "train_mask")
        else None,
        split_name="train",
    )

    val_dataset = TransductiveSplitDataset(
        preprocessor=preprocessor,
        split_config=split_config,
        mask=preprocessor.graph_data.val_mask
        if hasattr(preprocessor.graph_data, "val_mask")
        else None,
        split_name="val",
    )

    test_dataset = TransductiveSplitDataset(
        preprocessor=preprocessor,
        split_config=split_config,
        mask=preprocessor.graph_data.test_mask
        if hasattr(preprocessor.graph_data, "test_mask")
        else None,
        split_name="test",
    )

    return train_dataset, val_dataset, test_dataset


def load_coauthorship_hypergraph_splits(data, parameters, train_prop=0.5):
    r"""Load the split generated by rand_train_test_idx function.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.
    train_prop : float
        Proportion of training data.

    Returns
    -------
    torch_geometric.data.Data:
        Graph dataset with the specified split.
    """

    data_dir = os.path.join(
        parameters["data_split_dir"], f"train_prop={train_prop}"
    )
    load_path = f"{data_dir}/split_{parameters['data_seed']}.npz"
    splits = np.load(load_path, allow_pickle=True)

    # Upload masks
    data.train_mask = torch.from_numpy(splits["train"])
    data.val_mask = torch.from_numpy(splits["valid"])
    data.test_mask = torch.from_numpy(splits["test"])

    # Check that all nodes assigned to splits
    assert (
        torch.unique(
            torch.concat([data.train_mask, data.val_mask, data.test_mask])
        ).shape[0]
        == data.num_nodes
    ), "Not all nodes within splits"
    return DataloadDataset([data]), None, None
