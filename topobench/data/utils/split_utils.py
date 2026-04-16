"""Split utilities."""

import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from topobench.dataloader import DataloadDataset


def _generate_or_load_cached_splits(split_dir, fold, generator):
    """Load split for ``fold`` from ``split_dir``, generating if missing.

    If ``{split_dir}/{fold}.npz`` is missing, calls ``generator()`` (which
    must return a sequence of split dicts, one per fold) and writes each
    to ``{fold_n}.npz`` before loading.

    Parameters
    ----------
    split_dir : str
        Directory holding cached split .npz files.
    fold : int
        Which fold to load.
    generator : callable
        Returns a sequence of split dicts with keys 'train', 'valid', 'test'.

    Returns
    -------
    NpzFile
        Loaded split with keys 'train', 'valid', 'test'.
    """
    split_path = os.path.join(split_dir, f"{fold}.npz")
    if not os.path.isfile(split_path):
        if not os.path.isdir(split_dir):
            os.makedirs(split_dir)
        for fold_n, split in enumerate(generator()):
            np.savez(os.path.join(split_dir, f"{fold_n}.npz"), **split)
    return np.load(split_path)


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

    def generate():
        """Generate all `k` stratified k-fold splits.

        Returns
        -------
        list[dict]
            List of split dicts, one per fold.
        """
        n = len(labels)
        x_idx = np.random.permutation(np.arange(n))
        labels_perm = labels[x_idx]
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        out = []
        for train_idx, valid_idx in skf.split(x_idx, labels_perm):
            assert np.all(
                np.sort(np.array(train_idx.tolist() + valid_idx.tolist()))
                == np.sort(np.arange(len(labels)))
            ), "Not every sample has been loaded."
            out.append(
                {"train": train_idx, "valid": valid_idx, "test": valid_idx}
            )
        return out

    split_idx = _generate_or_load_cached_splits(split_dir, fold, generate)

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

    split_dir = os.path.join(
        data_dir, f"train_prop={train_prop}_global_seed={global_data_seed}"
    )

    def generate():
        """Generate 10 random train/valid/test splits.

        Returns
        -------
        list[dict]
            List of 10 split dicts.
        """
        torch.manual_seed(global_data_seed)
        np.random.seed(global_data_seed)
        n = len(labels)
        train_num = int(n * train_prop)
        valid_num = int(n * valid_prop)
        out = []
        for _ in range(10):
            perm = torch.as_tensor(np.random.permutation(n))
            out.append(
                {
                    "train": perm[:train_num],
                    "valid": perm[train_num : train_num + valid_num],
                    "test": perm[train_num + valid_num :],
                }
            )
        return out

    split_idx = _generate_or_load_cached_splits(split_dir, fold, generate)

    # Check that all nodes/graph have been assigned to some split
    assert np.unique(
        np.array(
            split_idx["train"].tolist()
            + split_idx["valid"].tolist()
            + split_idx["test"].tolist()
        )
    ).shape[0] == len(labels), "Not all nodes within splits"

    return split_idx


def assign_train_val_test_mask_to_graphs(dataset, split_idx):
    """Split the graph dataset into train, validation, and test datasets.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Considered dataset.
    split_idx : dict
        Dictionary containing the train, validation, and test indices.

    Returns
    -------
    tuple:
        Tuple containing the train, validation, and test datasets.
    """

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


def load_transductive_splits(dataset, parameters):
    r"""Load the graph dataset with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

    Returns
    -------
    list:
        List containing the train, validation, and test splits.
    """
    # Extract labels from dataset object
    assert len(dataset) == 1, (
        "Dataset should have only one graph in a transductive setting."
    )

    data = dataset.data_list[0]
    labels = data.y.numpy()

    # Ensure labels are one dimensional array
    assert len(labels.shape) == 1, "Labels should be one dimensional array"

    root = (
        dataset.dataset.get_data_dir()
        if hasattr(dataset.dataset, "get_data_dir")
        else None
    )

    if parameters.split_type == "random":
        splits = random_splitting(labels, parameters, root=root)

    elif parameters.split_type == "k-fold":
        splits = k_fold_split(labels, parameters, root=root)

    else:
        raise NotImplementedError(
            f"split_type {parameters.split_type} not valid. Choose either 'random' or 'k-fold'"
        )

    # Assign train val test masks to the graph
    data.train_mask = torch.from_numpy(splits["train"])
    data.val_mask = torch.from_numpy(splits["valid"])
    data.test_mask = torch.from_numpy(splits["test"])

    if parameters.get("standardize", False):
        # Standardize the node features respecting train mask
        data.x = (data.x - data.x[data.train_mask].mean(0)) / data.x[
            data.train_mask
        ].std(0)
        data.y = (data.y - data.y[data.train_mask].mean(0)) / data.y[
            data.train_mask
        ].std(0)

    return DataloadDataset([data]), None, None


def load_inductive_splits(dataset, parameters):
    r"""Load multiple-graph datasets with the specified split.

    Parameters
    ----------
    dataset : torch_geometric.data.Dataset
        Graph dataset.
    parameters : DictConfig
        Configuration parameters.

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
        assign_train_val_test_mask_to_graphs(dataset, split_idx)
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
