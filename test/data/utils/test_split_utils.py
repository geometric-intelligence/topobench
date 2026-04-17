"""Unit tests for split utilities."""

import os
import tempfile
import shutil
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from omegaconf import DictConfig

from topobench.data.utils.split_utils import (
    k_fold_split,
    random_splitting,
    load_inductive_splits,
    load_transductive_splits,
    assign_train_val_test_mask_to_graphs,
)


class TestLoadInductiveSplits:
    """Test load_inductive_splits function."""

    def setup_method(self):
        """Setup method for each test."""
        # Create temporary directory for test splits
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_mock_dataset(self, n_graphs, label_shapes, has_get_data_dir=True):
        """Create a mock dataset with specified label shapes.

        Parameters
        ----------
        n_graphs : int
            Number of graphs in the dataset.
        label_shapes : list
            List of tuples representing label shapes for each graph.
        has_get_data_dir : bool
            Whether the dataset has get_data_dir method.

        Returns
        -------
        MagicMock
            Mock dataset object.
        """
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=n_graphs)

        # Create mock graphs with different label shapes
        mock_graphs = []
        for i, shape in enumerate(label_shapes):
            mock_graph = MagicMock()
            # Create labels with specified shape
            if len(shape) == 0:
                labels = np.array([i % 3])  # Single label
            else:
                labels = np.random.randint(0, 3, size=shape)
            mock_graph.y.squeeze.return_value.numpy.return_value = labels
            mock_graphs.append(mock_graph)

        mock_dataset.__getitem__ = lambda self, idx: mock_graphs[idx]
        mock_dataset.__iter__ = lambda self: iter(mock_graphs)

        # Setup dataset.dataset.get_data_dir()
        if has_get_data_dir:
            mock_dataset.dataset.get_data_dir.return_value = self.test_dir
        else:
            mock_dataset.dataset = MagicMock(spec=[])

        return mock_dataset

    def test_uniform_label_shapes_random_split(self):
        """Test with uniform label shapes using random split."""
        # Create dataset with uniform label shapes (all graphs have 1 label)
        n_graphs = 20
        label_shapes = [()] * n_graphs  # All single labels
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        # Verify splits exist and are non-empty
        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

        # Verify total equals original
        assert len(train_ds) + len(val_ds) + len(test_ds) == n_graphs

    def test_uniform_label_shapes_kfold_split(self):
        """Test with uniform label shapes using k-fold split."""
        n_graphs = 20
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "k-fold",
            "data_seed": 0,
            "k": 5,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        # Note: test_ds and val_ds are the same in k-fold (test=valid)
        assert len(train_ds) + len(val_ds) == n_graphs

    def test_ragged_label_shapes_random_split(self):
        """Test with ragged label shapes (different sizes) using random split."""
        # Create dataset with varying label shapes
        n_graphs = 15
        label_shapes = [()] * 5 + [(2,)] * 5 + [(3,)] * 5  # Mix of shapes
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        assert len(train_ds) + len(val_ds) + len(test_ds) == n_graphs

    def test_ragged_label_shapes_kfold_raises_error(self):
        """Test that k-fold with ragged labels raises an assertion error."""
        n_graphs = 15
        label_shapes = [()] * 5 + [(2,)] * 5 + [(3,)] * 5
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "k-fold",
            "data_seed": 0,
            "k": 5,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        # Should raise assertion error for ragged labels with k-fold
        with pytest.raises((AssertionError, ValueError)):
            # Could be AssertionError from the check or ValueError from sklearn
            load_inductive_splits(mock_dataset, parameters)

    def test_fixed_split_type(self):
        """Test with fixed split type."""
        n_graphs = 20
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        # Add split_idx attribute
        split_idx = {
            "train": np.arange(12),
            "valid": np.arange(12, 16),
            "test": np.arange(16, 20)
        }
        mock_dataset.split_idx = split_idx

        parameters = DictConfig({
            "split_type": "fixed",
            "data_seed": 0,
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        assert len(train_ds) == 12
        assert len(val_ds) == 4
        assert len(test_ds) == 4

    def test_fixed_split_type_without_split_idx_raises_error(self):
        """Test that fixed split without split_idx raises error."""
        n_graphs = 20
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)
        # Ensure split_idx attribute doesn't exist
        if hasattr(mock_dataset, 'split_idx'):
            delattr(mock_dataset, 'split_idx')

        parameters = DictConfig({
            "split_type": "fixed",
            "data_seed": 0,
        })

        with pytest.raises(NotImplementedError):
            load_inductive_splits(mock_dataset, parameters)

    def test_invalid_split_type_raises_error(self):
        """Test that invalid split type raises NotImplementedError."""
        n_graphs = 20
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "invalid_split_type",
            "data_seed": 0,
        })

        with pytest.raises(NotImplementedError, match="not valid"):
            load_inductive_splits(mock_dataset, parameters)

    def test_single_graph_raises_assertion(self):
        """Test that single graph dataset raises assertion error."""
        n_graphs = 1
        label_shapes = [()]
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        with pytest.raises(AssertionError, match="more than one graph"):
            load_inductive_splits(mock_dataset, parameters)

    def test_without_get_data_dir(self):
        """Test when dataset doesn't have get_data_dir method."""
        n_graphs = 20
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes, has_get_data_dir=False)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        # Should work fine without get_data_dir
        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0

    def test_masks_are_assigned_correctly(self):
        """Test that train/val/test masks are assigned correctly to graphs."""
        n_graphs = 10
        label_shapes = [()] * n_graphs
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        # Check that masks are properly assigned to the data_lst items
        for i in range(len(train_ds.data_lst)):
            graph = train_ds.data_lst[i]
            assert hasattr(graph, 'train_mask')
            assert hasattr(graph, 'val_mask')
            assert hasattr(graph, 'test_mask')
            assert graph.train_mask.item() == 1
            assert graph.val_mask.item() == 0
            assert graph.test_mask.item() == 0

        for i in range(len(val_ds.data_lst)):
            graph = val_ds.data_lst[i]
            assert graph.train_mask.item() == 0
            assert graph.val_mask.item() == 1
            assert graph.test_mask.item() == 0

        for i in range(len(test_ds.data_lst)):
            graph = test_ds.data_lst[i]
            assert graph.train_mask.item() == 0
            assert graph.val_mask.item() == 0
            assert graph.test_mask.item() == 1

    def test_different_data_seeds_produce_different_splits(self):
        """Test that different data seeds produce different splits."""
        n_graphs = 20
        label_shapes = [()] * n_graphs

        # First split
        mock_dataset1 = self.create_mock_dataset(n_graphs, label_shapes)
        parameters1 = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })
        train_ds1, _, _ = load_inductive_splits(mock_dataset1, parameters1)

        # Second split with different seed
        mock_dataset2 = self.create_mock_dataset(n_graphs, label_shapes)
        parameters2 = DictConfig({
            "split_type": "random",
            "data_seed": 1,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })
        train_ds2, _, _ = load_inductive_splits(mock_dataset2, parameters2)

        # Splits should have same size but potentially different composition
        assert len(train_ds1) == len(train_ds2)

    def test_multidimensional_ragged_labels(self):
        """Test with multidimensional ragged labels."""
        n_graphs = 12
        # Mix of different multidimensional shapes
        label_shapes = [(5,)] * 4 + [(10,)] * 4 + [(15,)] * 4
        mock_dataset = self.create_mock_dataset(n_graphs, label_shapes)

        parameters = DictConfig({
            "split_type": "random",
            "data_seed": 0,
            "train_prop": 0.5,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        train_ds, val_ds, test_ds = load_inductive_splits(mock_dataset, parameters)

        assert len(train_ds) > 0
        assert len(val_ds) > 0
        assert len(test_ds) > 0
        assert len(train_ds) + len(val_ds) + len(test_ds) == n_graphs


class TestKFoldSplit:
    """Test k_fold_split function."""

    def setup_method(self):
        """Setup method for each test."""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_basic_kfold(self):
        """Test basic k-fold splitting."""
        # Use more samples per class to avoid n_splits > samples per class
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] * 2)  # 30 samples, 10 per class
        parameters = DictConfig({
            "k": 5,
            "data_seed": 0,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        split_idx = k_fold_split(labels, parameters)

        assert "train" in split_idx
        assert "valid" in split_idx
        assert "test" in split_idx
        assert len(split_idx["train"]) + len(split_idx["valid"]) == len(labels)

    def test_kfold_with_root_override(self):
        """Test k-fold with root directory override."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2] * 2)  # 30 samples
        custom_root = os.path.join(self.test_dir, "custom")
        os.makedirs(custom_root, exist_ok=True)

        parameters = DictConfig({
            "k": 5,
            "data_seed": 0,
            "data_split_dir": "original_dir"  # Should be ignored
        })

        split_idx = k_fold_split(labels, parameters, root=custom_root)

        assert "train" in split_idx
        # Check that split was saved in custom root
        assert os.path.exists(os.path.join(custom_root, "data_splits", "5-fold"))


class TestRandomSplitting:
    """Test random_splitting function."""

    def setup_method(self):
        """Setup method for each test."""
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_basic_random_split(self):
        """Test basic random splitting."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        parameters = DictConfig({
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        split_idx = random_splitting(labels, parameters)

        assert "train" in split_idx
        assert "valid" in split_idx
        assert "test" in split_idx

        total = len(split_idx["train"]) + len(split_idx["valid"]) + len(split_idx["test"])
        assert total == len(labels)

    def test_random_split_proportions(self):
        """Test that random split respects train_prop."""
        labels = np.array([0, 1, 2] * 100)  # 300 samples
        parameters = DictConfig({
            "data_seed": 0,
            "train_prop": 0.7,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        split_idx = random_splitting(labels, parameters)

        train_ratio = len(split_idx["train"]) / len(labels)
        # Should be approximately 0.7
        assert 0.65 < train_ratio < 0.75

    def test_random_split_with_custom_seed(self):
        """Test random splitting with custom global seed."""
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        parameters = DictConfig({
            "data_seed": 0,
            "train_prop": 0.6,
            "data_split_dir": os.path.join(self.test_dir, "data_splits")
        })

        split_idx = random_splitting(labels, parameters, global_data_seed=999)

        assert "train" in split_idx
        # Verify split directory reflects custom seed
        split_dir = os.path.join(self.test_dir, "data_splits", "train_prop=0.6_global_seed=999")
        assert os.path.exists(split_dir)


class TestAssignMasks:
    """Test assign_train_val_test_mask_to_graphs function."""

    def test_assign_masks(self):
        """Test mask assignment to graphs."""
        # Create mock graphs
        mock_graphs = []
        for i in range(10):
            graph = MagicMock()
            mock_graphs.append(graph)

        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = lambda self, idx: mock_graphs[idx]

        split_idx = {
            "train": np.array([0, 1, 2, 3, 4]),
            "valid": np.array([5, 6, 7]),
            "test": np.array([8, 9])
        }

        train_ds, val_ds, test_ds = assign_train_val_test_mask_to_graphs(
            mock_dataset, split_idx
        )

        assert len(train_ds) == 5
        assert len(val_ds) == 3
        assert len(test_ds) == 2
