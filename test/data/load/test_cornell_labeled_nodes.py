"""Unit tests for Cornell labeled nodes hypergraph datasets."""

import os
import shutil
import pytest
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig

from topobench.data.datasets.cornell_labeled_nodes_dataset import (
    CornellLabeledNodesDataset,
)
from topobench.data.loaders.hypergraph import CornellLabeledNodesDatasetLoader


class TestCornellLabeledNodesDataset:
    """Test suite for CornellLabeledNodesDataset class."""

    # Expected statistics for each dataset
    DATASET_STATS = {
        "walmart-trips": {
            "num_nodes": 88860,
            "num_hyperedges": 69906,
            "num_classes": 11,
            "num_edges": 460630,
            "mean_hyperedge_size": 6.59,
            "median_hyperedge_size": 5,
            "max_hyperedge_size": 25,
        },
        "house-committees": {
            "num_nodes": 1290,
            "num_hyperedges": 341,
            "num_classes": 2,
            "num_edges": 11843,  # Deduplicated (was 11863 with duplicates)
            "mean_hyperedge_size": 34.7,  # Recalculated after deduplication
            "median_hyperedge_size": 40.0,
            "max_hyperedge_size": 81,  # Reduced from 82 (one hyperedge had duplicate)
        },
        "senate-committees": {
            "num_nodes": 282,
            "num_hyperedges": 315,
            "num_classes": 2,
            "num_edges": 5408,  # Deduplicated (was 5430 with duplicates)
            "mean_hyperedge_size": 17.2,  # Stays same after rounding
            "median_hyperedge_size": 19.0,
            "max_hyperedge_size": 31,
        },
        "house-bills": {
            "num_nodes": 1494,
            "num_hyperedges": 60987,
            "num_classes": 2,
            "num_edges": 1248666,
            "mean_hyperedge_size": 20.5,
            "median_hyperedge_size": 9.0,
            "max_hyperedge_size": 399,
        },
        "senate-bills": {
            "num_nodes": 294,
            "num_hyperedges": 29157,
            "num_classes": 2,
            "num_edges": 232147,
            "mean_hyperedge_size": 8.0,
            "median_hyperedge_size": 4.0,
            "max_hyperedge_size": 99,
        },
        "contact-primary-school": {
            "num_nodes": 242,
            "num_hyperedges": 12704,
            "num_classes": 11,
            "num_edges": 30729,
            "mean_hyperedge_size": 2.4,
            "median_hyperedge_size": 2.0,
            "max_hyperedge_size": 5,
        },
        "contact-high-school": {
            "num_nodes": 327,
            "num_hyperedges": 7818,
            "num_classes": 9,
            "num_edges": 18192,
            "mean_hyperedge_size": 2.3,
            "median_hyperedge_size": 2.0,
            "max_hyperedge_size": 5,
        },
        "amazon-reviews": {
            "num_nodes": 2268264,  # Actual from dataset (Cornell website says 2,268,231)
            "num_hyperedges": 4285363,
            "num_classes": 29,
            "num_edges": 73141425,  # Actual edge_index shape[1]
            "mean_hyperedge_size": 17.1,
            "median_hyperedge_size": 8.0,
            "max_hyperedge_size": 9350,
        },
    }

    @pytest.fixture(scope="session")
    def test_data_dir(self):
        """Create persistent cache directory for test data.
        
        Uses session scope so datasets are downloaded once and reused
        across all test runs, which is important for large datasets
        like amazon-reviews.

        Returns
        -------
        Path
            Persistent cache directory path.
        """
        cache_dir = Path(".test_tmp/cornell_datasets")
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @pytest.fixture
    def walmart_params(self, test_data_dir):
        """Create parameters for walmart-trips dataset.

        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.

        Returns
        -------
        DictConfig
            Configuration parameters.
        """
        return DictConfig({
            "data_dir": str(test_data_dir),
            "data_name": "walmart-trips"
        })

    @pytest.mark.parametrize("dataset_name", ["walmart-trips", "house-committees", "senate-committees", "house-bills", "senate-bills", "contact-primary-school", "contact-high-school", "amazon-reviews"])
    def test_dataset_initialization(self, test_data_dir, dataset_name):
        """Test that dataset initializes correctly.

        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_name : str
            Name of dataset to test.
        """
        dataset = CornellLabeledNodesDataset(
            data_dir=str(test_data_dir),
            data_name=dataset_name,
        )
        
        assert dataset is not None
        assert dataset.name == dataset_name
        assert len(dataset) == 1  # Single hypergraph

    @pytest.mark.parametrize("dataset_name", ["walmart-trips", "house-committees", "senate-committees", "house-bills", "senate-bills", "contact-primary-school", "contact-high-school", "amazon-reviews"])
    def test_dataset_structure(self, test_data_dir, dataset_name):
        """Test that loaded dataset has correct structure.

        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_name : str
            Name of dataset to test.
        """
        dataset = CornellLabeledNodesDataset(
            data_dir=str(test_data_dir),
            data_name=dataset_name,
        )
        data = dataset[0]
        
        # Check required attributes exist
        assert hasattr(data, 'x'), "Missing node features"
        assert hasattr(data, 'y'), "Missing labels"
        assert hasattr(data, 'edge_index'), "Missing edge_index"
        assert hasattr(data, 'incidence_hyperedges'), "Missing incidence matrix"
        assert hasattr(data, 'num_nodes'), "Missing num_nodes"
        assert hasattr(data, 'num_hyperedges'), "Missing num_hyperedges"
        assert hasattr(data, 'num_class'), "Missing num_class"

    @pytest.mark.parametrize("dataset_name", ["walmart-trips", "house-committees", "senate-committees", "house-bills", "senate-bills", "contact-primary-school", "contact-high-school", "amazon-reviews"])
    def test_dataset_statistics(self, test_data_dir, dataset_name):
        """Test that dataset has expected statistics.

        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_name : str
            Name of dataset to test.
        """
        dataset = CornellLabeledNodesDataset(
            data_dir=str(test_data_dir),
            data_name=dataset_name,
        )
        data = dataset[0]
        
        # Get expected statistics for this dataset
        expected = self.DATASET_STATS[dataset_name]
        
        # Check node/hyperedge/class counts
        assert data.num_nodes == expected["num_nodes"], "Incorrect number of nodes"
        assert data.num_hyperedges == expected["num_hyperedges"], "Incorrect number of hyperedges"
        assert data.num_class == expected["num_classes"], "Incorrect number of classes"
        
        # Check shapes
        assert data.x.shape == (expected["num_nodes"], 1), "Incorrect feature shape"
        assert data.y.shape == (expected["num_nodes"],), "Incorrect label shape"
        assert data.edge_index.shape[0] == 2, "Edge index should be 2xN"
        assert data.edge_index.shape[1] == expected["num_edges"], "Incorrect number of edges"

    @pytest.mark.parametrize("dataset_name", ["walmart-trips", "house-committees", "senate-committees", "house-bills", "senate-bills", "contact-primary-school", "contact-high-school", "amazon-reviews"])
    def test_label_indexing(self, test_data_dir, dataset_name):
        """Test that labels are correctly 0-indexed.

        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_name : str
            Name of dataset to test.
        """
        dataset = CornellLabeledNodesDataset(
            data_dir=str(test_data_dir),
            data_name=dataset_name,
        )
        data = dataset[0]
        
        expected = self.DATASET_STATS[dataset_name]
        
        # Labels should be 0-indexed
        assert data.y.min().item() == 0, "Labels should start from 0"
        assert data.y.max().item() == expected["num_classes"] - 1, \
            "Labels should end at num_classes-1"
        
        # Check all labels are valid
        unique_labels = torch.unique(data.y).tolist()
        assert unique_labels == list(range(expected["num_classes"])), \
            f"Labels should be [0, 1, ..., {expected['num_classes']-1}]"

    @pytest.mark.parametrize("dataset_name", ["walmart-trips", "house-committees", "senate-committees", "house-bills", "senate-bills", "contact-primary-school", "contact-high-school", "amazon-reviews"])
    def test_node_indexing(self, test_data_dir, dataset_name):
        """Test that node IDs in edge_index are 0-indexed.

        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_name : str
            Name of dataset to test.
        """
        dataset = CornellLabeledNodesDataset(
            data_dir=str(test_data_dir),
            data_name=dataset_name,
        )
        data = dataset[0]
        
        # Node IDs should be 0-indexed
        node_ids = data.edge_index[0]
        assert node_ids.min().item() == 0, "Node IDs should start from 0"
        assert node_ids.max().item() == data.num_nodes - 1, \
            "Max node ID should be num_nodes-1"

    @pytest.mark.parametrize("dataset_name", ["walmart-trips", "house-committees", "senate-committees", "house-bills", "senate-bills", "contact-primary-school", "contact-high-school", "amazon-reviews"])
    def test_hyperedge_statistics(self, test_data_dir, dataset_name):
        """Test hyperedge size statistics.

        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_name : str
            Name of dataset to test.
        """
        dataset = CornellLabeledNodesDataset(
            data_dir=str(test_data_dir),
            data_name=dataset_name,
        )
        data = dataset[0]
        
        expected = self.DATASET_STATS[dataset_name]
        
        # Calculate hyperedge sizes
        hyperedge_sizes = torch.bincount(data.edge_index[1])
        mean_size = hyperedge_sizes.float().mean().item()
        median_size = hyperedge_sizes.float().median().item()
        max_size = hyperedge_sizes.max().item()
        
        # Check statistics (use 0.1 tolerance for mean to account for rounding)
        assert abs(mean_size - expected["mean_hyperedge_size"]) < 0.1, \
            f"Incorrect mean hyperedge size for {dataset_name}: {mean_size} vs {expected['mean_hyperedge_size']}"
        assert median_size == expected["median_hyperedge_size"], \
            f"Incorrect median hyperedge size for {dataset_name}"
        assert max_size == expected["max_hyperedge_size"], \
            f"Incorrect max hyperedge size for {dataset_name}"

    @pytest.mark.parametrize("dataset_name", ["walmart-trips", "house-committees", "senate-committees", "house-bills", "senate-bills", "contact-primary-school", "contact-high-school", "amazon-reviews"])
    def test_incidence_matrix(self, test_data_dir, dataset_name):
        """Test that incidence matrix is correctly formed.

        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_name : str
            Name of dataset to test.
        """
        dataset = CornellLabeledNodesDataset(
            data_dir=str(test_data_dir),
            data_name=dataset_name,
        )
        data = dataset[0]
        
        expected = self.DATASET_STATS[dataset_name]
        
        # Check incidence matrix properties
        assert data.incidence_hyperedges.is_sparse, "Incidence should be sparse"
        assert data.incidence_hyperedges.shape == (expected["num_nodes"], expected["num_hyperedges"]), \
            "Incorrect incidence matrix shape"
        
        # Check that incidence matrix is binary by verifying construction
        # The incidence matrix is constructed from edge_index with all 1.0 values
        # When coalesced, duplicate entries are summed, but there shouldn't be duplicates
        # in a properly constructed incidence matrix
        
        # Verify the number of stored values matches edge count
        assert data.incidence_hyperedges._nnz() == data.edge_index.shape[1], \
            "Number of non-zero entries should equal number of edges"

    @pytest.mark.parametrize("dataset_name", ["walmart-trips", "house-committees", "senate-committees", "house-bills", "senate-bills", "contact-primary-school", "contact-high-school", "amazon-reviews"])
    def test_data_consistency(self, test_data_dir, dataset_name):
        """Test consistency between edge_index and incidence matrix.

        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        dataset_name : str
            Name of dataset to test.
        """
        dataset = CornellLabeledNodesDataset(
            data_dir=str(test_data_dir),
            data_name=dataset_name,
        )
        data = dataset[0]
        
        # Number of edges should match incidence matrix nnz
        num_edges = data.edge_index.shape[1]
        nnz = data.incidence_hyperedges._nnz()
        assert num_edges == nnz, \
            "Edge count mismatch between edge_index and incidence matrix"

    def test_invalid_dataset_name(self, test_data_dir):
        """Test that invalid dataset name raises error.

        Parameters
        ----------
        test_data_dir : Path
            Temporary directory for test data.
        """
        with pytest.raises(ValueError, match="Unknown dataset"):
            CornellLabeledNodesDataset(
                data_dir=str(test_data_dir),
                data_name="invalid-dataset",
            )

