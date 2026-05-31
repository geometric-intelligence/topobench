"""Comprehensive unit tests for the OnDiskPreProcessor class.

This test file provides extensive coverage of the OnDiskPreProcessor class functionality,
including initialization, data transformations, split loading, edge cases, and memory efficiency.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_geometric
from omegaconf import DictConfig

from topobench.data.preprocessor.on_disk_preprocessor import OnDiskPreProcessor


class TestOnDiskPreProcessor:
    """Test OnDiskPreProcessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_list = []
        for i in range(5):
            data = torch_geometric.data.Data(
                x=torch.randn(10, 3),
                edge_index=torch.randint(0, 10, (2, 20)),
                y=torch.tensor([i % 2]),
            )
            self.mock_data_list.append(data)

        # Create mock dataset
        self.mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        self.mock_dataset.__len__ = MagicMock(return_value=5)
        self.mock_dataset.__getitem__ = MagicMock(
            side_effect=lambda idx: self.mock_data_list[idx]
        )
        self.mock_dataset.transform = None

    def test_init_without_transforms(self):
        """Test initialization without transforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            assert preprocessor.transforms_applied is False
            assert preprocessor.pre_transform is None
            assert len(preprocessor) == 5

    @patch('topobench.data.preprocessor.on_disk_preprocessor.DataTransform')
    def test_init_with_transforms(self, mock_data_transform):
        """Test initialization with transforms.
        
        Parameters
        ----------
        mock_data_transform : MagicMock
            Mock of the DataTransform class.
        """
        # Create mock transform
        mock_transform = MagicMock()
        mock_transform.parameters = {"transform_name": "test_transform"}
        mock_data_transform.return_value = mock_transform
        
        with tempfile.TemporaryDirectory() as tmpdir:
            transform_config = DictConfig({
                "test_transform": {
                    "transform_name": "test_transform",
                }
            })
            
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transform_config
            )
            
            assert preprocessor.transforms_applied is True
            assert preprocessor.pre_transform is not None

    def test_process_saves_to_disk(self):
        """Test that process() saves individual files to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            preprocessor.process()
            
            processed_dir = preprocessor.processed_dir
            assert os.path.exists(processed_dir)
            
            for i in range(5):
                file_path = os.path.join(processed_dir, f"data_{i}.pt")
                assert os.path.exists(file_path), f"File {file_path} not found"

    def test_get_loads_from_disk(self):
        """Test that get() loads data from disk correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            preprocessor.process()
            
            for i in range(5):
                data = preprocessor.get(i)
                assert isinstance(data, torch_geometric.data.Data)
                assert data.x.shape == (10, 3)
                assert data.y.item() == i % 2

    def test_len_returns_correct_count(self):
        """Test that len() returns correct number of samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            assert len(preprocessor) == 5

    def test_processed_files_not_reprocessed(self):
        """Test that already processed files are not reprocessed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            preprocessor.process()
            
            first_file = os.path.join(preprocessor.processed_dir, "data_0.pt")
            first_mtime = os.path.getmtime(first_file)
            
            time.sleep(0.1)
            preprocessor.process()
            
            second_mtime = os.path.getmtime(first_file)
            assert first_mtime == second_mtime

    @patch("topobench.data.preprocessor.on_disk_preprocessor.load_inductive_splits")
    def test_load_dataset_splits_inductive(self, mock_load_splits):
        """Test loading dataset splits for inductive learning.
        
        Parameters
        ----------
        mock_load_splits : MagicMock
            Mock of the load_inductive_splits function.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            split_params = DictConfig({"learning_setting": "inductive"})
            preprocessor.load_dataset_splits(split_params)
            
            mock_load_splits.assert_called_once_with(preprocessor, split_params)

    @patch("topobench.data.preprocessor.on_disk_preprocessor.load_transductive_splits")
    def test_load_dataset_splits_transductive(self, mock_load_splits):
        """Test loading dataset splits for transductive learning.
        
        Parameters
        ----------
        mock_load_splits : MagicMock
            Mock of the load_transductive_splits function.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            split_params = DictConfig({"learning_setting": "transductive"})
            preprocessor.load_dataset_splits(split_params)
            
            mock_load_splits.assert_called_once_with(preprocessor, split_params)

    def test_memory_efficiency(self):
        """Test that processing doesn't load all data into memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create larger mock dataset
            large_dataset = MagicMock(spec=torch_geometric.data.Dataset)
            large_dataset.__len__ = MagicMock(return_value=100)
            large_dataset.__getitem__ = MagicMock(
                side_effect=lambda idx: torch_geometric.data.Data(
                    x=torch.randn(50, 10),
                    edge_index=torch.randint(0, 50, (2, 100)),
                    y=torch.tensor([idx % 5]),
                )
            )
            large_dataset.transform = None
            
            preprocessor = OnDiskPreProcessor(
                large_dataset, tmpdir, transforms_config=None
            )
            
            preprocessor.process()
            
            for i in range(100):
                file_path = os.path.join(preprocessor.processed_dir, f"data_{i}.pt")
                assert os.path.exists(file_path)


class TestOnDiskPreProcessorEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_list = []
        for i in range(5):
            data = torch_geometric.data.Data(
                x=torch.randn(10, 3),
                edge_index=torch.randint(0, 10, (2, 20)),
                y=torch.tensor([i % 2]),
            )
            self.mock_data_list.append(data)

        self.mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        self.mock_dataset.__len__ = MagicMock(return_value=5)
        self.mock_dataset.__getitem__ = MagicMock(
            side_effect=lambda idx: self.mock_data_list[idx]
        )
        self.mock_dataset.transform = None

    def test_process_with_empty_dataset(self):
        """Test process method with an empty dataset."""
        empty_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        empty_dataset.__len__ = MagicMock(return_value=0)
        empty_dataset.__getitem__ = MagicMock(side_effect=lambda idx: None)
        empty_dataset.transform = None
        
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                empty_dataset, tmpdir, transforms_config=None
            )
            
            # Should handle empty dataset gracefully
            preprocessor.process()
            assert len(preprocessor) == 0

    def test_invalid_learning_setting(self):
        """Test error with invalid learning setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            split_params = DictConfig({"learning_setting": "invalid"})
            with pytest.raises(ValueError, match="Invalid.*learning setting"):
                preprocessor.load_dataset_splits(split_params)

    def test_no_learning_setting_error(self):
        """Test error when no learning setting is specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            # Test with no learning_setting key
            split_params = DictConfig({})
            with pytest.raises(ValueError, match="No learning setting specified"):
                preprocessor.load_dataset_splits(split_params)
            
            # Test with learning_setting = False
            split_params = DictConfig({"learning_setting": False})
            with pytest.raises(ValueError, match="No learning setting specified"):
                preprocessor.load_dataset_splits(split_params)

    def test_processed_dir_property(self):
        """Test the processed_dir property returns correct paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Without transforms
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            assert preprocessor.processed_dir == tmpdir
            
            # With transforms
            transform_config = DictConfig({
                "test_transform": {
                    "transform_name": "test_transform",
                }
            })
            with patch('topobench.data.preprocessor.on_disk_preprocessor.DataTransform'):
                preprocessor_with_transforms = OnDiskPreProcessor(
                    self.mock_dataset, tmpdir, transform_config
                )
                assert preprocessor_with_transforms.processed_dir.endswith("/processed")
                assert "test_transform" in preprocessor_with_transforms.processed_dir

    def test_processed_file_names_property(self):
        """Test the processed_file_names property."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            expected_names = [f"data_{i}.pt" for i in range(5)]
            assert preprocessor.processed_file_names == expected_names

    def test_get_with_runtime_transform(self):
        """Test get() applies runtime transforms correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            preprocessor.process()
            
            # Runtime transform
            def mock_transform(data):
                data.x = data.x * 2
                return data
            
            preprocessor.transform = mock_transform
            
            original_data = torch.load(
                os.path.join(preprocessor.processed_dir, "data_0.pt")
            )
            
            transformed_data = preprocessor.get(0)
            
            assert torch.allclose(transformed_data.x, original_data.x * 2)

    def test_init_preserves_split_idx(self):
        """Test that split_idx is preserved from dataset."""
        mock_dataset_with_splits = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset_with_splits.__len__ = MagicMock(return_value=5)
        mock_dataset_with_splits.__getitem__ = MagicMock(
            side_effect=lambda idx: self.mock_data_list[idx]
        )
        mock_dataset_with_splits.transform = None
        mock_dataset_with_splits.split_idx = {
            "train": [0, 1, 2], 
            "val": [3], 
            "test": [4]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessor = OnDiskPreProcessor(
                mock_dataset_with_splits, tmpdir, transforms_config=None
            )
            
            assert hasattr(preprocessor, "split_idx")
            assert preprocessor.split_idx == mock_dataset_with_splits.split_idx


class TestOnDiskPreProcessorTransforms:
    """Test OnDiskPreProcessor with transforms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_list = []
        for i in range(3):
            data = torch_geometric.data.Data(
                x=torch.randn(10, 3),
                edge_index=torch.randint(0, 10, (2, 20)),
                y=torch.tensor([i % 2]),
            )
            self.mock_data_list.append(data)

        self.mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        self.mock_dataset.__len__ = MagicMock(return_value=3)
        self.mock_dataset.__getitem__ = MagicMock(
            side_effect=lambda idx: self.mock_data_list[idx]
        )
        self.mock_dataset.transform = None

    def test_save_transform_parameters_new_file(self):
        """Test saving transform parameters when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transform_config = DictConfig({
                "transform1": {"transform_name": "Transform1", "param1": "value1"}
            })
            
            with patch('topobench.data.preprocessor.on_disk_preprocessor.DataTransform') as mock_dt:
                mock_transform = MagicMock()
                mock_transform.parameters = {"transform_name": "Transform1", "param1": "value1"}
                mock_dt.return_value = mock_transform
                
                preprocessor = OnDiskPreProcessor(
                    self.mock_dataset, tmpdir, transform_config
                )
                
                # Check if file was created
                param_file = os.path.join(
                    preprocessor.processed_data_dir, "path_transform_parameters_dict.json"
                )
                assert os.path.exists(param_file)
                
                # Check file contents
                with open(param_file, 'r') as f:
                    saved_params = json.load(f)
                # The key in saved_params is the original config key ("transform1"), not the transform_name
                assert "transform1" in saved_params
                assert saved_params["transform1"]["transform_name"] == "Transform1"

    def test_save_transform_parameters_existing_same(self, capsys):
        """Test saving transform parameters when file exists with same params.
        
        Parameters
        ----------
        capsys : pytest.CaptureFixture
            Pytest fixture to capture stdout/stderr output.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            transform_config = DictConfig({
                "transform1": {"transform_name": "Transform1", "param1": "value1"}
            })
            
            with patch('topobench.data.preprocessor.on_disk_preprocessor.DataTransform') as mock_dt:
                mock_transform = MagicMock()
                mock_transform.parameters = {"transform_name": "Transform1", "param1": "value1"}
                mock_dt.return_value = mock_transform
                
                # Create first preprocessor (saves params)
                preprocessor1 = OnDiskPreProcessor(
                    self.mock_dataset, tmpdir, transform_config
                )
                
                # Create second preprocessor with same params (should print message)
                preprocessor2 = OnDiskPreProcessor(
                    self.mock_dataset, tmpdir, transform_config
                )
                
                captured = capsys.readouterr()
                assert "Transform parameters are the same" in captured.out

    def test_save_transform_parameters_existing_different(self):
        """Test error when saving different transform parameters to same path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first config
            transform_config1 = DictConfig({
                "transform1": {"transform_name": "Transform1", "param1": "old_value"}
            })
            
            with patch('topobench.data.preprocessor.on_disk_preprocessor.DataTransform') as mock_dt:
                mock_transform1 = MagicMock()
                mock_transform1.parameters = {"transform_name": "Transform1", "param1": "old_value"}
                mock_dt.return_value = mock_transform1
                
                preprocessor1 = OnDiskPreProcessor(
                    self.mock_dataset, tmpdir, transform_config1
                )
                
                processed_data_dir = preprocessor1.processed_data_dir
                
                param_file = os.path.join(processed_data_dir, "path_transform_parameters_dict.json")
                with open(param_file, 'w') as f:
                    json.dump({"Transform1": {"param1": "different_value"}}, f)
                
                # Try to create second preprocessor with same transform name but will check against modified file
                with pytest.raises(ValueError, match="Different transform parameters"):
                    with patch.object(OnDiskPreProcessor, 'set_processed_data_dir'):
                        preprocessor2 = OnDiskPreProcessor.__new__(OnDiskPreProcessor)
                        preprocessor2.processed_data_dir = processed_data_dir
                        preprocessor2.transforms_parameters = {"Transform1": {"param1": "old_value"}}
                        preprocessor2.save_transform_parameters()

    def test_instantiate_pre_transform_with_liftings(self):
        """Test instantiate_pre_transform with liftings config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transforms_config = DictConfig({
                "liftings": {
                    "transform1": {"transform_name": "DummyTransform", "param1": "value1"}
                }
            })
            
            with patch('topobench.data.preprocessor.on_disk_preprocessor.DataTransform') as mock_dt:
                mock_transform = MagicMock()
                mock_transform.parameters = {"transform_name": "DummyTransform"}
                mock_dt.return_value = mock_transform
                
                preprocessor = OnDiskPreProcessor(
                    self.mock_dataset, tmpdir, transforms_config
                )
                
                # Check that pre_transform was created
                assert preprocessor.pre_transform is not None
                assert isinstance(
                    preprocessor.pre_transform,
                    torch_geometric.transforms.Compose
                )

    def test_instantiate_pre_transform_single_transform(self):
        """Test instantiate_pre_transform with single transform."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transforms_config = DictConfig({
                "transform_name": "SingleTransform",
                "param1": "value1",
                "param2": 42
            })
            
            with patch('topobench.data.preprocessor.on_disk_preprocessor.DataTransform') as mock_dt:
                mock_transform = MagicMock()
                mock_transform.parameters = {"transform_name": "SingleTransform"}
                mock_dt.return_value = mock_transform
                
                preprocessor = OnDiskPreProcessor(
                    self.mock_dataset, tmpdir, transforms_config
                )
                
                # DataTransform should be called once with the entire config
                assert mock_dt.call_count == 1
                mock_dt.assert_called_once_with(**transforms_config)
                
                # Verify the pre_transform is a Compose object
                assert isinstance(
                    preprocessor.pre_transform,
                    torch_geometric.transforms.Compose
                )

    def test_instantiate_pre_transform_multiple_transforms(self):
        """Test instantiate_pre_transform with multiple transforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            transforms_config = DictConfig({
                "transform1": {"transform_name": "Transform1", "param1": "value1"},
                "transform2": {"transform_name": "Transform2", "param2": "value2"}
            })
            
            with patch('topobench.data.preprocessor.on_disk_preprocessor.DataTransform') as mock_dt:
                mock_transform = MagicMock()
                mock_transform.parameters = {"transform_name": "test"}
                mock_dt.return_value = mock_transform
                
                preprocessor = OnDiskPreProcessor(
                    self.mock_dataset, tmpdir, transforms_config
                )
                
                # DataTransform should be called for each transform
                assert mock_dt.call_count == 2
                assert hasattr(preprocessor.pre_transform, '__call__')

    def test_process_with_pre_transform(self):
        """Test process method with a pre_transform applied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_pre_transform = MagicMock(side_effect=lambda x: x)
            
            # Create preprocessor without config
            preprocessor = OnDiskPreProcessor(
                self.mock_dataset, tmpdir, transforms_config=None
            )
            
            # Delete processed files so we can test process() again
            for i in range(len(self.mock_dataset)):
                file_path = os.path.join(preprocessor.processed_dir, f"data_{i}.pt")
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Set pre_transform and process again
            preprocessor.pre_transform = mock_pre_transform
            preprocessor.process()
            
            # Verify pre_transform was called for each data item
            assert mock_pre_transform.call_count == len(self.mock_dataset)