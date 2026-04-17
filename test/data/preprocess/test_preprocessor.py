"""Comprehensive unit tests for the PreProcessor class.

This test file provides extensive coverage of the PreProcessor class functionality,
including initialization, data transformations, split loading, and edge cases.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch, mock_open
import torch
import torch_geometric.data
from omegaconf import DictConfig

from topobench.data.preprocessor.preprocessor import PreProcessor


class MockTorchDataset(torch.utils.data.Dataset):
    """A mock of the torch.utils.data.Dataset class.

    Parameters
    ----------
    data : Any
        The data to store in the dataset.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        """Return the length of the data.

        Returns
        -------
        int
            The length of the data.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Return the data at the given index.

        Parameters
        ----------
        idx : int
            The index of the data to return.

        Returns
        -------
        Any
            The data at the given index.
        """
        return self.data[idx]


class TestPreProcessorBasic:
    """Test basic PreProcessor functionality."""

    def test_init_without_transforms(self):
        """Test PreProcessor initialization without transforms."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([
            torch_geometric.data.Data(x=torch.randn(3, 4)),
            torch_geometric.data.Data(x=torch.randn(5, 4)),
        ]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    assert preprocessor.transforms_applied == False
                    assert hasattr(preprocessor, 'data_list')

    def test_init_preserves_split_idx(self):
        """Test that split_idx is preserved from dataset."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.split_idx = {"train": [0, 1], "val": [2], "test": [3]}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    assert hasattr(preprocessor, "split_idx")
                    assert preprocessor.split_idx == mock_dataset.split_idx

    def test_processed_file_names(self):
        """Test the processed_file_names property."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    assert preprocessor.processed_file_names == "data.pt"

    @patch("topobench.data.preprocessor.preprocessor.load_inductive_splits")
    def test_load_dataset_splits_inductive(self, mock_load_inductive_splits):
        """Test loading dataset splits for inductive learning.

        Parameters
        ----------
        mock_load_inductive_splits : MagicMock
            Mock of the load_inductive_splits function.
        """
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    split_params = DictConfig({"learning_setting": "inductive"})
                    preprocessor.load_dataset_splits(split_params)

                    mock_load_inductive_splits.assert_called_once_with(
                        preprocessor, split_params
                    )

    @patch("topobench.data.preprocessor.preprocessor.load_transductive_splits")
    def test_load_dataset_splits_transductive(self, mock_load_transductive_splits):
        """Test loading dataset splits for transductive learning.

        Parameters
        ----------
        mock_load_transductive_splits : MagicMock
            Mock of the load_transductive_splits function.
        """
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    split_params = DictConfig({"learning_setting": "transductive"})
                    preprocessor.load_dataset_splits(split_params)

                    mock_load_transductive_splits.assert_called_once_with(
                        preprocessor, split_params
                    )

    def test_invalid_learning_setting(self):
        """Test error with invalid learning setting."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    split_params = DictConfig({"learning_setting": "invalid"})
                    with pytest.raises(ValueError, match="Invalid.*learning setting"):
                        preprocessor.load_dataset_splits(split_params)

    def test_no_learning_setting_error(self):
        """Test error when no learning setting is specified."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}
        mock_dataset.__iter__ = MagicMock(return_value=iter([]))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("torch_geometric.data.InMemoryDataset.__init__"):
                with patch.object(PreProcessor, "load"):
                    preprocessor = PreProcessor(mock_dataset, tmpdir, None)

                    # Test with no learning_setting key
                    split_params = DictConfig({})
                    with pytest.raises(ValueError, match="No learning setting specified"):
                        preprocessor.load_dataset_splits(split_params)

                    # Test with learning_setting = False
                    split_params = DictConfig({"learning_setting": False})
                    with pytest.raises(ValueError, match="No learning setting specified"):
                        preprocessor.load_dataset_splits(split_params)


class TestPreProcessorProcessing:
    """Test PreProcessor data processing methods."""

    def test_process_with_torch_utils_dataset(self):
        """Test process method with torch.utils.data.Dataset."""
        mock_data = [
            torch_geometric.data.Data(x=torch.randn(3, 4)),
            torch_geometric.data.Data(x=torch.randn(5, 4)),
        ]
        mock_dataset = MockTorchDataset(mock_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.dataset = mock_dataset
                preprocessor.pre_transform = None
                preprocessor.collate = MagicMock(
                    return_value=(torch_geometric.data.Data(), {})
                )
                preprocessor.save = MagicMock()

                # Mock the processed_paths property
                with patch.object(type(preprocessor), 'processed_paths', new_callable=lambda: property(lambda self: [f"{tmpdir}/data.pt"])):
                    preprocessor.process()

                    assert len(preprocessor.data_list) == len(mock_data)
                    preprocessor.collate.assert_called_once()
                    preprocessor.save.assert_called_once()

    def test_process_with_torch_geometric_data(self):
        """Test process method with torch_geometric.data.Data."""
        mock_data = torch_geometric.data.Data(x=torch.randn(3, 4))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.dataset = mock_data
                preprocessor.pre_transform = None
                preprocessor.collate = MagicMock(
                    return_value=(torch_geometric.data.Data(), {})
                )
                preprocessor.save = MagicMock()

                # Mock the processed_paths property
                with patch.object(type(preprocessor), 'processed_paths', new_callable=lambda: property(lambda self: [f"{tmpdir}/data.pt"])):
                    preprocessor.process()

                    assert preprocessor.data_list == [mock_data]
                    preprocessor.collate.assert_called_once_with([mock_data])

    def test_process_with_pre_transform(self):
        """Test process method with a pre_transform applied."""
        mock_data = [
            torch_geometric.data.Data(x=torch.randn(3, 4)),
            torch_geometric.data.Data(x=torch.randn(5, 4))
        ]
        mock_dataset = MockTorchDataset(mock_data)
        mock_pre_transform = MagicMock(side_effect=lambda x: x)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.dataset = mock_dataset
                preprocessor.pre_transform = mock_pre_transform
                preprocessor.collate = MagicMock(
                    return_value=(torch_geometric.data.Data(), {})
                )
                preprocessor.save = MagicMock()

                # Mock the processed_paths property
                with patch.object(type(preprocessor), 'processed_paths', new_callable=lambda: property(lambda self: [f"{tmpdir}/data.pt"])):
                    preprocessor.process()

                    # Verify pre_transform was called for each data item
                    assert mock_pre_transform.call_count == len(mock_data)


class TestPreProcessorLoad:
    """Test PreProcessor load method."""

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_backward_compatibility_2_elements(self, mock_torch_load):
        """Test load method with 2 elements (backward compatibility).

        Parameters
        ----------
        mock_torch_load : MagicMock
            Mock of the torch_load function.
        """
        mock_data = torch_geometric.data.Data()
        mock_slices = {"x": torch.tensor([0, 3])}
        mock_torch_load.return_value = (mock_data, mock_slices)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.load("/fake/path")

                # Use _data as that's what the actual code uses
                assert preprocessor._data == mock_data
                assert preprocessor.slices == mock_slices

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_backward_compatibility_3_elements(self, mock_torch_load):
        """Test load method with 3 elements (backward compatibility).

        Parameters
        ----------
        mock_torch_load : MagicMock
            Mock of the torch_load function.
        """
        mock_data = torch_geometric.data.Data()
        mock_slices = {"x": torch.tensor([0, 3])}
        mock_data_cls = torch_geometric.data.Data
        mock_torch_load.return_value = (mock_data, mock_slices, mock_data_cls)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.load("/fake/path")

                assert preprocessor._data == mock_data
                assert preprocessor.slices == mock_slices

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_with_4_elements(self, mock_torch_load):
        """Test load method with 4 elements (TU Datasets format).

        Parameters
        ----------
        mock_torch_load : MagicMock
            Mock of the torch_load function.
        """
        mock_data = torch_geometric.data.Data()
        mock_slices = {"x": torch.tensor([0, 3])}
        mock_sizes = {"x": 3}
        mock_data_cls = torch_geometric.data.Data
        mock_torch_load.return_value = (mock_data, mock_slices, mock_sizes, mock_data_cls)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.load("/fake/path")

                assert preprocessor._data == mock_data
                assert preprocessor.slices == mock_slices

    @patch("topobench.data.preprocessor.preprocessor.fs.torch_load")
    def test_load_with_dict_data(self, mock_torch_load):
        """Test load method when data is a dictionary.

        Parameters
        ----------
        mock_torch_load : MagicMock
            Mock of the torch_load function.
        """
        mock_data_dict = {
            "x": torch.randn(3, 4),
            "edge_index": torch.tensor([[0, 1], [1, 2]])
        }
        mock_slices = {"x": torch.tensor([0, 3])}
        mock_data_cls = MagicMock()
        mock_reconstructed_data = torch_geometric.data.Data()
        mock_data_cls.from_dict.return_value = mock_reconstructed_data
        mock_torch_load.return_value = (mock_data_dict, mock_slices, mock_data_cls)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.load("/fake/path")

                mock_data_cls.from_dict.assert_called_once_with(mock_data_dict)
                assert preprocessor._data == mock_reconstructed_data
                assert preprocessor.slices == mock_slices


class TestPreProcessorTransforms:
    """Test PreProcessor with transforms."""

    def test_save_transform_parameters_new_file(self):
        """Test saving transform parameters when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.processed_data_dir = tmpdir
                preprocessor.transforms_parameters = {
                    "transform1": {"param": "value"}
                }

                preprocessor.save_transform_parameters()

                # Check if file was created
                param_file = os.path.join(
                    tmpdir, "path_transform_parameters_dict.json"
                )
                assert os.path.exists(param_file)

                # Check file contents
                with open(param_file, 'r') as f:
                    saved_params = json.load(f)
                assert saved_params == preprocessor.transforms_parameters

    def test_save_transform_parameters_existing_same(self, capsys):
        """Test saving transform parameters when file exists with same params.

        Parameters
        ----------
        capsys : pytest.CaptureFixture
            Pytest fixture to capture stdout/stderr output.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing params file
            params = {"transform1": {"param": "value"}}
            param_file = os.path.join(
                tmpdir, "path_transform_parameters_dict.json"
            )
            with open(param_file, 'w') as f:
                json.dump(params, f)

            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.processed_data_dir = tmpdir
                preprocessor.transforms_parameters = params

                preprocessor.save_transform_parameters()

                # Check that message was printed
                captured = capsys.readouterr()
                assert "Transform parameters are the same" in captured.out

    def test_save_transform_parameters_existing_different(self):
        """Test error when saving different transform parameters to same path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create existing params file with different params
            existing_params = {"transform1": {"param": "old_value"}}
            param_file = os.path.join(
                tmpdir, "path_transform_parameters_dict.json"
            )
            with open(param_file, 'w') as f:
                json.dump(existing_params, f)

            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.processed_data_dir = tmpdir
                preprocessor.transforms_parameters = {
                    "transform1": {"param": "new_value"}
                }

                with pytest.raises(ValueError, match="Different transform parameters"):
                    preprocessor.save_transform_parameters()

    def test_instantiate_pre_transform_with_liftings(self):
        """Test instantiate_pre_transform with liftings config."""
        mock_dataset = MagicMock(spec=torch_geometric.data.Dataset)
        mock_dataset.transform = None
        mock_dataset._data = torch_geometric.data.Data()
        mock_dataset.slices = {}

        transforms_config = DictConfig({
            "liftings": {
                "transform1": {"transform_name": "DummyTransform", "param1": "value1"}
            }
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create preprocessor instance
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)

                # Mock DataTransform to avoid needing real transforms
                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    mock_dt.return_value = MagicMock()
                    preprocessor.set_processed_data_dir = MagicMock()

                    pre_transform = preprocessor.instantiate_pre_transform(
                        tmpdir, transforms_config
                    )

                    # Check that a Compose object was created
                    assert hasattr(pre_transform, '__call__')

    def test_instantiate_pre_transform_multiple_transforms(self):
        """Test instantiate_pre_transform with multiple transforms (else branch)."""
        transforms_config = DictConfig({
            "transform1": {"transform_name": "Transform1", "param1": "value1"},
            "transform2": {"transform_name": "Transform2", "param2": "value2"}
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)

                # Mock DataTransform
                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    mock_dt.return_value = MagicMock()

                    # Mock set_processed_data_dir
                    preprocessor.set_processed_data_dir = MagicMock()

                    pre_transform = preprocessor.instantiate_pre_transform(
                        tmpdir, transforms_config
                    )

                    # DataTransform should be called for each transform
                    assert mock_dt.call_count == 2
                    assert hasattr(pre_transform, '__call__')

    def test_instantiate_pre_transform_single_transform(self):
        """Test instantiate_pre_transform with single transform (if branch)."""
        transforms_config = DictConfig({
            "transform_name": "SingleTransform",
            "param1": "value1",
            "param2": 42
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)

                # Mock DataTransform
                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    # Mock DataTransform to return a mock object
                    mock_transform = MagicMock()
                    mock_dt.return_value = mock_transform

                    # Mock set_processed_data_dir
                    preprocessor.set_processed_data_dir = MagicMock()

                    pre_transform = preprocessor.instantiate_pre_transform(
                        tmpdir, transforms_config
                    )

                    # DataTransform should be called once with the entire config
                    assert mock_dt.call_count == 1
                    # Should be called with all the config parameters
                    mock_dt.assert_called_once_with(**transforms_config)

                    # Verify the pre_transform is a Compose object
                    assert isinstance(
                        pre_transform,
                        torch_geometric.transforms.Compose
                    )

    def test_instantiate_pre_transform_calls_set_processed_data_dir(self):
        """Test that instantiate_pre_transform calls set_processed_data_dir."""
        transforms_config = DictConfig({
            "transform1": {"transform_name": "Transform1", "param1": "value1"}
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)

                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    mock_dt.return_value = MagicMock()
                    # Mock set_processed_data_dir
                    preprocessor.set_processed_data_dir = MagicMock()

                    pre_transform = preprocessor.instantiate_pre_transform(
                        tmpdir, transforms_config
                    )

                    # Verify set_processed_data_dir was called
                    preprocessor.set_processed_data_dir.assert_called_once()
                    call_args = preprocessor.set_processed_data_dir.call_args
                    assert call_args[0][1] == tmpdir
                    assert call_args[0][2] == transforms_config

    def test_instantiate_pre_transform_returns_compose(self):
        """Test that instantiate_pre_transform returns a Compose object."""
        transforms_config = DictConfig({
            "transform1": {"transform_name": "Transform1", "param1": "value1"}
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)

                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    mock_dt.return_value = MagicMock()
                    preprocessor.set_processed_data_dir = MagicMock()

                    pre_transform = preprocessor.instantiate_pre_transform(
                        tmpdir, transforms_config
                    )

                    # Check it's a Compose instance
                    assert isinstance(
                        pre_transform,
                        torch_geometric.transforms.Compose
                    )

    def test_instantiate_pre_transform_single_vs_multiple(self):
        """Test that the method correctly distinguishes between single and multiple transforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.set_processed_data_dir = MagicMock()

                with patch("topobench.data.preprocessor.preprocessor.DataTransform") as mock_dt:
                    mock_dt.return_value = MagicMock()

                    # Test single transform (has transform_name key)
                    single_config = DictConfig({
                        "transform_name": "SingleTransform",
                        "param1": "value1"
                    })

                    preprocessor.instantiate_pre_transform(tmpdir, single_config)

                    # Should be called once with all parameters
                    assert mock_dt.call_count == 1
                    mock_dt.assert_called_with(**single_config)

                    # Reset mock
                    mock_dt.reset_mock()

                    # Test multiple transforms (no transform_name key at top level)
                    multiple_config = DictConfig({
                        "transform1": {"transform_name": "Transform1", "param1": "value1"},
                        "transform2": {"transform_name": "Transform2", "param2": "value2"}
                    })

                    preprocessor.instantiate_pre_transform(tmpdir, multiple_config)

                    # Should be called twice, once for each transform
                    assert mock_dt.call_count == 2


class TestPreProcessorEdgeCases:
    """Test edge cases and error handling."""

    def test_process_with_empty_dataset(self):
        """Test process method with an empty dataset."""
        mock_dataset = MockTorchDataset([])

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.dataset = mock_dataset
                preprocessor.pre_transform = None
                # For empty list, collate should return a single empty Data object
                preprocessor.collate = MagicMock(
                    return_value=(torch_geometric.data.Data(), {})
                )
                preprocessor.save = MagicMock()

                # Mock the processed_paths property
                with patch.object(type(preprocessor), 'processed_paths', new_callable=lambda: property(lambda self: [f"{tmpdir}/data.pt"])):
                    preprocessor.process()

                    assert preprocessor.data_list == []

    def test_processed_dir_property(self):
        """Test the processed_dir property returns correct paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Without transforms
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.root = tmpdir
                preprocessor.transforms_applied = False

                assert preprocessor.processed_dir == tmpdir

            # With transforms
            with patch.object(PreProcessor, '__init__', lambda self, *args, **kwargs: None):
                preprocessor = PreProcessor(None, tmpdir, None)
                preprocessor.root = tmpdir
                preprocessor.transforms_applied = True

                assert preprocessor.processed_dir == tmpdir + "/processed"
