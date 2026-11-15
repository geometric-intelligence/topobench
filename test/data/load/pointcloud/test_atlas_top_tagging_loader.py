"""Test suite for ATLAS Top Tagging Dataset Loader."""

import pytest
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

# Try to import the loader class
try:
    from topobench.data.loaders.pointcloud.atlas_top_tagging_loader import (
        ATLASTopTaggingDatasetLoader
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


@pytest.fixture
def minimal_config():
    """Minimal loader configuration for testing.

    Returns
    -------
    DictConfig
        Configuration dictionary for loader.
    """
    config = {
        'data_dir': '/tmp/test_data',  # MOVED TO TOP LEVEL
        'data_domain': 'pointcloud',
        'data_name': 'atlas_top_tagging',
        'split': 'train',
        'subset': 0.01,
        'max_constituents': 80,
        'use_high_level': True,
        'verbose': False
    }
    return OmegaConf.create(config)


@pytest.fixture
def test_config():
    """Test configuration with different parameters.

    Returns
    -------
    DictConfig
        Configuration dictionary for loader.
    """
    config = {
        'data_dir': '/tmp/test_data',  # MOVED TO TOP LEVEL
        'data_domain': 'pointcloud',
        'data_name': 'atlas_top_tagging',
        'split': 'test',
        'subset': 0.01,
        'max_constituents': 50,
        'use_high_level': False,
        'verbose': True
    }
    return OmegaConf.create(config)


class TestATLASLoaderImport:
    """Test 1-2: Basic imports and class existence."""
    
    def test_can_import_loader_class(self):
        """Test that the loader class can be imported."""
        assert IMPORT_SUCCESS, f"Failed to import: {IMPORT_ERROR if not IMPORT_SUCCESS else ''}"
        assert ATLASTopTaggingDatasetLoader is not None
    
    def test_class_inherits_from_abstract_loader(self):
        """Test that loader inherits from AbstractLoader."""
        from topobench.data.loaders.base import AbstractLoader
        assert issubclass(ATLASTopTaggingDatasetLoader, AbstractLoader)


class TestATLASLoaderStructure:
    """Test 3-7: Loader has required structure."""
    
    def test_has_init_method(self):
        """Test __init__ method exists."""
        assert hasattr(ATLASTopTaggingDatasetLoader, '__init__')
        assert callable(ATLASTopTaggingDatasetLoader.__init__)
    
    def test_has_load_dataset_method(self):
        """Test load_dataset method exists."""
        assert hasattr(ATLASTopTaggingDatasetLoader, 'load_dataset')
        assert callable(ATLASTopTaggingDatasetLoader.load_dataset)
    
    def test_has_initialize_dataset_method(self):
        """Test _initialize_dataset helper method exists."""
        assert hasattr(ATLASTopTaggingDatasetLoader, '_initialize_dataset')
        assert callable(ATLASTopTaggingDatasetLoader._initialize_dataset)
    
    def test_has_redefine_data_dir_method(self):
        """Test _redefine_data_dir helper method exists."""
        assert hasattr(ATLASTopTaggingDatasetLoader, '_redefine_data_dir')
        assert callable(ATLASTopTaggingDatasetLoader._redefine_data_dir)
    
    def test_loader_has_docstring(self):
        """Test that loader class has docstring."""
        assert ATLASTopTaggingDatasetLoader.__doc__ is not None
        assert len(ATLASTopTaggingDatasetLoader.__doc__) > 0


class TestATLASLoaderInitialization:
    """Test 8-12: Loader initialization."""
    
    def test_can_initialize_with_minimal_config(self, minimal_config):
        """Test loader can be initialized with minimal config.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        assert loader is not None
    
    def test_can_initialize_with_test_config(self, test_config):
        """Test loader can be initialized with test config.
        
        Parameters
        ----------
        test_config : DictConfig
            Test configuration with different parameters.
        """
        loader = ATLASTopTaggingDatasetLoader(test_config)
        assert loader is not None
    
    def test_config_is_stored(self, minimal_config):
        """Test that config is stored as instance variable.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        assert hasattr(loader, 'parameters')
    
    def test_parameters_accessible(self, minimal_config):
        """Test that parameters are accessible.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        assert loader.parameters is not None
    
    def test_inherits_from_abstract_loader_instance(self, minimal_config):
        """Test that instance is of AbstractLoader type.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        from topobench.data.loaders.base import AbstractLoader
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        assert isinstance(loader, AbstractLoader)


class TestATLASLoaderParameterHandling:
    """Test 13-20: Parameter extraction and defaults."""
    
    def test_split_parameter_extracted(self, minimal_config):
        """Test split parameter is accessible.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        assert loader.parameters.get('split') == 'train'
    
    def test_subset_parameter_extracted(self, minimal_config):
        """Test subset parameter is accessible.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        assert loader.parameters.get('subset') == 0.01
    
    def test_max_constituents_parameter_extracted(self, minimal_config):
        """Test max_constituents parameter is accessible.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        assert loader.parameters.get('max_constituents') == 80
    
    def test_use_high_level_parameter_extracted(self, minimal_config):
        """Test use_high_level parameter is accessible.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        assert loader.parameters.get('use_high_level') is True
    
    def test_verbose_parameter_extracted(self, minimal_config):
        """Test verbose parameter is accessible.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        assert loader.parameters.get('verbose') is False
    
    def test_parameters_with_different_values(self, test_config):
        """Test parameters with different values.
        
        Parameters
        ----------
        test_config : DictConfig
            Test configuration with different parameters.
        """
        loader = ATLASTopTaggingDatasetLoader(test_config)
        assert loader.parameters.get('split') == 'test'
        assert loader.parameters.get('subset') == 0.01
        assert loader.parameters.get('max_constituents') == 50
        assert loader.parameters.get('use_high_level') is False
        assert loader.parameters.get('verbose') is True
    
    def test_default_split_parameter(self):
        """Test default split parameter."""
        config = OmegaConf.create({
            'data_dir': '/tmp/test'
        })
        loader = ATLASTopTaggingDatasetLoader(config)
        # Should use default value in _initialize_dataset
        assert loader.parameters.get('split', 'train') == 'train'
    
    def test_default_subset_parameter(self):
        """Test default subset parameter."""
        config = OmegaConf.create({
            'data_dir': '/tmp/test'
        })
        loader = ATLASTopTaggingDatasetLoader(config)
        # Should use default value in _initialize_dataset
        assert loader.parameters.get('subset', 0.01) == 0.01


class TestATLASLoaderInitializeDataset:
    """Test 21-25: _initialize_dataset method."""
    
    @patch('topobench.data.loaders.pointcloud.atlas_top_tagging_loader.ATLASTopTaggingDataset')
    def test_initialize_dataset_creates_instance(self, mock_dataset_class, minimal_config):
        """Test _initialize_dataset creates dataset instance.
        
        Parameters
        ----------
        mock_dataset_class : Mock
            Mocked dataset class.
        minimal_config : DictConfig
            Minimal configuration fixture.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        loader._initialize_dataset()
        mock_dataset_class.assert_called_once()
    
    @patch('topobench.data.loaders.pointcloud.atlas_top_tagging_loader.ATLASTopTaggingDataset')
    def test_initialize_dataset_passes_split(self, mock_dataset_class, minimal_config):
        """Test split parameter is passed to dataset.
        
        Parameters
        ----------
        mock_dataset_class : Mock
            Mocked dataset class.
        minimal_config : DictConfig
            Minimal configuration fixture.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        loader._initialize_dataset()
        call_kwargs = mock_dataset_class.call_args.kwargs
        assert call_kwargs['split'] == 'train'
    
    @patch('topobench.data.loaders.pointcloud.atlas_top_tagging_loader.ATLASTopTaggingDataset')
    def test_initialize_dataset_passes_subset(self, mock_dataset_class, minimal_config):
        """Test subset parameter is passed to dataset.
        
        Parameters
        ----------
        mock_dataset_class : Mock
            Mocked dataset class.
        minimal_config : DictConfig
            Minimal configuration fixture.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        loader._initialize_dataset()
        call_kwargs = mock_dataset_class.call_args.kwargs
        assert call_kwargs['subset'] == 0.01
    
    @patch('topobench.data.loaders.pointcloud.atlas_top_tagging_loader.ATLASTopTaggingDataset')
    def test_initialize_dataset_passes_max_constituents(self, mock_dataset_class, minimal_config):
        """Test max_constituents parameter is passed to dataset.
        
        Parameters
        ----------
        mock_dataset_class : Mock
            Mocked dataset class.
        minimal_config : DictConfig
            Minimal configuration fixture.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        loader._initialize_dataset()
        call_kwargs = mock_dataset_class.call_args.kwargs
        assert call_kwargs['max_constituents'] == 80
    
    @patch('topobench.data.loaders.pointcloud.atlas_top_tagging_loader.ATLASTopTaggingDataset')
    def test_initialize_dataset_passes_use_high_level(self, mock_dataset_class, minimal_config):
        """Test use_high_level parameter is passed to dataset.
        
        Parameters
        ----------
        mock_dataset_class : Mock
            Mocked dataset class.
        minimal_config : DictConfig    
            Minimal configuration fixture.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        loader._initialize_dataset()
        call_kwargs = mock_dataset_class.call_args.kwargs
        assert call_kwargs['use_high_level'] is True


class TestATLASLoaderRedefineDataDir:
    """Test 26-28: _redefine_data_dir method."""
    
    def test_redefine_data_dir_returns_path(self, minimal_config):
        """Test _redefine_data_dir returns Path object.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        mock_dataset = Mock()
        mock_dataset.processed_dir = '/tmp/test/processed'
        
        result = loader._redefine_data_dir(mock_dataset)
        assert isinstance(result, Path)
    
    def test_redefine_data_dir_uses_processed_dir(self, minimal_config):
        """Test _redefine_data_dir uses dataset.processed_dir.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        mock_dataset = Mock()
        mock_dataset.processed_dir = '/tmp/test/processed'
        
        result = loader._redefine_data_dir(mock_dataset)
        assert str(result) == '/tmp/test/processed'
    
    def test_redefine_data_dir_converts_to_path_object(self, minimal_config):
        """Test _redefine_data_dir converts string to Path.
        
        Parameters
        ----------
        minimal_config : DictConfig
            Minimal configuration.
        """
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        mock_dataset = Mock()
        mock_dataset.processed_dir = '/tmp/different/path'
        
        result = loader._redefine_data_dir(mock_dataset)
        assert result == Path('/tmp/different/path')


class TestATLASLoaderLoadDataset:
    """Test 29-32: load_dataset method."""
    
    @patch('topobench.data.loaders.pointcloud.atlas_top_tagging_loader.ATLASTopTaggingDataset')
    def test_load_dataset_calls_initialize(self, mock_dataset_class, minimal_config):
        """Test load_dataset calls _initialize_dataset.
        
        Parameters
        ----------
        mock_dataset_class : Mock
            Mocked dataset class.
        minimal_config : DictConfig
            Minimal configuration fixture.
        """
        mock_instance = Mock()
        mock_instance.processed_dir = '/tmp/processed'
        mock_dataset_class.return_value = mock_instance
        
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        result = loader.load_dataset()
        
        mock_dataset_class.assert_called_once()
    
    @patch('topobench.data.loaders.pointcloud.atlas_top_tagging_loader.ATLASTopTaggingDataset')
    def test_load_dataset_returns_dataset(self, mock_dataset_class, minimal_config):
        """Test load_dataset returns dataset instance.
        
        Parameters
        ----------
        mock_dataset_class : Mock
            Mocked dataset class.
        minimal_config : DictConfig
            Minimal configuration fixture.
        """
        mock_instance = Mock()
        mock_instance.processed_dir = '/tmp/processed'
        mock_dataset_class.return_value = mock_instance
        
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        result = loader.load_dataset()
        
        assert result is not None
        assert result == mock_instance
    
    @patch('topobench.data.loaders.pointcloud.atlas_top_tagging_loader.ATLASTopTaggingDataset')
    def test_load_dataset_sets_data_dir(self, mock_dataset_class, minimal_config):
        """Test load_dataset sets data_dir attribute.
        
        Parameters
        ----------
        mock_dataset_class : Mock
            Mocked dataset class.
        minimal_config : DictConfig    
            Minimal configuration fixture.
        """
        mock_instance = Mock()
        mock_instance.processed_dir = '/tmp/processed'
        mock_dataset_class.return_value = mock_instance
        
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        loader.load_dataset()
        
        assert hasattr(loader, 'data_dir')
        assert loader.data_dir == Path('/tmp/processed')
    
    @patch('topobench.data.loaders.pointcloud.atlas_top_tagging_loader.ATLASTopTaggingDataset')
    def test_load_dataset_returns_correct_type(self, mock_dataset_class, minimal_config):
        """Test load_dataset returns Dataset-like object.
        
        Parameters
        ----------
        mock_dataset_class : Mock
            Mocked dataset class.
        minimal_config : DictConfig
            Minimal configuration fixture.
        """
        from torch_geometric.data import Dataset
        mock_instance = Mock(spec=Dataset)
        mock_instance.processed_dir = '/tmp/processed'
        mock_dataset_class.return_value = mock_instance
        
        loader = ATLASTopTaggingDatasetLoader(minimal_config)
        result = loader.load_dataset()
        
        # Check that result has Dataset-like interface
        assert result == mock_instance


class TestATLASLoaderIntegration:
    """Test 33-35: Integration and edge cases."""
    
    def test_loader_with_omegaconf_dict_config(self):
        """Test loader works with OmegaConf DictConfig."""
        config = OmegaConf.create({
            'data_dir': '/tmp/test',
            'split': 'train',
            'subset': 0.01,
            'max_constituents': 80,
            'use_high_level': True,
            'verbose': False
        })
        loader = ATLASTopTaggingDatasetLoader(config)
        assert isinstance(loader.parameters, (dict, DictConfig))
    
    def test_loader_docstring_has_parameters_section(self):
        """Test loader docstring documents parameters."""
        docstring = ATLASTopTaggingDatasetLoader.__doc__
        assert 'Parameters' in docstring or 'parameters' in docstring
    
    def test_loader_docstring_has_returns_section(self):
        """Test loader docstring documents return values."""
        docstring = ATLASTopTaggingDatasetLoader.load_dataset.__doc__
        assert docstring is not None
        assert 'Returns' in docstring or 'Dataset' in docstring