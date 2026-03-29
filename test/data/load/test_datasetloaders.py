"""Comprehensive test suite for all dataset loaders."""

import os
import pytest
import torch
import hydra
from pathlib import Path
from typing import List, Tuple, Dict, Any
from omegaconf import DictConfig
from topobench.data.preprocessor.preprocessor import PreProcessor
class TestLoaders:

    """Comprehensive test suite for all dataset loaders."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment before each test method."""
        # Existing setup code remains the same
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        base_dir = Path(__file__).resolve().parents[3]
        self.config_files = self._gather_config_files(base_dir)
        self.relative_config_dir = "../../../configs"
        self.test_splits = ['train', 'val', 'test']

    # Existing helper methods remain the same
    def _gather_config_files(self, base_dir: Path) -> List[str]:
        """Gather all relevant config files.
        
        Parameters
        ----------
        base_dir : Path
            Base directory to start searching for config files.

        Returns
        -------
        List[str]
          List of config file paths.
        """
        config_files = []
        config_base_dir = base_dir / "configs/dataset"
        # Below the datasets that have some default transforms manually overriten with no_transform,
        exclude_datasets = {"karate_club.yaml",
                            # Below the datasets that have some default transforms with we manually overriten with no_transform,
                            # due to lack of default transform for domain2domain
                            "REDDIT-BINARY.yaml", "IMDB-MULTI.yaml", "IMDB-BINARY.yaml", #"ZINC.yaml"
                            "ogbg-molpcba.yaml", "manual_dataset.yaml" # "ogbg-molhiv.yaml"
                            }
        
        # Below the datasets that takes quite some time to load and process                            
        self.long_running_datasets = {"mantra_name.yaml", "mantra_orientation.yaml", "mantra_genus.yaml", "mantra_betti_numbers.yaml"}

        
        for dir_path in config_base_dir.iterdir():
            curr_dir = str(dir_path).split('/')[-1]
            if dir_path.is_dir():
                config_files.extend([
                    (curr_dir, f.name) for f in dir_path.glob("*.yaml")
                    if f.name not in exclude_datasets
                ])
        return config_files

    def _load_dataset(self, data_domain: str, config_file: str) -> Tuple[Any, Dict]:
        """Load dataset with given config file.

        Parameters
        ----------
        data_domain : str
            Name of the data domain.
        config_file : str
          Name of the config file.
        
        Returns
        -------
        Tuple[Any, Dict]
          Tuple containing the dataset and dataset directory.
        """
        with hydra.initialize(
            version_base="1.3",
            config_path=self.relative_config_dir,
            job_name="run"
        ):
            print('Current config file: ', config_file)
            parameters = hydra.compose(
                config_name="run.yaml",
                overrides=[f"dataset={data_domain}/{config_file}", f"model=graph/gat"], 
                return_hydra_config=True, 
            )
            dataset_loader = hydra.utils.instantiate(parameters.dataset.loader)
            print(repr(dataset_loader))

            if config_file in self.long_running_datasets:
                dataset, data_dir = dataset_loader.load(slice=100)
            else:
                dataset, data_dir = dataset_loader.load()
            return dataset, data_dir

    #def test_dataset_loading_states(self):
     #   """Test different states and scenarios during dataset loading."""
      #  for config_data in self.config_files:
       #     data_domain, config_file = config_data
        #    dataset, _ = self._load_dataset(data_domain, config_file)
            
            # Test dataset size and dimensions
         #   if hasattr(dataset, "data"):
          #      assert dataset.data.x.size(0) > 0, "Empty node features"
           #     assert dataset.data.y.size(0) > 0, "Empty labels"
            
            # Below brakes with manual dataset
            # else: 
            #     assert dataset[0].x.size(0) > 0, "Empty node features"
            #     assert dataset[0].y.size(0) > 0, "Empty labels"
            
            # Test node feature dimensions
           # if hasattr(dataset, 'num_node_features'):
            #    assert dataset.data.x.size(1) == dataset.num_node_features
            
            # Below brakes with manual dataset
            # # Test label dimensions
            # if hasattr(dataset, 'num_classes'):
            #     assert torch.max(dataset.data.y) < dataset.num_classes

            #repr(dataset)
    
       


    
                

