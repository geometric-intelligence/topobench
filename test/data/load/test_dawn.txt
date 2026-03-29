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
        # Only test DAWN dataset
        exclude_datasets = {"karate_club.yaml",
                            "REDDIT-BINARY.yaml", "IMDB-MULTI.yaml", "IMDB-BINARY.yaml",
                            "ogbg-molpcba.yaml", "manual_dataset.yaml",
                            "mantra_name.yaml", "mantra_orientation.yaml", "mantra_genus.yaml", 
                            "mantra_betti_numbers.yaml", "Mushroom.yaml", "NTU2012.yaml",
                            "zoo.yaml", "20newsgroup.yaml", "coauthorship_cora.yaml",
                            "coauthorship_dblp.yaml", "cocitation_citeseer.yaml",
                            "cocitation_cora.yaml", "cocitation_pubmed.yaml", "ModelNet40.yaml"}
        
        # Below the datasets that takes quite some time to load and process                            
        self.long_running_datasets = set()

        
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

    def test_dataset_loading_states(self):
        """Test different states and scenarios during dataset loading."""
        for config_data in self.config_files:
            data_domain, config_file = config_data
            dataset, _ = self._load_dataset(data_domain, config_file)
            
            # Test dataset size and dimensions
            if hasattr(dataset, "data"):
                assert dataset.data.x.size(0) > 0, "Empty node features"
                # DAWN dataset doesn't have labels, so y can be None
                if dataset.data.y is not None:
                    assert dataset.data.y.size(0) > 0, "Empty labels"
            
            # Below brakes with manual dataset
            # else: 
            #     assert dataset[0].x.size(0) > 0, "Empty node features"
            #     assert dataset[0].y.size(0) > 0, "Empty labels"
            
            # Test node feature dimensions
            if hasattr(dataset, 'num_node_features'):
                assert dataset.data.x.size(1) == dataset.num_node_features
            
            # Below brakes with manual dataset
            # # Test label dimensions
            # if hasattr(dataset, 'num_classes'):
            #     assert torch.max(dataset.data.y) < dataset.num_classes

            repr(dataset)


class TestDawnDatasetLoader:
    """Test suite for DawnDatasetLoader class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def loader_parameters(self, temp_dir):
        """Create loader parameters for testing."""
        from omegaconf import OmegaConf
        return OmegaConf.create(
            {
                "data_dir": temp_dir,
                "data_name": "DAWN",
                "data_domain": "hypergraph",
                "data_type": "temporal",
            }
        )

    @pytest.fixture
    def loader(self, loader_parameters):
        """Create a DawnDatasetLoader instance."""
        from topobench.data.loaders.hypergraph.dawn_dataset_loader import DawnDatasetLoader
        return DawnDatasetLoader(loader_parameters)

    def test_init(self, loader, loader_parameters):
        """Test DawnDatasetLoader initialization."""
        assert loader.parameters == loader_parameters
        assert str(loader.root_data_dir) == loader_parameters["data_dir"]

    def test_get_data_dir(self, loader):
        """Test get_data_dir method."""
        data_dir = loader.get_data_dir()
        expected = os.path.join(loader.root_data_dir, "DAWN")
        assert str(data_dir) == expected

    def test_initialize_dataset(self, loader):
        """Test _initialize_dataset method."""
        dataset = loader._initialize_dataset()
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        assert isinstance(dataset, DawnDataset)
        assert dataset.name == "DAWN"
        assert dataset.parameters == loader.parameters

    def test_load_dataset(self, loader, temp_dir):
        """Test load_dataset method."""
        # Create minimal test data files
        raw_dir = os.path.join(temp_dir, "DAWN", "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Create minimal test files
        with open(os.path.join(raw_dir, "DAWN-nverts.txt"), "w") as f:
            f.write("2\n1\n3\n")
        with open(os.path.join(raw_dir, "DAWN-simplices.txt"), "w") as f:
            f.write("1\n2\n3\n4\n5\n6\n")
        with open(os.path.join(raw_dir, "DAWN-times.txt"), "w") as f:
            f.write("8017\n8018\n8019\n")
        with open(os.path.join(raw_dir, "DAWN-node-labels.txt"), "w") as f:
            f.write("1 D00001 DRUG1\n2 D00002 DRUG2\n")

        # Process and load
        dataset = loader.load_dataset()
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        assert isinstance(dataset, DawnDataset)
        assert hasattr(dataset, "data")

    def test_repr(self, loader):
        """Test __repr__ method."""
        repr_str = repr(loader)
        assert "DawnDatasetLoader" in repr_str
        assert "parameters" in repr_str


class TestDawnHypergraphDataset:
    """Test suite for DawnHypergraphDataset class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def dataset_parameters(self):
        """Create dataset parameters for testing."""
        from omegaconf import OmegaConf
        return OmegaConf.create(
            {
                "num_features": 1,
                "num_classes": 2,
                "task": "classification",
            }
        )

    @pytest.fixture
    def raw_data_files(self, temp_dir):
        """Create raw data files for testing."""
        raw_dir = os.path.join(temp_dir, "DAWN", "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Create test data matching DAWN format
        # 3 simplices: {1,2}, {3}, {4,5,6} at times 8017, 8018, 8019
        with open(os.path.join(raw_dir, "DAWN-nverts.txt"), "w") as f:
            f.write("2\n1\n3\n")

        with open(os.path.join(raw_dir, "DAWN-simplices.txt"), "w") as f:
            f.write("1\n2\n3\n4\n5\n6\n")

        with open(os.path.join(raw_dir, "DAWN-times.txt"), "w") as f:
            f.write("8017\n8018\n8019\n")

        with open(os.path.join(raw_dir, "DAWN-node-labels.txt"), "w") as f:
            f.write("1 D00001 DRUG1\n2 D00002 DRUG2\n3 D00003 DRUG3\n")
            f.write("4 D00004 DRUG4\n5 D00005 DRUG5\n6 D00006 DRUG6\n")

        return raw_dir

    def test_init(self, temp_dir, dataset_parameters):
        """Test DawnHypergraphDataset initialization."""
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        dataset = DawnDataset(
            root=temp_dir, name="DAWN", parameters=dataset_parameters
        )
        assert dataset.name == "DAWN"
        assert dataset.parameters == dataset_parameters

    def test_raw_dir(self, temp_dir, dataset_parameters):
        """Test raw_dir property."""
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        dataset = DawnDataset(
            root=temp_dir, name="DAWN", parameters=dataset_parameters
        )
        expected = os.path.join(temp_dir, "DAWN", "raw")
        assert dataset.raw_dir == expected

    def test_processed_dir(self, temp_dir, dataset_parameters):
        """Test processed_dir property."""
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        dataset = DawnDataset(
            root=temp_dir, name="DAWN", parameters=dataset_parameters
        )
        expected = os.path.join(temp_dir, "DAWN", "processed")
        assert dataset.processed_dir == expected

    def test_raw_file_names(self, temp_dir, dataset_parameters):
        """Test raw_file_names property."""
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        dataset = DawnDataset(
            root=temp_dir, name="DAWN", parameters=dataset_parameters
        )
        expected = [
            "DAWN-nverts.txt",
            "DAWN-simplices.txt",
            "DAWN-times.txt",
            "DAWN-node-labels.txt",
        ]
        assert dataset.raw_file_names == expected

    def test_processed_file_names(self, temp_dir, dataset_parameters):
        """Test processed_file_names property."""
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        dataset = DawnDataset(
            root=temp_dir, name="DAWN", parameters=dataset_parameters
        )
        assert dataset.processed_file_names == "data.pt"

    def test_download(self, temp_dir, dataset_parameters, raw_data_files):
        """Test download method."""
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        dataset = DawnDataset(
            root=temp_dir, name="DAWN", parameters=dataset_parameters
        )

        # Files should already exist, download should not fail
        dataset.download()

        # Verify files exist
        for fname in dataset.raw_file_names:
            assert os.path.exists(os.path.join(dataset.raw_dir, fname))

    def test_process(self, temp_dir, dataset_parameters, raw_data_files):
        """Test process method."""
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        dataset = DawnDataset(
            root=temp_dir, name="DAWN", parameters=dataset_parameters
        )

        # Process the dataset
        dataset.process()

        # Verify processed file exists
        processed_path = os.path.join(dataset.processed_dir, "data.pt")
        assert os.path.exists(processed_path)

        # Verify data structure
        assert hasattr(dataset, "data")
        from torch_geometric.data import Data
        assert isinstance(dataset.data, Data)
        assert hasattr(dataset.data, "x")
        assert hasattr(dataset.data, "edge_index")
        assert hasattr(dataset.data, "edge_timestamps")

    def test_process_data_structure(self, temp_dir, dataset_parameters, raw_data_files):
        """Test that processed data has correct structure."""
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        dataset = DawnDataset(
            root=temp_dir, name="DAWN", parameters=dataset_parameters
        )
        dataset.process()

        data = dataset.data

        # Check node features
        assert data.x.shape[0] == 6  # 6 unique nodes (1-6)
        assert data.x.shape[1] == 1  # 1 feature dimension

        # Check edge_index (incidence matrix)
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0

        # Check timestamps
        assert len(data.edge_timestamps) == 3  # 3 simplices

        # Check node IDs are 0-indexed
        assert data.edge_index[0].min() >= 0
        assert data.edge_index[0].max() < 6

    def test_repr(self, temp_dir, dataset_parameters):
        """Test __repr__ method."""
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        dataset = DawnDataset(
            root=temp_dir, name="DAWN", parameters=dataset_parameters
        )
        repr_str = repr(dataset)
        # Check that repr contains key information
        assert "DAWN" in repr_str or "DawnDataset" in repr_str
        assert "root" in repr_str or "name" in repr_str

    def test_validate_and_normalize(self, temp_dir, dataset_parameters):
        """Test validate_and_normalize method."""
        from topobench.data.datasets.dawn_hypergraph_dataset import DawnDataset
        import torch
        dataset = DawnDataset(
            root=temp_dir, name="DAWN", parameters=dataset_parameters
        )
        
        # Test with valid inputs
        num_nodes = 5
        x = torch.ones((5, 1), dtype=torch.float)
        y = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)
        edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
        
        num_nodes, x, y = dataset.validate_and_normalize(num_nodes, x, y, edge_index)
        assert num_nodes == 5
        assert x is not None
        assert y is not None

