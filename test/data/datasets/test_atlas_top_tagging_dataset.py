"""Tests for ATLAS Top Tagging Dataset."""

import pytest
from pathlib import Path


class TestATLASDatasetImport:
    """Test dataset can be imported."""

    def test_can_import_dataset_class(self):
        """Test ATLASTopTaggingDataset can be imported."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert ATLASTopTaggingDataset is not None

    def test_class_inherits_from_in_memory_dataset(self):
        """Test class inherits from InMemoryDataset."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        from torch_geometric.data import InMemoryDataset
        assert issubclass(ATLASTopTaggingDataset, InMemoryDataset)


class TestATLASDatasetClassAttributes:
    """Test dataset class attributes."""

    def test_has_urls_attribute(self):
        """Test class has URLS attribute."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'URLS')
        assert isinstance(ATLASTopTaggingDataset.URLS, dict)

    def test_has_constituent_branches(self):
        """Test class has CONSTITUENT_BRANCHES."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'CONSTITUENT_BRANCHES')
        assert len(ATLASTopTaggingDataset.CONSTITUENT_BRANCHES) == 4

    def test_has_high_level_branches(self):
        """Test class has HIGH_LEVEL_BRANCHES."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'HIGH_LEVEL_BRANCHES')
        assert len(ATLASTopTaggingDataset.HIGH_LEVEL_BRANCHES) == 15

    def test_has_jet_branches(self):
        """Test class has JET_BRANCHES."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'JET_BRANCHES')
        assert len(ATLASTopTaggingDataset.JET_BRANCHES) == 4

    def test_constituent_branches_content(self):
        """Test CONSTITUENT_BRANCHES has expected content."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        branches = ATLASTopTaggingDataset.CONSTITUENT_BRANCHES
        # Check for actual ATLAS branch names
        assert 'fjet_clus_pt' in branches
        assert 'fjet_clus_eta' in branches
        assert 'fjet_clus_phi' in branches
        assert 'fjet_clus_E' in branches

    def test_urls_are_strings(self):
        """Test URLS values are strings."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        for url in ATLASTopTaggingDataset.URLS.values():
            assert isinstance(url, str)


class TestATLASDatasetMethods:
    """Test dataset methods exist."""

    def test_has_download_method(self):
        """Test download method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'download')

    def test_has_process_method(self):
        """Test process method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'process')

    def test_has_preprocess_method(self):
        """Test _preprocess method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, '_preprocess')

    def test_has_load_h5_flexible_method(self):
        """Test _load_h5_file_flexible method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, '_load_h5_file_flexible')

    def test_has_expected_filenames_method(self):
        """Test _expected_filenames method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, '_expected_filenames')

    def test_has_total_files_for_split_method(self):
        """Test _total_files_for_split method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, '_total_files_for_split')


class TestATLASDatasetProperties:
    """Test dataset properties."""

    def test_has_num_classes_property(self):
        """Test num_classes property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'num_classes')

    def test_has_num_features_property(self):
        """Test num_features property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'num_features')

    def test_has_num_high_level_features_property(self):
        """Test num_high_level_features property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'num_high_level_features')

    def test_has_raw_file_names_property(self):
        """Test raw_file_names property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'raw_file_names')

    def test_has_processed_file_names_property(self):
        """Test processed_file_names property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'processed_file_names')

    def test_has_pre_processed_path_property(self):
        """Test pre_processed_path property exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'pre_processed_path')

    def test_has_stats_method(self):
        """Test stats method exists."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        assert hasattr(ATLASTopTaggingDataset, 'stats')


class TestATLASDatasetInitialization:
    """Test dataset initialization and parameters."""

    def test_subset_parameter_validation_too_high(self):
        """Test subset validation rejects values > 1.0."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        with pytest.raises(AssertionError):
            ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=1.5)

    def test_subset_parameter_validation_too_low(self):
        """Test subset validation rejects values <= 0."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        with pytest.raises(AssertionError):
            ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.0)

    def test_split_parameter_validation(self):
        """Test split parameter validation."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        with pytest.raises(AssertionError):
            ATLASTopTaggingDataset(root='/tmp/nonexistent', split='invalid', subset=0.003)

    def test_max_constituents_parameter_stored(self, monkeypatch):
        """Test max_constituents parameter is stored.
        
        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest fixture for mocking.
        """
        # Mock torch.load to avoid file I/O
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003, max_constituents=50)
        assert dataset.max_constituents == 50

    def test_use_high_level_parameter_stored(self):
        """Test use_high_level parameter is stored."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003, use_high_level=False)
        assert dataset.use_high_level == False

    def test_verbose_parameter_stored(self):
        """Test verbose parameter is stored."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003, verbose=True)
        assert dataset.verbose == True


class TestATLASDatasetHelperMethods:
    """Test helper methods."""

    def test_total_files_for_split_train(self):
        """Test _total_files_for_split returns 930 for train."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        # Create instance with train split to test the method
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003)
        assert dataset._total_files_for_split() == 930

    def test_total_files_for_split_test(self):
        """Test _total_files_for_split returns 100 for test."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        # Create instance with test split to test the method
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='test', subset=0.003)
        assert dataset._total_files_for_split() == 100

    def test_expected_filenames_format(self):
        """Test _expected_filenames returns correct format."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003)
        filenames = dataset._expected_filenames()
        assert len(filenames) > 0
        assert all('.h5.gz' in f for f in filenames)

    def test_expected_filenames_respects_subset(self):
        """Test _expected_filenames respects subset parameter."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        small_dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003)
        large_dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.005)
        small = small_dataset._expected_filenames()
        large = large_dataset._expected_filenames()
        assert len(small) < len(large)

    def test_processed_file_names_format(self):
        """Test processed_file_names has correct format."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003)
        assert 'atlas_top_tagging' in dataset.processed_file_names[0]
        assert '.pt' in dataset.processed_file_names[0]


class TestATLASDatasetPropertiesValues:
    """Test property values."""

    def test_num_classes_equals_two(self):
        """Test num_classes returns 2."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003)
        assert dataset.num_classes == 2

    def test_num_features_equals_four(self):
        """Test num_features returns 4."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003)
        assert dataset.num_features == 4

    def test_num_high_level_features_with_flag_true(self):
        """Test num_high_level_features returns 15 when use_high_level=True."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003, use_high_level=True)
        assert dataset.num_high_level_features == 15

    def test_num_high_level_features_with_flag_false(self):
        """Test num_high_level_features returns 0 when use_high_level=False."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003, use_high_level=False)
        assert dataset.num_high_level_features == 0

    def test_raw_file_names_includes_split_directory(self):
        """Test raw_file_names includes split directory."""
        from topobench.data.datasets.atlas_top_tagging_dataset import ATLASTopTaggingDataset
        dataset = ATLASTopTaggingDataset(root='/tmp/nonexistent', split='train', subset=0.003)
        assert 'train' in str(dataset.raw_file_names)


class TestATLASDatasetDownload:
    """Extra tests for download() error handling."""

    def test_download_logs_failures_and_raises(self, tmp_path, monkeypatch, capsys):
        """Cover error messages and final FileNotFoundError when all downloads fail.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory provided by pytest.
        monkeypatch : pytest.MonkeyPatch
            MonkeyPatch fixture for mocking.
        capsys : pytest.CaptureFixture
            CaptureFixture for capturing stdout/stderr.
        """
        from topobench.data.datasets.atlas_top_tagging_dataset import (
            ATLASTopTaggingDataset,
        )
        import urllib.request
        import time

        dataset = ATLASTopTaggingDataset.__new__(ATLASTopTaggingDataset)
        dataset.split = "train"
        dataset.subset = 0.001
        # Set root; raw_dir is derived from it by the base class property
        dataset.root = str(tmp_path)

        def fake_urlretrieve(_url, _path):
            """Fake urlretrieve that always raises RuntimeError.

            Parameters
            ----------
            _url : str
                URL to download from.
            _path : str
                Path to save the file to.

            Raises
            ------
            RuntimeError
                Always raised to simulate download failure.
            """
            raise RuntimeError("network down")

        monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
        monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)

        # (1) per-file failure messages, (2) final FileNotFoundError when downloaded == 0
        with pytest.raises(FileNotFoundError):
            dataset.download()

        out = capsys.readouterr().out
        assert "Failed to download" in out
        assert "You can manually download from" in out



class TestATLASDatasetH5Loading:
    """Extra tests for flexible HDF5 loading."""

    def test_load_h5_file_uses_suffix_for_compression_flag(self, monkeypatch):
        """Cover is_compressed flag and delegation to _load_h5_file_flexible().
        
        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            MonkeyPatch fixture for mocking.
        """
        from topobench.data.datasets.atlas_top_tagging_dataset import (
            ATLASTopTaggingDataset,
        )

        calls: list[tuple[str, bool, int | None]] = []

        def fake_flexible(self, file_path, use_compressed=True, num_jets=None):
            """Fake _load_h5_file_flexible to capture parameters.
            
            Parameters
            ----------
            file_path : str
                Path to the HDF5 file.
            use_compressed : bool
                Whether the file is compressed.
            num_jets : int | None
                Number of jets to load.
            
            Returns
            -------
            dict
                Empty dictionary for testing.
            """
            calls.append((file_path, use_compressed, num_jets))
            return {}

        monkeypatch.setattr(
            ATLASTopTaggingDataset,
            "_load_h5_file_flexible",
            fake_flexible,
            raising=False,
        )

        dataset = ATLASTopTaggingDataset.__new__(ATLASTopTaggingDataset)

        # (5) .gz → use_compressed=True, .h5 → False
        dataset._load_h5_file("file.h5.gz", num_jets=3)
        dataset._load_h5_file("file.h5", num_jets=None)

        assert calls[0] == ("file.h5.gz", True, 3)
        assert calls[1] == ("file.h5", False, None)

    def test_load_h5_flexible_uncompressed_with_num_jets(self, tmp_path):
        """Cover uncompressed path + label slicing and total_jets logic.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory provided by pytest.
        """
        from topobench.data.datasets.atlas_top_tagging_dataset import (
            ATLASTopTaggingDataset,
        )
        import h5py
        import numpy as np

        h5_path = tmp_path / "dummy.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("labels", data=np.array([0, 1, 1], dtype="i4"))
            # Minimal constituent branches
            for name in (
                "fjet_clus_pt",
                "fjet_clus_eta",
                "fjet_clus_phi",
                "fjet_clus_E",
            ):
                f.create_dataset(name, data=np.ones((3, 4), dtype="f4"))

        dataset = ATLASTopTaggingDataset.__new__(ATLASTopTaggingDataset)
        dataset.use_high_level = False  # keep it minimal

        data = dataset._load_h5_file_flexible(
            str(h5_path),
            use_compressed=False,
            num_jets=2,
        )

        # (3) with h5py.File(..., "r") + (4) label slicing and total_jets
        assert list(data["labels"]) == [0, 1]
        assert data["fjet_clus_pt"].shape == (2, 4)


class TestATLASDatasetPreprocess:
    """Extra tests for _preprocess() fallback and error branches."""

    def test_preprocess_falls_back_to_h5_and_prints_progress(
        self, tmp_path, monkeypatch, capsys
    ):
        """Cover .h5 fallback, 'Found ... decompressed', and verbose 'Loaded ... jets'.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory provided by pytest.
        monkeypatch : pytest.MonkeyPatch
            MonkeyPatch fixture for mocking.
        capsys : pytest.CaptureFixture
            CaptureFixture for capturing stdout/stderr.
        """
        from pathlib import Path
        from topobench.data.datasets.atlas_top_tagging_dataset import (
            ATLASTopTaggingDataset,
        )
        import numpy as np
        import torch

        dataset = ATLASTopTaggingDataset.__new__(ATLASTopTaggingDataset)
        dataset.split = "train"
        dataset.subset = 0.5
        dataset.verbose = True
        dataset.root = str(tmp_path)

        # Create a dummy decompressed .h5 file under raw/<split>_nominal
        split_dir = Path(dataset.raw_dir) / "train_nominal"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train_nominal_000.h5").write_text("dummy")

        def fake_load_flexible(self, file_path, use_compressed=True, num_jets=None):
            """Fake _load_h5_file_flexible returning small arrays.

            Parameters
            ----------
            file_path : str
                Path to the HDF5 file.
            use_compressed : bool
                Whether the file is compressed.
            num_jets : int | None
                Number of jets to load.
            
            Returns
            -------
            dict
                Dictionary with small arrays for testing.
            """
            # Return small arrays so concatenation and trimming logic can run
            labels = np.array([0, 1], dtype="i4")
            feats = np.ones((2, 4), dtype="f4")
            return {
                "labels": labels,
                "fjet_clus_pt": feats,
                "fjet_clus_eta": feats,
                "fjet_clus_phi": feats,
                "fjet_clus_E": feats,
            }

        monkeypatch.setattr(
            ATLASTopTaggingDataset,
            "_load_h5_file_flexible",
            fake_load_flexible,
            raising=False,
        )

        # Avoid writing anything heavy to disk
        monkeypatch.setattr(
            "topobench.data.datasets.atlas_top_tagging_dataset.torch.save",
            lambda *_args, **_kwargs: None,
        )

        dataset._preprocess()
        out = capsys.readouterr().out

        # (6) .h5 fallback branch + verbose progress print
        assert "Found 1 decompressed .h5 files" in out
        assert "Loaded" in out

    def test_preprocess_raises_when_no_files_found(self, tmp_path):
        """Cover FileNotFoundError when no .h5.gz and no .h5 exist.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory provided by pytest.
        """
        from topobench.data.datasets.atlas_top_tagging_dataset import (
            ATLASTopTaggingDataset,
        )

        dataset = ATLASTopTaggingDataset.__new__(ATLASTopTaggingDataset)
        dataset.split = "train"
        dataset.subset = 0.1
        dataset.root = str(tmp_path)

        with pytest.raises(FileNotFoundError):
            dataset._preprocess()


class TestATLASDatasetProcess:
    """Extra tests for process() control flow."""

    def test_process_skips_jets_with_no_constituents(self, tmp_path, monkeypatch):
        """Cover 'num_valid == 0: continue' skip of empty jets.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory provided by pytest.
        monkeypatch : pytest.MonkeyPatch
            MonkeyPatch fixture for mocking.
        """
        from topobench.data.datasets.atlas_top_tagging_dataset import (
            ATLASTopTaggingDataset,
        )
        import torch

        dataset = ATLASTopTaggingDataset.__new__(ATLASTopTaggingDataset)
        dataset.split = "train"
        dataset.subset = 0.003
        dataset.max_constituents = 10
        dataset.use_high_level = False
        dataset.pre_filter = None
        dataset.pre_transform = None
        dataset.post_filter = None
        dataset.root = str(tmp_path)

        # Two jets: first empty (pt all 0), second has one valid constituent
        data_dict = {
            "labels": torch.tensor([0, 1]),
            "fjet_clus_pt": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            "fjet_clus_eta": torch.zeros((2, 2)),
            "fjet_clus_phi": torch.zeros((2, 2)),
            "fjet_clus_E": torch.ones((2, 2)),
        }

        # Pretend preprocessed file exists and contains our dict
        monkeypatch.setattr(
            "topobench.data.datasets.atlas_top_tagging_dataset.osp.exists",
            lambda _path: True,
        )
        monkeypatch.setattr(
            "topobench.data.datasets.atlas_top_tagging_dataset.torch.load",
            lambda _path, weights_only=False: data_dict,
        )
        monkeypatch.setattr(
            "topobench.data.datasets.atlas_top_tagging_dataset.knn_graph",
            lambda x, k, loop: torch.empty((2, 0), dtype=torch.long),
        )

        captured: dict[str, list] = {}

        def fake_collate(self, data_list):
            """Capture data_list passed to collate().

            Parameters
            ----------
            data_list : list
                List of data objects to collate.

            Returns
            -------
            tuple
                The original data_list and an empty dictionary.
            """
            captured["data_list"] = list(data_list)
            return data_list, {}

        monkeypatch.setattr(
            ATLASTopTaggingDataset,
            "collate",
            fake_collate,
            raising=False,
        )
        monkeypatch.setattr(
            "topobench.data.datasets.atlas_top_tagging_dataset.torch.save",
            lambda *_args, **_kwargs: None,
        )

        dataset.process()
        graphs = captured["data_list"]

        # (7) only the non-empty jet becomes a graph
        assert len(graphs) == 1

    def test_process_applies_pre_and_post_filters(self, tmp_path, monkeypatch):
        """Cover pre_filter, pre_transform, and post_filter branches.
        
        Parameters
        ----------
        tmp_path : pathlib.Path
            Temporary directory provided by pytest.
        monkeypatch : pytest.MonkeyPatch
            MonkeyPatch fixture for mocking.
        """
        from topobench.data.datasets.atlas_top_tagging_dataset import (
            ATLASTopTaggingDataset,
        )
        import torch

        dataset = ATLASTopTaggingDataset.__new__(ATLASTopTaggingDataset)
        dataset.split = "train"
        dataset.subset = 0.003
        dataset.max_constituents = 10
        dataset.use_high_level = False
        dataset.root = str(tmp_path)

        data_dict = {
            "labels": torch.tensor([0, 1]),
            "fjet_clus_pt": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "fjet_clus_eta": torch.zeros((2, 2)),
            "fjet_clus_phi": torch.zeros((2, 2)),
            "fjet_clus_E": torch.ones((2, 2)),
        }

        monkeypatch.setattr(
            "topobench.data.datasets.atlas_top_tagging_dataset.osp.exists",
            lambda _path: True,
        )
        monkeypatch.setattr(
            "topobench.data.datasets.atlas_top_tagging_dataset.torch.load",
            lambda _path, weights_only=False: data_dict,
        )
        monkeypatch.setattr(
            "topobench.data.datasets.atlas_top_tagging_dataset.knn_graph",
            lambda x, k, loop: torch.empty((2, 0), dtype=torch.long),
        )

        def pre_filter(data):
            """Keep only signal jets (label == 1).
            
            Parameters
            ----------
            data : Data
                Data object to filter.

            Returns
            -------
            bool
                True if data.y == 1, False otherwise.
            """
            return int(data.y.item()) == 1

        def pre_transform(data):
            """Set flag attribute to True.
            
            Parameters
            ----------
            data : Data
                Data object to transform.

            Returns
            -------
            Data
                Transformed data with flag attribute set to True.
            """
            data.flag = True
            return data

        def post_filter(data):
            """Keep only data with flag attribute set to True.
            
            Parameters
            ----------
            data : Data
                Data object to filter.

            Returns
            -------
            bool
                True if data.flag is True, False otherwise.
            """
            return getattr(data, "flag", False)

        dataset.pre_filter = pre_filter
        dataset.pre_transform = pre_transform
        dataset.post_filter = post_filter

        captured: dict[str, list] = {}

        def fake_collate(self, data_list):
            """Capture data_list passed to collate().

            Parameters
            ----------
            data_list : list
                List of data objects to collate.
            
            Returns
            -------
            tuple
                The original data_list and an empty dictionary.
            """
            captured["data_list"] = list(data_list)
            return data_list, {}

        monkeypatch.setattr(
            ATLASTopTaggingDataset,
            "collate",
            fake_collate,
            raising=False,
        )
        monkeypatch.setattr(
            "topobench.data.datasets.atlas_top_tagging_dataset.torch.save",
            lambda *_args, **_kwargs: None,
        )

        dataset.process()
        graphs = captured["data_list"]

        # (10), (11), (12): filter/transform branches executed, one graph survives
        assert len(graphs) == 1
        assert getattr(graphs[0], "flag", False)


class TestATLASDatasetStats:
    """Extra test for stats()."""

    def test_stats_prints_summary_and_distribution(self, monkeypatch, capsys):
        """Cover stats summary and class distribution prints.
        
        Parameters
        ----------
        monkeypatch : pytest.MonkeyPatch
            Pytest fixture for mocking.
        capsys : pytest.CaptureFixture
            Pytest fixture for capturing stdout/stderr.
        """
        from topobench.data.datasets.atlas_top_tagging_dataset import (
            ATLASTopTaggingDataset,
        )
        import torch

        dataset = ATLASTopTaggingDataset.__new__(ATLASTopTaggingDataset)
        dataset.split = "train"
        dataset.max_constituents = 80
        dataset.use_high_level = True

        monkeypatch.setattr(
            ATLASTopTaggingDataset,
            "__len__",
            lambda _self: 3,
            raising=False,
        )

        def fake_get(self, idx):
            """Create Dummy data with controlled labels and num_nodes.
            
            Parameters
            ----------
            idx : int
                Index of the data point to retrieve.

            Returns
            -------
            Dummy
                Dummy object with y and num_nodes attributes.
            """
            labels = [1, 0, 1]
            nodes = [10, 5, 15]

            class Dummy:
                """Dummy data object with y and num_nodes attributes.
                
                Parameters
                ----------
                label : int
                        Class label for the data point.
                num_nodes : int
                        Number of nodes (constituents) in the data point.
                """
                def __init__(self, label, num_nodes):
                    self.y = torch.tensor(label)
                    self.num_nodes = num_nodes

            return Dummy(labels[idx], nodes[idx])

        monkeypatch.setattr(
            ATLASTopTaggingDataset,
            "get",
            fake_get,
            raising=False,
        )

        dataset.stats()
        out = capsys.readouterr().out

        # (8) header + basic info
        assert "ATLAS Top Tagging Dataset" in out
        assert "Split: train" in out
        # (9) class distribution + average constituents
        assert "Signal jets: 2" in out
        assert "Background jets: 1" in out
        assert "Average constituents per jet: 10.0" in out