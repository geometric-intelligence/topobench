"""Unit tests for config resolvers."""

import pytest
from omegaconf import OmegaConf
import hydra
from topobench.utils.config_resolvers import (
    infer_in_channels,
    infer_num_cell_dimensions,
    infer_topotune_num_cell_dimensions,
    get_default_metrics,
    get_default_trainer,
    get_default_transform,
    get_flattened_channels,
    get_non_relational_out_channels,
    get_monitor_metric,
    get_monitor_mode,
    get_required_lifting,
    check_pses_in_transforms,
)

class TestConfigResolvers:
    """Test config resolvers."""

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.dataset_config_1 = OmegaConf.load("configs/dataset/graph/MUTAG.yaml")
        self.dataset_config_2 = OmegaConf.load("configs/dataset/graph/cocitation_cora.yaml")
        self.cliq_lift_transform = OmegaConf.load("configs/transforms/liftings/graph2simplicial/clique.yaml")
        self.feature_lift_transform = OmegaConf.load("configs/transforms/feature_liftings/concatenate.yaml")
        hydra.initialize(version_base="1.3", config_path="../../configs", job_name="job")
        
    def test_get_default_trainer(self):
        """Test get_default_trainer."""
        out = get_default_trainer()
        assert isinstance(out, str)
        
    def test_get_default_metrics(self):
        """Test get_default_metrics."""
        out = get_default_metrics("classification")
        assert out == ["accuracy", "precision", "recall", "auroc"]

        out = get_default_metrics("regression")
        assert out == ["mse", "mae"]

        with pytest.raises(ValueError, match="Invalid task") as e:
            get_default_metrics("some_task")

    def test_get_default_transform(self):
        """Test get_default_transform."""
        out = get_default_transform("graph/MUTAG", "graph/gat")
        assert out == "no_transform"
        
        out = get_default_transform("graph/MUTAG", "non_relational/mlp")
        assert out == "no_transform"

        out = get_default_transform("graph/MUTAG", "cell/can")
        assert out == "liftings/graph2cell_default"

        out = get_default_transform("graph/ZINC", "cell/can")
        assert out == "dataset_defaults/ZINC"

        out = get_default_transform("graph/MUTAG", "graph/gps")
        assert out == "model_defaults/gps"

        out = get_default_transform("graph/ZINC", "graph/gps")
        assert out == "dataset_model_defaults/ZINC_gps"


    def test_get_flattened_channels(self):
        """Test get_flattened_channels."""
        out = get_flattened_channels(10, 5)
        assert out == 50

    def test_non_relational_out_channels(self):
        """Test get_non_relational_out_channels."""
        out = get_non_relational_out_channels(10, 5, "node")
        assert out == 50

        out = get_non_relational_out_channels(10, 5, "graph")
        assert out == 5

        with pytest.raises(ValueError, match="Invalid task level") as e:
            get_non_relational_out_channels(10, 5, "some_task")

    def test_get_required_lifting(self):
        """Test get_required_lifting."""
        out = get_required_lifting("graph", "graph/gat")
        assert out == "no_lifting"

        out = get_required_lifting("graph", "cell/can")
        assert out == "graph2cell_default"
    
    def test_get_monitor_metric(self):
        """Test get_monitor_metric."""
        out = get_monitor_metric("classification", "F1")
        assert out == "val/F1" 
        
        with pytest.raises(ValueError, match="Invalid task") as e:
            get_monitor_metric("mix", "F1")

    def test_get_monitor_mode(self):
        """Test get_monitor_mode."""
        out = get_monitor_mode("regression")
        assert out == "min"
        
        out = get_monitor_mode("classification")
        assert out == "max"
        
        with pytest.raises(ValueError, match="Invalid task") as e:
            get_monitor_mode("mix")

    def test_infer_in_channels(self):
        """Test infer_in_channels."""
        in_channels = infer_in_channels(self.dataset_config_1, self.cliq_lift_transform)
        assert in_channels == [7]
        
        in_channels = infer_in_channels(self.dataset_config_2, None)
        assert in_channels == [1433]
        
        cfg = hydra.compose(config_name="run.yaml", overrides=["model=simplicial/topotune", "dataset=graph/MUTAG"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [7,4,4,4]
        
        cfg = hydra.compose(config_name="run.yaml", overrides=["model=simplicial/topotune", "dataset=graph/MUTAG", "dataset.parameters.preserve_edge_attr_if_lifted=False"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [7,7,7,7]
        
        cfg = hydra.compose(config_name="run.yaml", overrides=["model=simplicial/topotune", "dataset=graph/MUTAG", "dataset.parameters.preserve_edge_attr_if_lifted=False", "transforms.graph2simplicial_lifting.feature_lifting=Concatenation"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [7,14,42,168]
        
        cfg = hydra.compose(config_name="run.yaml", overrides=["model=simplicial/topotune", "dataset=graph/MUTAG", "transforms.graph2simplicial_lifting.feature_lifting=Concatenation"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [7,4,4,4]
        
        cfg = hydra.compose(config_name="run.yaml", overrides=["model=simplicial/topotune", "dataset=graph/cocitation_cora", "transforms.graph2simplicial_lifting.feature_lifting=Concatenation"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [1433,2866,8598,34392]
        
        cfg = hydra.compose(config_name="run.yaml", overrides=["model=simplicial/topotune", "dataset=graph/cocitation_cora"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [1433,1433,1433,1433]
        
        cfg = hydra.compose(config_name="run.yaml", overrides=["model=graph/gcn", "dataset=simplicial/mantra_orientation"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [1]

        cfg = hydra.compose(config_name="run.yaml", overrides=["model=simplicial/scn", "dataset=graph/cocitation_cora"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [1433,1433,1433,1433]

    def test_infer_num_cell_dimensions(self):
        """Test infer_num_cell_dimensions."""
        out = infer_num_cell_dimensions(None, [7, 7, 7])
        assert out == 3

        out = infer_num_cell_dimensions([1, 2, 3], [7, 7])
        assert out == 3

    def test_infer_topotune_num_cell_dimensions(self):
        """Test infer_topotune_num_cell_dimensions."""
        neighborhoods = ["up_adjacency-1"]
        out = infer_topotune_num_cell_dimensions(neighborhoods)
        assert out == 3

        neighborhoods = ["up_incidence-0"]
        out = infer_topotune_num_cell_dimensions(neighborhoods)
        assert out == 2

        neighborhoods = ["down_incidence-2"]
        out = infer_topotune_num_cell_dimensions(neighborhoods)
        assert out == 3
        
    def test_get_default_metrics(self):
        """Test get_default_metrics."""
        out = get_default_metrics("classification", ["accuracy", "precision"])
        assert out == ["accuracy", "precision"]
        
        out = get_default_metrics("classification")
        assert out == ["accuracy", "precision", "recall", "auroc"]

        out = get_default_metrics("regression")
        assert out == ["mse", "mae"]

        with pytest.raises(ValueError, match="Invalid task") as e:
            get_default_metrics("some_task")

    def test_check_pses_in_transforms_empty(self):
        """Test check_pses_in_transforms with no encodings."""
        transforms = OmegaConf.create({})
        result = check_pses_in_transforms(transforms)
        assert result == 0
        
    def test_single_transform_lappe_with_eigenvalues(self):
        """Test single transform with LapPE including eigenvalues."""
        transforms = OmegaConf.create({
            "transform_name": "LapPE",
            "include_eigenvalues": True,
            "max_pe_dim": 8
        })
        result = check_pses_in_transforms(transforms)
        assert result == 16  # 8 * 2

    def test_single_transform_lappe_without_eigenvalues(self):
        """Test single transform with LapPE without eigenvalues."""
        transforms = OmegaConf.create({
            "transform_name": "LapPE",
            "include_eigenvalues": False,
            "max_pe_dim": 8
        })
        result = check_pses_in_transforms(transforms)
        assert result == 8

    def test_single_transform_rwse(self):
        """Test single transform with RWSE."""
        transforms = OmegaConf.create({
            "transform_name": "RWSE",
            "max_pe_dim": 16
        })
        result = check_pses_in_transforms(transforms)
        assert result == 16

    def test_check_pses_in_transforms_lappe_only(self):
        """Test check_pses_in_transforms with only LapPE encoding."""
        # LapPE without eigenvalues
        transforms = OmegaConf.create({
            "LapPE": {
                "max_pe_dim": 8,
                "include_eigenvalues": False,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 8

    def test_check_pses_in_transforms_lappe_with_eigenvalues(self):
        """Test check_pses_in_transforms with LapPE including eigenvalues."""
        transforms = OmegaConf.create({
            "LapPE": {
                "max_pe_dim": 8,
                "include_eigenvalues": True,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 16  # 8 * 2

    def test_check_pses_in_transforms_rwse_only(self):
        """Test check_pses_in_transforms with only RWSE encoding."""
        transforms = OmegaConf.create({
            "RWSE": {
                "max_pe_dim": 8,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 8

    def test_check_pses_in_transforms_combined_pses_lappe_rwse(self):
        """Test check_pses_in_transforms with CombinedPSEs containing both LapPE and RWSE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": False,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 12  # 8 + 4

    def test_check_pses_in_transforms_combined_pses_with_eigenvalues(self):
        """Test check_pses_in_transforms with CombinedPSEs where LapPE includes eigenvalues."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": True,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 20  # (8 * 2) + 4

    def test_check_pses_in_transforms_combined_pses_lappe_only(self):
        """Test check_pses_in_transforms with CombinedPSEs containing only LapPE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 16,
                        "include_eigenvalues": False,
                        "concat_to_x": False
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 16

    def test_check_pses_in_transforms_combined_pses_rwse_only(self):
        """Test check_pses_in_transforms with CombinedPSEs containing only RWSE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["RWSE"],
                "parameters": {
                    "RWSE": {
                        "max_pe_dim": 12,
                        "concat_to_x": False
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 12

    def test_check_pses_in_transforms_multiple_separate_transforms(self):
        """Test check_pses_in_transforms with multiple separate encoding transforms."""
        transforms = OmegaConf.create({
            "LapPE_1": {
                "max_pe_dim": 8,
                "include_eigenvalues": False,
                "concat_to_x": True
            },
            "RWSE_1": {
                "max_pe_dim": 4,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 12  # 8 + 4

    def test_check_pses_in_transforms_multiple_lappe_transforms(self):
        """Test check_pses_in_transforms with multiple LapPE transforms."""
        transforms = OmegaConf.create({
            "LapPE_first": {
                "max_pe_dim": 8,
                "include_eigenvalues": False,
                "concat_to_x": True
            },
            "LapPE_second": {
                "max_pe_dim": 4,
                "include_eigenvalues": True,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 16  # 8 + (4 * 2)

    def test_check_pses_in_transforms_mixed_transforms(self):
        """Test check_pses_in_transforms with mixed transform types."""
        transforms = OmegaConf.create({
            "some_other_transform": {
                "param1": "value1"
            },
            "LapPE": {
                "max_pe_dim": 8,
                "include_eigenvalues": False,
                "concat_to_x": True
            },
            "another_transform": {
                "param2": "value2"
            },
            "RWSE": {
                "max_pe_dim": 4,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 12  # 8 + 4

    def test_check_pses_in_transforms_combined_and_separate(self):
        """Test check_pses_in_transforms with both CombinedPSEs and separate encodings."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": False,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    }
                }
            },
            "LapPE_extra": {
                "max_pe_dim": 2,
                "include_eigenvalues": False,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 14  # (8 + 4) + 2

    def test_check_pses_in_transforms_different_dimensions(self):
        """Test check_pses_in_transforms with various dimension sizes."""
        # Test with different max_pe_dim values
        for dim in [1, 2, 4, 8, 16, 32]:
            transforms = OmegaConf.create({
                "RWSE": {
                    "max_pe_dim": dim,
                    "concat_to_x": True
                }
            })
            result = check_pses_in_transforms(transforms)
            assert result == dim

    def test_check_pses_in_transforms_combined_pses_empty_encodings(self):
        """Test check_pses_in_transforms with CombinedPSEs but empty encodings list."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": [],
                "parameters": {}
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 0

    def test_check_pses_in_transforms_complex_scenario(self):
        """Test check_pses_in_transforms with a complex scenario."""
        transforms = OmegaConf.create({
            "preprocessing": {
                "some_param": "value"
            },
            "CombinedPSEs_1": {
                "encodings": ["LapPE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 16,
                        "include_eigenvalues": True,
                        "concat_to_x": True
                    }
                }
            },
            "other_transform": {
                "param": "value"
            },
            "RWSE_custom": {
                "max_pe_dim": 8,
                "concat_to_x": False
            },
            "CombinedPSEs_2": {
                "encodings": ["RWSE"],
                "parameters": {
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 44  # (16 * 2) + 8 + 4

    @pytest.mark.parametrize("max_pe_dim,include_eigenvalues,expected", [
        (4, False, 4),
        (4, True, 8),
        (8, False, 8),
        (8, True, 16),
        (16, False, 16),
        (16, True, 32),
    ])
    def test_check_pses_in_transforms_lappe_parametrized(self, max_pe_dim, include_eigenvalues, expected):
        """Parametrized test for LapPE with different configurations.
        
        Parameters
        ----------
        max_pe_dim : int
            Maximum positional encoding dimension for LapPE.
        include_eigenvalues : bool
            Whether to include eigenvalues in the encoding.
        expected : int
            Expected dimension of the positional encoding.
        """
        transforms = OmegaConf.create({
            "LapPE": {
                "max_pe_dim": max_pe_dim,
                "include_eigenvalues": include_eigenvalues,
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == expected

    @pytest.mark.parametrize("lappe_dim,rwse_dim,expected", [
        (4, 4, 8),
        (8, 4, 12),
        (4, 8, 12),
        (16, 8, 24),
        (8, 16, 24),
    ])
    def test_check_pses_in_transforms_combined_parametrized(self, lappe_dim, rwse_dim, expected):
        """Parametrized test for CombinedPSEs with different dimension combinations.
        
        Parameters
        ----------
        lappe_dim : int
            Dimension for LapPE encoding.
        rwse_dim : int
            Dimension for RWSE encoding.
        expected : int
            Expected combined dimension of both encodings.
        """
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": lappe_dim,
                        "include_eigenvalues": False,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": rwse_dim,
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == expected
