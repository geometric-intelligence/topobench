"""Unit tests for config resolvers."""

import pytest
from omegaconf import OmegaConf
import hydra
import torch
import os
from topobench.utils.config_resolvers import (
    register_all_resolvers,
    define_task_level,
    infer_in_channels,
    infer_num_cell_dimensions,
    infer_in_khop_feature_dim,
    infer_topotune_num_cell_dimensions,
    get_default_metrics,
    get_default_trainer,
    get_default_transform,
    get_flattened_channels,
    get_non_relational_out_channels,
    get_monitor_metric,
    get_monitor_mode,
    get_required_lifting,
    set_preserve_edge_attr,
    check_pses_in_transforms,
    check_fes_in_transforms,
    get_fes_dimensions,
    get_all_encoding_dimensions,
    get_pse_dimensions,
    get_routes_from_neighborhoods,
    infer_list_length,
    infer_list_length_plus_one,
    get_list_element,
    infer_in_hasse_graph_agg_dim,
)

class TestConfigResolvers:
    """Test config resolvers."""

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        register_all_resolvers()
        self.dataset_config_1 = OmegaConf.load("configs/dataset/graph/MUTAG.yaml")
        self.dataset_config_2 = OmegaConf.load("configs/dataset/graph/cocitation_cora.yaml")
        self.cliq_lift_transform = OmegaConf.load("configs/transforms/liftings/graph2simplicial/clique.yaml")
        self.feature_lift_transform = OmegaConf.load("configs/transforms/feature_liftings/concatenate.yaml")
        hydra.initialize(version_base="1.3", config_path="../../configs", job_name="job")

    def test_define_task_level(self):
        """Test define_task_level."""
        # node + inductive -> node_inductive (the bug-fix branch)
        assert define_task_level("node", "inductive") == "node_inductive"

        # else branch: any other combination returns dataset_task_level unchanged
        assert define_task_level("node", "transductive") == "node"
        assert define_task_level("graph", "inductive") == "graph"
        assert define_task_level("graph", "transductive") == "graph"

    def test_get_default_trainer(self):
        """Test get_default_trainer."""
        out = get_default_trainer()
        assert isinstance(out, str)

    def test_get_default_metrics(self):
        """Test get_default_metrics."""
        out = get_default_metrics("classification", 10)
        assert out == ["accuracy", "precision", "recall", "auroc", "f1"]

        out = get_default_metrics("regression", 1)
        assert out == ["mse", "mae"]

        # Test for multioutput and multilabel classification
        out = get_default_metrics("multioutput classification", 2)
        expected = ["accuracy-0", "precision-0", "recall-0", "f1-0", "accuracy-1", "precision-1", "recall-1", "f1-1"]
        assert out == expected

        out = get_default_metrics("multilabel classification", 2)
        assert out == expected

        with pytest.raises(ValueError, match="Invalid task") as e:
            get_default_metrics("some_task", 2)

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
        assert out == "model_dataset_defaults/gps_ZINC"


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

        assert get_monitor_metric("multioutput classification", "accuracy") == "val/accuracy"
        assert get_monitor_metric("multilabel classification", "f1") == "val/f1"

        with pytest.raises(ValueError, match="Invalid task") as e:
            get_monitor_metric("mix", "F1")

    def test_get_monitor_mode(self):
        """Test get_monitor_mode."""
        out = get_monitor_mode("regression")
        assert out == "min"

        out = get_monitor_mode("classification")
        assert out == "max"

        assert get_monitor_mode("multilabel classification") == "max"
        assert get_monitor_mode("multioutput classification") == "min"

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

        cfg = hydra.compose(config_name="run.yaml", overrides=["model=simplicial/topotune", "dataset=graph/cocitation_cora", "transforms.graph2simplicial_lifting.feature_lifting=Concatenation", "transforms.graph2simplicial_lifting.complex_dim=3"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [1433,2866,8598,34392]

        cfg = hydra.compose(config_name="run.yaml", overrides=["model=simplicial/topotune", "dataset=graph/cocitation_cora", "transforms.graph2simplicial_lifting.complex_dim=3"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [1433,1433,1433,1433]

        cfg = hydra.compose(config_name="run.yaml", overrides=["model=graph/gcn", "dataset=simplicial/mantra_orientation"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [1]

        cfg = hydra.compose(config_name="run.yaml", overrides=["model=simplicial/scn", "dataset=graph/cocitation_cora", "transforms.graph2simplicial_lifting.complex_dim=3"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [1433,1433,1433,1433]

        cfg = hydra.compose(config_name="run.yaml", overrides=["model=graph/gcn", "dataset=graph/MUTAG", "transforms=combined_fe"], return_hydra_config=True)
        in_channels = infer_in_channels(cfg.dataset, cfg.transforms)
        assert in_channels == [48]

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
        assert out == 2

        neighborhoods = ["up_incidence-0"]
        out = infer_topotune_num_cell_dimensions(neighborhoods)
        assert out == 2

        neighborhoods = ["down_incidence-2"]
        out = infer_topotune_num_cell_dimensions(neighborhoods)
        assert out == 3

    def test_get_default_metrics_with_params(self):
        """Test get_default_metrics with explicit metrics."""
        out = get_default_metrics("classification", 10, ["accuracy", "precision"])
        assert out == ["accuracy", "precision"]

        out = get_default_metrics("classification", 10)
        assert out == ["accuracy", "precision", "recall", "auroc", "f1"]

        out = get_default_metrics("regression", 1)
        assert out == ["mse", "mae"]

        with pytest.raises(ValueError, match="Invalid task") as e:
            get_default_metrics("some_task", 2)

    def test_set_preserve_edge_attr(self):
        """Test set_preserve_edge_attr."""
        default = True

        out = set_preserve_edge_attr(model_name="sann", default=default)
        assert out == False

        out = set_preserve_edge_attr(model_name="san", default=default)
        assert out == True

    def test_infer_in_khop_feature_dim(self):
        """Test infer_in_khop_feature_dim."""
        dataset_in_channels = [7, 7, 7]
        max_hop = 3
        out = infer_in_khop_feature_dim(dataset_in_channels, max_hop)
        assert out == [[7, 14, 42, 133], [7, 28, 91, 294], [7, 21, 70, 231]]

    def test_infer_in_khop_feature_dim_with_complex_dim(self):
        """Test infer_in_khop_feature_dim with complex_dim truncation."""
        # dataset_in_channels has 4 elements (from lifting complex_dim=3)
        # but transform only processes 3 ranks (complex_dim=3)
        dataset_in_channels = [7, 7, 7, 7]
        max_hop = 2
        # Without truncation: rank 2 hop 1 = 28 (wrong, includes rank 3 neighbor)
        out_no_trunc = infer_in_khop_feature_dim(dataset_in_channels, max_hop)
        assert out_no_trunc[2][1] == 28
        # With truncation: rank 2 hop 1 = 21 (correct, no rank 3 neighbor)
        out_trunc = infer_in_khop_feature_dim(dataset_in_channels, max_hop, complex_dim=3)
        assert out_trunc[2][1] == 21
        assert len(out_trunc) == 3

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

    def test_check_pses_in_transforms_electrostatic_pe_only(self):
        """Test check_pses_in_transforms with only ElectrostaticPE encoding."""
        transforms = OmegaConf.create({
            "ElectrostaticPE": {
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 7

    def test_check_pses_in_transforms_hkdiag_se_only(self):
        """Test check_pses_in_transforms with only HKdiagSE encoding."""
        transforms = OmegaConf.create({
            "HKdiagSE": {
                "kernel_param_HKdiagSE": [1, 5],
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 4  # range(1, 5) = 4

    def test_check_pses_in_transforms_hkdiag_se_different_ranges(self):
        """Test check_pses_in_transforms with HKdiagSE using different kernel param ranges."""
        transforms = OmegaConf.create({
            "HKdiagSE": {
                "kernel_param_HKdiagSE": [1, 9],
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 8  # range(1, 9) = 8

    def test_check_pses_in_transforms_combined_pses_electrostatic_pe(self):
        """Test check_pses_in_transforms with CombinedPSEs containing ElectrostaticPE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["ElectrostaticPE"],
                "parameters": {
                    "ElectrostaticPE": {
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 7

    def test_check_pses_in_transforms_combined_pses_hkdiag_se(self):
        """Test check_pses_in_transforms with CombinedPSEs containing HKdiagSE."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["HKdiagSE"],
                "parameters": {
                    "HKdiagSE": {
                        "kernel_param_HKdiagSE": [1, 5],
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 4

    def test_check_pses_in_transforms_combined_all_four(self):
        """Test check_pses_in_transforms with CombinedPSEs containing all four encoding types."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE", "ElectrostaticPE", "HKdiagSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": False,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    },
                    "ElectrostaticPE": {
                        "concat_to_x": True
                    },
                    "HKdiagSE": {
                        "kernel_param_HKdiagSE": [1, 4],
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 22  # 8 + 4 + 7 + 3

    def test_check_pses_in_transforms_combined_all_four_with_eigenvalues(self):
        """Test check_pses_in_transforms with all four encodings and LapPE eigenvalues."""
        transforms = OmegaConf.create({
            "CombinedPSEs": {
                "encodings": ["LapPE", "RWSE", "ElectrostaticPE", "HKdiagSE"],
                "parameters": {
                    "LapPE": {
                        "max_pe_dim": 8,
                        "include_eigenvalues": True,
                        "concat_to_x": True
                    },
                    "RWSE": {
                        "max_pe_dim": 4,
                        "concat_to_x": True
                    },
                    "ElectrostaticPE": {
                        "concat_to_x": True
                    },
                    "HKdiagSE": {
                        "kernel_param_HKdiagSE": [1, 4],
                        "concat_to_x": True
                    }
                }
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 30  # (8*2) + 4 + 7 + 3

    def test_check_pses_in_transforms_separate_all_four(self):
        """Test check_pses_in_transforms with all four encodings as separate transforms."""
        transforms = OmegaConf.create({
            "LapPE": {
                "max_pe_dim": 8,
                "include_eigenvalues": False,
                "concat_to_x": True
            },
            "RWSE": {
                "max_pe_dim": 4,
                "concat_to_x": True
            },
            "ElectrostaticPE": {
                "concat_to_x": True
            },
            "HKdiagSE": {
                "kernel_param_HKdiagSE": [1, 5],
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 23  # 8 + 4 + 7 + 4

    def test_check_pses_in_transforms_mixed_combined_and_separate_with_new_encodings(self):
        """Test check_pses_in_transforms with CombinedPSEs and separate ElectrostaticPE/HKdiagSE."""
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
            "ElectrostaticPE_extra": {
                "concat_to_x": True
            },
            "HKdiagSE_extra": {
                "kernel_param_HKdiagSE": [1, 6],
                "concat_to_x": True
            }
        })
        result = check_pses_in_transforms(transforms)
        assert result == 24  # (8 + 4) + 7 + 5

    def test_check_fes_in_transforms_empty(self):
        """Test check_fes_in_transforms with no encodings."""
        transforms = OmegaConf.create({})
        assert check_fes_in_transforms(transforms) == 0

    def test_check_fes_single_transform_pprfe_list(self):
        """Single flat PPRFE: alpha list uses second element (ListConfig path)."""
        transforms = OmegaConf.create(
            {
                "transform_name": "PPRFE",
                "alpha_param_PPRFE": [0.1, 5],
            }
        )
        assert check_fes_in_transforms(transforms) == 5

    def test_check_fes_single_transform_pprfe_scalar(self):
        """Single flat PPRFE: scalar alpha counts as fixed dimension."""
        transforms = OmegaConf.create(
            {
                "transform_name": "PPRFE",
                "alpha_param_PPRFE": 4,
            }
        )
        assert check_fes_in_transforms(transforms) == 4

    def test_check_fes_single_transform_sheaf(self):
        """Single flat SheafConnLapPE uses max_pe_dim."""
        transforms = OmegaConf.create(
            {"transform_name": "SheafConnLapPE", "max_pe_dim": 6}
        )
        assert check_fes_in_transforms(transforms) == 6

    def test_check_fes_keyed_pprfe_list(self):
        """Keyed transform whose name contains PPRFE."""
        transforms = OmegaConf.create(
            {"run_PPRFE_1": {"alpha_param_PPRFE": [0.1, 8]}}
        )
        assert check_fes_in_transforms(transforms) == 8

    def test_check_fes_keyed_pprfe_scalar(self):
        """Keyed PPRFE with scalar alpha_param."""
        transforms = OmegaConf.create({"extra_PPRFE": {"alpha_param_PPRFE": 3}})
        assert check_fes_in_transforms(transforms) == 3

    def test_check_fes_keyed_sheaf(self):
        """Keyed transform whose name contains SheafConnLapPE."""
        transforms = OmegaConf.create(
            {"custom_SheafConnLapPE": {"max_pe_dim": 9}}
        )
        assert check_fes_in_transforms(transforms) == 9

    def test_check_fes_combined_fes_pprfe_and_sheaf(self):
        """Test CombinedFEs inner loop with PPRFE list and SheafConnLapPE."""
        transforms = OmegaConf.create(
            {
                "CombinedFEs": {
                    "encodings": ["PPRFE", "SheafConnLapPE"],
                    "parameters": {
                        "PPRFE": {"alpha_param_PPRFE": [0.1, 7], "concat_to_x": True},
                        "SheafConnLapPE": {
                            "max_pe_dim": 4,
                            "stalk_dim": 2,
                            "concat_to_x": True,
                        },
                    },
                }
            }
        )
        assert check_fes_in_transforms(transforms) == 7 + 4

    def test_check_fes_combined_fes_pprfe_default_alpha(self):
        """Test CombinedFEs PPRFE with missing alpha_param using default [0.1, 10]."""
        transforms = OmegaConf.create(
            {
                "CombinedFEs": {
                    "encodings": ["PPRFE"],
                    "parameters": {"PPRFE": {"concat_to_x": True}},
                }
            }
        )
        assert check_fes_in_transforms(transforms) == 10

    def test_check_fes_combined_fes_pprfe_scalar_alpha(self):
        """Test CombinedFEs PPRFE with scalar alpha_param."""
        transforms = OmegaConf.create(
            {
                "CombinedFEs": {
                    "encodings": ["PPRFE"],
                    "parameters": {
                        "PPRFE": {"alpha_param_PPRFE": 11, "concat_to_x": True}
                    },
                }
            }
        )
        assert check_fes_in_transforms(transforms) == 11

    def test_check_fes_in_transforms_hkfe(self):
        """Test check_fes_in_transforms with HKFE."""
        transforms = OmegaConf.create({
            "transform_name": "HKFE",
            "kernel_param_HKFE": [1, 5]
        })
        assert check_fes_in_transforms(transforms) == 4

        transforms = OmegaConf.create({
            "transform_name": "HKFE",
            "kernel_param_HKFE": 3
        })
        assert check_fes_in_transforms(transforms) == 3

        transforms = OmegaConf.create({
            "CombinedFEs": {
                "encodings": ["HKFE"],
                "parameters": {
                    "HKFE": {"kernel_param_HKFE": [2, 7]}
                }
            }
        })
        assert check_fes_in_transforms(transforms) == 5

        transforms = OmegaConf.create({
            "HKFE_extra": {"kernel_param_HKFE": [0, 10]}
        })
        assert check_fes_in_transforms(transforms) == 10

    def test_check_fes_in_transforms_khopfe(self):
        """Test check_fes_in_transforms with KHopFE."""
        transforms = OmegaConf.create({
            "transform_name": "KHopFE",
            "max_hop": 5
        })
        assert check_fes_in_transforms(transforms) == 4

        transforms = OmegaConf.create({
            "CombinedFEs": {
                "encodings": ["KHopFE"],
                "parameters": {
                    "KHopFE": {"max_hop": 3}
                }
            }
        })
        assert check_fes_in_transforms(transforms) == 2

        transforms = OmegaConf.create({
            "KHopFE_extra": {"max_hop": 6}
        })
        assert check_fes_in_transforms(transforms) == 5

    def test_get_fes_dimensions_khopfe(self):
        """Test get_fes_dimensions with KHopFE using max_hop - 1."""
        encodings = ["KHopFE"]
        parameters = {"KHopFE": {"max_hop": 5}}
        assert get_fes_dimensions(encodings, parameters) == [4]

    def test_get_fes_dimensions_hkfe(self):
        """Test get_fes_dimensions with HKFE."""
        encodings = ["HKFE"]
        parameters = {"HKFE": {"kernel_param_HKFE": [1, 5]}}
        assert get_fes_dimensions(encodings, parameters) == [4]

        parameters = {"HKFE": {"kernel_param_HKFE": 3}}
        assert get_fes_dimensions(encodings, parameters) == [3]

    def test_get_fes_dimensions_pprfe_list_tuple(self):
        """Test get_fes_dimensions with PPRFE alpha as tuple returning second element."""
        encodings = ["PPRFE"]
        parameters = {"PPRFE": {"alpha_param_PPRFE": (0.1, 6)}}
        assert get_fes_dimensions(encodings, parameters) == [6]

    def test_get_fes_dimensions_pprfe_omegaconf_list(self):
        """Test get_fes_dimensions with PPRFE alpha as OmegaConf list."""
        parameters = OmegaConf.create(
            {"PPRFE": {"alpha_param_PPRFE": [0.1, 12]}}
        )
        assert get_fes_dimensions(["PPRFE"], parameters) == [12]

    def test_get_fes_dimensions_pprfe_scalar(self):
        """Test get_fes_dimensions with PPRFE scalar alpha."""
        encodings = ["PPRFE"]
        parameters = {"PPRFE": {"alpha_param_PPRFE": 5}}
        assert get_fes_dimensions(encodings, parameters) == [5]

    def test_get_fes_dimensions_pprfe_missing_uses_default(self):
        """Test get_fes_dimensions with missing PPRFE block using default alpha upper bound 10."""
        assert get_fes_dimensions(["PPRFE"], {}) == [10]

    def test_get_fes_dimensions_sheaf(self):
        """Test get_fes_dimensions with SheafConnLapPE."""
        encodings = ["SheafConnLapPE"]
        parameters = {"SheafConnLapPE": {"max_pe_dim": 8}}
        assert get_fes_dimensions(encodings, parameters) == [8]

    def test_get_all_encoding_dimensions_khopfe_pprfe_sheaf(self):
        """Test get_all_encoding_dimensions with KHopFE, PPRFE list, and SheafConnLapPE branches."""
        encodings = ["KHopFE", "PPRFE", "SheafConnLapPE"]
        parameters = {
            "KHopFE": {"max_hop": 4},
            "PPRFE": {"alpha_param_PPRFE": [0.1, 9]},
            "SheafConnLapPE": {"max_pe_dim": 3},
        }
        assert get_all_encoding_dimensions(encodings, parameters) == [3, 9, 3]

    def test_get_all_encoding_dimensions_pprfe_scalar(self):
        """Test get_all_encoding_dimensions with PPRFE scalar alpha."""
        assert get_all_encoding_dimensions(
            ["PPRFE"], {"PPRFE": {"alpha_param_PPRFE": 2}}
        ) == [2]

    def test_get_all_encoding_dimensions_pprfe_missing_uses_default(self):
        """Test get_all_encoding_dimensions with PPRFE absent from parameters using default 10."""
        assert get_all_encoding_dimensions(["PPRFE"], {}) == [10]

    def test_get_all_encoding_dimensions_exhaustive(self):
        """Test get_all_encoding_dimensions for all supported encodings."""
        encodings = ["LapPE", "RWSE", "ElectrostaticPE", "HKdiagSE", "HKFE", "KHopFE", "PPRFE", "SheafConnLapPE"]
        parameters = {
            "LapPE": {"max_pe_dim": 8, "include_eigenvalues": True},
            "RWSE": {"max_pe_dim": 4},
            "ElectrostaticPE": {},
            "HKdiagSE": {"kernel_param_HKdiagSE": [1, 5]},
            "HKFE": {"kernel_param_HKFE": [2, 8]},
            "KHopFE": {"max_hop": 3},
            "PPRFE": {"alpha_param_PPRFE": [0.1, 7]},
            "SheafConnLapPE": {"max_pe_dim": 5}
        }
        expected = [16, 4, 7, 4, 6, 2, 7, 5]
        assert get_all_encoding_dimensions(encodings, parameters) == expected

        # Test scalar params
        parameters["HKdiagSE"]["kernel_param_HKdiagSE"] = 10
        parameters["HKFE"]["kernel_param_HKFE"] = 12
        parameters["PPRFE"]["alpha_param_PPRFE"] = 15
        parameters["LapPE"]["include_eigenvalues"] = False
        expected = [8, 4, 7, 10, 12, 2, 15, 5]
        assert get_all_encoding_dimensions(encodings, parameters) == expected


class TestNewHopseResolvers:
    """Targeted tests for the HOPSE-era resolvers added in this branch."""

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    # ---- get_routes_from_neighborhoods ----

    def test_get_routes_from_neighborhoods_basic(self):
        nbhds = [
            "up_adjacency-0",
            "up_incidence-0",
            "down_incidence-1",
            "2-up_incidence-0",
        ]
        routes = get_routes_from_neighborhoods(nbhds)
        assert routes == [[0, 0], [0, 1], [1, 0], [0, 2]]

    def test_get_routes_from_neighborhoods_rejects_unknown(self):
        with pytest.raises(Exception, match="Invalid neighborhood"):
            get_routes_from_neighborhoods(["something_weird-0"])

    # ---- get_pse_dimensions ----

    @pytest.mark.parametrize(
        "encodings, parameters, expected",
        [
            (
                ["LapPE"],
                {"LapPE": {"max_pe_dim": 4, "include_eigenvalues": False}},
                [4],
            ),
            (
                ["LapPE"],
                {"LapPE": {"max_pe_dim": 4, "include_eigenvalues": True}},
                [8],
            ),
            (["RWSE"], {"RWSE": {"max_pe_dim": 6}}, [6]),
            (["ElectrostaticPE"], {}, [7]),
            (
                ["HKdiagSE"],
                {"HKdiagSE": {"kernel_param_HKdiagSE": [1, 5]}},
                [4],
            ),
            (
                ["HKdiagSE"],
                {"HKdiagSE": {"kernel_param_HKdiagSE": 3}},
                [3],
            ),
        ],
    )
    def test_get_pse_dimensions_parametrized(
        self, encodings, parameters, expected
    ):
        assert get_pse_dimensions(encodings, parameters) == expected

    # ---- infer_list_length / +1 / get_list_element ----

    def test_infer_list_length(self):
        assert infer_list_length([]) == 0
        assert infer_list_length([1, 2, 3]) == 3

    def test_infer_list_length_plus_one(self):
        assert infer_list_length_plus_one([]) == 1
        assert infer_list_length_plus_one([10, 20]) == 3

    def test_get_list_element(self):
        assert get_list_element([10, 20, 30], 0) == 10
        assert get_list_element([10, 20, 30], -1) == 30
        with pytest.raises(IndexError):
            get_list_element([1], 5)

    # ---- infer_in_hasse_graph_agg_dim ----

    def test_infer_in_hasse_graph_agg_dim_copy_initial_scalar_dim_in(self):
        out = infer_in_hasse_graph_agg_dim(
            neighborhoods=["up_adjacency-0", "up_incidence-0"],
            dim_pses=[5, 7],
            complex_dim=2,
            max_hop=3,
            dim_in=4,
            dim_hidden_graph=16,
            dim_hidden_node=8,
            copy_initial=True,
            use_edge_attr=False,
        )
        assert isinstance(out, list)
        assert len(out) == 3  # complex_dim + 1
        assert all(len(row) == 3 for row in out)  # max_hop
        # Hop 0 should be the input dim (since copy_initial=True).
        assert out[0][0] == 4
        assert out[1][0] == 4

    def test_infer_in_hasse_graph_agg_dim_with_edge_attr(self):
        out = infer_in_hasse_graph_agg_dim(
            neighborhoods=["up_adjacency-0"],
            dim_pses=[5],
            complex_dim=1,
            max_hop=2,
            dim_in=[3, 9],
            dim_hidden_graph=16,
            dim_hidden_node=8,
            copy_initial=True,
            use_edge_attr=True,
        )
        # Hop 0 for dim 0 is dim_in[0]; for dim 1 is dim_in[1] (edge attr).
        assert out[0][0] == 3
        assert out[1][0] == 9

    def test_infer_in_hasse_graph_agg_dim_no_copy_initial(self):
        out = infer_in_hasse_graph_agg_dim(
            neighborhoods=["up_adjacency-0"],
            dim_pses=[5],
            complex_dim=1,
            max_hop=2,
            dim_in=4,
            dim_hidden_graph=16,
            dim_hidden_node=8,
            copy_initial=False,
            use_edge_attr=False,
        )
        # All initial entries == dim_hidden_graph + dim_hidden_node.
        assert out[0][0] == 24
        assert out[1][0] == 24
