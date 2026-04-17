"""Unit tests for config instantiators."""

import pytest
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
from unittest.mock import MagicMock
from topobench.utils.utils import extras, get_metric_value, task_wrapper

# initialize(config_path="../../configs", job_name="job")

class TestUtils:
    """Test config instantiators."""

    def setup_method(self):
        """Setup method."""
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.metric_dict = {'accuracy': torch.tensor([90])}
        hydra.initialize(version_base="1.3", config_path="../../configs", job_name="job")
        self.cfg = hydra.compose(config_name="run.yaml", overrides=["extras.ignore_warnings=True","tags=False"], return_hydra_config=True)

    def test_get_metric_value(self):
        """Test get_metric_value."""
        out = get_metric_value(self.metric_dict, "accuracy")
        assert out == 90.

        with pytest.raises(Exception) as e:
            get_metric_value(self.metric_dict, "some_metric")

        out = get_metric_value(self.metric_dict, None)
        assert out is None

    def test_extras(self):
        """Test extras."""
        # extras(self.cfg)
        extras({})


    def test_task_wrapper(self):
        """Test task_wrapper."""
        d = DictConfig({'paths': {'output_dir': 'logs/'}})

        def task_func(cfg: DictConfig):
            """Task function for testing task_wrapper.

            Parameters
            ----------
            cfg : DictConfig
                A DictConfig object containing the config tree.

            Returns
            -------
            dict[str, Any], dict[str, Any]
                The metric and object dictionaries.
            """
            return {'accuracy': torch.tensor([90])}, {'model': 'model'}


        out = task_wrapper(task_func)(d)

        assert out[0]['accuracy'] == 90., "Metric dictionary not returned correctly."
        assert out[1]['model'] == 'model', "Object dictionary not returned correctly."

        mock_task_func = MagicMock(side_effect=Exception("Test exception"))
        with pytest.raises(Exception, match="Test exception"):
            task_wrapper(mock_task_func)(d)
