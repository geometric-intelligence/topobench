"""Unit tests for config instantiators."""

import pytest
from omegaconf import OmegaConf, DictConfig
from topobench.utils.instantiators import (
    instantiate_callbacks,
    instantiate_loggers
)

class TestConfigInstantiators:
    """Test config instantiators."""

    def setup_method(self):
        """Setup method."""
        self.callback = OmegaConf.load("configs/callbacks/model_summary.yaml")
        self.logger = DictConfig(
            {
            '_target_': 'lightning.pytorch.loggers.wandb.WandbLogger',
            'save_dir': '/',
            'offline': False,
            'id': None,
            'anonymous': None,
            'project': 'None',
            'log_model': False,
            'prefix': '',
            'group': '',
            'tags': [],
            'job_type': ''
            }
        )

    def test_instantiate_callbacks(self):
        """Test instantiate_callbacks."""
        result = instantiate_callbacks(self.callback)
        assert type(result) == list

    def test_instantiate_loggers(self):
        """Test instantiate_loggers."""
        result = instantiate_loggers(self.logger)
        assert type(result) == list
