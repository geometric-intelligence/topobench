"""Unit tests for the optimizer manager class."""

import pytest
import torch

from topobench.optimizer import TBOptimizer


class TestTBOptimizer:
    """Test the TBOptimizer class."""

    def setup_method(self):
        """Setup method."""
        self.optimizer_config_with_scheduler = {
            "optimizer_id": "Adam",
            "parameters": {"lr": 0.001},
            "scheduler": {"scheduler_id": "StepLR", "scheduler_params": {"step_size": 30, "gamma": 0.1}}
        }
        self.optimizer_config_without_scheduler = {
            "optimizer_id": "Adam",
            "parameters": {"lr": 0.001}
        }
        self.params = {torch.Tensor([0,3,4])}

    def test_configure_optimizer(self):
        """Test the configure_optimizer method."""
        # Check with scheduler
        optimizer = TBOptimizer(**self.optimizer_config_with_scheduler)
        out = optimizer.configure_optimizer(self.params)
        assert "optimizer" in out
        assert "lr_scheduler" in out

        # Check without scheduler
        optimizer = TBOptimizer(**self.optimizer_config_without_scheduler)
        out = optimizer.configure_optimizer(self.params)
        assert "optimizer" in out
        assert "lr_scheduler" not in out
