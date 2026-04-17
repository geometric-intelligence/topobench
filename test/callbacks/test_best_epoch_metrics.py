"""Test the BestEpochMetricsCallback class."""
import pytest
import torch
from unittest.mock import MagicMock, Mock
from lightning.pytorch.callbacks import ModelCheckpoint
from topobench.callbacks import BestEpochMetricsCallback


class TestBestEpochMetricsCallback:
    """Test the BestEpochMetricsCallback class."""

    def test_init(self):
        """Test callback initialization."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")
        assert callback.monitor_metric == "val/loss"
        assert callback.mode == "min"
        assert callback.best_monitored_value is None
        assert callback.best_epoch_number is None
        assert callback.best_epoch_metrics == {}

    def test_init_with_max_mode(self):
        """Test callback initialization with max mode."""
        callback = BestEpochMetricsCallback(monitor="val/accuracy", mode="max")
        assert callback.monitor_metric == "val/accuracy"
        assert callback.mode == "max"

    def test_on_train_start_finds_checkpoint_callback(self):
        """Test that on_train_start finds ModelCheckpoint callback."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        # Create mock trainer with ModelCheckpoint callback
        trainer = Mock()
        checkpoint_callback = ModelCheckpoint()
        trainer.callbacks = [checkpoint_callback, Mock()]
        pl_module = Mock()

        callback.on_train_start(trainer, pl_module)

        assert callback.checkpoint_callback is checkpoint_callback

    def test_on_train_start_without_checkpoint_callback(self):
        """Test that on_train_start works without ModelCheckpoint (checkpoint_callback stays None)."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        # Create mock trainer without ModelCheckpoint callback
        trainer = Mock()
        trainer.callbacks = [Mock(), Mock()]
        pl_module = Mock()

        callback.on_train_start(trainer, pl_module)

        assert callback.checkpoint_callback is None

    def test_on_train_epoch_end_captures_metrics(self):
        """Test that training metrics are captured at end of training epoch."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        trainer = Mock()
        trainer.callback_metrics = {
            "train/loss": torch.tensor(0.5),
            "train/accuracy": torch.tensor(0.85),
            "val/loss": torch.tensor(0.6),  # Should not be captured
        }
        pl_module = Mock()

        callback.on_train_epoch_end(trainer, pl_module)

        assert "train/loss" in callback.current_epoch_train_metrics
        assert "train/accuracy" in callback.current_epoch_train_metrics
        assert "val/loss" not in callback.current_epoch_train_metrics
        assert callback.current_epoch_train_metrics["train/loss"] == 0.5

    def test_on_validation_epoch_end_first_epoch(self):
        """Test that first epoch is always considered best."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/loss": 0.5,
            "train/accuracy": 0.8
        }

        trainer = Mock()
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.6),
            "val/accuracy": torch.tensor(0.75),
        }
        pl_module = Mock()

        callback.on_validation_epoch_end(trainer, pl_module)

        assert callback.best_monitored_value == pytest.approx(0.6)
        assert callback.best_epoch_number == 0
        assert "train/loss" in callback.best_epoch_metrics
        assert "val/loss" in callback.best_epoch_metrics

    def test_on_validation_epoch_end_min_mode_improvement(self):
        """Test detection of improvement in min mode."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")
        callback.best_monitored_value = 0.6
        callback.best_epoch_number = 0

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/loss": 0.3,
        }

        trainer = Mock()
        trainer.current_epoch = 1
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.4),  # Better (lower)
        }
        pl_module = Mock()

        callback.on_validation_epoch_end(trainer, pl_module)

        assert callback.best_monitored_value == pytest.approx(0.4)
        assert callback.best_epoch_number == 1

    def test_on_validation_epoch_end_min_mode_no_improvement(self):
        """Test that worse values don't update in min mode."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")
        callback.best_monitored_value = 0.4
        callback.best_epoch_number = 1
        old_metrics = {"val/loss": 0.4}
        callback.best_epoch_metrics = old_metrics.copy()

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/loss": 0.5,
        }

        trainer = Mock()
        trainer.current_epoch = 2
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.6),  # Worse (higher)
        }
        pl_module = Mock()

        callback.on_validation_epoch_end(trainer, pl_module)

        # Should not update
        assert callback.best_monitored_value == 0.4
        assert callback.best_epoch_number == 1

    def test_on_validation_epoch_end_max_mode_improvement(self):
        """Test detection of improvement in max mode."""
        callback = BestEpochMetricsCallback(monitor="val/accuracy", mode="max")
        callback.best_monitored_value = 0.7
        callback.best_epoch_number = 0

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/accuracy": 0.85,
        }

        trainer = Mock()
        trainer.current_epoch = 1
        trainer.callback_metrics = {
            "val/accuracy": torch.tensor(0.8),  # Better (higher)
        }
        pl_module = Mock()

        callback.on_validation_epoch_end(trainer, pl_module)

        assert callback.best_monitored_value == pytest.approx(0.8)
        assert callback.best_epoch_number == 1

    def test_on_validation_epoch_end_max_mode_no_improvement(self):
        """Test that worse values don't update in max mode."""
        callback = BestEpochMetricsCallback(monitor="val/accuracy", mode="max")
        callback.best_monitored_value = 0.8
        callback.best_epoch_number = 1

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/accuracy": 0.75,
        }

        trainer = Mock()
        trainer.current_epoch = 2
        trainer.callback_metrics = {
            "val/accuracy": torch.tensor(0.7),  # Worse (lower)
        }
        pl_module = Mock()

        callback.on_validation_epoch_end(trainer, pl_module)

        # Should not update
        assert callback.best_monitored_value == 0.8
        assert callback.best_epoch_number == 1

    def test_best_epoch_logging(self):
        """Test that best epoch number and metrics are logged."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        # Setup training metrics
        callback.current_epoch_train_metrics = {
            "train/loss": 0.5,
            "train/accuracy": 0.8
        }

        trainer = Mock()
        trainer.current_epoch = 5
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.3),
            "val/accuracy": torch.tensor(0.85),
        }
        pl_module = Mock()

        callback.on_validation_epoch_end(trainer, pl_module)

        # Check that log was called with best epoch number
        pl_module.log.assert_any_call("best_epoch", 5, prog_bar=False)

        # Check that metrics were logged with best_epoch prefix
        calls = [call[0] for call in pl_module.log.call_args_list]
        assert any("best_epoch/train/loss" in str(call) for call in calls)
        assert any("best_epoch/val/loss" in str(call) for call in calls)

    def test_on_train_end_with_checkpoint(self):
        """Test on_train_end logs checkpoint path when available."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        # Setup checkpoint callback
        checkpoint_callback = Mock()
        checkpoint_callback.best_model_path = "/path/to/checkpoint.ckpt"
        callback.checkpoint_callback = checkpoint_callback

        trainer = Mock()
        pl_module = Mock()
        pl_module.logger = Mock()
        pl_module.logger.experiment.summary = {}

        callback.on_train_end(trainer, pl_module)

        # Check that checkpoint path was logged
        assert pl_module.logger.experiment.summary["best_epoch/checkpoint"] == "/path/to/checkpoint.ckpt"
        assert pl_module.logger.experiment.summary["monitored_metric"] == "val/loss (min)"

    def test_on_train_end_without_checkpoint(self):
        """Test on_train_end handles missing checkpoint gracefully."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")
        callback.checkpoint_callback = None

        trainer = Mock()
        pl_module = Mock()

        # Should not raise error
        callback.on_train_end(trainer, pl_module)

    def test_on_train_end_without_logger(self):
        """Test on_train_end handles missing logger gracefully."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        checkpoint_callback = Mock()
        checkpoint_callback.best_model_path = "/path/to/checkpoint.ckpt"
        callback.checkpoint_callback = checkpoint_callback

        trainer = Mock()
        pl_module = Mock()
        pl_module.logger = None

        # Should not raise error
        callback.on_train_end(trainer, pl_module)

    def test_handles_tensor_values(self):
        """Test that callback properly converts tensor values to floats."""
        callback = BestEpochMetricsCallback(monitor="val/loss", mode="min")

        trainer = Mock()
        trainer.current_epoch = 0
        trainer.callback_metrics = {
            "val/loss": torch.tensor(0.5),
            "val/accuracy": torch.tensor(0.85),
        }
        callback.current_epoch_train_metrics = {}
        pl_module = Mock()

        callback.on_validation_epoch_end(trainer, pl_module)

        # Values should be converted to floats
        assert isinstance(callback.best_monitored_value, float)
        assert callback.best_monitored_value == 0.5
