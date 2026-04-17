"""Test the PipelineTimer callback."""
import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from topobench.callbacks.timer_callback import PipelineTimer


class TestPipelineTimer:
    """Test the PipelineTimer callback."""

    def test_init(self):
        """Test callback initialization."""
        timer = PipelineTimer()

        # Check that all stage dictionaries are initialized
        expected_stages = ["train_batch", "train_epoch", "val_batch", "val_epoch", "test_batch", "test_epoch"]
        for stage in expected_stages:
            assert stage in timer.sums
            assert stage in timer.counts
            assert timer.sums[stage] == []
            assert timer.counts[stage] == 0

        assert timer.times == {}
        assert timer.skip_first_n == 10

    def test_start_timer(self):
        """Test that _start_timer records the start time."""
        timer = PipelineTimer()

        timer._start_timer("train_batch")

        assert "train_batch" in timer.times
        assert isinstance(timer.times["train_batch"], float)

    def test_end_timer(self):
        """Test that _end_timer calculates and stores elapsed time."""
        timer = PipelineTimer()

        timer._start_timer("train_batch")
        time.sleep(0.01)  # Small delay
        timer._end_timer("train_batch")

        assert len(timer.sums["train_batch"]) == 1
        assert timer.counts["train_batch"] == 1
        assert timer.sums["train_batch"][0] > 0
        assert timer.sums["train_batch"][0] >= 0.01

    def test_end_timer_without_start(self):
        """Test that _end_timer handles missing start gracefully."""
        timer = PipelineTimer()

        # Should not raise error
        timer._end_timer("train_batch")

        # Nothing should be recorded
        assert len(timer.sums["train_batch"]) == 0
        assert timer.counts["train_batch"] == 0

    def test_multiple_timer_cycles(self):
        """Test multiple start/end cycles accumulate correctly."""
        timer = PipelineTimer()

        for _ in range(5):
            timer._start_timer("train_batch")
            time.sleep(0.001)
            timer._end_timer("train_batch")

        assert len(timer.sums["train_batch"]) == 5
        assert timer.counts["train_batch"] == 5

    def test_train_batch_hooks(self):
        """Test train batch timing hooks."""
        timer = PipelineTimer()

        timer.on_train_batch_start()
        assert "train_batch" in timer.times

        time.sleep(0.001)
        timer.on_train_batch_end()

        assert len(timer.sums["train_batch"]) == 1
        assert timer.counts["train_batch"] == 1

    def test_train_epoch_hooks(self):
        """Test train epoch timing hooks."""
        timer = PipelineTimer()

        timer.on_train_epoch_start()
        assert "train_epoch" in timer.times

        time.sleep(0.001)
        timer.on_train_epoch_end()

        assert len(timer.sums["train_epoch"]) == 1
        assert timer.counts["train_epoch"] == 1

    def test_validation_batch_hooks(self):
        """Test validation batch timing hooks."""
        timer = PipelineTimer()

        timer.on_validation_batch_start()
        assert "val_batch" in timer.times

        time.sleep(0.001)
        timer.on_validation_batch_end()

        assert len(timer.sums["val_batch"]) == 1
        assert timer.counts["val_batch"] == 1

    def test_validation_epoch_hooks(self):
        """Test validation epoch timing hooks."""
        timer = PipelineTimer()

        timer.on_validation_epoch_start()
        assert "val_epoch" in timer.times

        time.sleep(0.001)
        timer.on_validation_epoch_end()

        assert len(timer.sums["val_epoch"]) == 1
        assert timer.counts["val_epoch"] == 1

    def test_test_batch_hooks(self):
        """Test test batch timing hooks."""
        timer = PipelineTimer()

        timer.on_test_batch_start()
        assert "test_batch" in timer.times

        time.sleep(0.001)
        timer.on_test_batch_end()

        assert len(timer.sums["test_batch"]) == 1
        assert timer.counts["test_batch"] == 1

    def test_test_epoch_hooks(self):
        """Test test epoch timing hooks."""
        timer = PipelineTimer()

        timer.on_test_epoch_start()
        assert "test_epoch" in timer.times

        time.sleep(0.001)
        timer.on_test_epoch_end()

        assert len(timer.sums["test_epoch"]) == 1
        assert timer.counts["test_epoch"] == 1

    def test_log_hyperparams_with_single_logger(self):
        """Test _log_hyperparams with a single logger."""
        timer = PipelineTimer()

        trainer = Mock()
        logger = Mock()
        logger.log_hyperparams = Mock()
        trainer.logger = logger

        params = {"metric1": 1.5, "metric2": 2.5}
        timer._log_hyperparams(trainer, params)

        logger.log_hyperparams.assert_called_once_with(params)

    def test_log_hyperparams_with_multiple_loggers(self):
        """Test _log_hyperparams with multiple loggers."""
        timer = PipelineTimer()

        trainer = Mock()
        logger1 = Mock()
        logger1.log_hyperparams = Mock()
        logger2 = Mock()
        logger2.log_hyperparams = Mock()
        trainer.logger = [logger1, logger2]

        params = {"metric1": 1.5}
        timer._log_hyperparams(trainer, params)

        logger1.log_hyperparams.assert_called_once_with(params)
        logger2.log_hyperparams.assert_called_once_with(params)

    def test_log_hyperparams_without_logger(self):
        """Test _log_hyperparams handles missing logger gracefully."""
        timer = PipelineTimer()

        trainer = Mock()
        trainer.logger = None

        # Should not raise error
        timer._log_hyperparams(trainer, {"metric1": 1.5})

    def test_log_hyperparams_without_log_hyperparams_method(self):
        """Test _log_hyperparams handles loggers without log_hyperparams method."""
        timer = PipelineTimer()

        trainer = Mock()
        logger = Mock(spec=[])  # Logger without log_hyperparams method
        trainer.logger = logger

        # Should not raise error
        timer._log_hyperparams(trainer, {"metric1": 1.5})

    def test_log_hyperparams_exception_suppression(self):
        """Test that exceptions in log_hyperparams are suppressed."""
        timer = PipelineTimer()

        trainer = Mock()
        logger = Mock()
        logger.log_hyperparams = Mock(side_effect=Exception("Test error"))
        trainer.logger = logger

        # Should not raise error
        timer._log_hyperparams(trainer, {"metric1": 1.5})

    def test_log_averages_skips_first_n(self):
        """Test that _log_averages skips first N measurements for non-test stages."""
        timer = PipelineTimer()
        timer.skip_first_n = 3

        # Add some measurements
        for i in range(10):
            timer.sums["train_batch"].append(float(i))
            timer.counts["train_batch"] += 1

        trainer = Mock()
        logger = Mock()
        logger.log_hyperparams = Mock()
        trainer.logger = logger

        timer._log_averages(trainer)

        # Should be called with averaged values (skipping first 3)
        logger.log_hyperparams.assert_called_once()
        call_args = logger.log_hyperparams.call_args[0][0]

        # Average of [3, 4, 5, 6, 7, 8, 9] = 6.0
        assert "AvgTime/train_batch_mean" in call_args
        assert call_args["AvgTime/train_batch_mean"] == pytest.approx(6.0)

    def test_log_averages_test_stages_no_skip(self):
        """Test that _log_averages doesn't skip measurements for test stages."""
        timer = PipelineTimer()
        timer.skip_first_n = 3

        # Add test measurements
        for i in range(5):
            timer.sums["test_batch"].append(float(i))
            timer.counts["test_batch"] += 1

        trainer = Mock()
        logger = Mock()
        logger.log_hyperparams = Mock()
        trainer.logger = logger

        timer._log_averages(trainer)

        call_args = logger.log_hyperparams.call_args[0][0]

        # Average of all [0, 1, 2, 3, 4] = 2.0
        assert "AvgTime/test_batch_mean" in call_args
        assert call_args["AvgTime/test_batch_mean"] == pytest.approx(2.0)
        # Std for test should be 0.0 (as specified in implementation)
        assert call_args["AvgTime/test_batch_std"] == 0.0

    def test_log_averages_computes_std(self):
        """Test that _log_averages computes standard deviation."""
        timer = PipelineTimer()
        timer.skip_first_n = 0

        # Add measurements with known std
        timer.sums["train_batch"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        timer.counts["train_batch"] = 5

        trainer = Mock()
        logger = Mock()
        logger.log_hyperparams = Mock()
        trainer.logger = logger

        timer._log_averages(trainer)

        call_args = logger.log_hyperparams.call_args[0][0]

        assert "AvgTime/train_batch_mean" in call_args
        assert "AvgTime/train_batch_std" in call_args
        assert call_args["AvgTime/train_batch_mean"] == pytest.approx(3.0)
        assert call_args["AvgTime/train_batch_std"] > 0  # Should have positive std

    def test_log_averages_empty_stage(self):
        """Test that _log_averages handles stages with no measurements."""
        timer = PipelineTimer()

        # Don't add any measurements
        trainer = Mock()
        logger = Mock()
        logger.log_hyperparams = Mock()
        trainer.logger = logger

        # Should not raise error
        timer._log_averages(trainer)

        # Should still be called (but with empty or minimal data)
        logger.log_hyperparams.assert_called_once()

    def test_on_train_end_calls_log_averages(self):
        """Test that on_train_end calls _log_averages."""
        timer = PipelineTimer()
        timer.skip_first_n = 0  # Don't skip any measurements to avoid empty slices

        # Add some measurements to all stages to avoid warnings
        for stage in ["train_batch", "train_epoch", "val_batch", "val_epoch", "test_batch", "test_epoch"]:
            timer.sums[stage] = [1.0, 2.0, 3.0]
            timer.counts[stage] = 3

        trainer = Mock()
        logger = Mock()
        logger.log_hyperparams = Mock()
        trainer.logger = logger

        timer.on_train_end(trainer)

        # Should have called log_hyperparams through _log_averages
        logger.log_hyperparams.assert_called_once()

    def test_full_training_cycle(self):
        """Test a complete training cycle with all hooks."""
        timer = PipelineTimer()

        # Simulate training loop
        for epoch in range(2):
            timer.on_train_epoch_start()

            for batch in range(3):
                timer.on_train_batch_start()
                time.sleep(0.001)
                timer.on_train_batch_end()

            timer.on_train_epoch_end()

            # Validation
            timer.on_validation_epoch_start()

            for batch in range(2):
                timer.on_validation_batch_start()
                time.sleep(0.001)
                timer.on_validation_batch_end()

            timer.on_validation_epoch_end()

        # Check all stages were timed
        assert timer.counts["train_batch"] == 6
        assert timer.counts["train_epoch"] == 2
        assert timer.counts["val_batch"] == 4
        assert timer.counts["val_epoch"] == 2

        # Check measurements are positive
        assert all(t > 0 for t in timer.sums["train_batch"])
        assert all(t > 0 for t in timer.sums["val_batch"])

    def test_all_stages_logged(self):
        """Test that _log_averages logs all stages with measurements."""
        timer = PipelineTimer()
        timer.skip_first_n = 0

        # Add measurements to all stages
        all_stages = ["train_batch", "train_epoch", "val_batch", "val_epoch", "test_batch", "test_epoch"]
        for stage in all_stages:
            timer.sums[stage] = [1.0, 2.0, 3.0]
            timer.counts[stage] = 3

        trainer = Mock()
        logger = Mock()
        logger.log_hyperparams = Mock()
        trainer.logger = logger

        timer._log_averages(trainer)

        call_args = logger.log_hyperparams.call_args[0][0]

        # All stages should have mean and std logged
        for stage in all_stages:
            assert f"AvgTime/{stage}_mean" in call_args
            assert f"AvgTime/{stage}_std" in call_args
