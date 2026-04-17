"""Unit tests for logging utils."""
import pytest
from unittest.mock import MagicMock, patch
from topobench.utils import log_hyperparameters

@patch("topobench.utils.logging_utils.pylogger.RankedLogger.warning")
@patch("topobench.utils.logging_utils.OmegaConf.to_container")
def test_log_hyperparameters(mock_to_container, mock_warning):
    """Test the log_hyperparameters function.

    Parameters
    ----------
    mock_to_container : MagicMock
        Mock of OmegaConf.to_container.
    mock_warning : MagicMock
        Mock of pylogger.RankedLogger.warning.
    """
    # Mock the input dictionary
    mock_cfg = MagicMock()
    mock_model = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.logger = True  # Ensure logger is present

    object_dict = {
        "cfg": mock_cfg,
        "model": mock_model,
        "trainer": mock_trainer,
    }

    # Mock the OmegaConf.to_container return value to include all required keys
    mock_to_container.return_value = {
        "model": "mock_model",
        "dataset": "mock_dataset",
        "trainer": "mock_trainer",
        "callbacks": "mock_callbacks",
        "extras": "mock_extras",
        "task_name": "mock_task_name",
        "tags": "mock_tags"
    }

    # Call the function
    log_hyperparameters(object_dict)

    # Check if OmegaConf.to_container was called with the correct arguments
    mock_to_container.assert_called_once_with(mock_cfg, resolve=True)

    # Check if the warning was not called
    mock_warning.assert_not_called()

    # Now test the case where the logger is not present
    mock_trainer.logger = None

    # Call the function again
    log_hyperparameters(object_dict)

    # Check if the warning was called
    mock_warning.assert_called_once_with("Logger not found! Skipping hyperparameter logging...")

if __name__ == "__main__":
    pytest.main()
