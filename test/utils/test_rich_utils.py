"""Unit tests for rich_utils."""
import pytest
from unittest.mock import MagicMock, patch
from topobench.utils.rich_utils import enforce_tags, print_config_tree
from omegaconf import DictConfig

@patch("topobench.utils.rich_utils.pylogger.RankedLogger.info")
@patch("topobench.utils.rich_utils.rich.tree.Tree")
@patch("topobench.utils.rich_utils.rich.syntax.Syntax")
@patch("topobench.utils.rich_utils.rich.print")
@patch("topobench.utils.rich_utils.Path.write_text")
@patch("topobench.utils.rich_utils.HydraConfig.get")
def test_print_config_tree(mock_hydra_config_get, mock_write_text, mock_rich_print, mock_syntax, mock_tree, mock_info):
    '''Test the print_config_tree function.

    Parameters
    ----------
    mock_hydra_config_get : MagicMock
        Mock of HydraConfig.get.
    mock_write_text : MagicMock
        Mock of Path.write_text.
    mock_rich_print : MagicMock
        Mock of rich.print.
    mock_syntax : MagicMock
        Mock of rich.syntax.Syntax.
    mock_tree : MagicMock
        Mock of rich.tree.Tree.
    mock_info : MagicMock
        Mock of pylogger.RankedLogger.info.
    '''
    # Mock the input DictConfig
    mock_cfg = DictConfig({
        "data": "mock_data",
        "model": "mock_model",
        "callbacks": "mock_callbacks",
        "logger": "mock_logger",
        "trainer": "mock_trainer",
        "paths": {"output_dir": "mock_output_dir"},
        "extras": "mock_extras"
    })

    # Mock the HydraConfig.get return value
    mock_hydra_config_get.return_value = {"run": {"dir": "mock_dir"}}

    # Call the function with save_to_file=False
    print_config_tree(mock_cfg, save_to_file=False)

    # Check if rich.print was called
    assert mock_rich_print.called

    # Check if the info logger was not called
    mock_info.assert_not_called()

    with pytest.raises(FileNotFoundError):
        # Call the function with save_to_file=True
        print_config_tree(mock_cfg, save_to_file=True)


@patch("topobench.utils.rich_utils.HydraConfig")
@patch("topobench.utils.rich_utils.Prompt.ask")
@patch("topobench.utils.rich_utils.pylogger.RankedLogger.warning")
@patch("topobench.utils.rich_utils.pylogger.RankedLogger.info")
@patch("topobench.utils.rich_utils.rich.print")
def test_enforce_tags_no_tags(mock_rich_print, mock_info, mock_warning, mock_prompt_ask, mock_hydra_config):
    """Test the enforce_tags function when no tags are provided in the config.

    Parameters
    ----------
    mock_rich_print : MagicMock
        Mock of rich.print.
    mock_info : MagicMock
        Mock of pylogger.RankedLogger.info.
    mock_warning : MagicMock
        Mock of pylogger.RankedLogger.warning.
    mock_prompt_ask : MagicMock
        Mock of Prompt.ask.
    mock_hydra_config : MagicMock
        Mock of HydraConfig.
    """
    # Mock the input DictConfig without tags
    mock_cfg = DictConfig({
        "paths": {"output_dir": "mock_output_dir"}
    })

    # Mock the HydraConfig
    mock_hydra_config().cfg.hydra.job = {}

    # Mock the Prompt.ask return value
    mock_prompt_ask.return_value = "test_tag"

    # Call the function with save_to_file=False
    enforce_tags(mock_cfg, save_to_file=False)

    # Check if the warning was called
    mock_warning.assert_called_once_with("No tags provided in config. Prompting user to input tags...")

    # Check if the Prompt.ask was called
    mock_prompt_ask.assert_called_once_with("Enter a list of comma separated tags", default="dev")

    # Check if the info logger was called
    mock_info.assert_called_once_with("Tags: ['test_tag']")

    # Check if the tags were added to the config
    assert mock_cfg.tags == ["test_tag"]

    # Check if rich.print was not called
    mock_rich_print.assert_not_called()


if __name__ == "__main__":
    pytest.main()
