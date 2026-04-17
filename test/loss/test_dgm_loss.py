"""Test the DGMLoss class."""

import pytest
import torch
import torch_geometric
from unittest.mock import MagicMock
from topobench.loss.model.DGMLoss import DGMLoss

@pytest.fixture
def mock_batch():
    """
    Create a mock batch of data for testing.

    Returns
    -------
    MagicMock
        A mock object representing a batch of data.
    """
    batch = MagicMock(spec=torch_geometric.data.Data)
    batch.keys.return_value = ["logprobs_1", "logprobs_2", "model_state", "train_mask", "val_mask", "test_mask"]
    batch.model_state = "Training"
    batch.train_mask = torch.tensor([True, False])
    batch.val_mask = torch.tensor([False, True])
    batch.test_mask = torch.tensor([False, True])
    batch.logprobs_1 = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    batch.logprobs_2 = torch.tensor([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
    batch.__getitem__.side_effect = lambda key: getattr(batch, key)
    return batch

@pytest.fixture
def mock_model_out():
    """
    Create a mock model output for testing.

    Returns
    -------
    dict
        A dictionary containing mock logits and labels.
    """
    return {
        "logits": torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]]),
        "labels": torch.tensor([1, 0, 1])
    }

def test_dgm_loss_init():
    """
    Test the initialization of the DGMLoss class.

    This function tests the DGMLoss class to ensure that it initializes correctly
    with the default loss weight and that the average accuracy is set to None.
    """
    loss_fn = DGMLoss(loss_weight=0.7)
    assert loss_fn.loss_weight == 0.7
    assert loss_fn.avg_accuracy is None

def test_dgm_loss_repr():
    """Test the string representation of the DGMLoss class."""
    loss_fn = DGMLoss()
    assert repr(loss_fn) == "DGMLoss()"

def test_dgm_loss_forward(mock_batch, mock_model_out):
    """
    Test the forward pass of the DGMLoss function.

    Parameters
    ----------
    mock_batch : torch.Tensor
        A mock batch of input data.
    mock_model_out : torch.Tensor
        A mock output from the model.
    """

    loss_fn = DGMLoss()
    loss = loss_fn.forward(mock_model_out, mock_batch)
    assert isinstance(loss, torch.Tensor)

    # Check if the loss value is a scalar tensor
    assert loss.dim() == 0

def test_dgm_loss_forward_with_different_masks(mock_batch, mock_model_out):
    """
    Test the DGMLoss forward method with different model states.

    This function tests the `DGMLoss` forward method using different
    model states (Training, Validation, and Test) to ensure that the
    loss is computed correctly and returns a tensor in each case.

    Parameters
    ----------
    mock_batch : Mock
        A mock object representing the batch input to the model,
        with an attribute `model_state` that can be set to different
        states (Training, Validation, Test).
    mock_model_out : Mock
        A mock object representing the output of the model.

    Raises
    ------
    AssertionError
        If the loss computed is not an instance of `torch.Tensor`.
    """
    loss_fn = DGMLoss()

    # Test with training mask
    mock_batch.model_state = "Training"
    loss = loss_fn.forward(mock_model_out, mock_batch)
    assert isinstance(loss, torch.Tensor)

    # Test with validation mask
    mock_batch.model_state = "Validation"
    loss = loss_fn.forward(mock_model_out, mock_batch)
    assert isinstance(loss, torch.Tensor)

    # Test with test mask
    mock_batch.model_state = "Test"
    loss = loss_fn.forward(mock_model_out, mock_batch)
    assert isinstance(loss, torch.Tensor)

if __name__ == "__main__":
    pytest.main()
