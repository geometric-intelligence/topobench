""" Test the TBEvaluator class."""
import pytest
import torch
import torch_geometric

from topobench.loss.dataset import DatasetLoss

class TestDatasetLoss:
    """ Test the TBEvaluator class."""

    def setup_method(self):
        """ Setup the test."""
        dataset_loss = {"task": "classification", "loss_type": "cross_entropy"}
        self.dataset1 = DatasetLoss(dataset_loss)
        dataset_loss = {"task": "regression", "loss_type": "mse"}
        self.dataset2 = DatasetLoss(dataset_loss)
        dataset_loss = {"task": "regression", "loss_type": "mae"}
        self.dataset3 = DatasetLoss(dataset_loss)
        dataset_loss = {"task": "multilabel classification", "loss_type": "BCE"}
        self.dataset4 = DatasetLoss(dataset_loss)

        dataset_loss = {"task": "wrong", "loss_type": "wrong"}
        with pytest.raises(Exception):
            DatasetLoss(dataset_loss)

        dataset_loss = {"task": "classification", "loss_type": "cross_entropy"}
        self.dataset5 = DatasetLoss(dataset_loss)

        repr = self.dataset1.__repr__()
        assert repr == "DatasetLoss(task=classification, loss_type=cross_entropy)"

    def test_forward(self):
        """ Test the forward method."""
        batch = torch_geometric.data.Data()

        model_out = {"logits": torch.tensor([0.1, 0.2, 0.3]), "labels": torch.tensor([0.1, 0.2, 0.3])}
        out = self.dataset1.forward(model_out, batch)
        assert out.item() >= 0

        model_out = {"logits": torch.tensor([0.1, 0.2, 0.3]), "labels": torch.tensor([0.1, 0.2, 0.3])}
        out = self.dataset3.forward(model_out, batch)
        assert out.item() >= 0

        model_out = {"logits": torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]), "labels": torch.tensor([[0.1, float('nan'), 0.3], [0.1, 0.2, float('nan')]])}
        out = self.dataset4.forward(model_out, batch)
        assert out.item() >= 0

        self.dataset5.task = 'not defined'
        with pytest.raises(Exception):
            self.dataset5(model_out, batch)
