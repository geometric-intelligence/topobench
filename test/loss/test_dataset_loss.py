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

        # multioutput classification forward
        dataset_moc = DatasetLoss({"task": "multioutput classification", "loss_type": "mse"})
        logits_moc = torch.randn(4, 3)
        labels_moc = torch.randint(0, 3, (4, 3)).float()
        out_moc = dataset_moc.forward({"logits": logits_moc, "labels": labels_moc}, batch)
        assert out_moc.item() >= 0


class TestDatasetLossMultioutput:
    """Additional tests for multioutput classification in DatasetLoss."""

    def test_init_multioutput_classification(self):
        """DatasetLoss accepts task=multioutput classification with loss_type=mse."""
        loss = DatasetLoss({"task": "multioutput classification", "loss_type": "mse"})
        assert loss.task == "multioutput classification"
        assert isinstance(loss.criterion, torch.nn.MSELoss)

    def test_repr_multioutput(self):
        """__repr__ includes task and loss_type."""
        loss = DatasetLoss({"task": "multioutput classification", "loss_type": "mse"})
        r = repr(loss)
        assert "multioutput classification" in r
        assert "mse" in r

    def test_forward_criterion_multioutput(self):
        """forward_criterion casts targets to float and returns a scalar loss."""
        loss = DatasetLoss({"task": "multioutput classification", "loss_type": "mse"})
        logits = torch.randn(8, 4)
        target = torch.randint(0, 4, (8, 4))

        result = loss.forward_criterion(logits, target)

        assert result.ndim == 0  # scalar
        assert result.item() >= 0.0

    def test_forward_with_model_out_dict(self):
        """forward() unpacks model_out dict correctly."""
        loss = DatasetLoss({"task": "multioutput classification", "loss_type": "mse"})
        batch = torch_geometric.data.Data()
        model_out = {
            "logits": torch.randn(5, 3),
            "labels": torch.randint(0, 3, (5, 3)),
        }

        result = loss.forward(model_out, batch)

        assert result.item() >= 0.0
