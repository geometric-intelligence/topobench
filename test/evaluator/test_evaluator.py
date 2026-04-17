"""Test the TBEvaluator class."""
import pytest
import torch
from topobench.evaluator import TBEvaluator


class TestTBEvaluator:
    """Test the TBEvaluator class."""

    def setup_method(self):
        """Setup the test."""
        self.classification_metrics = ["accuracy", "precision", "recall", "auroc"]
        self.evaluator_classification = TBEvaluator(
            task="classification",
            num_classes=3,
            metrics=self.classification_metrics
        )
        self.regression_metrics = ["example", "mae", "mse", "rmse", "r2"]
        self.evaluator_regression = TBEvaluator(
            task="regression",
            num_classes=1,
            metrics=self.regression_metrics
        )

    def test_repr(self):
        """Test the __repr__ method."""
        repr_str = self.evaluator_classification.__repr__()
        assert "TBEvaluator" in repr_str
        assert "classification" in repr_str

        repr_str = self.evaluator_regression.__repr__()
        assert "TBEvaluator" in repr_str
        assert "regression" in repr_str

    def test_classification_update_and_compute(self):
        """Test the update and compute methods for classification."""
        # Create deterministic data for testing
        logits = torch.tensor([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])
        labels = torch.tensor([0, 1, 2])

        self.evaluator_classification.update({"logits": logits, "labels": labels})
        out = self.evaluator_classification.compute()

        # Check all metrics are present
        for metric in self.classification_metrics:
            assert metric in out, f"Metric {metric} not found in output"

        # Check accuracy is 1.0 (perfect prediction)
        assert out["accuracy"] == pytest.approx(1.0, abs=0.01)

    def test_regression_update_and_compute(self):
        """Test the update and compute methods for regression."""
        # Create deterministic data
        logits = torch.tensor([[1.0], [2.0], [3.0]])
        labels = torch.tensor([1.1, 2.1, 3.1])

        self.evaluator_regression.update({"logits": logits, "labels": labels})
        out = self.evaluator_regression.compute()

        # Check all metrics are present
        for metric in self.regression_metrics:
            assert metric in out, f"Metric {metric} not found in output"

        # MAE should be approximately 0.1
        assert out["mae"] < 0.2

    def test_regression_with_different_shapes(self):
        """Test regression with different tensor shapes."""
        # Test with 2D predictions (required shape for regression)
        logits = torch.randn(10, 1)
        labels = torch.randn(10)

        self.evaluator_regression.update({"logits": logits, "labels": labels})
        out = self.evaluator_regression.compute()
        assert "mae" in out

    def test_binary_classification(self):
        """Test binary classification (num_classes=2)."""
        evaluator = TBEvaluator(
            task="classification",
            num_classes=2,
            metrics=["accuracy", "f1", "precision", "recall"]
        )

        logits = torch.tensor([[2.0, 0.1], [0.1, 2.0], [2.0, 0.1]])
        labels = torch.tensor([0, 1, 0])

        evaluator.update({"logits": logits, "labels": labels})
        out = evaluator.compute()

        assert out["accuracy"] == pytest.approx(1.0, abs=0.01)

    def test_multiclass_classification(self):
        """Test multiclass classification with more classes."""
        evaluator = TBEvaluator(
            task="classification",
            num_classes=5,
            metrics=["accuracy", "f1_macro", "f1_weighted"]
        )

        logits = torch.randn(20, 5)
        labels = torch.randint(0, 5, (20,))

        evaluator.update({"logits": logits, "labels": labels})
        out = evaluator.compute()

        assert "accuracy" in out
        assert "f1_macro" in out
        assert "f1_weighted" in out

    def test_confusion_matrix_metric(self):
        """Test that confusion matrix metric works."""
        evaluator = TBEvaluator(
            task="classification",
            num_classes=3,
            metrics=["accuracy", "confusion_matrix"]
        )

        logits = torch.tensor([[2.0, 0.1, 0.1], [0.1, 2.0, 0.1], [0.1, 0.1, 2.0]])
        labels = torch.tensor([0, 1, 2])

        evaluator.update({"logits": logits, "labels": labels})
        out = evaluator.compute()

        assert "confusion_matrix" in out
        # Should be identity matrix for perfect prediction
        cm = out["confusion_matrix"]
        assert cm.shape == (3, 3)

    def test_not_implemented_task(self):
        """Test error handling for invalid tasks."""
        with pytest.raises(ValueError, match="Invalid task"):
            TBEvaluator(task="wrong_task", num_classes=2, metrics=["accuracy"])

    def test_multilabel_not_implemented(self):
        """Test that multilabel classification raises NotImplementedError."""
        evaluator = TBEvaluator(
            task="multilabel classification",
            num_classes=3,
            metrics=["accuracy"]
        )

        with pytest.raises(NotImplementedError, match="Multilabel classification"):
            evaluator.update({
                "logits": torch.tensor([[1, 0, 0], [0, 1, 1]]),
                "labels": torch.tensor([[1, 1, 0], [0, 1, 1]])
            })

    def test_reset(self):
        """Test the reset method."""
        # Update with some data
        self.evaluator_regression.update({
            "logits": torch.randn(10, 1),
            "labels": torch.randn(10)
        })
        out_before = self.evaluator_regression.compute()

        # Reset
        self.evaluator_regression.reset()

        # Update with new data
        self.evaluator_regression.update({
            "logits": torch.randn(10, 1),
            "labels": torch.randn(10)
        })
        out_after = self.evaluator_regression.compute()

        # Results should be different (computed on different data)
        assert out_before["mae"] != out_after["mae"]

    def test_rmse_configuration(self):
        """Test that RMSE is correctly configured with squared=False."""
        """Test that RMSE is correctly configured with squared=False."""
        evaluator = TBEvaluator(
            task="regression",
            num_classes=1,
            metrics=["mse", "rmse"]
        )

        # Create data where we know MSE and RMSE
        # If predictions = [1, 2, 3] and labels = [1, 2, 3], both should be 0
        logits = torch.tensor([[1.0], [2.0], [3.0]])
        labels = torch.tensor([1.0, 2.0, 3.0])

        evaluator.update({"logits": logits, "labels": labels})
        out = evaluator.compute()

        # Both should be very close to 0
        assert out["mse"] < 0.01
        assert out["rmse"] < 0.01

        # For non-perfect predictions
        evaluator.reset()
        logits = torch.tensor([[1.0], [2.0]])
        labels = torch.tensor([2.0, 3.0])  # Off by 1 each

        evaluator.update({"logits": logits, "labels": labels})
        out = evaluator.compute()

        # MSE should be 1.0, RMSE should also be 1.0
        assert out["mse"] == pytest.approx(1.0, abs=0.01)
        assert out["rmse"] == pytest.approx(1.0, abs=0.01)

    def test_f1_variants(self):
        """Test different F1 score variants (macro, weighted)."""
        evaluator = TBEvaluator(
            task="classification",
            num_classes=3,
            metrics=["f1", "f1_macro", "f1_weighted"]
        )

        logits = torch.randn(30, 3)
        labels = torch.randint(0, 3, (30,))

        evaluator.update({"logits": logits, "labels": labels})
        out = evaluator.compute()

        assert "f1" in out
        assert "f1_macro" in out
        assert "f1_weighted" in out

    def test_multiple_updates_before_compute(self):
        """Test multiple updates before computing metrics."""
        evaluator = TBEvaluator(
            task="classification",
            num_classes=2,
            metrics=["accuracy"]
        )

        # Multiple updates (accumulating)
        for _ in range(5):
            logits = torch.randn(10, 2)
            labels = torch.randint(0, 2, (10,))
            evaluator.update({"logits": logits, "labels": labels})

        # Compute accumulated metrics
        out = evaluator.compute()
        assert "accuracy" in out
        assert 0 <= out["accuracy"] <= 1
