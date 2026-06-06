"""Evaluators for model evaluation."""

from torchmetrics.classification import (
    AUROC,
    Accuracy,
    ConfusionMatrix,
    F1Score,
    Precision,
    Recall,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)

from .metrics import ExampleRegressionMetric

# Define metrics
METRICS = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1": F1Score,
    "auroc": AUROC,
    "f1_macro": F1Score,
    "f1_weighted": F1Score,
    "confusion_matrix": ConfusionMatrix,
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "rmse": MeanSquaredError,  # We'll configure this with squared=False
    "r2": R2Score,
    "example": ExampleRegressionMetric,
}

from .base import AbstractEvaluator  # noqa: E402
from .evaluator import TBEvaluator  # noqa: E402

__all__ = [
    "METRICS",
    "AbstractEvaluator",
    "TBEvaluator",
]
