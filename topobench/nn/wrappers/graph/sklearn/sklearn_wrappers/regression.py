from typing import Any

import numpy as np
import torch

from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.base import (
    BaseWrapper,
)


class RegressorWrapper(BaseWrapper):
    """
    Regression version of ClassificationWrapper:
    - always returns a scalar prediction per node
    - falls back to the global mean if no neighbours
    - trains backbone per‐node on its sampled neighbours
    """

    def __init__(self, backbone: Any, sampler: Any | None = None, **kwargs):
        super().__init__(backbone, sampler=sampler, **kwargs)
        self.global_mean_: float = 0.0

    def _init_targets(self, y_train: np.ndarray) -> None:
        self.global_mean_ = torch.tensor(float(np.mean(y_train)))

    def get_predictions(self, node_features, test_mask):
        # Predict probabilities for the whole dataset (to allow compatibility with the rest of the code)
        output = self.backbone.predict(node_features[test_mask])

        preds_tensor = torch.from_numpy(output).float().view(-1, 1)

        # Prepare the output
        preds = torch.zeros(len(node_features), 1)
        preds[test_mask] = preds_tensor

        return preds
