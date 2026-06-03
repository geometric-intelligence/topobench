import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseWrapper(torch.nn.Module, ABC):
    def __init__(self, backbone: Any, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.use_embeddings = kwargs.get("use_embeddings", True)
        self.use_node_features = kwargs.get("use_node_features", True)
        self.sampler = kwargs.get("sampler")

        assert self.use_embeddings or self.use_node_features, (
            "Either use_embeddings or use_node_features could be False, not both."
        )
        self.logger = kwargs.get("logger")
        # Initialize the counters
        self.num_no_neighbors = 0
        self.num_one_neighbor = 0
        self.num_all_same_label = 0
        self.num_all_feat_constant = 0
        self.num_model_trained = 0

    def fit(self, x: np.ndarray, y: np.ndarray):
        # common: store target dtype, do nothing else
        return self

    def log_model_stat(self, num_test_points):
        total_ratio = (
            self.num_no_neighbors / num_test_points
            + self.num_one_neighbor / num_test_points
            + self.num_all_feat_constant / num_test_points
            + self.num_model_trained / num_test_points
            + self.num_all_same_label / num_test_points
        )

        assert math.isclose(total_ratio, 1.0, rel_tol=1e-9, abs_tol=1e-6), (
            f"The sum of the ratios should be 1 (within tolerance), but got {total_ratio:.10f}"
        )
        self.logger(
            "test/no_neighbors",
            np.round((100 * self.num_no_neighbors / num_test_points), 2),
            prog_bar=True,
            on_step=False,
        )
        self.logger(
            "test/one_neighbor",
            np.round((100 * self.num_one_neighbor / num_test_points), 2),
            prog_bar=True,
            on_step=False,
        )
        self.logger(
            "test/all_features_constant",
            np.round((100 * self.num_all_feat_constant / num_test_points), 2),
            prog_bar=True,
            on_step=False,
        )
        self.logger(
            "test/num_all_same_label",
            np.round((100 * self.num_all_same_label / num_test_points), 2),
            prog_bar=True,
            on_step=False,
        )
        self.logger(
            "test/model_trained",
            np.round((100 * self.num_model_trained / num_test_points), 2),
            prog_bar=True,
            on_step=False,
        )

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        train_mask = batch.get("train_mask").cpu().numpy().copy()
        # val_mask = batch.get("val_mask").cpu().numpy().copy()
        test_mask = batch.get("test_mask").cpu().numpy().copy()

        # encoded node features
        rank0_features = []

        if self.use_embeddings:
            all_keys = batch.keys()
            # adding all the rank0 features
            rank0_features.extend([s for s in all_keys if s.startswith("x_")])
        else:
            rank0_features = ["x_0"]

        if self.use_node_features is False:
            rank0_features.remove("x0_0")  # remove x0_0 if using embeddings

        # Concatenate tensors along the column dimension (dim=1)
        tensors_to_concat = [batch[k] for k in rank0_features]
        tensor_features = torch.cat(tensors_to_concat, dim=1)

        node_features = tensor_features.cpu().numpy().copy()

        X_train = node_features[train_mask].copy()
        y_train = batch["y"][train_mask].cpu().numpy().copy()

        self._init_targets(y_train)

        # If sample is None training the network on the whole dataset
        self.backbone.fit(X_train, y_train)

        prob_logits = self.get_predictions(
            node_features=node_features, test_mask=test_mask
        ).to(batch["x_0"].device)

        self.num_model_trained += len(test_mask)

        num_test_points = test_mask.shape[0]

        # Log the metrics calculated within wrapper
        if self.logger is not None:
            self.log_model_stat(num_test_points)

        return {
            "labels": batch["y"],
            "batch_0": batch["batch_0"],
            "x_0": prob_logits,
        }

    @abstractmethod
    def get_predictions(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        """Implement forward pass"""

    @abstractmethod
    def _init_targets(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        """Implement forward pass"""
