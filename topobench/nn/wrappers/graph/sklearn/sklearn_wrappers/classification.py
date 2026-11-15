import numpy as np
import torch

from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.base import (
    BaseWrapper,
)


class ClassifierWrapper(BaseWrapper):
    def _init_targets(self, y_train):
        self.classes_ = np.unique(y_train)
        self.num_classes_ = len(self.classes_)
        self.uniform_ = np.ones(len(self.classes_)) / len(self.classes_)

    def get_predictions(self, node_features, test_mask):
        output = self.backbone.predict_proba(node_features[test_mask])

        prob_tensor = torch.from_numpy(output).float()

        # Prepare the output
        prob_logits = torch.zeros(len(node_features), self.num_classes_)
        prob_logits[test_mask] = prob_tensor

        return prob_logits
