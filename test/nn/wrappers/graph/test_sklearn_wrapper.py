# test_wrappers.py
import numpy as np
import torch
import pytest

# Adjust these imports according to your project structure
from topobench.nn.wrappers.graph.sklearn.sklearn_wrappers.base import BaseWrapper
from topobench.nn.wrappers.graph.sklearn import ClassifierWrapper
from topobench.nn.wrappers.graph.sklearn import RegressorWrapper


class DummyClassifierBackbone:
    """
    Fake sklearn-style classifier:
    - fit(X, y) stores X, y, and the number of classes
    - predict_proba(X) returns a uniform probability distribution over classes
    """

    def __init__(self):
        self.X_fit = None
        self.y_fit = None
        self.fit_called = False
        self.num_classes_ = None

    def fit(self, X, y):
        self.X_fit = X
        self.y_fit = y
        self.fit_called = True
        self.num_classes_ = len(np.unique(y))

    def predict_proba(self, X):
        # Returns uniform probabilities with shape (len(X), num_classes_)
        assert self.num_classes_ is not None, "fit must be called before predict_proba."
        n = len(X)
        probs = np.ones((n, self.num_classes_), dtype=float) / self.num_classes_
        return probs


class DummyRegressorBackbone:
    """
    Fake sklearn-style regressor:
    - fit(X, y) stores X and y
    - predict(X) always returns the mean of y_fit
    """

    def __init__(self):
        self.X_fit = None
        self.y_fit = None
        self.fit_called = False
        self.mean_y_ = 0.0

    def fit(self, X, y):
        self.X_fit = X
        self.y_fit = y
        self.fit_called = True
        self.mean_y_ = float(np.mean(y))

    def predict(self, X):
        # Always returns the mean with shape (len(X),)
        return np.full(shape=(len(X),), fill_value=self.mean_y_, dtype=float)


class DummyClassifierBackbone:
    """
    Fake sklearn-style classifier:
    - fit(X, y) stores X, y, and the number of classes
    - predict_proba(X) returns a uniform probability distribution over classes
    """

    def __init__(self):
        self.X_fit = None
        self.y_fit = None
        self.fit_called = False
        self.num_classes_ = None

    def fit(self, X, y):
        self.X_fit = X
        self.y_fit = y
        self.fit_called = True
        self.num_classes_ = len(np.unique(y))

    def predict_proba(self, X):
        # Returns uniform probabilities with shape (len(X), num_classes_)
        assert self.num_classes_ is not None, "fit must be called before predict_proba."
        n = len(X)
        probs = np.ones((n, self.num_classes_), dtype=float) / self.num_classes_
        return probs


def test_classifier_init_targets():
    y_train = np.array([0, 1, 1, 2, 2, 2])

    backbone = DummyClassifierBackbone()
    wrapper = ClassifierWrapper(backbone=backbone)

    wrapper._init_targets(y_train)

    # sorted unique classes
    assert np.array_equal(wrapper.classes_, np.array([0, 1, 2]))
    assert wrapper.num_classes_ == 3

    # expected uniform distribution
    expected_uniform = np.ones(3) / 3
    assert np.allclose(wrapper.uniform_, expected_uniform)


def test_classifier_get_predictions_respects_test_mask_and_shape():
    torch.manual_seed(0)
    np.random.seed(0)

    num_nodes = 5
    num_features = 4

    # Node features (content irrelevant, only shape matters)
    node_features = np.random.randn(num_nodes, num_features)

    # Example: test nodes 1 and 3
    test_mask = np.array([False, True, False, True, False])

    # Training targets, used only to initialize class info
    y_train = np.array([0, 1, 2, 1, 0])

    backbone = DummyClassifierBackbone()
    wrapper = ClassifierWrapper(backbone=backbone)

    # Initialize classes
    wrapper._init_targets(y_train)
    backbone.fit(node_features, y_train)  # Fit called for consistency

    prob_logits = wrapper.get_predictions(node_features=node_features, test_mask=test_mask)

    # Must be a torch tensor with shape (num_nodes, num_classes)
    assert isinstance(prob_logits, torch.Tensor)
    assert prob_logits.shape == (num_nodes, wrapper.num_classes_)

    # Nodes outside test_mask must be all zeros
    non_test_idx = np.where(~test_mask)[0]
    assert torch.all(prob_logits[non_test_idx] == 0)

    # Test nodes must have uniform probabilities
    test_idx = np.where(test_mask)[0]
    for idx in test_idx:
        row = prob_logits[idx].numpy()
        expected = np.ones(wrapper.num_classes_) / wrapper.num_classes_
        assert np.allclose(row, expected)


def test_regressor_init_targets_sets_global_mean():
    y_train = np.array([1.0, 2.0, 3.0, 4.0])

    backbone = DummyRegressorBackbone()
    wrapper = RegressorWrapper(backbone=backbone)

    wrapper._init_targets(y_train)

    expected_mean = float(np.mean(y_train))

    # global_mean_ must be a torch tensor, and its value must equal the mean
    assert isinstance(wrapper.global_mean_, torch.Tensor)
    assert float(wrapper.global_mean_.item()) == pytest.approx(expected_mean)


def test_regressor_get_predictions_respects_test_mask_and_shape():
    torch.manual_seed(0)
    np.random.seed(0)

    num_nodes = 6
    num_features = 3

    node_features = np.random.randn(num_nodes, num_features)

    test_mask = np.array([True, False, True, False, False, True])

    y_train = np.array([10.0, 12.0, 14.0, 16.0])
    backbone = DummyRegressorBackbone()
    wrapper = RegressorWrapper(backbone=backbone)

    wrapper._init_targets(y_train)

    # Fit called for consistency with expected forward logic
    backbone.fit(node_features[test_mask], y_train[: test_mask.sum()])

    preds = wrapper.get_predictions(node_features=node_features, test_mask=test_mask)

    # Must be a torch tensor with shape (num_nodes, 1)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (num_nodes, 1)

    # Non-test nodes must be zero
    non_test_idx = np.where(~test_mask)[0]
    assert torch.all(preds[non_test_idx] == 0)

    # Test nodes must equal mean_y_ from the backbone
    test_idx = np.where(test_mask)[0]
    expected_val = backbone.mean_y_
    for idx in test_idx:
        assert float(preds[idx].item()) == pytest.approx(expected_val)
