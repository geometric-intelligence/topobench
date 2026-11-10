"""Unit tests for TBModel.process_outputs."""

import pytest
import torch
from unittest.mock import MagicMock

from topobench.model.model import TBModel


def make_model(task_level):
    """Instantiate TBModel with mocked dependencies for a given task_level.

    Parameters
    ----------
    task_level : str
        The task level to assign to the readout mock.

    Returns
    -------
    TBModel
        A TBModel instance with mocked backbone, readout, loss, evaluator and optimizer.
    """
    backbone = MagicMock()
    backbone.parameters.return_value = []

    readout = MagicMock()
    readout.task_level = task_level
    readout.parameters.return_value = []

    loss = MagicMock()

    evaluator = MagicMock()

    optimizer = MagicMock()
    optimizer.configure_optimizer.return_value = {"optimizer": MagicMock()}

    feature_encoder = MagicMock()
    feature_encoder.parameters.return_value = []

    model = TBModel(
        backbone=backbone,
        readout=readout,
        loss=loss,
        feature_encoder=feature_encoder,
        evaluator=evaluator,
        optimizer=optimizer,
    )
    return model


class TestProcessOutputs:
    """Tests for TBModel.process_outputs covering all branches."""

    def _make_batch(self, n=10):
        """Create a simple batch mock with train/val/test masks.

        Parameters
        ----------
        n : int
            Number of nodes.

        Returns
        -------
        MagicMock
            Batch mock with boolean masks.
        """
        batch = MagicMock()
        # First 6 are train, next 2 val, last 2 test
        batch.train_mask = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.bool)
        batch.val_mask = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=torch.bool)
        batch.test_mask = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=torch.bool)
        return batch

    def _make_model_out(self, n=10):
        """Create a sample model output dict.

        Parameters
        ----------
        n : int
            Number of nodes.

        Returns
        -------
        dict
            Model output with logits, labels, and an extra key.
        """
        return {
            "logits": torch.randn(n, 3),
            "labels": torch.randint(0, 3, (n,)),
            "x_0": torch.randn(n, 8),
        }

    # ------------------------------------------------------------------
    # node-level masking
    # ------------------------------------------------------------------

    def test_node_training_filters_by_train_mask(self):
        """process_outputs with task_level='node' and state 'Training' filters by train_mask."""
        model = make_model("node")
        model.state_str = "Training"
        batch = self._make_batch()
        model_out = self._make_model_out()

        result = model.process_outputs(model_out, batch)

        n_train = batch.train_mask.sum().item()
        assert result["logits"].shape[0] == n_train
        assert result["labels"].shape[0] == n_train
        # non-masked keys are untouched
        assert result["x_0"].shape[0] == 10

    def test_node_validation_filters_by_val_mask(self):
        """process_outputs with task_level='node' and state 'Validation' filters by val_mask."""
        model = make_model("node")
        model.state_str = "Validation"
        batch = self._make_batch()
        model_out = self._make_model_out()

        result = model.process_outputs(model_out, batch)

        n_val = batch.val_mask.sum().item()
        assert result["logits"].shape[0] == n_val
        assert result["labels"].shape[0] == n_val

    def test_node_test_filters_by_test_mask(self):
        """process_outputs with task_level='node' and state 'Test' filters by test_mask."""
        model = make_model("node")
        model.state_str = "Test"
        batch = self._make_batch()
        model_out = self._make_model_out()

        result = model.process_outputs(model_out, batch)

        n_test = batch.test_mask.sum().item()
        assert result["logits"].shape[0] == n_test
        assert result["labels"].shape[0] == n_test

    def test_node_invalid_state_raises_value_error(self):
        """process_outputs with task_level='node' and an invalid state_str raises ValueError."""
        model = make_model("node")
        model.state_str = "Invalid"
        batch = self._make_batch()
        model_out = self._make_model_out()

        with pytest.raises(ValueError, match="Invalid state_str"):
            model.process_outputs(model_out, batch)

    # ------------------------------------------------------------------
    # no-op task levels
    # ------------------------------------------------------------------

    def test_graph_level_returns_unchanged(self):
        """process_outputs with task_level='graph' returns model_out unchanged."""
        model = make_model("graph")
        model.state_str = "Training"
        batch = self._make_batch()
        model_out = self._make_model_out()
        original_logits = model_out["logits"].clone()

        result = model.process_outputs(model_out, batch)

        assert result["logits"].shape == original_logits.shape
        assert torch.equal(result["logits"], original_logits)

    def test_node_inductive_returns_unchanged(self):
        """process_outputs with task_level='node_inductive' returns model_out unchanged (inductive bug-fix path)."""
        model = make_model("node_inductive")
        model.state_str = "Training"
        batch = self._make_batch()
        model_out = self._make_model_out()
        original_logits = model_out["logits"].clone()

        result = model.process_outputs(model_out, batch)

        assert result["logits"].shape == original_logits.shape
        assert torch.equal(result["logits"], original_logits)


def _bare_model():
    """TBModel without running ``__init__`` (no hyperparameters needed)."""
    from torch import nn

    module = TBModel.__new__(TBModel)
    nn.Module.__init__(module)
    return module


def test_plateau_scheduler_steps_with_validation():
    """Plateau schedulers step only on epochs that produce the metric."""
    from torch import nn

    from topobench.optimizer import TBOptimizer

    class Holder:
        check_val_every_n_epoch = 5

    module = _bare_model()
    module.backbone, module.readout = nn.Linear(1, 1), nn.Identity()
    module.feature_encoder = nn.Identity()
    module.optimizer = TBOptimizer(
        "AdamW",
        {"lr": 1e-3},
        scheduler={
            "scheduler_id": "ReduceLROnPlateau",
            "monitor": "val/accuracy",
            "scheduler_params": {"mode": "max"},
        },
    )
    module._trainer = Holder()
    config = module.configure_optimizers()
    assert config["lr_scheduler"]["frequency"] == 5
    assert "ReduceLROnPlateau" in repr(module.optimizer)


def _cpu_autocast_enabled():
    # ``torch.is_autocast_enabled(device_type)`` is unavailable before torch 2.4.
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:
        return torch.is_autocast_cpu_enabled()


@pytest.mark.parametrize("training", [False, True])
def test_evaluation_can_disable_autocast(training):
    """``evaluation_autocast=False`` evaluates without autocast."""
    from torch import nn

    module = _bare_model()
    module.evaluation_autocast, module.training = False, training
    module.backbone, module.state_str = nn.Identity(), "Validation"
    module._device = torch.device("cpu")
    seen = []

    def forward(batch):
        seen.append(_cpu_autocast_enabled())
        return {}

    module.forward = forward
    module.process_outputs = lambda model_out, batch: model_out
    module.loss = lambda model_out, batch: model_out
    module.evaluator = type("E", (), {"update": lambda self, out: None})()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        module.model_step({})
    assert seen == [training]


def test_layer_compilation_preserves_state_keys(monkeypatch):
    """``compile_scope=layers`` compiles in place, once per layer."""
    from types import SimpleNamespace

    from torch import nn

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])

        def compilable_layers(self):
            return list(self.layers)

    wrapped = TBModel(
        Backbone(),
        SimpleNamespace(task_level="graph"),
        None,
        compile=True,
        compile_scope="layers",
    )
    before = set(wrapped.state_dict())
    calls = []

    def compile_layer(layer):
        calls.append(layer)
        layer._compiled_call_impl = lambda *args: None

    monkeypatch.setattr(torch.nn.Module, "compile", compile_layer)
    wrapped.configure_compilation()
    wrapped.configure_compilation()
    assert len(calls) == 2
    assert set(wrapped.state_dict()) == before
    assert all(not key.startswith("backbone._orig_mod") for key in before)


def test_layer_compilation_requires_compilable_layers():
    """Backbones without ``compilable_layers`` reject ``compile_scope=layers``."""
    from types import SimpleNamespace

    from torch import nn

    wrapped = TBModel(
        nn.Linear(2, 2),
        SimpleNamespace(task_level="graph"),
        None,
        compile=True,
        compile_scope="layers",
    )
    with pytest.raises(ValueError, match="compilable_layers"):
        wrapped.configure_compilation()


def test_host_fields_are_copied_before_transfer():
    """Backbones listing ``host_fields`` receive CPU copies of those fields."""
    from torch import nn
    from torch_geometric.data import Data

    module = _bare_model()
    module.backbone = nn.Identity()
    module.backbone.host_fields = ("counts", "absent")
    batch = module.on_before_batch_transfer(
        Data(counts=torch.tensor([3, 4]), x=torch.ones(2, 1)), 0
    )
    assert batch.host_copies.keys() == {"counts"}
    assert batch.host_copies["counts"].tolist() == [3, 4]
    plain = _bare_model()
    plain.backbone = nn.Identity()
    assert "host_copies" not in plain.on_before_batch_transfer(Data(x=torch.ones(1)), 0)
