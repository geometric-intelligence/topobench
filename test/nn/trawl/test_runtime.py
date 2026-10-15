"""Runtime optimizations must preserve checkpoint contracts."""

from types import SimpleNamespace

import torch

from topobench.model.model import TBModel

from .test_trawl import model, prepare


def test_layer_compilation_preserves_state_keys(monkeypatch):
    backbone = model([prepare()])
    wrapped = TBModel(
        backbone,
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
