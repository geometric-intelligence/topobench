"""Tests for the ``RenameFields`` transform."""

import warnings

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.rename_fields import RenameFields


def _data():
    return Data(x=torch.arange(6.0).reshape(3, 2), foo=torch.tensor([1, 2, 3]))


def test_init_validates_lengths():
    with pytest.raises(AssertionError):
        RenameFields(init_field_name=["a"], new_field_name=["a", "b"])


def test_repr_includes_field_names():
    t = RenameFields(
        init_field_name=["a"], new_field_name=["b"], note="hello"
    )
    text = repr(t)
    assert "RenameFields" in text
    assert "'a'" in text
    assert "'b'" in text


def test_rename_existing_field_moves_value():
    t = RenameFields(init_field_name=["foo"], new_field_name=["bar"])
    data = t(_data())
    assert hasattr(data, "bar")
    assert not hasattr(data, "foo")
    assert torch.equal(data.bar, torch.tensor([1, 2, 3]))


def test_rename_missing_field_is_no_op():
    t = RenameFields(init_field_name=["nope"], new_field_name=["bar"])
    data = t(_data())
    assert not hasattr(data, "bar")


def test_rename_overwrites_existing_target_with_warning():
    t = RenameFields(init_field_name=["foo"], new_field_name=["x"])
    data = _data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = t(data)
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert torch.equal(out.x, torch.tensor([1, 2, 3]))
