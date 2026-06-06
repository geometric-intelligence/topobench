"""Tests for KeepSelectedTargetIndices transform."""

import pytest
import torch
from torch_geometric.data import Data

from topobench.transforms.data_manipulations.keep_selected_target_indices import (
    KeepSelectedTargetIndices,
)


class TestKeepSelectedTargetIndices:
    """Unit tests for KeepSelectedTargetIndices."""

    def test_repr(self):
        """__repr__ includes class name and parameters."""
        t = KeepSelectedTargetIndices(target_indices=[0, 2])
        r = repr(t)
        assert "KeepSelectedTargetIndices" in r
        assert "keep_selected_target_indices" in r

    def test_init_stores_parameters(self):
        """__init__ stores kwargs in self.parameters."""
        t = KeepSelectedTargetIndices(target_indices=[1, 3], extra_param="foo")
        assert t.parameters["target_indices"] == [1, 3]
        assert t.parameters["extra_param"] == "foo"

    def test_forward_selects_columns(self):
        """forward() keeps only the requested target columns."""
        t = KeepSelectedTargetIndices(target_indices=[0, 2])
        data = Data(y=torch.tensor([[10, 20, 30], [40, 50, 60]]))

        out = t(data)

        # Should keep columns 0 and 2 → shape [2, 2], then squeeze(0)
        assert out.y.shape[1] == 2
        assert torch.equal(out.y[:, 0], torch.tensor([10, 40]))
        assert torch.equal(out.y[:, 1], torch.tensor([30, 60]))

    def test_forward_single_target(self):
        """forward() with a single index squeezes correctly."""
        t = KeepSelectedTargetIndices(target_indices=[1])
        y = torch.arange(12).reshape(4, 3)
        data = Data(y=y)

        out = t(data)

        # Column 1 selected; squeeze(0) doesn't change shape for [4, 1]
        assert out.y.shape[0] == 4

    def test_call_delegates_to_forward(self):
        """__call__ is equivalent to calling forward."""
        t = KeepSelectedTargetIndices(target_indices=[0])
        data = Data(y=torch.tensor([[1, 2, 3], [4, 5, 6]]))
        out = t(data)
        assert hasattr(out, "y")
