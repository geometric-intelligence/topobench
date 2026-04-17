"""Test data field manipulation transforms."""

import torch
from torch_geometric.data import Data
from topobench.transforms.data_manipulations import KeepSelectedDataFields


class TestDataFieldTransforms:
    """Test data field manipulation transforms."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        x = torch.tensor([
            [2, 2],
            [2.2, 2],
            [2.1, 1.5],
        ])

        self.data = Data(
            x=x,
            num_nodes=len(x),
            field_1="some text",
            field_2=x.clone(),
            preserve_1=123,
            preserve_2=torch.tensor((1, 2, 3)),
        )

        self.keep_selected_fields = KeepSelectedDataFields(
            base_fields=["x", "num_nodes"],
            preserved_fields=["preserve_1", "preserve_2"],
        )

    def test_keep_selected_data_fields(self):
        """Test keeping selected fields.

        Verifies that only specified fields are kept and others are removed.
        """
        data = self.keep_selected_fields(self.data.clone())
        expected_fields = set(
            self.keep_selected_fields.parameters["base_fields"] +
            self.keep_selected_fields.parameters["preserved_fields"]
        )
        assert set(data.keys()) == expected_fields, "Some fields are not deleted"
