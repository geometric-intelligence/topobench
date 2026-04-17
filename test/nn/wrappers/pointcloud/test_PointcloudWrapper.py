"""Tests for the PointcloudWrapper."""

from unittest.mock import Mock

import pytest
import torch

from topobench.nn.wrappers.pointcloud.pointcloud_wrapper import (
    PointcloudWrapper,
)


class TestPointcloudWrapper:
    """Tests for the PointcloudWrapper."""

    @pytest.fixture
    def mock_backbone(self):
        """Fixture to create a mock backbone model.

        Returns
        -------
        Mock
            A mock backbone model that returns embeddings.
        """
        backbone = Mock()
        backbone.return_value = torch.randn(10, 64)
        return backbone

    @pytest.fixture
    def wrapper_kwargs(self):
        """Fixture providing the required parameters for PointcloudWrapper.

        Returns
        -------
        dict
            Dictionary with required initialization parameters.
        """
        return {
            "out_channels": 64,
            "num_cell_dimensions": 0,  # For pointclouds, we work with 0-cells (nodes)
        }

    @pytest.fixture
    def pointcloud_wrapper(self, mock_backbone, wrapper_kwargs):
        """Fixture to create a PointcloudWrapper instance.

        Parameters
        ----------
        mock_backbone : Mock
            A fixture providing a mock backbone model.
        wrapper_kwargs : dict
            A fixture providing required initialization parameters.

        Returns
        -------
        PointcloudWrapper
            A PointcloudWrapper instance for testing.
        """
        wrapper = PointcloudWrapper(
            backbone=mock_backbone,
            **wrapper_kwargs
        )
        return wrapper

    @pytest.fixture
    def sample_batch(self):
        """Fixture to create a sample batch of pointcloud data.

        Returns
        -------
        Mock
            A mock batch object with pointcloud data.
        """
        batch = Mock()
        batch.x_0 = torch.randn(10, 3)  # 10 points with 3D coordinates
        batch.y = torch.randint(0, 5, (10,))  # Labels for 10 points
        batch.batch_0 = torch.zeros(10, dtype=torch.long)  # Single graph
        return batch

    @pytest.fixture
    def multi_graph_batch(self):
        """Fixture to create a batch with multiple pointclouds.

        Returns
        -------
        Mock
            A mock batch object with multiple pointclouds.
        """
        batch = Mock()
        batch.x_0 = torch.randn(20, 3)  # 20 points total
        batch.y = torch.randint(0, 5, (20,))  # Labels for 20 points
        batch.batch_0 = torch.cat([torch.zeros(10), torch.ones(10)]).long()  # Two pointclouds
        return batch

    def test_initialization(self, mock_backbone, wrapper_kwargs):
        """Test that PointcloudWrapper initializes correctly.

        Parameters
        ----------
        mock_backbone : Mock
            A fixture providing a mock backbone model.
        wrapper_kwargs : dict
            A fixture providing required initialization parameters.
        """
        wrapper = PointcloudWrapper(
            backbone=mock_backbone,
            **wrapper_kwargs
        )
        assert isinstance(wrapper, PointcloudWrapper)
        assert wrapper.backbone == mock_backbone

    def test_forward_returns_dict(self, pointcloud_wrapper, sample_batch):
        """Test that forward method returns a dictionary.

        Parameters
        ----------
        pointcloud_wrapper : PointcloudWrapper
            A fixture to create a PointcloudWrapper instance.
        sample_batch : Mock
            A fixture to create a sample batch of pointcloud data.
        """
        output = pointcloud_wrapper.forward(sample_batch)
        assert isinstance(output, dict)

    def test_forward_contains_required_keys(self, pointcloud_wrapper, sample_batch):
        """Test that forward output contains required keys.

        Parameters
        ----------
        pointcloud_wrapper : PointcloudWrapper
            A fixture to create a PointcloudWrapper instance.
        sample_batch : Mock
            A fixture to create a sample batch of pointcloud data.
        """
        output = pointcloud_wrapper.forward(sample_batch)

        assert "x_0" in output
        assert "labels" in output
        assert "batch_0" in output

    def test_forward_calls_backbone(self, pointcloud_wrapper, sample_batch):
        """Test that forward method calls the backbone with correct arguments.

        Parameters
        ----------
        pointcloud_wrapper : PointcloudWrapper
            A fixture to create a PointcloudWrapper instance.
        sample_batch : Mock
            A fixture to create a sample batch of pointcloud data.
        """
        pointcloud_wrapper.forward(sample_batch)

        pointcloud_wrapper.backbone.assert_called_once_with(sample_batch.x_0)

    def test_forward_output_x_0_shape(self, mock_backbone, wrapper_kwargs, sample_batch):
        """Test that x_0 output has the correct shape.

        Parameters
        ----------
        mock_backbone : Mock
            A fixture providing a mock backbone model.
        wrapper_kwargs : dict
            A fixture providing required initialization parameters.
        sample_batch : Mock
            A fixture to create a sample batch of pointcloud data.
        """
        expected_output = torch.randn(10, 64)
        mock_backbone.return_value = expected_output

        wrapper = PointcloudWrapper(
            backbone=mock_backbone,
            **wrapper_kwargs
        )
        output = wrapper.forward(sample_batch)

        assert output["x_0"].shape == expected_output.shape
        assert torch.equal(output["x_0"], expected_output)

    def test_forward_preserves_labels(self, pointcloud_wrapper, sample_batch):
        """Test that forward method preserves labels from batch.

        Parameters
        ----------
        pointcloud_wrapper : PointcloudWrapper
            A fixture to create a PointcloudWrapper instance.
        sample_batch : Mock
            A fixture to create a sample batch of pointcloud data.
        """
        output = pointcloud_wrapper.forward(sample_batch)

        assert torch.equal(output["labels"], sample_batch.y)

    def test_forward_preserves_batch_0(self, pointcloud_wrapper, sample_batch):
        """Test that forward method preserves batch_0 from batch.

        Parameters
        ----------
        pointcloud_wrapper : PointcloudWrapper
            A fixture to create a PointcloudWrapper instance.
        sample_batch : Mock
            A fixture to create a sample batch of pointcloud data.
        """
        output = pointcloud_wrapper.forward(sample_batch)

        assert torch.equal(output["batch_0"], sample_batch.batch_0)

    def test_forward_with_multiple_pointclouds(self, pointcloud_wrapper, multi_graph_batch):
        """Test forward pass with multiple pointclouds in a batch.

        Parameters
        ----------
        pointcloud_wrapper : PointcloudWrapper
            A fixture to create a PointcloudWrapper instance.
        multi_graph_batch : Mock
            A fixture to create a batch with multiple pointclouds.
        """
        output = pointcloud_wrapper.forward(multi_graph_batch)

        assert "x_0" in output
        assert "labels" in output
        assert "batch_0" in output
        assert torch.equal(output["batch_0"], multi_graph_batch.batch_0)

    def test_forward_with_different_feature_dimensions(self, mock_backbone, wrapper_kwargs):
        """Test forward pass with different input feature dimensions.

        Parameters
        ----------
        mock_backbone : Mock
            A fixture providing a mock backbone model.
        wrapper_kwargs : dict
            A fixture providing required initialization parameters.
        """
        wrapper = PointcloudWrapper(
            backbone=mock_backbone,
            **wrapper_kwargs
        )

        for input_dim in [3, 6, 128]:
            batch = Mock()
            batch.x_0 = torch.randn(10, input_dim)
            batch.y = torch.randint(0, 5, (10,))
            batch.batch_0 = torch.zeros(10, dtype=torch.long)

            mock_backbone.return_value = torch.randn(10, 64)
            output = wrapper.forward(batch)

            assert output["x_0"].shape == (10, 64)
            mock_backbone.assert_called_with(batch.x_0)

    def test_forward_with_different_num_points(self, mock_backbone, wrapper_kwargs):
        """Test forward pass with different numbers of points.

        Parameters
        ----------
        mock_backbone : Mock
            A fixture providing a mock backbone model.
        wrapper_kwargs : dict
            A fixture providing required initialization parameters.
        """
        wrapper = PointcloudWrapper(
            backbone=mock_backbone,
            **wrapper_kwargs
        )

        for num_points in [5, 10, 50, 100]:
            batch = Mock()
            batch.x_0 = torch.randn(num_points, 3)
            batch.y = torch.randint(0, 5, (num_points,))
            batch.batch_0 = torch.zeros(num_points, dtype=torch.long)

            mock_backbone.return_value = torch.randn(num_points, 64)
            output = wrapper.forward(batch)

            assert output["x_0"].shape[0] == num_points
            assert len(output["labels"]) == num_points
            assert len(output["batch_0"]) == num_points

    def test_forward_backbone_exception_propagates(self, mock_backbone, wrapper_kwargs, sample_batch):
        """Test that exceptions from backbone are propagated.

        Parameters
        ----------
        mock_backbone : Mock
            A fixture providing a mock backbone model.
        wrapper_kwargs : dict
            A fixture providing required initialization parameters.
        sample_batch : Mock
            A fixture to create a sample batch of pointcloud data.
        """
        mock_backbone.side_effect = RuntimeError("Backbone error")
        wrapper = PointcloudWrapper(
            backbone=mock_backbone,
            **wrapper_kwargs
        )

        with pytest.raises(RuntimeError, match="Backbone error"):
            wrapper.forward(sample_batch)

    def test_forward_with_empty_batch(self, mock_backbone, wrapper_kwargs):
        """Test forward pass with an empty batch (edge case).

        Parameters
        ----------
        mock_backbone : Mock
            A fixture providing a mock backbone model.
        wrapper_kwargs : dict
            A fixture providing required initialization parameters.
        """
        wrapper = PointcloudWrapper(
            backbone=mock_backbone,
            **wrapper_kwargs
        )

        batch = Mock()
        batch.x_0 = torch.randn(0, 3)  # Empty pointcloud
        batch.y = torch.tensor([])
        batch.batch_0 = torch.tensor([], dtype=torch.long)

        mock_backbone.return_value = torch.randn(0, 64)
        output = wrapper.forward(batch)

        assert output["x_0"].shape == (0, 64)
        assert len(output["labels"]) == 0
        assert len(output["batch_0"]) == 0

    def test_inheritance_from_abstract_wrapper(self):
        """Test that PointcloudWrapper inherits from AbstractWrapper."""
        from topobench.nn.wrappers.base import AbstractWrapper
        assert issubclass(PointcloudWrapper, AbstractWrapper)

    def test_forward_output_consistency(self, pointcloud_wrapper, sample_batch):
        """Test that multiple forward passes produce consistent structure.

        Parameters
        ----------
        pointcloud_wrapper : PointcloudWrapper
            A fixture to create a PointcloudWrapper instance.
        sample_batch : Mock
            A fixture to create a sample batch of pointcloud data.
        """
        output1 = pointcloud_wrapper.forward(sample_batch)
        output2 = pointcloud_wrapper.forward(sample_batch)

        assert set(output1.keys()) == set(output2.keys())
        assert output1["x_0"].shape == output2["x_0"].shape

    def test_forward_with_real_tensors(self, mock_backbone, wrapper_kwargs):
        """Test forward pass with realistic tensor values.

        Parameters
        ----------
        mock_backbone : Mock
            A fixture providing a mock backbone model.
        wrapper_kwargs : dict
            A fixture providing required initialization parameters.
        """
        wrapper = PointcloudWrapper(
            backbone=mock_backbone,
            **wrapper_kwargs
        )

        # Create a realistic batch with actual PyTorch tensors
        batch = Mock()
        batch.x_0 = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        batch.y = torch.tensor([0, 1, 2])
        batch.batch_0 = torch.tensor([0, 0, 0], dtype=torch.long)

        mock_backbone.return_value = torch.randn(3, 64)
        output = wrapper.forward(batch)

        assert output["x_0"].dtype == torch.float32
        assert output["labels"].dtype == torch.long
        assert output["batch_0"].dtype == torch.long

    def test_residual_connections_flag(self, mock_backbone):
        """Test initialization with residual_connections parameter.

        Parameters
        ----------
        mock_backbone : Mock
            A fixture providing a mock backbone model.
        """
        wrapper_with_residual = PointcloudWrapper(
            backbone=mock_backbone,
            out_channels=64,
            num_cell_dimensions=0,
            residual_connections=True
        )
        assert wrapper_with_residual.residual_connections is True

        wrapper_without_residual = PointcloudWrapper(
            backbone=mock_backbone,
            out_channels=64,
            num_cell_dimensions=0,
            residual_connections=False
        )
        assert wrapper_without_residual.residual_connections is False
