"""Tests for the MLPReadout layer."""

import pytest
import torch
import torch_geometric.data as tg_data
from unittest.mock import Mock, patch

from topobench.nn.readouts.mlp_readout import MLPReadout


class TestMLPReadout:
    """Tests for the MLPReadout layer."""

    @pytest.fixture
    def base_kwargs(self):
        """Fixture providing the required base parameters.

        Returns
        -------
        dict
            The base parameters for the MLPReadout layer.
        """
        return {
            'in_channels': 64,
            'hidden_layers': [32, 16],
            'out_channels': 8,
            'task_level': 'node'
        }

    @pytest.fixture
    def graph_level_kwargs(self):
        """Fixture providing parameters for graph-level tasks.

        Returns
        -------
        dict
            The parameters for graph-level MLPReadout.
        """
        return {
            'in_channels': 64,
            'hidden_layers': [32, 16],
            'out_channels': 8,
            'task_level': 'graph',
            'pooling_type': 'sum'
        }

    @pytest.fixture
    def readout_layer(self, base_kwargs):
        """Fixture to create a MLPReadout instance for testing.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.

        Returns
        -------
        MLPReadout
            A MLPReadout instance for testing.
        """
        return MLPReadout(**base_kwargs)

    @pytest.fixture
    def graph_readout_layer(self, graph_level_kwargs):
        """Fixture to create a graph-level MLPReadout instance.

        Parameters
        ----------
        graph_level_kwargs : dict
            A fixture providing parameters for graph-level tasks.

        Returns
        -------
        MLPReadout
            A graph-level MLPReadout instance for testing.
        """
        return MLPReadout(**graph_level_kwargs)

    @pytest.fixture
    def sample_model_output(self):
        """Fixture to create a sample model output dictionary.

        Returns
        -------
        dict
            A sample model output dictionary.
        """
        return {
            'x_0': torch.randn(10, 64),
            'edge_indices': torch.randint(0, 10, (2, 15))
        }

    @pytest.fixture
    def sample_batch(self):
        """Fixture to create a sample batch of graph data.

        Returns
        -------
        dict
            A sample batch dictionary.
        """
        return {
            'batch_0': torch.zeros(10, dtype=torch.long)
        }

    @pytest.fixture
    def multi_graph_batch(self):
        """Fixture to create a batch with multiple graphs.

        Returns
        -------
        dict
            A batch dictionary with multiple graphs.
        """
        return {
            'batch_0': torch.cat([torch.zeros(5), torch.ones(5)]).long()
        }

    def test_initialization_node_level(self, base_kwargs):
        """Test that MLPReadout initializes correctly for node-level tasks.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        """
        readout = MLPReadout(**base_kwargs)
        assert isinstance(readout, MLPReadout)
        assert readout.pooling_type == "sum"
        assert readout.task_level == "node"

    def test_initialization_graph_level(self, graph_level_kwargs):
        """Test that MLPReadout initializes correctly for graph-level tasks.

        Parameters
        ----------
        graph_level_kwargs : dict
            A fixture providing parameters for graph-level tasks.
        """
        readout = MLPReadout(**graph_level_kwargs)
        assert isinstance(readout, MLPReadout)
        assert readout.pooling_type == "sum"
        assert readout.task_level == "graph"
        assert hasattr(readout, 'graph_readout_layer')
        assert hasattr(readout, 'graph_readout_activation')

    def test_pooling_type_parameter(self, base_kwargs):
        """Test that different pooling types are correctly set.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        """
        for pooling in ['sum', 'mean', 'max']:
            kwargs = {**base_kwargs, 'pooling_type': pooling}
            readout = MLPReadout(**kwargs)
            assert readout.pooling_type == pooling

    def test_forward_node_level(self, readout_layer, sample_model_output, sample_batch):
        """Test forward pass for node-level tasks.

        Parameters
        ----------
        readout_layer : MLPReadout
            A fixture to create a MLPReadout instance for testing.
        sample_model_output : dict
            A fixture to create a sample model output dictionary.
        sample_batch : dict
            A fixture to create a sample batch of graph data.
        """
        output = readout_layer.forward(sample_model_output.copy(), sample_batch)

        assert 'x_0' in output
        assert output['x_0'].shape[0] == 10  # Same number of nodes
        assert output['x_0'].shape[1] == 8   # out_channels

    def test_forward_graph_level(self, graph_readout_layer, sample_model_output, sample_batch):
        """Test forward pass for graph-level tasks.

        Parameters
        ----------
        graph_readout_layer : MLPReadout
            A fixture to create a graph-level MLPReadout instance.
        sample_model_output : dict
            A fixture to create a sample model output dictionary.
        sample_batch : dict
            A fixture to create a sample batch of graph data.
        """
        output = graph_readout_layer.forward(sample_model_output.copy(), sample_batch)

        assert 'x_0' in output
        assert output['x_0'].shape[0] == 1  # Single graph
        assert output['x_0'].shape[1] == 8  # out_channels

    def test_forward_graph_level_multiple_graphs(self, graph_readout_layer, multi_graph_batch):
        """Test forward pass for multiple graphs in a batch.

        Parameters
        ----------
        graph_readout_layer : MLPReadout
            A fixture to create a graph-level MLPReadout instance.
        multi_graph_batch : dict
            A fixture to create a batch with multiple graphs.
        """
        model_output = {'x_0': torch.randn(10, 64)}
        output = graph_readout_layer.forward(model_output, multi_graph_batch)

        assert 'x_0' in output
        assert output['x_0'].shape[0] == 2  # Two graphs
        assert output['x_0'].shape[1] == 8  # out_channels

    def test_call_method_adds_logits(self, readout_layer, sample_model_output, sample_batch):
        """Test that __call__ method adds logits to output.

        Parameters
        ----------
        readout_layer : MLPReadout
            A fixture to create a MLPReadout instance for testing.
        sample_model_output : dict
            A fixture to create a sample model output dictionary.
        sample_batch : dict
            A fixture to create a sample batch of graph data.
        """
        output = readout_layer(sample_model_output.copy(), sample_batch)

        assert 'logits' in output
        assert 'x_0' in output
        assert torch.equal(output['logits'], output['x_0'])

    def test_different_pooling_types_graph_level(self, graph_level_kwargs):
        """Test that different pooling types work correctly for graph-level tasks.

        Parameters
        ----------
        graph_level_kwargs : dict
            A fixture providing parameters for graph-level tasks.
        """
        model_output = {'x_0': torch.randn(10, 64)}
        batch = {'batch_0': torch.zeros(10, dtype=torch.long)}

        for pooling in ['sum', 'mean', 'max']:
            kwargs = {**graph_level_kwargs, 'pooling_type': pooling}
            readout = MLPReadout(**kwargs)
            output = readout(model_output.copy(), batch)

            assert 'x_0' in output
            assert output['x_0'].shape == (1, 8)

    def test_dropout_parameter(self, base_kwargs):
        """Test that dropout parameter is correctly set.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        """
        kwargs = {**base_kwargs, 'dropout': 0.5}
        readout = MLPReadout(**kwargs)
        assert isinstance(readout, MLPReadout)

    def test_normalization_parameter(self, base_kwargs):
        """Test that normalization parameter is correctly set.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        """
        kwargs = {**base_kwargs, 'norm': 'batch_norm'}
        readout = MLPReadout(**kwargs)
        assert isinstance(readout, MLPReadout)

    def test_activation_parameters(self, base_kwargs):
        """Test that activation function parameters are correctly set.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.
        """
        kwargs = {**base_kwargs, 'act': 'gelu', 'final_act': 'sigmoid'}
        readout = MLPReadout(**kwargs)
        assert isinstance(readout, MLPReadout)

    def test_empty_hidden_layers(self):
        """Test initialization with empty hidden layers."""
        kwargs = {
            'in_channels': 64,
            'hidden_layers': [],
            'out_channels': 8,
            'task_level': 'graph',
            'pooling_type': 'sum'
        }
        readout = MLPReadout(**kwargs)
        assert isinstance(readout, MLPReadout)

        model_output = {'x_0': torch.randn(10, 64)}
        batch = {'batch_0': torch.zeros(10, dtype=torch.long)}
        output = readout(model_output, batch)

        assert output['x_0'].shape == (1, 8)

    def test_output_shape_consistency(self, readout_layer, sample_batch):
        """Test that output shapes are consistent across different input sizes.

        Parameters
        ----------
        readout_layer : MLPReadout
            A fixture to create a MLPReadout instance for testing.
        sample_batch : dict
            A fixture to create a sample batch of graph data.
        """
        for num_nodes in [5, 10, 20]:
            model_output = {'x_0': torch.randn(num_nodes, 64)}
            batch = {'batch_0': torch.zeros(num_nodes, dtype=torch.long)}
            output = readout_layer(model_output, batch)

            assert output['x_0'].shape[0] == num_nodes
            assert output['x_0'].shape[1] == 8

    def test_preserves_other_keys_in_model_output(self, readout_layer, sample_batch):
        """Test that forward pass preserves other keys in model_out.

        Parameters
        ----------
        readout_layer : MLPReadout
            A fixture to create a MLPReadout instance for testing.
        sample_batch : dict
            A fixture to create a sample batch of graph data.
        """
        model_output = {
            'x_0': torch.randn(10, 64),
            'edge_indices': torch.randint(0, 10, (2, 15)),
            'extra_data': torch.randn(10, 32)
        }
        output = readout_layer(model_output, sample_batch)

        assert 'edge_indices' in output
        assert 'extra_data' in output
        assert torch.equal(output['edge_indices'], model_output['edge_indices'])
        assert torch.equal(output['extra_data'], model_output['extra_data'])
