"""Tests for the PropagateSignalDown layer."""

import pytest
import torch
import torch_geometric.data as tg_data
import topomodelx
from topobench.nn.readouts.propagate_signal_down import PropagateSignalDown


class TestPropagateSignalDown:
    """Tests for the PropagateSignalDown layer."""
    @pytest.fixture
    def base_kwargs(self):
        """Fixture providing the required base parameters.

        Returns
        -------
        dict
            The base parameters for the PropagateSignalDown layer.
        """
        return {
            'hidden_dim': 64,
            'out_channels': 32,
            'task_level': 'graph',
            'num_cell_dimensions': 2,  # Need at least 2 dimensions for signal propagation
            'readout_name': 'test_readout'
        }

    @pytest.fixture
    def readout_layer(self, base_kwargs):
        """Fixture to create a PropagateSignalDown instance for testing.

        Parameters
        ----------
        base_kwargs : dict
            A fixture providing the required base parameters.

        Returns
        -------
        PropagateSignalDown
            A PropagateSignalDown instance for testing.
        """
        layer = PropagateSignalDown(**base_kwargs)
        layer.hidden_dim = base_kwargs['hidden_dim']
        return layer

    @pytest.fixture
    def create_sparse_incidence_matrix(self):
        """Helper fixture to create sparse incidence matrices.

        Returns
        -------
        callable
            A function to create sparse incidence matrices.
        """
        def _create_matrix(num_source, num_target, sparsity=0.3):
            """Create a sparse incidence matrix.

            Parameters
            ----------
            num_source : int
                The number of source nodes.
            num_target : int
                The number of target nodes.
            sparsity : float
                The sparsity of the matrix.

            Returns
            -------
            torch.sparse.FloatTensor
                A sparse incidence matrix.
            """
            num_entries = int(num_source * num_target * sparsity)
            indices = torch.zeros((2, num_entries), dtype=torch.long)
            values = torch.ones(num_entries)

            for i in range(num_entries):
                source = torch.randint(0, num_source, (1,))
                target = torch.randint(0, num_target, (1,))
                indices[0, i] = source
                indices[1, i] = target
                values[i] = torch.randint(0, 2, (1,)) * 2 - 1  # {-1, 1} values

            sparse_matrix = torch.sparse_coo_tensor(
                indices=torch.stack([indices[1], indices[0]]),
                values=values,
                size=(num_target, num_source)
            ).coalesce()

            return sparse_matrix
        return _create_matrix

    @pytest.fixture
    def sample_batch(self, create_sparse_incidence_matrix):
        """Fixture to create a sample batch with required incidence matrices.

        Parameters
        ----------
        create_sparse_incidence_matrix : callable
            A fixture to create sparse incidence matrices.

        Returns
        -------
        torch_geometric.data.Data
            A sample batch with required incidence matrices.
        """
        num_nodes = 10
        num_edges = 15

        return tg_data.Data(
            x=torch.randn(num_nodes, 64),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            batch_0=torch.zeros(num_nodes, dtype=torch.long),
            incidence_1=create_sparse_incidence_matrix(num_edges, num_nodes)
        )

    @pytest.fixture
    def sample_model_output(self, sample_batch):
        """Fixture to create a sample model output with cell embeddings.

        Parameters
        ----------
        sample_batch : torch_geometric.data.Data
            A fixture to create a sample batch with required incidence matrices.

        Returns
        -------
        dict
            A sample model output with cell embeddings.
        """
        hidden_dim = 64

        num_nodes = sample_batch.x.size(0)
        num_edges = sample_batch.edge_index.size(1)

        return {
            'logits': torch.randn(num_nodes, hidden_dim),
            'x_0': torch.randn(num_nodes, hidden_dim),
            'x_1': torch.randn(num_edges, hidden_dim),
        }

    def test_forward_propagation(self, readout_layer, sample_model_output, sample_batch):
        """Test the forward pass with detailed assertions.

        Parameters
        ----------
        readout_layer : PropagateSignalDown
            A fixture to create a PropagateSignalDown instance.
        sample_model_output : dict
            A fixture to create a sample model output with cell embeddings.
        sample_batch : torch_geometric.data.Data
            A fixture to create a sample batch with required incidence matrices.
        """
        initial_output = {k: v.clone() for k, v in sample_model_output.items()}
        sample_model_output['x_0'] = sample_model_output['logits']

        output = readout_layer(sample_model_output, sample_batch)

        assert 'x_0' in output
        assert output['x_0'].shape == initial_output['logits'].shape
        assert output['x_0'].dtype == torch.float32

        assert 'x_1' in output
        assert output['x_1'].shape == initial_output['x_1'].shape
        assert output['x_1'].dtype == torch.float32

    @pytest.mark.parametrize('missing_key', ['incidence_1'])
    def test_missing_incidence_matrix(self, readout_layer, sample_model_output, sample_batch, missing_key):
        """Test handling of missing incidence matrices.

        Parameters
        ----------
        readout_layer : PropagateSignalDown
            A fixture to create a PropagateSignalDown instance.
        sample_model_output : dict
            A fixture to create a sample model output with cell embeddings.
        sample_batch : torch_geometric.data.Data
            A fixture to create a sample batch with required incidence matrices.
        missing_key : str
            The key to remove from the sample batch.
        """
        invalid_batch = tg_data.Data(**{k: v for k, v in sample_batch.items() if k != missing_key})
        sample_model_output['x_0'] = sample_model_output['logits']

        with pytest.raises(KeyError):
            readout_layer(sample_model_output, invalid_batch)

    @pytest.mark.parametrize('missing_key', ['x_1'])  # Changed to only test x_1
    def test_missing_cell_features(self, readout_layer, sample_model_output, sample_batch, missing_key):
        """Test handling of missing cell features.

        Parameters
        ----------
        readout_layer : PropagateSignalDown
            A fixture to create a PropagateSignalDown instance.
        sample_model_output : dict
            A fixture to create a sample model output with cell embeddings.
        sample_batch : torch_geometric.data.Data
            A fixture to create a sample batch with required incidence matrices.
        missing_key : str
            The key to remove from the sample model output.
        """
        invalid_output = {k: v for k, v in sample_model_output.items() if k != missing_key}
        invalid_output['x_0'] = invalid_output['logits']  # Always map logits to x_0

        with pytest.raises(KeyError):
            readout_layer(invalid_output, sample_batch)

    def test_gradient_flow(self, readout_layer, sample_model_output, sample_batch):
        """Test gradient flow through the network.

        Parameters
        ----------
        readout_layer : PropagateSignalDown
            A fixture to create a PropagateSignalDown instance.
        sample_model_output : dict
            A fixture to create a sample model output with cell embeddings.
        sample_batch : torch_geometric.data.Data
            A fixture to create a sample batch with required incidence matrices.
        """
        # Create a copy of logits tensor to track gradients properly
        logits = sample_model_output['logits'].clone().detach().requires_grad_(True)
        x_1 = sample_model_output['x_1'].clone().detach().requires_grad_(True)

        model_output = {
            'logits': logits,
            'x_0': logits,  # Share the same tensor
            'x_1': x_1
        }

        output = readout_layer(model_output, sample_batch)
        loss = output['x_0'].sum()
        loss.backward()

        # Check gradient flow
        assert logits.grad is not None
        assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad))
        assert x_1.grad is not None
        assert not torch.allclose(x_1.grad, torch.zeros_like(x_1.grad))
