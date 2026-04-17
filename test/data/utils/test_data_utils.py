"""Unit tests for data utils."""

import omegaconf
import pytest
import torch_geometric
import torch
from topobench.data.utils import *
import toponetx as tnx
from toponetx.classes import CellComplex

class TestDataUtils:
    """Test data utils functions."""

    def setup_method(self):
        """Setup method."""
        self.complex = CellComplex()
        self.complex.add_cell([1, 2, 3],rank=2)
        self.complex.add_cell([3, 4, 5],rank=2)
        self.complex.add_cell([5, 6, 7],rank=2)
        self.neighborhoods1 = ['up_adjacency-0','2-up_adjacency-0','2-down_laplacian-2','2-down_adjacency-2','2-up_incidence-0','2-down_incidence-2']
        self.neighborhoods2 = ['down_incidence-1', 'up_laplacian-0', 'down_laplacian-1', 'up_adjacency-0', 'hodge_laplacian-1']


    def test_get_complex_connectivity(self):
        """Test get_complex_connectivity."""
        out = get_complex_connectivity(self.complex, 3, neighborhoods=self.neighborhoods2)
        assert 'up_laplacian-0' in out.keys()

    def test_get_combinatorial_complex_connectivity(self):
        """Test get_complex_connectivity."""
        out = get_combinatorial_complex_connectivity(self.complex, 3)
        assert 'adjacency_0' in out.keys()

    def test_select_neighborhoods_of_interest(self):
        """Test select_neighborhoods_of_interest."""
        connectivity = get_complex_connectivity(self.complex, 2)
        out = select_neighborhoods_of_interest(connectivity, self.neighborhoods1)
        assert '2-down_laplacian-2' in out.keys()
        assert 'incidence_1' in out.keys()

        with pytest.raises(ValueError) as e:
            select_neighborhoods_of_interest(connectivity, ['invalid_neighborhood'])

    def test_generate_zero_sparse_connectivity(self):
        """Test generate_zero_sparse_connectivity."""
        out = generate_zero_sparse_connectivity(10, 10)
        assert out.shape == (10, 10)
        assert torch.sum(out) == 0

    def test_load_cell_complex_dataset(self):
        """Test load_cell_complex_dataset."""
        with pytest.raises(NotImplementedError) as e:
            load_cell_complex_dataset({})

    def test_load_simplicial_dataset(self):
        """Test load_simplicial_dataset."""
        with pytest.raises(NotImplementedError) as e:
            load_simplicial_dataset({})

    def test_load_manual_graph(self):
        """Test load_manual_graph."""
        out = load_manual_graph()
        assert isinstance(out, torch_geometric.data.Data)

    def test_make_hash(self):
        """Test make_hash."""
        out = make_hash('test')
        assert isinstance(out, int)

    def test_ensure_serializable(self):
        """Test ensure_serializable."""
        objects = ['test', 1, 1.0, [1, 2, 3], {'a': 1, 'b': 2}, set([1, 2, 3]), omegaconf.dictconfig.DictConfig({'a': 1, 'b': 2}), torch_geometric.data.Data()]
        for obj in objects:
            out = ensure_serializable(obj)
