"""Test Kernel Lifting."""

import pytest
import torch
import torch.nn.functional as F
import torch_geometric

from topobench.data.utils.utils import load_manual_graph
from topobench.transforms.liftings.graph2hypergraph.kernel_lifting import (
     HypergraphKernelLifting,
     get_combination,
     get_feat_kernel,
     get_graph_kernel,
     graph_heat_kernel,
     graph_matern_kernel,
)


def cos_sim(A, B, dim=1):
    """Compute the cosine similarity between two matrices.

    Parameters
    ----------
    A : torch.Tensor
        The first matrix.
    B : torch.Tensor
        The second matrix.
    dim : int, optional
        The dimension along which to compute the cosine similarity, by default 1.

    Returns
    -------
    torch.Tensor
        The cosine similarity matrix.
    """
    return torch.mm(
        F.normalize(A, p=2, dim=dim),
        F.normalize(B, p=2, dim=dim).transpose(0, 1)
    )


def rbf_kernel(x1, x2, gamma=1.0):
    """ Compute the RBF kernel between two matrices.

    Parameters
    ----------
    x1 : torch.Tensor
        The first matrix.
    x2 : torch.Tensor
        The second matrix.
    gamma : float, optional
        The gamma parameter of the RBF kernel, by default 1.0.

    Returns
    -------
    torch.Tensor
        The RBF kernel matrix.
    """
    dist_sq = torch.cdist(x1, x2, p=2) ** 2
    return torch.exp(-gamma * dist_sq)


class TestHypergraphKernelLifting:
    """Test the HypergraphKernelLifting class."""

    def setup_method(self):
        """Initialise the data and the HypergraphKernelLifting class."""
        # Load the graph
        self.data = load_manual_graph()
        self.data.edge_index = torch_geometric.utils.to_undirected(self.data.edge_index)

        # Initialise the HypergraphKernelLifting class
        self.lifting = HypergraphKernelLifting()

        # Initialise other classes for utilities testing
        self.small_laplacian = torch.tensor([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32)
        self.small_features = F.normalize(torch.tensor([[1, 1, 1, 1], [100, 1000, 100, 1000], [1.1, 1.1, 1.1, 1.1]], dtype=torch.float32), p=2, dim=1)

    def test_graph_heat_kernel(self):
        """Test the graph heat kernel."""
        assert torch.allclose(graph_heat_kernel(self.small_laplacian, 0), torch.eye(self.small_laplacian.shape[0]))
        assert torch.allclose(graph_heat_kernel(self.small_laplacian, 100), torch.tensor([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]]))
        assert torch.allclose(graph_heat_kernel(self.small_laplacian, 1), torch.tensor([[0.5677, 0.4323, 0.0], [0.4323, 0.5677, 0.0], [0.0, 0.0, 1.0]]), atol=1e-4)

    def test_graph_matern_kernel(self):
        """Test the graph matern kernel."""
        assert torch.allclose(graph_matern_kernel(self.small_laplacian, nu=0, kappa=0.1), torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))
        assert torch.allclose(graph_matern_kernel(self.small_laplacian, nu=100, kappa=2), torch.zeros(self.small_laplacian.shape[0], self.small_laplacian.shape[1]), atol=1e-7)
        assert torch.allclose(graph_matern_kernel(self.small_laplacian, nu=2, kappa=2), torch.tensor([[0.5556, 0.4444, 0.0], [0.4444, 0.5556, 0.0], [0.0, 0.0, 1.0]]), atol=1e-4)

    def test_validations(self):
        """Test the validations of the kernel lifting."""
        with pytest.raises(ValueError):
            get_graph_kernel(self.small_laplacian, "does not exist")

        with pytest.raises(ValueError):
            get_feat_kernel(self.small_features, "does not exist")

        edges = (torch.tensor([0]), torch.tensor([1]))
        graph = torch_geometric.data.Data(
            edge_index=edges,
            num_nodes=10,
        )
        with pytest.raises(ValueError):
            self.lifting.forward(graph)

    def test_get_graph_kernel(self):
        """Test the get_graph_kernel function."""
        assert torch.all(get_graph_kernel(self.small_laplacian, "heat", t=1) == graph_heat_kernel(self.small_laplacian, 1))
        assert torch.all(get_graph_kernel(self.small_laplacian, "matern", nu=2, kappa=2) == graph_matern_kernel(self.small_laplacian, nu=2, kappa=2))
        # test custom kernel
        assert torch.all(get_graph_kernel(self.small_laplacian, lambda L: 2 * L) == 2 * self.small_laplacian)

    def test_remove_self_loops(self):
        """Test the remove_self_loops function."""
        # edges indexed 2 and 5 are self loops
        incidence_with_self_loop_edges = torch.tensor([[1, 0, 0, 1, 0], [1, 1, 0, 1, 0], [1, 1, 1, 0, 0], [1, 1, 0, 1, 0], [1, 1, 0, 0, 1]])
        incidence_without_self_loops = torch.tensor([[1, 0, 1], [1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 1, 0]])
        assert torch.all(self.lifting._remove_empty_edges(incidence_with_self_loop_edges) == incidence_without_self_loops)
        assert torch.all(self.lifting._remove_empty_edges(incidence_without_self_loops) == incidence_without_self_loops)

    def test_get_feat_kernel(self):
        """Test the get_feat_kernel function."""
        assert torch.all(get_feat_kernel(self.small_features, "identity") == torch.ones((self.small_features.shape[0], self.small_features.shape[0])))

        # test custom kernel (1 - cosine distance, the features are normalized in the test)
        assert torch.allclose(
            get_feat_kernel(self.small_features, lambda X: torch.mm(X, X.t())),
            torch.tensor([[1.0, 0.7740, 1.0], [0.774, 1.0, 0.774], [1.0, 0.774, 1.0]]), atol=1e-4)

    def test_get_combination(self):
        """Test the get_combination function."""
        assert (get_combination("prod")(self.small_laplacian, self.small_laplacian) == self.small_laplacian * self.small_laplacian).all(), "Graph prod combination failed"
        assert (get_combination("prod")(self.small_features, self.small_features) == self.small_features * self.small_features).all(), "Feature prod combination failed"

        with pytest.raises(ValueError):
            get_combination("does not exist")

    def reset_lifting(self, **kwargs):
        """Reset the HypergraphKernelLifting class with new parameters.

        Parameters
        ----------
        **kwargs : dict
            The parameters to reset the HypergraphKernelLifting class.
        """
        self.lifting = HypergraphKernelLifting(**kwargs)

    def test_lifting_heat(self):
        """Test the lifting with the heat kernel."""
        # Test the lift_topology method with HEAT kernel and t=0
        self.reset_lifting(graph_kernel="heat", t=0, fraction=0.1)
        lifted_data = self.lifting.forward(self.data.clone())
        expected_incidence_1 = torch.tensor([])
        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Incorrect hyperedges incidence, heat kernel, t=0."

        # Test the lift_topology method with HEAT kernel and t=1
        self.reset_lifting(graph_kernel="heat", t=1)
        lifted_data = self.lifting.forward(self.data.clone())
        expected_n_hyperedges = 7
        expected_incidence_1 = torch.tensor([
            [0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 1., 1., 1.],
            [0., 0., 1., 1., 1., 1., 1.],
            [0., 1., 1., 0., 0., 0., 1.],
            [0., 0., 0., 0., 1., 1., 1.],
            [1., 1., 0., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0., 0., 0.],
            [1., 0., 0., 1., 0., 1., 1.]
        ])

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Incorrect hyperedges incidence, heat kernel, t=1."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Incorrect number of hyperedges, heat kernel, t=1."

        # Test the lift_topology method with HEAT kernel and t=2
        self.reset_lifting(graph_kernel="heat", t=2)
        lifted_data = self.lifting.forward(self.data.clone())
        expected_n_hyperedges = 7
        expected_incidence_1 = torch.tensor([
            [0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 1., 1., 1.],
            [0., 0., 1., 1., 1., 1., 1.],
            [1., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 1.],
            [0., 1., 1., 1., 0., 0., 1.],
            [1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 1., 1., 0., 1., 1.]
        ])

        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Incorrect hyperedges, heat kernel, t=2."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Incorrect number of hyperedges, heat kernel, t=2."

    def test_lifting_matern(self):
        """Test the lifting with the matern kernel."""
        # Test the lift_topology method with MATERN kernel and nu=1 and kappa=2
        self.reset_lifting(graph_kernel="matern", nu=1, kappa=2)
        lifted_data = self.lifting.forward(self.data.clone())
        expected_incidence_1 = torch.tensor([
            [0., 0., 0., 1., 1., 1., 1.],
            [0., 0., 0., 0., 1., 1., 1.],
            [0., 0., 1., 1., 1., 1., 1.],
            [1., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 1., 1.],
            [0., 1., 1., 1., 0., 0., 1.],
            [1., 1., 1., 0., 0., 0., 0.],
            [0., 0., 1., 1., 0., 1., 1.]
        ])
        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Incorrect hyperedges incidence, Matern kernel, nu=1, kappa=2."
        assert (
            lifted_data.num_hyperedges == 7
        ), "Incorrect number of hyperedges, Matern kernel, nu=1, kappa=2"

    def test_features_lifting(self):
        """Test the feature lifting."""
        self.reset_lifting(
            graph_kernel="identity",
            feat_kernel=lambda X: rbf_kernel(X, X),
            C="prod",
            fraction=0.2
        )
        lifted_data = self.lifting.forward(self.data.clone())

        expected_n_hyperedges = 1
        expected_incidence_1 = torch.tensor([
            [1.],
            [1.],
            [1.],
            [0.],
            [0.],
            [0.],
            [0.],
            [0.]
        ])
        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Wrong incidence_hyperedges: feature lifting with rbf_kernel."
        assert expected_n_hyperedges == lifted_data.num_hyperedges, "Wrong number of hyperedges: feature lifting with rbf_kernel."

    def test_combinations(self):
        """Test the combinations of the graph and feature liftings."""
        self.reset_lifting(
            graph_kernel="heat", t=2,
            feat_kernel=lambda X: rbf_kernel(X, X),
            C="sum", fraction=0.2)
        lifted_data = self.lifting.forward(self.data.clone())

        expected_n_hyperedges = 3
        expected_incidence_1 = torch.tensor([
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ])
        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Wrong incidence_hyperedges: feature and graph lifting combined."
        assert (
            expected_n_hyperedges == lifted_data.num_hyperedges
        ), "Wrong number of hyperedges: feature and graph lifting combined."

        self.reset_lifting(
            graph_kernel="heat", t=2,
            feat_kernel=lambda X: rbf_kernel(X, X),
            C="prod", fraction=0.3)
        lifted_data = self.lifting.forward(self.data.clone())
        expected_incidence_1 = torch.tensor([
            [1.],
            [1.],
            [1.],
            [1.],
            [1.],
            [1.],
            [1.],
            [1.]
        ])
        assert (
            expected_incidence_1 == lifted_data.incidence_hyperedges.to_dense()
        ).all(), "Wrong incidence_hyperedges: feature and graph lifting combined."
        assert (
            lifted_data.num_hyperedges == 1
        ), "Wrong number of hyperedges: feature and graph lifting combined."
