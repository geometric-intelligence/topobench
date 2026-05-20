"""Loaders for HypBench dataset as graph."""

import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.data import Data, InMemoryDataset

from topobench.data.loaders.base import AbstractLoader


class HypBenchDatasetLoader(AbstractLoader):
    """Load HypBench dataset with configurable parameters.

    HypBench: Hyperbolic Benchmarking framework for Graph Neural Networks (GNNs).

    This module provides a PyTorch interface to generate and evaluate synthetic
    graph-structured datasets based on the S1/H2 geometric soft configuration model
    and its bipartite extension. The framework enables controlled manipulation of
    key network properties, including:

        - Degree distributions
        - Clustering coefficients
        - Homophily
        - Topology–feature correlations

    HypBench allows systematic benchmarking of GNN architectures on tasks such as
    node classification and link prediction, offering a reproducible and tunable
    environment for fair model comparison. By coupling topology and node features
    within a hyperbolic similarity space, it provides insights into how structural
    properties of networks affect model performance.

    Parameters
    ----------
    parameters : DictConfig
        Configuration object with the following required fields:
            N_n (int): Number of nodes. Must be > 0.
            beta (float): Inverse temperature, controlling the clustering coefficient. Must be > 1.
            gamma (float): Exponent of the power-law distribution for hidden degrees for unipartite network. Must be > 2.
            kmean (float): The mean degree of the unipartite network. Must be > 0.
            N_f (int): Number of features. Must be > 0.
            beta_b (float): Controls bipartite clustering. Must be > 1.
            gamma_n (float): Exponent of the power-law distribution for hidden degrees of nodes in the bipartite network. Must be > 2.
            gamma_f (float): Exponent of the power-law distribution for hidden degrees of features in the bipartite network. Must be > 2.
            kmean_n (float): The mean degree nodes in the bipartite network. Must be > 0.
            N_c (int): Number of classes. Must be > 1.
            alpha (float): Tunes the level of homophily in the network.
    **kwargs : dict, optional
        Additional keyword arguments for dataset initialization (currently unused).

    Raises
    ------
    ValueError
        If any required parameter is missing or does not satisfy its constraint.

    References
    ----------
    Roya Aliakbarisani, Robert Jankowski, M. Ángeles Serrano, Marián Boguñá.
    "Hyperbolic Benchmarking Unveils Network Topology-Feature Relationship in GNN Performance."
    arXiv:2406.02772, 2024. https://arxiv.org/abs/2406.02772
    """

    def __init__(self, parameters: DictConfig, **kwargs) -> None:
        super().__init__(parameters, **kwargs)

        # List of required parameters
        required_params = [
            "N_n",
            "beta",
            "gamma",
            "kmean",
            "N_f",
            "beta_b",
            "gamma_n",
            "gamma_f",
            "kmean_n",
            "N_c",
            "alpha",
        ]
        missing = [p for p in required_params if not hasattr(parameters, p)]
        if missing:
            raise ValueError(
                f"Missing required parameters in DictConfig: {missing}"
            )

        # Load parameters from DictConfig
        self.N_n = parameters.N_n
        self.beta = parameters.beta
        self.gamma = parameters.gamma
        self.kmean = parameters.kmean
        self.N_f = parameters.N_f
        self.beta_b = parameters.beta_b
        self.gamma_n = parameters.gamma_n
        self.gamma_f = parameters.gamma_f
        self.kmean_n = parameters.kmean_n
        self.N_c = parameters.N_c
        self.alpha = parameters.alpha

        # Value checks for parameters
        if not (self.N_n > 0):
            raise ValueError("N_n must be > 0")
        if not (self.beta > 1):
            raise ValueError("beta must be > 1")
        if not (self.gamma > 2):
            raise ValueError("gamma must be > 2")
        if not (self.kmean > 0):
            raise ValueError("kmean must be > 0")
        if not (self.N_f > 0):
            raise ValueError("N_f must be > 0")
        if not (self.beta_b > 1):
            raise ValueError("beta_b must be > 1")
        if not (self.gamma_n > 2):
            raise ValueError("gamma_n must be > 2")
        if not (self.gamma_f > 2):
            raise ValueError("gamma_f must be > 2")
        if not (self.kmean_n > 0):
            raise ValueError("kmean_n must be > 0")
        if not (self.N_c > 1):
            raise ValueError("N_c must be > 1")

        self.R = self.N_n / (2 * np.pi)

        # Generate hidden degrees
        self.kappas = self._generate_power_law_distribution(
            self.N_n, self.gamma, self.kmean
        )
        self.kappas_n = self._generate_power_law_distribution(
            self.N_n, self.gamma_n, self.kmean_n
        )
        kmean_f = self.N_n / self.N_f * self.kmean_n
        self.kappas_f = self._generate_power_law_distribution(
            self.N_f, self.gamma_f, kmean_f
        )

        # Generate angular positions
        self.thetas = np.random.uniform(0, 2 * np.pi, self.N_n)
        self.thetas_f = np.random.uniform(0, 2 * np.pi, self.N_f)

        # Compute parameters mu and mu_b
        self.mu = (
            self.beta / (2 * np.pi * self.kmean) * np.sin(np.pi / self.beta)
        )
        self.mu_b = (
            self.beta_b
            / (2 * np.pi * self.kmean_n)
            * np.sin(np.pi / self.beta_b)
        )

        # Compute radial coordinates
        kappa_min = np.min(self.kappas)
        Rhat = 2 * np.log(2 * self.R / (self.mu * kappa_min**2))
        self.radii = [
            Rhat - 2 * np.log(kappa / kappa_min) for kappa in self.kappas
        ]
        kappa_n_min = np.min(self.kappas_n)
        kappa_f_min = np.min(self.kappas_f)
        Rhat_b = 2 * np.log(
            2 * self.R / (self.mu_b * kappa_n_min * kappa_f_min)
        )
        self.radii_n = [
            Rhat_b - 2 * np.log(kappa_n / kappa_n_min)
            for kappa_n in self.kappas_n
        ]
        self.radii_f = [
            Rhat_b - 2 * np.log(kappa_f / kappa_f_min)
            for kappa_f in self.kappas_f
        ]

    def load_dataset(self, **kwargs):
        """Generate a synthetic graph dataset using the HypBench model.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for dataset initialization (currently unused).

        Returns
        -------
        Data
            A torch_geometric.data.Data object containing:
                - x: Node feature matrix (torch.Tensor)
                - edge_index: Edge list (torch.Tensor)
                - y: Node labels (torch.Tensor)
                - thetas, kappas, radii, thetas_f, kappas_f, kappas_n, radii_n, radii_f: Model parameters (numpy arrays)
                - num_nodes, num_node_features, num_classes: Dataset statistics

        Raises
        ------
        RuntimeError
            If dataset generation fails.
        """

        # Generate unipartite network
        source_nodes = []
        target_nodes = []
        for u in range(self.N_n):
            for v in range(u):
                angle = self._get_angle(self.thetas[u], self.thetas[v])
                chi = (
                    self.R
                    * angle
                    / (self.mu * self.kappas[u] * self.kappas[v])
                )
                p_ij = 1 / (1 + np.power(chi, self.beta))
                if random.random() < p_ij:
                    source_nodes.append(u)
                    target_nodes.append(v)
                    source_nodes.append(v)
                    target_nodes.append(u)
        edge_index = torch.tensor(
            [source_nodes, target_nodes], dtype=torch.long
        )

        # Generate bipartite network representing nodes' features
        x = torch.zeros((self.N_n, self.N_f), dtype=torch.float)
        for n in range(self.N_n):
            for f in range(self.N_f):
                angle = self._get_angle(self.thetas[n], self.thetas_f[f])
                chi = (
                    self.R
                    * angle
                    / (self.mu_b * self.kappas_n[n] * self.kappas_f[f])
                )
                p_ij = 1 / (1 + np.power(chi, self.beta_b))
                if random.random() < p_ij:
                    x[n, f] = 1

        y = self._generate_labels(self.N_c, self.alpha, self.thetas)

        dataset = Data(
            x=x,
            edge_index=edge_index,
            thetas=self.thetas,
            kappas=self.kappas,
            radii=self.radii,
            thetas_f=self.thetas_f,
            kappas_f=self.kappas_f,
            kappas_n=self.kappas_n,
            radii_n=self.radii_n,
            radii_f=self.radii_f,
            y=torch.tensor(y),
            num_nodes=self.N_n,
            num_node_features=self.N_f,
            num_classes=self.N_c,
        )

        return SingleDataInMemoryDataset(dataset)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"N_n={self.N_n}, "
            f"beta={self.beta}, "
            f"gamma={self.gamma}, "
            f"kmean={self.kmean}, "
            f"N_f={self.N_f}, "
            f"beta_b={self.beta_b}, "
            f"gamma_n={self.gamma_n}, "
            f"gamma_f={self.gamma_f}, "
            f"kmean_n={self.kmean_n}, "
            f"N_c={self.N_c}, "
            f"alpha={self.alpha}"
            f")"
        )

    def _generate_power_law_distribution(
        self, n: int, gamma: float, kmean: float
    ) -> list[float]:
        """Generate a power-law distributed sequence of hidden degrees.

        Parameters
        ----------
        n : int
            Number of nodes.
        gamma : float
            Exponent of the power-law distribution. Must be > 2.
        kmean : float
            Desired mean of the distribution. Must be > 0.

        Returns
        -------
        list[float]
            List of hidden degrees following a power-law distribution.

        Raises
        ------
        ValueError
            If gamma <= 2 or kmean <= 0.
        """
        if gamma <= 2:
            raise ValueError("gamma must be > 2")
        if kmean <= 0:
            raise ValueError("kmean must be > 0")

        gam_ratio = (gamma - 2) / (gamma - 1)
        kappa_0 = kmean * gam_ratio * (1 - 1 / n) / (1 - 1 / n**gam_ratio)
        base = 1 - 1 / n
        power = 1 / (1 - gamma)
        kappas = [
            kappa_0 * (1 - random.random() * base) ** power for _ in range(n)
        ]
        return kappas

    def _generate_labels(self, N_c: int, alpha: float, thetas) -> list[int]:
        """Generate node labels based on angular positions and homophily parameter.

        Parameters
        ----------
        N_c : int
            Number of classes.
        alpha : float
            Homophily tuning parameter.
        thetas : array-like
            Angular positions of nodes.

        Returns
        -------
        list[int]
            List of node labels.
        """
        centers = np.random.uniform(0, 2 * np.pi, size=N_c)
        labels = []
        for t in thetas:
            total_distance = sum(
                [np.power(self._get_angle(t, c), -alpha) for c in centers]
            )
            prob = [
                np.power(self._get_angle(t, c), -alpha) / total_distance
                for c in centers
            ]
            label = np.random.choice(len(prob), size=1, p=prob)[0]
            labels.append(label)
        return labels

    def _get_angle(self, t1: float, t2: float) -> float:
        """Compute the angular distance between two angles on a circle.

        Parameters
        ----------
        t1 : float
            First angle in radians.
        t2 : float
            Second angle in radians.

        Returns
        -------
        float
            The angular distance between t1 and t2.
        """
        return np.pi - np.fabs(np.pi - np.fabs(t1 - t2))


class SingleDataInMemoryDataset(InMemoryDataset):
    """A wrapper to store a single Data object in an InMemoryDataset.

    Parameters
    ----------
    data : Data
        A torch_geometric.data.Data object to be stored in the dataset.

    Returns
    -------
    InMemoryDataset
        An InMemoryDataset containing the single Data object.
    """

    def __init__(self, data):
        super().__init__(None)
        self.data = data
        self.slices = None

    def __len__(self):
        """
        Return the length of the dataset.

        Returns
        -------
        int
            The length of the dataset (always 1).
        """
        return 1

    def get(self, idx):
        """
        Get the data object at the specified index.

        Parameters
        ----------
        idx : int
            Index of the data object to retrieve.

        Returns
        -------
        Data
            The Data object stored in the dataset.
        """
        return self.data
