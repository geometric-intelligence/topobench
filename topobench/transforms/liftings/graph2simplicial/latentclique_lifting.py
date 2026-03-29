"""This module implements the LatentCliqueLifting class, which lifts graphs to simplicial complexes."""

import numpy as np
import torch
import torch_geometric
from scipy import stats
from scipy.special import gammaln, logsumexp
from tqdm.auto import tqdm

from topobench.transforms.liftings.graph2simplicial.clique_lifting import (
    Graph2SimplicialLifting,
    SimplicialCliqueLifting,
)


class LatentCliqueLifting(Graph2SimplicialLifting):
    r"""Lift graphs to cell complexes by identifying the cycles as 2-cells.

    Parameters
    ----------
    edge_prob_mean : float = 0.9
        Mean of the prior distribution of pie ~ Beta
        where edge_prob_mean must be in (0, 1).
        When edge_prob_mean is one, the value of edge_prob is fixed and not sampled.
    edge_prob_var : float = 0.05
        Uncertainty of the prior distribution of pie ~ Beta(a, b)
        where edge_prob_var must be in [0, inf). When edge_prob_var is zero,
        the value of edge_prob is fixed and not sampled. It is require dthat
        edge_prob_var < edge_prob_mean * (1 - edge_prob_mean). When this is not the case
        the value of edge_prob_var is set to edge_prob_mean * (1 - edge_prob_mean) - 1e-6.
    it : int, optional
        Number of iterations for sampling, by default None.
    init : str, optional
        Initialization method for the clique cover matrix, by default "edges".
    do_gibbs : bool, optional
        Whether to perform Gibbs sampling, by default False.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        edge_prob_mean: float = 0.9,
        edge_prob_var: float = 0.05,
        it=20,
        init="edges",
        do_gibbs=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.edge_prob_mean = edge_prob_mean
        min_var = self.edge_prob_mean * (1 - self.edge_prob_mean)
        self.edge_prob_var = min(edge_prob_var, 0.5 * min_var)
        self.it = it
        self.init = init
        self.do_gibbs = do_gibbs

    def lift_topology(
        self, data: torch_geometric.data.Data, verbose: bool = True
    ) -> dict:
        r"""Find the cycles of a graph and lifts them to 2-cells.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.
        verbose : bool, optional
            Whether to display verbose output, by default False.

        Returns
        -------
        dict
            The lifted topology.
        """
        # Make adjacency matrix from data
        N = data.num_nodes
        adj = np.zeros((N, N))
        for j in range(data.edge_index.shape[1]):
            adj[data.edge_index[0, j], data.edge_index[1, j]] = 1
            adj[data.edge_index[1, j], data.edge_index[0, j]] = 1

        # Create the latent clique model and fit using Gibbs sampling
        mod = _LatentCliqueModel(
            adj,
            init=self.init,
            edge_prob_mean=self.edge_prob_mean,
            edge_prob_var=self.edge_prob_var,
        )
        it = self.it if self.it is not None else data.num_edges
        mod.sample(
            sample_hypers=True,
            num_iters=it,
            do_gibbs=self.do_gibbs,
            verbose=verbose,
        )

        # # Translate fitted model to a new topology
        Zs = torch.from_numpy(mod.Z).to_sparse().float()
        cic_torch = torch.sparse.mm(Zs.T, Zs)
        cic = cic_torch.to_dense().numpy().astype(int)
        adj = np.minimum(cic - np.diag(np.diag(cic)), 1)
        edges = np.array(np.where(adj == 1))
        edges = torch.LongTensor(edges).to(data.edge_index.device)
        new_data = torch_geometric.data.Data(x=data.x, edge_index=edges)
        return SimplicialCliqueLifting(
            complex_dim=self.complex_dim
        ).lift_topology(new_data)


class _LatentCliqueModel:
    """Latent clique cover model for network data in the Partial Observability Setting.

    Extended Summary
    ----------------
    This model corresponds to the Partial Observability Setting of the Random Clique Cover
    (Williamson & Tec, 2020) and is based on the Stable Beta-Indian Buffet Process (SB-IBP).

    Williamson & Tec (2020). "Random clique covers for graphs with local density and global sparsity".
    UAI 2020. http://proceedings.mlr.press/v115/williamson20a/williamson20a.pdf

    The model relies on four parameters: alpha, sigma, c, and pie. Parameters alpha, sigma, and c
    come from the SB-IBP (Teh & Gorur, 2010), while pie determines the prior probability of edge
    unobservability. This setup helps model graphs with local density and global sparsity.

    Parameters
    ----------
    adj : np.ndarray
        Adjacency matrix of the input graph.
    edge_prob_mean : float
        Mean of the prior distribution of pie ~ Beta
        where edge_prob_mean must be in (0, 1].
        When edge_prob_var is one, the value of edge_prob is fixed and not sampled.
    edge_prob_var : float
        Uncertainty of the prior distribution of pie ~ Beta(a, b)
        where edge_prob_var must be in [0, inf). When edge_prob_var > 0, the value of pie is sampled.
    init : str, optional
        Initialization method for the clique cover matrix, by default "edges".
    seed : int, optional
        Random seed for reproducibility, by default None.

    Attributes
    ----------
    adj : np.ndarray
        Adjacency matrix of the input graph of shape (num_nodes, num_nodes).
    num_nodes : int
        Number of nodes in the graph.
    edges : np.ndarray of shape (num_edges, 2)
        Edges of the graph.
    num_edges : int
        Number of edges in the graph.
    Z : np.ndarray of shape (num_cliques, num_nodes)
        Clique cover matrix such that Zkj = 1 if node j is in clique k.
    K : int
        Number of cliques.
    alpha : float
        Parameter of the SB-iBP taking values in (0, inf).
    sigma : float
        Parameter of the SB-iBP taking values in (0, 1).
    c : float
        Parameter of the SB-iBP taking values in (-c, inf).
    edge_prob : float
        Probability of an edge observation.
    lamb : float
        Rate parameter of the Poisson distribution for the number of cliques.
        It does not influence parameter learning but is sampled for the
        likelihood computation.

    Notes
    -----
    The values of (K, N) are used interchangeably from the paper notation.
    """

    def __init__(
        self,
        adj,
        edge_prob_mean=0.9,
        edge_prob_var=0.05,
        init="edges",
        seed=None,
    ):
        self.init = init
        self.adj = adj
        self.num_nodes = adj.shape[0]
        mask = np.triu(np.ones((self.num_nodes, self.num_nodes)), 1)
        half_adj = np.multiply(adj, mask)
        self.edges = np.array(np.where(half_adj == 1)).T
        self.num_edges = len(self.edges)
        self.rng = np.random.default_rng(seed)

        # Initialize clique cover matrix
        self._init_Z()

        # Initialize parameters
        self._init_params()
        # Initialize hyperparameters
        self._init_hyperparams(edge_prob_mean, edge_prob_var)

        # Current number of clusters
        self.K = self.Z.shape[0]

    def _init_params(self):
        """Initialize the parameters of the model."""
        self.alpha = 1.0
        self.sigma = 0.5
        self.c = 0.5
        self.edge_prob = 0.98

    def _init_hyperparams(self, edge_prob_mean, edge_prob_var):
        """Initialize the hyperparameters of the model.

        Parameters
        ----------
        edge_prob_mean : float
            Mean of the prior distribution of pie ~ Beta
            where edge_prob_mean must be in (0, 1).
        edge_prob_var : float
            Uncertainty of the prior distribution of pie ~ Beta(a, b)
            where edge_prob_var must be in [0, inf).
        """
        # Validate the edge probability parameters
        assert 0 < edge_prob_mean <= 1
        assert edge_prob_var >= 0

        # Parameter prior hyper-parameters
        # The priors of alpha, sigma, c and uninformative, so their are set to
        # a default value governing the prior distribution that has little effect on the posterior
        self._alpha_params = [1.0, 1.0]
        self._sigma_params = [1.0, 1.0]
        self._c_params = [1.0, 1.0]

        # Prior for the probability of an edge observation which influences
        # the clique cover matrix. With a lower prob, there will be more latent edges
        # and therefore larger cliques. The mean, var parameterization is transformed
        # to the alpha, beta parameterization of the Beta distribution.
        self._sample_edge_prob = edge_prob_var > 0 and edge_prob_mean < 1
        self._edge_prob_params = _get_beta_params(
            edge_prob_mean, edge_prob_var
        )

    def _init_Z(self):
        """Initialize the clique cover matrix Z."""
        if self.init == "edges":
            self.Z = np.zeros((self.num_edges, self.num_nodes), dtype=int)
            for i in range(self.num_edges):
                self.Z[i, self.edges[i][0]] = 1
                self.Z[i, self.edges[i][1]] = 1
            self.lamb = self.num_edges
        elif self.init == "single":
            self.Z = np.ones((1, self.num_nodes), dtype=int)
            self.lamb = 1

    def sample(
        self,
        num_iters=1000,
        num_sm=5,
        sample_hypers=True,
        do_gibbs=False,
        verbose=False,
    ):
        """Sample from the model.

        Parameters
        ----------
        num_iters : int, optional
            Number of iterations, by default 1000.
        num_sm : int, optional
            Number of split-merge steps, by default 5.
        sample_hypers : bool, optional
            Whether to sample hyperparameters, by default True.
        do_gibbs : bool, optional
            Whether to perform Gibbs sampling, by default False.
        verbose : bool, optional
            Whether to display a progress bar, by default False.
        """
        if self.num_nodes < 500:
            num_sm = (
                20  # Increase number of split-merge steps for small graphs
            )
        pbar = tqdm(
            range(num_iters),
            desc=f"#cliques={self.K}",
            leave=False,
            disable=not verbose or self.num_nodes < 500,
        )
        for _ in pbar:
            if sample_hypers:
                self.sample_hypers()

            if do_gibbs:
                self.gibbs()

            for _ in range(num_sm):
                self.splitmerge()
            pbar.set_description(f"#cliques={self.K}")

    def log_lik(
        self, alpha=None, sigma=None, c=None, alpha_only=False, include_K=False
    ):
        """Efficient implementation of the Stable Beta-Indian Buffet Process likelihood.

        The likelihood is computed as:

        P(Z1,...,ZK) = alpha^N * exp( - alpha * A * B) * C * D^N
                   A = sum_k=1^K Gam(k - 1 + c + sigma) / Gam(k + c)
                   B = Gam(1 + c) / Gam(c + sigma)
                   C = prod_i=1^N Gam(mi - sigma) * Gam(N - mi + c + sigma)
                   D = Gam(1 + c) / Gam(c + sigma) / Gam(1 - sigma) / Gam(n + c)

        where K is the number of cliques, N is the number of nodes, and mi is the number of nodes in clique i.

        Or, equivalently:

        logP(Z1,...,ZK) = N * log(alpha) - alpha * A * B + logC + N * logD
                   A = as before
                   B = as before
                logC = sum_i=1^N log(Gam(mi - sigma)) + log(Gam(N - mi + c + sigma)
                logD = log(Gam(1 + c)) - log(Gam(c + sigma)) - log(Gam(1 - sigma)) - log(Gam(n + c))

        See Eq. 10 in Teh and Gorur (2010), "Indian Buffet Processes with Power-Law Behavior,
        Advances in Neural Information Processing Systems 23" for details.

        Parameters
        ----------
        alpha : float, optional
            Alpha parameter, by default None.
        sigma : float, optional
            Sigma parameter, by default None.
        c : float, optional
            Parameter, by default None.
        alpha_only : bool, optional
            Whether to compute likelihood with alpha only, by default False.
        include_K : bool, optional
            Whether to include the probability of the number of cliques, by default False.

        Returns
        -------
        float
            Log-likelihood value.
        """
        alpha = alpha if alpha is not None else self.alpha
        sigma = sigma if sigma is not None else self.sigma
        c = c if c is not None else self.c

        # Number of nodes and number of cliques
        N = self.num_nodes
        K = self.K

        # Compute A
        k_seq = np.arange(1, K + 1)
        A_terms = gammaln(k_seq - 1 + c + sigma) - gammaln(k_seq + c)
        A = np.exp(np.clip(A_terms, -20, 20)).sum()

        # Compute B
        B = gammaln(1 + c) - gammaln(c + sigma)

        # Compute first part of likelihood involving alpha
        ll = N * np.log(alpha) - alpha * A * B
        if alpha_only:
            return ll

        # Compute logC
        cliques_per_node = np.sum(self.Z, 0)
        logC = (
            gammaln(cliques_per_node - sigma).sum()
            + gammaln(K - cliques_per_node + c + sigma).sum()
        )

        # Compute logD
        logD = (
            gammaln(1 + c)
            - gammaln(c + sigma)
            - gammaln(1 - sigma)
            - gammaln(N + c)
        )

        # Compute the rest of the likelihood
        ll = ll + logC + N * logD

        if include_K:
            ll = ll + stats.poisson.logpmf(K, self.lamb)

        return ll

    def sample_hypers(self, step_size=0.1):
        """Sample hyperparameters using Metropolis-Hastings updates.

        Parameters
        ----------
        step_size : float, optional
            Step size for the proposal distribution, by default 0.01.
        """
        # Sample alpha
        alpha_prop = self.alpha + step_size * self.rng.normal()
        if alpha_prop > 0:
            lp_ratio = (self._alpha_params[0] - 1) * (
                np.log(alpha_prop) - np.log(self.alpha)
            ) + self._alpha_params[1] * (self.alpha - alpha_prop)

            ll_new = self.log_lik(alpha=alpha_prop, alpha_only=True)
            ll_old = self.log_lik(alpha_only=True)
            lratio = ll_new - ll_old + lp_ratio
            r = np.log(self.rng.random())
            if r < lratio:
                self.alpha = alpha_prop

        # Sample sigma
        sigma_prop = self.sigma + 0.1 * step_size * self.rng.normal()
        if 0 < sigma_prop < 1:
            ll_new = self.log_lik(sigma=sigma_prop)
            ll_old = self.log_lik()

            lp_ratio = (self._sigma_params[0] - 1) * (
                np.log(sigma_prop) - np.log(self.sigma)
            ) + (self._sigma_params[1] - 1) * (
                np.log(1 - sigma_prop) - np.log(1 - self.sigma)
            )
            lratio = ll_new - ll_old + lp_ratio
            r = np.log(self.rng.random())

            if r < lratio:
                self.sigma = sigma_prop

        # Sample pie
        edge_prob_prop = self.edge_prob + 0.1 * step_size * self.rng.normal()
        if self._sample_edge_prob and 0 < edge_prob_prop < 1:
            ll_new = self.loglikZ(pie=edge_prob_prop)
            ll_old = self.loglikZ()
            a = self._edge_prob_params[0]
            b = self._edge_prob_params[1]
            # lp ratio comes from a beta distribution
            lp_ratio = (a - 1) * (
                np.log(edge_prob_prop) - np.log(self.edge_prob)
            ) + (b - 1) * (
                np.log(1 - edge_prob_prop) - np.log(1 - self.edge_prob)
            )
            lratio = ll_new - ll_old + lp_ratio
            r = np.log(self.rng.random())
            if r < lratio:
                self.edge_prob = edge_prob_prop

        c_prop = self.c + step_size * self.rng.normal()
        if c_prop > -1 * self.sigma:
            ll_new = self.log_lik(c=c_prop)
            c_diff_new = c_prop + self.sigma
            lp_new = stats.gamma.logpdf(
                c_diff_new, self._c_params[0], scale=1 / self._c_params[1]
            )

            ll_old = self.log_lik()
            c_diff_old = self.c + self.sigma
            lp_old = stats.gamma.logpdf(
                c_diff_old, self._c_params[0], scale=1 / self._c_params[1]
            )

            lratio = ll_new - ll_old + lp_new - lp_old
            r = np.log(self.rng.random())
            if r < lratio:
                self.c = c_prop
        # Sample c
        c_prop = self.c + step_size * self.rng.normal()

        # Sample lamb, which is the rate for the number of cliques
        # in the Poisson distribution. It does not influence parameter learning.
        self.lamb = self.rng.gamma(1 + self.K, 1 / 2)

    def gibbs(self):
        """Perform Gibbs sampling step to update Z."""
        mk = np.sum(self.Z, 0)
        for node in range(self.num_nodes):
            for clique in range(self.K):
                if self.Z[clique, node] == 1:
                    self.Z[clique, node] = 0
                    ll_0 = self.loglikZn(node)
                    self.Z[clique, node] = 1
                    if not np.isinf(ll_0):
                        ll_1 = self.loglikZn(node)
                        mk[node] -= 1
                        if mk[node] == 0:
                            continue

                        prior0 = (self.K - mk[node]) / (self.K - self.sigma)
                        prior1 = 1 - prior0
                        if prior0 <= 0 or prior1 <= 0:
                            raise ValueError("prior is negative")

                        lp0 = np.log(prior0 + 1e-3) + ll_0
                        lp1 = np.log(prior1 + 1e-3) + ll_1
                        lp0 = lp0 - logsumexp([lp0, lp1])
                        r = np.log(self.rng.random())
                        if r < lp0:
                            self.Z[clique, node] = 0
                        else:
                            mk[node] += 1
                else:
                    self.Z[clique, node] = 1
                    ll_1 = self.loglikZn(node)
                    self.Z[clique, node] = 0
                    if not np.isinf(ll_1):
                        ll_0 = self.loglikZn(node)

                        if mk[node] == 0:
                            continue

                        prior0 = (self.K - mk[node]) / (self.K - self.sigma)
                        prior1 = 1 - prior0
                        lp0 = np.log(prior0 + 1e-3) + ll_0
                        lp1 = np.log(prior1 + 1e-3) + ll_1
                        lp1 = lp1 - logsumexp([lp0, lp1])
                        r = np.log(self.rng.random())
                        if r < lp1:
                            self.Z[clique, node] = 1
                            mk[node] += 1

    def loglikZ(self, Z=None, pie=None):
        """Compute the log-likelihood of the current state Z.

        Parameters
        ----------
        Z : np.ndarray, optional
            Clique cover matrix, by default None.
        pie : float, optional
            Parameter for the model, by default None.

        Returns
        -------
        float
            Log-likelihood value.
        """
        if Z is None:
            Z = self.Z
        if pie is None:
            pie = self.edge_prob
        Zs = torch.from_numpy(Z).to_sparse().float()
        cic = torch.sparse.mm(Zs.T, Zs)
        diag_idxs = torch.where(cic.indices()[0, :] == cic.indices()[1, :])[0]
        new_idxs = cic.indices()[:, diag_idxs]
        new_values = cic.values()[diag_idxs]
        size = cic.size()
        cic_diag = torch.sparse_coo_tensor(
            indices=new_idxs, values=new_values, size=size
        )
        cic = cic - cic_diag
        cic = cic.coalesce()
        cic_numpy = cic.to_dense().numpy().astype(int)

        zero_check = (1 - np.minimum(cic_numpy, 1)) * self.adj
        if np.sum(zero_check) == 0:
            p0 = (1 - pie) ** cic_numpy
            p1 = 1 - p0
            network_mask = self.adj + 1
            network_mask = np.triu(network_mask, 1) - 1
            lp_0 = np.sum(np.log(1e-6 + p0[np.where(network_mask == 0)]))
            lp_1 = np.sum(np.log(1e-6 + p1[np.where(network_mask == 1)]))
            lp = lp_0 + lp_1
        else:
            lp = -np.inf
        return lp

    def loglikZn(self, node, Z=None):
        """Compute the log-likelihood of node-specific Z.

        Parameters
        ----------
        node : int
            Node index.
        Z : np.ndarray, optional
            Clique cover matrix, by default None.

        Returns
        -------
        float
            Log-likelihood value.
        """
        if Z is None:
            Z = self.Z
        cic = np.dot(Z[:, node].T, Z)
        cic[node] = 0

        zero_check = (1 - np.minimum(cic, 1)) * self.adj[node, :]
        if np.sum(zero_check) == 0:
            p0 = (1 - self.edge_prob) ** cic
            p1 = 1 - p0
            lp0 = np.sum(np.log(1e-3 + p0[np.where(self.adj[node, :] == 0)]))
            lp1 = np.sum(np.log(1e-3 + p1[np.where(self.adj[node, :] == 1)]))
            lp = lp0 + lp1
        else:
            lp = -np.inf
        return lp

    def splitmerge(self):
        """Perform split-merge step to update Z."""
        link_id = self.rng.choice(self.num_edges)
        if self.rng.random() < 0.5:
            sender = self.edges[link_id][0]
            receiver = self.edges[link_id][1]
        else:
            sender = self.edges[link_id][1]
            receiver = self.edges[link_id][0]

        valid_cliques_i = np.where(self.Z[:, sender] == 1)[0]
        clique_i = self.rng.choice(valid_cliques_i)

        valid_cliques_j = np.where(self.Z[:, receiver] == 1)[0]
        clique_j = self.rng.choice(valid_cliques_j)

        if clique_i == clique_j:
            clique_size = self.Z[clique_i].sum()
            if clique_size <= 2:
                return  # noqa

            Z_prop = self.Z.copy()
            Z_prop = np.delete(Z_prop, clique_i, 0)
            Z_prop = np.vstack((Z_prop, np.zeros((2, self.num_nodes))))

            lqsplit = 0
            lpsplit = 0

            mk = np.sum(self.Z, 0)

            for node in range(self.num_nodes):
                if self.Z[clique_i, node] == 1:
                    if node == sender:
                        Z_prop[self.K - 1, node] = 1

                        r = self.rng.random()
                        if r < 0.5:
                            Z_prop[self.K, node] = 1
                            lpsplit = (
                                lpsplit
                                + np.log(mk[node] + 1 - self.sigma)
                                - np.log(self.K + 1 - self.sigma)
                            )
                        else:
                            lpsplit = (
                                lpsplit
                                + np.log(self.K + 1 - mk[node] - 1 + 1e-3)
                                - np.log(self.K + 1 - self.sigma)
                            )
                        lqsplit -= np.log(2)
                    elif node == receiver:
                        Z_prop[self.K, node] = 1
                        r = self.rng.random()
                        if r < 0.5:
                            Z_prop[self.K - 1, node] = 1
                            lpsplit = (
                                lpsplit
                                + np.log(mk[node] + 1 - self.sigma)
                                - np.log(self.K + 1 - self.sigma)
                            )
                        else:
                            lpsplit = (
                                lpsplit
                                + np.log(self.K - mk[node] + 1e-3)
                                - np.log(self.K + 1 - self.sigma)
                            )
                        lqsplit -= np.log(2)
                    else:
                        r = self.rng.random()
                        if r < (1 / 3):
                            Z_prop[self.K - 1, node] = 1
                            lpsplit = (
                                lpsplit
                                + np.log(self.K - mk[node] + 1e-3)
                                - np.log(self.K + 1 - self.sigma)
                            )
                        elif r < (2 / 3):
                            Z_prop[self.K, node] = 1
                            lpsplit = (
                                lpsplit
                                + np.log(self.K - mk[node] + 1e-3)
                                - np.log(self.K + 1 - self.sigma)
                            )
                        else:
                            Z_prop[self.K - 1, node] = 1
                            Z_prop[self.K, node] = 1
                            lpsplit = (
                                lpsplit
                                + np.log(mk[node] + 1 - self.sigma)
                                - np.log(self.K + 1 - self.sigma)
                            )
                        lqsplit -= np.log(3)
                else:
                    lpsplit = (
                        lpsplit
                        + np.log(self.K + 1 - mk[node])
                        - np.log(self.K + 1 - self.sigma)
                    )

            ll_prop = self.loglikZ(Z_prop)
            if not np.isinf(ll_prop):
                ll_old = self.loglikZ()
                lqsplit = (
                    lqsplit
                    - np.log(np.sum(self.Z[:, sender]))
                    - np.log(np.sum(self.Z[:, receiver]))
                )
                lqmerge = -np.log(
                    np.sum(self.Z[:, sender])
                    - self.Z[clique_i, sender]
                    + np.sum(Z_prop[:, sender])
                ) - np.log(
                    np.sum(self.Z[:, receiver])
                    - self.Z[clique_i, receiver]
                    + np.sum(Z_prop[:, receiver])
                )

                lpsplit += np.log(self.lamb / (self.K + 1))
                laccept = lpsplit - lqsplit + lqmerge + ll_prop - ll_old
                r = np.log(self.rng.random())

                if r < laccept:
                    self.Z = Z_prop.copy()
                    self.K += 1
        else:
            Z_sum = self.Z[clique_i, :] + self.Z[clique_j, :]
            Z_prop = self.Z.copy()
            Z_prop[clique_i] = np.minimum(Z_sum, 1)
            Z_prop = np.delete(Z_prop, clique_j, 0)
            ll_prop = self.loglikZ(Z_prop)
            if not np.isinf(ll_prop):
                mk = np.sum(self.Z, 0) - Z_sum
                num_affected = np.sum(Z_prop)
                if num_affected < 2:
                    raise ValueError("num_affected<2")
                lqmerge = -np.log(np.sum(self.Z[:, sender])) - np.log(
                    np.sum(self.Z[:, receiver])
                )
                lqsplit = -np.log(
                    np.sum(self.Z[:, sender])
                    - self.Z[clique_i, sender]
                    - self.Z[clique_j, sender]
                    + 1
                ) - np.log(
                    np.sum(self.Z[:, receiver])
                    - self.Z[clique_i, receiver]
                    - self.Z[clique_j, receiver]
                    + 1
                )

                lpsplit = 0
                for node in range(self.num_nodes):
                    if Z_sum[node] == 0:
                        lpsplit = (
                            lpsplit
                            + np.log(self.K - mk[node])
                            - np.log(self.K - self.sigma)
                        )
                    elif Z_sum[node] == 1:
                        lpsplit = (
                            lpsplit
                            + np.log(self.K - mk[node] - 1)
                            - np.log(self.K - self.sigma)
                        )
                    else:
                        lpsplit = (
                            lpsplit
                            + np.log(mk[node] + 1 - self.sigma)
                            - np.log(self.K - self.sigma)
                        )

                lpmerge = np.log(self.K / self.lamb)
                ll_old = self.loglikZ()

                laccept = (
                    lpmerge - lpsplit + lqsplit - lqmerge + ll_prop - ll_old
                )
                r = np.log(self.rng.random())

                if r < laccept:
                    self.Z = Z_prop.copy()
                    self.K -= 1


def _get_beta_params(mean, var):
    """Compute the parameters of a Beta distribution given the mean and variance.

    Parameters
    ----------
    mean : float
        Mean of the Beta distribution.
    var : float
        Variance of the Beta distribution.

    Returns
    -------
    tuple
        Tuple of the Beta distribution parameters.
    """
    if var == 0:
        return 1, 1
    a = mean * (mean * (1 - mean) / var - 1)
    b = a * (1 - mean) / mean
    return a, b


# def _sample_from_ibp(K, alpha, sigma, c, seed=None):
#     """
#     Auxiliary function to sample from the Indian Buffet Process.

#     Parameters
#     ----------
#     K : int
#         Number of random cliques.
#     alpha : float
#         Alpha parameter of the IBP.
#     sigma : float
#         Sigma parameter of the IBP.
#     c : float
#         Parameter of the IBP.
#     seed : int, optional
#         Random seed, by default None.

#     Returns
#     -------
#     csr_matrix
#         A sparse matrix, compressed by rows, representing the clique membership matrix.
#         Recover the adjacency matrix with min(Z'Z, 1).
#     """
#     rng = np.random.default_rng(seed)

#     k_seq = np.arange(K, dtype=float)
#     lpp = (
#         np.log(alpha)
#         + gammaln(1.0 + c)
#         - gammaln(c + sigma)
#         + gammaln(k_seq + c + sigma)
#         - gammaln(k_seq + 1.0 + c)
#     )
#     pp = np.exp(lpp)
#     new_nodes = rng.poisson(pp)
#     Ncols = new_nodes.sum()
#     node_count = np.zeros(Ncols)

#     colidx = []
#     rowidx = []
#     rightmost_node = 0

#     for n in range(K):
#         for k in range(rightmost_node):
#             prob_repeat = (node_count[k] - sigma) / (n + c)
#             if rng.random() < prob_repeat:
#                 rowidx.append(n)
#                 colidx.append(k)
#                 node_count[k] += 1

#         for k in range(rightmost_node, rightmost_node + new_nodes[n]):
#             rowidx.append(n)
#             colidx.append(k)
#             node_count[k] += 1

#         rightmost_node += new_nodes[n]

#     data = np.ones(len(rowidx), int)
#     shape = (K, Ncols)
#     Z = csr_matrix((data, (rowidx, colidx)), shape).todense()

#     # delte empty cliques
#     return Z[np.where(Z.sum(1) > 1)[0]]


# if __name__ == "__main__":
#     import networkx as nx

#     K, alpha, sigma, c, pie = 200, 3, 0.7, 1, 1.0
#     Z = _sample_from_ibp(K, alpha, sigma, c)

#     cic = Z.T @ Z
#     adj = np.minimum(cic - np.diag(np.diag(cic)), 1)

#     # delete edges with prob 1 - exp(pi^2)
#     prob = np.exp(-((1 - pie) ** 2))
#     triu_mask = np.triu(np.ones_like(adj), 1)
#     adj = np.multiply(adj, triu_mask)
#     adj = np.multiply(adj, np.random.binomial(1, prob, adj.shape))
#     adj = adj + adj.T

#     g = nx.from_numpy_matrix(adj)
#     print("Number of edges:", g.number_of_edges())
#     print("Number of nodes:", g.number_of_nodes())

#     # Transform to a torch geometric data object
#     data = torch_geometric.utils.from_networkx(g)
#     data.x = torch.ones(data.num_nodes, 1)

#     # Lift the topology to a cell complex
#     lifting = LatentCliqueLifting(edge_prob_mean=0.99, edge_prob_var=0.0)
#     complex = lifting.lift_topology(data, verbose=True)
