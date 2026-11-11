"""PhenomNN: from hypergraph energy functions to hypergraph neural networks.

This module implements PhenomNN, introduced in

    Yuxin Wang, Quan Gan, Xipeng Qiu, Xuanjing Huang, David Wipf.
    "From Hypergraph Energy Functions to Hypergraph Neural Networks."
    ICML 2023. https://arxiv.org/abs/2306.09623

Reference implementation: https://github.com/yxzwang/PhenomNN
(``models/phenomnn.py``, ``models/phenomnn_s.py`` and the operator construction
``B2A`` in ``train_faster.py``).

Notes
-----
PhenomNN defines a hypergraph-regularized energy whose two smoothness terms
correspond to the **clique expansion** (subscript :math:`\\beta`) and the
**star expansion** (subscript :math:`\\gamma`) of the hypergraph, plus a fidelity
term to the encoded input features :math:`Y^{(0)}`. A single PhenomNN layer is an
unfolded proximal-gradient step that minimizes this energy, iterated for
``prop_step`` steps (Eq. (6) of the paper):

.. math::

    \\hat Y = \\lambda_0\\big(A_\\beta Y (H_\\beta+H_\\beta^\\top) - D_\\beta Y H_\\beta H_\\beta^\\top\\big)
            + Y^{(0)}
            + \\lambda_1\\big(L_\\gamma Y + A_\\gamma Y (H_\\gamma+H_\\gamma^\\top)
                              - D_\\gamma Y H_\\gamma H_\\gamma^\\top\\big),

    Y \\leftarrow (1-\\alpha) Y + \\alpha\\, \\tilde Q^{-1} \\hat Y,
    \\qquad \\tilde Q = \\lambda_0 D_\\beta + \\lambda_1 D_\\gamma + I,

with :math:`L_\\gamma = D_\\gamma - A_\\gamma` and learnable compatibility matrices
:math:`H_\\beta, H_\\gamma`. Setting ``compatibility=False`` (i.e. :math:`H = I`)
recovers the parameter-free *PhenomNN_simple* update
:math:`\\hat Y = \\lambda_0 A_\\beta Y + Y^{(0)} + \\lambda_1 A_\\gamma Y`.

Both expansion operators are reconstructed from the node-hyperedge incidence
matrix :math:`B` (TopoBench's ``incidence_hyperedges``):

* clique (node-normalized):
  :math:`A_\\beta = \\hat D^{-1/2}(BB^\\top + I)\\hat D^{-1/2}`,
* star (edge-size normalized):
  :math:`A_\\gamma = \\hat D^{-1/2}(B D_E^{-1} B^\\top + I)\\hat D^{-1/2}`,
  with :math:`D_E = \\mathrm{diag}(\\mathbf{1}^\\top B)` the hyperedge sizes,

and :math:`D_\\beta, D_\\gamma` are the row-sum degrees of the renormalized
adjacencies. The input/output linear maps of the original architecture are
provided by TopoBench's feature encoder and readout, so the backbone operates on
pre-encoded, equal-dimensional node features (as EDGNN does).
"""

import torch
import torch.nn.functional as F
from torch import nn


def _renormalize(adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetrically normalize a dense adjacency and return its degree vector.

    Computes :math:`\\hat A = \\hat D^{-1/2} A \\hat D^{-1/2}` with
    :math:`\\hat D = \\mathrm{diag}(A\\mathbf{1})`, then returns the renormalized
    adjacency together with its (recomputed) row-sum degree vector.

    Parameters
    ----------
    adj : torch.Tensor
        Dense ``(n, n)`` adjacency matrix (already including self-loops).

    Returns
    -------
    tuple of torch.Tensor
        The renormalized adjacency ``(n, n)`` and its degree vector ``(n,)``.
    """
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)
    adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    deg = adj.sum(dim=1)
    return adj, deg


def build_expansion_operators(
    incidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the clique- and star-expansion operators from a hypergraph incidence.

    Parameters
    ----------
    incidence : torch.Tensor
        Node-hyperedge incidence matrix :math:`B` of shape
        ``(n_nodes, n_hyperedges)`` (sparse or dense; values are treated as
        unsigned membership).

    Returns
    -------
    tuple of torch.Tensor
        ``(A_beta, A_gamma, D_beta, D_gamma)`` where the adjacencies are dense
        ``(n, n)`` tensors and the degree vectors are ``(n,)``.
    """
    if incidence.is_sparse:
        incidence = incidence.to_dense()
    b = incidence.abs()
    n = b.shape[0]
    identity = torch.eye(n, device=b.device, dtype=b.dtype)

    # Clique expansion (node-normalized): A_beta = D^-1/2 (B B^T + I) D^-1/2.
    a_beta = b @ b.t() + identity
    a_beta, d_beta = _renormalize(a_beta)

    # Star expansion (edge-size normalized): A_gamma = D^-1/2 (B D_E^-1 B^T + I) D^-1/2.
    edge_size_inv = b.sum(dim=0).pow(-1)
    edge_size_inv.masked_fill_(torch.isinf(edge_size_inv), 0.0)
    a_gamma = (b * edge_size_inv.unsqueeze(0)) @ b.t() + identity
    a_gamma, d_gamma = _renormalize(a_gamma)

    return a_beta, a_gamma, d_beta, d_gamma


class PhenomNNConv(nn.Module):
    """A single unfolded PhenomNN energy-descent layer.

    Parameters
    ----------
    channels : int
        Node feature dimension.
    prop_step : int
        Number of unfolded iterations (``T`` in the paper).
    lam0 : float
        Weight :math:`\\lambda_0` of the clique-expansion energy term.
    lam1 : float
        Weight :math:`\\lambda_1` of the star-expansion energy term.
    alpha : float
        Step size of the unfolded update.
    compatibility : bool
        If ``True``, use learnable compatibility matrices :math:`H_\\beta,
        H_\\gamma`; if ``False``, reduce to the parameter-free simple update.
    """

    def __init__(
        self,
        channels: int,
        prop_step: int,
        lam0: float,
        lam1: float,
        alpha: float,
        compatibility: bool,
    ):
        super().__init__()
        self.channels = channels
        self.prop_step = prop_step
        self.lam0 = lam0
        self.lam1 = lam1
        self.alpha = alpha
        self.compatibility = compatibility

        if compatibility:
            self.H_beta = nn.Parameter(torch.empty(channels, channels))
            self.H_gamma = nn.Parameter(torch.empty(channels, channels))
        else:
            self.register_parameter("H_beta", None)
            self.register_parameter("H_gamma", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize each compatibility matrix as identity plus small noise."""
        if self.compatibility:
            bound = 1.0 / self.channels
            for h in (self.H_beta, self.H_gamma):
                nn.init.normal_(h, mean=0.0, std=bound)
                with torch.no_grad():
                    h.add_(torch.eye(self.channels, device=h.device))

    def forward(
        self,
        x: torch.Tensor,
        operators: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> torch.Tensor:
        """Run the unfolded energy descent.

        Parameters
        ----------
        x : torch.Tensor
            Encoded node features ``(n, channels)`` used as both the initial
            iterate and the fidelity anchor :math:`Y^{(0)}`.
        operators : tuple of torch.Tensor
            ``(A_beta, A_gamma, D_beta, D_gamma)`` from
            :func:`build_expansion_operators`.

        Returns
        -------
        torch.Tensor
            Smoothed node features ``(n, channels)``.
        """
        a_beta, a_gamma, d_beta, d_gamma = operators
        q_inv = (self.lam0 * d_beta + self.lam1 * d_gamma + 1.0).pow(-1)
        q_inv = q_inv.unsqueeze(1)

        y = y0 = x
        if self.compatibility:
            h_beta = self.H_beta + self.H_beta.t()
            hh_beta = self.H_beta @ self.H_beta.t()
            h_gamma = self.H_gamma + self.H_gamma.t()
            hh_gamma = self.H_gamma @ self.H_gamma.t()
            d_beta_col = d_beta.unsqueeze(1)
            d_gamma_col = d_gamma.unsqueeze(1)

        for _ in range(self.prop_step):
            if self.compatibility:
                beta_term = self.lam0 * (
                    a_beta @ y @ h_beta - d_beta_col * (y @ hh_beta)
                )
                # L_gamma @ Y = D_gamma Y - A_gamma Y
                l_gamma_y = d_gamma_col * y - a_gamma @ y
                gamma_term = self.lam1 * (
                    l_gamma_y
                    + a_gamma @ y @ h_gamma
                    - d_gamma_col * (y @ hh_gamma)
                )
                y_hat = beta_term + y0 + gamma_term
            else:
                y_hat = (
                    self.lam0 * (a_beta @ y) + y0 + self.lam1 * (a_gamma @ y)
                )
            y = (1 - self.alpha) * y + self.alpha * (q_inv * y_hat)
        return y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(channels={self.channels}, "
            f"prop_step={self.prop_step}, lam0={self.lam0}, lam1={self.lam1}, "
            f"alpha={self.alpha:.4g}, compatibility={self.compatibility})"
        )


class PhenomNN(nn.Module):
    """PhenomNN hypergraph backbone.

    Stacks ``num_layers`` unfolded energy-descent blocks (each a
    :class:`PhenomNNConv`), with dropout and a ReLU nonlinearity between blocks.
    The expansion operators are built once per forward pass from the hypergraph
    incidence matrix and shared across blocks.

    Parameters
    ----------
    num_features : int
        Dimension of the (encoded) input node features and of every block.
    num_layers : int, optional
        Number of stacked PhenomNN blocks (default: 1).
    prop_step : int, optional
        Number of unfolded iterations within each block (default: 16).
    lam0 : float, optional
        Clique-expansion energy weight :math:`\\lambda_0` (default: 10.0).
    lam1 : float, optional
        Star-expansion energy weight :math:`\\lambda_1` (default: 10.0).
    alpha : float or None, optional
        Step size; if ``None`` it defaults to ``1 / (1 + lam0 + lam1)`` as in the
        reference implementation (default: None).
    dropout : float, optional
        Dropout applied before each block (default: 0.5).
    compatibility : bool, optional
        Use learnable compatibility matrices (full PhenomNN) when ``True``;
        otherwise the parameter-free PhenomNN_simple update (default: True).
    """

    def __init__(
        self,
        num_features: int,
        num_layers: int = 1,
        prop_step: int = 16,
        lam0: float = 10.0,
        lam1: float = 10.0,
        alpha: float | None = None,
        dropout: float = 0.5,
        compatibility: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be a positive integer."
        assert prop_step >= 1, "prop_step must be a positive integer."

        resolved_alpha = (
            alpha if alpha is not None else 1.0 / (1.0 + lam0 + lam1)
        )
        self.dropout = dropout
        self.convs = nn.ModuleList(
            [
                PhenomNNConv(
                    channels=num_features,
                    prop_step=prop_step,
                    lam0=lam0,
                    lam1=lam1,
                    alpha=resolved_alpha,
                    compatibility=compatibility,
                )
                for _ in range(num_layers)
            ]
        )

    def reset_parameters(self) -> None:
        """Reset all learnable parameters of the backbone."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        incidence: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Encoded node features ``(n_nodes, num_features)``.
        incidence : torch.Tensor
            Node-hyperedge incidence matrix ``(n_nodes, n_hyperedges)``.

        Returns
        -------
        tuple
            ``(x_0, None)`` where ``x_0`` is the updated node embedding; the
            second element follows the hypergraph wrapper's ``(nodes, hyperedges)``
            return contract (PhenomNN produces no hyperedge embedding).
        """
        operators = build_expansion_operators(incidence)
        h = x
        for conv in self.convs:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(conv(h, operators))
        return h, None
