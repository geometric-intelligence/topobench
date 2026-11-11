"""Generalized Simplicial Attention Neural Network (GSAN) backbone.

This module implements GSAN, introduced in

    Claudio Battiloro, Lucia Testa, Lorenzo Giusti, Stefania Sardellitti,
    Paolo Di Lorenzo, Sergio Barbarossa.
    "Generalized Simplicial Attention Neural Networks."
    IEEE Transactions on Signal and Information Processing over Networks, 2024.
    https://arxiv.org/abs/2309.02138

Reference implementation: https://github.com/luciatesta97/Generalized-Simplicial-Attention-Neural-Networks
(the single-order ``SALayer`` in ``GSAN_SAN/layers/simplicial_attention_layer.py``
and the all-orders coupled ``SAN`` in ``GSAN_Joint/NCI1/net.py``).

Notes
-----
GSAN generalizes the (edge-only) Simplicial Attention Network to signals living
on simplices of *every* order and couples the orders through the incidence maps.
For a signal :math:`x_k` on order-k simplices, a masked multi-head attention is
run separately over the **lower** (irrotational) neighborhood — masked by the
sparsity of :math:`L_\\downarrow^{(k)} = B_k^\\top B_k` — and the **upper**
(solenoidal) neighborhood — masked by :math:`L_\\uparrow^{(k)} = B_{k+1} B_{k+1}^\\top`.
Each branch applies a ``K``-tap polynomial (higher-hop) diffusion of the learned,
topologically masked attention matrix :math:`\\alpha`:

.. math::

    z = \\sum_{j=0}^{K-1} \\big(\\alpha \\cdot L^{\\,j}\\big)\\, x\\, W_j ,
    \\qquad L^0 = I,

following the *correct* ``SALayer`` indexing (the public ``GSAN_Joint`` loop
double-counts the ``j=0`` tap and raises the attention matrix rather than the
static Laplacian — we use the paper form).

The orders are coupled (``GSAN_Joint``): node features receive a lower projection
of the edge signal (via :math:`B_1`), edge features receive both a lower
projection of nodes (:math:`B_1^\\top`) and an upper projection of triangles
(:math:`B_2`), and triangle features receive a lower projection of edges
(:math:`B_2^\\top`). The harmonic component is disabled, exactly as in the
reference joint model.

Within TopoBench the operators are taken directly from the simplicial batch:
``hodge_laplacian_0`` (= :math:`L_\\uparrow^{(0)}`), ``down_laplacian_1``,
``up_laplacian_1``, ``down_laplacian_2`` and the incidences ``incidence_1``,
``incidence_2``; the full edge Laplacian is recovered as
``down_laplacian_1 + up_laplacian_1``.
"""

import torch
import torch.nn.functional as F
from torch import nn

_NEG_INF = -9e15


def _to_dense(matrix: torch.Tensor) -> torch.Tensor:
    """Return a dense view of a (possibly sparse) operator.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse or dense 2D tensor.

    Returns
    -------
    torch.Tensor
        Dense 2D tensor.
    """
    return matrix.to_dense() if matrix.is_sparse else matrix


class GSANLayer(nn.Module):
    """A single all-orders Generalized Simplicial Attention layer.

    Each call updates node, edge and triangle signals with masked attention over
    their lower/upper neighborhoods plus the cross-order incidence projections.

    Parameters
    ----------
    channels : int
        Common feature dimension of all orders.
    K : int
        Number of polynomial filter taps (``kappa`` in the paper).
    dropout : float
        Dropout applied to the attention coefficients.
    alpha_leaky_relu : float
        Negative slope of the LeakyReLU used in attention scoring.
    update_func : str or None
        Nonlinearity applied to each updated signal: ``"sigmoid"`` (reference
        joint model), ``"relu"`` or ``None``.
    """

    def __init__(
        self,
        channels: int,
        K: int,
        dropout: float,
        alpha_leaky_relu: float,
        update_func: str | None,
    ):
        super().__init__()
        self.channels = channels
        self.K = K
        self.dropout = dropout
        self.update_func = update_func
        self.leaky_relu = nn.LeakyReLU(alpha_leaky_relu)

        # Self (W) and cross-order (A) tap weights, shared across orders.
        self.W = nn.Parameter(torch.empty(K, channels, channels))
        self.A = nn.Parameter(torch.empty(K, channels, channels))
        # One GAT-style attention vector per (masked) attention term:
        # nodes: self (00), edge->node (01);
        # edges: self (10), node->edge (11), triangle->edge (12);
        # triangles: self (20), edge->triangle (21).
        self.att = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(2 * channels * K, 1))
                for name in ("a00", "a01", "a10", "a11", "a12", "a20", "a21")
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize all learnable parameters with Xavier uniform."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W, gain=gain)
        nn.init.xavier_uniform_(self.A, gain=gain)
        for p in self.att.values():
            nn.init.xavier_uniform_(p, gain=gain)

    def _attention(
        self,
        x_att: torch.Tensor,
        taps: torch.Tensor,
        att_vec: torch.Tensor,
        laplacian: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a topologically masked attention matrix.

        Parameters
        ----------
        x_att : torch.Tensor
            Features used to score attention, shape ``(M, channels)``.
        taps : torch.Tensor
            Tap weight tensor ``(K, channels, channels)``.
        att_vec : torch.Tensor
            GAT attention vector ``(2 * channels * K, 1)``.
        laplacian : torch.Tensor
            Dense neighborhood operator ``(M, M)`` whose nonzero pattern is the
            attention mask.

        Returns
        -------
        torch.Tensor
            Row-stochastic attention matrix ``(M, M)``.
        """
        split = self.channels * self.K
        x_f = torch.cat([x_att @ taps[k] for k in range(self.K)], dim=1)
        scores = self.leaky_relu(
            (x_f @ att_vec[:split]) + (x_f @ att_vec[split:]).T
        )
        scores = torch.where(
            laplacian != 0, scores, torch.full_like(scores, _NEG_INF)
        )
        alpha = F.softmax(scores, dim=1)
        return F.dropout(alpha, self.dropout, training=self.training)

    def _branch(
        self,
        x_att: torch.Tensor,
        x_val: torch.Tensor,
        taps: torch.Tensor,
        att_vec: torch.Tensor,
        laplacian: torch.Tensor,
    ) -> torch.Tensor:
        """Masked-attention + K-tap polynomial diffusion for one branch.

        Returns the zero tensor (correct shape) when the branch is empty or its
        neighborhood operator has no nonzero entries (e.g. graphs without
        triangles).

        Parameters
        ----------
        x_att : torch.Tensor
            Features that drive the attention scores, shape ``(M, channels)``.
        x_val : torch.Tensor
            Features that are diffused (the message), shape ``(M, channels)``.
        taps : torch.Tensor
            Tap weights ``(K, channels, channels)``.
        att_vec : torch.Tensor
            GAT attention vector.
        laplacian : torch.Tensor
            Dense neighborhood operator ``(M, M)``.

        Returns
        -------
        torch.Tensor
            Branch contribution of shape ``(M, channels)``.
        """
        if x_val.shape[0] == 0 or laplacian.numel() == 0:
            return torch.zeros_like(x_val)
        if not torch.any(laplacian != 0):
            return torch.zeros_like(x_val)

        alpha = self._attention(x_att, taps, att_vec, laplacian)
        accum = alpha
        out = accum @ (x_val @ taps[0])
        for j in range(1, self.K):
            accum = accum @ laplacian
            out = out + accum @ (x_val @ taps[j])
        return out

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the configured update nonlinearity.

        Parameters
        ----------
        x : torch.Tensor
            Pre-activation features.

        Returns
        -------
        torch.Tensor
            Activated features.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.relu(x)
        return x

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        operators: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update node, edge and triangle signals.

        Parameters
        ----------
        x_0, x_1, x_2 : torch.Tensor
            Node, edge and triangle features, each ``(N_k, channels)``.
        operators : dict of torch.Tensor
            Dense operators with keys ``l0, ld1, lu1, l1, ld2, b1, b2``.

        Returns
        -------
        tuple of torch.Tensor
            Updated ``(x_0, x_1, x_2)``.
        """
        l0 = operators["l0"]
        ld1, lu1, l1 = operators["ld1"], operators["lu1"], operators["l1"]
        ld2 = operators["ld2"]
        b1, b2 = operators["b1"], operators["b2"]

        # Nodes: self-attention over L0 + lower projection of edges (B1 @ x_1).
        z0 = self._branch(x_0, x_0, self.W, self.att["a00"], l0)
        z0 = z0 + self._branch(x_0, b1 @ x_1, self.A, self.att["a01"], l0)

        # Edges: self over L1 + node lower projection + triangle upper projection.
        z1 = self._branch(x_1, x_1, self.W, self.att["a10"], l1)
        z1 = z1 + self._branch(
            b1.T @ x_0, b1.T @ x_0, self.A, self.att["a11"], ld1
        )
        z1 = z1 + self._branch(
            b2 @ x_2, b2 @ x_2, self.A, self.att["a12"], lu1
        )

        # Triangles: self over L_down2 + edge lower projection.
        z2 = self._branch(x_2, x_2, self.W, self.att["a20"], ld2)
        z2 = z2 + self._branch(
            b2.T @ x_1, b2.T @ x_1, self.A, self.att["a21"], ld2
        )

        return self._activate(z0), self._activate(z1), self._activate(z2)


class GSAN(nn.Module):
    """Generalized Simplicial Attention Network backbone.

    Projects the per-order input features to a common hidden width and applies a
    stack of :class:`GSANLayer` blocks coupling nodes, edges and triangles.

    Parameters
    ----------
    in_channels_all : tuple of int
        Input feature dimensions ``(nodes, edges, triangles)``.
    hidden_channels_all : tuple of int
        Hidden feature dimensions ``(nodes, edges, triangles)``; all three must
        be equal (the coupled attention requires a shared width).
    K : int, optional
        Number of polynomial filter taps (default: 2).
    n_layers : int, optional
        Number of stacked GSAN layers (default: 2).
    dropout : float, optional
        Attention dropout (default: 0.2).
    alpha_leaky_relu : float, optional
        LeakyReLU negative slope in attention scoring (default: 0.2).
    update_func : str or None, optional
        Per-layer nonlinearity: ``"sigmoid"``, ``"relu"`` or ``None``
        (default: ``"sigmoid"``).
    """

    def __init__(
        self,
        in_channels_all: tuple[int, int, int],
        hidden_channels_all: tuple[int, int, int],
        K: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
        alpha_leaky_relu: float = 0.2,
        update_func: str | None = "sigmoid",
    ):
        super().__init__()
        assert K >= 1, "K (filter taps) must be a positive integer."
        assert n_layers >= 1, "n_layers must be a positive integer."
        channels = hidden_channels_all[0]
        assert all(h == channels for h in hidden_channels_all), (
            "GSAN couples orders and requires a single shared hidden width "
            "across all ranks."
        )

        self.in_linear_0 = nn.Linear(in_channels_all[0], channels)
        self.in_linear_1 = nn.Linear(in_channels_all[1], channels)
        self.in_linear_2 = nn.Linear(in_channels_all[2], channels)
        self.layers = nn.ModuleList(
            [
                GSANLayer(
                    channels=channels,
                    K=K,
                    dropout=dropout,
                    alpha_leaky_relu=alpha_leaky_relu,
                    update_func=update_func,
                )
                for _ in range(n_layers)
            ]
        )

    def reset_parameters(self) -> None:
        """Reset all learnable parameters of the backbone."""
        for lin in (self.in_linear_0, self.in_linear_1, self.in_linear_2):
            lin.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self,
        x_all: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        laplacian_all: tuple[torch.Tensor, ...],
        incidence_all: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x_all : tuple of torch.Tensor
            Per-rank features ``(x_0, x_1, x_2)``.
        laplacian_all : tuple of torch.Tensor
            ``(hodge_laplacian_0, down_laplacian_1, up_laplacian_1,
            down_laplacian_2, up_laplacian_2)`` as in the SCCNN wrapper.
        incidence_all : tuple of torch.Tensor
            ``(incidence_1, incidence_2)``.

        Returns
        -------
        tuple of torch.Tensor
            Updated ``(x_0, x_1, x_2)`` at the hidden width.
        """
        x_0, x_1, x_2 = x_all
        x_0 = self.in_linear_0(x_0)
        x_1 = self.in_linear_1(x_1)
        x_2 = self.in_linear_2(x_2)

        l0, ld1, lu1, ld2, _lu2 = (_to_dense(m) for m in laplacian_all)
        b1, b2 = (_to_dense(m) for m in incidence_all)
        operators = {
            "l0": l0,
            "ld1": ld1,
            "lu1": lu1,
            "l1": ld1 + lu1,
            "ld2": ld2,
            "b1": b1,
            "b2": b2,
        }

        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, x_2, operators)
        return x_0, x_1, x_2
