"""JacobiConv spectral graph backbone with polynomial coefficient decomposition."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class _NormalizedPropagation(MessagePassing):
    """Apply one normalized graph-propagation step."""

    def __init__(self) -> None:
        """Initialize additive message passing."""
        super().__init__(aggr="add")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propagate node features over the normalized graph.

        Parameters
        ----------
        x : torch.Tensor
            Node-feature matrix of shape ``[num_nodes, num_features]``.
        edge_index : torch.Tensor
            Graph connectivity in COO format with shape ``[2, num_edges]``.
        edge_weight : torch.Tensor or None, optional
            Optional scalar edge weights.

        Returns
        -------
        torch.Tensor
            Propagated node features with the same shape as ``x``.
        """
        edge_index, norm = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes=x.size(0),
            add_self_loops=True,
            dtype=x.dtype,
        )
        return self.propagate(edge_index, x=x, norm=norm)

    def message(
        self,
        x_j: torch.Tensor,
        norm: torch.Tensor,
    ) -> torch.Tensor:
        """Construct normalized messages from neighbouring nodes.

        Parameters
        ----------
        x_j : torch.Tensor
            Source-node features for each edge.
        norm : torch.Tensor
            Scalar normalization coefficient for each edge.

        Returns
        -------
        torch.Tensor
            Normalized edge messages.
        """
        return norm.view(-1, 1) * x_j


class JacobiConv(nn.Module):
    """Jacobi-polynomial spectral graph neural network.

    The model computes a collection of Jacobi polynomial graph signals and
    combines them using learnable output-channel-specific coefficients.

    Parameters
    ----------
    in_channels : int or None, optional
        Number of input node-feature channels.
    hidden_channels : int or None, optional
        Hidden feature dimension retained for TopoBench configuration
        compatibility.
    out_channels : int or None, optional
        Number of output node-feature channels.
    polynomial_order : int, optional
        Maximum order of the Jacobi polynomial expansion.
    jacobi_alpha : float, optional
        First Jacobi polynomial parameter. Must be greater than ``-1``.
    jacobi_beta : float, optional
        Second Jacobi polynomial parameter. Must be greater than ``-1``.
    pcd_scale : float, optional
        Scale applied to the polynomial coefficient decomposition factors.
    dropout : float, optional
        Dropout probability applied before the input projection.
    input_dim : int or None, optional
        Alias for ``in_channels`` used by some TopoBench configurations.
    hidden_dim : int or None, optional
        Alias for ``hidden_channels``.
    **kwargs
        Additional unused keyword arguments accepted for wrapper
        compatibility.

    Raises
    ------
    ValueError
        If required dimensions are missing or a hyperparameter is outside
        its valid range.
    """

    def __init__(
        self,
        in_channels: int | None = None,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        polynomial_order: int = 10,
        jacobi_alpha: float = 1.0,
        jacobi_beta: float = 1.0,
        pcd_scale: float = 1.0,
        dropout: float = 0.5,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        in_channels = in_channels if in_channels is not None else input_dim
        hidden_channels = (
            hidden_channels if hidden_channels is not None else hidden_dim
        )
        if in_channels is None:
            raise ValueError("in_channels or input_dim must be provided")
        hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        out_channels = (
            hidden_channels if out_channels is None else out_channels
        )
        if polynomial_order < 0:
            raise ValueError("polynomial_order must be non-negative")
        if jacobi_alpha <= -1.0 or jacobi_beta <= -1.0:
            raise ValueError("Jacobi alpha and beta must be greater than -1")
        if pcd_scale <= 0:
            raise ValueError("pcd_scale must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.polynomial_order = polynomial_order
        self.jacobi_alpha = jacobi_alpha
        self.jacobi_beta = jacobi_beta
        self.pcd_scale = pcd_scale
        self.dropout = dropout

        # The paper first linearly transforms X and uses one filter per output channel.
        self.input_projection = nn.Linear(in_channels, out_channels, bias=True)
        self.filter_coefficients = nn.Parameter(
            torch.ones(polynomial_order + 1, out_channels)
        )
        # PCD: alpha_{k,l} = beta_{k,l} prod_{i=1}^k gamma_i,
        # with gamma_i = gamma_0 tanh(eta_i).
        self.pcd_logits = nn.Parameter(torch.zeros(polynomial_order))
        self.propagate_once = _NormalizedPropagation()

    @staticmethod
    def _coerce_edge_weight(
        edge_weight: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Select scalar edge weights from the available edge inputs.

        Parameters
        ----------
        edge_weight : torch.Tensor or None
            Explicit scalar edge weights.
        edge_attr : torch.Tensor or None
            Optional edge attributes.

        Returns
        -------
        torch.Tensor or None
            One-dimensional edge weights when available, otherwise ``None``.
        """
        if edge_weight is not None:
            return edge_weight
        if edge_attr is None:
            return None
        if edge_attr.dim() == 1:
            return edge_attr
        if edge_attr.dim() == 2 and edge_attr.size(-1) == 1:
            return edge_attr.view(-1)
        return None

    def _jacobi_terms(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        """Compute Jacobi polynomial graph-signal terms.

        Parameters
        ----------
        x : torch.Tensor
            Projected node features.
        edge_index : torch.Tensor
            Graph connectivity in COO format.
        edge_weight : torch.Tensor or None
            Optional scalar edge weights.

        Returns
        -------
        list of torch.Tensor
            Polynomial terms from order zero through ``polynomial_order``.
        """
        terms = [x]
        if self.polynomial_order == 0:
            return terms

        a, b = self.jacobi_alpha, self.jacobi_beta
        gamma = self.pcd_scale * torch.tanh(self.pcd_logits)
        ax = self.propagate_once(x, edge_index, edge_weight)
        p1 = gamma[0] * (((a - b) / 2.0) * x + ((a + b + 2.0) / 2.0) * ax)
        terms.append(p1)

        for k in range(2, self.polynomial_order + 1):
            kf = float(k)
            common = 2.0 * kf + a + b
            denom = 2.0 * kf * (kf + a + b) * (common - 2.0)
            theta = (common - 1.0) * common * (common - 2.0) / denom
            theta_prime = (common - 1.0) * (a * a - b * b) / denom
            theta_double = (
                2.0 * (kf + a - 1.0) * (kf + b - 1.0) * common / denom
            )
            propagated = self.propagate_once(
                terms[-1], edge_index, edge_weight
            )
            current = (
                gamma[k - 1] * (theta * propagated + theta_prime * terms[-1])
                - gamma[k - 1] * gamma[k - 2] * theta_double * terms[-2]
            )
            terms.append(current)
        return terms

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute JacobiConv node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node-feature matrix of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Graph connectivity in COO format with shape ``[2, num_edges]``.
        batch : torch.Tensor or None, optional
            Graph assignment vector accepted for wrapper compatibility.
        edge_weight : torch.Tensor or None, optional
            Optional scalar edge weights.
        edge_attr : torch.Tensor or None, optional
            Optional edge attributes. One-dimensional attributes may be used
            as scalar edge weights.
        **kwargs
            Additional unused keyword arguments.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``[num_nodes, out_channels]``.
        """
        del batch, kwargs
        edge_weight = self._coerce_edge_weight(edge_weight, edge_attr)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_projection(x)
        terms = self._jacobi_terms(x, edge_index, edge_weight)
        stacked = torch.stack(terms, dim=0)
        return (stacked * self.filter_coefficients[:, None, :]).sum(dim=0)
