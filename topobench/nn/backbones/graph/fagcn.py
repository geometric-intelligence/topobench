"""Frequency Adaptation Graph Convolutional Network.

This module implements FAGCN from:

    Deyu Bo, Xiao Wang, Chuan Shi, and Huawei Shen,
    "Beyond Low-frequency Information in Graph Convolutional Networks",
    AAAI 2021.

The implementation follows the official architecture while using PyTorch
Geometric's :class:`torch_geometric.nn.FAConv` operator.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import FAConv


class FAGCN(nn.Module):
    """Frequency Adaptation Graph Convolutional Network backbone.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int
        Number of hidden channels used by all FAConv layers.
    out_channels : int, optional
        Number of output channels. When omitted, ``hidden_channels`` is used.
    num_layers : int, optional
        Number of frequency-adaptation layers.
    dropout : float, optional
        Dropout probability applied to node features and attention
        coefficients.
    eps : float, optional
        Residual coefficient multiplying the initial hidden representation.
    cached : bool, optional
        Whether normalized graph structure should be cached. This should
        remain ``False`` for inductive mini-batch training.
    add_self_loops : bool, optional
        Whether FAConv adds self-loops before normalization.
    normalize : bool, optional
        Whether FAConv computes symmetric normalization coefficients.
    **kwargs
        Additional arguments accepted for TopoBench compatibility.

    Notes
    -----
    The official FAGCN architecture first projects the input features, stores
    that representation as the residual signal, applies several adaptive
    frequency layers, adds ``eps`` times the residual signal at every layer,
    and finally applies a linear output projection.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.5,
        eps: float = 0.1,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if out_channels is not None and out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must lie in [0, 1]")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels or hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.eps = eps

        self.input_linear = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList(
            [
                FAConv(
                    channels=hidden_channels,
                    eps=eps,
                    dropout=dropout,
                    cached=cached,
                    add_self_loops=add_self_loops,
                    normalize=normalize,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_linear = nn.Linear(hidden_channels, self.out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset all trainable parameters."""
        nn.init.xavier_normal_(self.input_linear.weight, gain=1.414)
        nn.init.zeros_(self.input_linear.bias)
        nn.init.xavier_normal_(self.output_linear.weight, gain=1.414)
        nn.init.zeros_(self.output_linear.bias)
        for layer in self.layers:
            layer.reset_parameters()

    @staticmethod
    def _resolve_edge_weight(
        edge_weight: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Resolve scalar edge weights from TopoBench graph inputs.

        Parameters
        ----------
        edge_weight : torch.Tensor or None
            Explicit scalar edge weights.
        edge_attr : torch.Tensor or None
            Optional edge attributes. One-dimensional attributes, or
            two-dimensional attributes with one channel, are interpreted as
            scalar edge weights.

        Returns
        -------
        torch.Tensor or None
            Scalar edge weights accepted by :class:`FAConv`.
        """
        if edge_weight is not None:
            return edge_weight
        if edge_attr is None:
            return None
        if edge_attr.ndim == 1:
            return edge_attr
        if edge_attr.ndim == 2 and edge_attr.shape[-1] == 1:
            return edge_attr.reshape(-1)
        return None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute FAGCN node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Graph connectivity with shape ``[2, num_edges]``.
        batch : torch.Tensor or None, optional
            Node-to-graph assignment vector. FAGCN does not use it directly.
        edge_weight : torch.Tensor or None, optional
            Scalar edge weights.
        edge_attr : torch.Tensor or None, optional
            Optional edge attributes. Scalar attributes are used as edge
            weights when ``edge_weight`` is absent.
        **kwargs
            Additional arguments accepted for wrapper compatibility.

        Returns
        -------
        torch.Tensor
            Node embeddings with shape ``[num_nodes, out_channels]``.
        """
        del batch, kwargs

        if x.ndim != 2:
            raise ValueError("x must be a rank-two tensor")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"received {x.shape[-1]}"
            )

        # PyG's FAConv requires ``edge_weight`` to be ``None`` whenever
        # internal symmetric normalization is enabled. FAGCN's reference
        # architecture uses that normalized path, so optional TopoBench edge
        # attributes are accepted by the interface but intentionally ignored.
        resolved_weight = None
        if not self.layers[0].normalize:
            resolved_weight = self._resolve_edge_weight(edge_weight, edge_attr)

        hidden = F.dropout(x, p=self.dropout, training=self.training)
        hidden = F.relu(self.input_linear(hidden))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        initial_hidden = hidden

        for layer in self.layers:
            hidden = layer(
                hidden,
                initial_hidden,
                edge_index,
                edge_weight=resolved_weight,
            )

        return self.output_linear(hidden)
