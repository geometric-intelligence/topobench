"""EHNN: Equivariant Hypergraph Neural Networks (linear / MLP variant).

This module implements the EHNN-MLP (``linear``) model introduced in

    Jinwoo Kim, Saeyoon Oh, Sungjun Cho, Seunghoon Hong.
    "Equivariant Hypergraph Neural Networks." ECCV 2022.
    https://arxiv.org/abs/2208.10428

Reference implementation: https://github.com/jw9730/ehnn
(``models.py``, ``linear.py``, ``hypernetwork.py``, ``mlp.py``).

Notes
-----
EHNN derives the *maximally expressive* linear equivariant layer for hypergraphs
of arbitrary, mixed order. A layer is built from order-equivariant aggregation
blocks whose parameters are produced by small hypernetworks conditioned on the
**order** (size) of each hyperedge, so that hyperedges of different cardinalities
are treated by a shared but order-aware basis. The ``linear`` variant alternates
two such blocks:

* :class:`LinearV2E` maps node signals to a (node, hyperedge) pair using the
  normalized incidence aggregation :math:`x_e = H^\\top x / |e|` together with a
  global term, each refined by positional MLPs keyed on the hyperedge order;
* :class:`LinearE2V` maps the (node, hyperedge) pair back to nodes via
  :math:`x = (x_v + H x_e) / (1 + \\deg)` plus a global term.

Everything is computed from the binary node-hyperedge incidence matrix
:math:`H` (TopoBench's ``incidence_hyperedges``) and the derived hyperedge sizes
and node degrees; no coordinates or precomputed positional features are needed.

In TopoBench the original input linear and MLP classifier head are provided by
the feature encoder and readout, so this backbone consumes pre-encoded node
features and returns node embeddings at the hidden width.
"""

import math

import torch
from torch import nn


class InnerMLP(nn.Module):
    """Plain feed-forward MLP used inside every EHNN block.

    Mirrors ``mlp.py``'s ``MLP``: a stack of linear layers with ReLU and dropout
    between them (no internal normalization in the forward pass).

    Parameters
    ----------
    dim_in : int
        Input dimension.
    dim_out : int
        Output dimension.
    dim_hidden : int
        Hidden dimension for intermediate layers.
    n_layers : int
        Number of linear layers (``>= 1``).
    dropout : float
        Dropout probability between layers.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        assert n_layers > 0, "InnerMLP requires at least one layer."
        self.n_layers = n_layers
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(nn.Linear(dim_in, dim_out))
        else:
            for i in range(n_layers):
                self.layers.append(
                    nn.Linear(
                        dim_hidden if i > 0 else dim_in,
                        dim_hidden if i < n_layers - 1 else dim_out,
                    )
                )

    def reset_parameters(self) -> None:
        """Reset the linear layers."""
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Transformed features.
        """
        for i in range(self.n_layers - 1):
            x = self.dropout(self.act(self.layers[i](x)))
        return self.layers[-1](x)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding indexed by an integer position.

    Parameters
    ----------
    dim : int
        Encoding dimension.
    max_pos : int
        Number of distinct positions (e.g. ``max_order + 1``).
    """

    def __init__(self, dim: int, max_pos: int):
        super().__init__()
        self.max_pos = max_pos
        position = torch.arange(max_pos).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(max_pos, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Look up the encoding for a batch of integer positions.

        Parameters
        ----------
        idx : torch.Tensor
            Long tensor of positions.

        Returns
        -------
        torch.Tensor
            Positional encodings.
        """
        return self.pe[idx]


class PositionalMLP(nn.Module):
    """Hypernetwork that maps an integer position to a feature vector.

    Used to produce order-conditioned hyperedge biases.

    Parameters
    ----------
    dim_out : int
        Output dimension.
    max_pos : int
        Number of distinct positions.
    dim_pe : int
        Positional-encoding dimension.
    dim_hidden : int
        Hidden dimension of the hypernetwork MLP.
    n_layers : int
        Number of layers in the hypernetwork MLP.
    dropout : float
        Dropout probability inside the hypernetwork MLP.
    """

    def __init__(
        self,
        dim_out: int,
        max_pos: int,
        dim_pe: int,
        dim_hidden: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.max_pos = max_pos
        self.pe = PositionalEncoding(dim_pe, max_pos)
        self.input = nn.Linear(dim_pe, dim_hidden)
        self.mlp = InnerMLP(dim_hidden, dim_out, dim_hidden, n_layers, dropout)

    def reset_parameters(self) -> None:
        """Reset the hypernetwork parameters."""
        self.input.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Map positions to feature vectors.

        Parameters
        ----------
        idx : torch.Tensor
            Long tensor of positions.

        Returns
        -------
        torch.Tensor
            Feature vectors.
        """
        return self.mlp(self.input(self.pe(idx)))


class BiasE(nn.Module):
    """Order-conditioned hyperedge bias (and a fixed node bias).

    Parameters
    ----------
    dim_out : int
        Output dimension.
    max_l : int
        Maximum hyperedge order.
    pe_dim : int
        Positional-encoding dimension of the bias hypernetwork.
    hyper_dim : int
        Hidden width of the bias hypernetwork.
    hyper_layers : int
        Number of layers in the bias hypernetwork MLP.
    hyper_dropout : float
        Dropout inside the bias hypernetwork.
    """

    def __init__(
        self,
        dim_out: int,
        max_l: int,
        pe_dim: int,
        hyper_dim: int,
        hyper_layers: int,
        hyper_dropout: float,
    ):
        super().__init__()
        self.dim_out = dim_out
        self.b = PositionalMLP(
            dim_out, max_l + 1, pe_dim, hyper_dim, hyper_layers, hyper_dropout
        )

    def reset_parameters(self) -> None:
        """Reset the bias hypernetwork."""
        self.b.reset_parameters()

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        edge_orders: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add order-conditioned bias to hyperedges, fixed bias to nodes.

        Parameters
        ----------
        x : tuple of torch.Tensor
            ``(x_v, x_e)`` node and hyperedge features.
        edge_orders : torch.Tensor
            Long tensor of hyperedge orders ``[|E|]``.

        Returns
        -------
        tuple of torch.Tensor
            Biased ``(x_v, x_e)``.
        """
        x_v, x_e = x
        b_e = self.b(edge_orders)
        b_1 = self.b(torch.ones(1, dtype=torch.long, device=x_v.device)).view(
            1, self.dim_out
        )
        return x_v + b_1, x_e + b_e


class BiasV(nn.Module):
    """Learnable node bias.

    Parameters
    ----------
    dim_out : int
        Output dimension.
    """

    def __init__(self, dim_out: int):
        super().__init__()
        self.dim_out = dim_out
        self.b = nn.Parameter(torch.empty(1, dim_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the node bias."""
        stdv = 1.0 / math.sqrt(self.dim_out)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the node bias.

        Parameters
        ----------
        x : torch.Tensor
            Node features.

        Returns
        -------
        torch.Tensor
            Biased node features.
        """
        return x + self.b


class LinearV2E(nn.Module):
    """Equivariant node-to-(node, hyperedge) block.

    Parameters
    ----------
    dim : int
        Shared input/inner/output dimension.
    max_l : int
        Maximum hyperedge order.
    pe_dim : int
        Positional-encoding dimension of the bias hypernetwork.
    hyper_dim : int
        Hidden width of the hypernetworks.
    hyper_layers : int
        Number of layers in each hypernetwork MLP.
    hyper_dropout : float
        Dropout inside the hypernetworks.
    """

    def __init__(
        self,
        dim: int,
        max_l: int,
        pe_dim: int,
        hyper_dim: int,
        hyper_layers: int,
        hyper_dropout: float,
    ):
        super().__init__()
        self.dim = dim
        self.max_l = max_l
        self.mlp1 = InnerMLP(dim, dim, hyper_dim, hyper_layers, hyper_dropout)
        self.mlp2 = InnerMLP(
            dim * 2, dim, hyper_dim, hyper_layers, hyper_dropout
        )
        self.mlp3 = InnerMLP(
            dim * 2, dim, hyper_dim, hyper_layers, hyper_dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.pe2 = PositionalEncoding(dim, 2)
        self.pe3 = PositionalEncoding(dim, max_l + 1)
        self.b = BiasE(
            dim, max_l, pe_dim, hyper_dim, hyper_layers, hyper_dropout
        )

    def reset_parameters(self) -> None:
        """Reset all submodules."""
        for m in (
            self.mlp1,
            self.mlp2,
            self.mlp3,
            self.norm1,
            self.norm2,
            self.norm3,
            self.b,
        ):
            m.reset_parameters()

    def forward(
        self, x: torch.Tensor, cache: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate node features into node and hyperedge representations.

        Parameters
        ----------
        x : torch.Tensor
            Node features ``[N, dim]``.
        cache : dict
            Operators ``incidence`` (sparse ``[N, |E|]``), ``edge_orders``
            ``[|E|]`` and ``prefix_normalizer`` ``[|E|]``.

        Returns
        -------
        tuple of torch.Tensor
            ``(x_v, x_e)`` node and hyperedge features.
        """
        incidence = cache["incidence"]
        edge_orders = cache["edge_orders"]
        prefix_normalizer = cache["prefix_normalizer"]

        x = x + self.mlp1(self.norm1(x))
        x0 = x.mean(dim=0, keepdim=True)
        x1_v = x
        x1_e = torch.sparse.mm(incidence.t(), x) / prefix_normalizer[:, None]

        pe2_s0 = self.pe2(
            torch.zeros(1, dtype=torch.long, device=x.device)
        ).view(1, self.dim)
        pe2_s1 = self.pe2(
            torch.ones(1, dtype=torch.long, device=x.device)
        ).view(1, self.dim)
        x0 = x0 + self.mlp2(torch.cat((self.norm2(x0), pe2_s0), dim=-1))
        x1_v = x1_v + self.mlp2(
            torch.cat((self.norm2(x1_v), pe2_s1.expand(x1_v.shape)), dim=-1)
        )
        x1_e = x1_e + self.mlp2(
            torch.cat((self.norm2(x1_e), pe2_s1.expand(x1_e.shape)), dim=-1)
        )
        x_v = x0 + x1_v
        x_e = x0 + x1_e

        pe3_l1 = self.pe3(
            torch.ones(1, dtype=torch.long, device=x.device)
        ).view(1, self.dim)
        pe3_l = self.pe3(edge_orders)
        x_v = x_v + self.mlp3(
            torch.cat((self.norm3(x_v), pe3_l1.expand(x_v.shape)), dim=-1)
        )
        x_e = x_e + self.mlp3(torch.cat((self.norm3(x_e), pe3_l), dim=-1))
        return self.b((x_v, x_e), edge_orders)


class LinearE2V(nn.Module):
    """Equivariant (node, hyperedge)-to-node block.

    Parameters
    ----------
    dim : int
        Shared input/inner/output dimension.
    max_k : int
        Maximum hyperedge order.
    hyper_dim : int
        Hidden width of the hypernetworks.
    hyper_layers : int
        Number of layers in each hypernetwork MLP.
    hyper_dropout : float
        Dropout inside the hypernetworks.
    """

    def __init__(
        self,
        dim: int,
        max_k: int,
        hyper_dim: int,
        hyper_layers: int,
        hyper_dropout: float,
    ):
        super().__init__()
        self.dim = dim
        self.max_k = max_k
        self.mlp1 = InnerMLP(
            dim * 2, dim, hyper_dim, hyper_layers, hyper_dropout
        )
        self.mlp2 = InnerMLP(
            dim * 2, dim, hyper_dim, hyper_layers, hyper_dropout
        )
        self.mlp3 = InnerMLP(dim, dim, hyper_dim, hyper_layers, hyper_dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.pe1 = PositionalEncoding(dim, max_k + 1)
        self.pe2 = PositionalEncoding(dim, 2)
        self.b = BiasV(dim)

    def reset_parameters(self) -> None:
        """Reset all submodules."""
        for m in (
            self.mlp1,
            self.mlp2,
            self.mlp3,
            self.norm1,
            self.norm2,
            self.norm3,
            self.b,
        ):
            m.reset_parameters()

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
        cache: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate node and hyperedge features back into node features.

        Parameters
        ----------
        x : tuple of torch.Tensor
            ``(x_v, x_e)`` node and hyperedge features.
        cache : dict
            Operators ``incidence`` (sparse ``[N, |E|]``), ``edge_orders``
            ``[|E|]`` and ``suffix_normalizer`` (node degrees) ``[N]``.

        Returns
        -------
        torch.Tensor
            Node features ``[N, dim]``.
        """
        incidence = cache["incidence"]
        edge_orders = cache["edge_orders"]
        suffix_normalizer = cache["suffix_normalizer"]
        x_v, x_e = x

        pe1_k1 = self.pe1(
            torch.ones(1, dtype=torch.long, device=x_v.device)
        ).view(1, self.dim)
        pe1_k = self.pe1(edge_orders)
        x_v = x_v + self.mlp1(
            torch.cat((self.norm1(x_v), pe1_k1.expand(x_v.shape)), dim=-1)
        )
        x_e = x_e + self.mlp1(torch.cat((self.norm1(x_e), pe1_k), dim=-1))

        x0 = torch.cat((x_v, x_e)).mean(dim=0, keepdim=True)
        x1 = (x_v + torch.sparse.mm(incidence, x_e)) / (
            1 + suffix_normalizer[:, None]
        )

        pe2_s0 = self.pe2(
            torch.zeros(1, dtype=torch.long, device=x_v.device)
        ).view(1, self.dim)
        pe2_s1 = self.pe2(
            torch.ones(1, dtype=torch.long, device=x_v.device)
        ).view(1, self.dim)
        x0 = x0 + self.mlp2(torch.cat((self.norm2(x0), pe2_s0), dim=-1))
        x1 = x1 + self.mlp2(
            torch.cat((self.norm2(x1), pe2_s1.expand(x1.shape)), dim=-1)
        )
        x = x0 + x1
        x = x + self.mlp3(self.norm3(x))
        return self.b(x)


class EHNN(nn.Module):
    """Equivariant Hypergraph Neural Network backbone (linear variant).

    Builds the incidence-derived operator cache, then applies the V2E and E2V
    equivariant blocks. The returned node embeddings are mapped to task logits by
    the TopoBench readout.

    Parameters
    ----------
    num_features : int
        Dimension of the (encoded) input node features.
    hidden_channels : int, optional
        Shared hidden width of the two blocks and of the returned embedding
        (default: 64).
    max_edge_order : int, optional
        Upper bound on hyperedge order used to size the order hypernetworks;
        larger hyperedges are clamped to this value (default: 100).
    dropout : float, optional
        Dropout applied to the input and between the blocks (default: 0.5).
    pe_dim : int or None, optional
        Positional-encoding dimension of the bias hypernetwork; defaults to
        ``hidden_channels`` when ``None`` (default: None).
    hyper_dim : int or None, optional
        Hidden width of the hypernetworks; defaults to ``hidden_channels`` when
        ``None`` (default: None).
    hyper_layers : int, optional
        Number of layers in each hypernetwork MLP (default: 3).
    hyper_dropout : float, optional
        Dropout inside the hypernetworks (default: 0.0).
    input_dropout : float, optional
        Dropout applied to the input features (default: 0.0).
    """

    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 64,
        max_edge_order: int = 100,
        dropout: float = 0.5,
        pe_dim: int | None = None,
        hyper_dim: int | None = None,
        hyper_layers: int = 3,
        hyper_dropout: float = 0.0,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        assert max_edge_order >= 1, (
            "max_edge_order must be a positive integer."
        )
        dim = hidden_channels
        pe_dim = pe_dim if pe_dim is not None else dim
        hyper_dim = hyper_dim if hyper_dim is not None else dim
        self.max_edge_order = max_edge_order

        self.input_dropout = nn.Dropout(input_dropout)
        self.input = nn.Linear(num_features, dim)
        self.layer1 = LinearV2E(
            dim, max_edge_order, pe_dim, hyper_dim, hyper_layers, hyper_dropout
        )
        self.layer2 = LinearE2V(
            dim, max_edge_order, hyper_dim, hyper_layers, hyper_dropout
        )
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        """Reset all learnable parameters of the backbone."""
        self.input.reset_parameters()
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def _build_cache(self, incidence: torch.Tensor) -> dict[str, torch.Tensor]:
        """Derive the EHNN operator cache from a hypergraph incidence matrix.

        Parameters
        ----------
        incidence : torch.Tensor
            Node-hyperedge incidence ``[N, |E|]`` (sparse or dense).

        Returns
        -------
        dict
            Cache with ``incidence`` (sparse, coalesced, unsigned),
            ``edge_orders`` (clamped hyperedge sizes), ``prefix_normalizer``
            (hyperedge sizes) and ``suffix_normalizer`` (node degrees).
        """
        if not incidence.is_sparse:
            incidence = incidence.to_sparse_coo()
        incidence = incidence.coalesce()
        incidence = torch.sparse_coo_tensor(
            incidence.indices(),
            incidence.values().abs(),
            size=incidence.shape,
        ).coalesce()

        edge_sizes = torch.sparse.sum(incidence, dim=0).to_dense()
        node_degrees = torch.sparse.sum(incidence, dim=1).to_dense()
        edge_orders = (
            edge_sizes.round().long().clamp(min=1, max=self.max_edge_order)
        )
        prefix_normalizer = edge_sizes.clone()
        prefix_normalizer[prefix_normalizer == 0] = 1e-5
        suffix_normalizer = node_degrees.clone()
        suffix_normalizer[suffix_normalizer == 0] = 1e-5
        return {
            "incidence": incidence,
            "edge_orders": edge_orders,
            "prefix_normalizer": prefix_normalizer,
            "suffix_normalizer": suffix_normalizer,
        }

    def forward(
        self,
        x: torch.Tensor,
        incidence: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Encoded node features ``[N, num_features]``.
        incidence : torch.Tensor
            Node-hyperedge incidence matrix ``[N, |E|]``.

        Returns
        -------
        tuple
            ``(x_0, None)`` node embeddings and the (unused) hyperedge slot,
            matching the hypergraph wrapper's ``(nodes, hyperedges)`` contract.
        """
        cache = self._build_cache(incidence)
        x = self.input(self.input_dropout(x))
        x_v, x_e = self.layer1(x, cache)
        x_v, x_e = self.dropout(self.act(x_v)), self.dropout(self.act(x_e))
        x = self.layer2((x_v, x_e), cache)
        return x, None
