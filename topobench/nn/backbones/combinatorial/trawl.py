"""TRAWL: topological random-walk sequence backbone.

TRAWL encodes random walks over the cells of a lifted complex with
configurable sequence layers (Mamba, SISA, hybrid, transformer, GRU, MLP or
any Hydra-instantiated module). This module holds the learnable components;
``TRAWLWrapper`` samples the walks and handles batching.
"""

import copy

import numpy as np
import torch
from torch import nn

from topobench.nn.backbones.combinatorial.trawl_utils.layers import (
    NeighborhoodFusion,
    make_layer,
)
from topobench.nn.backbones.combinatorial.trawl_utils.sampling import upload


class TRAWL(nn.Module):
    """Encode random walks over lifted cells with configurable sequence layers.

    Input widths left as None are inferred by ``initialize``.

    Parameters
    ----------
    hidden_dim : int, optional
        Width of walk states (default: 128).
    max_rank : int, optional
        Highest cell rank (default: 2).
    in_channels : list of int, optional
        Feature width per rank; inferred if None (default: None).
    pe_dim : int, optional
        Positional encoding width; inferred if None (default: None).
    layers : list, optional
        Explicit ``make_layer`` configs; overrides ``architecture`` and
        ``depth`` (default: None).
    architecture : str, optional
        ``"mamba"``, ``"sisa"``, ``"hybrid"`` (alternating), ``"gru"``,
        ``"transformer"`` or ``"mlp"`` (default: "hybrid").
    depth : int, optional
        Number of preset layers (default: 5).
    layer_options : dict, optional
        Options per layer kind for presets (default: None).
    walk_scope : str, optional
        ``"union"`` walks all neighborhoods jointly, ``"separate"`` walks
        each one (default: "union").
    num_neighborhoods : int, optional
        Number of relations built by the transform (default: 2).
    encoder_sharing : str, optional
        ``"shared"`` or ``"independent"`` encoders for separate walks
        (default: "shared").
    fusion : str, optional
        ``NeighborhoodFusion`` mode for separate walks (default: "mean").
    pooling : str, optional
        Time pooling: ``"mean"``, ``"max"`` or ``"mean_max"``
        (default: "mean_max").
    rank_embedding : bool, optional
        Add a learned rank embedding (default: True).
    num_colors : int, optional
        Number of color embeddings; 0 disables them (default: 0).
    move_embedding : bool, optional
        Embed rank moves (down, same, up) along walks (default: True).
    dropout : float, optional
        Dropout on walk inputs (default: 0.0).
    checkpoint_layers : bool, optional
        Activation checkpointing during training (default: False).
    occurrence_pooling : str, optional
        Pooling of a cell's walk occurrences: ``"mean"`` or
        ``"attention"`` (default: "mean").
    graph_readout : str, optional
        Graph embedding from ``"walks"`` or ``"cells"`` (default: "walks").
    """

    def __init__(
        self,
        hidden_dim=128,
        max_rank=2,
        in_channels=None,
        pe_dim=None,
        layers=None,
        architecture="hybrid",
        depth=5,
        layer_options=None,
        walk_scope="union",
        num_neighborhoods=2,
        encoder_sharing="shared",
        fusion="mean",
        pooling="mean_max",
        rank_embedding=True,
        num_colors=0,
        move_embedding=True,
        dropout=0.0,
        checkpoint_layers=False,
        occurrence_pooling="mean",
        graph_readout="walks",
    ):
        super().__init__()
        choices = {
            "walk_scope": (walk_scope, {"union", "separate"}),
            "encoder_sharing": (encoder_sharing, {"shared", "independent"}),
            "pooling": (pooling, {"mean", "max", "mean_max"}),
            "occurrence_pooling": (occurrence_pooling, {"mean", "attention"}),
            "graph_readout": (graph_readout, {"walks", "cells"}),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        if depth < 1 or num_neighborhoods < 1:
            raise ValueError("Depth and neighborhood count must be positive")
        self.hidden_dim, self.max_rank = hidden_dim, max_rank
        self.walk_scope, self.pooling = walk_scope, pooling
        self.num_neighborhoods = num_neighborhoods
        self.checkpoint_layers = checkpoint_layers
        self.occurrence_pooling = occurrence_pooling
        self.graph_readout = graph_readout
        self.occurrence_attention = (
            nn.Linear(hidden_dim, 1)
            if occurrence_pooling == "attention"
            else None
        )
        self.features = nn.ModuleList(
            [
                nn.Linear(in_channels[r], hidden_dim)
                if in_channels is not None
                else nn.LazyLinear(hidden_dim)
                for r in range(max_rank + 1)
            ]
        )
        self.position = (
            nn.LazyLinear(hidden_dim, bias=False)
            if pe_dim is None
            else nn.Linear(pe_dim, hidden_dim, bias=False)
        )
        self.rank = (
            nn.Embedding(max_rank + 1, hidden_dim) if rank_embedding else None
        )
        self.color = (
            nn.Embedding(num_colors, hidden_dim) if num_colors else None
        )
        self.move = nn.Embedding(3, hidden_dim) if move_embedding else None
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        options = dict(layer_options or {})
        if layers is None:
            presets = {"mamba", "sisa", "hybrid", "gru", "transformer", "mlp"}
            if architecture not in presets:
                raise ValueError(
                    "Custom architecture requires an explicit layer list"
                )
            kinds = [
                ("mamba" if i % 2 == 0 else "sisa")
                if architecture == "hybrid"
                else architecture
                for i in range(depth)
            ]
            layers = [
                {"kind": kind, **dict(options.get(kind, {}))} for kind in kinds
            ]
        if not layers:
            raise ValueError("Provide at least one sequence layer")
        encoder = nn.ModuleList(
            [make_layer(config, hidden_dim) for config in layers]
        )
        count = (
            num_neighborhoods
            if walk_scope == "separate" and encoder_sharing == "independent"
            else 1
        )
        self.encoders = nn.ModuleList(
            [encoder] + [copy.deepcopy(encoder) for _ in range(count - 1)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim * (2 if pooling == "mean_max" else 1)
        self.fusion = NeighborhoodFusion(
            self.output_dim, num_neighborhoods, fusion
        )

    def initialize(self, signal_widths, pe_width):
        """Materialize lazily sized input projections.

        Parameters
        ----------
        signal_widths : list of int
            Input feature width of each rank.
        pe_width : int
            Width of the positional encodings.
        """
        with torch.no_grad():
            for projection, width in zip(
                self.features, signal_widths, strict=True
            ):
                if isinstance(projection, nn.LazyLinear):
                    projection(
                        torch.zeros(1, width, device=projection.weight.device)
                    )
            if isinstance(self.position, nn.LazyLinear):
                self.position(
                    torch.zeros(
                        1, pe_width, device=self.position.weight.device
                    )
                )

    def compilable_layers(self):
        """Return the sequence layers that ``torch.compile`` may compile.

        Returns
        -------
        list of torch.nn.Module
            Every layer of every encoder stack.
        """
        return [layer for stack in self.encoders for layer in stack]

    def pool(self, x):
        """Pool walk states over time.

        Parameters
        ----------
        x : torch.Tensor
            States of shape ``[walk, time, hidden]``.

        Returns
        -------
        torch.Tensor
            Pooled embeddings of shape ``[walk, output_dim]``.
        """
        if self.pooling == "mean":
            return x.mean(dim=1)
        if self.pooling == "max":
            return x.amax(dim=1)
        return torch.cat((x.amax(dim=1), x.mean(dim=1)), dim=-1)

    def encode(self, x, paths, ranks, relation):
        """Encode sampled walks with the sequence layers.

        Parameters
        ----------
        x : torch.Tensor
            Cell states of shape ``[cells, hidden]``.
        paths : torch.Tensor
            Walk cell indices of shape ``[walk, time]``.
        ranks : torch.Tensor
            Rank of each cell, shape ``[cells]``.
        relation : int
            Neighborhood group index selecting the encoder.

        Returns
        -------
        torch.Tensor
            Encoded walk states of shape ``[walk, time, hidden]``.
        """
        tokens = x[paths]
        if self.move is not None:
            path_ranks = ranks[paths]
            movement = torch.zeros_like(path_ranks)
            movement[:, 1:] = torch.sign(
                path_ranks[:, 1:] - path_ranks[:, :-1]
            )
            tokens = tokens + self.move(movement + 1)
        tokens = self.dropout(self.input_norm(tokens))
        encoder = self.encoders[0 if len(self.encoders) == 1 else relation]
        for layer in encoder:
            if self.checkpoint_layers and self.training:
                from torch.utils.checkpoint import checkpoint

                tokens = checkpoint(layer, tokens, use_reentrant=False)
            else:
                tokens = layer(tokens)
        return self.output_norm(tokens)

    def forward(self, signals, pe, ranks, walk_groups, colors=None):
        """Encode the sampled walks of one complex.

        Parameters
        ----------
        signals : list of torch.Tensor
            Raw cell features per rank, each of shape ``[cells_r, width_r]``.
        pe : torch.Tensor
            Positional encodings of all cells, shape ``[cells, pe_dim]``.
        ranks : torch.Tensor
            Rank of each cell, shape ``[cells]``.
        walk_groups : list of list of tuple
            One entry per walk group (one for ``"union"``, one per
            neighborhood for ``"separate"``), each a list of
            ``(view, paths)`` pairs with ``paths`` of shape ``[walk, time]``.
        colors : torch.Tensor, optional
            Color ID of each cell; required when color embeddings are used
            (default: None).

        Returns
        -------
        dict
            ``contextual`` cell states ``[cells, hidden]``, the pooled
            ``graph_embedding`` ``[output_dim]`` and ``walks``, a list of
            ``(view, embeddings)`` pairs with embeddings of shape
            ``[walk, output_dim]``.
        """
        projected = [
            self.features[rank](raw)
            if len(raw)
            else pe.new_empty(0, self.hidden_dim)
            for rank, raw in enumerate(signals)
        ]
        features = torch.cat(projected)
        total = len(features)
        x = features + self.position(pe)
        if self.rank is not None:
            x = x + self.rank(ranks)
        if self.color is not None:
            x = x + self.color(colors)
        fallback = self.pool(features[None])[0]
        # Accumulate occurrences in full precision under autocast. Embedding
        # additions can promote encoded states independently of projections.
        accumulation_dtype = (
            torch.float32
            if features.dtype in {torch.float16, torch.bfloat16}
            else features.dtype
        )
        contextual = torch.zeros_like(features, dtype=accumulation_dtype)
        visits = contextual.new_zeros(total, 1)
        occurrence_values, occurrence_ids = [], []
        per_group, availability, walks = [], [], []
        for group_id, sampled in enumerate(walk_groups):
            pooled_views = []
            for view, paths in sampled:
                encoded = self.encode(x, paths, ranks, group_id)
                pooled = self.pool(encoded)
                pooled_views.append(pooled.mean(dim=0))
                walks.append((view, pooled))
                contextual.index_add_(
                    0,
                    paths.flatten(),
                    encoded.flatten(0, 1).to(contextual),
                )
                if self.occurrence_attention is not None:
                    occurrence_values.append(encoded.flatten(0, 1))
                    occurrence_ids.append(paths.flatten())
                visits.index_add_(
                    0,
                    paths.flatten(),
                    visits.new_ones(paths.numel(), 1),
                )
            availability.append(bool(pooled_views))
            per_group.append(
                torch.stack(pooled_views).mean(dim=0)
                if pooled_views
                else fallback * 0
            )
        if self.walk_scope == "union":
            graph_embedding = per_group[0] if availability[0] else fallback
        else:
            graph_embedding = self.fusion(
                torch.stack(per_group),
                upload(np.array(availability), x.device).bool(),
                fallback,
            )
        contextual = torch.where(
            visits > 0, contextual / visits.clamp_min(1), features
        )
        if occurrence_values:
            from torch_geometric.utils import softmax

            values, ids = (
                torch.cat(occurrence_values),
                torch.cat(occurrence_ids),
            )
            weights = softmax(
                self.occurrence_attention(values), ids, num_nodes=total
            )
            attended = torch.zeros_like(contextual).index_add(
                0, ids, (values * weights).to(contextual)
            )
            contextual = torch.where(visits > 0, attended, features)
        if self.graph_readout == "cells":
            graph_embedding = self.pool(contextual[None])[0]
        return {
            "contextual": contextual,
            "graph_embedding": graph_embedding,
            "walks": walks,
        }
