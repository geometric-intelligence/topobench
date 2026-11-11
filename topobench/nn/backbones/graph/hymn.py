"""Hybrid Marking Network (HyMN) graph backbone.

This module implements the centrality structural encodings, centrality-guided
node marking, and shared message passing proposed by Southern et al. [1].  The
implementation follows Algorithm 1 and Equations (4) and (45)--(48) of the
paper, and the authors' reference implementation [2].

References
----------
[1] J. Southern et al., "Balancing Efficiency and Expressiveness: Subgraph
    GNNs with Walk-Based Centrality," ICML 2025,
    https://proceedings.mlr.press/v267/southern25a.html.
[2] J. Southern et al., HyMN reference implementation, revision
    ``adde55268307ff69527375757ec31a146d59ccae``,
    https://github.com/jks17/HyMN.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
from torch import nn
from torch_geometric.nn import GINConv

# The reference implementation computes CSEs once in the dataset transform.
# TopoBench constructs a fresh backbone for every benchmark run, so a bounded
# process cache preserves that preprocessing behavior across model instances.
_GLOBAL_STATISTICS_CACHE: OrderedDict[
    tuple[int, int, int, bytes], tuple[tuple[int, ...], torch.Tensor]
] = OrderedDict()


class HyMN(nn.Module):
    r"""Centrality-guided Hybrid Marking Network.

    For every input graph, HyMN computes the order-:math:`K` Centrality
    Structural Encoding (CSE)

    .. math::
        C^{\mathrm{CSE}}_{v,k} = (A^k)_{vv} / k!,

    selects the :math:`T` nodes with the largest truncated Subgraph Centrality,
    and processes an unmarked view plus the :math:`T` marked views with shared
    GIN layers.  These are Algorithm 1, Equation (4), and Equations (45)--(48)
    in [1].  As in the reference implementation, the CSE is batch-normalized,
    linearly encoded, and concatenated with the input node representation.

    Views are represented as a disjoint expanded graph during message passing,
    then reduced back to node-aligned embeddings before TopoBench's task
    readout.  This is algebraically the same node-wise view aggregation used by
    ``MeanAveraging``/``SumAveraging`` in [2], while satisfying TopoBench's
    backbone API for both node- and graph-level tasks.

    Parameters
    ----------
    in_channels : int
        Number of input node features after TopoBench's feature encoder.
    hidden_channels : int
        Width of the shared marked GIN and output node embeddings.
    num_layers : int, optional
        Number of shared GIN message-passing layers.
    num_samples : int, optional
        Total number of views.  Following the augmented policy in Equation
        (45), this includes one unmarked view and ``num_samples - 1`` marked
        views.
    cse_steps : int, optional
        Maximum walk length :math:`K` in Equation (4).
    cse_channels : int, optional
        Width of the linearly encoded CSE.
    dropout : float, optional
        Dropout after each marked GIN layer.
    train_eps : bool, optional
        Whether GIN's epsilon parameters are trainable.
    batch_norm : bool, optional
        Whether the two-layer GIN MLPs use Batch Normalization, as in [2].
    residual : bool, optional
        Whether to use the layer-wise residual connections reported in [1].
    use_centrality_encoding : bool, optional
        Whether to concatenate CSEs to node features.  Disabling this exposes
        the paper's ``HyMN no CSE`` ablation without changing node selection.
    sample_aggregation : str, optional
        ``"mean"`` or ``"sum"`` aggregation over the augmented bag, both used
        in the reference configurations.
    cache_size : int, optional
        Maximum number of per-graph CSE computations retained by a model.
    global_cache_size : int, optional
        Maximum number of CSE computations shared across model instances in a
        process.  This mirrors the reference implementation's dataset-level
        preprocessing when TopoBench creates a model per benchmark run.
    **kwargs : dict, optional
        Extra configuration values accepted for TopoBench compatibility.

    Notes
    -----
    Node features are expected to be ordered graph-by-graph when ``batch`` is
    supplied, as they are in :class:`torch_geometric.data.Batch`.

    The authors use marked GINE layers when molecular bond attributes are
    available and marked GIN layers for unattributed counting graphs.  The
    standard TopoBench :class:`~topobench.nn.wrappers.GNNWrapper` does not
    expose bond attributes, so this backbone follows the latter, edge-less
    reference path (``counting_substructures/conv.py`` in [2]).  It contains
    no GraphUniverse-specific features, labels, or task logic.

    Exact centrality ties are resolved by stable node-index order.  This is a
    deterministic instance of the arbitrary tie-breaking policy explicitly
    allowed by [1]; as with the authors' ``torch.argsort`` implementation,
    tied selections need not be permutation equivariant.

    Both statistics caches store only the deterministic output of Algorithm
    1.  They reproduce the authors' dataset-level preprocessing without
    changing the mathematical forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        num_samples: int = 3,
        cse_steps: int = 16,
        cse_channels: int = 16,
        dropout: float = 0.0,
        train_eps: bool = True,
        batch_norm: bool = True,
        residual: bool = True,
        use_centrality_encoding: bool = True,
        sample_aggregation: str = "mean",
        cache_size: int = 4096,
        global_cache_size: int = 65536,
        **kwargs,
    ) -> None:
        super().__init__()
        if in_channels < 1 or hidden_channels < 1:
            raise ValueError(
                "in_channels and hidden_channels must be positive"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if num_samples < 1:
            raise ValueError("num_samples must include the unmarked view")
        if cse_steps < 1:
            raise ValueError("cse_steps must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between zero and one")
        if sample_aggregation not in {"mean", "sum"}:
            raise ValueError("sample_aggregation must be mean or sum")
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")
        if global_cache_size < 0:
            raise ValueError("global_cache_size cannot be negative")
        if use_centrality_encoding and not 0 < cse_channels < hidden_channels:
            raise ValueError(
                "cse_channels must lie between zero and hidden_channels"
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.cse_steps = cse_steps
        self.cse_channels = cse_channels
        self.dropout = dropout
        self.train_eps = train_eps
        self.batch_norm = batch_norm
        self.residual = residual
        self.use_centrality_encoding = use_centrality_encoding
        self.sample_aggregation = sample_aggregation
        self.cache_size = cache_size
        self.global_cache_size = global_cache_size

        node_channels = (
            hidden_channels - cse_channels
            if use_centrality_encoding
            else hidden_channels
        )
        self.node_encoder = (
            nn.Identity()
            if in_channels == node_channels
            else nn.Linear(in_channels, node_channels)
        )
        if use_centrality_encoding:
            self.cse_norm = nn.BatchNorm1d(cse_steps)
            self.cse_encoder = nn.Linear(cse_steps, cse_channels)
        else:
            self.cse_norm = None
            self.cse_encoder = None

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels + 1, hidden_channels),
                nn.BatchNorm1d(hidden_channels)
                if batch_norm
                else nn.Identity(),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
        self.dropout_layer = nn.Dropout(dropout)
        self._statistics_cache: OrderedDict[
            tuple[int, bytes], tuple[tuple[int, ...], torch.Tensor]
        ] = OrderedDict()

    @staticmethod
    def _graph_key(
        num_nodes: int,
        local_edge_index: torch.Tensor,
    ) -> tuple[int, bytes]:
        """Return an exact, edge-order-independent key for the CSE cache.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        local_edge_index : torch.Tensor
            Local COO edge indices of shape ``[2, num_edges]``.

        Returns
        -------
        tuple[int, bytes]
            Node count and sorted linearized edge list.
        """
        if local_edge_index.numel() == 0:
            return num_nodes, b""
        linear_edges = local_edge_index[0].to(
            torch.int64
        ) * num_nodes + local_edge_index[1].to(torch.int64)
        edge_bytes = torch.sort(linear_edges.cpu()).values.numpy().tobytes()
        return num_nodes, edge_bytes

    @staticmethod
    def _compute_graph_statistics(
        num_nodes: int,
        local_edge_index: torch.Tensor,
        num_marked_views: int,
        cse_steps: int = 16,
    ) -> tuple[tuple[int, ...], torch.Tensor]:
        r"""Compute Algorithm 1's CSE and top-centrality marked nodes.

        The returned columns are :math:`\operatorname{diag}(A^k)/k!` for
        :math:`k=1,\ldots,K`, matching ``centrality_posenc`` in [2].  Their sum
        is the paper's truncated estimate of Subgraph Centrality.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        local_edge_index : torch.Tensor
            Local COO edge indices of shape ``[2, num_edges]``.
        num_marked_views : int
            Number :math:`T` of centrality-ranked nodes to mark.
        cse_steps : int, optional
            Maximum walk length :math:`K`.

        Returns
        -------
        tuple[tuple[int, ...], torch.Tensor]
            Selected local node indices and the ``[num_nodes, cse_steps]`` CSE.
        """
        if num_nodes < 0:
            raise ValueError("num_nodes cannot be negative")
        if num_marked_views < 0:
            raise ValueError("num_marked_views cannot be negative")

        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        if local_edge_index.numel():
            edges = local_edge_index.to(device="cpu", dtype=torch.long)
            adjacency.index_put_(
                (edges[0], edges[1]),
                torch.ones(edges.size(1), dtype=adjacency.dtype),
                accumulate=True,
            )

        encodings = []
        adjacency_power = adjacency
        for walk_length in range(1, cse_steps + 1):
            encodings.append(
                torch.diagonal(adjacency_power) / math.factorial(walk_length)
            )
            adjacency_power = adjacency_power @ adjacency
        cse = torch.stack(encodings, dim=-1)

        if num_nodes == 0:
            roots: tuple[int, ...] = ()
        else:
            order = torch.argsort(-cse.sum(dim=-1), stable=True)
            # The authors' implementation retains T views even when T > |V|,
            # filling the remaining marks with node zero.
            selected = torch.zeros(num_marked_views, dtype=torch.long)
            available = min(num_nodes, num_marked_views)
            selected[:available] = order[:available]
            roots = tuple(int(index) for index in selected)
        return roots, cse

    def _statistics_for_batch(
        self,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute or retrieve Algorithm 1 statistics for a PyG batch.

        Parameters
        ----------
        edge_index : torch.Tensor
            Batched COO edge indices.
        batch : torch.Tensor
            Graph assignment for every node.
        num_nodes : int
            Total number of nodes.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Marker matrix ``[num_nodes, num_samples]`` and CSE matrix
            ``[num_nodes, cse_steps]`` on ``edge_index.device``.
        """
        if batch.ndim != 1 or batch.numel() != num_nodes:
            raise ValueError("batch must assign every node to one graph")
        if batch.numel() and int(batch.min()) < 0:
            raise ValueError("batch graph indices cannot be negative")

        markers = torch.zeros(
            (num_nodes, self.num_samples),
            dtype=torch.float32,
            device=edge_index.device,
        )
        cse = torch.zeros(
            (num_nodes, self.cse_steps),
            dtype=torch.float32,
            device=edge_index.device,
        )
        num_graphs = int(batch.max()) + 1 if batch.numel() else 0
        graph_sizes = (
            torch.bincount(batch, minlength=num_graphs).cpu().tolist()
        )

        node_offset = 0
        for graph_index, graph_size in enumerate(graph_sizes):
            next_offset = node_offset + graph_size
            if graph_size == 0:
                node_offset = next_offset
                continue
            if not torch.all(batch[node_offset:next_offset] == graph_index):
                raise ValueError("batch nodes must be grouped by graph")

            edge_mask = batch[edge_index[0]] == graph_index
            local_edges = (edge_index[:, edge_mask] - node_offset).cpu()
            key = self._graph_key(graph_size, local_edges)
            cached = self._statistics_cache.get(key)
            if cached is None:
                global_key = (
                    self.cse_steps,
                    self.num_samples - 1,
                    *key,
                )
                cached = _GLOBAL_STATISTICS_CACHE.get(global_key)
                if cached is None:
                    cached = self._compute_graph_statistics(
                        graph_size,
                        local_edges,
                        self.num_samples - 1,
                        self.cse_steps,
                    )
                    if self.global_cache_size:
                        _GLOBAL_STATISTICS_CACHE[global_key] = cached
                        if (
                            len(_GLOBAL_STATISTICS_CACHE)
                            > self.global_cache_size
                        ):
                            _GLOBAL_STATISTICS_CACHE.popitem(last=False)
                elif self.global_cache_size:
                    _GLOBAL_STATISTICS_CACHE.move_to_end(global_key)
                if self.cache_size:
                    self._statistics_cache[key] = cached
                    if len(self._statistics_cache) > self.cache_size:
                        self._statistics_cache.popitem(last=False)
            elif self.cache_size:
                self._statistics_cache.move_to_end(key)

            roots, graph_cse = cached
            for sample_index, root in enumerate(roots, start=1):
                markers[node_offset + root, sample_index] = 1.0
            cse[node_offset:next_offset] = graph_cse.to(edge_index.device)
            node_offset = next_offset
        return markers, cse

    def _expand_views(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        markers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Materialize the augmented bag as one disjoint expanded graph.

        Parameters
        ----------
        x : torch.Tensor
            Node features shared by every view.
        edge_index : torch.Tensor
            Original batched COO edge indices.
        markers : torch.Tensor
            Marker matrix with one column per view.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Repeated features, view-offset edges, and flattened markers.
        """
        num_nodes = x.size(0)
        expanded_x = x.repeat(self.num_samples, 1)
        offsets = (
            torch.arange(self.num_samples, device=edge_index.device)
            * num_nodes
        )
        expanded_edges = edge_index.unsqueeze(0) + offsets[:, None, None]
        expanded_edges = expanded_edges.permute(1, 0, 2).reshape(2, -1)
        expanded_markers = markers.transpose(0, 1).reshape(-1, 1)
        return expanded_x, expanded_edges, expanded_markers

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode the unmarked graph and its top-centrality marked views.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            COO edge indices of shape ``[2, num_edges]``.
        batch : torch.Tensor, optional
            Graph assignment for each node.  A single graph is assumed when
            omitted.
        **kwargs : dict, optional
            Extra wrapper arguments, such as unused edge weights.

        Returns
        -------
        torch.Tensor
            Node-aligned view-aggregated embeddings of shape
            ``[num_nodes, hidden_channels]``.
        """
        if x.ndim != 2:
            raise ValueError("x must be a two-dimensional node feature matrix")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        with torch.no_grad():
            markers, cse = self._statistics_for_batch(
                edge_index,
                batch,
                x.size(0),
            )

        node_features = self.node_encoder(x)
        if self.use_centrality_encoding:
            cse = cse.to(dtype=x.dtype)
            cse_features = self.cse_encoder(self.cse_norm(cse))
            node_features = torch.cat((node_features, cse_features), dim=-1)

        hidden, expanded_edges, expanded_markers = self._expand_views(
            node_features,
            edge_index,
            markers.to(dtype=x.dtype),
        )
        for conv in self.convs:
            marked_hidden = torch.cat((hidden, expanded_markers), dim=-1)
            update = torch.relu(conv(marked_hidden, expanded_edges))
            update = self.dropout_layer(update)
            hidden = hidden + update if self.residual else update

        view_embeddings = hidden.reshape(
            self.num_samples,
            x.size(0),
            self.hidden_channels,
        )
        if self.sample_aggregation == "mean":
            return view_embeddings.mean(dim=0)
        return view_embeddings.sum(dim=0)
