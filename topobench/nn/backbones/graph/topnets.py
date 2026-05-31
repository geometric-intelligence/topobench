"""TopNets graph route operator with learned-filtration PH features.

This module adapts the challenge-relevant graph-classification route from the
reference TopNets repository. The original graph models combine a GCN/GIN
state evolution with a TOGL/RePHINE-style topological layer: node embeddings
parameterize graph filtrations, persistence pairs are transformed by learned
coordinate functions, and the resulting topological features are fused back
with message-passing features.

The reference code calls compiled ``ph_cpu``/``rephine_mt`` extensions. This
TopoBench implementation keeps the learned standard-filtration interface while
replacing compiled persistence with portable PyTorch proxy pairs from the
learned filtration values.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.utils import scatter


class _TriangleTransform(nn.Module):
    """Triangle coordinate function from TOGL."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.t_param = nn.Parameter(torch.randn(output_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x[:, 1, None] - torch.abs(self.t_param - x[:, 0, None]))


class _GaussianTransform(nn.Module):
    """Gaussian coordinate function from TOGL."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.t_param = nn.Parameter(torch.randn(output_dim) * 0.1)
        self.sigma = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma.abs().clamp_min(1e-6)
        return torch.exp(
            -((x[:, :, None] - self.t_param).pow(2).sum(dim=1))
            / (2 * sigma.pow(2))
        )


class _LineTransform(nn.Module):
    """Linear coordinate function from TOGL."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _RationalHatTransform(nn.Module):
    """Rational hat coordinate function used for persistence barcodes."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.c_param = nn.Parameter(torch.randn(1, output_dim) * 0.1)
        self.r_param = nn.Parameter(torch.randn(1, output_dim) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        distance = torch.norm(x[:, :, None] - self.c_param, p=1, dim=1)
        radius_gap = torch.abs(self.r_param.abs() - distance)
        return (1 / (1 + distance)) - (1 / (1 + radius_gap))


class _GraphConvBlock(nn.Module):
    """GCN/GIN block matching the graph layers used by TopNets."""

    def __init__(
        self,
        gnn_type: str,
        in_channels: int,
        out_channels: int,
        dropout: float,
        activation: bool,
    ) -> None:
        super().__init__()
        self.gnn_type = gnn_type.lower()
        self.activation = nn.ReLU() if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        if self.gnn_type == "gcn":
            self.conv = GCNConv(in_channels, out_channels)
        elif self.gnn_type == "gin":
            gin_mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            self.conv = GINConv(gin_mlp)
        else:
            msg = "gnn_type must be either 'gcn' or 'gin'"
            raise ValueError(msg)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one graph-convolution block."""
        if self.gnn_type == "gcn":
            x = self.conv(x, edge_index, edge_weight=edge_weight)
        else:
            x = self.conv(x, edge_index)
        return self.dropout(self.activation(x))


class _TopologicalFiltrationLayer(nn.Module):
    """Learned-filtration layer used inside TopNets."""

    def __init__(
        self,
        channels: int,
        num_filtrations: int = 8,
        filtration_hidden: int = 16,
        coord_fun_count: int = 3,
        use_dim1: bool = True,
        sigmoid_filtrations: bool = True,
    ) -> None:
        super().__init__()
        final_activation = nn.Sigmoid() if sigmoid_filtrations else nn.Identity()
        self.filtrations = nn.Sequential(
            nn.Linear(channels, filtration_hidden),
            nn.ReLU(),
            nn.Linear(filtration_hidden, num_filtrations),
            final_activation,
        )

        self.coord_modules = nn.ModuleList(
            [
                _TriangleTransform(coord_fun_count),
                _GaussianTransform(coord_fun_count),
                _LineTransform(coord_fun_count),
                _RationalHatTransform(coord_fun_count),
            ]
        )
        self.num_filtrations = num_filtrations
        self.coord_dim = coord_fun_count * len(self.coord_modules)
        self.topological_dim = self.num_filtrations * self.coord_dim
        self.use_dim1 = use_dim1

        self.out = nn.Linear(self.topological_dim, channels)
        self.bn = nn.BatchNorm1d(channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse learned-filtration topological features with node features."""
        num_graphs = _num_graphs(batch)
        filtrations = self.filtrations(x)
        persistence0, persistence1 = self._compute_persistence(
            filtrations=filtrations,
            edge_index=edge_index,
            batch=batch,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

        node_topology = self._coordinate_activations(persistence0)
        update = self.out(node_topology)
        if update.shape[0] > 1:
            update = self.bn(update)
        x = x + update

        if self.use_dim1:
            graph_topology = self._collapse_dim1(
                persistence1=persistence1,
                edge_batch=edge_batch,
                num_graphs=num_graphs,
            )
        else:
            graph_topology = scatter(
                node_topology,
                batch,
                dim=0,
                reduce="mean",
                dim_size=num_graphs,
            )
        return x, graph_topology

    def _coordinate_activations(
        self, persistence: torch.Tensor
    ) -> torch.Tensor:
        """Apply TOGL coordinate functions to persistence pairs."""
        num_filtrations, num_items, _ = persistence.shape
        if num_items == 0:
            return persistence.new_zeros((0, self.topological_dim))

        flat_pairs = persistence.reshape(num_filtrations * num_items, 2)
        flat_activations = torch.cat(
            [module(flat_pairs) for module in self.coord_modules],
            dim=1,
        )
        return (
            flat_activations.view(
                num_filtrations,
                num_items,
                self.coord_dim,
            )
            .transpose(0, 1)
            .reshape(num_items, self.topological_dim)
        )

    def _collapse_dim1(
        self,
        persistence1: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """Pool 1-dimensional persistence coordinates to graph features."""
        if edge_batch.numel() == 0:
            return persistence1.new_zeros(num_graphs, self.topological_dim)

        edge_topology = self._coordinate_activations(persistence1)
        mask = (persistence1 != 0).any(dim=2).any(dim=0)
        if not mask.any():
            return edge_topology.new_zeros(num_graphs, self.topological_dim)

        return scatter(
            edge_topology[mask],
            edge_batch[mask],
            dim=0,
            reduce="sum",
            dim_size=num_graphs,
        )

    def _compute_persistence(
        self,
        filtrations: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute batched proxy H0 and H1 persistence pairs."""
        num_nodes = filtrations.shape[0]
        graph_max = scatter(
            filtrations,
            batch,
            dim=0,
            reduce="max",
            dim_size=num_graphs,
        )

        if edge_index.numel() == 0:
            persistence0 = torch.stack(
                [filtrations.t(), filtrations.t()],
                dim=2,
            )
            persistence1 = filtrations.new_zeros(
                (self.num_filtrations, 0, 2)
            )
            return persistence0, persistence1

        row, col = edge_index
        edge_values = self._edge_filtration_values(filtrations, edge_index)

        incident_nodes = torch.cat([row, col], dim=0)
        incident_values = torch.cat([edge_values, edge_values], dim=0)
        min_incident_death = scatter(
            incident_values,
            incident_nodes,
            dim=0,
            reduce="min",
            dim_size=num_nodes,
        )

        has_incident_edge = torch.zeros(
            num_nodes,
            dtype=torch.bool,
            device=filtrations.device,
        )
        has_incident_edge[incident_nodes] = True

        graph_edge_count = scatter(
            torch.ones(
                edge_batch.shape[0],
                dtype=filtrations.dtype,
                device=filtrations.device,
            ),
            edge_batch,
            dim=0,
            reduce="sum",
            dim_size=num_graphs,
        )
        graph_has_edges = graph_edge_count > 0

        death0 = torch.where(
            has_incident_edge[:, None],
            min_incident_death,
            graph_max[batch],
        )
        death0 = torch.where(
            graph_has_edges[batch, None],
            death0,
            filtrations,
        )
        persistence0 = torch.stack([filtrations.t(), death0.t()], dim=2)

        persistence1 = torch.stack(
            [edge_values.t(), graph_max[edge_batch].t()],
            dim=2,
        )
        return persistence0, persistence1

    @staticmethod
    def _edge_filtration_values(
        values: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        return torch.maximum(values[edge_index[0]], values[edge_index[1]])


class _TopNetsRouteOperator(nn.Module):
    """Continuous TopNets-style graph backbone for TopoBench.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden node feature dimension used by the GNN and topology layer.
    num_steps : int, optional
        Number of fixed integration steps for the continuous TopNets dynamics.
    gnn_type : str, optional
        Message-passing layer type, either ``"gcn"`` or ``"gin"``.
    num_filtrations : int, optional
        Number of learned filtration functions.
    filtration_hidden : int, optional
        Hidden dimension of the filtration MLP.
    coord_fun_count : int, optional
        Number of outputs per TOGL coordinate transform.
    dropout : float, optional
        Dropout applied in graph-convolution blocks.
    use_dim1 : bool, optional
        Whether to include cycle-based 1-dimensional topological summaries.
    sigmoid_filtrations : bool, optional
        Whether to constrain filtration values with a sigmoid.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_steps: int = 4,
        gnn_type: str = "gcn",
        num_filtrations: int = 8,
        filtration_hidden: int = 16,
        coord_fun_count: int = 3,
        dropout: float = 0.0,
        use_dim1: bool = True,
        sigmoid_filtrations: bool = True,
    ) -> None:
        super().__init__()
        if num_steps <= 0:
            msg = "num_steps must be positive"
            raise ValueError(msg)

        self.hidden_channels = hidden_channels
        self.out_channels = hidden_channels
        self.num_steps = num_steps
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.first_gnn = _GraphConvBlock(
            gnn_type=gnn_type,
            in_channels=hidden_channels + 1,
            out_channels=hidden_channels,
            dropout=dropout,
            activation=True,
        )
        self.topology = _TopologicalFiltrationLayer(
            channels=hidden_channels,
            num_filtrations=num_filtrations,
            filtration_hidden=filtration_hidden,
            coord_fun_count=coord_fun_count,
            use_dim1=use_dim1,
            sigmoid_filtrations=sigmoid_filtrations,
        )
        self.second_gnn = _GraphConvBlock(
            gnn_type=gnn_type,
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=dropout,
            activation=False,
        )
        self.node_readout = nn.Sequential(
            nn.Linear(hidden_channels, 2 * hidden_channels),
            nn.ReLU(),
            nn.Linear(2 * hidden_channels, hidden_channels),
        )
        self.topology_projection = nn.Linear(
            self.topology.topological_dim, hidden_channels
        )
        self.output_projection = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run fixed-step TopNets dynamics and return node embeddings."""
        del kwargs
        batch = _ensure_batch(batch, x.shape[0], x.device)
        unique_edges, edge_batch = _unique_undirected_edges(
            edge_index=edge_index,
            batch=batch,
            num_nodes=x.shape[0],
        )

        h = self.input_projection(x.float())
        graph_topology = []
        step_size = 1.0 / self.num_steps
        denom = max(self.num_steps - 1, 1)

        for step in range(self.num_steps):
            time = h.new_full((h.shape[0], 1), step / denom)
            rhs = self.first_gnn(
                torch.cat([time, h], dim=1),
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            rhs, topo = self.topology(
                rhs,
                edge_index=unique_edges,
                batch=batch,
                edge_batch=edge_batch,
            )
            rhs = self.second_gnn(
                rhs,
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            h = h + step_size * rhs
            graph_topology.append(topo)

        topo_embedding = torch.stack(graph_topology).mean(dim=0)
        topo_embedding = self.topology_projection(topo_embedding)
        node_embedding = self.node_readout(h)
        return self.output_projection(
            torch.cat([node_embedding, topo_embedding[batch]], dim=1)
        )


def _ensure_batch(
    batch: torch.Tensor | None, num_nodes: int, device: torch.device
) -> torch.Tensor:
    if batch is None:
        return torch.zeros(num_nodes, dtype=torch.long, device=device)
    return batch.to(device=device, dtype=torch.long)


def _num_graphs(batch: torch.Tensor) -> int:
    if batch.numel() == 0:
        return 1
    return int(batch.max().item()) + 1


def _unique_undirected_edges(
    edge_index: torch.Tensor,
    batch: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one undirected edge per graph edge for persistence."""
    if edge_index.numel() == 0:
        empty_edges = edge_index.new_empty((2, 0))
        empty_batch = batch.new_empty((0,))
        return empty_edges, empty_batch

    row, col = edge_index
    source = torch.minimum(row, col)
    target = torch.maximum(row, col)
    mask = (source != target) & (batch[source] == batch[target])

    if not mask.any():
        empty_edges = edge_index.new_empty((2, 0))
        empty_batch = batch.new_empty((0,))
        return empty_edges, empty_batch

    source = source[mask]
    target = target[mask]
    keys = source * num_nodes + target
    order = torch.argsort(keys)
    sorted_keys = keys[order]
    keep = torch.ones(
        sorted_keys.shape[0], dtype=torch.bool, device=edge_index.device
    )
    keep[1:] = sorted_keys[1:] != sorted_keys[:-1]
    unique_positions = order[keep]
    unique_edges = torch.stack(
        [source[unique_positions], target[unique_positions]], dim=0
    )
    return unique_edges, batch[unique_edges[0]]
