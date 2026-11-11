"""SGFormer backbone for graph representation learning.

This implementation adapts the SGFormer architecture for TopoBench by
returning node embeddings instead of classifier log-probabilities.

References
----------
Wu, Q., Zhao, W., Yang, C., Zhang, H., Nie, F., Jiang, H., Bian, Y.,
and Yan, J. "SGFormer: Simplifying and Empowering Transformers for
Large-Graph Representations." NeurIPS 2023.
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class _SGFormerAttention(nn.Module):
    """Linear-complexity global attention used by SGFormer.

    Parameters
    ----------
    channels : int
        Input feature dimension.
    heads : int, optional
        Number of attention heads.
    head_channels : int, optional
        Hidden dimension per attention head. Defaults to ``channels``.
    qkv_bias : bool, optional
        Whether query, key, and value projections use bias terms.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        head_channels: int | None = None,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_channels = head_channels or channels

        inner_channels = self.heads * self.head_channels
        self.q = nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.k = nn.Linear(channels, inner_channels, bias=qkv_bias)
        self.v = nn.Linear(channels, inner_channels, bias=qkv_bias)

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        self.q.reset_parameters()
        self.k.reset_parameters()
        self.v.reset_parameters()

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Run SGFormer attention over a dense graph batch.

        Parameters
        ----------
        x : torch.Tensor
            Dense node features with shape ``[batch_size, num_nodes, channels]``.
        mask : torch.Tensor, optional
            Boolean mask indicating valid nodes in each graph.

        Returns
        -------
        torch.Tensor
            Updated dense node features.
        """
        batch_size, num_nodes, _ = x.shape
        query, key, value = self.q(x), self.k(x), self.v(x)
        query, key, value = (
            tensor.view(
                batch_size,
                num_nodes,
                self.heads,
                self.head_channels,
            )
            for tensor in (query, key, value)
        )

        if mask is None:
            mask = x.new_ones((batch_size, num_nodes), dtype=torch.bool)

        valid_mask = mask[:, :, None, None]
        key = key.masked_fill(~valid_mask, 0.0)
        value = value.masked_fill(~valid_mask, 0.0)

        eps = torch.finfo(x.dtype).eps
        query = query / torch.linalg.norm(
            query, ord=2, dim=-1, keepdim=True
        ).clamp_min(eps)
        key = key / torch.linalg.norm(
            key, ord=2, dim=-1, keepdim=True
        ).clamp_min(eps)

        kv = torch.einsum("bnhc,bnhd->bhcd", key, value)
        numerator = torch.einsum("bnhc,bhcd->bnhd", query, kv)

        valid_counts = mask.sum(dim=1).clamp_min(1).view(batch_size, 1, 1, 1)
        numerator = numerator + valid_counts * value

        key_sum = key.sum(dim=1)
        denominator = torch.einsum("bnhc,bhc->bnh", query, key_sum)
        denominator = denominator.unsqueeze(-1) + valid_counts
        denominator = torch.where(
            denominator.abs() > eps,
            denominator,
            denominator.new_full(denominator.shape, eps),
        )

        return (numerator / denominator).mean(dim=2)


class _SGModule(nn.Module):
    """Global SGFormer attention branch.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden embedding dimension.
    num_layers : int, optional
        Number of global attention layers.
    num_heads : int, optional
        Number of attention heads.
    dropout : float, optional
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_norm = nn.LayerNorm(hidden_channels)
        self.attns = nn.ModuleList(
            [
                _SGFormerAttention(
                    hidden_channels,
                    heads=num_heads,
                    head_channels=hidden_channels,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_channels) for _ in range(num_layers)]
        )
        self.dropout = dropout

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        self.input_proj.reset_parameters()
        self.input_norm.reset_parameters()
        for attn in self.attns:
            attn.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x: Tensor, batch: Tensor | None = None) -> Tensor:
        """Run the global branch.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape ``[num_nodes, in_channels]``.
        batch : torch.Tensor, optional
            Graph assignment vector for each node.

        Returns
        -------
        torch.Tensor
            Node embeddings with shape ``[num_nodes, hidden_channels]``.
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        batch, indices = batch.sort(stable=True)
        reverse = torch.empty_like(indices)
        reverse[indices] = torch.arange(indices.numel(), device=indices.device)

        x = x[indices]
        x, mask = to_dense_batch(x, batch)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for attn, norm in zip(self.attns, self.norms, strict=True):
            residual = x
            x = attn(x, mask)
            x = (x + residual) / 2.0
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x[mask][reverse]


class _GraphModule(nn.Module):
    """Local GCN branch used by SGFormer.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden embedding dimension.
    num_layers : int, optional
        Number of graph convolution layers.
    dropout : float, optional
        Dropout probability.
    use_bn : bool, optional
        Whether to use ``BatchNorm1d`` instead of ``LayerNorm``.
    use_edge_weight : bool, optional
        Whether to pass edge weights to the local graph convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        use_bn: bool = True,
        use_edge_weight: bool = False,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_norm = (
            nn.BatchNorm1d(hidden_channels)
            if use_bn
            else nn.LayerNorm(hidden_channels)
        )
        self.convs = nn.ModuleList(
            [
                GCNConv(hidden_channels, hidden_channels)
                for _ in range(num_layers)
            ]
        )
        norm_cls = nn.BatchNorm1d if use_bn else nn.LayerNorm
        self.norms = nn.ModuleList(
            [norm_cls(hidden_channels) for _ in range(num_layers)]
        )
        self.dropout = dropout
        self.use_edge_weight = use_edge_weight

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        self.input_proj.reset_parameters()
        self.input_norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Run the local graph branch.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Edge indices with shape ``[2, num_edges]``.
        edge_weight : torch.Tensor, optional
            Optional scalar weights for each edge.

        Returns
        -------
        torch.Tensor
            Node embeddings with shape ``[num_nodes, hidden_channels]``.
        """
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, norm in zip(self.convs, self.norms, strict=True):
            residual = x
            if self.use_edge_weight:
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual

        return x


class SGFormer(nn.Module):
    """SGFormer backbone returning node embeddings for TopoBench readouts.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden dimension for both SGFormer branches.
    out_channels : int, optional
        Output embedding dimension. Defaults to ``hidden_channels``.
    trans_num_layers : int, optional
        Number of global attention layers.
    trans_num_heads : int, optional
        Number of heads in the global attention branch.
    trans_dropout : float, optional
        Dropout rate in the global attention branch.
    gnn_num_layers : int, optional
        Number of local graph convolution layers.
    gnn_dropout : float, optional
        Dropout rate in the local graph branch.
    graph_weight : float, optional
        Weight assigned to the local graph branch when ``aggregate="add"``.
    aggregate : {"add", "cat"}, optional
        How to combine global and local branches.
    gnn_use_bn : bool, optional
        Whether to use BatchNorm1d in the local branch. LayerNorm is used when
        false, which is robust to one-node training batches.
    gnn_use_edge_weight : bool, optional
        Whether to pass ``edge_weight`` to the local GCN branch. Disabled by
        default to match the reference SGFormer path.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int | None = None,
        trans_num_layers: int = 1,
        trans_num_heads: int = 1,
        trans_dropout: float = 0.5,
        gnn_num_layers: int = 2,
        gnn_dropout: float = 0.5,
        graph_weight: float = 0.5,
        aggregate: Literal["add", "cat"] = "add",
        gnn_use_bn: bool = True,
        gnn_use_edge_weight: bool = False,
    ) -> None:
        super().__init__()
        if aggregate not in {"add", "cat"}:
            raise ValueError(f"Invalid aggregate type: {aggregate}")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels or hidden_channels
        self.graph_weight = graph_weight
        self.aggregate = aggregate

        self.trans_conv = _SGModule(
            in_channels,
            hidden_channels,
            trans_num_layers,
            trans_num_heads,
            trans_dropout,
        )
        self.graph_conv = _GraphModule(
            in_channels,
            hidden_channels,
            gnn_num_layers,
            gnn_dropout,
            use_bn=gnn_use_bn,
            use_edge_weight=gnn_use_edge_weight,
        )

        fc_in_channels = (
            hidden_channels if aggregate == "add" else 2 * hidden_channels
        )
        self.fc = nn.Linear(fc_in_channels, self.out_channels)

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        self.trans_conv.reset_parameters()
        self.graph_conv.reset_parameters()
        self.fc.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        edge_weight: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Edge indices with shape ``[2, num_edges]``.
        batch : torch.Tensor, optional
            Graph assignment vector for each node.
        edge_weight : torch.Tensor, optional
            Optional edge weights for the GCN branch. Ignored unless
            ``gnn_use_edge_weight`` is enabled.
        **kwargs : dict
            Extra keyword arguments ignored for wrapper compatibility.

        Returns
        -------
        torch.Tensor
            Node embeddings with shape ``[num_nodes, out_channels]``.
        """
        del kwargs
        x_trans = self.trans_conv(x, batch)
        x_graph = self.graph_conv(x, edge_index, edge_weight=edge_weight)

        if self.aggregate == "add":
            x = (
                self.graph_weight * x_graph
                + (1.0 - self.graph_weight) * x_trans
            )
        else:
            x = torch.cat((x_trans, x_graph), dim=-1)

        return self.fc(x)
