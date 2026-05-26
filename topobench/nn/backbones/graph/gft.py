"""GFT backbone adapted for TopoBench graph tasks.

This module implements the encoder core of "GFT: Graph Foundation Model with
Transferable Tree Vocabulary" for supervised TopoBench graph tasks.  The
official GFT pretrains a message-passing encoder and a transferable
computation-tree vocabulary with reconstruction losses, then reuses that
vocabulary for task adaptation.  TopoBench challenge runs do not ship the
pretrained vocabulary, LLM/text features, reconstruction heads, or prototype
task heads, so this backbone keeps the dependency-free core:

* a SAGE/GCN/GIN/GAT message-passing encoder modeled after the reference
  encoder choices;
* a structural computation-tree descriptor built by repeated neighborhood
  aggregation of root structural statistics;
* a cosine vector-quantized tree vocabulary fallback that is learned with the
  supervised task instead of loaded from GFT pretraining.

The forward pass returns node embeddings of shape ``[num_nodes, out_channels]``
and is compatible with :class:`topobench.nn.wrappers.GNNWrapper`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.utils import degree, scatter, to_undirected


class _TreeVocabulary(nn.Module):
    """Small vector-quantized vocabulary for computation-tree tokens."""

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        code_dim: int,
        num_heads: int,
        use_cosine_sim: bool = True,
        learnable_codebook: bool = True,
    ) -> None:
        super().__init__()
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if code_dim <= 0:
            raise ValueError("code_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")

        self.dim = dim
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.num_heads = num_heads
        self.use_cosine_sim = use_cosine_sim

        codebook_input_dim = code_dim * num_heads
        self.project_in = (
            nn.Linear(dim, codebook_input_dim)
            if codebook_input_dim != dim
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_input_dim, dim)
            if codebook_input_dim != dim
            else nn.Identity()
        )

        codebook = torch.empty(num_heads, codebook_size, code_dim)
        nn.init.xavier_uniform_(codebook)
        if learnable_codebook:
            self.codebook = nn.Parameter(codebook)
        else:
            self.register_buffer("codebook", codebook)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize tree queries into vocabulary codes.

        Parameters
        ----------
        x : torch.Tensor
            Tree-query embeddings with shape ``[num_nodes, dim]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Quantized embeddings, selected token ids, and a commitment loss
            value exposed for diagnostics.
        """
        if x.size(0) == 0:
            token_ids = x.new_empty((0, self.num_heads), dtype=torch.long)
            return x, token_ids, x.sum() * 0

        projected = self.project_in(x)
        projected = projected.view(x.size(0), self.num_heads, self.code_dim)

        codebook = self.codebook
        query = projected
        if self.use_cosine_sim:
            query = F.normalize(projected, p=2, dim=-1)
            codebook = F.normalize(codebook, p=2, dim=-1)
            scores = torch.einsum("nhd,hkd->nhk", query, codebook)
        else:
            scores = -(
                projected.unsqueeze(2) - codebook.unsqueeze(0)
            ).square().sum(dim=-1)

        token_ids = scores.argmax(dim=-1)
        expanded_codebook = codebook.unsqueeze(0).expand(
            x.size(0),
            -1,
            -1,
            -1,
        )
        gather_index = token_ids[..., None, None].expand(
            -1,
            -1,
            1,
            self.code_dim,
        )
        tokens = expanded_codebook.gather(2, gather_index).squeeze(2)
        commitment_loss = F.mse_loss(projected, tokens.detach())

        if self.training:
            tokens = tokens + projected - projected.detach()

        tokens = tokens.reshape(x.size(0), self.num_heads * self.code_dim)
        return self.project_out(tokens), token_ids, commitment_loss


class GFT(nn.Module):
    """TopoBench GFT core encoder.

    This is not a full reproduction of the official GFT training system.  It
    omits cross-domain pretraining, tree/feature reconstruction losses, LLM
    textual feature construction, and prototype task heads.  In their place it
    provides an in-model, dependency-free tree vocabulary fallback that can be
    optimized by the supervised TopoBench objective.

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden encoder dimension.
    out_channels : int, optional
        Output node embedding dimension. Defaults to ``hidden_channels``.
    num_layers : int, optional
        Number of message-passing layers. Defaults to 2.
    backbone : str, optional
        Local message-passing backbone: ``"sage"``, ``"gcn"``, ``"gin"``, or
        ``"gat"``. Defaults to ``"sage"``, matching the reference defaults.
    normalize : str, optional
        Normalization type: ``"none"``, ``"batch"``, or ``"layer"``.
    dropout : float, optional
        Dropout applied between encoder layers and before output projection.
    activation : str, optional
        Activation name: ``"relu"``, ``"gelu"``, ``"elu"``, or
        ``"leaky_relu"``.
    tree_depth : int, optional
        Number of neighborhood aggregation steps used to summarize rooted
        computation trees.
    codebook_size : int, optional
        Number of vocabulary entries per codebook head.
    codebook_heads : int, optional
        Number of independent vocabulary heads.
    code_dim : int, optional
        Dimension of each code. Defaults to ``hidden_channels``.
    use_cosine_sim : bool, optional
        Use cosine similarity for nearest-code lookup, as in the reference
        implementation.
    learnable_codebook : bool, optional
        If true, the fallback vocabulary is a trainable parameter. If false,
        it remains a fixed randomly initialized buffer.
    input_dim : int, optional
        Alias for ``in_channels`` for compatibility with other TopoBench
        graph encoders.
    hidden_dim : int, optional
        Alias for ``hidden_channels`` for compatibility with other TopoBench
        graph encoders.
    """

    def __init__(
        self,
        in_channels: int | None = None,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        num_layers: int = 2,
        backbone: str = "sage",
        normalize: str | None = "none",
        dropout: float = 0.15,
        activation: str = "relu",
        tree_depth: int = 2,
        codebook_size: int = 128,
        codebook_heads: int = 4,
        code_dim: int | None = None,
        use_cosine_sim: bool = True,
        learnable_codebook: bool = True,
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
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = hidden_channels
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if tree_depth < 0:
            raise ValueError("tree_depth must be non-negative")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.backbone = backbone
        self.normalize = "none" if normalize is None else normalize
        self.dropout_rate = dropout
        self.tree_depth = tree_depth
        self.codebook_size = codebook_size
        self.codebook_heads = codebook_heads
        self.code_dim = hidden_channels if code_dim is None else code_dim

        self.activation = self._make_activation(activation)
        self.dropout = nn.Dropout(dropout)

        dims = [in_channels, *([hidden_channels] * num_layers)]
        self.convs = nn.ModuleList(
            [
                self._make_conv(dims[i], dims[i + 1], backbone)
                for i in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [self._make_norm(self.normalize, hidden_channels) for _ in dims[1:]]
        )

        tree_descriptor_dim = 4 * (tree_depth + 1)
        self.tree_encoder = nn.Sequential(
            nn.Linear(tree_descriptor_dim, hidden_channels),
            self._make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.tree_query = nn.Linear(hidden_channels * 2, hidden_channels)
        self.vocabulary = _TreeVocabulary(
            dim=hidden_channels,
            codebook_size=codebook_size,
            code_dim=self.code_dim,
            num_heads=codebook_heads,
            use_cosine_sim=use_cosine_sim,
            learnable_codebook=learnable_codebook,
        )
        self.output_proj = nn.Linear(hidden_channels * 3, out_channels)
        self.output_norm = nn.LayerNorm(out_channels)

        self.last_tree_token_ids: torch.Tensor | None = None
        self.last_commitment_loss: torch.Tensor | None = None

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "elu":
            return nn.ELU()
        if name == "leaky_relu":
            return nn.LeakyReLU()
        raise ValueError(
            "Unsupported activation. Expected one of: relu, gelu, elu, "
            "leaky_relu"
        )

    @staticmethod
    def _make_norm(name: str, channels: int) -> nn.Module:
        name = name.lower()
        if name == "none":
            return nn.Identity()
        if name in {"batch", "batch_norm"}:
            return nn.BatchNorm1d(channels)
        if name in {"layer", "layer_norm"}:
            return nn.LayerNorm(channels)
        raise ValueError(
            "Unsupported normalize. Expected one of: none, batch, layer"
        )

    @staticmethod
    def _make_conv(
        in_channels: int,
        out_channels: int,
        backbone: str,
    ) -> nn.Module:
        backbone = backbone.lower()
        if backbone == "sage":
            return SAGEConv(in_channels, out_channels)
        if backbone == "gcn":
            return GCNConv(in_channels, out_channels)
        if backbone == "gin":
            return GINConv(
                nn.Sequential(
                    nn.Linear(in_channels, out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels, out_channels),
                )
            )
        if backbone == "gat":
            return GATConv(in_channels, out_channels, heads=1, concat=False)
        raise ValueError(
            "Unsupported backbone. Expected one of: sage, gcn, gin, gat"
        )

    @staticmethod
    def _coerce_edge_weight(
        edge_weight: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if edge_weight is not None:
            return edge_weight
        if edge_attr is None:
            return None
        if edge_attr.dim() == 1:
            return edge_attr
        if edge_attr.dim() == 2 and edge_attr.size(-1) == 1:
            return edge_attr.view(-1)
        return None

    @staticmethod
    def _apply_conv(
        conv: nn.Module,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(conv, GCNConv):
            return conv(x, edge_index, edge_weight=edge_weight)
        return conv(x, edge_index)

    def _tree_descriptors(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Build rooted computation-tree descriptors from local structure."""
        num_nodes = x.size(0)
        if num_nodes == 0:
            return x.new_empty((0, 4 * (self.tree_depth + 1)))

        tree_edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        source, target = tree_edge_index[0], tree_edge_index[1]

        deg = degree(target, num_nodes=num_nodes, dtype=x.dtype).view(-1, 1)
        feat_mean = x.mean(dim=-1, keepdim=True)
        feat_mean = feat_mean.sign() * torch.log1p(feat_mean.abs())
        feat_norm = torch.log1p(torch.linalg.vector_norm(x, dim=-1)).view(
            -1,
            1,
        )
        state = torch.cat(
            [
                torch.log1p(deg),
                feat_mean,
                feat_norm,
                torch.ones_like(deg),
            ],
            dim=-1,
        )

        descriptors = [state]
        for _ in range(self.tree_depth):
            if tree_edge_index.numel() == 0:
                state = torch.zeros_like(state)
            else:
                state = scatter(
                    state[source],
                    target,
                    dim=0,
                    dim_size=num_nodes,
                    reduce="mean",
                )
            descriptors.append(state)

        return torch.cat(descriptors, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        return_aux: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return node embeddings with shape ``[num_nodes, out_channels]``."""
        del batch, kwargs
        edge_weight = self._coerce_edge_weight(edge_weight, edge_attr)

        z = x
        for index, conv in enumerate(self.convs):
            z = self._apply_conv(conv, z, edge_index, edge_weight)
            z = self.norms[index](z)
            if index < self.num_layers - 1:
                z = self.dropout(self.activation(z))

        tree_descriptors = self._tree_descriptors(x, edge_index)
        tree_repr = self.tree_encoder(tree_descriptors)
        tree_query = self.tree_query(torch.cat([z, tree_repr], dim=-1))
        tree_tokens, token_ids, commitment_loss = self.vocabulary(tree_query)

        self.last_tree_token_ids = token_ids.detach()
        self.last_commitment_loss = commitment_loss.detach()

        fused = torch.cat([z, tree_repr, tree_tokens], dim=-1)
        out = self.output_norm(self.output_proj(self.dropout(fused)))

        if not return_aux:
            return out

        aux = {
            "tree_token_ids": token_ids,
            "commitment_loss": commitment_loss,
        }
        return out, aux
