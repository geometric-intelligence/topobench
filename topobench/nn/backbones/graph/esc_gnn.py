"""GINE backbone driven by cached ESC structure."""

import torch
from torch import nn
from torch_geometric.nn import GINEConv
from torch_geometric.utils import scatter

from topobench.data.utils.esc import ESC_NUM_STRUCTURAL_CODES


def _reset_sequential(module: nn.Sequential) -> None:
    """Reset stateful children in a sequential block.

    Parameters
    ----------
    module : torch.nn.Sequential
        Block with learned children to reset.
    """
    for layer in module:
        reset_parameters = getattr(layer, "reset_parameters", None)
        if reset_parameters is not None:
            reset_parameters()


class ESCGNN(nn.Module):
    r"""Run GINE layers with cached ESC edge feats.

    Code entries get embedded, summed per directed edge, and reused by three
    GINE layers. Input plus layer states feed final projection. This matches
    challenge form of ESC-GNN Equation (4).

    Parameters
    ----------
    in_channels : int, optional
        Encoded node width. Must be ``64``.
    hidden_channels : int, optional
        Message width. Must be ``64``.
    num_layers : int, optional
        GINE layer count. Must be ``3``.
    num_structural_codes : int, optional
        Codebook size. Must be ``387``.
    structural_channels : int, optional
        Structural feat width. Must be ``64``.
    dropout : float, optional
        GINE MLP dropout. Must be ``0.0``.
    train_eps : bool, optional
        Learnable GINE epsilon flag. Must be ``True``.
    edge_dim : int, optional
        GINE edge width. Must be ``64``.

    Notes
    -----
    Raw ``edge_attr`` stays out. ESC histograms provide learned edge feats.
    Theorems 5.2 and 5.3 give existence results, not an exact-counting promise
    for this fixed model.

    References
    ----------
    Zuoyu Yan et al., "An Efficient Subgraph GNN with Provable Substructure
    Counting Power," KDD 2024, Equation (4), Section 5.1, and Theorems 5.2-5.3.

    Examples
    --------
    >>> model = ESCGNN().eval()
    >>> x = torch.zeros(2, 64)
    >>> edge_index = torch.tensor([[0, 1], [1, 0]])
    >>> code_id = torch.tensor([0, 0])
    >>> count = torch.ones(2, dtype=torch.float32)
    >>> nnz = torch.ones(2, dtype=torch.long)
    >>> model(x, edge_index, code_id, count, nnz).shape
    torch.Size([2, 64])
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_structural_codes: int = ESC_NUM_STRUCTURAL_CODES,
        structural_channels: int = 64,
        dropout: float = 0.0,
        train_eps: bool = True,
        edge_dim: int = 64,
    ) -> None:
        super().__init__()
        frozen = {
            "in_channels": (in_channels, 64),
            "hidden_channels": (hidden_channels, 64),
            "num_layers": (num_layers, 3),
            "num_structural_codes": (
                num_structural_codes,
                ESC_NUM_STRUCTURAL_CODES,
            ),
            "structural_channels": (structural_channels, 64),
            "dropout": (dropout, 0.0),
            "train_eps": (train_eps, True),
            "edge_dim": (edge_dim, 64),
        }
        for name, (value, expected) in frozen.items():
            assert value == expected, (
                f"ESCGNN requires {name}={expected!r}; received {value!r}"
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_structural_codes = num_structural_codes
        self.structural_channels = structural_channels

        self.structural_embedding = nn.Embedding(
            num_structural_codes, structural_channels
        )
        self.structural_mlp = nn.Sequential(
            nn.BatchNorm1d(structural_channels),
            nn.ReLU(),
            nn.Linear(structural_channels, edge_dim),
            nn.BatchNorm1d(edge_dim),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.Dropout(dropout),
                        nn.BatchNorm1d(hidden_channels),
                        nn.ReLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.Dropout(dropout),
                        nn.BatchNorm1d(hidden_channels),
                        nn.ReLU(),
                    ),
                    eps=0.0,
                    train_eps=train_eps,
                    edge_dim=edge_dim,
                    aggr="add",
                )
                for _ in range(num_layers)
            ]
        )
        self.jk_projection = nn.Sequential(
            nn.Linear((num_layers + 1) * hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        esc_code_id: torch.Tensor,
        esc_code_count: torch.Tensor,
        esc_nnz_per_edge: torch.Tensor,
    ) -> torch.Tensor:
        """Build node feats with cached structural edges.

        Parameters
        ----------
        x : torch.Tensor
            Encoded node feats, shape ``[N, 64]``.
        edge_index : torch.Tensor
            Directed edges, shape ``[2, E]``.
        esc_code_id : torch.Tensor
            Flat structural code IDs, shape ``[K]``.
        esc_code_count : torch.Tensor
            Positive code counts, shape ``[K]``.
        esc_nnz_per_edge : torch.Tensor
            Stored-code counts per edge, shape ``[E]``.

        Returns
        -------
        torch.Tensor
            Node feats, shape ``[N, 64]``.
        """
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("ESCGNN: x must be a two-dimensional tensor")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"ESCGNN: x must have {self.in_channels} feature channels"
            )
        if x.device != edge_index.device:
            raise ValueError("ESCGNN: x and edge_index must share a device")

        edge_structure = self._encode_structure(
            edge_index.size(1),
            esc_code_id,
            esc_code_count,
            esc_nnz_per_edge,
        )
        states = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_structure)
            states.append(x)
        return self.jk_projection(torch.cat(states, dim=-1))

    def _encode_structure(
        self,
        num_edges: int,
        esc_code_id: torch.Tensor,
        esc_code_count: torch.Tensor,
        esc_nnz_per_edge: torch.Tensor,
    ) -> torch.Tensor:
        """Turn sparse code rows into edge feats.

        Fixed output size keeps zero-code edges in message order.

        Parameters
        ----------
        num_edges : int
            Directed edge count.
        esc_code_id : torch.Tensor
            Flat structural code IDs, shape ``[K]``.
        esc_code_count : torch.Tensor
            Positive code counts, shape ``[K]``.
        esc_nnz_per_edge : torch.Tensor
            Stored-code counts per edge, shape ``[E]``.

        Returns
        -------
        torch.Tensor
            Structural edge feats, shape ``[E, 64]``.

        Examples
        --------
        >>> model = ESCGNN().eval()
        >>> ids = torch.tensor([0], dtype=torch.long)
        >>> counts = torch.tensor([2.0], dtype=torch.float32)
        >>> nnz = torch.tensor([1, 0], dtype=torch.long)
        >>> model._encode_structure(2, ids, counts, nnz).shape
        torch.Size([2, 64])
        """
        if num_edges == 0:
            return self.structural_embedding.weight.new_empty(
                (0, self.structural_channels)
            )

        edge_id = torch.repeat_interleave(
            torch.arange(num_edges, device=esc_nnz_per_edge.device),
            esc_nnz_per_edge,
        )
        code_vectors = self.structural_embedding(esc_code_id)
        weighted_codes = code_vectors * esc_code_count[:, None]
        edge_structure = scatter(
            weighted_codes,
            edge_id,
            dim=0,
            dim_size=num_edges,
            reduce="sum",
        )
        return self.structural_mlp(edge_structure)

    def reset_parameters(self) -> None:
        """Reset all learned ESC layers."""
        self.structural_embedding.reset_parameters()
        _reset_sequential(self.structural_mlp)
        for conv in self.convs:
            conv.reset_parameters()
        _reset_sequential(self.jk_projection)


__all__ = ["ESCGNN"]
