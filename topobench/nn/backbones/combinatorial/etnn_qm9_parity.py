"""Protocol-matched ETNN core for the controlled QM9 reproduction.

This module isolates the model-side comparison between the TopoBench ETNN core
and the pinned NSAPH ``experiment_1`` implementation.  It consumes
:class:`~topobench.nn.backbones.combinatorial.etnn_qm9_adapter._QM9ParityBatch`
objects produced from canonical native QM9CC batches.  Consequently, the
wrapper does not lift molecules, recompute physical invariants, or alter
targets and split membership.

The architecture matches the selected native configuration:

* ranks 0 and 1 with 15- and 19-channel input features;
* relations ``0_0_2`` and ``1_1_2`` with five invariant channels;
* one shared affine-free invariant BatchNorm per relation;
* seven 128-channel gated ETNN layers;
* ``Linear-SiLU-Linear-SiLU`` relation-message MLPs;
* residual ``Linear-SiLU-Linear`` rank updates;
* rank-wise pre-pool MLPs, sum pooling, and a scalar post-pool readout.

The class is reproduction-specific.  It does not replace the coordinate-policy
backbone or change any standard GraphUniverse/QM9 model default.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from topobench.nn.backbones.combinatorial.etnn_coordinate_policy import (
    _ETNNMessagePassing,
    _make_mlp,
)
from topobench.nn.backbones.combinatorial.etnn_qm9_adapter import (
    _QM9ParityBatch,
)

_EXPERIMENT_1_RELATIONS = ("0_0_2", "1_1_2")
_EXPERIMENT_1_RELATION_RANK = {"0_0_2": 0, "1_1_2": 1}
_EXPERIMENT_1_VISIBLE_RANKS = (0, 1)
_EXPERIMENT_1_FEATURE_CHANNELS = {0: 15, 1: 19}
_EXPERIMENT_1_INVARIANT_CHANNELS = 5
_EXPERIMENT_1_HIDDEN_CHANNELS = 128
_EXPERIMENT_1_NUM_LAYERS = 7
_EXPERIMENT_1_NUM_PARAMETERS = 1_497_871


class ETNNQM9Parity(nn.Module):
    """Reproduce the pinned NSAPH ``experiment_1`` model with TopoBench code.

    Parameters
    ----------
    hidden_channels : int, optional
        Shared hidden width.  The controlled protocol requires 128.
    num_layers : int, optional
        Number of ETNN message/update layers.  The protocol requires seven.
    dropout : float, optional
        Feature dropout applied after each layer.  The protocol requires zero.
    normalize_invariants : bool, optional
        Whether to use one shared affine-free BatchNorm per relation.  The
        controlled protocol requires this to be enabled.

    Raises
    ------
    ValueError
        If a constructor option differs from the pinned ``experiment_1``
        architecture.
    RuntimeError
        If the assembled trainable-parameter count differs from the native
        model.
    """

    def __init__(
        self,
        hidden_channels: int = _EXPERIMENT_1_HIDDEN_CHANNELS,
        num_layers: int = _EXPERIMENT_1_NUM_LAYERS,
        dropout: float = 0.0,
        normalize_invariants: bool = True,
    ) -> None:
        super().__init__()
        _validate_experiment_1_architecture(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            normalize_invariants=normalize_invariants,
        )

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize_invariants = normalize_invariants

        # Keep native attribute names where possible.  This makes architecture
        # auditing and state-dict mapping explicit rather than heuristic.
        self.inv_normalizer = nn.ModuleDict(
            {
                relation: nn.BatchNorm1d(
                    _EXPERIMENT_1_INVARIANT_CHANNELS,
                    affine=False,
                )
                for relation in _EXPERIMENT_1_RELATIONS
            }
        )
        self.feature_embedding = nn.ModuleDict(
            {
                str(rank): nn.Sequential(
                    nn.Linear(
                        _EXPERIMENT_1_FEATURE_CHANNELS[rank],
                        hidden_channels,
                    )
                )
                for rank in _EXPERIMENT_1_VISIBLE_RANKS
            }
        )
        self.layers = nn.ModuleList(
            [
                _ETNNQM9ParityLayer(hidden_channels=hidden_channels)
                for _ in range(num_layers)
            ]
        )
        self.pre_pool = nn.ModuleDict(
            {
                str(rank): nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels),
                    nn.SiLU(),
                    nn.Linear(hidden_channels, hidden_channels),
                )
                for rank in _EXPERIMENT_1_VISIBLE_RANKS
            }
        )
        self.post_pool = nn.Sequential(
            nn.Linear(
                len(_EXPERIMENT_1_VISIBLE_RANKS) * hidden_channels,
                hidden_channels,
            ),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

        parameter_count = sum(
            parameter.numel() for parameter in self.parameters()
        )
        if parameter_count != _EXPERIMENT_1_NUM_PARAMETERS:
            raise RuntimeError(
                "ETNNQM9Parity architecture drifted from the pinned native "
                "parameter count: "
                f"{parameter_count} != {_EXPERIMENT_1_NUM_PARAMETERS}."
            )

    def forward(self, batch: _QM9ParityBatch) -> Tensor:
        """Predict molecular dipole magnitude for one adapted QM9CC batch.

        Parameters
        ----------
        batch : _QM9ParityBatch
            Canonical native tensors validated and copied by
            :func:`adapt_nsaph_qm9_batch`.

        Returns
        -------
        Tensor
            One scalar prediction per molecule, with shape ``[num_graphs]``.
            Target normalization and de-normalization belong to the controlled
            training harness, matching the native driver.
        """
        _validate_parity_batch_surface(batch)

        hidden = {
            rank: self.feature_embedding[str(rank)](batch.features[rank])
            for rank in _EXPERIMENT_1_VISIBLE_RANKS
        }
        invariants = {
            relation: self.inv_normalizer[relation](
                batch.raw_invariants[relation]
            )
            for relation in _EXPERIMENT_1_RELATIONS
        }

        # Physical positions are intentionally not updated in native QM9.  The
        # canonical static invariants are normalized once and reused by every
        # layer, exactly as in the pinned implementation.
        for layer in self.layers:
            hidden = layer(
                hidden=hidden,
                edge_index=batch.edge_index,
                invariants=invariants,
            )
            if self.dropout > 0:
                hidden = {
                    rank: F.dropout(
                        features,
                        p=self.dropout,
                        training=self.training,
                    )
                    for rank, features in hidden.items()
                }

        pre_pooled = {
            rank: self.pre_pool[str(rank)](hidden[rank])
            for rank in _EXPERIMENT_1_VISIBLE_RANKS
        }
        pooled = {
            rank: _global_add_pool(
                features=pre_pooled[rank],
                cell_batch=batch.cell_batch[rank],
                num_graphs=batch.num_graphs,
            )
            for rank in _EXPERIMENT_1_VISIBLE_RANKS
        }
        state = torch.cat(
            [pooled[rank] for rank in _EXPERIMENT_1_VISIBLE_RANKS],
            dim=-1,
        )
        return self.post_pool(state).squeeze(-1)


class _ETNNQM9ParityLayer(nn.Module):
    """One native-shaped gated ETNN message and residual-update layer.

    Parameters
    ----------
    hidden_channels : int
        Shared width of cell features, messages, and rank updates.
    """

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.message_passing = nn.ModuleDict(
            {
                relation: _ETNNMessagePassing(
                    hidden_channels=hidden_channels,
                    edge_channels=_EXPERIMENT_1_INVARIANT_CHANNELS,
                    dropout=0.0,
                    activation="silu",
                    use_batch_norm=False,
                    final_activation=True,
                )
                for relation in _EXPERIMENT_1_RELATIONS
            }
        )
        self.update = nn.ModuleDict(
            {
                str(rank): _make_mlp(
                    in_channels=2 * hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    dropout=0.0,
                    activation="silu",
                    use_batch_norm=False,
                )
                for rank in _EXPERIMENT_1_VISIBLE_RANKS
            }
        )

    def forward(
        self,
        hidden: dict[int, Tensor],
        edge_index: Mapping[str, Tensor],
        invariants: Mapping[str, Tensor],
    ) -> dict[int, Tensor]:
        """Apply relation messages followed by rank-wise residual updates.

        Parameters
        ----------
        hidden : dict[int, Tensor]
            Current rank-0 and rank-1 hidden features.
        edge_index : Mapping[str, Tensor]
            Canonical same-rank native relations.
        invariants : Mapping[str, Tensor]
            Shared normalized invariant channels aligned with each relation.

        Returns
        -------
        dict[int, Tensor]
            Updated hidden features for ranks 0 and 1.
        """
        messages: dict[int, Tensor] = {}
        for relation in _EXPERIMENT_1_RELATIONS:
            rank = _EXPERIMENT_1_RELATION_RANK[relation]
            messages[rank] = self.message_passing[relation](
                x_src=hidden[rank],
                x_dst=hidden[rank],
                edge_index=edge_index[relation],
                edge_attr=invariants[relation],
            )

        return {
            rank: hidden[rank]
            + self.update[str(rank)](
                torch.cat([hidden[rank], messages[rank]], dim=-1)
            )
            for rank in _EXPERIMENT_1_VISIBLE_RANKS
        }


def load_nsaph_experiment_1_state_dict(
    model: ETNNQM9Parity,
    native_state_dict: Mapping[str, Tensor],
) -> None:
    """Load pinned NSAPH parameters into the parity wrapper.

    The two implementations contain the same trainable tensors.  Their module
    paths differ only because TopoBench's shared MLP helper keeps a zero-rate
    ``Dropout`` module in each block and names the sigmoid gate ``edge_gate``.
    This function performs that deterministic key translation and then uses a
    strict state-dict load.

    Parameters
    ----------
    model : ETNNQM9Parity
        Destination parity model.
    native_state_dict : Mapping[str, Tensor]
        State dictionary from the pinned NSAPH ``experiment_1`` model.

    Raises
    ------
    KeyError
        If two native keys map to the same destination key.
    RuntimeError
        If the translated dictionary does not exactly cover the parity model.
    """
    translated: dict[str, Tensor] = {}
    for native_key, value in native_state_dict.items():
        target_key = _translate_native_state_key(native_key)
        if target_key in translated:
            raise KeyError(
                "Multiple NSAPH parameters mapped to TopoBench key "
                f"`{target_key}`."
            )
        translated[target_key] = value

    model.load_state_dict(translated, strict=True)


def _translate_native_state_key(native_key: str) -> str:
    """Translate one pinned NSAPH state path to the parity-module path.

    Parameters
    ----------
    native_key : str
        State-dictionary key emitted by the pinned NSAPH model.

    Returns
    -------
    str
        Corresponding state-dictionary key in :class:`ETNNQM9Parity`.
    """
    target_key = native_key.replace(".edge_inf_mlp.", ".edge_gate.")

    # Native lean=False MLPs place their second Linear at index 2.  The shared
    # TopoBench helper keeps a parameter-free Dropout at index 2, moving that
    # Linear to index 3.
    if ".message_mlp.2." in target_key:
        target_key = target_key.replace(
            ".message_mlp.2.",
            ".message_mlp.3.",
        )
    key_parts = target_key.split(".")
    if "update" in key_parts and key_parts[-2] == "2":
        key_parts[-2] = "3"
    return ".".join(key_parts)


def _global_add_pool(
    features: Tensor,
    cell_batch: Tensor,
    num_graphs: int,
) -> Tensor:
    """Sum cell features independently for every molecule in a batch.

    Parameters
    ----------
    features : Tensor
        Cell features with shape ``[num_cells, num_channels]``.
    cell_batch : Tensor
        Molecule index for every cell row.
    num_graphs : int
        Number of molecules in the batch.

    Returns
    -------
    Tensor
        Summed features with shape ``[num_graphs, num_channels]``.
    """
    pooled = features.new_zeros((num_graphs, features.shape[1]))
    pooled.index_add_(0, cell_batch, features)
    return pooled


def _validate_experiment_1_architecture(
    *,
    hidden_channels: int,
    num_layers: int,
    dropout: float,
    normalize_invariants: bool,
) -> None:
    """Reject architecture drift from the controlled native protocol.

    Parameters
    ----------
    hidden_channels : int
        Requested hidden feature width.
    num_layers : int
        Requested number of message/update layers.
    dropout : float
        Requested feature-dropout probability.
    normalize_invariants : bool
        Whether shared relation-wise invariant normalization is enabled.

    Raises
    ------
    ValueError
        If any requested value differs from pinned ``experiment_1``.
    """
    expected = {
        "hidden_channels": _EXPERIMENT_1_HIDDEN_CHANNELS,
        "num_layers": _EXPERIMENT_1_NUM_LAYERS,
        "dropout": 0.0,
        "normalize_invariants": True,
    }
    observed = {
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "dropout": dropout,
        "normalize_invariants": normalize_invariants,
    }
    mismatches = [
        f"{name}={observed[name]!r} (expected {value!r})"
        for name, value in expected.items()
        if observed[name] != value
    ]
    if mismatches:
        raise ValueError(
            "ETNNQM9Parity only supports the pinned experiment_1 "
            f"architecture: {', '.join(mismatches)}."
        )


def _validate_parity_batch_surface(batch: _QM9ParityBatch) -> None:
    """Defend the model boundary against incomplete adapter output.

    Parameters
    ----------
    batch : _QM9ParityBatch
        Validated adapter output supplied to the parity model.

    Raises
    ------
    ValueError
        If required ranks or relations are absent, unexpected entries are
        present, or the batch contains no molecules.
    """
    if set(batch.features) != set(_EXPERIMENT_1_VISIBLE_RANKS):
        raise ValueError(
            "ETNNQM9Parity requires rank-0 and rank-1 feature tensors."
        )
    if set(batch.cell_batch) != set(_EXPERIMENT_1_VISIBLE_RANKS):
        raise ValueError(
            "ETNNQM9Parity requires rank-0 and rank-1 graph assignments."
        )
    if set(batch.edge_index) != set(_EXPERIMENT_1_RELATIONS):
        raise ValueError(
            "ETNNQM9Parity requires exactly the 0_0_2 and 1_1_2 relations."
        )
    if set(batch.raw_invariants) != set(_EXPERIMENT_1_RELATIONS):
        raise ValueError(
            "ETNNQM9Parity requires one canonical invariant tensor per "
            "configured relation."
        )
    if batch.num_graphs < 1:
        raise ValueError("ETNNQM9Parity requires at least one molecule.")
