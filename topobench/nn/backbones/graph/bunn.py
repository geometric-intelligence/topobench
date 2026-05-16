"""Bundle Neural Network backbone for graph data.

This module implements a compact BuNN-style encoder from Bamberger et al.,
"Bundle Neural Networks for message diffusion on graphs", arXiv:2405.15540.
The implementation follows the flat vector bundle formulation in the paper:
node-wise orthogonal maps synchronize features to a global frame, a learnable
channel update is applied, and a truncated Taylor heat-kernel approximation
diffuses the synchronized features over the graph before returning them to
local frames.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    """Return an activation module from a small supported set."""
    activations = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "identity": nn.Identity,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    if name not in activations:
        supported = ", ".join(sorted(activations))
        raise ValueError(
            f"Unsupported activation '{name}'. Use one of {supported}."
        )
    return activations[name]()


def _require_positive_int(name: str, value: int) -> int:
    """Validate positive integer hyperparameters before module construction."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _require_non_negative_int(name: str, value: int) -> int:
    """Validate non-negative integer hyperparameters."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _require_bool(name: str, value: bool) -> bool:
    """Validate boolean feature flags from configs and CLI overrides."""
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean.")
    return value


def _require_non_negative_float(name: str, value: float) -> float:
    """Validate finite non-negative scalar hyperparameters."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a numeric scalar.")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def _require_probability(name: str, value: float) -> float:
    """Validate probability-valued regularization hyperparameters."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a probability in [0, 1].")
    value = float(value)
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be a finite probability in [0, 1].")
    return value


def _require_node_feature_matrix(
    x: torch.Tensor, expected_channels: int, channel_name: str
) -> None:
    """Validate node-feature matrices before learnable projections."""
    if (
        not isinstance(x, torch.Tensor)
        or x.dim() != 2
        or x.shape[1] != expected_channels
    ):
        raise ValueError(f"x must have shape [num_nodes, {channel_name}].")


class BuNNLayer(nn.Module):
    r"""Single flat-bundle heat diffusion layer.

    The layer approximates the BuNN update described in Equations (1)--(4) and
    Algorithm 1 of Bamberger et al. (2024). For scalability and compatibility
    with arbitrary PyG mini-batches, it uses multiple two-dimensional flat
    bundles and the truncated Taylor approximation of the random-walk graph
    heat kernel.

    Parameters
    ----------
    hidden_channels : int
        Width of the node representation. It must be divisible by
        ``num_bundles * bundle_dim``.
    num_bundles : int, optional
        Number of independent flat vector bundles. Default is 8.
    bundle_dim : int, optional
        Dimension of each bundle. Only 2D bundles are currently supported,
        matching the direct angle parameterization used in the BuNN paper's
        experiments. Default is 2.
    t : float, optional
        Heat diffusion time. Larger values mix information over a larger graph
        scale. Default is 1.0.
    taylor_degree : int, optional
        Degree of the truncated Taylor approximation to the heat kernel.
        Default is 3.
    dropout : float, optional
        Dropout probability applied after the layer activation. Default is 0.0.
    act : str, optional
        Activation name. One of ``relu``, ``gelu``, ``elu``, ``tanh``, or
        ``identity``. Default is ``gelu``.
    angle_hidden_channels : int or None, optional
        Hidden width of the angle network. If ``None``, twice the number of
        bundles is used, matching the compact angle-network width described
        for the paper's LRGB experiments. Default is ``None``.
    include_reflections : bool, optional
        Whether to parameterize half of the 2D bundles as determinant -1
        orthogonal maps. Default is ``True``, following Appendix E.
    residual : bool, optional
        Whether to add a residual connection around the BuNN layer. Default is
        ``False``, matching the core Equations (1)--(4).
    norm : str or None, optional
        Optional normalization applied after the layer. Use ``layer_norm`` for
        ``torch.nn.LayerNorm``. Default is ``None``.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_bundles: int = 8,
        bundle_dim: int = 2,
        t: float = 1.0,
        taylor_degree: int = 3,
        dropout: float = 0.0,
        act: str = "gelu",
        angle_hidden_channels: int | None = None,
        include_reflections: bool = True,
        residual: bool = False,
        norm: str | None = None,
    ) -> None:
        super().__init__()

        hidden_channels = _require_positive_int(
            "hidden_channels", hidden_channels
        )
        num_bundles = _require_positive_int("num_bundles", num_bundles)
        bundle_dim = _require_positive_int("bundle_dim", bundle_dim)
        taylor_degree = _require_non_negative_int(
            "taylor_degree", taylor_degree
        )
        if angle_hidden_channels is not None:
            angle_hidden_channels = _require_positive_int(
                "angle_hidden_channels", angle_hidden_channels
            )
        include_reflections = _require_bool(
            "include_reflections", include_reflections
        )
        residual = _require_bool("residual", residual)
        diffusion_time = _require_non_negative_float("t", t)
        dropout = _require_probability("dropout", dropout)

        if bundle_dim != 2:
            raise ValueError("BuNNLayer currently supports bundle_dim=2 only.")
        if include_reflections and num_bundles % 2 != 0:
            raise ValueError(
                "num_bundles must be even when include_reflections=True."
            )

        bundle_width = num_bundles * bundle_dim
        if hidden_channels % bundle_width != 0:
            raise ValueError(
                "hidden_channels must be divisible by "
                "num_bundles * bundle_dim."
            )

        self.hidden_channels = hidden_channels
        self.num_bundles = num_bundles
        self.bundle_dim = bundle_dim
        self.channels_per_bundle = hidden_channels // bundle_width
        self.t = diffusion_time
        self.taylor_degree = taylor_degree

        self.include_reflections = include_reflections
        self.residual = residual
        self.norm_name = norm

        angle_hidden_channels = angle_hidden_channels or 2 * num_bundles
        self.angle_network = nn.Sequential(
            nn.Linear(hidden_channels, angle_hidden_channels),
            nn.GELU(),
            nn.Linear(angle_hidden_channels, num_bundles),
        )
        self.channel_mixer = nn.Linear(hidden_channels, hidden_channels)
        self.activation = _get_activation(act)
        self.dropout = nn.Dropout(dropout)
        if norm is None:
            self.norm = nn.Identity()
        elif norm == "layer_norm":
            self.norm = nn.LayerNorm(hidden_channels)
        else:
            raise ValueError("Unsupported norm. Use None or 'layer_norm'.")

    def _compute_bundle_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Compute one 2D orthogonal matrix per node and bundle."""
        angles = self.angle_network(x)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        maps = torch.stack((cos, -sin, sin, cos), dim=-1).reshape(
            x.shape[0], self.num_bundles, 2, 2
        )
        if not self.include_reflections:
            return maps

        reflection_start = self.num_bundles // 2
        reflection_maps = torch.stack(
            (
                cos[:, reflection_start:],
                sin[:, reflection_start:],
                sin[:, reflection_start:],
                -cos[:, reflection_start:],
            ),
            dim=-1,
        ).reshape(x.shape[0], self.num_bundles - reflection_start, 2, 2)
        return torch.cat((maps[:, :reflection_start], reflection_maps), dim=1)

    def _to_bundle_fields(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape node features into bundle and vector-field channels."""
        return x.reshape(
            x.shape[0],
            self.num_bundles,
            self.channels_per_bundle,
            self.bundle_dim,
        )

    def _from_bundle_fields(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten bundle and vector-field channels back to node features."""
        return x.reshape(x.shape[0], self.hidden_channels)

    @staticmethod
    def _random_walk_laplacian(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the random-walk graph Laplacian to node features.

        BuNN is derived for undirected graphs, so the COO edge list is
        symmetrized before computing random-walk neighbor averages. Duplicate
        reverse edges do not change the normalized average.
        """
        if x.dim() != 2:
            raise ValueError("x must have shape [num_nodes, num_features].")
        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must be a torch.long tensor.")

        source, target = edge_index
        if source.numel() > 0 and (
            (edge_index < 0).any() or edge_index.max() >= x.shape[0]
        ):
            raise ValueError("edge_index values must be in [0, num_nodes).")

        if edge_weight is None:
            edge_weight = x.new_ones(source.shape[0])
        else:
            edge_weight = edge_weight.to(dtype=x.dtype, device=x.device).view(
                -1
            )
            if edge_weight.shape[0] != source.shape[0]:
                raise ValueError(
                    "edge_weight must have one scalar per edge_index column."
                )
            if not torch.isfinite(edge_weight).all():
                raise ValueError("edge_weight must be finite.")
            if (edge_weight < 0).any():
                raise ValueError("edge_weight must be non-negative.")

        if source.numel() == 0:
            return torch.zeros_like(x)

        non_loop = source != target
        source = source[non_loop]
        target = target[non_loop]
        edge_weight = edge_weight[non_loop]
        if source.numel() == 0:
            return torch.zeros_like(x)

        source, target = (
            torch.cat((source, target), dim=0),
            torch.cat((target, source), dim=0),
        )
        edge_weight = torch.cat((edge_weight, edge_weight), dim=0)

        weighted_messages = x[source] * edge_weight.unsqueeze(-1)
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, target, weighted_messages)

        degree = x.new_zeros(x.shape[0])
        degree.index_add_(0, target, edge_weight)

        laplacian = torch.zeros_like(x)
        non_isolated = degree > 0
        laplacian[non_isolated] = x[non_isolated] - aggregated[
            non_isolated
        ] / degree[non_isolated].unsqueeze(-1)
        return laplacian

    def _heat_kernel_taylor(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Approximate ``exp(-t L_rw) x`` with a Taylor polynomial."""
        if self.t == 0.0 or self.taylor_degree == 0:
            return x

        output = x
        term = x
        coefficient = 1.0
        for degree in range(1, self.taylor_degree + 1):
            term = self._random_walk_laplacian(
                term, edge_index=edge_index, edge_weight=edge_weight
            )
            coefficient *= -self.t / degree
            output = output + coefficient * term
        return output

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Apply one BuNN layer.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, hidden_channels]``.
        edge_index : torch.Tensor
            Graph connectivity in PyG COO format.
        edge_weight : torch.Tensor or None, optional
            Optional scalar edge weights. Default is ``None``.

        Returns
        -------
        torch.Tensor
            Updated node features of shape ``[num_nodes, hidden_channels]``.
        """
        _require_node_feature_matrix(
            x, self.hidden_channels, "hidden_channels"
        )
        residual = x
        bundle_maps = self._compute_bundle_maps(x)
        fields = self._to_bundle_fields(x)

        synchronized = torch.einsum("nbij,nbcj->nbci", bundle_maps, fields)
        synchronized = self.channel_mixer(
            self._from_bundle_fields(synchronized)
        )
        synchronized = self._to_bundle_fields(synchronized)

        diffused = self._heat_kernel_taylor(
            self._from_bundle_fields(synchronized),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        diffused = self._to_bundle_fields(diffused)

        desynchronized = torch.einsum("nbji,nbcj->nbci", bundle_maps, diffused)
        out = self._from_bundle_fields(desynchronized)
        out = self.activation(out)
        out = self.dropout(out)
        if self.residual:
            out = out + residual
        return self.norm(out)

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        for module in self.modules():
            if module is self:
                continue
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


class BuNN(nn.Module):
    r"""Bundle Neural Network graph encoder.

    BuNN performs global message diffusion through flat vector bundles while
    staying compatible with TopoBench's standard graph wrapper. The backbone
    accepts the same call pattern as PyG graph models and returns node
    embeddings for downstream TopoBench readouts.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    hidden_channels : int
        Hidden and output feature dimension.
    num_layers : int, optional
        Number of BuNN layers. Default is 2.
    num_bundles : int, optional
        Number of parallel 2D bundles. Default is 8.
    bundle_dim : int, optional
        Bundle dimension. Only 2 is currently supported. Default is 2.
    t : float, optional
        Heat diffusion time used in each layer. Default is 1.0.
    taylor_degree : int, optional
        Taylor approximation degree for the heat kernel. Default is 3.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    act : str, optional
        Activation name. Default is ``gelu``.
    angle_hidden_channels : int or None, optional
        Hidden width of each angle network. If ``None``, twice the number of
        bundles is used. Default is ``None``.
    include_reflections : bool, optional
        Whether to parameterize half of the 2D bundles as determinant -1
        orthogonal maps. Default is ``True``.
    residual : bool, optional
        Whether to add residual connections around BuNN layers. Default is
        ``False``.
    norm : str or None, optional
        Optional layer normalization mode. Use ``layer_norm`` to enable
        ``torch.nn.LayerNorm``. Default is ``None``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        num_bundles: int = 8,
        bundle_dim: int = 2,
        t: float = 1.0,
        taylor_degree: int = 3,
        dropout: float = 0.0,
        act: str = "gelu",
        angle_hidden_channels: int | None = None,
        include_reflections: bool = True,
        residual: bool = False,
        norm: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        in_channels = _require_positive_int("in_channels", in_channels)
        hidden_channels = _require_positive_int(
            "hidden_channels", hidden_channels
        )
        num_layers = _require_positive_int("num_layers", num_layers)
        num_bundles = _require_positive_int("num_bundles", num_bundles)
        bundle_dim = _require_positive_int("bundle_dim", bundle_dim)
        taylor_degree = _require_non_negative_int(
            "taylor_degree", taylor_degree
        )
        if angle_hidden_channels is not None:
            angle_hidden_channels = _require_positive_int(
                "angle_hidden_channels", angle_hidden_channels
            )
        include_reflections = _require_bool(
            "include_reflections", include_reflections
        )
        residual = _require_bool("residual", residual)
        diffusion_time = _require_non_negative_float("t", t)
        dropout = _require_probability("dropout", dropout)
        if bundle_dim != 2:
            raise ValueError("BuNN currently supports bundle_dim=2 only.")
        if include_reflections and num_bundles % 2 != 0:
            raise ValueError(
                "num_bundles must be even when include_reflections=True."
            )
        if hidden_channels % (num_bundles * bundle_dim) != 0:
            raise ValueError(
                "hidden_channels must be divisible by "
                "num_bundles * bundle_dim."
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_bundles = num_bundles
        self.bundle_dim = bundle_dim
        self.t = diffusion_time
        self.taylor_degree = taylor_degree
        self.include_reflections = include_reflections

        if in_channels == hidden_channels:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList(
            [
                BuNNLayer(
                    hidden_channels=hidden_channels,
                    num_bundles=num_bundles,
                    bundle_dim=bundle_dim,
                    t=diffusion_time,
                    taylor_degree=taylor_degree,
                    dropout=dropout,
                    act=act,
                    angle_hidden_channels=angle_hidden_channels,
                    include_reflections=include_reflections,
                    residual=residual,
                    norm=norm,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Compute node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            Graph connectivity in PyG COO format.
        batch : torch.Tensor or None, optional
            Batch assignment vector. It is accepted for wrapper compatibility
            but not used by the diffusion operator. Default is ``None``.
        edge_weight : torch.Tensor or None, optional
            Optional scalar edge weights. Default is ``None``.
        **kwargs
            Additional keyword arguments accepted for PyG compatibility.

        Returns
        -------
        torch.Tensor
            Node embeddings of shape ``[num_nodes, hidden_channels]``.
        """
        del batch, kwargs

        _require_node_feature_matrix(x, self.in_channels, "in_channels")
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
        return x

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        if hasattr(self.input_projection, "reset_parameters"):
            self.input_projection.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_bundles={self.num_bundles}, "
            f"bundle_dim={self.bundle_dim}, "
            f"include_reflections={self.include_reflections}, "
            f"t={self.t}, "
            f"taylor_degree={self.taylor_degree})"
        )
