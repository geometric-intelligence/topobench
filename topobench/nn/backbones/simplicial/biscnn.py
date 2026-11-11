"""Binarized Simplicial Convolutional Neural Network.

This module implements Bi-SCNN from:

    Yi Yan and Ercan E. Kuruoglu,
    "Binarized Simplicial Convolutional Neural Networks",
    Neural Networks 183 (2025), 106928.
    arXiv:2405.04098.

The implementation follows equations (21)--(27) of the paper. Features are
binarized with the hard-tanh approximation specified by the authors, while
the per-simplex magnitude is retained through row-wise L1 normalization.
Only features are binarized; trainable weights remain full precision.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _matmul(
    operator: torch.Tensor | None, features: torch.Tensor
) -> torch.Tensor:
    """Multiply a dense or sparse operator by a feature matrix.

    Parameters
    ----------
    operator : torch.Tensor or None
        Dense or sparse square simplicial operator. ``None`` represents a
        missing lower or upper Hodge component.
    features : torch.Tensor
        Feature matrix of shape ``[num_simplices, num_features]``.

    Returns
    -------
    torch.Tensor
        Propagated feature matrix.
    """
    if operator is None:
        return torch.zeros_like(features)
    if operator.layout == torch.strided:
        return operator @ features
    return torch.sparse.mm(operator, features)


def _hard_sign(features: torch.Tensor) -> torch.Tensor:
    """Approximate the sign function with hard tanh.

    Parameters
    ----------
    features : torch.Tensor
        Input features.

    Returns
    -------
    torch.Tensor
        Features clipped to the interval ``[-1, 1]``.

    Notes
    -----
    Section 4.5 of the paper states that ``Sign`` is approximated by hard
    tanh so ordinary gradient-based optimization can be used.
    """
    return torch.nn.functional.hardtanh(features, min_val=-1.0, max_val=1.0)


def _row_l1_mean(features: torch.Tensor) -> torch.Tensor:
    """Compute the feature-normalization vector from equation (23).

    Parameters
    ----------
    features : torch.Tensor
        Feature matrix of shape ``[num_simplices, num_features]``.

    Returns
    -------
    torch.Tensor
        Nonnegative row-wise mean absolute value with shape
        ``[num_simplices, 1]``.
    """
    if features.shape[-1] == 0:
        raise ValueError("features must contain at least one channel")
    return features.abs().mean(dim=-1, keepdim=True)


class BiSCNNLayer(nn.Module):
    """One weighted binary-sign simplicial convolution layer.

    For simplex rank ``k``, this layer computes the length-one simplicial
    convolution from equation (24), then separates it into the magnitude and
    binary-sign paths used by equations (25)--(27).

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    use_lower : bool
        Whether to include the lower-Hodge term.
    use_upper : bool
        Whether to include the upper-Hodge term.
    bias : bool, optional
        Whether to add a full-precision bias to the preactivation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_lower: bool,
        use_upper: bool,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("channel dimensions must be positive")
        if not use_lower and not use_upper:
            raise ValueError("at least one Hodge component must be enabled")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_lower = use_lower
        self.use_upper = use_upper

        self.lower_weight = (
            nn.Parameter(torch.empty(in_channels, out_channels))
            if use_lower
            else None
        )
        self.upper_weight = (
            nn.Parameter(torch.empty(in_channels, out_channels))
            if use_upper
            else None
        )
        self.harmonic_weight = nn.Parameter(
            torch.empty(in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize all full-precision trainable matrices."""
        if self.lower_weight is not None:
            nn.init.xavier_uniform_(self.lower_weight)
        if self.upper_weight is not None:
            nn.init.xavier_uniform_(self.upper_weight)
        nn.init.xavier_uniform_(self.harmonic_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        features: torch.Tensor,
        lower_laplacian: torch.Tensor | None,
        upper_laplacian: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the magnitude and binary-sign outputs of one layer.

        Parameters
        ----------
        features : torch.Tensor
            Input simplex features of shape
            ``[num_simplices, in_channels]``.
        lower_laplacian : torch.Tensor or None
            Lower Hodge Laplacian for the current simplex rank.
        upper_laplacian : torch.Tensor or None
            Upper Hodge Laplacian for the current simplex rank.

        Returns
        -------
        tuple of torch.Tensor
            Row-wise normalization ``M`` and hard-sign output ``Q``. Their
            shapes are ``[num_simplices, 1]`` and
            ``[num_simplices, out_channels]``.
        """
        if features.ndim != 2:
            raise ValueError("features must be a rank-two tensor")
        if features.shape[-1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, "
                f"received {features.shape[-1]}"
            )

        preactivation = features @ self.harmonic_weight
        if self.lower_weight is not None:
            preactivation = (
                preactivation
                + _matmul(lower_laplacian, features) @ self.lower_weight
            )
        if self.upper_weight is not None:
            preactivation = (
                preactivation
                + _matmul(upper_laplacian, features) @ self.upper_weight
            )
        if self.bias is not None:
            preactivation = preactivation + self.bias

        magnitude = _row_l1_mean(preactivation)
        binary = _hard_sign(preactivation)
        return magnitude, binary


class BiSCNN(nn.Module):
    """Binarized Simplicial Convolutional Neural Network backbone.

    A separate Bi-SCNN stack is applied to ranks 0, 1, and 2. In accordance
    with the paper, intermediate layers propagate only the binary-sign output.
    Per-layer normalization vectors bypass the binary path and are multiplied
    into the final output.

    Parameters
    ----------
    in_channels_all : sequence of int
        Input dimensions for node, edge, and face features.
    hidden_channels_all : sequence of int
        Output dimensions for node, edge, and face features.
    n_layers : int, optional
        Number of Bi-SCNN layers. Must be at least one.
    sc_order : int, optional
        Maximum simplicial-complex order. Rank-2 upper propagation is enabled
        when ``sc_order > 2``.
    bias : bool, optional
        Whether layers use full-precision biases.
    **kwargs
        Additional arguments accepted for TopoBench compatibility.

    Notes
    -----
    The paper binarizes features but explicitly does not binarize trainable
    weights. This implementation preserves that distinction.
    """

    def __init__(
        self,
        in_channels_all: Sequence[int],
        hidden_channels_all: Sequence[int],
        n_layers: int = 2,
        sc_order: int = 3,
        bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs

        if len(in_channels_all) != 3 or len(hidden_channels_all) != 3:
            raise ValueError(
                "in_channels_all and hidden_channels_all must contain "
                "node, edge, and face dimensions"
            )
        if n_layers < 1:
            raise ValueError("n_layers must be at least one")
        if sc_order < 2:
            raise ValueError("sc_order must be at least two")

        self.in_channels_all = tuple(int(v) for v in in_channels_all)
        self.hidden_channels_all = tuple(int(v) for v in hidden_channels_all)
        self.out_channels = self.hidden_channels_all
        self.n_layers = n_layers
        self.sc_order = sc_order

        rank_flags = (
            (False, True),
            (True, True),
            (True, sc_order > 2),
        )
        self.rank_layers = nn.ModuleList()
        for rank, (use_lower, use_upper) in enumerate(rank_flags):
            layers = nn.ModuleList()
            for layer_index in range(n_layers):
                input_dim = (
                    self.in_channels_all[rank]
                    if layer_index == 0
                    else self.hidden_channels_all[rank]
                )
                layers.append(
                    BiSCNNLayer(
                        in_channels=input_dim,
                        out_channels=self.hidden_channels_all[rank],
                        use_lower=use_lower,
                        use_upper=use_upper,
                        bias=bias,
                    )
                )
            self.rank_layers.append(layers)

    @staticmethod
    def _split_laplacians(
        laplacian_all: Sequence[torch.Tensor],
        sc_order: int,
    ) -> tuple[
        tuple[torch.Tensor | None, torch.Tensor | None],
        tuple[torch.Tensor | None, torch.Tensor | None],
        tuple[torch.Tensor | None, torch.Tensor | None],
    ]:
        """Map TopoBench Laplacian tuples to lower/upper rank pairs.

        Parameters
        ----------
        laplacian_all : sequence of torch.Tensor
            For ``sc_order > 2`` this must contain ``L0, L1_down, L1_up,
            L2_down, L2_up``. For ``sc_order == 2`` it must contain
            ``L0, L1_down, L1_up, L2``.
        sc_order : int
            Simplicial-complex order.

        Returns
        -------
        tuple
            Lower/upper operator pairs for ranks 0, 1, and 2.
        """
        expected = 5 if sc_order > 2 else 4
        if len(laplacian_all) != expected:
            raise ValueError(
                f"expected {expected} Laplacians for sc_order={sc_order}, "
                f"received {len(laplacian_all)}"
            )

        if sc_order > 2:
            lap_0, lap_1_down, lap_1_up, lap_2_down, lap_2_up = laplacian_all
        else:
            lap_0, lap_1_down, lap_1_up, lap_2_down = laplacian_all
            lap_2_up = None

        return (
            (None, lap_0),
            (lap_1_down, lap_1_up),
            (lap_2_down, lap_2_up),
        )

    @staticmethod
    def _run_rank(
        features: torch.Tensor,
        layers: nn.ModuleList,
        lower_laplacian: torch.Tensor | None,
        upper_laplacian: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run one simplex rank through a Bi-SCNN stack.

        Parameters
        ----------
        features : torch.Tensor
            Input features for one simplex rank.
        layers : torch.nn.ModuleList
            Rank-specific Bi-SCNN layers.
        lower_laplacian : torch.Tensor or None
            Lower Hodge Laplacian.
        upper_laplacian : torch.Tensor or None
            Upper Hodge Laplacian.

        Returns
        -------
        torch.Tensor
            Final weighted binary-sign features.
        """
        binary = features
        accumulated_magnitude: torch.Tensor | None = None
        for layer in layers:
            magnitude, binary = layer(
                binary,
                lower_laplacian=lower_laplacian,
                upper_laplacian=upper_laplacian,
            )
            accumulated_magnitude = (
                magnitude
                if accumulated_magnitude is None
                else accumulated_magnitude * magnitude
            )

        if accumulated_magnitude is None:
            raise RuntimeError("Bi-SCNN stack contains no layers")
        return accumulated_magnitude * binary

    def forward(
        self,
        x_all: Sequence[torch.Tensor],
        laplacian_all: Sequence[torch.Tensor],
        incidence_all: Sequence[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute node-, edge-, and face-level representations.

        Parameters
        ----------
        x_all : sequence of torch.Tensor
            Node, edge, and face feature matrices.
        laplacian_all : sequence of torch.Tensor
            TopoBench Hodge-Laplacian tuple.
        incidence_all : sequence of torch.Tensor or None, optional
            Incidence matrices accepted for wrapper compatibility. Bi-SCNN
            uses the derived Hodge operators directly.

        Returns
        -------
        tuple of torch.Tensor
            Output feature matrices for ranks 0, 1, and 2.
        """
        del incidence_all
        if len(x_all) != 3:
            raise ValueError(
                "x_all must contain node, edge, and face features"
            )

        laplacian_pairs = self._split_laplacians(
            laplacian_all, sc_order=self.sc_order
        )
        outputs = []
        for features, layers, (lower, upper) in zip(
            x_all, self.rank_layers, laplacian_pairs, strict=True
        ):
            outputs.append(
                self._run_rank(
                    features,
                    layers,
                    lower_laplacian=lower,
                    upper_laplacian=upper,
                )
            )
        return tuple(outputs)
