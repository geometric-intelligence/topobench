"""Learnable directional maps for copresheaf message passing."""

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseCopresheafMap(nn.Module, ABC):
    r"""Base class for maps :math:`\rho_{y\to x}:F(y)\to F(x)`.

    A map is evaluated independently for every directed neighborhood edge and
    every head. Subclasses return tensors of shape
    ``[num_edges, heads, stalk_dimension, stalk_dimension]``.

    Parameters
    ----------
    channels : int
        Dimension of the cell features used to condition the map.
    heads : int
        Number of independent stalk bundles.
    stalk_dimension : int, optional
        Dimension of one stalk. By default, ``channels // heads``.
    dropout : float
        Dropout applied to learned map perturbations.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 1,
        stalk_dimension: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")

        stalk_dimension = stalk_dimension or channels // heads
        if heads * stalk_dimension != channels:
            raise ValueError(
                "channels must equal heads * stalk_dimension; got "
                f"{channels}, {heads}, and {stalk_dimension}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.channels = channels
        self.heads = heads
        self.stalk_dimension = stalk_dimension
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "identity",
            torch.eye(stalk_dimension).view(
                1, 1, stalk_dimension, stalk_dimension
            ),
            persistent=False,
        )

    def _validate(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Validate that edge features have a compatible shape.

        Parameters
        ----------
        source : torch.Tensor
            Source-cell features for each edge, shape
            ``[num_edges, channels]``.
        target : torch.Tensor
            Target-cell features for each edge, shape
            ``[num_edges, channels]``.
        """
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError(
                "source and target features must be rank-2 tensors"
            )
        if source.shape != target.shape:
            raise ValueError(
                "source and target edge features must have the same shape"
            )
        if source.size(-1) != self.channels:
            raise ValueError(
                f"expected {self.channels} features, got {source.size(-1)}"
            )

    def _eye(self, num_edges: int) -> torch.Tensor:
        """Broadcast the stored identity to a batch of edges.

        Parameters
        ----------
        num_edges : int
            Number of directed edges to broadcast the identity over.

        Returns
        -------
        torch.Tensor
            Identity transport of shape
            ``[num_edges, heads, stalk_dimension, stalk_dimension]``.
        """
        return self.identity.expand(
            num_edges,
            self.heads,
            self.stalk_dimension,
            self.stalk_dimension,
        )

    @abstractmethod
    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Construct a transport matrix for each directed edge.

        Parameters
        ----------
        source : torch.Tensor
            Source-cell features for each edge, shape
            ``[num_edges, channels]``.
        target : torch.Tensor
            Target-cell features for each edge, shape
            ``[num_edges, channels]``.

        Returns
        -------
        torch.Tensor
            Transport tensor of shape
            ``[num_edges, heads, stalk_dimension, stalk_dimension]``.
        """


class IdentityCopresheafMap(BaseCopresheafMap):
    """Return identity transport on every edge.

    This recovers ordinary message passing and is useful as a controlled
    ablation of the learned copresheaf maps.
    """

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Return the identity transport for each edge.

        Parameters
        ----------
        source : torch.Tensor
            Source-cell features for each edge, shape
            ``[num_edges, channels]``.
        target : torch.Tensor
            Target-cell features for each edge, shape
            ``[num_edges, channels]``.

        Returns
        -------
        torch.Tensor
            Identity transport of shape
            ``[num_edges, heads, stalk_dimension, stalk_dimension]``.
        """
        self._validate(source, target)
        return self._eye(source.size(0))


class DiagonalCopresheafMap(BaseCopresheafMap):
    r"""Learn :math:`\rho=I+\operatorname{diag}(\tanh(W[h_x;h_y]))`.

    This is the map used by the CopresheafGCN and CopresheafSAGE
    experiments in Section 6.2 of the CTNN paper.

    Parameters
    ----------
    *args : tuple
        Positional arguments forwarded to
        :class:`BaseCopresheafMap`.
    init_std : float, optional
        Standard deviation used to initialize the linear layer's
        weights. By default, ``0.01``.
    **kwargs : dict
        Keyword arguments forwarded to :class:`BaseCopresheafMap`.
    """

    def __init__(self, *args, init_std: float = 0.01, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(2 * self.channels, self.channels)
        nn.init.normal_(self.linear.weight, std=init_std)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Construct a diagonal identity-plus-perturbation transport.

        Parameters
        ----------
        source : torch.Tensor
            Source-cell features for each edge, shape
            ``[num_edges, channels]``.
        target : torch.Tensor
            Target-cell features for each edge, shape
            ``[num_edges, channels]``.

        Returns
        -------
        torch.Tensor
            Transport tensor of shape
            ``[num_edges, heads, stalk_dimension, stalk_dimension]``.
        """
        self._validate(source, target)
        pair = torch.cat((target, source), dim=-1)
        diagonal = torch.tanh(self.linear(pair)).view(
            -1, self.heads, self.stalk_dimension
        )
        delta = torch.diag_embed(self.dropout(diagonal))
        return self._eye(source.size(0)) + delta


class FullCopresheafMap(BaseCopresheafMap):
    r"""Learn a full SheafFC map :math:`\rho=I+\tanh(W[h_x;h_y])`.

    The zero initialization makes every map initially equal to the identity,
    as prescribed for the SheafFC family in the paper's map catalogue.

    Parameters
    ----------
    *args : tuple
        Positional arguments forwarded to
        :class:`BaseCopresheafMap`.
    zero_init : bool, optional
        Whether to zero-initialize the linear layer so every map
        starts as the identity. By default, ``True``.
    **kwargs : dict
        Keyword arguments forwarded to :class:`BaseCopresheafMap`.
    """

    def __init__(self, *args, zero_init: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        output_size = self.heads * self.stalk_dimension * self.stalk_dimension
        self.linear = nn.Linear(2 * self.channels, output_size)
        if zero_init:
            nn.init.zeros_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Construct a full identity-plus-perturbation transport.

        Parameters
        ----------
        source : torch.Tensor
            Source-cell features for each edge, shape
            ``[num_edges, channels]``.
        target : torch.Tensor
            Target-cell features for each edge, shape
            ``[num_edges, channels]``.

        Returns
        -------
        torch.Tensor
            Transport tensor of shape
            ``[num_edges, heads, stalk_dimension, stalk_dimension]``.
        """
        self._validate(source, target)
        pair = torch.cat((target, source), dim=-1)
        delta = torch.tanh(self.linear(pair)).view(
            -1,
            self.heads,
            self.stalk_dimension,
            self.stalk_dimension,
        )
        return self._eye(source.size(0)) + self.dropout(delta)


class SharedLocalCopresheafMap(BaseCopresheafMap):
    r"""Modulate a full transport with a local edge gate.

    This implements the CT-SharedLoc option from Section 6.3:
    :math:`\rho=I+\sigma(g(h_x,h_y))\Delta(h_x,h_y)`.

    Parameters
    ----------
    *args : tuple
        Positional arguments forwarded to
        :class:`BaseCopresheafMap`.
    hidden_channels : int, optional
        Width of the hidden layer in the map and gate MLPs. By
        default, ``channels``.
    **kwargs : dict
        Keyword arguments forwarded to :class:`BaseCopresheafMap`.
    """

    def __init__(self, *args, hidden_channels: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_channels = hidden_channels or self.channels
        pair_channels = 2 * self.channels
        map_channels = self.heads * self.stalk_dimension * self.stalk_dimension
        self.map_mlp = nn.Sequential(
            nn.Linear(pair_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, map_channels),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(pair_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, self.heads),
        )
        nn.init.normal_(self.map_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.map_mlp[-1].bias)

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Construct a gated identity-plus-perturbation transport.

        Parameters
        ----------
        source : torch.Tensor
            Source-cell features for each edge, shape
            ``[num_edges, channels]``.
        target : torch.Tensor
            Target-cell features for each edge, shape
            ``[num_edges, channels]``.

        Returns
        -------
        torch.Tensor
            Transport tensor of shape
            ``[num_edges, heads, stalk_dimension, stalk_dimension]``.
        """
        self._validate(source, target)
        pair = torch.cat((target, source), dim=-1)
        delta = torch.tanh(self.map_mlp(pair)).view(
            -1,
            self.heads,
            self.stalk_dimension,
            self.stalk_dimension,
        )
        gate = torch.sigmoid(self.gate_mlp(pair)).view(-1, self.heads, 1, 1)
        return self._eye(source.size(0)) + self.dropout(gate * delta)


class OuterProductCopresheafMap(BaseCopresheafMap):
    r"""Learn an outer-product transport map.

    Per head, :math:`\rho=I+\tanh((W_qh_x)(W_kh_y)^\top)`. The map is a
    residualized version of the outer-product family used in the paper's
    physics experiments, which keeps deep stacks stable at initialization.

    Parameters
    ----------
    *args : tuple
        Positional arguments forwarded to
        :class:`BaseCopresheafMap`.
    residual : bool, optional
        Whether to add the identity to the outer-product term. By
        default, ``True``.
    **kwargs : dict
        Keyword arguments forwarded to :class:`BaseCopresheafMap`.
    """

    def __init__(self, *args, residual: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual = residual
        self.query = nn.Linear(self.channels, self.channels, bias=False)
        self.key = nn.Linear(self.channels, self.channels, bias=False)

    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Construct an outer-product transport for each edge.

        Adds the identity to the outer-product term when
        ``self.residual`` is ``True``; otherwise returns the
        outer-product term unmodified.

        Parameters
        ----------
        source : torch.Tensor
            Source-cell features for each edge, shape
            ``[num_edges, channels]``.
        target : torch.Tensor
            Target-cell features for each edge, shape
            ``[num_edges, channels]``.

        Returns
        -------
        torch.Tensor
            Transport tensor of shape
            ``[num_edges, heads, stalk_dimension, stalk_dimension]``.
        """
        self._validate(source, target)
        query = self.query(target).view(-1, self.heads, self.stalk_dimension)
        key = self.key(source).view(-1, self.heads, self.stalk_dimension)
        transport = torch.tanh(torch.einsum("ehd,ehk->ehdk", query, key))
        transport = self.dropout(transport)
        if self.residual:
            transport = self._eye(source.size(0)) + transport
        return transport


COPRESHEAF_MAPS = {
    "identity": IdentityCopresheafMap,
    "diagonal": DiagonalCopresheafMap,
    "full": FullCopresheafMap,
    "outer_product": OuterProductCopresheafMap,
    "shared_local": SharedLocalCopresheafMap,
}


def create_copresheaf_map(map_type: str, **kwargs) -> BaseCopresheafMap:
    """Create a copresheaf map from its configuration name.

    Parameters
    ----------
    map_type : str
        One of ``COPRESHEAF_MAPS``, for example ``"diagonal"`` or ``"full"``.
    **kwargs : dict
        Constructor arguments forwarded to the selected map class, for
        example ``channels``, ``heads``, and ``stalk_dimension``.

    Returns
    -------
    BaseCopresheafMap
        The instantiated map module.
    """
    try:
        map_class = COPRESHEAF_MAPS[map_type.lower()]
    except KeyError as error:
        choices = ", ".join(sorted(COPRESHEAF_MAPS))
        raise ValueError(
            f"unknown copresheaf map {map_type!r}; choose from {choices}"
        ) from error
    return map_class(**kwargs)
