r"""GAMMA graph backbone.

This module implements the routing layer from *GAMMA: Gated Multi-hop Message
Passing for Homophily-Agnostic Node Representation in GNNs* (NeurIPS 2025).
The implementation follows Algorithm 1 in the paper:

1. Project once with a shared matrix, :math:`H^{(0)} = XW`.
2. Recurrently propagate, :math:`H^{(p)} = AH^{(p-1)}`.
3. Apply hop-specific channel gates and node-wise L2 normalization.
4. Mix hops with node-specific routing coefficients and capsule squash.

The paper and its reference implementation disagree in several places. The
paper specifies normalized adjacency, L2 hop normalization, exact capsule
squash, one shared projection, and recurrent sparse propagation. The reference
code uses raw adjacency, affine LayerNorm, an epsilon inside the squash square
root, repeated projections, and walks recomputed from zero. This module uses
the paper-faithful path by default and exposes explicit compatibility settings
to reproduce and test the published reference code. That profile is
``propagation="raw"``, ``normalization="layer_norm"``,
``squash_mode="reference"``, and ``projection_bias=True``.

References
----------
Ghazizadeh, A., Ewetz, R., and Zheng, H. "GAMMA: Gated Multi-hop Message
Passing for Homophily-Agnostic Node Representation in GNNs." NeurIPS 2025.
https://proceedings.neurips.cc/paper_files/paper/2025/file/1129f729097a28bfb5b836ec4bf94478-Paper-Conference.pdf

Official implementation, pinned for reproducibility:
https://github.com/amir-ghz/GAMMA/tree/36d054b4ae53c0c8e360d85d056f5207d61c8281
"""

import torch
import torch.nn as nn
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class GAMMA(nn.Module):
    r"""Gated multi-hop message-passing backbone.

    For input features :math:`X \in \mathbb{R}^{N \times d_{in}}`, the
    backbone computes candidate embeddings

    .. math::

        U_i^{(p)} =
        \operatorname{L2Norm}\left(
        \left(A^p XW\right)_i \odot \gamma_p\right),

    then applies Algorithm 1's routing-by-agreement updates. Routing logits are
    activations, not parameters, and are reset to zero for every forward pass.
    Consequently, one routing iteration is uniform mixing; at least two
    iterations are required for an adaptive mixture. This maps the paper's
    Eq. (5) to hop normalization, Eqs. (6)--(9) to routing and agreement, and
    Eq. (10) to recurrent propagation.

    ``edge_index`` defines :math:`A` using PyG's source-to-target convention.
    The paper defines :math:`A` as symmetrically normalized adjacency with
    self-loops, which is the default here. The authors' code instead uses raw
    adjacency and ignores its self-loop option; ``propagation="raw"``
    reproduces that behavior.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    max_hops : int, optional
        Maximum propagation power :math:`K`. The model uses all powers from
        zero through ``max_hops``. Default is 2.
    num_routing_iterations : int, optional
        Number of routing passes :math:`R`. Default is 2.
    propagation : {"normalized", "raw"}, optional
        Sparse propagation operator. ``"normalized"`` applies standard
        symmetric normalization to the weighted adjacency after inserting
        remaining self-loops, as specified by the paper. ``"raw"`` uses the
        supplied edges and weights exactly, matching the released code.
        Default is ``"normalized"``.
    normalization : {"l2", "layer_norm"}, optional
        Hop normalization. ``"l2"`` follows Algorithm 1.
        ``"layer_norm"`` reproduces the authors' released code. Default is
        ``"l2"``.
    squash_mode : {"paper", "reference"}, optional
        Capsule squash implementation. ``"paper"`` uses the exact equation.
        ``"reference"`` places ``1e-8`` inside the square root, matching the
        authors' code even for tiny nonzero vectors. Default is ``"paper"``.
    projection_bias : bool, optional
        Add a bias to the shared projection. The paper's :math:`XW` has no
        projection bias, while the reference code does. Default is ``False``.
    bias : bool, optional
        Add the optional final bias from Algorithm 1. Default is ``True``.
    eps : float, optional
        Numerical floor used by L2 normalization and squash. Default is
        ``1e-8``.

    Notes
    -----
    The forward pass performs one dense projection and exactly ``max_hops``
    sparse propagations. It never materializes a dense adjacency matrix or an
    adjacency power. The paper's degree notation around ``A + I`` is
    ambiguous; normalized mode uses the conventional degree of the
    self-loop-augmented adjacency, as implemented by PyG's ``gcn_norm``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_hops: int = 2,
        num_routing_iterations: int = 2,
        propagation: str = "normalized",
        normalization: str = "l2",
        squash_mode: str = "paper",
        projection_bias: bool = False,
        bias: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive")
        if max_hops < 0:
            raise ValueError("max_hops must be non-negative")
        if num_routing_iterations < 1:
            raise ValueError("num_routing_iterations must be at least 1")
        if propagation not in {"normalized", "raw"}:
            raise ValueError(
                "propagation must be either 'normalized' or 'raw'"
            )
        if normalization not in {"l2", "layer_norm"}:
            raise ValueError(
                "normalization must be either 'l2' or 'layer_norm'"
            )
        if squash_mode not in {"paper", "reference"}:
            raise ValueError(
                "squash_mode must be either 'paper' or 'reference'"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_hops = max_hops
        self.num_routing_iterations = num_routing_iterations
        self.propagation = propagation
        self.normalization = normalization
        self.squash_mode = squash_mode
        self.eps = eps

        self.projection = nn.Linear(
            in_channels,
            out_channels,
            bias=projection_bias,
        )
        self.hop_scale = nn.Parameter(torch.empty(max_hops + 1, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        if normalization == "layer_norm":
            self.hop_normalizer = nn.LayerNorm(out_channels, eps=1e-5)
        else:
            self.hop_normalizer = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the projection, hop gates, normalizer, and final bias."""
        self.projection.reset_parameters()
        nn.init.normal_(self.hop_scale, mean=1.0, std=0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if self.hop_normalizer is not None:
            self.hop_normalizer.reset_parameters()

    @staticmethod
    def squash(s: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        r"""Apply Algorithm 1's capsule squash.

        .. math::

            \operatorname{squash}(s) =
            \frac{\lVert s\rVert_2^2}{1 + \lVert s\rVert_2^2}
            \frac{s}{\lVert s\rVert_2}.

        Parameters
        ----------
        s : torch.Tensor
            Tensor whose final dimension contains representation channels.
        eps : float, optional
            Numerical floor for a zero norm. Default is ``1e-8``.

        Returns
        -------
        torch.Tensor
            Squashed tensor with the same shape, dtype, and device as ``s``.
        """
        norm = torch.linalg.vector_norm(s, dim=-1, keepdim=True)
        scale = norm.square() / (1.0 + norm.square())
        safe_eps = max(eps, torch.finfo(s.dtype).tiny)
        return scale * s / norm.clamp_min(safe_eps)

    def _apply_squash(self, s: torch.Tensor) -> torch.Tensor:
        """Apply the configured paper or reference squash.

        Parameters
        ----------
        s : torch.Tensor
            Routed pre-activation vectors.

        Returns
        -------
        torch.Tensor
            Squashed vectors with the same shape, dtype, and device.
        """
        if self.squash_mode == "paper":
            return self.squash(s, eps=self.eps)

        norm_squared = torch.sum(s.square(), dim=-1, keepdim=True)
        scale = norm_squared / (1.0 + norm_squared)
        reference_eps = max(1e-8, torch.finfo(s.dtype).tiny)
        norm = torch.sqrt(norm_squared + reference_eps)
        return scale * s / norm

    def _validate_inputs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None,
        edge_weight: torch.Tensor | None,
    ) -> None:
        """Validate the inexpensive structural parts of a forward input.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix.
        edge_index : torch.Tensor
            COO graph connectivity.
        batch : torch.Tensor or None
            Optional graph index for each node.
        edge_weight : torch.Tensor or None
            Optional raw propagation weight for each edge.
        """
        if x.ndim != 2:
            raise ValueError("x must have shape [num_nodes, in_channels]")
        if x.size(-1) != self.in_channels:
            raise ValueError(
                f"x has {x.size(-1)} channels; expected {self.in_channels}"
            )
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.dtype != torch.long:
            raise TypeError("edge_index must have dtype torch.long")
        if edge_index.device != x.device:
            raise ValueError("edge_index and x must be on the same device")
        if edge_weight is not None:
            if edge_weight.numel() != edge_index.size(1):
                raise ValueError("edge_weight must contain one value per edge")
            if edge_weight.device != x.device:
                raise ValueError(
                    "edge_weight and x must be on the same device"
                )
        if batch is not None:
            if batch.ndim != 1 or batch.numel() != x.size(0):
                raise ValueError("batch must contain one graph index per node")
            if batch.device != x.device:
                raise ValueError("batch and x must be on the same device")

    @staticmethod
    def _propagate(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply one sparse, weighted source-to-target aggregation.

        Parameters
        ----------
        x : torch.Tensor
            Current node embeddings.
        edge_index : torch.Tensor
            COO graph connectivity.
        edge_weight : torch.Tensor or None
            Optional raw propagation weight for each edge.

        Returns
        -------
        torch.Tensor
            Aggregated node embeddings with the same shape as ``x``.
        """
        source, target = edge_index
        messages = x.index_select(0, source)
        if edge_weight is not None:
            weights = edge_weight.reshape(-1, 1).to(dtype=x.dtype)
            messages = messages * weights
        output = x.new_zeros(x.shape)
        return output.index_add(0, target, messages)

    def _prepare_propagation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare the sparse operator selected by ``propagation``.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix used to infer node count and dtype.
        edge_index : torch.Tensor
            COO graph connectivity.
        edge_weight : torch.Tensor or None
            Optional raw adjacency weight for each edge.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor or None]
            Connectivity and weights for recurrent sparse propagation.
        """
        if self.propagation == "raw":
            return edge_index, edge_weight

        weights = (
            None
            if edge_weight is None
            else edge_weight.reshape(-1).to(dtype=x.dtype)
        )
        normalized_edges, normalized_weights = gcn_norm(
            edge_index,
            weights,
            num_nodes=x.size(0),
            add_self_loops=True,
            flow="source_to_target",
            dtype=x.dtype,
        )
        return normalized_edges, normalized_weights

    def _compute_hop_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute and normalize all candidates from hop zero through K.

        This is Algorithm 1, Lines 1--7, and the recurrent form in Eq. (10).

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix.
        edge_index : torch.Tensor
            COO graph connectivity.
        edge_weight : torch.Tensor or None
            Optional raw propagation weight for each edge.

        Returns
        -------
        torch.Tensor
            Normalized candidates with shape
            ``[num_nodes, max_hops + 1, out_channels]``.
        """
        current = self.projection(x)
        hop_outputs = [current]
        for _ in range(self.max_hops):
            current = self._propagate(
                current,
                edge_index,
                edge_weight,
            )
            hop_outputs.append(current)

        hops = torch.stack(hop_outputs, dim=1)
        hops = hops * self.hop_scale.unsqueeze(0)

        if self.hop_normalizer is not None:
            return self.hop_normalizer(hops)
        safe_eps = max(self.eps, torch.finfo(hops.dtype).tiny)
        norm = torch.linalg.vector_norm(hops, dim=-1, keepdim=True)
        normalized = hops / norm.clamp_min(safe_eps)
        return torch.where(
            norm > safe_eps,
            normalized,
            torch.zeros_like(normalized),
        )

    def _route(
        self,
        hop_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route candidates and return the weights actually used.

        This implements Algorithm 1, Lines 8--16, corresponding to the
        routing, squash, and agreement updates in Eqs. (6)--(9).

        Parameters
        ----------
        hop_embeddings : torch.Tensor
            Normalized candidates with shape
            ``[num_nodes, max_hops + 1, out_channels]``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Routed embeddings and the node-hop coefficients used to form them.
        """
        logits = hop_embeddings.new_zeros(hop_embeddings.shape[:2])
        routing_weights = torch.softmax(logits, dim=1)
        output = hop_embeddings.new_zeros(
            hop_embeddings.size(0),
            hop_embeddings.size(2),
        )

        for iteration in range(self.num_routing_iterations):
            routing_weights = torch.softmax(logits, dim=1)
            mixed = torch.sum(
                routing_weights.unsqueeze(-1) * hop_embeddings,
                dim=1,
            )
            output = self._apply_squash(mixed)

            if iteration + 1 < self.num_routing_iterations:
                agreement = torch.sum(
                    hop_embeddings * output.unsqueeze(1),
                    dim=-1,
                )
                logits = logits + agreement

        return output, routing_weights

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Compute GAMMA node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            COO connectivity with shape ``[2, num_edges]``. Messages travel
            from row zero to row one.
        edge_weight : torch.Tensor, optional
            One raw propagation weight per edge.
        batch : torch.Tensor, optional
            Graph index for each node. GAMMA's routing is node-wise, so the
            vector is validated but no cross-node reduction is required.

        Returns
        -------
        torch.Tensor
            Node embeddings with shape ``[num_nodes, out_channels]``.
        """
        self._validate_inputs(x, edge_index, batch, edge_weight)
        propagation_edges, propagation_weights = self._prepare_propagation(
            x,
            edge_index,
            edge_weight,
        )
        hop_embeddings = self._compute_hop_embeddings(
            x,
            propagation_edges,
            propagation_weights,
        )
        output, _ = self._route(hop_embeddings)
        if self.bias is not None:
            output = output + self.bias
        return output

    def get_routing_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the node-hop weights that produce the returned embedding.

        Unlike the authors' helper, this method returns the final softmax used
        to form the output, not a softmax after an unused last logit update.

        Parameters
        ----------
        x : torch.Tensor
            Node features with shape ``[num_nodes, in_channels]``.
        edge_index : torch.Tensor
            COO connectivity with shape ``[2, num_edges]``.
        edge_weight : torch.Tensor, optional
            One raw propagation weight per edge.
        batch : torch.Tensor, optional
            Graph index for each node.

        Returns
        -------
        torch.Tensor
            Routing weights with shape ``[num_nodes, max_hops + 1]``.
        """
        self._validate_inputs(x, edge_index, batch, edge_weight)
        with torch.no_grad():
            propagation_edges, propagation_weights = self._prepare_propagation(
                x,
                edge_index,
                edge_weight,
            )
            hop_embeddings = self._compute_hop_embeddings(
                x,
                propagation_edges,
                propagation_weights,
            )
            _, routing_weights = self._route(hop_embeddings)
        return routing_weights

    def extra_repr(self) -> str:
        """Return the compact configuration shown by ``repr``.

        Returns
        -------
        str
            Human-readable architectural settings.
        """
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"max_hops={self.max_hops}, "
            "num_routing_iterations="
            f"{self.num_routing_iterations}, "
            f"propagation={self.propagation!r}, "
            f"normalization={self.normalization!r}, "
            f"squash_mode={self.squash_mode!r}"
        )
