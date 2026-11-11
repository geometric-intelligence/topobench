"""Gated cross-rank readout: direct, learnably-weighted rank-to-node pooling.

Motivation. TopoBench (Telyatnikov et al., 2025) flags higher-order pooling as
an open problem: a simplicial model may update edge or triangle representations
while pooling only over nodes, and the sequential cascade of
``PropagateSignalDown`` forces rank-``r`` signal through every intermediate
projection, with no mechanism to weight ranks by task relevance. This module
gives every rank its **own direct path** to rank 0 and fuses the paths with a
**learned convex gate** (softmax over ranks), initialised rank-0-biased so the
fusion is *opt-in*: near-identity by default, opening higher-rank paths only
where they help. The learned weights are exposed (``rank_weights``) and describe
the readout's mixing; the companion gate-diagnostics study documents that they
must NOT be read as attribution of the *backbone's* topology reliance (a
negative result: with cross-rank message passing underneath, readout paths are
redundant and gates carry no such signal).
"""

import topomodelx
import torch
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class GatedCrossRankReadout(AbstractZeroCellReadOut):
    r"""Cross-rank readout with per-rank direct paths and a learned convex gate.

    For each rank :math:`r \ge 1`, features are pushed down to rank 0 through a
    dedicated chain of incidence convolutions (rank :math:`j \to j-1` via
    :math:`B_j`), layer-normalised at every hop. The rank-0 representation is
    the convex combination

    .. math::
        x_0^{\mathrm{out}} \;=\; \sum_{r=0}^{D-1} w_r \, P_r(x_r),
        \qquad w = \mathrm{softmax}(g),

    where :math:`P_0` is the identity, :math:`P_r` is rank :math:`r`'s direct
    path, and :math:`g \in \mathbb{R}^{D}` are learnable gate logits. Unlike a
    sequential cascade, each rank reaches the nodes without passing through the
    projections of intermediate ranks; the learned mixing weights :math:`w` are
    exposed (see :attr:`rank_weights`) but describe readout mixing only, not
    backbone topology reliance.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments. It should contain the following keys:
        - num_cell_dimensions (int): Highest order of cells considered by the model.
        - hidden_dim (int): Dimension of the cell representations.
        - readout_name (str): Readout name.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = kwargs["readout_name"]
        self.num_cell_dimensions = kwargs["num_cell_dimensions"]
        self.hidden_dim = kwargs["hidden_dim"]

        # One direct path per source rank r >= 1: hops r -> r-1 -> ... -> 0.
        for r in range(1, self.num_cell_dimensions):
            for j in range(r, 0, -1):
                setattr(
                    self,
                    f"conv_{r}_{j}",
                    topomodelx.base.conv.Conv(
                        self.hidden_dim, self.hidden_dim, aggr_norm=False
                    ),
                )
                setattr(
                    self, f"ln_{r}_{j}", torch.nn.LayerNorm(self.hidden_dim)
                )

        # Learnable gate logits over ranks {0, ..., D-1}; softmax -> convex
        # weights. Initialisation is rank-0-biased ("opt-in"): the readout starts
        # near-identity on x_0 and *opens* higher-rank paths only where they help.
        # Rationale: with uniform init, most of the fused signal comes from
        # higher-rank paths, which injects noise on tasks where those ranks carry
        # little information and can cripple learning (documented in the companion
        # gate-diagnostics study, v1 -> v2).
        rank0_bias = float(kwargs.get("rank0_bias", 2.0))
        init = torch.zeros(self.num_cell_dimensions)
        init[0] = rank0_bias
        self.gate_logits = torch.nn.Parameter(init)

    @property
    def rank_weights(self) -> torch.Tensor:
        r"""Learned convex weights over ranks (softmax of the gate logits).

        Returns
        -------
        torch.Tensor
            Tensor of shape ``[num_cell_dimensions]`` summing to one; entry
            ``r`` is the contribution of rank ``r`` to the node representation.
        """
        return torch.softmax(self.gate_logits, dim=0)

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Fuse all ranks into the rank-0 representation via gated direct paths.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output (``x_0``, ``x_1``, ...).
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data (``incidence_j``).

        Returns
        -------
        dict
            Dictionary with ``x_0`` replaced by the gated cross-rank fusion.
        """
        weights = torch.softmax(self.gate_logits, dim=0)
        fused = weights[0] * model_out["x_0"]

        for r in range(1, self.num_cell_dimensions):
            h = model_out[f"x_{r}"]
            for j in range(r, 0, -1):
                h = getattr(self, f"conv_{r}_{j}")(h, batch[f"incidence_{j}"])
                h = getattr(self, f"ln_{r}_{j}")(h)
            fused = fused + weights[r] * h

        model_out["x_0"] = fused
        return model_out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_cell_dimensions="
            f"{self.num_cell_dimensions}, hidden_dim={self.hidden_dim}, "
            f"readout_name={self.name})"
        )
