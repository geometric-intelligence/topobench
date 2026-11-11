"""Scalable Multi-Cellular Network (SMCN) backbone.

SMCN extends higher-order message passing with features indexed by
(node, edge) pairs — a bag holding one marked copy of the node set per
edge, processed with subgraph-GNN-style updates (SCL layers). This
mitigates provable expressivity limitations of standard topological
message passing (diameter, orientability, homology; Theorem 4.3 of the
paper). The assembly follows the model the authors evaluate on graph
benchmarks: CIN-style blocks, bag initialization with distance marking,
a stack of SCL layers over the (0, 1) pair space (Eq. 55; the
GNN-SSWL+ instantiation), and sum-pooling back to the cochains.

The optional ``learned_lifting`` mode selects the 2-cells produced by
the cycle lifting with a straight-through scorer (DiffLift), learning
which cycles enter the complex end-to-end.

References
----------
Eitan et al. "Topological Blindspots: Understanding and Extending
Topological Deep Learning Through the Lens of Expressivity." ICLR 2025.
https://arxiv.org/abs/2408.05486 (official implementation:
https://github.com/yoavgelberg/SMCN)
Franco et al. "Differentiable Lifting for Topological Neural Networks."
https://openreview.net/forum?id=eC89CbINIw
"""

import torch
from torch import nn

from topobench.nn.backbones.cell.smcn_utils.layers import (
    BagInit,
    BagPool,
    CINBlock,
    SCLLayer,
    TwoCellInit,
)
from topobench.nn.backbones.cell.smcn_utils.structures import (
    build_smcn_structures,
)
from topobench.nn.liftings.difflift import (
    CellScorer,
    DiffLiftEncoder,
)


class SMCN(nn.Module):
    """SMCN backbone operating on 2-dimensional cell complexes.

    Parameters
    ----------
    in_channels : int
        Feature dimension of all cochains (as produced by the feature
        encoder); also the CIN-block width.
    sub_channels : int, optional
        Width of the subcomplex (SCL) layers. Default is ``in_channels``.
    n_cin_layers : int, optional
        Number of CIN blocks before the bag. Default is 1.
    n_scl_layers : int, optional
        Number of SCL layers (at least 2: the first maps into
        ``sub_channels``, the last maps back). Default is 3.
    num_mlp_layers : int, optional
        Depth of the CIN convolution MLPs. Default is 2.
    max_rank_out : int, optional
        Highest rank updated by the final reduced CIN block. Default 1.
    max_dist : int, optional
        Distance cutoff of the bag marking. Default is 10.
    dropout : float, optional
        Dropout applied between stages. Default is 0.0.
    learned_lifting : bool, optional
        If True, candidate 2-cells are gated by a straight-through
        scorer (DiffLift) instead of all being kept. Default is False.
    sharpening : float, optional
        Logit sharpening of the 2-cell scorer. Default is 10.0.

    Raises
    ------
    ValueError
        If ``n_scl_layers`` is smaller than 2.
    """

    def __init__(
        self,
        in_channels,
        sub_channels=None,
        n_cin_layers=1,
        n_scl_layers=3,
        num_mlp_layers=2,
        max_rank_out=1,
        max_dist=10,
        dropout=0.0,
        learned_lifting=False,
        sharpening=10.0,
    ):
        super().__init__()
        if n_scl_layers < 2:
            raise ValueError("n_scl_layers must be at least 2")
        sub_channels = sub_channels or in_channels
        self.out_channels = in_channels
        self.max_dist = max_dist
        self.dropout = nn.Dropout(dropout)
        self.learned_lifting = learned_lifting

        if learned_lifting:
            self.lift_encoder = DiffLiftEncoder(in_channels)
            self.select = CellScorer(32, sharpening=sharpening)

        self.two_cell_init = TwoCellInit(in_channels, num_mlp_layers)
        self.cin_blocks = nn.ModuleList(
            CINBlock(in_channels, num_mlp_layers) for _ in range(n_cin_layers)
        )
        self.bag_init = BagInit(in_channels, max_dist)
        dims = (
            [(in_channels, sub_channels)]
            + [(sub_channels, sub_channels)] * (n_scl_layers - 2)
            + [(sub_channels, in_channels)]
        )
        self.scl_layers = nn.ModuleList(
            SCLLayer(d_in, d_out, edge_dim=in_channels, max_dist=max_dist)
            for d_in, d_out in dims
        )
        self.bag_pool = BagPool()
        self.final_block = CINBlock(
            in_channels, num_mlp_layers, max_rank=max_rank_out
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        incidence_1,
        incidence_2,
        batch_0,
        batch_1,
        batch_2,
    ):
        """Run the SMCN forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features ``[n0, in_channels]``.
        x_1 : torch.Tensor
            Edge features ``[n1, in_channels]``.
        x_2 : torch.Tensor
            2-cell features ``[n2, in_channels]``.
        incidence_1 : torch.Tensor
            Node-edge incidence ``[n0, n1]`` (sparse or dense).
        incidence_2 : torch.Tensor
            Edge-2-cell incidence ``[n1, n2]`` (sparse or dense).
        batch_0 : torch.Tensor
            Graph index of every node, ``[n0]``.
        batch_1 : torch.Tensor
            Graph index of every edge, ``[n1]``.
        batch_2 : torch.Tensor
            Graph index of every 2-cell, ``[n2]``.

        Returns
        -------
        tuple of torch.Tensor
            Updated ``(x_0, x_1, x_2)``.
        """
        with torch.no_grad():
            structures = build_smcn_structures(
                incidence_1,
                incidence_2,
                batch_0,
                batch_1,
                batch_2,
                max_dist=self.max_dist,
            )
        s = structures

        gate = None
        if self.learned_lifting:
            z = self.lift_encoder(x_0, s.a01_pairs)
            gate = self.select(z, s.inc02_pairs, batch_2)

        x_2 = self.two_cell_init(x_0, x_2, s.inc02_pairs)
        if gate is not None:
            x_2 = gate.unsqueeze(-1) * x_2

        a12_gate = gate[s.a12_bridge] if gate is not None else None
        for block in self.cin_blocks:
            x_0, x_1, x_2 = block(x_0, x_1, x_2, s, a12_gate=a12_gate)
            if gate is not None:
                x_2 = gate.unsqueeze(-1) * x_2
            x_0, x_1 = self.dropout(x_0), self.dropout(x_1)

        x_bag = self.bag_init(x_0, x_1, s)
        for layer in self.scl_layers:
            x_bag = self.dropout(layer(x_bag, x_1, s))
        x_0, x_1 = self.bag_pool(x_bag, s, x_0.size(0), x_1.size(0))

        x_0, x_1, x_2 = self.final_block(x_0, x_1, x_2, s, a12_gate=a12_gate)
        if gate is not None:
            x_2 = gate.unsqueeze(-1) * x_2
        return x_0, x_1, x_2
