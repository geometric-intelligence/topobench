"""Wrapper for the SheafTSP model.

Routes cell complex data from TopoBench's batch format into the SheafTSP
backbone, which operates on 1-cells using the down and up Laplacians.
Produces embeddings for all cell dimensions via incidence propagation.
"""

import torch
import torch.nn as nn

from topobench.nn.backbones.cell.sheaf_tsp import SheafConvLayer
from topobench.nn.wrappers.base import AbstractWrapper


class SheafTSPWrapper(AbstractWrapper):
    r"""Wrapper for the SheafTSP model.

    The SheafTSP backbone operates on 1-cell (edge) features using the
    Hodge Laplacians to derive sheaf connectivity.  This wrapper:
      1. Feeds x_1, down_laplacian_1, up_laplacian_1 to the backbone.
      2. Recovers 0-cell embeddings via incidence_1 (boundary map),
         preserving the encoded 0-cell features via a residual sum so
         node-level signal is not discarded.
      3. Exposes the backbone's sheaf Dirichlet energy (Eq. 15 of
         Tandon et al.) as ``model_out["sheaf_dirichlet"]`` during
         training, for the ``SheafDirichletLoss`` regularizer.

    This follows the same pattern as the CCCNWrapper.

    Parameters
    ----------
    backbone : torch.nn.Module
        The SheafTSP backbone.
    **kwargs : dict
        Arguments for AbstractWrapper (``out_channels``,
        ``num_cell_dimensions``).
    """

    def __init__(self, backbone, **kwargs):
        super().__init__(backbone, **kwargs)
        # Learned embedding of the rank-2 degree signal (per-node count
        # of incident (edge, 2-cell) pairs).  This is the DC component
        # of a rank-2 stalk section: with triangle 2-cells its graph sum
        # equals 6x the triangle count, so a linear readout can recover
        # the count exactly.  Zero-initialized so tasks that do not
        # benefit (e.g. community detection) start unaffected and can
        # keep it switched off.
        self.tri_embed = nn.Linear(1, kwargs["out_channels"], bias=False)
        nn.init.zeros_(self.tri_embed.weight)
        # Warm-start one channel so the count signal is readable from
        # epoch 0: with dropout in the feature path, a fully zero init
        # never matures before early stopping fires on the plateau of
        # the crude density solution. ``tri_warm`` sets the strength;
        # large values make the count channel dominate the regression
        # from the start (the count is exact by construction, so the
        # readout only has to learn a scale).
        with torch.no_grad():
            self.tri_embed.weight[0, 0] = kwargs.get("tri_warm", 0.1)
        # Learnable gate on the feature streams entering x_0. Init 1.0
        # (identity at epoch 0). Each task trains its own weights, so
        # a counting task may drive the gate toward 0, which leaves the
        # sum-pooled representation an exact linear image of the
        # triangle count; a node-classification task keeps it near 1.
        self.use_stream_gate = kwargs.get("stream_gate", False)
        if self.use_stream_gate:
            self.stream_gate = nn.Parameter(torch.ones(1))
        # Sheaf-petals branch: sheaf convolution on the triangle
        # co-membership graph (nodes are adjacent when they share a
        # triangle). The substrate follows HiGCN's order-2 petal
        # observation; the mechanism is our own — learned O(d)
        # transports and kernel weighting via SheafConvLayer. Petal
        # edges come from A .* A^2 > 0 on the raw adjacency, so the
        # branch is exact and independent of the cycle-basis lifting.
        # Zero-init fusion keeps epoch-0 behavior identical.
        # Count-signal source: "auto" prefers the TriangleDegree
        # transform when present; "incidence" always derives t_v from
        # the lifted complex as |B_1||B_2|1 — exact under the clique
        # lifting (each triangle at a node contributes 2), endogenous
        # to the same incidence structure all lifted models consume.
        self.count_source = kwargs.get("count_source", "auto")
        self.use_petals = kwargs.get("petals", False)
        if self.use_petals:
            c = kwargs["out_channels"]
            self.petals = SheafConvLayer(
                in_channels=c,
                out_channels=c,
                stalk_dim=kwargs.get("petals_stalk_dim", 2),
                filter_order=kwargs.get("petals_filter_order", 3),
                mlp_dropout=kwargs.get("petals_mlp_dropout", 0.5),
            )
            self.petals_fuse = nn.Linear(c, c, bias=False)
            nn.init.zeros_(self.petals_fuse.weight)

    @staticmethod
    def _petal_edges(batch, n_nodes):
        """Compute the edges of the triangle co-membership graph.

        Two nodes are petal-adjacent when they are graph-adjacent and
        share at least one common neighbor, i.e. their edge lies in a
        triangle: the nonzero pattern of A .* A^2 restricted to u < v.
        Computed from ``incidence_1`` (exact, lifting-independent).

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch with ``incidence_1``.
        n_nodes : int
            Number of 0-cells in the batch.

        Returns
        -------
        torch.Tensor or None
            Edge index of shape (2, E_petal), or None.
        """
        if not hasattr(batch, "incidence_1"):
            return None
        inc1 = torch.abs(batch.incidence_1.coalesce())
        # A = |B1||B1|^T minus the degree diagonal
        aa = torch.sparse.mm(inc1, inc1.transpose(0, 1)).coalesce()
        idx, val = aa.indices(), aa.values()
        off = idx[0] != idx[1]
        A = torch.sparse_coo_tensor(
            idx[:, off],
            torch.ones(int(off.sum()), device=val.device),
            (n_nodes, n_nodes),
        ).coalesce()
        # Common-neighbor counts on adjacent pairs: A .* (A @ A)
        A2 = torch.sparse.mm(A, A)
        tri = A.mul(A2).coalesce()
        ti, tv = tri.indices(), tri.values()
        keep = (tv > 0) & (ti[0] < ti[1])
        return ti[:, keep]

    def forward(self, batch):
        r"""Forward pass for the SheafTSP wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.
            Expected attributes:
            - ``x_1``: 1-cell features.
            - ``down_laplacian_1``: Down Laplacian of rank 1.
            - ``up_laplacian_1``: Up Laplacian of rank 1.
            - ``incidence_1``: Boundary operator (0-cells × 1-cells).
            - ``y``: Labels.
            - ``batch_0``: Batch assignment for 0-cells.

        Returns
        -------
        dict
            Dictionary containing:
            - ``labels``: Ground truth labels.
            - ``batch_0``: Batch indices for 0-cells.
            - ``x_0``: 0-cell embeddings (propagated from 1-cells,
              plus the embedded rank-2 degree signal).
            - ``x_1``: 1-cell embeddings (direct backbone output).
        """
        x_1 = self.backbone(
            batch.x_1,
            batch.down_laplacian_1.coalesce(),
            batch.up_laplacian_1.coalesce(),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}

        # 1-cell embeddings from backbone
        model_out["x_1"] = x_1

        # Propagate to 0-cells via boundary map: x_0 = B_1 @ x_1,
        # plus a residual with the encoded 0-cell features so the
        # original node-level signal is preserved.
        x_0 = torch.sparse.mm(batch.incidence_1, x_1)
        if hasattr(batch, "x_0") and batch.x_0.shape == x_0.shape:
            x_0 = x_0 + batch.x_0
        if self.use_stream_gate:
            x_0 = self.stream_gate * x_0

        # Rank-2 count signal, injected through the zero-initialized
        # embedding (see __init__). Preferred source: exact per-node
        # triangle counts from the TriangleDegree transform (the cycle
        # lifting attaches a cycle basis only, which undercounts
        # triangles on dense graphs). Fallback: the lifting-derived
        # degree t_v = |B_1||B_2|1.
        t_v = None
        use_transform = self.count_source == "auto" and hasattr(
            batch, "tri_degree"
        )
        if use_transform:
            t_v = batch.tri_degree.to(dtype=x_1.dtype)
        elif hasattr(batch, "incidence_2"):
            inc2 = batch.incidence_2.coalesce()
            if inc2.shape[1] > 0:
                ones_2 = torch.ones(
                    inc2.shape[1], 1, device=x_1.device, dtype=x_1.dtype
                )
                t_e = torch.sparse.mm(torch.abs(inc2), ones_2)
                t_v = torch.sparse.mm(
                    torch.abs(batch.incidence_1.coalesce()), t_e
                )
        if t_v is not None and t_v.shape[0] == x_0.shape[0]:
            x_0 = x_0 + self.tri_embed(t_v)

        # Sheaf-petals branch on the triangle co-membership graph.
        if self.use_petals and hasattr(batch, "x_0"):
            petal_ei = self._petal_edges(batch, x_0.shape[0])
            if petal_ei is not None and petal_ei.shape[1] > 0:
                x_p = self.petals(batch.x_0, petal_ei)
                x_0 = x_0 + self.petals_fuse(x_p)
        model_out["x_0"] = x_0

        # Expose the sheaf Dirichlet energy (Eq. 15 regularizer) only
        # during training; val/test losses stay pure task losses.
        reg = getattr(self.backbone, "dirichlet_energy", None)
        if self.backbone.training and reg is not None:
            model_out["sheaf_dirichlet"] = reg

        return model_out
