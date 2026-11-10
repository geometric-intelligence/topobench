"""Wrapper for the MPSN (Message Passing Simplicial Network) model.

MPSN: Bodnar et al., "Weisfeiler and Lehman Go Topological: Message Passing
Simplicial Networks", ICML 2021, arXiv:2103.03212.
"""

from topobench.nn.wrappers.base import AbstractWrapper


class MPSNWrapper(AbstractWrapper):
    r"""Wrapper for the MPSN model.

    Unpacks the lifted batch and drives the MPSN backbone. Only the per-rank
    features (``x_0/x_1/x_2``) and the signed or unsigned clique-lifted
    incidence matrices (``incidence_1``, ``incidence_2``) are passed to the
    backbone. Their nonzero support is used solely for routing the boundary and
    upper-adjacency messages (no Hodge Laplacians). The backbone returns the
    embeddings of the cells of rank 0, 1 and 2, which the readout consumes.
    """

    def forward(self, batch):
        r"""Forward pass for the MPSN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data. Must carry ``x_0``,
            ``x_1``, ``x_2``, ``incidence_1``, ``incidence_2``, ``batch_0`` and
            ``y``.

        Returns
        -------
        dict
            Dictionary with ``labels``, ``batch_0`` and the per-rank embeddings
            ``x_0``, ``x_1``, ``x_2``.
        """
        x_all = (batch.x_0, batch.x_1, batch.x_2)
        incidence_all = (batch.incidence_1, batch.incidence_2)

        x_0, x_1, x_2 = self.backbone(x_all, incidence_all)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2

        return model_out
