"""Wrapper for the Cellular Transformer (CT) model."""

from topobench.nn.wrappers.base import AbstractWrapper


class CellularTransformerWrapper(AbstractWrapper):
    r"""Wrapper for the Cellular Transformer model.

    This wrapper defines the forward pass of the Cellular Transformer
    (`Ballester et al., 2024 <https://arxiv.org/abs/2405.14094>`_). It
    forwards the cell features of ranks 0, 1 and 2 together with the
    neighborhood matrices required by the pairwise cellular attention
    (Section 4.2.1 of the paper): the rank-0 upper adjacency, the
    rank-1 and rank-2 lower adjacencies (coadjacencies) and the
    non-signed incidences between adjacent ranks. The model returns the
    updated embeddings of all three ranks.
    """

    def forward(self, batch):
        r"""Forward pass for the Cellular Transformer wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched cell complex data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        # --- NEW: pull precomputed PE (None if transform not used) ---
        rwpe_0 = getattr(batch, "rwpe_0", None)
        rwpe_1 = getattr(batch, "rwpe_1", None)
        rwpe_2 = getattr(batch, "rwpe_2", None)

        x_0, x_1, x_2 = self.backbone(
            x_0=batch.x_0,
            x_1=batch.x_1,
            x_2=batch.x_2,
            adjacency_0=batch.adjacency_0.coalesce(),
            coadjacency_1=batch.coadjacency_1.coalesce(),
            coadjacency_2=batch.coadjacency_2.coalesce(),
            incidence_1=batch.incidence_1.coalesce(),
            incidence_2=batch.incidence_2.coalesce(),
            rwpe_0=rwpe_0,
            rwpe_1=rwpe_1,
            rwpe_2=rwpe_2,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out
