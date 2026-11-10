"""Wrapper for the PolynomialFilterTNN model."""

from topobench.nn.wrappers.base import AbstractWrapper


class PolyFilterTNNWrapper(AbstractWrapper):
    r"""Wrapper for the PolynomialFilterTNN model.

    Feeds the per-rank signals and the full Hodge Laplacians
    ``(L_0, L_1, L_2)`` to the backbone and returns the filtered per-rank
    embeddings (ranks 0, 1, 2).
    """

    def forward(self, batch):
        r"""Forward pass for the PolynomialFilterTNN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output (per-rank embeddings).
        """
        x_all = (batch.x_0, batch.x_1, batch.x_2)
        laplacian_all = (
            batch.hodge_laplacian_0,
            batch.hodge_laplacian_1,
            batch.hodge_laplacian_2,
        )
        x_0, x_1, x_2 = self.backbone(x_all, laplacian_all)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out
