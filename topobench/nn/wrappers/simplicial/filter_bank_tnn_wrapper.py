"""Wrapper for the FilterBankTNN (SCNN) model."""

from topobench.nn.wrappers.base import AbstractWrapper


class FilterBankTNNWrapper(AbstractWrapper):
    r"""Wrapper for the FilterBankTNN (SCNN) model.

    Feeds each rank's lower/upper Hodge Laplacians to the backbone. Rank 0
    (nodes) has no lower simplices, so its down operator is ``None`` and
    its up operator is the full Hodge Laplacian ``L_0`` (the graph
    Laplacian).
    """

    def forward(self, batch):
        r"""Forward pass for the FilterBankTNN wrapper.

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
        down_all = (None, batch.down_laplacian_1, batch.down_laplacian_2)
        up_all = (
            batch.hodge_laplacian_0,
            batch.up_laplacian_1,
            batch.up_laplacian_2,
        )
        x_0, x_1, x_2 = self.backbone(x_all, down_all, up_all)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out
