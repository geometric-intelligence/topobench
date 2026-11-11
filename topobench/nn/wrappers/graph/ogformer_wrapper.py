"""Wrapper for the OGFormer backbone."""

from topobench.nn.wrappers.base import AbstractWrapper


class OGFormerWrapper(AbstractWrapper):
    r"""Wrapper for the OGFormer backbone.

    Forwards the rank-0 cell features, the edge index and the batch
    vector to the backbone, and exposes the backbone's auxiliary outputs
    (per-layer queries and attention scores) to
    :class:`topobench.loss.model.OGFormerLoss` through the model output
    dictionary. OGFormer applies its own residual connections internally
    (Eq. (5)), so the wrapper-level residual connection
    should be disabled in the model configuration.
    """

    def forward(self, batch):
        r"""Forward pass for the OGFormer wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_0, aux = self.backbone(
            batch.x_0,
            batch.edge_index,
            batch=batch.batch_0,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        if aux is not None:  # Training mode
            model_out["ogformer_queries"] = aux["queries"]
            model_out["ogformer_attention"] = aux["attention_scores"]

        return model_out
