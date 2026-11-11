"""Wrapper for the GSN backbone models."""

from topobench.nn.wrappers.base import AbstractWrapper


class GSNWrapper(AbstractWrapper):
    r"""Wrapper for the GSN backbone models.

    The GSN backbones take tensors (``edge_index``, ``x``, ``gsn_embeddings``,
    ...) directly, mirroring their per-layer signature. This wrapper is the
    adapter that pulls those fields off the TopoBench batch: node features from
    ``x_0`` if the cell machinery populated it, else the raw ``x`` that plain
    graph datasets carry; the structural encodings attached by
    ``GSNFeatureEncoder`` from ``backbone.gsn_kword`` (e.g.
    ``node_gsn_encodings``); and the node-to-graph assignment from ``batch_0``.
    """

    def forward(self, batch):
        r"""Forward pass for the GSN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data. It must carry the
            structural encodings under ``backbone.gsn_kword``, node features
            under ``x_0`` or ``x``, and ``batch_0``.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        gsn_embeddings = batch[self.backbone.gsn_kword]
        edge_attr = getattr(batch, "edge_attr", None)

        # rank-0 features are stored under `x_0` once the cell machinery
        # populates them; plain graph datasets only carry `x`.
        x = batch.x_0 if getattr(batch, "x_0", None) is not None else batch.x

        x_0 = self.backbone(
            batch.edge_index,
            x,
            gsn_embeddings,
            edge_attr=edge_attr,
            batch=batch.batch_0,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0

        return model_out
