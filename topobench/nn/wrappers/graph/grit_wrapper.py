"""Wrapper for the GRIT backbone."""

from topobench.nn.wrappers.base import AbstractWrapper


class GRITWrapper(AbstractWrapper):
    r"""Wrapper for the GRIT backbone.

    This wrapper defines the forward pass of the model. Besides the node
    features and connectivity, it forwards the RRWP positional encodings
    (``rrwp``, ``rrwp_index``, ``rrwp_val``) and the log-degrees
    (``log_deg``) attached to the batch by the ``AddRRWP`` transform. When
    these attributes are absent, the backbone computes them on the fly.
    The GRIT backbone returns the embeddings of the cells of rank 0.
    """

    def forward(self, batch):
        r"""Forward pass for the GRIT wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_0 = self.backbone(
            batch.x_0,
            batch.edge_index,
            batch=batch.batch_0,
            edge_attr=batch.get("edge_attr", None),
            rrwp=batch.get("rrwp", None),
            rrwp_index=batch.get("rrwp_index", None),
            rrwp_val=batch.get("rrwp_val", None),
            log_deg=batch.get("log_deg", None),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0

        return model_out
