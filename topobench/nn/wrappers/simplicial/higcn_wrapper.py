"""Wrapper for the HiGCN model."""

from topobench.nn.wrappers.base import AbstractWrapper


class HiGCNWrapper(AbstractWrapper):
    r"""Wrapper for the HiGCN model.

    HiGCN is a node-centric higher-order model: it carries features on the
    nodes (rank 0) and uses the higher-order simplicial structure only through
    the Flower-Petals Laplacians, which the backbone reconstructs from the
    incidence matrices. This wrapper therefore feeds the node features together
    with the simplicial incidence matrices to the backbone and returns the
    updated node embeddings.
    """

    def forward(self, batch):
        r"""Forward pass for the HiGCN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        incidences = (batch.incidence_1, batch.incidence_2)
        x_0 = self.backbone(batch.x_0, incidences)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0

        return model_out
