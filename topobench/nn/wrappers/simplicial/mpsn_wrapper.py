"""Wrapper for the MPSN model."""

from topobench.nn.wrappers.base import AbstractWrapper


class MPSNWrapper(AbstractWrapper):
    r"""Wrapper for Message Passing Simplicial Networks (MPSN).

    Feeds the node/edge/face embeddings and the boundary (incidence) matrices
    :math:`B_1` and :math:`B_2` to the backbone, which derives the four MPSN
    adjacencies internally, and returns the embeddings of the cells of rank
    0, 1 and 2.
    """

    def forward(self, batch):
        r"""Forward pass for the MPSN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_all = (batch.x_0, batch.x_1, batch.x_2)
        incidence_all = (batch.incidence_1, batch.incidence_2)
        x_0, x_1, x_2 = self.backbone(x_all, incidence_all)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2

        return model_out
