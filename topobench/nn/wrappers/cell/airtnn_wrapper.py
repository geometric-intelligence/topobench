"""Wrapper for the AirTNN model."""

import torch

from topobench.nn.wrappers.base import AbstractWrapper


class AirTNNWrapper(AbstractWrapper):
    r"""Wrapper for the AirTNN model.

    Mirrors the sibling cell wrappers: the backbone processes the rank-1
    (edge) signal over the lower/upper neighbourhood operators, per Sec. 3
    of arXiv:2502.10070; rank-0 embeddings are obtained by pushing the edge
    embeddings down through ``incidence_1``.
    """

    def forward(self, batch):
        r"""Forward pass for the AirTNN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        x_1 = self.backbone(
            batch.x_1,
            batch.down_laplacian_1.coalesce(),
            batch.up_laplacian_1.coalesce(),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}

        model_out["x_1"] = x_1
        model_out["x_0"] = torch.sparse.mm(batch.incidence_1, x_1)
        return model_out
