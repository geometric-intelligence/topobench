"""Wrapper for the DirSNN model."""

import torch

from topobench.nn.wrappers.base import AbstractWrapper


class DirSNNWrapper(AbstractWrapper):
    r"""Wrapper for the DirSNN model.

    This wrapper defines the forward pass of the DirSNN backbone within the
    TopoBench pipeline. DirSNN operates exclusively on 1-simplex (edge)
    features using directed incidence matrices incidence_1 and incidence_2.

    Following the pattern established by SANWrapper, the 0-cell (node)
    embeddings required by the downstream readout are computed via signal
    down-propagation: x_0 = incidence_1 @ x_1. This matches the
    PropagateSignalDown readout strategy recommended for models that do not
    internally update 0-cell representations (TopoBench ablation, Table 2).

    Initialization parameters (backbone, out_channels, num_cell_dimensions)
    are handled by the parent AbstractWrapper.__init__.
    """

    def forward(self, batch):
        r"""Forward pass for the DirSNN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing x_1, incidence_1, incidence_2, y, batch_0.

        Returns
        -------
        dict
            Dictionary with keys: labels, batch_0, x_0, x_1.
        """
        x_1 = self.backbone(batch)
        x_0 = torch.sparse.mm(batch.incidence_1, x_1)
        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        return model_out
