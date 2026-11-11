"""Wrapper for the SMCN model."""

import torch

from topobench.nn.wrappers.base import AbstractWrapper


class SMCNWrapper(AbstractWrapper):
    r"""Wrapper for the SMCN model.

    This wrapper defines the forward pass of the model. The SMCN model
    returns the embeddings of the cells of rank 0, 1, and 2.
    """

    def forward(self, batch):
        r"""Forward pass for the SMCN wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        def _batch_vector(rank, size):
            """Return the batch vector of a rank, or zeros if absent.

            Parameters
            ----------
            rank : int
                Cell rank whose batch vector is requested.
            size : int
                Number of cells of that rank in the batch.

            Returns
            -------
            torch.Tensor
                Long tensor assigning each cell to its graph.
            """
            vec = batch.get(f"batch_{rank}", None)
            if not torch.is_tensor(vec):
                vec = torch.zeros(
                    size, dtype=torch.long, device=batch.x_0.device
                )
            return vec

        x_0, x_1, x_2 = self.backbone(
            batch.x_0,
            batch.x_1,
            batch.x_2,
            batch.incidence_1,
            batch.incidence_2,
            _batch_vector(0, batch.x_0.size(0)),
            _batch_vector(1, batch.x_1.size(0)),
            _batch_vector(2, batch.x_2.size(0)),
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0
        model_out["x_1"] = x_1
        model_out["x_2"] = x_2
        return model_out
