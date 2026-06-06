"""Wrapper for the HOPSE model."""

from topobench.nn.wrappers.base import AbstractWrapper


class HOPSEWrapper(AbstractWrapper):
    r"""Wrapper for the HOPSE.

    Parameters
    ----------
    backbone : torch.nn.Module
        Backbone model.
    **kwargs : dict
        Additional arguments for the class. It should contain the following keys:
        - out_channels (int): Number of output channels.
        - num_cell_dimensions (int): Number of cell dimensions.
    """

    def __init__(self, backbone, **kwargs):
        super().__init__(backbone, **kwargs)
        self.complex_dim = kwargs["complex_dim"]
        self.max_hop = kwargs["max_hop"]

    def __call__(self, batch):
        r"""Forward pass for the model.

        This method calls the forward method and adds the residual connection.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        model_out = self.forward(batch)
        return model_out

    def forward(self, batch):
        """Forward pass of the HOPSE.

        Parameters
        ----------
        batch : Dict
            Dictionary containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        # Prepare the input data for the backbone
        # by aggregating the data in a dictionary
        # (source_simplex_dim, (target_simplex_dim, torch.Tensor with embeddings))
        x_all = tuple(
            tuple(batch[f"x{i}_{j}"] for j in range(self.max_hop))
            for i in range(self.complex_dim + 1)
        )

        x_out = self.backbone(x_all)

        # Get all the batch tensors according to the max_simplex_dim
        model_out = {
            f"batch_{i}": batch[f"batch_{i}"]
            for i in range(self.complex_dim + 1)
        }
        # Add the target labels
        model_out["labels"] = batch.y

        for cell_idx in range(self.complex_dim + 1):
            for hop_idx in range(
                self.max_hop
            ):  # when I run HOPSE_classic range(max_hop_dim) but when I run HOPSE_zero range(max_hop_dim + 1)
                model_out[f"x{cell_idx}_{hop_idx}"] = x_out[cell_idx][hop_idx]
        return model_out
