"""Readout layer that does not perform any operation on the node embeddings."""

import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class NoReadOut(AbstractZeroCellReadOut):
    r"""No readout layer.

    This readout layer does not perform any operation on the node embeddings.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(self, **kwargs):
        self.name = "NoReadOut"
        super().__init__(**kwargs)

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass of the no readout layer.

        It returns the model output without any modification.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output.
        """
        return model_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
