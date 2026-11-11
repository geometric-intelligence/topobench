"""Readout for OGFormer exposing unmasked node logits."""

import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class OGFormerReadOut(AbstractZeroCellReadOut):
    r"""Readout layer for OGFormer.

    Behaves like :class:`topobench.nn.readouts.NoReadOut` but additionally
    keeps a reference to the logits over *all* nodes under the
    ``node_logits`` key. In transductive node classification the trainer
    masks ``logits`` down to the current split, while
    :class:`topobench.loss.model.OGFormerLoss` needs full-graph
    predictions to build pseudo-labels for unlabeled nodes
    (:math:`Y_{train} \| Y_{pred}` in Eq. (15)).

    Parameters
    ----------
    **kwargs : dict, optional
        Additional keyword arguments forwarded to the base readout.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass of the OGFormer readout layer.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output with logits and their
            unmasked ``node_logits`` alias.
        """
        model_out["logits"] = self.compute_logits(
            model_out["x_0"], batch["batch_0"]
        )
        model_out["node_logits"] = model_out["logits"]
        return model_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
