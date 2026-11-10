"""Wrapper for the TopoU-Net model."""

from topobench.nn.wrappers.base import AbstractWrapper


class TopoUNetWrapper(AbstractWrapper):
    r"""Wrapper for the TopoU-Net model (arXiv:2605.10091).

    Routes the rank-0 cochain and the consecutive incidence matrices
    :math:`B_{r-1,r}` of the batched combinatorial complex into the
    TopoU-Net backbone, and exposes the decoder states of every rank of the
    encoder rank path to the readout. Following Definition 3.3 of the
    paper, the model consumes a single input cochain at the input rank
    :math:`s_0`; all higher-rank states are computed by the encoder through
    incidence transport.
    """

    def forward(self, batch):
        r"""Forward pass for the TopoU-Net wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """
        path = self.backbone.encoder_rank_path
        x_all = {path[0]: batch[f"x_{path[0]}"]}
        incidence_all = {
            rank: batch[f"incidence_{rank}"] for rank in range(1, path[-1] + 1)
        }

        decoder_states = self.backbone(x_all, incidence_all)

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        for rank, x in decoder_states.items():
            model_out[f"x_{rank}"] = x
        return model_out
