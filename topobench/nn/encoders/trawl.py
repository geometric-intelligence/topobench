"""Adapter for using existing TopoBench rank-feature encoders before TRAWL."""

from topobench.nn.encoders.base import AbstractFeatureEncoder


class TRAWLFeatureEncoder(AbstractFeatureEncoder):
    """Synchronize a native encoder's x_r outputs with TRAWL state features.

    When using this adapter, set backbone.in_channels to the encoder's output
    widths; input shape inference runs before learned feature encoding.

    Parameters
    ----------
    encoder : torch.nn.Module
        TopoBench feature encoder producing ``x_r`` outputs.
    ranks : iterable of int
        Ranks whose encoded features replace ``trawl_signal_r``.
    """

    def __init__(self, encoder, ranks):
        super().__init__()
        self.encoder, self.ranks = encoder, list(ranks)

    def forward(self, batch):
        """Encode features and copy them into TRAWL state signals.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Input batch.

        Returns
        -------
        torch_geometric.data.Batch
            Batch with encoded ``x_r`` and matching ``trawl_signal_r``.
        """
        batch = self.encoder(batch)
        for rank in self.ranks:
            batch[f"trawl_signal_{rank}"] = batch[f"x_{rank}"]
        return batch
