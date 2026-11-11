"""TopoBench wrapper for the combinatorial copresheaf backbone."""

from topobench.nn.wrappers.base import AbstractWrapper


class CopresheafCCWrapper(AbstractWrapper):
    """Extract rank features and neighborhoods from a TopoBench batch."""

    def forward(self, batch):
        """Run higher-order copresheaf message passing on ``batch``.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batched combinatorial-complex data with per-rank features
            ``x_{rank}``, per-rank batch indices ``batch_{rank}``, and one
            connectivity tensor per configured neighborhood.

        Returns
        -------
        dict
            Per-rank output features, per-rank batch indices, and labels,
            following the TopoBench wrapper contract.
        """
        features = {rank: batch[f"x_{rank}"] for rank in self.backbone.ranks}
        connectivities = {
            name: batch[name] for name in self.backbone.neighborhoods
        }
        output = self.backbone(features, connectivities)

        model_out = {f"x_{rank}": value for rank, value in output.items()}
        model_out.update(
            {f"batch_{rank}": batch[f"batch_{rank}"] for rank in output}
        )
        model_out["labels"] = batch.y
        return model_out
