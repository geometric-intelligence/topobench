"""TopoBench boundary for ESC-GNN batches."""

from torch_geometric.data import Data

from topobench.data.utils.esc import require_esc_tensors, validate_esc_tensors
from topobench.nn.wrappers.base import AbstractWrapper


class ESCGNNWrapper(AbstractWrapper):
    """Pass encoded graphs through ESC-GNN.

    Cache gets checked again at model boundary. Output keeps TopoBench
    rank-zero shape.

    Parameters
    ----------
    backbone : torch.nn.Module
        Configured ESC-GNN backbone.
    **kwargs : Any
        Wrapper metadata, including ``out_channels`` and cell dims.

    Notes
    -----
    Outer residual stays off. Backbone already joins input and three GINE
    states.

    Examples
    --------
    >>> from topobench.nn.backbones.graph.esc_gnn import ESCGNN
    >>> wrapper = ESCGNNWrapper(
    ...     ESCGNN(),
    ...     out_channels=64,
    ...     num_cell_dimensions=1,
    ...     residual_connections=False,
    ... )
    >>> wrapper.residual_connections
    False
    """

    def __init__(self, backbone, **kwargs) -> None:
        super().__init__(backbone, **kwargs)
        if not self.residual_connections:
            for dimension in self.dimensions:
                layer_name = f"ln_{dimension}"
                if hasattr(self, layer_name):
                    delattr(self, layer_name)

    def forward(self, batch: Data) -> dict:
        """Run backbone and return rank-zero output.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Encoded batch with ``x_0``, ``batch_0``, labels, edges, and cache.

        Returns
        -------
        dict
            ``labels``, ``batch_0``, and updated ``x_0`` feats.

        Raises
        ------
        ValueError
            When cache, encoded feats, or batch vector is missing or bad.
        """
        esc_tensors = require_esc_tensors(batch, context="ESCGNNWrapper")
        validate_esc_tensors(
            *esc_tensors,
            num_structural_codes=self.backbone.num_structural_codes,
            context="ESCGNNWrapper",
        )
        if batch.get("x_0") is None:
            raise ValueError(
                "ESCGNNWrapper: missing encoded node features x_0"
            )
        if batch.get("batch_0") is None:
            raise ValueError(
                "ESCGNNWrapper: missing node batch vector batch_0"
            )

        x_0 = self.backbone(batch.x_0, *esc_tensors)
        return {"labels": batch.y, "batch_0": batch.batch_0, "x_0": x_0}


__all__ = ["ESCGNNWrapper"]
