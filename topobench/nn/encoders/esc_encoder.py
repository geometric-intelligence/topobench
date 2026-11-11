"""Node feat encoding with an ESC cache check."""

from typing import Any

from torch_geometric.data import Data

from topobench.data.utils.esc import (
    ESC_NUM_STRUCTURAL_CODES,
    require_esc_tensors,
    validate_esc_tensors,
)
from topobench.nn.encoders.all_cell_encoder import AllCellFeatureEncoder


class ESCFeatureEncoder(AllCellFeatureEncoder):
    """Encode node feats after checking ESC cache.

    Uses :class:`AllCellFeatureEncoder` for projection. Cache gets checked
    here, and no histogram rebuild.

    Parameters
    ----------
    in_channels : list[int]
        Input widths for selected cell dims.
    out_channels : int
        Encoded node width.
    structural_codebook_size : int, optional
        ESC vocabulary size. Must be ``387``.
    proj_dropout : float, optional
        Dropout for TopoBench feat projection.
    selected_dimensions : list[int], optional
        Cell dims to encode.
    **kwargs : Any
        Extra TopoBench encoder metadata.

    Notes
    -----
    Codebook stays at 387 so cache, embedding, and checks cannot drift.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        structural_codebook_size: int = ESC_NUM_STRUCTURAL_CODES,
        proj_dropout: float = 0.0,
        selected_dimensions=None,
        **kwargs: Any,
    ) -> None:
        if structural_codebook_size != ESC_NUM_STRUCTURAL_CODES:
            raise ValueError(
                "ESCFeatureEncoder requires structural_codebook_size=387"
            )
        self.structural_codebook_size = structural_codebook_size
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            proj_dropout=proj_dropout,
            selected_dimensions=selected_dimensions,
            **kwargs,
        )

    def forward(self, data: Data) -> Data:
        """Check cache, then encode node feats.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph with raw node feats, ``edge_index``, and ESC cache.

        Returns
        -------
        torch_geometric.data.Data
            Same data with encoded feats in ``x_0``.

        Raises
        ------
        ValueError
            When ESC cache is missing or malformed.
        """
        esc_tensors = require_esc_tensors(data, context="ESCFeatureEncoder")
        validate_esc_tensors(
            *esc_tensors,
            num_structural_codes=self.structural_codebook_size,
            context="ESCFeatureEncoder",
        )
        return super().forward(data)


__all__ = ["ESCFeatureEncoder"]
