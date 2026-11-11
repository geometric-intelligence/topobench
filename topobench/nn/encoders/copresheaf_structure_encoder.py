"""Feature encoder for copresheaf combinatorial-complex inputs."""

import torch
import torch_geometric

from topobench.nn.encoders.all_cell_encoder import BaseEncoder
from topobench.nn.encoders.base import AbstractFeatureEncoder


class CopresheafStructureFeatureEncoder(AbstractFeatureEncoder):
    """Encode cell signals together with scale-stable structural statistics.

    The submitted graph-to-combinatorial lifting stores four log-scaled
    statistics in
    ``structure_{rank}``: cell size, lower incidence degree, same-rank degree,
    and upper incidence degree. Keeping these separate from ``x_{rank}``
    preserves TopoBench's input-channel inference while allowing the model to
    retain multiplicity information lost by mean feature projection.

    Parameters
    ----------
    in_channels : list[int]
        Input feature dimension for each cell rank, before any
        structural channels are appended.
    out_channels : int
        Output feature dimension shared by every encoded rank.
    structure_channels : int, optional
        Number of columns in each ``structure_{rank}`` tensor
        (default: 4).
    proj_dropout : float, optional
        Dropout used inside each per-rank ``BaseEncoder`` (default: 0.0).
    selected_dimensions : list[int], optional
        Cell ranks to encode (default: all ranks in ``in_channels``).
    use_structure : bool, optional
        If ``True``, concatenate the structural statistics to the raw
        features before projecting (default: True).
    **kwargs : dict, optional
        Additional keyword arguments.
    """

    def __init__(
        self,
        in_channels,
        out_channels: int,
        structure_channels: int = 4,
        proj_dropout: float = 0.0,
        selected_dimensions=None,
        use_structure: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.structure_channels = structure_channels
        self.use_structure = use_structure
        self.dimensions = (
            selected_dimensions
            if selected_dimensions is not None
            else range(len(in_channels))
        )
        extra_channels = structure_channels if use_structure else 0
        for rank in self.dimensions:
            setattr(
                self,
                f"encoder_{rank}",
                BaseEncoder(
                    in_channels[rank] + extra_channels,
                    out_channels,
                    dropout=proj_dropout,
                ),
            )

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        """Encode every selected cell rank in place.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data object which should contain ``x_{rank}`` and,
            when ``use_structure`` is set, ``structure_{rank}``
            features for each selected rank.

        Returns
        -------
        torch_geometric.data.Data
            Data object with updated ``x_{rank}`` features.
        """
        if not hasattr(data, "x_0"):
            data.x_0 = data.x
        for rank in self.dimensions:
            feature_key = f"x_{rank}"
            if feature_key not in data:
                continue
            features = data[feature_key]
            if self.use_structure:
                structure_key = f"structure_{rank}"
                if structure_key not in data:
                    raise KeyError(
                        f"missing structural features {structure_key!r}"
                    )
                structure = data[structure_key].to(
                    device=features.device, dtype=features.dtype
                )
                if structure.size(1) != self.structure_channels:
                    raise ValueError(
                        f"expected {self.structure_channels} structural "
                        f"channels at rank {rank}, got {structure.size(1)}"
                    )
                features = torch.cat((features, structure), dim=-1)
            data[feature_key] = getattr(self, f"encoder_{rank}")(
                features, data[f"batch_{rank}"]
            )
        return data
