"""Rank-aware neighborhood routes for copresheaf message passing."""

from dataclasses import dataclass

import torch

from topobench.data.utils import get_routes_from_neighborhoods


@dataclass(frozen=True)
class CopresheafRoute:
    """A directed message route induced by one neighborhood function."""

    neighborhood: str
    source_rank: int
    target_rank: int

    @classmethod
    def from_neighborhood(cls, neighborhood: str) -> "CopresheafRoute":
        """Parse a TopoBench neighborhood name into its directed ranks.

        Parameters
        ----------
        neighborhood : str
            TopoBench neighborhood name, for example ``up_incidence-0``.

        Returns
        -------
        CopresheafRoute
            The route induced by the named neighborhood.
        """
        try:
            source_rank, target_rank = get_routes_from_neighborhoods(
                [neighborhood]
            )[0]
        except (IndexError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid TopoBench neighborhood name: {neighborhood!r}"
            ) from error
        return cls(neighborhood, source_rank, target_rank)

    def edge_index(
        self, connectivity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a sparse connectivity matrix to ``source -> target`` edges.

        TopoBench's selected neighborhood matrices consistently store target
        cells on rows and source cells on columns. The returned PyG-style
        index therefore reverses the matrix indices.

        Parameters
        ----------
        connectivity : torch.Tensor
            Sparse or dense neighborhood matrix with target cells on rows
            and source cells on columns.

        Returns
        -------
        tuple of torch.Tensor
            A ``[2, num_edges]`` PyG-style ``(source, target)`` edge index
            and the corresponding edge weights.
        """
        if not connectivity.is_sparse:
            connectivity = connectivity.to_sparse()
        connectivity = connectivity.coalesce()
        row, column = connectivity.indices()

        source, target = column, row
        edge_index = torch.stack((source, target), dim=0)
        return edge_index, connectivity.values().abs()
