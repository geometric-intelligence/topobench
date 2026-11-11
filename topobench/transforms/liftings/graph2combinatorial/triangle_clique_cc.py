"""Triangle-clique lifting of graphs to combinatorial complexes."""

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import CombinatorialComplex

from topobench.data.utils import (
    get_combinatorial_complex_connectivity,
    select_neighborhoods_of_interest,
)
from topobench.transforms.liftings.graph2combinatorial.base import (
    Graph2CombinatorialLifting,
)


class GraphTriangleCliqueCCLifting(Graph2CombinatorialLifting):
    r"""Lift graph triangles as explicit rank-2 combinatorial cells.

    Rank 0 is the input nodes, rank 1 is the input graph edges, and rank 2
    contains one cell for every graph triangle. The lift gives the submitted
    copresheaf model a deterministic topological domain while keeping the
    preprocessing small and easy to audit.
    """

    def lift_topology(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data | dict:
        """Lift one graph into a triangle combinatorial complex.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input graph data, expected to carry undirected edges
            and node features ``data.x``.

        Returns
        -------
        dict
            The lifted topology: rank-0/1/2 connectivity, node features,
            and structural features.
        """
        graph = self._generate_graph_from_data(data)
        if graph.is_directed():
            raise ValueError(
                "GraphTriangleCliqueCCLifting expects undirected graphs"
            )

        combinatorial_complex = CombinatorialComplex(graph)
        for triangle in self._triangles(graph):
            combinatorial_complex.add_cell(triangle, rank=2)

        lifted_topology = self._connectivity(combinatorial_complex)
        lifted_topology["x_0"] = data.x
        lifted_topology.update(
            self._structural_features(
                lifted_topology,
                dtype=data.x.dtype,
                device=data.x.device,
            )
        )
        return lifted_topology

    @staticmethod
    def _triangles(graph: nx.Graph) -> list[tuple[int, int, int]]:
        """Return sorted node triples that form graph triangles.

        Parameters
        ----------
        graph : nx.Graph
            The input undirected graph.

        Returns
        -------
        list of tuple of int
            Sorted node triples, one per triangle (3-clique) found
            in ``graph``.
        """
        triangles = set()
        for clique in nx.enumerate_all_cliques(graph):
            if len(clique) < 3:
                continue
            if len(clique) > 3:
                break
            triangles.add(tuple(sorted(clique)))
        return sorted(triangles)

    def _connectivity(self, combinatorial_complex) -> dict[str, torch.Tensor]:
        """Return the raw rank-0/1/2 connectivity of a complex.

        Parameters
        ----------
        combinatorial_complex : CombinatorialComplex
            The complex whose connectivity tensors are computed.

        Returns
        -------
        dict
            Connectivity tensors (incidence, adjacency, and
            Laplacian matrices) for ranks ``0`` through
            ``self.complex_dim``.
        """
        connectivity = get_combinatorial_complex_connectivity(
            combinatorial_complex,
            self.complex_dim,
            neighborhoods=None,
        )
        if self.neighborhoods is not None:
            connectivity.update(
                select_neighborhoods_of_interest(
                    connectivity, self.neighborhoods
                )
            )
        return connectivity

    def _structural_features(
        self,
        connectivity: dict[str, torch.Tensor],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Compute scale-stable size/degree features per complex rank.

        Parameters
        ----------
        connectivity : dict
            Rank connectivity tensors, including ``"shape"`` and,
            for each rank, ``"incidence_{rank}"`` and
            ``"adjacency_{rank}"``.
        dtype : torch.dtype
            Dtype used for the returned feature tensors.
        device : torch.device
            Device used for the returned feature tensors.

        Returns
        -------
        dict
            A ``"structure_{rank}"`` tensor for every rank from
            ``0`` to ``self.complex_dim``, each stacking log1p-
            scaled cell size, lower degree, same-rank degree, and
            upper degree.
        """
        shape = list(connectivity["shape"])
        output = {}
        node_membership = None
        for rank in range(self.complex_dim + 1):
            num_cells = int(shape[rank]) if rank < len(shape) else 0
            if rank == 0:
                cell_size = torch.ones(num_cells, dtype=dtype, device=device)
                lower_degree = torch.zeros(
                    num_cells, dtype=dtype, device=device
                )
                node_membership = torch.eye(
                    num_cells, dtype=dtype, device=device
                )
            else:
                incidence = (
                    connectivity[f"incidence_{rank}"].coalesce().to(device)
                )
                lower_degree = self._fit_length(
                    torch.sparse.sum(incidence, dim=0).to_dense().to(dtype),
                    num_cells,
                )
                if node_membership is None:
                    node_membership = torch.zeros(
                        shape[0], num_cells, dtype=dtype, device=device
                    )
                else:
                    node_membership = (
                        node_membership @ incidence.to_dense().abs().to(dtype)
                    ).clamp_max(1)
                cell_size = node_membership.sum(0)

            adjacency = connectivity[f"adjacency_{rank}"].coalesce().to(device)
            same_degree = self._fit_length(
                torch.sparse.sum(adjacency, dim=1).to_dense().to(dtype),
                num_cells,
            )
            if rank < self.complex_dim:
                upper_incidence = (
                    connectivity[f"incidence_{rank + 1}"].coalesce().to(device)
                )
                upper_degree = self._fit_length(
                    torch.sparse.sum(upper_incidence, dim=1)
                    .to_dense()
                    .to(dtype),
                    num_cells,
                )
            else:
                upper_degree = torch.zeros(
                    num_cells, dtype=dtype, device=device
                )

            output[f"structure_{rank}"] = torch.stack(
                (
                    torch.log1p(cell_size),
                    torch.log1p(lower_degree),
                    torch.log1p(same_degree),
                    torch.log1p(upper_degree),
                ),
                dim=-1,
            )
        return output

    @staticmethod
    def _fit_length(values: torch.Tensor, length: int) -> torch.Tensor:
        """Trim TopoNetX placeholder degrees or pad missing degrees.

        Parameters
        ----------
        values : torch.Tensor
            A 1-D tensor of degree values to resize.
        length : int
            Target number of entries.

        Returns
        -------
        torch.Tensor
            ``values`` trimmed to ``length`` entries, or right-
            padded with zeros if it has fewer than ``length``
            entries.
        """
        if values.numel() == length:
            return values
        if values.numel() > length:
            return values[:length]
        return torch.cat((values, values.new_zeros(length - values.numel())))
