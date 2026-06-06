"""Transform that performs barycentric subdivision on a simplicial complex."""

import torch
import torch_geometric
from topomodelx.utils.sparse import from_sparse
from toponetx import SimplicialComplex

from topobench.data.utils import data2simplicial
from topobench.data.utils.utils import get_complex_connectivity


class BarycentricSubdivisionTransform(
    torch_geometric.transforms.BaseTransform
):
    r"""A transform that performs barycentric subdivision on a simplicial complex.

    The barycentric subdivision of a simplicial complex K is a new simplicial complex Sd(K)
    where each simplex in K is replaced by a collection of simplices, resulting in a finer
    triangulation of the underlying space.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the base transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "domain2domain"
        self.parameters = kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the barycentric subdivision to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data, expected to contain a simplicial complex.

        Returns
        -------
        torch_geometric.data.Data
            The data with a subdivided simplicial complex.
        """
        keys_to_keep = self.parameters["keys_to_keep"]
        # Transform torch_geometric.data.Data to simplicial complex
        simplicial_complex = data2simplicial(data)

        # Apply barycentric subdivision
        subdivided_complex, simplex_to_index = self._barycentric_subdivision(
            simplicial_complex
        )

        lifted_topology = get_complex_connectivity(
            subdivided_complex,
            self.parameters["complex_dim"],
            neighborhoods=self.parameters["neighborhoods"],
            signed=self.parameters["signed"],
        )

        # Get rid of the old keys
        for key in list(data.keys()):
            if key not in keys_to_keep:
                data.pop(key)

        # Assign new topology
        for key in lifted_topology:
            data[key] = lifted_topology[key]

        # Use shape from lifted_topology to get cell counts
        zero_cells, one_cells, two_cells, three_cells = lifted_topology[
            "shape"
        ]

        data["shape"] = torch.tensor(
            [zero_cells, one_cells, two_cells, three_cells]
        )
        for idx, n in enumerate(data["shape"]):
            if idx == 0:
                data["x"] = torch.ones((n, 1))
            if n > 0:
                data[f"x_{idx}"] = torch.ones((n, 1))

        data["edge_index"] = torch.Tensor(
            from_sparse(subdivided_complex.adjacency_matrix(rank=0)).indices()
        )

        return data

    def _barycentric_subdivision(self, K: SimplicialComplex) -> tuple:
        """Perform barycentric subdivision on a simplicial complex.

        Parameters
        ----------
        K : SimplicialComplex
            The input simplicial complex.

        Returns
        -------
        tuple
            A tuple containing the subdivided simplicial complex and a mapping from
            simplices to indices.
        """
        # Create a new SimplicialComplex to store the subdivision
        Sd_K = SimplicialComplex()

        # Check if K has the required attributes
        if not hasattr(K, "simplices"):
            raise AttributeError(
                "The simplicial complex must have a 'simplices' attribute"
            )
        if not hasattr(K, "dim"):
            raise AttributeError(
                "The simplicial complex must have a 'dim' property"
            )

        new_simplices = {dim: set() for dim in range(K.dim + 1)}

        # Add new vertices to Sd_K. Each simplex of Sd_K is a chain of simplices of K
        for simplex in K.simplices:
            new_simplices[0].add((simplex,))

        # Give now an index to each simplex
        simplex_to_index = {
            simplex[0]: i for i, simplex in enumerate(new_simplices[0])
        }

        # Now, we add simplices from dimension 1 to K.dim
        for dim in range(1, K.dim + 1):
            # Get all simplices of the previous dimension, and try to add more simplices to the chain
            previous_simplices = new_simplices[dim - 1]
            for simplex_sub in previous_simplices:
                last_simplex = simplex_sub[-1]
                for simplex in K.simplices:
                    # Check if simplex is a face of simplex_sub
                    # Note: The '<' operator is assumed to check if last_simplex is a face of simplex
                    if last_simplex < simplex:
                        new_simplices[dim].add(simplex_sub + (simplex,))

        # Now convert the simplices to indexes
        all_simplices = [
            [simplex_to_index[or_simplex] for or_simplex in simplex]
            for dim in range(K.dim + 1)
            for simplex in new_simplices[dim]
        ]

        # Add the simplices to the new SimplicialComplex
        Sd_K.add_simplices_from(all_simplices)

        return Sd_K, simplex_to_index
