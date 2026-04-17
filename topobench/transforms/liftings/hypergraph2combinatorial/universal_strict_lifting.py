"""Universal strict lifting of hypergraphs to combinatorial complexes."""

import torch
import torch_geometric
from toponetx.classes import CombinatorialComplex

from topobench.transforms.liftings.hypergraph2combinatorial.base import (
    Hypergraph2CombinatorialLifting,
)


class UniversalStrictLifting(Hypergraph2CombinatorialLifting):
    r"""Lift hypergraphs to combinatorial complexes.

    It works by assigning the smallest rank values such that subcells of any cell have strictly smaller rank.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def lift_topology(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data | dict:
        r"""Lift the topology of a hypergraph to a combinatorial complex.

        It works by setting the rank of a hyperedge equal to the maximum of the ranks of its sub-hyperedges plus 1.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        # Incidence matrix is transposed for easier handling of hyperedges
        incidence_hyperedges = data["incidence_hyperedges"].t().coalesce()
        incidence_indices = incidence_hyperedges.indices()
        num_hyperedges = incidence_hyperedges.size()[0]

        # Create a list of pairs (he_start, he_length) sorted by the second entry
        sorted_indices = self._sorted_hyperedge_indices(incidence_indices[0])

        combinatorial_complex = CombinatorialComplex()

        # Initialize all ranks to 1
        ranks = torch.zeros(num_hyperedges)

        # Assign a rank to each hyperedge
        for i, (he_start, he_length) in enumerate(sorted_indices):
            hyperedge = set(
                node.item()
                for node in incidence_indices[1][
                    he_start : he_start + he_length
                ]
            )
            # If the hyperedge has at least two nodes, set its rank to at least 1
            # Otherwise CombinatorialComplex will complain
            if len(hyperedge) >= 2:
                ranks[i] = 1

            # Iterate over sub-hyperedges
            for j in range(i):
                sub_he_start, sub_he_length = sorted_indices[j]
                subhyperedge = set(
                    node.item()
                    for node in incidence_indices[1][
                        sub_he_start : sub_he_start + sub_he_length
                    ]
                )

                # Set the rank of the hyperedge to 1 + max(ranks of subhyperedges)
                if subhyperedge < hyperedge:
                    ranks[i] = max(ranks[i], ranks[j] + 1)

            if ranks[i] == 0:
                # If the rank is still 0, the hyperedge is a node, so we add the features.
                combinatorial_complex.add_cell(
                    hyperedge, 0
                )  # , **{'x_0': data.x[next(iter(hyperedge)),:]})
            else:
                combinatorial_complex.add_cell(hyperedge, int(ranks[i]))

        lifted_topology = self._get_lifted_topology(combinatorial_complex)

        # Feature liftings
        lifted_topology["x_0"] = data.x

        return lifted_topology

    def _sorted_hyperedge_indices(self, hyperedges):
        """
        Create a list of pairs with the starts and lengths of hyperedges in ascending order of hyperedge size.

        Parameters
        ----------
        hyperedges : torch.tensor
            A tensor with two rows: the first one for hyperedge indices, the second one for node indices.

        Returns
        -------
        list
            A list of pairs (start, length) sorted according to length (ascending).
        """
        # Identify where the changes occur
        changes = torch.cat(
            [torch.tensor([True]), hyperedges[1:] != hyperedges[:-1]]
        )
        change_indices = torch.where(changes)[0]

        # Calculate the size of each hyperedge
        lengths = change_indices[1:] - change_indices[:-1]
        lengths = torch.cat(
            [lengths, torch.tensor([len(hyperedges) - change_indices[-1]])]
        )

        # Sort the list according to the lengths entry
        indices = list(
            zip(change_indices.tolist(), lengths.tolist(), strict=False)
        )
        indices.sort(key=lambda x: x[1])

        return indices
