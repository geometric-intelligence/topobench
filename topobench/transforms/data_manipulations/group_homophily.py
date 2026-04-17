"""A transform that canculates group combinatorial homophily of the input hypergraph."""

import torch
import torch_geometric
from scipy.special import comb


class GroupCombinatorialHomophily(torch_geometric.transforms.BaseTransform):
    r"""Calculates group combinatorial homophily of the input hypergraph.

    This transformation implements the methodology from the paper:
    "Combinatorial Characterizations and Impossibilities for Higher-order Homophily".
    It computes homophily metrics for hypergraphs by analyzing the relationship between
    node labels within hyperedges.

    Parameters
    ----------
    **kwargs : dict, optional
        Additional parameters for the transform.
        - top_k : int, default=3
            Number of top hyperedge cardinalities to analyze.

    Attributes
    ----------
    type : str
        Identifier for the transform type.
    top_k : int
        Number of top hyperedge cardinalities to analyze.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "calcualte_group_combinatorial_homophily"
        self.top_k = kwargs.get("top_k", 3)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r})"

    def forward(self, data: torch_geometric.data.Data):
        r"""Apply the transform to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        torch_geometric.data.Data
            The transformed data.
        """
        labels = data.y
        unique_labels, count_labels = torch.unique(labels, return_counts=True)
        # Crate a dictionary with the number of nodes in each class

        # {label: number of nodes in class}
        # DO WE NEED TO CALCULATE THIS FOR EACH K? OR once for all graph?
        unique_labels = dict(
            zip(unique_labels.numpy(), count_labels.numpy(), strict=False)
        )

        # Enhancement: avoid to_dense
        H = data.incidence_hyperedges.to_dense()
        he_cardinalities = H.sum(0)

        # DO WE NEED TO CALCULATE THIS FOR EACH K? OR once for all graph?
        class_node_idxs = {
            label: torch.where(labels == label)[0] for label in unique_labels
        }
        n_nodes = H.shape[0]

        unique_he, unique_he_counts = torch.unique(
            he_cardinalities, return_counts=True
        )
        idx_sorted = torch.argsort(unique_he_counts, descending=True, axis=0)[
            : self.top_k
        ]

        out = {}
        # temp_global_idx = 0
        for _, idx in enumerate(idx_sorted):
            max_k = int(unique_he[idx])
            num_he_size_k = int(unique_he_counts[idx])

            if max_k != 1:
                H_k = H[:, torch.where(he_cardinalities == max_k)[0]].clone()

                he_cardinalities_k = torch.tensor(H_k.sum(0), dtype=torch.long)
                Dt, D = self.calculate_D_matrix(
                    H_k,
                    labels,
                    he_cardinalities_k,
                    unique_labels,
                    class_node_idxs,
                )
                Bt = self.calculate_baseline_matrix(
                    he_cardinalities_k,
                    unique_labels,
                    class_node_idxs,
                    count_labels,
                    n_nodes,
                )

                out[f"he_card={max_k}"] = {
                    "D": D,
                    "Dt": Dt,
                    "Bt": Bt,
                    "num_hyperedges": num_he_size_k,
                }
        data["group_combinatorial_homophily"] = out
        return data

    def calculate_affinity_score(self, n_nodes, X_mod, t, k):
        """Calculate affinity score.

        Parameters
        ----------
        n_nodes : int
            Total number of nodes.
        X_mod : int
            Total number of nodes in a class.
        t : int
            Type-t degree.
        k : int
            Max hyperedge cardinality.

        Returns
        -------
        torch.Tensor
            The affinity matrix.
        """
        # Probability to extract t-1 nodes with the same label from target label nodes
        term_1 = comb(X_mod - 1, t - 1)

        # Probability to extract k-t nodes with different labels from rest of the nodes
        term_2 = comb(n_nodes - X_mod, k - t)

        # Probability to extract k-1 nodes from hypergraph
        term_3 = comb(n_nodes - 1, k - 1)

        return (term_1 * term_2) / term_3

    def calculate_D_matrix(
        self, H, labels, he_cardinalities, unique_labels, class_node_idxs
    ):
        """Calculate the degree matrices D and D_t for the hypergraph.

        Parameters
        ----------
        H : torch.Tensor
            Dense incidence matrix of the hypergraph.
        labels : torch.Tensor
            Node labels.
        he_cardinalities : torch.Tensor
            Cardinality of each hyperedge.
        unique_labels : dict
            Dictionary mapping labels to their counts.
        class_node_idxs : dict
            Dictionary mapping labels to node indices.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - D_t_class : Type-t degree distribution matrix for each class
            - D : Degree matrix counting same-label nodes in hyperedges
        """
        max_k = max(he_cardinalities)
        D = torch.zeros((H.shape[0], max_k))

        # Calculate D matrix
        for he, he_card in zip(H.T, he_cardinalities, strict=False):
            # Node indices belonging to hyperedge
            node_idxs = torch.where(he == 1)[0]

            # Node labels belonging to hyperedge
            he_lables = labels[node_idxs]

            # Go over all nodes in hyperedge
            for idx in range(he_card):
                # Extract node index and label
                node_idx, node_label = node_idxs[idx], he_lables[idx]

                # Check number of nodes with the same label in hyperedge
                n_same_class_nodes = (he_lables == node_label).sum()
                D[node_idx, n_same_class_nodes - 1] += 1

        # Calculate numerator of (1)
        D_t_class = torch.zeros((len(unique_labels), max_k))
        for unique_class in class_node_idxs:
            # Node indices belonging to current 'unique_class'
            node_idxs = class_node_idxs[unique_class]

            # Extract from D matrix only rows corresponding to nodes belonging to current 'unique_class'
            # Transpose to be aligned with paper
            D_class = D[node_idxs, :].T

            # Denominator of (1)
            d = D_class.sum()

            # Numerator of (1)
            # Sum over nodes to get type t degrees
            d_t = D_class.sum(1)
            # Calculate (1)
            h_t_class = d_t / d

            # Record (1) for a particular class
            D_t_class[unique_class, :] = h_t_class

        return D_t_class, D

    def calculate_baseline_matrix(
        self,
        he_cardinalities,
        unique_labels,
        class_node_idxs,
        count_labels,
        n_nodes,
    ):
        r"""Calculate the baseline affinity matrix for comparison.

        Parameters
        ----------
        he_cardinalities : torch.Tensor
            Cardinality of each hyperedge.
        unique_labels : dict
            Dictionary mapping labels to their counts.
        class_node_idxs : dict
            Dictionary mapping labels to node indices.
        count_labels : torch.Tensor
            Count of nodes for each label.
        n_nodes : int
            Total number of nodes in the hypergraph.

        Returns
        -------
        torch.Tensor
            Baseline matrix containing expected affinity scores for each class
            and degree type.
        """
        max_k = max(he_cardinalities)
        # Calculate matrix of affity scores
        B = torch.zeros((len(unique_labels), max_k))
        # Go over all nodes in hyperedge
        for unique_class in class_node_idxs:
            # Extract node index and label
            X_mod = count_labels[unique_class]
            for t in range(max_k):
                B[unique_class, t] = self.calculate_affinity_score(
                    n_nodes=n_nodes, X_mod=X_mod, t=t + 1, k=max_k
                )
        return B
