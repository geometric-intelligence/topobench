"""Precompute the features based on KHop neighbourhoods transform."""

import torch
import torch_geometric


class PrecomputeKHopFeatures(torch_geometric.transforms.BaseTransform):
    r"""Class for precomputing the features of a k-hop neighbourhood features transform.

    A transform that computes an aggregation of injective transformations of the k-hop neighbourhood.

    Parameters
    ----------
    max_hop : int
        The maximum hop neighbourhood.
    complex_dim : int
        The maximum dimension of the complex to evaluate.
    use_initial_features : bool
        Whether to use the initial features as the 0-hop features.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(
        self,
        max_hop: int,
        complex_dim: int,
        use_initial_features: bool,
        **kwargs,
    ) -> None:
        super().__init__()
        self.type = "precompute_khop_features"
        self.complex_dim = complex_dim
        self.max_hop = (
            max_hop - 1
        )  # The 0-th hop is always the features themselves
        self.use_initial_features = use_initial_features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, max_hop={self.max_hop}, complex_dim={self.complex_dim}, use_initial_features={self.use_initial_features})"

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
        T = self.max_hop
        K = self.complex_dim
        B = [torch.abs(data[f"incidence_{i + 1}"]) for i in range(K)]
        UP = [torch.mm(B[i], B[i].T) for i in range(K)]
        DOWN = [torch.mm(B[i].T, B[i]) for i in range(K)]
        Bs = [B[i].to_dense() for i in range(K)]
        Bc = [B[i].T.to_dense() for i in range(K)]

        Bs_new = Bc  # [torch.ones_like(Bs[0])] * len(Bs)
        Bc_new = Bs  # [torch.ones_like(Bc[0])] * len(Bc)

        x_is = {}
        for i in range(K):
            if i == 0:
                x_is[i] = torch.ones((UP[i].size())[0], 1)
            else:
                x_is[i] = torch.ones((DOWN[i - 1].size())[0], 1)

        # Create a dictionary that stores the i-simplices, the
        # j-th hop features matrix
        x_all = {
            f"x{i}_{j}": torch.ones(1).float()
            for i in range(K)
            for j in range(T + 1)
        }

        # TODO Sometimes te normalizations results in 0
        # for non-connected simplices which blows up
        # Compute normalization matrices D_k
        for i in range(K):
            LS = torch.mm(UP[i], x_is[i]).flatten()
            D_i = LS  # - torch.diag(torch.abs(UP[i].to_dense())).flatten()
            # Check no zeroes that blow up the division
            D_i = torch.diagflat(torch.div(1, torch.sqrt(D_i + 1)))
            # TODO Check this is correct

            if i > 0:
                DOWN[i - 1] = torch.mm(torch.mm(D_i, DOWN[i - 1]), D_i)
                D_i_i_minus_1 = torch.mm(
                    torch.abs(Bc[i - 1]), x_is[i - 1]
                ).flatten()
                D_i_i_minus_1 = torch.diagflat(
                    torch.div(1, torch.sqrt(D_i_i_minus_1 + 1))
                )

                D_i_minus_1_i = torch.mm(Bs[i - 1], x_is[i]).flatten()
                D_i_minus_1_i = torch.diagflat(
                    torch.div(1, torch.sqrt(D_i_minus_1_i + 1))
                )

                # D_{i, i-1} B_{i-1} D_{i-1, i}
                Bs_new[i - 1] = torch.mm(
                    torch.mm(D_i_i_minus_1, Bc[i - 1]), D_i_minus_1_i
                )

            if i < (K - 1):
                D_i_i_plus_1 = torch.mm(Bs[i], x_is[i + 1]).flatten()
                D_i_i_plus_1 = torch.diagflat(
                    torch.div(1, torch.sqrt(D_i_i_plus_1 + 1))
                )

                D_i_plus_1_i = torch.mm(torch.abs(Bc[i]), x_is[i]).flatten()
                D_i_plus_1_i = torch.diagflat(
                    torch.div(1, torch.sqrt(D_i_plus_1_i + 1))
                )

                # D_{i, i-1} B_{i-1} D_{i-1, i}
                Bc_new[i] = torch.mm(
                    torch.mm(D_i_i_plus_1, Bs[i]), D_i_plus_1_i
                )
            # Update UP at the end
            UP[i] = torch.mm(torch.mm(D_i, UP[i]), D_i)

        # Set the information for the 0-hop embeddings
        # if the initial features are not to be used
        # then the normalization is done over one vectors
        for i in range(K):
            if self.use_initial_features:
                x_all[f"x{i}_0"] = data[f"x_{i}"].float()
            else:
                x_all[f"x{i}_0"] = torch.ones_like(data[f"x_{i}"]).float()

        # For each hop t=1,...,T
        for t in range(1, T + 1):
            # For each k-simplex, k=0,...,K
            for k in range(K):
                # Set everything to `None` as
                # some representations are not available
                Y_U_1 = None
                Y_L_1 = None
                Y_B_1 = None
                Y_C_1 = None

                adjacencies_embedding = []
                # The highest order simplex does not have a higher order adjacency
                Y_U_1 = torch.mm(UP[k], x_all[f"x{k}_{t - 1}"])
                adjacencies_embedding.append(Y_U_1)
                if k < (K - 1):
                    Y_C_1 = torch.mm(Bc_new[k], x_all[f"x{k + 1}_{t - 1}"])

                    adjacencies_embedding.append(Y_C_1)
                # The 0-simplex does not have a lower-adjacency
                if k > 0:
                    Y_L_1 = torch.mm(DOWN[k - 1], x_all[f"x{k}_{t - 1}"])
                    Y_B_1 = torch.mm(Bs_new[k - 1], x_all[f"x{k - 1}_{t - 1}"])
                    adjacencies_embedding.append(Y_L_1)
                    adjacencies_embedding.append(Y_B_1)

                # Concat the feature dimension of the adjacy normalized features
                X = torch.cat(adjacencies_embedding, dim=1)
                x_all[f"x{k}_{t}"] = X
        return torch_geometric.data.Data(**data, **x_all)
