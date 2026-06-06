"""A transform that adds positional information to the graph."""

import torch
import torch_geometric
import torch_geometric.data
from torch_geometric.data import Data

from topobench.data.utils import get_routes_from_neighborhoods
from topobench.transforms.data_manipulations.all_encodings import (
    CombinedEncodings,
    SelectDestinationEncodings,
)


class dotdict(dict):
    """Dot.notation access to dictionary attributes."""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class HOPSE_PE_Information(torch_geometric.transforms.BaseTransform):
    r"""A transform that uses a positional and structural information added to the graph.

    Parameters
    ----------
    **kwargs : optional
        Parameters for the transform.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.type = "hopse_ps_information"
        self.parameters = kwargs

        self.max_rank = kwargs["max_rank"]
        self.copy_initial = kwargs["copy_initial"]
        self.neighborhoods = kwargs["neighborhoods"]
        self.encodings = kwargs["encodings"]
        self.dim_all_encodings = kwargs.get("dim_all_encodings", [])
        self.in_channels = kwargs["in_channels"]

        self.device = (
            "cpu" if kwargs["device"] == "cpu" else f"cuda:{kwargs['cuda'][0]}"
        )

        # Create combined encoding transform
        self.encoding_transform = CombinedEncodings(
            encodings=self.encodings,
            parameters=kwargs.get("parameters", {}),
        )
        self.select_dst_encodings = SelectDestinationEncodings(self.encodings)

        self.num_pe_considered = len(kwargs["encodings"])
        self.hidden_dim = self.parameters["dim_target_node"]

    def _data_to_device(self, data):
        """Move all tensors in a Data object to self.device.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data whose tensor attributes should be moved to
            ``self.device``. Non-tensor attributes are kept as-is.

        Returns
        -------
        torch_geometric.data.Data
            A new ``Data`` object with all tensor attributes on
            ``self.device``.
        """
        moved = {}
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                moved[key] = val.to(self.device)
            else:
                moved[key] = val
        return Data(**moved)

    def _make_zero_encoding_data(self, n_cells, encodings, dims):
        """Create a Data object with zero tensors for each encoding.

        Parameters
        ----------
        n_cells : int
            Number of cells (rows) for each encoding tensor.
        encodings : list[str]
            List of encoding names.
        dims : list[int]
            Dimension for each encoding. Must have same length as encodings.

        Returns
        -------
        torch_geometric.data.Data
            Data object with each encoding key mapped to a zero tensor
            of shape [n_cells, dims[i]].
        """
        zero_data = {
            enc: torch.zeros(
                (n_cells, dim),
                dtype=torch.float,
                device=self.device,
            )
            for enc, dim in zip(encodings, dims, strict=True)
        }
        return Data(**zero_data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.type!r}, parameters={self.parameters!r})"

    def intrarank_expand(self, params, src_rank, nbhd):
        """Expand the complex into an intrarank Hasse graph.

        Parameters
        ----------
        params : dict
            The parameters of the batch, containting the complex.
        src_rank : int
            The source rank.
        nbhd : str
            The neighborhood to use.

        Returns
        -------
        torch_geometric.data.Data
            The expanded batch of intrarank Hasse graphs for this route.
        """

        setattr(params, nbhd, getattr(params, nbhd).coalesce())

        batch_route = Data(
            x=getattr(params, f"x_{src_rank}"),
            edge_index=getattr(params, nbhd).indices(),
            edge_weight=getattr(params, nbhd).values().squeeze(),
            edge_attr=getattr(params, nbhd).values().squeeze(),
            requires_grad=True,
        )

        return batch_route

    def interrank_expand(self, params, src_rank, dst_rank, nbhd_cache):
        """Expand the complex into an interrank Hasse graph.

        Parameters
        ----------
        params : dict
            The parameters of the batch, containting the complex.
        src_rank : int
            The source rank.
        dst_rank : int
            The destination rank.
        nbhd_cache : dict
            The neighborhood cache containing the expanded boundary index and edge attributes.

        Returns
        -------
        torch_geometric.data.Data
            The expanded batch of interrank Hasse graphs for this route.
        """
        src_batch = params[f"x_{src_rank}"]
        dst_batch = params[f"x_{dst_rank}"]
        edge_index, edge_attr = nbhd_cache
        feat_on_dst = torch.zeros_like(
            getattr(params, f"x_{dst_rank}"), device=self.device
        )
        feat_on_src = getattr(params, f"x_{src_rank}").to(self.device)
        # Features on the source rank are more than the destination rank
        if feat_on_dst.shape[1] > src_batch.shape[1]:
            pad = (0, feat_on_dst.shape[1] - src_batch.shape[1])
            feat_on_src = torch.nn.functional.pad(
                feat_on_src, pad, "constant", 0
            )
            src_batch = torch.nn.functional.pad(src_batch, pad, "constant", 0)
        # Features on the source rank are more than the destination rank
        elif feat_on_dst.shape[1] < src_batch.shape[1]:
            pad = (0, feat_on_src.shape[1] - dst_batch.shape[1])
            feat_on_dst = torch.nn.functional.pad(
                feat_on_dst, pad, "constant", 0
            )
            dst_batch = torch.nn.functional.pad(dst_batch, pad, "constant", 0)

        x_in = torch.vstack([feat_on_dst, feat_on_src])
        batch_expanded = torch.cat([dst_batch, src_batch], dim=0)

        batch_route = Data(
            x=x_in.to(self.device),
            edge_index=edge_index.to(self.device),
            edge_attr=edge_attr.to(self.device),
            edge_weight=edge_attr.to(self.device),
            batch=batch_expanded.to(self.device),
        )

        return batch_route

    def aggregate_inter_nbhd(self, x_out_per_route):
        """Aggregate the outputs of the GNN for each rank.

        While the GNN takes care of intra-nbhd aggregation,
        this will take care of inter-nbhd aggregation.
        Default: sum.

        Parameters
        ----------
        x_out_per_route : dict
            The outputs of the GNN for each route.

        Returns
        -------
        dict
            The aggregated outputs of the GNN for each rank.
        """
        x_out_per_rank = {}
        for route_index, (_, dst_rank) in enumerate(self.routes):
            new_data = x_out_per_route[route_index]
            if dst_rank not in x_out_per_rank:
                x_out_per_rank[dst_rank] = new_data
            else:
                # Concatenate each encoding's features along dim=1 (feature dim)
                existing = x_out_per_rank[dst_rank]
                merged = {}
                for enc in self.encodings:
                    merged[enc] = torch.cat(
                        [existing[enc], new_data[enc]], dim=1
                    )
                x_out_per_rank[dst_rank] = Data(**merged)
        return x_out_per_rank

    def interrank_boundary_index(x_src, boundary_index, n_dst_nodes):
        """
        Recover lifted graph.

        Edge-to-node boundary relationships of a graph with n_nodes and n_edges
        can be represented as up-adjacency node relations. There are n_nodes+n_edges nodes in this lifted graph.
        Desgiend to work for regular (edge-to-node and face-to-edge) boundary relationships.

        Parameters
        ----------
        x_src : torch.tensor
            Source node features. Shape [n_src_nodes, n_features]. Should represent edge or face features.
        boundary_index : list of lists or list of tensors
            List boundary_index[0] stores node ids in the boundary of edge stored in boundary_index[1].
            List boundary_index[1] stores list of edges.
        n_dst_nodes : int
            Number of destination nodes.

        Returns
        -------
        edge_index : list of lists
            The edge_index[0][i] and edge_index[1][i] are the two nodes of edge i.
        edge_attr : tensor
            Edge features are given by feature of bounding node represnting an edge. Shape [n_edges, n_features].
        """
        node_ids = (
            boundary_index[0]
            if torch.is_tensor(boundary_index[0])
            else torch.tensor(boundary_index[0], dtype=torch.int32)
        )
        edge_ids = (
            boundary_index[1]
            if torch.is_tensor(boundary_index[1])
            else torch.tensor(boundary_index[1], dtype=torch.int32)
        )

        max_node_id = n_dst_nodes
        adjusted_edge_ids = edge_ids + max_node_id

        edge_index = torch.zeros(
            (2, node_ids.numel()), dtype=node_ids.dtype, device=x_src.device
        )
        edge_index[0, :] = node_ids
        edge_index[1, :] = adjusted_edge_ids

        edge_attr = x_src[edge_ids].squeeze()

        return edge_index, edge_attr

    def get_nbhd_cache(self, params):
        """Cache the nbhd information into a dict for the complex at hand.

        Parameters
        ----------
        params : dict
            The parameters of the batch, containing the complex.

        Returns
        -------
        dict
            The neighborhood cache.
        """
        nbhd_cache = {}
        for neighborhood, route in zip(
            self.neighborhoods, self.routes, strict=False
        ):
            src_rank, dst_rank = route
            if src_rank != dst_rank and (src_rank, dst_rank) not in nbhd_cache:
                n_dst_nodes = getattr(params, f"x_{dst_rank}").shape[0]

                # There is no neighbourhood in question
                if n_dst_nodes == 0:
                    nbhd_cache[(src_rank, dst_rank)] = None
                elif src_rank > dst_rank:
                    boundary = getattr(params, neighborhood).coalesce()
                    nbhd_cache[(src_rank, dst_rank)] = (
                        interrank_boundary_index(
                            getattr(params, f"x_{src_rank}"),
                            boundary.indices(),
                            n_dst_nodes,
                        )
                    )
                elif src_rank < dst_rank:
                    coboundary = getattr(params, neighborhood).coalesce()
                    nbhd_cache[(src_rank, dst_rank)] = (
                        interrank_boundary_index(
                            getattr(params, f"x_{src_rank}"),
                            coboundary.indices(),
                            n_dst_nodes,
                        )
                    )
        return nbhd_cache

    def forward_intrarank(
        self, src_rank, route_index, data: torch_geometric.data.Data
    ):
        """Forward for cells where src_rank==dst_rank.

        Parameters
        ----------
        src_rank : int
            Source rank of the transmitting cell.
        route_index : int
            The index of this particular message passing route.
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        data
            The data object with messages passed.
        """
        nbhd = self.neighborhoods[route_index]
        batch_route = self.intrarank_expand(data, src_rank, nbhd)

        # Build input graph with actual node features
        actual_features = getattr(data, f"x_{src_rank}").to(self.device)
        input_graph = torch_geometric.data.Data(
            x=actual_features,
            edge_index=batch_route.edge_index,
            batch=torch.zeros(
                batch_route.x.shape[0],
                dtype=torch.int64,
                device=self.device,
            ),
        ).to(self.device)

        # Compute all encodings (FEs use features, PSEs use graph structure)
        return self._data_to_device(self.encoding_transform(input_graph))

    def forward_interank(
        self, src_rank, dst_rank, nbhd_cache, data: torch_geometric.data.Data
    ):
        """Forward for cells where src_rank!=dst_rank.

        Parameters
        ----------
        src_rank : int
            Source rank of the transmitting cell.
        dst_rank : int
            Destination rank of the transmitting cell.
        nbhd_cache : dict
            Cache of the neighbourhood information.
        data : torch_geometric.data.Data
            The input data.

        Returns
        -------
        data
            The data object with messages passed.
        """
        # This has the boundary index
        nbhd = nbhd_cache[(src_rank, dst_rank)]

        # The actual data to pass to the GNN
        batch_route = self.interrank_expand(data, src_rank, dst_rank, nbhd)
        # The number of destination cells
        n_dst_cells = data[f"x_{dst_rank}"].shape[0]

        # Build features for the expanded graph (dst + src nodes)
        feat_on_dst = getattr(data, f"x_{dst_rank}").to(self.device)
        feat_on_src = getattr(data, f"x_{src_rank}").to(self.device)

        # Pad features to match dimensions if needed
        if feat_on_dst.shape[1] > feat_on_src.shape[1]:
            pad = (0, feat_on_dst.shape[1] - feat_on_src.shape[1])
            feat_on_src = torch.nn.functional.pad(
                feat_on_src, pad, "constant", 0
            )
        elif feat_on_dst.shape[1] < feat_on_src.shape[1]:
            pad = (0, feat_on_src.shape[1] - feat_on_dst.shape[1])
            feat_on_dst = torch.nn.functional.pad(
                feat_on_dst, pad, "constant", 0
            )

        x_expanded = torch.vstack([feat_on_dst, feat_on_src])

        input_graph = torch_geometric.data.Data(
            x=x_expanded,
            edge_index=batch_route.edge_index,
            batch=torch.zeros(
                batch_route.x.shape[0],
                dtype=torch.int64,
                device=self.device,
            ),
        ).to(self.device)

        # Compute all encodings on expanded graph
        expanded_out = self._data_to_device(
            self.encoding_transform(input_graph)
        )

        # Select only destination cells
        return self.select_dst_encodings(expanded_out, n_dst_cells)

    def _make_zero_data_for_all_encodings(self, n_cells, dims):
        """Create a Data object with zero tensors for all encodings.

        Parameters
        ----------
        n_cells : int
            Number of cells (rows) for each encoding tensor.
        dims : list[int]
            Dimension for each encoding. Must have same length as self.encodings.

        Returns
        -------
        torch_geometric.data.Data
            Data object with each encoding key mapped to a zero tensor.
        """
        zero_data = {}
        for enc, dim in zip(self.encodings, dims, strict=True):
            zero_data[enc] = torch.zeros(
                (n_cells, dim),
                dtype=torch.float,
                device=self.device,
            )
        return Data(**zero_data)

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

        # Copy initial values as first hop
        if self.copy_initial:
            for i in range(self.max_rank + 1):
                x_i = getattr(data, f"x_{i}").float().to(self.device)
                setattr(data, f"x{i}_0", x_i)
        self.routes = get_routes_from_neighborhoods(self.neighborhoods)
        nbhd_cache = self.get_nbhd_cache(data)

        # Get encoding dimensions for zero initialization
        all_dims = self.dim_all_encodings

        x_out_per_route = {}
        # Iterate over the routes (i, [0, 1])
        for route_index, route in enumerate(self.routes):
            src_rank, dst_rank = route

            if src_rank == dst_rank:
                # If there are no nodes in this rank, we skip
                if getattr(data, f"x_{src_rank}").shape[0] == 0:
                    x_out_per_route[route_index] = (
                        self._make_zero_data_for_all_encodings(0, all_dims)
                    )
                    continue
                # We cannot use PE embeddings on single-node graphs
                if getattr(data, f"x_{src_rank}").shape[0] == 1:
                    x_out_per_route[route_index] = (
                        self._make_zero_data_for_all_encodings(1, all_dims)
                    )
                    continue
                x_out = self.forward_intrarank(src_rank, route_index, data)
                x_out_per_route[route_index] = x_out

            elif src_rank != dst_rank:
                # If there is no neighborhood, we skip
                if nbhd_cache[(src_rank, dst_rank)] is None:
                    x_out_per_route[route_index] = (
                        self._make_zero_data_for_all_encodings(0, all_dims)
                    )
                    continue
                x_out = self.forward_interank(
                    src_rank, dst_rank, nbhd_cache, data
                )
                # Outputs of this particular route
                x_out_per_route[route_index] = x_out

        # aggregate across neighborhoods
        x_out_per_rank = self.aggregate_inter_nbhd(x_out_per_route)

        # If no information was passed to a rank, then we initialize an empty vector
        # with the output dimension of the pre-trained model
        # and set the features as 0
        hop_num = int(self.copy_initial)
        for rank in range(self.max_rank + 1):
            if rank not in x_out_per_rank:
                n_cells = data[f"x_{rank}"].shape[0]
                rank_dims = [
                    self.in_channels[rank][idx + hop_num]
                    for idx in range(len(self.encodings))
                ]
                x_out_per_rank[rank] = self._make_zero_data_for_all_encodings(
                    n_cells, rank_dims
                )

        for key, value in x_out_per_rank.items():
            for idx, enc in enumerate(self.encodings):
                data_key = f"x{key}_{idx + hop_num}"
                setattr(
                    data,
                    data_key,
                    value[enc].float().to(self.device),
                )

        return data


def interrank_boundary_index(x_src, boundary_index, n_dst_nodes):
    """
    Recover lifted graph.

    Edge-to-node boundary relationships of a graph with n_nodes and n_edges
    can be represented as up-adjacency node relations. There are n_nodes+n_edges nodes in this lifted graph.
    Desgiend to work for regular (edge-to-node and face-to-edge) boundary relationships.

    Parameters
    ----------
    x_src : torch.tensor
        Source node features. Shape [n_src_nodes, n_features]. Should represent edge or face features.
    boundary_index : list of lists or list of tensors
        List boundary_index[0] stores node ids in the boundary of edge stored in boundary_index[1].
        List boundary_index[1] stores list of edges.
    n_dst_nodes : int
        Number of destination nodes.

    Returns
    -------
    edge_index : list of lists
        The edge_index[0][i] and edge_index[1][i] are the two nodes of edge i.
    edge_attr : tensor
        Edge features are given by feature of bounding node represnting an edge. Shape [n_edges, n_features].
    """
    node_ids = (
        boundary_index[0]
        if torch.is_tensor(boundary_index[0])
        else torch.tensor(boundary_index[0], dtype=torch.int32)
    )
    edge_ids = (
        boundary_index[1]
        if torch.is_tensor(boundary_index[1])
        else torch.tensor(boundary_index[1], dtype=torch.int32)
    )

    max_node_id = n_dst_nodes
    adjusted_edge_ids = edge_ids + max_node_id

    edge_index = torch.zeros((2, node_ids.numel()), dtype=node_ids.dtype)
    edge_index[0, :] = node_ids
    edge_index[1, :] = adjusted_edge_ids

    edge_attr = x_src[edge_ids].squeeze()

    return edge_index, edge_attr
