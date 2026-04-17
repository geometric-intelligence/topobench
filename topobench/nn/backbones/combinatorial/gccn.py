"""Define the TopoTune class, which, given a choice of hyperparameters, instantiates a GCCN expecting a collection of strictly augmented Hasse graphs as input."""

import copy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from topobench.data.utils import get_routes_from_neighborhoods


class TopoTune(torch.nn.Module):
    """Tunes a GNN model using higher-order relations.

    This class takes a GNN and its kwargs as inputs, and tunes it with specified additional relations.

    Parameters
    ----------
    GNN : torch.nn.Module, a class not an object
        The GNN class to use. ex: GAT, GCN.
    neighborhoods : list of lists
        The neighborhoods of interest.
    layers : int
        The number of layers to use. Each layer contains one GNN.
    use_edge_attr : bool
        Whether to use edge attributes.
    activation : str
        The activation function to use. ex: 'relu', 'tanh', 'sigmoid'.
    """

    def __init__(
        self,
        GNN,
        neighborhoods,
        layers,
        use_edge_attr,
        activation,
    ):
        super().__init__()
        self.routes = get_routes_from_neighborhoods(neighborhoods)
        self.neighborhoods = neighborhoods
        self.layers = layers
        self.use_edge_attr = use_edge_attr
        self.max_rank = max([max(route) for route in self.routes])
        self.graph_routes = torch.nn.ModuleList()
        self.GNN = [i for i in GNN.named_modules()]
        self.activation = activation

        # Instantiate GNN layers
        num_routes = len(self.routes)
        for _ in range(self.layers):
            layer_routes = torch.nn.ModuleList()
            for _ in range(num_routes):
                layer_routes.append(copy.deepcopy(GNN))
            self.graph_routes.append(layer_routes)

        self.hidden_channels = GNN.hidden_channels
        self.out_channels = GNN.out_channels

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
                if src_rank > dst_rank:
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

    def intrarank_expand(self, params, src_rank, nbhd):
        """Expand the complex into an intrarank Hasse graph.

        Parameters
        ----------
        params : dict
            The parameters of the batch, containing the complex.
        src_rank : int
            The source rank.
        nbhd : str
            The neighborhood to use.

        Returns
        -------
        torch_geometric.data.Data
            The expanded batch of intrarank Hasse graphs for this route.
        """
        batch_route = Data(
            x=getattr(params, f"x_{src_rank}"),
            edge_index=getattr(params, nbhd).indices(),
            edge_weight=getattr(params, nbhd).values().squeeze(),
            edge_attr=getattr(params, nbhd).values().squeeze(),
            requires_grad=True,
        )

        return batch_route

    def intrarank_gnn_forward(self, batch_route, layer_idx, route_index):
        """Forward pass of the GNN (one layer) for an intrarank Hasse graph.

        Parameters
        ----------
        batch_route : torch_geometric.data.Data
            The batch of intrarank Hasse graphs for this route.
        layer_idx : int
            The index of the TopoTune layer.
        route_index : int
            The index of the route.

        Returns
        -------
        torch.tensor
            The output of the GNN (updated features).
        """
        if batch_route.x.shape[0] < 2:
            return batch_route.x
        out = self.graph_routes[layer_idx][route_index](
            batch_route.x,
            batch_route.edge_index,
            #    batch_route.edge_weight, # TODO Mathilde : some gnns take edge_weight (1d) and some take edge_attr.
            #    batch_route.edge_attr,
        )
        return out

    def interrank_expand(
        self, params, src_rank, dst_rank, nbhd_cache, membership
    ):
        """Expand the complex into an interrank Hasse graph.

        Parameters
        ----------
        params : dict
            The parameters of the batch, containing the complex.
        src_rank : int
            The source rank.
        dst_rank : int
            The destination rank.
        nbhd_cache : dict
            The neighborhood cache containing the expanded boundary index and edge attributes.
        membership : dict
            The batch membership of the graphs per rank.

        Returns
        -------
        torch_geometric.data.Data
            The expanded batch of interrank Hasse graphs for this route.
        """
        src_batch = membership[src_rank]
        dst_batch = membership[dst_rank]
        edge_index, edge_attr = nbhd_cache
        device = getattr(params, f"x_{src_rank}").device
        feat_on_dst = torch.zeros_like(getattr(params, f"x_{dst_rank}"))
        x_in = torch.vstack([feat_on_dst, getattr(params, f"x_{src_rank}")])
        batch_expanded = torch.cat([dst_batch, src_batch], dim=0)

        batch_route = Data(
            x=x_in,
            edge_index=edge_index.to(device),
            edge_attr=edge_attr.to(device),
            edge_weight=edge_attr.to(device),
            batch=batch_expanded.to(device),
        )

        return batch_route

    def interrank_gnn_forward(
        self, batch_route, layer_idx, route_index, n_dst_cells
    ):
        """Forward pass of the GNN (one layer) for an interrank Hasse graph.

        Parameters
        ----------
        batch_route : torch_geometric.data.Data
            The batch of interrank Hasse graphs for this route.
        layer_idx : int
            The index of the layer.
        route_index : int
            The index of the route.
        n_dst_cells : int
            The number of destination cells in the whole batch.

        Returns
        -------
        torch.tensor
            The output of the GNN (updated features).
        """
        expanded_out = self.graph_routes[layer_idx][route_index](
            batch_route.x,
            batch_route.edge_index,
            #    batch_route.edge_weight, # TODO : some gnns take edge_weight (1d) and some take edge_attr.
            #    batch_route.edge_attr,
        )
        out = expanded_out[:n_dst_cells]
        return out

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
            if dst_rank not in x_out_per_rank:
                x_out_per_rank[dst_rank] = x_out_per_route[route_index]
            else:
                x_out_per_rank[dst_rank] += x_out_per_route[route_index]
        return x_out_per_rank

    def generate_membership_vectors(self, batch: Data):
        """Generate membership vectors based on batch.cell_statistics.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            The batch membership of the graphs per rank.
        """
        max_dim = batch.cell_statistics.shape[1]
        cell_statistics = batch.cell_statistics
        membership = {
            j: torch.tensor(
                [
                    elem
                    for list in [
                        [i] * x for i, x in enumerate(cell_statistics[:, j])
                    ]
                    for elem in list
                ]
            )
            for j in range(max_dim)
        }
        return membership

    def forward(self, batch):
        """Forward pass of the model.

        Parameters
        ----------
        batch : Complex or ComplexBatch(Complex)
            The input data.

        Returns
        -------
        dict
            The output hidden states of the model per rank.
        """
        act = get_activation(self.activation)

        nbhd_cache = self.get_nbhd_cache(batch)
        membership = self.generate_membership_vectors(batch)

        x_out_per_route = {}
        for layer_idx in range(self.layers):
            for route_index, route in enumerate(self.routes):
                src_rank, dst_rank = route

                if src_rank == dst_rank:
                    nbhd = self.neighborhoods[route_index]
                    batch_route = self.intrarank_expand(batch, src_rank, nbhd)
                    x_out = self.intrarank_gnn_forward(
                        batch_route, layer_idx, route_index
                    )

                    x_out_per_route[route_index] = x_out

                elif src_rank != dst_rank:
                    nbhd = nbhd_cache[(src_rank, dst_rank)]

                    batch_route = self.interrank_expand(
                        batch, src_rank, dst_rank, nbhd, membership
                    )
                    x_out = self.interrank_gnn_forward(
                        batch_route,
                        layer_idx,
                        route_index,
                        getattr(batch, f"x_{dst_rank}").shape[0],
                    )

                    x_out_per_route[route_index] = x_out

            # aggregate across neighborhoods
            x_out_per_rank = self.aggregate_inter_nbhd(x_out_per_route)

            # update and replace the features for next layer
            for rank in x_out_per_rank:
                x_out_per_rank[rank] = act(x_out_per_rank[rank])
                setattr(batch, f"x_{rank}", x_out_per_rank[rank])

        for rank in range(self.max_rank + 1):
            if rank not in x_out_per_rank:
                x_out_per_rank[rank] = getattr(batch, f"x_{rank}")

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

    edge_index = torch.zeros((2, node_ids.numel()), dtype=node_ids.dtype)
    edge_index[0, :] = node_ids
    edge_index[1, :] = adjusted_edge_ids

    edge_attr = x_src[edge_ids].squeeze()

    return edge_index, edge_attr


def get_activation(nonlinearity, return_module=False):
    """Activation resolver from CWN.

    Parameters
    ----------
    nonlinearity : str
        The nonlinearity to use.
    return_module : bool
        Whether to return the module or the function.

    Returns
    -------
    module or function
        The module or the function.
    """
    if nonlinearity == "relu":
        module = torch.nn.ReLU
        function = F.relu
    elif nonlinearity == "elu":
        module = torch.nn.ELU
        function = F.elu
    elif nonlinearity == "id":
        module = torch.nn.Identity

        def function(x):
            return x
    elif nonlinearity == "sigmoid":
        module = torch.nn.Sigmoid
        function = F.sigmoid
    elif nonlinearity == "tanh":
        module = torch.nn.Tanh
        function = torch.tanh
    else:
        raise NotImplementedError(
            f"Nonlinearity {nonlinearity} is not currently supported."
        )
    if return_module:
        return module
    return function
