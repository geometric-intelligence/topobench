"""This module implements the MapperCover class."""

import networkx as nx
import torch
import torch_geometric
from torch_geometric.transforms import (
    AddLaplacianEigenvectorPE,
    Compose,
    SVDFeatureReduction,
    ToUndirected,
)

from topobench.transforms.liftings.graph2hypergraph.base import (
    Graph2HypergraphLifting,
)


class MapperCover:
    r"""The MapperCover class computes the cover used in constructing the Mapper for the MapperLifting class.

    Parameters
    ----------
    resolution : int, optional
        The number of intervals in the MapperCover. Default is 10.
    gain : float, optional
        The proportion of overlap between consecutive intervals
        in the MapperCover and should be value between 0 and 0.5.
        Default is 0.3.

    Attributes
    ----------
    cover_intervals : (resolution, 2) Tensor
        A tensor containing each interval in the MapperCover.
    """

    def __init__(self, resolution=10, gain=0.3):
        self.resolution = resolution
        self.gain = gain
        self._verify_cover_parameters()

    def fit_transform(self, filtered_data):
        r"""Construct an interval cover over filtered data.

        Parameters
        ----------
        filtered_data : (n_sample, 1) Tensor
            Filtered data to construct the Mapper cover.

        Returns
        -------
         < (n_sample, resolution) boolean Tensor.
            Mask which identifies which data points are
            in each cover set. Covers which are empty
            are removed so output tensor has at most
            size (n_sample, resolution).
        """
        # We add a slight buffer to the minimum and maximum
        # values to ensure that each data point is covered.
        data_min = torch.min(filtered_data) - 1e-3
        data_max = torch.max(filtered_data) + 1e-3
        data_range = data_max - data_min

        # width of each interval in the cover and last left endpoint
        cover_width = data_range / (
            self.resolution - (self.resolution - 1) * self.gain
        )
        last_lower_endpoint = data_min + cover_width * (
            self.resolution - 1
        ) * (1 - self.gain)
        lower_endpoints = torch.linspace(
            data_min, last_lower_endpoint, self.resolution
        )
        upper_endpoints = lower_endpoints + cover_width
        self.cover_intervals = torch.hstack(
            (
                lower_endpoints.reshape([self.resolution, 1]),
                upper_endpoints.reshape([self.resolution, 1]),
            )
        )
        # want a n x resolution Boolean tensor
        lower_values = torch.gt(filtered_data, lower_endpoints)
        upper_values = torch.lt(filtered_data, upper_endpoints)

        # need to check close values to deal with some endpoint issues
        lower_is_close_values = torch.isclose(filtered_data, lower_endpoints)
        upper_is_close_values = torch.isclose(filtered_data, upper_endpoints)

        # construct the boolean mask
        mask = torch.logical_and(
            torch.logical_or(lower_values, lower_is_close_values),
            torch.logical_or(upper_values, upper_is_close_values),
        )
        # assert every data point is covered
        assert torch.all(torch.any(mask, 1)), f"{torch.any(mask, 1)}"

        # remove empty intervals from cover
        non_empty_covers = torch.any(mask, 0)
        return mask[:, non_empty_covers]

    def _verify_cover_parameters(self):
        """Verify the cover parameters for the MapperCover class."""
        assert self.gain > 0 and self.gain <= 0.5, (
            f"Gain must be a proportion greater than 0 and at most 0.5. Currently, gain is {self.gain}."
        )
        assert self.resolution > 0, (
            f"Resolution should be greater than 0. Currently, resolution is {self.resolution}."
        )
        assert float(self.resolution).is_integer(), (
            f"Resolution must be an integer value. Currently, resolution is {self.resolution}."
        )


# Global filter dictionary for the MapperLifting class.
filter_dict = {
    "laplacian": Compose(
        [ToUndirected(), AddLaplacianEigenvectorPE(k=1, is_undirected=True)]
    ),
    "svd": SVDFeatureReduction(out_channels=1),
    "feature_sum": lambda data: torch.sum(data.x, dim=1).unsqueeze(1),
    "position_sum": lambda data: torch.sum(data.pos, dim=1).unsqueeze(1),
    "feature_pca": lambda data: torch.matmul(
        data.x, torch.pca_lowrank(data.x, q=1)[2][:, :1]
    ),
    "position_pca": lambda data: torch.matmul(
        data.pos, torch.pca_lowrank(data.pos, q=1)[2][:, :1]
    ),
}


class MapperLifting(Graph2HypergraphLifting):
    r"""Lift graphs to hypergraph domain using a Mapper construction for CC-pooling.

    Parameters
    ----------
    filter_attr : str, optional
        Name of the filter functional to filter data to 1-dimensional subspace.
        The filter attribute can be "laplacican", "svd", "pca", "feature_sum",
        "position_sum". You may also define your own filter_attr string if
        the filter_func parameter is defined.
        Default is "laplacian".
    resolution : int, optional
        The number of intervals to construct the MapperCover.
        Default is 10.
    gain : float, optional
        The percentage of overlap between consecutive intervals
        in MapperCover and should be a value between 0 and 0.5.
        Default is 0.3.
    filter_func : object, optional
        Filter function used for Mapper construction.
        Self defined lambda function or transform to filter data.
        Function must output an (n_sample, 1) Tensor.
        If filter_func is not None, user must define filter_attr
        as a string not already listed above.
        Default is None.
    **kwargs : optional
        Additional arguments for the class.

    Attributes
    ----------
    filtered_data : dict
        Filtered data used to compute the Mapper lifting.
        Dictionary is of the form
        {filter_attr: filter_func(data)}.
    cover : (k, resolution) boolean Tensor
        Mask computed from the MapperCover class
        to compute the Mapper lifting with k < n_sample.
    clusters : dict
        Distinct connected components in each cover set
        computed after fitting the Mapper cover.
        Dictionary has integer keys and tuple values
        of the form (cover_set_i, nodes_in_cluster).
        Each cluster is a rank 2 hyperedge in the
        hypergraph.

    Notes
    -----
    The following are common filter functions which can be called with
    filter_attr.

    1. "laplacian" : Converts data to an undirected graph and then applies the
    torch_geometric.transforms.AddLaplacianEigenvectorPE(k=1) transform and
    projects onto the smallest nonzero eigenvector.

    2. "svd" : Applies the torch_geometric.transforms.SVDFeatureReduction(out_channels=1)
    transform to the node feature matrix (ie. torch_geometric.Data.data.x)
    to project data to a 1-dimensional subspace.

    3. "feature_pca" : Applies torch.pca_lowrank(q=1) transform to node feature matrix
    (ie. torch_geometric.Data.data.x) and then projects to the 1st principal component.

    4. "position_pca" : Applies torch.pca_lowrank(q=1) transform to node position matrix
    (ie. torch_geometric.Data.data.pos) and then projects to the 1st principal component.

    5. "feature_sum" : Applies torch.sum(dim=1) to the node feature matrix in the graph
    (ie. torch_geometric.Data.data.x).

    6. "position_sum" : Applies torch.sum(dim=1) to the node position matrix in the graph
    (ie. torch_geometric.Data.data.pos).

    You may also construct your own filter_attr and filter_func:

    7. "my_filter_attr" : Name of a self defined function
    my_filter_func = lambda data : my_filter_func(data)
    where my_filter_func(data) outputs a (n_sample, 1) Tensor.

    References
    ----------
    .. [1] Hajij, M., Zamzmi, G., Papamarkou, T., Miolane, N., Guzmán-Sáenz,
        A., Ramamurthy, K. N., et al. (2022).
        Topological deep learning: Going beyond graph data.
        arXiv preprint arXiv:2206.00606.
    """

    def __init__(
        self,
        filter_attr="laplacian",
        resolution=10,
        gain=0.3,
        filter_func=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filter_attr = filter_attr
        self.resolution = resolution
        self.gain = gain
        self.filter_func = filter_func
        self._verify_filter_parameters(filter_attr, filter_func)

    def _verify_filter_parameters(self, filter_attr, filter_func):
        """Verify the filter parameters for the MapperLifting class.

        Parameters
        ----------
        filter_attr : str
            Name of the filter functional to filter data to 1-dimensional subspace.
        filter_func : object
            Filter function used for Mapper construction.
        """
        if filter_func is None:
            assert self.filter_attr in filter_dict, (
                f"Please add function to filter_func or choose filter_attr from {list(filter_dict)}. \
            Currently filter_func is {filter_func} and filter_attr is {filter_attr}."
            )
        if filter_func is not None:
            assert self.filter_attr not in filter_dict, (
                f"Assign new filter_attr not in {list(filter_dict)} or leave filter_func as None. \
            Currently filter_func is {filter_func} and filter_attr is {filter_attr}"
            )
            assert isinstance(filter_attr, str), (
                f"{filter_attr} must be a string."
            )

    def _filter(self, data):
        """Apply 1-dimensional filter function to torch_geometric.Data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be filtered.

        Returns
        -------
        torch.Tensor
            Filtered data of size (n_samples, 1).
        """
        if self.filter_attr in filter_dict:
            transform = filter_dict[self.filter_attr]
            transformed_data = transform(data)
            if self.filter_attr == "laplacian":
                filtered_data = transformed_data["laplacian_eigenvector_pe"]
            if self.filter_attr == "svd":
                filtered_data = transformed_data.x
            if self.filter_attr not in [
                "laplacian",
                "svd",
            ]:
                filtered_data = transformed_data

        else:
            transform = self.filter_func
            filtered_data = transform(data)

        assert filtered_data.size() == torch.Size([len(data.x), 1]), (
            f"filtered data should have size [n_samples, 1]. Currently filtered data has size {filtered_data.size()}."
        )
        self.filtered_data = {self.filter_attr: filtered_data}

        return filtered_data

    def _cluster(self, data, cover_mask):
        """Find clusters in each cover set within cover_mask.

        Extended Summary
        ----------------
        For each cover set, a cluster is a
        distinct connected component.
        Clusters are stored in the dictionary, self.clusters.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.
        cover_mask : (n_samples, resolution) bool Tensor
            Mask which identifies which data points are
            in each cover set.

        Returns
        -------
        dict
            Dictionary of clusters in each.
        """
        mapper_clusters = {}
        num_clusters = 0

        # convert data to undirected graph for clustering
        to_undirected = ToUndirected()
        data = to_undirected(data)

        # Each cover set is of the form [n_samples]
        for i, cover_set in enumerate(torch.t(cover_mask)):
            # Find indices of nodes which are in each cover set
            # cover_data = data.subgraph(cover_set.T) does not work
            # as it relabels node indices
            cover_data, _ = torch_geometric.utils.subgraph(
                cover_set, data["edge_index"]
            )

            edges = [
                (i.item(), j.item())
                for i, j in zip(cover_data[0], cover_data[1], strict=False)
            ]

            nodes = [i.item() for i in torch.where(cover_set)[0]]

            # build graph to find clusters
            cover_graph = nx.Graph()
            cover_graph.add_nodes_from(nodes)
            cover_graph.add_edges_from(edges)

            # find clusters
            clusters = nx.connected_components(cover_graph)

            for cluster in clusters:
                # index is the subset of nodes in data
                # contained in cluster
                index = torch.Tensor(list(cluster))

                # kth cluster is item in dictionary
                # of the form:
                # k : (cover_set_index, nodes_in_cluster)
                mapper_clusters[num_clusters] = (i, index)
                num_clusters += 1

        self.clusters = mapper_clusters

        return mapper_clusters

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the topology of a graph to hypergraph domain by Mapper on Graphs.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        # Filter the data to 1-dimensional subspace
        filtered_data = self._filter(data)

        # Define and fit the cover
        cover = MapperCover(self.resolution, self.gain)
        cover_mask = cover.fit_transform(filtered_data)
        self.cover = cover_mask

        # Find the clusters in the fitted cover
        mapper_clusters = self._cluster(data, cover_mask)
        self.clusters = mapper_clusters

        # Construct the hypergraph dictionary
        num_nodes = data["x"].shape[0]
        num_edges = data["edge_index"].size()[1]

        num_clusters = len(mapper_clusters)
        num_hyperedges = num_edges + num_clusters

        incidence_edges = torch.zeros(num_nodes, num_edges)

        for i, edge in enumerate(torch.t(data["edge_index"])):
            incidence_edges[edge[0], i] = 1
            incidence_edges[edge[1], i] = 1

        incidence_hyperedges = torch.zeros(num_nodes, num_clusters)

        for i, hyperedge in enumerate(mapper_clusters):
            for j in mapper_clusters[hyperedge][1]:
                incidence_hyperedges[j.int(), i] = 1

        # Incidence matrix is (num_nodes, num_edges + num_clusters) size matrix
        incidence = torch.hstack([incidence_edges, incidence_hyperedges])
        incidence = torch.Tensor(incidence).to_sparse_coo()

        return {
            "incidence_hyperedges": incidence,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
