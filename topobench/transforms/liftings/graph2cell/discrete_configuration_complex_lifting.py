"""This module implements the discrete configuratio complex lifting lifting for graphs to cell complexes."""

from itertools import permutations
from typing import ClassVar

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import CellComplex

from topobench.transforms.liftings.graph2cell.base import Graph2CellLifting

Vertex = int
Edge = tuple[Vertex, Vertex]
ConfigurationTuple = tuple[Vertex | Edge]


class DiscreteConfigurationComplexLifting(Graph2CellLifting):
    """Lift graphs to cell complexes.

    Lift graphs to cell complexes by generating the k-th discrete configuration complex D_k(G) of the graph. This is a cube complex, which is similar to a simplicial complex except each n-dimensional cell is homeomorphic to a n-dimensional cube rather than an n-dimensional simplex.

    The discrete configuration complex of order k consists of all sets of k unique edges or vertices of G, with the additional constraint that if an edge e is in a cell, then neither of the endpoints of e is in the cell. For examples of different graphs and their configuration complexes, see the tutorial.

    Note that since TopoNetx only supports cell complexes of dimension 2, if you generate a configuration complex of order k > 2 this will only produce the 2-skeleton.

    Parameters
    ----------
    k : int
        The order of the configuration complex, i.e. the number of 'agents' in a single configuration.
    preserve_edge_attr : bool, optional
        Whether to preserve edge attributes. Default is True.
    feature_aggregation : str, optional
        For a k-agent configuration, the method by which the features are aggregated. Can be "mean", "sum", or "concat". Default is "concat".
    **kwargs : dict, optional
        Additional arguments for the class.

    Notes
    -----
    The discrete configuration complex provides a way to model higher-order interactions in graphs, particularly useful for scenarios involving multiple agents or particles moving on a graph structure.
    """

    def __init__(
        self,
        k: int,
        preserve_edge_attr: bool = True,
        feature_aggregation="concat",
        **kwargs,
    ):
        self.k = k
        self.complex_dim = kwargs["complex_dim"]
        assert feature_aggregation in ["mean", "sum", "concat"], (
            "Feature_aggregation must be one of 'mean', 'sum', 'concat'"
        )
        self.feature_aggregation = feature_aggregation
        super().__init__(preserve_edge_attr=preserve_edge_attr, **kwargs)

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        r"""Apply the full lifting (topology + features) to the input data.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        torch_geometric.data.Data
            The lifted data.
        """
        # Unlike the base class, we do not pass the initial data to the final data
        # This is because the configuration complex has a completely different 1-skeleton from the original graph
        lifted_topology = self.lift_topology(data)
        lifted_topology = self.feature_lifting(lifted_topology)
        return torch_geometric.data.Data(y=data.y, **lifted_topology)

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Generate the cubical complex of discrete graph configurations.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        G = self._generate_graph_from_data(data)
        assert not G.is_directed(), "Directed Graphs are not supported."

        Configuration = generate_configuration_class(
            G, self.feature_aggregation, self.contains_edge_attr
        )

        # The vertices of the configuration complex are just tuples of k vertices
        for dim_0_configuration_tuple in permutations(G, self.k):
            configuration = Configuration(dim_0_configuration_tuple)
            configuration.generate_upwards_neighbors()

        cells = {i: [] for i in range(self.k + 1)}
        for conf in Configuration.instances.values():
            features = conf.features()
            attrs = {"features": features} if features is not None else {}
            cell = (conf.contents, attrs)
            cells[conf.dim].append(cell)

        # TopoNetX only supports cells of dimension <= 2
        cc = CellComplex()
        for node, attrs in cells[0]:
            cc.add_node(node, **attrs)
        for edge, attrs in cells[1]:
            cc.add_edge(edge[0], edge[1], **attrs)
        for cell, attrs in cells[2]:
            cell_vertices = edge_cycle_to_vertex_cycle(cell)
            cc.add_cell(cell_vertices, rank=2, **attrs)

        return self._get_lifted_topology(cc, G)


def edge_cycle_to_vertex_cycle(edge_cycle: list[list | tuple]):
    """Take a cycle represented by a list of edges and returns a vertex representation: [(1, 2), (0, 1), (1, 2)] -> [1, 2, 3].

    Parameters
    ----------
    edge_cycle : list[list | tuple]
        The cycle represented by a list of edges.

    Returns
    -------
    list
        The cycle represented by a list of vertices.
    """
    return [e[0] for e in nx.find_cycle(nx.Graph(edge_cycle))]


def generate_configuration_class(
    graph: nx.Graph, feature_aggregation: str, edge_features: bool
):
    """Factory for the Configuration class.

    Parameters
    ----------
    graph : nx.Graph
        The input graph.
    feature_aggregation : str
        The method by which the features are aggregated.
    edge_features : bool
        Whether edge features are present.

    Returns
    -------
    Configuration
        The Configuration.
    """

    class Configuration:
        """Configuration Class.

        Represents a single legal configuration of k agents on a graph G. A legal configuration is a tuple of k edges and vertices of G where all the vertices and endpoints are **distinct**, i.e., no two edges sharing an endpoint can simultaneously be in the configuration, and an adjacent (edge, vertex) pair can be contained in the configuration. Each configuration corresponds to a cell, and the number of edges in the configuration is the dimension.

        Parameters
        ----------
        configuration_tuple : tuple
            A tuple representing the configuration of k agents on the graph.

        Notes
        -----
        The `configuration_tuple` parameter was added to document its role in representing specific configurations. This class is used to model higher-order interactions in graphs by defining legal configurations based on graph topology.
        """

        instances: ClassVar[dict[ConfigurationTuple, "Configuration"]] = {}

        def __new__(cls, configuration_tuple: ConfigurationTuple):
            """Create a new configuration object if it does not already exist.

            Parameters
            ----------
            configuration_tuple : ConfigurationTuple
                The configuration tuple to be represented.

            Returns
            -------
            Configuration
                The configuration object.
            """
            # Ensure that a configuration tuple corresponds to a *unique* configuration object
            key = configuration_tuple
            if key not in cls.instances:
                cls.instances[key] = super().__new__(cls)

            return cls.instances[key]

        def __init__(self, configuration_tuple: ConfigurationTuple) -> None:
            # If this object was already initialized earlier, maintain current state
            if hasattr(self, "initialized"):
                return

            self.initialized = True
            self.configuration_tuple = configuration_tuple
            self.neighborhood = set()
            self.dim = 0
            for agent in configuration_tuple:
                if isinstance(agent, Vertex):
                    self.neighborhood.add(agent)
                else:
                    self.neighborhood.update(set(agent))
                    self.dim += 1

            if self.dim == 0:
                self.contents = configuration_tuple
            else:
                self.contents = []

            self._upwards_neighbors_generated = False

        def features(self):
            """Generate the features for the configuration by combining the edge and vertex features.

            Returns
            -------
            torch.Tensor
                The features of the configuration.
            """
            features = []
            for agent in self.configuration_tuple:
                if isinstance(agent, Vertex):
                    features.append(graph.nodes[agent]["features"])
                elif edge_features:
                    features.append(graph.edges[agent]["features"])

            if not features:
                return None

            if feature_aggregation == "mean":
                try:
                    return torch.stack(features, dim=0).mean(dim=0)
                except Exception as e:
                    raise ValueError(
                        "Failed to mean feature tensors. This may be because edge features and vertex features have different shapes. If this is the case, use feature_aggregation='concat', or disable edge features."
                    ) from e
            elif feature_aggregation == "sum":
                try:
                    return torch.stack(features, dim=0).sum(dim=0)
                except Exception as e:
                    raise ValueError(
                        "Failed to sum feature tensors. This may be because edge features and vertex features have different shapes. If this is the case, use feature_aggregation='concat', or disable edge features."
                    ) from e
            elif feature_aggregation == "concat":
                return torch.concatenate(features, dim=-1)
            else:
                raise ValueError(
                    f"Unrecognized feature_aggregation: {feature_aggregation}"
                )

        def generate_upwards_neighbors(self):
            """For the configuration self of dimension d, generate the configurations of dimension d+1 containing it."""
            if self._upwards_neighbors_generated:
                return
            self._upwards_neighbors_generated = True
            for i, agent in enumerate(self.configuration_tuple):
                if isinstance(agent, Vertex):
                    for neighbor in graph[agent]:
                        self._generate_single_neighbor(i, agent, neighbor)

        def _generate_single_neighbor(
            self, index: int, vertex_agent: int, neighbor: int
        ):
            """Generate a configuration containing self by moving an agent from a vertex onto an edge.

            Parameters
            ----------
            index : int
                The index of the vertex in the configuration tuple.
            vertex_agent : int
                The vertex to move.
            neighbor : int
                The neighbor to move the vertex to.

            Returns
            -------
            None
                The new configuration is added to the instances dictionary.
            """
            # If adding the edge (vertex_agent, neighbor) would produce an illegal configuration, ignore it
            if neighbor in self.neighborhood:
                return

            # We always orient edges (min -> max) to maintain uniqueness of configuration tuples
            new_edge = (
                min(vertex_agent, neighbor),
                max(vertex_agent, neighbor),
            )

            # Remove the vertex at index and replace it with new edge
            new_configuration_tuple = (
                *self.configuration_tuple[:index],
                new_edge,
                *self.configuration_tuple[index + 1 :],
            )
            new_configuration = Configuration(new_configuration_tuple)
            new_configuration.contents.append(self.contents)
            new_configuration.generate_upwards_neighbors()

    return Configuration
