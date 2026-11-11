"""
GSN (Graph Substructure Network) feature encoding utilities.

This module provides helper functions for converting between PyTorch
Geometric `Data` objects and `networkx.Graph` objects, and a
`GSNFeatureEncoder` that augments node and edge features with counts
of how often each node/edge participates in each automorphism orbit
of a fixed collection of substructures (motifs), following the
Graph Substructure Network (GSN) approach.
"""

import warnings
from collections import namedtuple

import networkx as nx
import numpy as np
import torch
import torch_geometric
from joblib import Parallel, delayed
from networkx.algorithms.isomorphism import GraphMatcher
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx

from topobench.nn.encoders.base import AbstractFeatureEncoder

OrbitInformation = namedtuple(
    "OrbitInformation",
    ["node_orbit_partition", "edge_orbit_partition", "number_automorphisms"],
)


# we need to make sure that the roles of the nodes are contiguous, i.e. start at 0,1,2,3,4 etc.
def make_contiguous(ii: dict) -> dict:
    """
    Remap dictionary values to contiguous integers starting at 0.

    The relative order of the (unique, sorted) values is preserved:
    the smallest original value is mapped to 0, the next-smallest to
    1, and so on. Keys are left unchanged.

    Parameters
    ----------
    ii : dict
        Dictionary with possibly non-contiguous integer values.

    Returns
    -------
    dict
        Dictionary with the same keys as `ii`, with values remapped to
        contiguous integers (0, 1, 2, ...).
    """
    unique_vals = list(set(ii.values()))
    unique_vals.sort()
    mapping = {unique_vals[k]: k for k in range(len(unique_vals))}

    nd = {}

    for k, v in ii.items():
        nd[k] = mapping[v]

    return nd


def invert_injective_dict(ii: dict) -> dict:
    """
    Invert a dictionary representing an injective mapping.

    Swaps the roles of keys and values, i.e. returns a dictionary that
    maps each original value back to its corresponding key.

    Parameters
    ----------
    ii : dict
        Dictionary representing an injective mapping (i.e. with unique
        values, so the inversion is well defined).

    Returns
    -------
    dict
        Dictionary with the roles of keys and values exchanged
        relative to `ii`.
    """
    inverted_dict: dict = {v: k for k, v in ii.items()}

    return inverted_dict


def pyg_to_nx(data: Data) -> nx.Graph:
    """
    Convert a PyTorch Geometric `Data` object to a `networkx.Graph`.

    Node, edge, and graph-level attributes are detected automatically
    (via `data.is_node_attr` / `data.is_edge_attr`) rather than having
    to be passed in explicitly, and are copied over onto the resulting
    graph. `edge_index` and `num_nodes` are excluded from attribute
    detection, since they are handled internally by
    `torch_geometric.utils.to_networkx`. The resulting graph is a
    simple, undirected `nx.Graph` with self-loops removed.

    Parameters
    ----------
    data : torch_geometric.data.Data
        `Data` object that will be transformed into a `networkx.Graph`
        instance.

    Returns
    -------
    nx.Graph
        The resulting `networkx.Graph`, with node, edge, and graph
        attributes copied over from `data`.

    Notes
    -----
    Written by Claude Sonnet 4.6 (Medium).
    """

    exclude = {"edge_index", "num_nodes"}
    keys = [k for k, _ in data if k not in exclude]

    node_attrs, edge_attrs, graph_attrs = [], [], []
    for k in keys:
        if data.is_node_attr(k):
            node_attrs.append(k)
        elif data.is_edge_attr(k):
            edge_attrs.append(k)
        else:
            graph_attrs.append(k)

    G = to_networkx(
        data,
        node_attrs=node_attrs,
        edge_attrs=edge_attrs,
        graph_attrs=graph_attrs,
        to_multi=False,  # keeps parallel edges instead of silently merging them
        to_undirected=True,  # keeps both directions distinct, avoids attr-merging ambiguity
        remove_self_loops=True,
    )

    return G


def nx_to_pyg(G: nx.Graph) -> Data:
    """
    Convert a `networkx.Graph` to a PyTorch Geometric `Data` object.

    Node and edge attributes are picked up automatically by name via
    `torch_geometric.utils.from_networkx`. Since `from_networkx` does
    not transfer graph-level attributes, any entries in `G.graph` are
    additionally copied over manually: each value is converted to a
    `torch.tensor` where possible, and stored as-is otherwise (e.g.
    for values that cannot be converted to a tensor).

    Parameters
    ----------
    G : nx.Graph
        A `networkx` graph to be turned into its PyTorch Geometric
        equivalent.

    Returns
    -------
    Data
        The PyTorch Geometric equivalent of the input graph, including
        any graph-level attributes from `G.graph`.

    Notes
    -----
    Written by Claude Sonnet 4.6 (Medium).
    """
    data = from_networkx(G)  # picks up all node/edge attrs by name already
    for (
        k,
        v,
    ) in G.graph.items():  # from_networkx ignores G.graph, so copy it manually
        try:
            data[k] = torch.tensor(v)
        except (ValueError, TypeError):
            data[k] = v
    return data


def normalized_edge(node1: int, node2: int) -> tuple[int, int]:
    """
    Return an undirected edge as a canonically ordered node pair.

    Orders the two endpoints so the smaller node id comes first, giving
    a single, orientation-independent key for an undirected edge. This
    lets ``(u, v)`` and ``(v, u)`` map to the same dictionary key, which
    is relied upon when keying and looking up edge orbits (whose stored
    orientation in ``substructure.edges`` is not guaranteed to be
    ascending).

    Parameters
    ----------
    node1 : int
        One endpoint of the edge.
    node2 : int
        The other endpoint of the edge.

    Returns
    -------
    tuple of (int, int)
        The endpoints as ``(min(node1, node2), max(node1, node2))``.
    """
    return min(node1, node2), max(node1, node2)


class GSNFeatureEncoder(AbstractFeatureEncoder):
    """
    Graph Substructure Network (GSN) feature encoder.

    Augments node and edge features with structural counts of how
    often each node/edge participates in each automorphism orbit of
    each substructure in a given collection. For every substructure,
    the node and edge automorphism orbits are computed once and given
    a global, contiguous numbering; then, for an input graph, every
    subgraph-isomorphic embedding of each substructure into the graph
    is found, and orbit occurrence counts are accumulated into a
    feature vector per node and per edge. Counts are normalized by the
    number of automorphisms of the substructure that the corresponding
    orbit belongs to.

    Parameters
    ----------
    substructures : list of nx.Graph
        Collection of substructures (motifs) to search for and count
        occurrences of within each input graph.
    pyg_kword : str, optional
        Base name used for the attributes that store the resulting
        encodings on the input/output `torch_geometric.data.Data`
        object. The node attribute is named ``f"node_{pyg_kword}"``
        and the edge attribute is named ``f"edge_{pyg_kword}"``.
        Default is ``"gsn_encodings"``.
    lazy : bool, optional
        If `False`, orbit partitions for all substructures are
        computed immediately upon construction (via `_precompute`).
        If `True` (default), this computation is deferred until the
        first call to `forward`.
    n_jobs : int, optional
        Number of parallel workers used by `_gsn_encoding` to annotate
        a graph, one job per substructure (passed straight to
        `joblib.Parallel`). ``-1`` (default) uses all available cores;
        ``1`` runs in-process with no serialization overhead.
        Parallelism is only worth it for graphs where the
        subgraph-isomorphism search dominates the per-task overhead of
        pickling the graph to each worker. Note that the encodings are
        computed by a cached pre-transform, so set this to ``1`` if that
        pre-transform itself already runs inside parallel workers, to
        prevent CPU oversubscription.

    **kwargs : dict, optional
        Additional keyword arguments. Ignored by the encoder; accepted so
        that config-level anchors (e.g. ``encoder_name`` / ``out_channels``
        referenced elsewhere in the model config) can be co-located on the
        encoder block, mirroring `AllCellFeatureEncoder`.

    Attributes
    ----------
    _substructures : list of nx.Graph
        The substructures passed at construction time.
    _node_pyg_kword : str
        Name of the node attribute used to store encodings.
    _edge_pyg_kword : str
        Name of the edge attribute used to store encodings.
    _lazy : bool
        Whether orbit-partition computation was deferred at
        construction time.
    _precomputed : bool
        Whether `_precompute` has already been run.
    _substructure_orbits : dict of int to OrbitInformation or None
        Per-substructure orbit information, keyed by substructure
        index, populated by `_precompute`.
    _node_channels : int
        Total number of distinct node orbits across all substructures
        (i.e. the length of the per-node encoding vector).
    _edge_channels : int
        Total number of distinct edge orbits across all substructures
        (i.e. the length of the per-edge encoding vector).
    _normalization_vector_n : np.array or None
        Per-orbit node normalization constants (number of
        automorphisms of the owning substructure), populated by
        `_precompute`.
    _normalization_vector_e : np.array or None
        Per-orbit edge normalization constants (number of
        automorphisms of the owning substructure), populated by
        `_precompute`.
    """

    def __init__(
        self,
        substructures: list[nx.Graph] | None = None,
        pyg_kword: str = "gsn_encodings",
        lazy: bool = True,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        Initialize the encoder and optionally precompute orbit partitions.

        See the class docstring for a description of the parameters.
        If `lazy` is `False`, this eagerly calls `_precompute` so that
        `_substructure_orbits` and the normalization vectors are ready
        before `forward` is first called. Extra keyword arguments are
        accepted and ignored (see ``**kwargs`` in the class docstring).
        """
        super().__init__()

        self._substructures = substructures

        self._node_pyg_kword = f"node_{pyg_kword}"
        self._edge_pyg_kword = f"edge_{pyg_kword}"

        self._lazy = lazy  # whether or not orbit partitions should be computed on instance creation or first call to forward
        self._n_jobs = n_jobs  # joblib workers used to parallelize over substructures in `_gsn_encoding`

        self._precomputed = False
        self._substructure_orbits: (
            dict[int, tuple[dict[int, int], dict[tuple[int, int], int], int]]
            | None
        ) = None
        self._node_channels: int = -1
        self._edge_channels: int = -1
        self._normalization_vector_n: torch.Tensor | None = None
        self._normalization_vector_e: torch.Tensor | None = None

        if not self._lazy:
            self._precompute()

    @staticmethod
    def _get_substructure_orbit_partition(
        substructure: nx.Graph,
    ) -> OrbitInformation:
        """
        Compute node and edge automorphism orbits of a substructure.

        The automorphism group of `substructure` is obtained by
        enumerating all self-isomorphisms (i.e.
        `GraphMatcher(substructure, substructure).isomorphisms_iter()`).
        Nodes (respectively, edges) that are mapped onto one another by
        at least one automorphism are merged into the same orbit, and
        orbit identifiers are then remapped to contiguous integers
        starting at 0 via `make_contiguous`.

        Parameters
        ----------
        substructure : nx.Graph
            The substructure (motif) whose automorphism orbits are to
            be computed.

        Returns
        -------
        OrbitInformation
            Named tuple with fields:

            node_orbit_partition : dict of int to int
                Mapping from each node id in `substructure` to its
                (contiguous) orbit id.
            edge_orbit_partition : dict of (int, int) to int
                Mapping from each edge `(u, v)` in `substructure` to
                its (contiguous) orbit id.
            number_automorphisms : int
                Number of automorphisms found for `substructure`.
        """
        # step 1: we need to find the orbits of the substructure first
        # how do we do this? -> easy, we just search for automorphisms via searching for isomorphisms
        # AUT(G) = list_isomorphisms(G,G)

        automorphism_group: list[dict[int, int]] = list(
            GraphMatcher(substructure, substructure).isomorphisms_iter()
        )

        # now we need to partition the nodes according to their orbits

        # we initialize all nodes as having separate roles and then collapse via the automorphism group
        node_orbits: dict[int, int] = {
            node_id: node_id for node_id in substructure.nodes
        }
        edge_orbits: dict[tuple[int, int], int] = {
            normalized_edge(u, v): idx
            for idx, (u, v) in enumerate(substructure.edges)
        }

        for isom in automorphism_group:
            for original_node_id, mapped_node_id in isom.items():
                node_role = min(
                    node_orbits[original_node_id], node_orbits[mapped_node_id]
                )

                node_orbits[original_node_id] = node_role
                node_orbits[mapped_node_id] = node_role

            # TODO: there must be a better way to do this, now we are iterating over all nodes and all edges
            # On the other hand the substructures are usually of limited size so this shouldnt make too much of a difference
            for ov, ou in substructure.edges:
                ov, ou = normalized_edge(ov, ou)  # normalize edge

                mv, mu = min(isom[ov], isom[ou]), max(isom[ov], isom[ou])

                edge_role = min(edge_orbits[(ov, ou)], edge_orbits[(mv, mu)])

                edge_orbits[(ov, ou)] = edge_role
                edge_orbits[(mv, mu)] = edge_role

        # make orbit ids contoguous integers such that we can easily use them for indexing
        node_orbits = make_contiguous(node_orbits)
        edge_orbits = make_contiguous(edge_orbits)

        return OrbitInformation(
            node_orbits, edge_orbits, len(automorphism_group)
        )

    def _precompute(self) -> None:
        """
        Compute and cache orbit partitions for all substructures.

        For every substructure in `self._substructures`, computes its
        node and edge orbit partitions via
        `_get_substructure_orbit_partition`, offsets the resulting
        orbit ids so they form a single global, contiguous numbering
        across all substructures, and stores the result in
        `self._substructure_orbits`. Also sets `self._node_channels`
        and `self._edge_channels` to the total number of distinct
        node/edge orbits across all substructures, and builds the
        per-orbit normalization vectors
        `self._normalization_vector_n` / `self._normalization_vector_e`,
        each entry of which holds the number of automorphisms of the
        substructure that the corresponding orbit belongs to.

        This method is idempotent: if orbit partitions have already
        been computed (`self._precomputed` is `True`), it returns
        immediately without recomputing anything.
        """
        if self._precomputed:
            return

        assert self._substructure_orbits is None

        self._substructure_orbits = dict()

        node_channels: int = 0
        edge_channels: int = 0

        nodenormvecs: list[np.array] = list()
        edgenormvecs: list[np.array] = list()

        for j, subs in enumerate(self._substructures):
            tmp = self._get_substructure_orbit_partition(subs)

            # how many different node / edge orbits are there in the current substructure
            num_orbits_n: int = len(
                list(set(tmp.node_orbit_partition.values()))
            )
            num_orbits_e: int = len(
                list(set(tmp.edge_orbit_partition.values()))
            )

            # we offset the orbit ids in the current substructure to get a global numbering
            for k in tmp.node_orbit_partition:
                tmp.node_orbit_partition[k] += node_channels
            for k in tmp.edge_orbit_partition:
                tmp.edge_orbit_partition[k] += edge_channels

            node_channels += num_orbits_n
            edge_channels += num_orbits_e

            self._substructure_orbits[j] = tmp

            nodenormvecs.append(
                np.ones(num_orbits_n)
                * self._substructure_orbits[j].number_automorphisms
            )

            edgenormvecs.append(
                np.ones(num_orbits_e)
                * self._substructure_orbits[j].number_automorphisms
            )

        self._node_channels = node_channels
        self._edge_channels = edge_channels

        # we should also create the normalization vectors here
        self._normalization_vector_n = torch.tensor(
            np.concatenate(nodenormvecs)
        )
        self._normalization_vector_e = torch.tensor(
            np.concatenate(edgenormvecs)
        )

        self._precomputed = True

    @staticmethod
    def _process_one_substructure(
        G: nx.Graph,
        H: nx.Graph,
        orbit_info: OrbitInformation,
        node_channels: int,
        edge_channels: int,
    ) -> tuple[dict[int, np.array], dict[tuple[int, int], np.array]]:
        """
        Accumulate raw orbit counts for a single substructure.

        Finds every subgraph isomorphism (embedding) of `H` into `G`
        and, for each embedding, increments the count at the
        corresponding global orbit index for every matched node and
        edge. Counts are accumulated into local dictionaries and
        returned rather than written onto `G`, so that this can run in
        a separate worker process (see `_gsn_encoding`) without sharing
        mutable state.

        Parameters
        ----------
        G : nx.Graph
            Target graph the substructure is searched in.
        H : nx.Graph
            Substructure (motif) to find embeddings of.
        orbit_info : OrbitInformation
            Orbit information for `H`, with node/edge orbit ids already
            offset into the global numbering (see `_precompute`).
        node_channels : int
            Length of the per-node count vector (total node orbits).
        edge_channels : int
            Length of the per-edge count vector (total edge orbits).

        Returns
        -------
        node_counts : dict of int to np.array
            Per-node raw orbit counts, keyed by node id in `G`.
        edge_counts : dict of (int, int) to np.array
            Per-edge raw orbit counts, keyed by normalized edge
            ``(min, max)`` in `G`.
        """
        node_counts: dict[int, np.array] = {
            node: np.zeros(node_channels) for node in G.nodes
        }
        edge_counts: dict[tuple[int, int], np.array] = {
            normalized_edge(u, v): np.zeros(edge_channels) for u, v in G.edges
        }

        # iterate over every embedding of the substructure into G and
        # accumulate counts (NB: this loop body must stay inside the loop —
        # accumulating only the last match is the bug in the original draft)
        for matching_map in GraphMatcher(G, H).subgraph_isomorphisms_iter():
            # pre-compute inverted matching_map
            inverted_matching_map: dict[int, int] = invert_injective_dict(
                matching_map
            )

            # nodes:
            for node_id_G, node_id_H in matching_map.items():
                # node orbits are globally contigous integers
                corresponding_orbit = orbit_info.node_orbit_partition[
                    node_id_H
                ]
                node_counts[node_id_G][corresponding_orbit] += 1

            # edges:
            for Hu, Hv in H.edges:
                Gu, Gv = normalized_edge(
                    inverted_matching_map[Hu], inverted_matching_map[Hv]
                )
                minH, maxH = normalized_edge(Hu, Hv)

                edge_role = orbit_info.edge_orbit_partition[minH, maxH]
                edge_counts[(Gu, Gv)][edge_role] += 1

        return node_counts, edge_counts

    def _gsn_encoding(self, G: nx.Graph) -> nx.Graph:
        """
        Annotate a graph with raw (unnormalized) GSN orbit counts.

        Initializes a zero-valued node attribute (named
        `self._node_pyg_kword`) of length `self._node_channels` on
        every node of `G`, and a zero-valued edge attribute (named
        `self._edge_pyg_kword`) of length `self._edge_channels` on
        every edge of `G`. Then processes each substructure (one
        `joblib` job per substructure, controlled by `self._n_jobs`),
        finding every subgraph isomorphism (embedding) of the
        substructure into `G` and, for each embedding found,
        incrementing the count at the corresponding global orbit index
        for every matched node and edge. Per-substructure counts are
        accumulated in the workers (see `_process_one_substructure`)
        and merged back into `G`.

        Parameters
        ----------
        G : nx.Graph
            Input graph to annotate. Modified in place.

        Returns
        -------
        nx.Graph
            The same graph object `G`, with `self._node_pyg_kword` and
            `self._edge_pyg_kword` attributes set on its nodes and
            edges respectively, containing raw (not yet normalized)
            orbit occurrence counts.

        Notes
        -----
        Requires `self._substructure_orbits` to already be populated
        (see `_precompute`).
        """
        # initialize empty GSN embedding vectors:
        nx.set_node_attributes(
            G,
            {k: np.zeros(self._node_channels) for k in G.nodes},
            self._node_pyg_kword,
        )
        nx.set_edge_attributes(
            G,
            {(u, v): np.zeros(self._edge_channels) for u, v in G.edges},
            self._edge_pyg_kword,
        )

        # run each substructure independently; n_jobs=1 uses joblib's
        # sequential backend (in-process, no pickling), so this is a single
        # code path for both the serial and parallel cases.
        results = Parallel(n_jobs=self._n_jobs)(
            delayed(self._process_one_substructure)(
                G,
                H,
                self._substructure_orbits[j],
                self._node_channels,
                self._edge_channels,
            )
            for j, H in enumerate(self._substructures)
        )

        # merge per-substructure counts back into G
        for node_counts, edge_counts in results:
            for node, counts in node_counts.items():
                G.nodes[node][self._node_pyg_kword] += counts
            for (u, v), counts in edge_counts.items():
                G.edges[u, v][self._edge_pyg_kword] += counts

        return G

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        """
        Return the batch with GSN encodings, computing them if missing.

        As a model feature encoder this class is normally a pure
        passthrough: the structural encodings are computed once and cached
        by the ``GSNEncodings`` data-manipulation pre-transform
        (auto-attached via ``configs/transforms/model_defaults/gsn.yaml``)
        and then ride along on every collated batch. If the encodings are
        missing the transform was not applied; rather than fail, we warn
        and fall back to computing them on the fly via `_encode`. This is
        expensive inside the training loop, so the warning flags that the
        pre-transform should be enabled.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input batch, expected to already carry the GSN encodings.

        Returns
        -------
        torch_geometric.data.Data
            The input ``data`` with the GSN encodings attached (either the
            precomputed ones, or freshly computed as a fallback).
        """
        if (
            self._node_pyg_kword in data
            and data[self._node_pyg_kword] is not None
        ):
            return data

        warnings.warn(
            f"'{self._node_pyg_kword}' not found on the batch; recomputing "
            "the GSN encodings on the fly. This is expensive inside the "
            "training loop -- precompute them with the `GSNEncodings` "
            "data-manipulation transform (auto-attached via "
            "configs/transforms/model_defaults/gsn.yaml) to avoid this.",
            stacklevel=2,
        )

        return self._encode(data)

    def _encode(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        """
        Compute and attach GSN structural encodings to a PyG graph.

        Converts `data` to a `networkx.Graph` (via `pyg_to_nx`),
        annotates it with raw GSN orbit-occurrence counts (via
        `_gsn_encoding`), converts the result back to a
        `torch_geometric.data.Data` object (via `nx_to_pyg`), and
        normalizes the resulting node and edge encodings by dividing
        by the number of automorphisms of the substructure that each
        orbit belongs to.

        This is the heavy path, driven once per graph by the
        ``GSNEncodings`` pre-transform; the model feature encoder never
        calls it (see ``forward``).

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph data object to encode.

        Returns
        -------
        torch_geometric.data.Data
            The input graph data, with two additional attributes set:
            ``f"node_{pyg_kword}"`` and ``f"edge_{pyg_kword}"``, each
            holding the normalized per-node / per-edge GSN orbit count
            vectors.
        """

        if self._lazy:
            self._precompute()

        # step 1 turn into networkx nx.Graph object
        data_nx: nx.Graph = pyg_to_nx(data)

        # annotate graph
        data_nx = self._gsn_encoding(data_nx)

        # final step: turn back into pyg.data.Data object:
        data_pyg = nx_to_pyg(data_nx)

        # the networkx round-trip rebuilds everything on CPU; remember the
        # original batch device so the result can be moved back. Otherwise a
        # CPU encoding fed into an on-device backbone raises e.g.
        # "Placeholder storage has not been allocated on MPS device!".
        ref_device = None
        for kw in ("x", "x_0", "edge_index"):
            t = data.get(kw, None)
            if t is not None:
                ref_device = t.device
                break

        # reference dtype for the encodings: match the node features. TopoBench
        # stores features under `x_0` (leaving `x` unset), so fall back
        # `x` -> `x_0` -> default float dtype rather than assuming `data.x`
        # exists.
        feature_dtype = None
        for kw in ("x", "x_0"):
            feat = data_pyg.get(kw, None)
            if feat is not None:
                feature_dtype = feat.dtype
                break
        if feature_dtype is None:
            feature_dtype = torch.get_default_dtype()

        # an edgeless (or nodeless) graph produces no corresponding attribute
        # in `from_networkx`, so materialize empty encodings to keep the
        # normalization below well-defined.
        if self._node_pyg_kword not in data_pyg:
            data_pyg[self._node_pyg_kword] = torch.zeros(
                (0, self._node_channels), dtype=feature_dtype
            )
        if self._edge_pyg_kword not in data_pyg:
            data_pyg[self._edge_pyg_kword] = torch.zeros(
                (0, self._edge_channels), dtype=feature_dtype
            )

        # here we batch-apply the normalization by number of matchings:
        data_pyg[self._node_pyg_kword] = (
            data_pyg[self._node_pyg_kword] / self._normalization_vector_n
        )
        data_pyg[self._edge_pyg_kword] = (
            data_pyg[self._edge_pyg_kword] / self._normalization_vector_e
        )

        # ensure the encodings share the node-feature dtype
        data_pyg[self._node_pyg_kword] = data_pyg[self._node_pyg_kword].to(
            dtype=feature_dtype
        )
        data_pyg[self._edge_pyg_kword] = data_pyg[self._edge_pyg_kword].to(
            dtype=feature_dtype
        )

        # move the reconstructed graph back onto the original batch's device
        # (the round-trip above produced CPU tensors)
        if ref_device is not None:
            data_pyg = data_pyg.to(ref_device)

        return data_pyg
