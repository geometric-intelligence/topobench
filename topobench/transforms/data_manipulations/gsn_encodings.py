"""GSN structural encodings exposed as a cached pre-transform.

The heavy lifting (automorphism-orbit partitioning and subgraph-isomorphism
counting) already lives in :class:`GSNFeatureEncoder`, whose ``forward`` maps a
``Data`` object to the same object annotated with node/edge orbit encodings --
exactly the signature of a ``torch_geometric`` transform. We therefore reuse it
verbatim and only add a thin subclass so the data-manipulation registry
auto-discovers it by name and so the substructures can be specified with plain
integer lengths in the transform config (the transform path does not run through
Hydra, so nested ``_target_`` substructure specs would not get instantiated).

Applied as a ``pre_transform`` by ``PreProcessor``, the encodings are computed
once and cached to disk; the model's ``GSNFeatureEncoder`` then short-circuits
(see its ``forward`` guard) because the fields are already present on the batch.
"""

import networkx as nx

from topobench.nn.encoders.gsn_encoder import GSNFeatureEncoder


class GSNEncodings(GSNFeatureEncoder):
    """GSN orbit-count encodings as a data-manipulation transform.

    Parameters
    ----------
    cycle_lengths : list of int, optional
        Lengths of the cycle-graph substructures to count. Default is
        ``[3, 4, 5, 6]``.
    path_lengths : list of int, optional
        Number of nodes of the path-graph substructures to count. Default is
        ``[3, 4, 5]``.
    pyg_kword : str, optional
        Base name for the attributes storing the encodings; the node/edge
        attributes are named ``f"node_{pyg_kword}"`` / ``f"edge_{pyg_kword}"``.
        Default is ``"gsn_encodings"``.
    **kwargs : dict, optional
        Additional keyword arguments (e.g. ``transform_name`` /
        ``transform_type`` injected by the transform registry). Absorbed by the
        base encoder's ``**kwargs``.
    """

    def __init__(
        self,
        cycle_lengths: list[int] | None = None,
        path_lengths: list[int] | None = None,
        pyg_kword: str = "gsn_encodings",
        **kwargs,
    ):
        cycle_lengths = (
            [3, 4, 5, 6] if cycle_lengths is None else cycle_lengths
        )
        path_lengths = [3, 4, 5] if path_lengths is None else path_lengths
        substructures = [nx.cycle_graph(n) for n in cycle_lengths] + [
            nx.path_graph(n) for n in path_lengths
        ]
        # eager precompute: this is a one-off pre_transform, not a hot path
        super().__init__(
            substructures=substructures,
            pyg_kword=pyg_kword,
            lazy=False,
            **kwargs,
        )

    def forward(self, data):
        """Compute and attach the GSN encodings to a single graph.

        Unlike the base class (a passthrough that expects the encodings to
        already be present), the transform is the producer: it drives the
        heavy per-graph computation via :meth:`GSNFeatureEncoder._encode`.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input graph to annotate.

        Returns
        -------
        torch_geometric.data.Data
            The graph with node/edge GSN orbit encodings attached.
        """
        return self._encode(data)
