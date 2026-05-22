"""Directed simplicial lifting for the Dir-SNN model.

Materialises the four directed lower edge adjacencies and six directed
upper edge adjacencies introduced by Lecha et al. 2024
(*Higher-Order Topological Directionality and Directed Simplicial Neural
Networks*, ICASSP 2025, arXiv:2409.08389; Eqs. (3)-(4)). The
implementation is a faithful port of ``compute_lower_adj`` and
``compute_upper_adj`` from ``compute_adj.py`` in the upstream
reference repository; the only
differences are (i) numpy -> torch sparse, and (ii) an explicit
"canonical orientation" rule that turns an undirected TopoBench input
into a digraph in a deterministic way (see :meth:`_directed_edge_list`).
"""

from itertools import combinations

import networkx as nx
import torch
import torch_geometric
from toponetx.classes import SimplicialComplex

from topobench.transforms.liftings.graph2simplicial.base import (
    Graph2SimplicialLifting,
)

# Names of the 10 directed adjacencies, in a stable order. The first 4
# are the directed lower adjacencies of Eq. (3); the last 6 are the
# directed upper adjacencies of Eq. (4), matching the return order of
# ``compute_lower_adj`` / ``compute_upper_adj`` in the upstream repo.
DIR_LOWER_KEYS: tuple[str, ...] = (
    "dir_lower_adj_100",
    "dir_lower_adj_101",
    "dir_lower_adj_110",
    "dir_lower_adj_111",
)
DIR_UPPER_KEYS: tuple[str, ...] = (
    "dir_upper_adj_101",
    "dir_upper_adj_102",
    "dir_upper_adj_112",
    "dir_upper_adj_110",
    "dir_upper_adj_120",
    "dir_upper_adj_121",
)
DIR_ADJ_KEYS: tuple[str, ...] = DIR_LOWER_KEYS + DIR_UPPER_KEYS


class DirectedSimplicialLifting(Graph2SimplicialLifting):
    r"""Lift a graph to a simplicial complex with directed edge adjacencies.

    This lifting performs a clique-style 2-complex construction (nodes,
    edges, triangles) so that downstream TopoBench components keep
    working (``incidence_1``, ``incidence_2``, ``x_0``, ``x_1`` are all
    produced as for :class:`SimplicialCliqueLifting`).

    In addition it materialises the ten directed edge adjacencies of
    Eqs. (3)-(4) of Lecha et al. 2024 (arXiv:2409.08389) and stores them
    on the lifted batch as sparse COO tensors of shape
    ``(n_edges, n_edges)``:

    Lower (Eq. (3); ``compute_lower_adj`` of the upstream repo):

    - ``dir_lower_adj_100`` -- both edges leave the shared face
      (same ``src`` vertex).
    - ``dir_lower_adj_101`` -- :math:`\sigma_1` leaves, :math:`\sigma_2`
      enters (line-graph head-to-tail).
    - ``dir_lower_adj_110`` -- :math:`\sigma_1` enters, :math:`\sigma_2`
      leaves (transpose of ``101``).
    - ``dir_lower_adj_111`` -- both enter the shared face (same ``dst``
      vertex).

    Upper (Eq. (4); ``compute_upper_adj`` of the upstream repo):

    - ``dir_upper_adj_{101,102,112,110,120,121}`` -- triangle-mediated
      edge pairs; see the upstream :mod:`compute_adj` for the exact
      indexing convention.

    Canonical orientation for undirected inputs
    -------------------------------------------
    TopoBench graphs are stored with a symmetric ``edge_index``. For
    every undirected edge :math:`\{u, v\}` (with :math:`u \ne v`) we
    pick the canonical direction :math:`u \to v` whenever :math:`u < v`
    (i.e. the natural ordering of the simplex tuple
    ``(min(u,v), max(u,v))``). This is exactly the simplex ordering
    used by :class:`toponetx.SimplicialComplex` for the 1-skeleton, so
    the directed adjacencies are indexed by the same edge order as
    ``incidence_1``. With this canonical choice every undirected
    triangle :math:`\{a, b, c\}` (with :math:`a < b < c`) lands in
    branch 1 of ``compute_upper_adj`` ((t0,t1), (t0,t2), (t1,t2) all
    present in the digraph), making the upper adjacencies
    deterministic.

    Directed-input path (deferred)
    ------------------------------
    The ``directed_input=True`` code path is intentionally guarded with
    a :class:`NotImplementedError`. Under that mode the digraph's raw
    orientations would be preserved (e.g. ``(2, 0)``), but
    :class:`toponetx.SimplicialComplex` stores 1-cells as
    ``tuple(sorted((u, v)))`` and therefore exposes ``incidence_1``
    columns in canonical ``(min, max)`` order. The 10 directed
    adjacency matrices would then be indexed by the raw-orientation
    edge list while ``incidence_1`` and ``x_1`` use the canonical
    ordering, silently misaligning rows/columns. A correct
    implementation requires reconciling these two orderings (e.g. by
    threading a sign / permutation through ``incidence_1``); that work
    is deferred. To make the bug undiscoverable from configuration we
    raise early and force the configuration default to ``False``.

    Parameters
    ----------
    directed_input : bool, optional
        Must currently be ``False``. Reserved for future support of
        genuinely directed input graphs (Sec. IV of the paper); the
        ``True`` branch raises :class:`NotImplementedError` because it
        does not preserve the row/column alignment between the ten
        directed adjacencies and ``incidence_1`` / ``x_1``.
    spectral_normalize : bool, optional
        If ``True``, each of the ten directed adjacency matrices is
        divided by its largest eigenvalue before being stored on the
        lifted batch, mirroring the upstream
        :func:`spectral_normalization` helper in
        the upstream ``utils.py:16-20`` (which
        :mod:`train.py:29-33` applies to the four lower adjacencies
        before feeding the model). The default (``False``) preserves
        the raw binary directed adjacencies and is bit-identical to the
        pre-review-3 behaviour (so existing evaluation results remain
        valid). Set to ``True`` only when reproducing the official
        source-localization training pipeline; the dedicated
        ``configs/transforms/model_defaults/dirsnn_official_lower.yaml``
        wires this on for the four-lower-adjacency setup.
    **kwargs : optional
        Additional arguments forwarded to
        :class:`Graph2SimplicialLifting`. ``complex_dim`` is forced to
        2 because the paper only defines edge-level (1-cell) adjacencies
        through triangles.

    Raises
    ------
    NotImplementedError
        If ``directed_input=True``.
    """

    def __init__(
        self,
        directed_input: bool = False,
        spectral_normalize: bool = False,
        **kwargs,
    ):
        # Force complex_dim = 2: the paper's upper adjacencies are
        # mediated by triangles, so anything beyond 2 is irrelevant.
        kwargs["complex_dim"] = 2
        super().__init__(**kwargs)
        if directed_input:
            # See class docstring: the raw-orientation edge list does
            # not line up with the canonical ordering used by
            # SimplicialComplex.skeleton(1) / incidence_1, so the
            # adjacencies and x_1 would be silently misaligned.
            raise NotImplementedError(
                "DirectedSimplicialLifting(directed_input=True) is not "
                "supported yet: the raw directed edge list breaks the "
                "row/column alignment between the 10 directed "
                "adjacencies and incidence_1 / x_1 (which use canonical "
                "(min, max) edge ordering). Re-enable this path only "
                "after threading canonical orientation through "
                "incidence_1."
            )
        self.directed_input = directed_input
        self.spectral_normalize = spectral_normalize

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"directed_input={self.directed_input!r}, "
            f"spectral_normalize={self.spectral_normalize!r})"
        )

    # ------------------------------------------------------------------
    # Spectral normalization helper (port of utils.spectral_normalization)
    # ------------------------------------------------------------------

    @staticmethod
    def _spectral_normalize(adj: torch.Tensor) -> torch.Tensor:
        r"""Divide a (possibly non-symmetric) adjacency by its largest eigenvalue.

        Faithful port of :func:`spectral_normalization` in
        the upstream ``utils.py:16-20``::

            max_eig = torch.linalg.eigh(torch.tensor(adj_matrix))[0][-1]
            adj_matrix = adj_matrix / max_eig

        Two implementation notes worth surfacing:

        * The reference calls :func:`torch.linalg.eigh` indiscriminately,
          including on the non-symmetric line-graph-style lower
          adjacency ``adj_low_101`` (and on the six non-symmetric upper
          adjacencies). :func:`torch.linalg.eigh` is defined only for
          Hermitian inputs; PyTorch silently treats the lower triangle
          as the Hermitian matrix and returns real eigenvalues. We
          preserve this verbatim so that on the symmetric source-
          localization adjacencies (which is what the upstream
          ``train.py`` actually feeds it) we match the reference output
          bit-for-bit.
        * When the upstream eigh-on-non-symmetric path returns
          ``max_eig == 0`` (e.g. on the strictly upper-triangular
          ``adj_low_101`` of a small directed path), dividing would
          produce NaNs. We fall back to
          :func:`torch.linalg.eigvals` (taking the absolute-value max
          of the complex spectrum) in that case; this is the only
          intentional deviation from the reference and only triggers
          on the non-symmetric branch where the reference itself would
          NaN out.

        Parameters
        ----------
        adj : torch.Tensor
            Dense ``(n, n)`` float tensor.

        Returns
        -------
        torch.Tensor
            ``adj`` divided by the largest eigenvalue (real, from
            :func:`torch.linalg.eigh`; or absolute-max from
            :func:`torch.linalg.eigvals` on the degenerate fallback
            branch). For ``n == 0`` returns ``adj`` unchanged.
        """
        if adj.numel() == 0:
            return adj
        # Match the upstream verbatim: eigh on the (possibly non-
        # symmetric) dense adjacency, take the last (= largest)
        # eigenvalue. ``eigh`` returns eigenvalues in ascending order,
        # so ``[0][-1]`` is the maximum.
        eigvals = torch.linalg.eigh(adj)[0]
        max_eig = eigvals[-1]
        if max_eig.abs().item() == 0.0:
            # Degenerate eigh branch (matches the reference only on
            # symmetric inputs; the upstream code would NaN here).
            # Fall back to the general non-Hermitian solver and pick
            # the largest |lambda|. Coverage: this branch is exercised
            # only by tiny non-symmetric adjacencies where the strict
            # lower triangle is zero -- not used in practice.
            general = torch.linalg.eigvals(adj).abs()
            max_eig = general.max()
            if max_eig.item() == 0.0:
                return adj
        return adj / max_eig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _directed_edge_list(
        self, data: torch_geometric.data.Data
    ) -> list[tuple[int, int]]:
        r"""Return the canonical directed edge list of the input graph.

        Every undirected edge ``{u, v}`` is oriented as
        ``(min(u, v), max(u, v))``; the returned list is sorted
        lexicographically so its indices match the
        :class:`toponetx.SimplicialComplex` 1-skeleton ordering. Only
        the ``directed_input=False`` path is supported -- see the class
        docstring for why the directed-input path is currently guarded.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input batch with ``edge_index`` of shape ``(2, n_e)``.

        Returns
        -------
        list of tuple of int
            Canonically oriented directed edges sorted
            lexicographically.
        """
        ei = data.edge_index
        edges: set[tuple[int, int]] = set()
        for k in range(ei.shape[1]):
            u, v = int(ei[0, k]), int(ei[1, k])
            if u == v:
                continue  # drop self-loops, as the reference does
            a, b = (u, v) if u < v else (v, u)
            edges.add((a, b))
        return sorted(edges)

    # ------------------------------------------------------------------
    # Core directed-adjacency computation (port of compute_adj.py)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_lower_adj(
        edge_list: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Port of ``compute_lower_adj`` (Eq. (3) of arXiv:2409.08389).

        Parameters
        ----------
        edge_list : list of tuple of int
            Directed edges in canonical (lexicographic) order. Used both
            for indexing and to define the digraph.

        Returns
        -------
        tuple of torch.Tensor
            Four dense ``float32`` tensors of shape
            ``(n_edges, n_edges)`` for ``adj_low_100``, ``adj_low_101``,
            ``adj_low_110``, ``adj_low_111`` respectively. The caller
            converts them to sparse COO before storing on the batch.

        Notes
        -----
        Mirrors :func:`compute_lower_adj` from
        upstream ``compute_adj.py``:

        * ``adj_low_101`` is the line-graph adjacency of the digraph
          (``dst -> src`` chain).
        * ``adj_low_110`` is its transpose.
        * ``adj_low_100`` matches edges that share the same source
          (diagonal is kept, matching the reference).
        * ``adj_low_111`` matches edges that share the same target
          (diagonal is kept, matching the reference).

        This matches the paper's intent under the canonical edge
        ordering (``edge_list`` is the lexicographic order used by
        :class:`toponetx.SimplicialComplex.skeleton(1)` and
        :attr:`incidence_1`, so adjacency rows/cols are aligned with
        the edge features ``x_1``). The upstream
        ``compute_adj.compute_lower_adj`` path obtains ``adj_low_101``
        as ``nx.adjacency_matrix(nx.line_graph(G))``, whose node order
        follows ``nx.line_graph(G).nodes()`` rather than ``G.edges()``
        and is **not** guaranteed to match the canonical edge order.
        Using that path against an ``incidence_1`` built from
        ``G.edges()`` would silently misindex the 4 lower adjacencies
        against the edge feature tensor; the upstream code relies on
        the empirical alignment that holds for the small graphs in
        its experiments. Our direct vectorised construction sidesteps
        that alignment risk entirely.
        """
        n = len(edge_list)
        if n == 0:
            zeros = torch.zeros(0, 0)
            return zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone()

        srcs = torch.tensor([e[0] for e in edge_list])
        dsts = torch.tensor([e[1] for e in edge_list])

        # 100: same source -> reference uses
        #     ``edge_index[:,0] == edge_index[:,0].T``
        # which is symmetric *and* keeps the diagonal (an edge trivially
        # shares its own src with itself). We preserve this verbatim.
        adj_low_100 = (srcs.unsqueeze(1) == srcs.unsqueeze(0)).float()

        # 111: same target. Also keeps the diagonal, matching the
        # reference.
        adj_low_111 = (dsts.unsqueeze(1) == dsts.unsqueeze(0)).float()

        # 101: line graph adjacency of the digraph. Edges
        # (u_i, v_i) and (u_j, v_j) are 101-adjacent iff v_i == u_j,
        # i.e. one edge ends where another starts. This is exactly the
        # adjacency matrix of ``nx.line_graph(G)`` (ordered by our
        # canonical edge_list) -- but computed vectorially to avoid
        # depending on NetworkX's internal node ordering for line
        # graphs.
        adj_low_101 = (dsts.unsqueeze(1) == srcs.unsqueeze(0)).float()
        # Drop self-loops: ``nx.line_graph`` does not include them, and
        # the reference does not either (an edge being adjacent to
        # itself via ``dst == src`` would require ``u == v``, which we
        # already filter out as a self-loop above).
        adj_low_101.fill_diagonal_(0.0)
        adj_low_110 = adj_low_101.t().contiguous()

        return adj_low_100, adj_low_101, adj_low_110, adj_low_111

    @staticmethod
    def compute_upper_adj(
        edge_list: list[tuple[int, int]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        r"""Port of ``compute_upper_adj`` (Eq. (4) of arXiv:2409.08389).

        Parameters
        ----------
        edge_list : list of tuple of int
            Directed edges in canonical (lexicographic) order.

        Returns
        -------
        tuple of torch.Tensor
            Six dense ``float32`` tensors of shape
            ``(n_edges, n_edges)`` in the order
            ``(adj_up_101, adj_up_102, adj_up_112,
            adj_up_110, adj_up_120, adj_up_121)``, matching the upstream
            return signature.

        Notes
        -----
        Algorithm (lifted verbatim from
        the upstream ``compute_adj.py``):

        1. Enumerate all undirected 3-cycles via ``nx.simple_cycles``
           with ``length_bound=3`` on the *undirected* projection of the
           digraph.
        2. For each cycle ``{a, b, c}``, rotate/reflect the three
           vertices into the unique ordering ``(t0, t1, t2)`` such that
           the directed edges ``(t0,t1)``, ``(t0,t2)``, ``(t1,t2)`` are
           all present in the digraph. (When several rotations match,
           the reference picks the first one; with our canonical
           orientation only the first branch is ever selected.)
        3. Populate the six adjacencies according to the six paired
           ordering rules of Eq. (4).

        Edges that are not part of any (oriented) triangle contribute
        nothing.
        """
        n = len(edge_list)
        if n == 0:
            zeros = torch.zeros(0, 0)
            return (zeros.clone(),) * 6  # type: ignore[return-value]

        edge_set: set[tuple[int, int]] = set(edge_list)
        edge_to_id = {e: i for i, e in enumerate(edge_list)}

        # Enumerate undirected triangles via the underlying simple
        # graph. The reference implementation uses
        # ``nx.simple_cycles(G.to_undirected(), length_bound=3)`` on
        # NetworkX >= 3.0; we replace it with an explicit triangle
        # enumeration so the lifting works against the NetworkX 2.8
        # currently pinned in the TopoBench env. Semantics are
        # identical: ``simple_cycles`` of length 3 on an undirected
        # simple graph returns exactly the 3-cliques (each triangle is
        # reported as a single sorted triple).
        und = nx.Graph()
        und.add_edges_from(edge_list)
        nbrs = {v: set(und.neighbors(v)) for v in und.nodes}
        seen: set[tuple[int, int, int]] = set()
        for u, v in und.edges():
            common = nbrs[u] & nbrs[v]
            for w in common:
                seen.add(tuple(sorted((u, v, w))))  # type: ignore[arg-type]
        all_triangles = sorted(seen)

        dir_triangles: list[list[int]] = []
        for t in all_triangles:
            if (
                (t[0], t[1]) in edge_set
                and (t[0], t[2]) in edge_set
                and (t[1], t[2]) in edge_set
            ):
                dir_triangles.append(list(t))
            elif (
                (t[0], t[2]) in edge_set
                and (t[0], t[1]) in edge_set
                and (t[2], t[1]) in edge_set
            ):
                dir_triangles.append([t[0], t[2], t[1]])
            elif (
                (t[1], t[0]) in edge_set
                and (t[1], t[2]) in edge_set
                and (t[0], t[2]) in edge_set
            ):
                dir_triangles.append([t[1], t[0], t[2]])
            elif (
                (t[1], t[2]) in edge_set
                and (t[1], t[0]) in edge_set
                and (t[2], t[0]) in edge_set
            ):
                dir_triangles.append([t[1], t[2], t[0]])
            elif (
                (t[2], t[1]) in edge_set
                and (t[2], t[0]) in edge_set
                and (t[1], t[0]) in edge_set
            ):
                dir_triangles.append([t[2], t[1], t[0]])
            elif (
                (t[2], t[0]) in edge_set
                and (t[2], t[1]) in edge_set
                and (t[0], t[1]) in edge_set
            ):
                dir_triangles.append([t[2], t[0], t[1]])

        adj_up_101 = torch.zeros(n, n)
        adj_up_102 = torch.zeros(n, n)
        adj_up_112 = torch.zeros(n, n)
        adj_up_110 = torch.zeros(n, n)
        adj_up_120 = torch.zeros(n, n)
        adj_up_121 = torch.zeros(n, n)

        for t in dir_triangles:
            e_01 = edge_to_id[(t[0], t[1])]
            e_02 = edge_to_id[(t[0], t[2])]
            e_12 = edge_to_id[(t[1], t[2])]

            adj_up_101[e_12, e_02] = 1.0
            adj_up_102[e_12, e_01] = 1.0
            adj_up_112[e_02, e_01] = 1.0
            adj_up_110[e_02, e_12] = 1.0
            adj_up_120[e_01, e_12] = 1.0
            adj_up_121[e_01, e_02] = 1.0

        return (
            adj_up_101,
            adj_up_102,
            adj_up_112,
            adj_up_110,
            adj_up_120,
            adj_up_121,
        )

    # ------------------------------------------------------------------
    # Lifting entry point
    # ------------------------------------------------------------------

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lift the graph to a 2-simplicial complex with directed adjacencies.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            A dictionary with the standard TopoBench keys
            (``x_0``, ``incidence_1``, ``incidence_2``, ``down/up
            laplacians`` etc., produced by
            :func:`get_complex_connectivity`) plus ten sparse
            directed-adjacency tensors keyed by
            :data:`DIR_LOWER_KEYS` / :data:`DIR_UPPER_KEYS`.
        """
        # 1) Standard clique 2-complex (nodes, edges, triangles) so the
        # rest of TopoBench keeps working unchanged. Self-loops are
        # stripped because ``toponetx.SimplicialComplex`` rejects them
        # (a 1-simplex must have two distinct vertices).
        graph = self._generate_graph_from_data(data)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        simplicial_complex = SimplicialComplex(graph)

        triangles: set[tuple[int, int, int]] = set()
        for clique in nx.find_cliques(graph):
            for c in combinations(sorted(clique), 3):
                triangles.add(c)
        simplicial_complex.add_simplices_from(list(triangles))

        lifted_topology = self._get_lifted_topology(simplicial_complex, graph)

        # 2) Directed adjacencies. Both the edge ordering inside
        # ``edge_list`` and the simplex ordering inside
        # ``simplicial_complex.skeleton(1)`` are obtained by sorting the
        # 2-tuples lexicographically, so the row/column indices of the
        # ten adjacency matrices line up with ``incidence_1``'s columns
        # by construction.
        edge_list = self._directed_edge_list(data)
        n_edges = len(edge_list)

        adj_low_100, adj_low_101, adj_low_110, adj_low_111 = (
            self.compute_lower_adj(edge_list)
        )
        (
            adj_up_101,
            adj_up_102,
            adj_up_112,
            adj_up_110,
            adj_up_120,
            adj_up_121,
        ) = self.compute_upper_adj(edge_list)

        dense_adjs = {
            "dir_lower_adj_100": adj_low_100,
            "dir_lower_adj_101": adj_low_101,
            "dir_lower_adj_110": adj_low_110,
            "dir_lower_adj_111": adj_low_111,
            "dir_upper_adj_101": adj_up_101,
            "dir_upper_adj_102": adj_up_102,
            "dir_upper_adj_112": adj_up_112,
            "dir_upper_adj_110": adj_up_110,
            "dir_upper_adj_120": adj_up_120,
            "dir_upper_adj_121": adj_up_121,
        }

        for key, dense in dense_adjs.items():
            # Optional opt-in spectral normalization for reference parity.
            # Mirrors ``repos/DirSNN/utils.py:16`` applied to each of the
            # 4 lower adjacencies in ``repos/DirSNN/train.py:29-33``. We
            # extend that to all 10 adjacencies for symmetry: the
            # ``dirsnn_official_lower`` wrapper only consumes the lower
            # four, so the upper six are still computed but ignored.
            if self.spectral_normalize:
                dense = self._spectral_normalize(dense)
            # Force a sparse COO layout regardless of density: TopoBench
            # collates sparse tensors block-diagonally during batching,
            # which is exactly what we want for the directed
            # adjacencies too.
            if dense.numel() == 0:
                lifted_topology[key] = torch.sparse_coo_tensor(
                    torch.empty(2, 0, dtype=torch.long),
                    torch.empty(0),
                    size=(n_edges, n_edges),
                ).coalesce()
            else:
                lifted_topology[key] = dense.to_sparse().coalesce()

        return lifted_topology
