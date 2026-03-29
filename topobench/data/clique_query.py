"""Clique query engine for efficient clique lookup in large graphs.

Detects and indexes cliques for on-demand querying during mini-batch training.
Cliques are stored in SQLite for memory-efficient access without loading
all cliques into RAM.
"""

from pathlib import Path
from typing import Any

import networkx as nx

from topobench.data.clique_detection import build_clique_index
from topobench.data.index import SQLiteIndexBackend


class CliqueQueryEngine:
    """Query engine for clique lookup during transductive mini-batch training.

    Parameters
    ----------
    graph : nx.Graph
        The graph to query cliques from.
    index_dir : str or Path
        Directory for storing the clique index.
    max_clique_size : int, optional
        Maximum clique size to index (e.g., 3 for triangles).
        If None, indexes all maximal cliques (default: None).
    backend : str, optional
        Backend type to use ('sqlite' or 'auto'). Default: 'auto'.
    force_rebuild : bool, optional
        If True, rebuild index even if it exists (default: False).

    Attributes
    ----------
    graph : nx.Graph
        The input graph.
    backend : AbstractIndexBackend
        The storage backend for clique queries.
    num_cliques : int
        Total number of cliques indexed.

    Examples
    --------
    >>> import networkx as nx
    >>> from topobench.data.clique_query import CliqueQueryEngine
    >>>
    >>> # Create query engine for Karate Club graph
    >>> G = nx.karate_club_graph()
    >>> engine = CliqueQueryEngine(G, index_dir="/tmp/karate_index", max_clique_size=3)
    >>> engine.build_index()
    >>>
    >>> # Query cliques for a batch of nodes
    >>> batch_nodes = [0, 1, 2, 3, 4]
    >>> cliques = engine.query_batch(batch_nodes)
    >>> print(f"Found {len(cliques)} cliques in batch")
    >>>
    >>> engine.close()
    """

    def __init__(
        self,
        graph: nx.Graph,
        index_dir: str | Path,
        max_clique_size: int | None = None,
        backend: str = "auto",
        force_rebuild: bool = False,
    ) -> None:
        """Initialize clique query engine.

        Parameters
        ----------
        graph : nx.Graph
            The graph to query cliques from.
        index_dir : str or Path
            Directory for storing the clique index.
        max_clique_size : int, optional
            Maximum clique size to index (default: None for all maximal).
        backend : str, optional
            Backend type ('sqlite' or 'auto', default: 'auto').
        force_rebuild : bool, optional
            Rebuild index even if exists (default: False).
        """
        self.graph = graph
        self.index_dir = Path(index_dir)
        self.max_clique_size = max_clique_size
        self.force_rebuild = force_rebuild

        # Initialize backend
        if backend == "auto" or backend == "sqlite":
            self.backend = SQLiteIndexBackend(data_dir=str(self.index_dir))
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self.num_cliques = 0
        self._is_open = False

    def open(self) -> None:
        """Open connection to index backend.

        Raises
        ------
        RuntimeError
            If opening backend fails.
        """
        self.backend.open()
        self._is_open = True

    def close(self) -> None:
        """Close connection to index backend."""
        if self._is_open:
            self.backend.close()
            self._is_open = False

    def build_index(self) -> None:
        """Build or load clique index.

        This method checks if an index exists. If not (or if force_rebuild=True),
        it builds the index by detecting all cliques and storing them.

        Notes
        -----
        This can take time for large graphs. Progress is displayed for large graphs.
        """
        if not self._is_open:
            raise RuntimeError("Backend not open. Call open() first.")

        # Check if index exists and we're not forcing rebuild
        if not self.force_rebuild and self.backend.exists():
            self.num_cliques = self.backend.count_cliques()
            print(f"Loaded existing index: {self.num_cliques} cliques")
            return

        # Clear if force rebuild
        if self.force_rebuild and self.backend.exists():
            self.backend.clear()

        # Build index
        print(
            f"Building index for graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges"
        )
        build_clique_index(
            self.graph, self.backend, max_size=self.max_clique_size
        )

        self.num_cliques = self.backend.count_cliques()
        print(f"Indexed {self.num_cliques} cliques")

    def query_batch(
        self, node_ids: list[int], fully_contained: bool = True
    ) -> list[tuple[int, list[int]]]:
        """Query cliques relevant to a batch of nodes.

        Parameters
        ----------
        node_ids : list of int
            Node IDs in the batch to query for.
        fully_contained : bool, optional
            If True, return only cliques where ALL nodes are in node_ids.
            If False, return cliques where ANY node is in node_ids
            (default: True).

        Returns
        -------
        list of (int, list of int)
            List of (clique_id, nodes) tuples for cliques matching the query.
        """
        if not self._is_open:
            raise RuntimeError("Backend not open. Call open() first.")

        return self.backend.query_by_nodes(
            node_ids, fully_contained=fully_contained
        )

    def query_node_cliques(self, node_id: int) -> list[tuple[int, list[int]]]:
        """Query all cliques containing a specific node.

        Parameters
        ----------
        node_id : int
            Node ID to query cliques for.

        Returns
        -------
        list of (int, list of int)
            List of (clique_id, nodes) tuples containing the node.
        """
        return self.query_batch([node_id], fully_contained=False)

    def query_cliques_by_id(
        self, clique_ids: list[int]
    ) -> list[tuple[int, list[int]]]:
        """Query cliques by their IDs.

        Parameters
        ----------
        clique_ids : list of int
            Clique IDs to retrieve.

        Returns
        -------
        list of (int, list of int)
            List of (clique_id, nodes) tuples for found cliques.
            Cliques that don't exist are silently skipped.

        Notes
        -----
        This is useful for clique-centric batching where you sample
        clique IDs directly and need to retrieve their nodes.

        Examples
        --------
        >>> engine = CliqueQueryEngine(graph, index_dir="/tmp/index")
        >>> engine.open()
        >>> engine.build_index()
        >>> cliques = engine.query_cliques_by_id([0, 1, 2])
        >>> print(f"Retrieved {len(cliques)} cliques")
        """
        if not self._is_open:
            raise RuntimeError("Backend not open. Call open() first.")

        cliques = []
        for sid in clique_ids:
            nodes = self.backend.get_clique(sid)
            if nodes is not None:
                cliques.append((sid, nodes))
        return cliques

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the indexed cliques.

        Returns
        -------
        dict
            Dictionary with statistics:
            - num_cliques: Total cliques indexed
            - num_nodes: Number of nodes in graph
            - num_edges: Number of edges in graph
            - index_dir: Path to index directory
            - max_clique_size: Maximum clique size indexed
        """
        return {
            "num_cliques": self.num_cliques,
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "index_dir": str(self.index_dir),
            "max_clique_size": self.max_clique_size,
        }

    def __enter__(self):
        """Context manager entry.

        Returns
        -------
        CliqueQueryEngine
            The engine instance.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit.

        Parameters
        ----------
        exc_type : type
            Exception type.
        exc_val : Exception
            Exception value.
        exc_tb : traceback
            Exception traceback.

        Returns
        -------
        bool
            False.
        """
        self.close()
        return False

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CliqueQueryEngine("
            f"nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()}, "
            f"cliques={self.num_cliques})"
        )
