"""Abstract base class for structure index backends.

This module provides the interface for different storage backends (SQLite3, RocksDB)
used to index topological structures in transductive learning scenarios.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class AbstractIndexBackend(ABC):
    """Abstract interface for structure index storage backends.

    This class defines the contract that all index backends must implement
    to store and query topological structures (simplices, cells, etc.) for
    transductive learning on large graphs.

    Parameters
    ----------
    data_dir : str
        Directory path for storing index data.
    **kwargs : dict
        Backend-specific configuration options.

    Notes
    -----
    Backends should support:
    - Insert: Add structure with associated node IDs
    - Query: Retrieve structures containing specific nodes
    - Batch operations: Efficient bulk insert/query
    - Persistence: Survive process restarts

    See Also
    --------
    topobench.data.index.sqlite_backend.SQLiteIndexBackend :
        SQLite3 implementation (default fallback).
    """

    def __init__(self, data_dir: str, **kwargs: Any) -> None:
        """Initialize backend.

        Parameters
        ----------
        data_dir : str
            Directory for index storage.
        **kwargs : dict
            Backend-specific options.
        """
        self.data_dir = data_dir

    @abstractmethod
    def open(self) -> None:
        """Open connection to backend storage.

        Raises
        ------
        RuntimeError
            If connection fails.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close connection to backend storage."""
        raise NotImplementedError

    @abstractmethod
    def insert(self, clique_id: int, nodes: list[int]) -> None:
        """Insert a clique with its constituent nodes.

        Parameters
        ----------
        clique_id : int
            Unique identifier for the clique.
        nodes : list of int
            Node IDs that comprise this clique.

        Notes
        -----
        Implementations should handle duplicate inserts gracefully.
        """
        raise NotImplementedError

    @abstractmethod
    def insert_batch(self, cliques: Iterator[tuple[int, list[int]]]) -> None:
        """Insert multiple cliques efficiently.

        Parameters
        ----------
        cliques : Iterator of (int, list of int)
            Iterator yielding (clique_id, nodes) tuples.

        Notes
        -----
        This should be optimized for bulk operations (e.g., transactions).
        """
        raise NotImplementedError

    @abstractmethod
    def query_by_nodes(
        self, node_ids: list[int], fully_contained: bool = True
    ) -> list[tuple[int, list[int]]]:
        """Query structures containing specified nodes.

        Parameters
        ----------
        node_ids : list of int
            Node IDs to search for.
        fully_contained : bool, optional
            If True, return only structures where ALL nodes are in node_ids.
            If False, return structures where ANY node is in node_ids
            (default: True).

        Returns
        -------
        list of (int, list of int)
            List of (structure_id, nodes) tuples matching the query.

        Notes
        -----
        For fully_contained=True, the structure's nodes must be a subset
        of node_ids (no nodes outside the batch).
        """
        raise NotImplementedError

    @abstractmethod
    def count_cliques(self) -> int:
        """Get total number of indexed cliques.

        Returns
        -------
        int
            Total clique count.
        """
        raise NotImplementedError

    @abstractmethod
    def get_clique(self, clique_id: int) -> list[int] | None:
        """Get nodes for a specific clique ID.

        Parameters
        ----------
        clique_id : int
            Clique ID to query.

        Returns
        -------
        list of int or None
            List of node IDs in the clique, or None if not found.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self) -> bool:
        """Check if index database exists.

        Returns
        -------
        bool
            True if index exists and is non-empty.
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Delete all indexed data.

        Notes
        -----
        Use with caution - this is destructive.
        """
        raise NotImplementedError

    def __enter__(self):
        """Context manager entry.

        Returns
        -------
        AbstractIndexBackend
            Self instance with opened connection.
        """
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception was raised.
        exc_val : Exception or None
            Exception instance if an exception was raised.
        exc_tb : traceback or None
            Exception traceback if an exception was raised.

        Returns
        -------
        bool or None
            None to propagate exceptions (default behavior).
        """
        self.close()
        return False
