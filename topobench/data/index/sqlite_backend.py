"""SQLite3 backend for structure index storage.

This module implements a SQLite3-based storage backend for indexing topological
structures. SQLite3 is used as the primary backend because:
1. Built-in to Python (zero dependencies)
2. Universally compatible
3. Good performance for read-heavy workloads
4. ACID transactions for consistency
"""

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from topobench.data.index.base import AbstractIndexBackend


class SQLiteIndexBackend(AbstractIndexBackend):
    """SQLite3 implementation of structure index backend.

    This backend stores structures in a SQLite database with efficient
    indexing for node-based queries. Each structure is stored with its
    constituent nodes, enabling fast retrieval during batch sampling.

    Parameters
    ----------
    data_dir : str
        Directory path for storing the SQLite database file.
    db_name : str, optional
        Database filename (default: "structure_index.db").
    **kwargs : dict
        Additional SQLite connection parameters.

    Attributes
    ----------
    db_path : Path
        Full path to the SQLite database file.
    conn : sqlite3.Connection or None
        Active database connection (None when closed).

    Examples
    --------
    >>> backend = SQLiteIndexBackend(data_dir="/tmp/index")
    >>> backend.open()
    >>> backend.insert(structure_id=0, nodes=[1, 2, 3])
    >>> results = backend.query_by_nodes([1, 2, 3, 4])
    >>> backend.close()

    Notes
    -----
    The database schema consists of two tables:
    - structures: (structure_id PRIMARY KEY, nodes_json TEXT)
    - node_index: (node_id INTEGER, structure_id INTEGER)

    The node_index table is indexed for fast node → structure lookups.

    See Also
    --------
    topobench.data.index.base.AbstractIndexBackend :
        Abstract base class defining the interface.
    """

    def __init__(
        self, data_dir: str, db_name: str = "structure_index.db", **kwargs: Any
    ) -> None:
        """Initialize SQLite backend.

        Parameters
        ----------
        data_dir : str
            Directory for database storage.
        db_name : str, optional
            Database filename (default: "structure_index.db").
        **kwargs : dict
            Additional SQLite connection parameters.
        """
        super().__init__(data_dir, **kwargs)
        self.db_path = Path(data_dir) / db_name
        self.conn: sqlite3.Connection | None = None
        self.kwargs = kwargs

        # Ensure directory exists
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    def open(self) -> None:
        """Open connection to SQLite database.

        Creates tables and indices if they don't exist.

        Raises
        ------
        RuntimeError
            If connection fails.
        """
        try:
            self.conn = sqlite3.connect(str(self.db_path), **self.kwargs)
            self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            self._create_tables()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to open SQLite database: {e}") from e

    def close(self) -> None:
        """Close connection to SQLite database."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _create_tables(self) -> None:
        """Create database tables and indices if they don't exist."""
        # Main structures table
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS structures (
                structure_id INTEGER PRIMARY KEY,
                nodes_json TEXT NOT NULL
            )
            """
        )

        # Node index for fast queries
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS node_index (
                node_id INTEGER NOT NULL,
                structure_id INTEGER NOT NULL,
                FOREIGN KEY (structure_id) REFERENCES structures(structure_id)
            )
            """
        )

        # Create index on node_id for fast lookups
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_node_id
            ON node_index(node_id)
            """
        )

        self.conn.commit()

    def insert(self, clique_id: int, nodes: list[int]) -> None:
        """Insert a clique with its constituent nodes.

        Parameters
        ----------
        clique_id : int
            Unique identifier for the clique.
        nodes : list of int
            Node IDs that comprise this clique.
        """
        # Insert into structures table
        nodes_json = json.dumps(sorted(nodes))
        self.conn.execute(
            "INSERT OR REPLACE INTO structures (structure_id, nodes_json) VALUES (?, ?)",
            (clique_id, nodes_json),
        )

        # Insert into node index
        self.conn.executemany(
            "INSERT INTO node_index (node_id, structure_id) VALUES (?, ?)",
            [(node_id, clique_id) for node_id in nodes],
        )

        self.conn.commit()

    def insert_batch(
        self, cliques: Iterator[tuple[int, list[int]]], batch_size: int = 10000
    ) -> None:
        """Insert multiple cliques efficiently using transactions.

        Parameters
        ----------
        cliques : Iterator of (int, list of int)
            Iterator yielding (clique_id, nodes) tuples.
        batch_size : int, optional
            Number of cliques to process in each chunk (default: 10000).
            Prevents memory issues with large graphs by streaming insertion.
        """
        # Process cliques in chunks to avoid loading all into memory
        structures_buffer = []
        node_index_buffer = []

        for struct_id, nodes in cliques:
            # Add to buffers
            structures_buffer.append(
                (struct_id, json.dumps(sorted([int(n) for n in nodes])))
            )
            node_index_buffer.extend(
                [(int(node_id), struct_id) for node_id in nodes]
            )

            # When buffer reaches batch_size, flush to database
            if len(structures_buffer) >= batch_size:
                self._flush_buffers(structures_buffer, node_index_buffer)
                structures_buffer.clear()
                node_index_buffer.clear()

        # Flush remaining structures
        if structures_buffer:
            self._flush_buffers(structures_buffer, node_index_buffer)

    def _flush_buffers(
        self, structures_buffer: list, node_index_buffer: list
    ) -> None:
        """Flush buffered structures to database in a transaction.

        Parameters
        ----------
        structures_buffer : list
            List of (structure_id, nodes_json) tuples.
        node_index_buffer : list
            List of (node_id, structure_id) tuples.
        """
        self.conn.execute("BEGIN TRANSACTION")

        try:
            # Insert structures
            self.conn.executemany(
                "INSERT OR REPLACE INTO structures (structure_id, nodes_json) VALUES (?, ?)",
                structures_buffer,
            )

            # Insert node indices
            self.conn.executemany(
                "INSERT INTO node_index (node_id, structure_id) VALUES (?, ?)",
                node_index_buffer,
            )

            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Batch insert failed: {e}") from e

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
        """
        if not node_ids:
            return []

        node_ids_set = set(node_ids)

        if fully_contained:
            # Find structures where all nodes are in node_ids
            # Strategy: Get all candidate structures, then filter
            placeholders = ",".join("?" * len(node_ids))
            query = f"""
                SELECT DISTINCT s.structure_id, s.nodes_json
                FROM structures s
                WHERE s.structure_id IN (
                    SELECT structure_id
                    FROM node_index
                    WHERE node_id IN ({placeholders})
                )
            """

            cursor = self.conn.execute(query, node_ids)
            results = []

            for struct_id, nodes_json in cursor:
                nodes = json.loads(nodes_json)
                # Check if all structure nodes are in our batch
                if set(nodes).issubset(node_ids_set):
                    results.append((struct_id, nodes))

            return results
        else:
            # Find structures where ANY node is in node_ids
            placeholders = ",".join("?" * len(node_ids))
            query = f"""
                SELECT DISTINCT s.structure_id, s.nodes_json
                FROM structures s
                INNER JOIN node_index ni ON s.structure_id = ni.structure_id
                WHERE ni.node_id IN ({placeholders})
            """

            cursor = self.conn.execute(query, node_ids)
            return [
                (struct_id, json.loads(nodes_json))
                for struct_id, nodes_json in cursor
            ]

    def count_cliques(self) -> int:
        """Get total number of indexed cliques.

        Returns
        -------
        int
            Total clique count.
        """
        cursor = self.conn.execute("SELECT COUNT(*) FROM structures")
        return cursor.fetchone()[0]

    def exists(self) -> bool:
        """Check if index database exists and is non-empty.

        Returns
        -------
        bool
            True if database file exists and contains cliques.
        """
        if not self.db_path.exists():
            return False

        # Check if tables exist and have data
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='structures'"
                )
                if cursor.fetchone()[0] == 0:
                    return False

                cursor = conn.execute("SELECT COUNT(*) FROM structures")
                return cursor.fetchone()[0] > 0
        except sqlite3.Error:
            return False

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
        cursor = self.conn.execute(
            "SELECT nodes_json FROM structures WHERE structure_id = ?",
            (clique_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def clear(self) -> None:
        """Delete all indexed data.

        Notes
        -----
        This deletes all rows but keeps the schema intact.
        """
        self.conn.execute("DELETE FROM node_index")
        self.conn.execute("DELETE FROM structures")
        self.conn.commit()
