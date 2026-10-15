"""Weighted non-backtracking random walks with coverage-biased starts."""

import hashlib
import json
from bisect import bisect_right
from collections import OrderedDict
from itertools import accumulate

import numpy as np

from topobench.data.utils.trawl import fast_walks


def walk_seed(*parts):
    """Collision-free sampling seed from integer seed components.

    Additive offsets alias distinct (identity, step, view) tuples; hashing
    through ``SeedSequence`` keeps every combination distinct.

    Parameters
    ----------
    *parts : tuple
        Integer-convertible seed components, reduced modulo ``2**63``.

    Returns
    -------
    int
        Seed derived from all components.
    """
    entropy = [int(part) % 2**63 for part in parts]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


class WalkSampler:
    """Bounded cache of CPU transition distributions; paths still resample.

    Cache keys include actual weights, so masked-topology pretraining cannot
    accidentally reuse an unmasked transition matrix. Derived cache state is
    intentionally excluded from neural checkpoints.

    Parameters
    ----------
    cache_size : int, optional
        Maximum number of cached transition distributions; 0 disables
        caching (default: 1024).
    """

    def __init__(self, cache_size=1024):
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __call__(self, matrix, guidance=None, **kwargs):
        """Sample walks on ``matrix`` using a cached transition distribution.

        Parameters
        ----------
        matrix : scipy.sparse.spmatrix
            Square weighted relation matrix over graph-local states.
        guidance : dict, optional
            Keyword arguments for ``guided_transition``; plain ``transition``
            is used when empty (default: None).
        **kwargs : dict
            Options forwarded to ``sample_neighbors`` (``k``, ``length``,
            ``seed``, ``start_policy``, ``epsilon``, ``reverse``).

        Returns
        -------
        numpy.ndarray
            Walks of graph-local state IDs with shape ``(num_walks, length)``.
        """
        from topobench.data.utils.trawl.encodings import (
            guided_transition,
            transition,
        )

        matrix = matrix.tocsr()
        digest = hashlib.sha256()
        for value in (matrix.data, matrix.indices, matrix.indptr):
            digest.update(value.tobytes())
        key = (
            matrix.shape,
            digest.digest(),
            json.dumps(dict(guidance or {}), sort_keys=True),
        )
        if key in self.cache:
            active, neighbors, probs, rows = self.cache.pop(key)
            self.cache[key] = (active, neighbors, probs, rows)
        else:
            active = np.flatnonzero(
                np.asarray((matrix + matrix.T).sum(axis=1)).ravel() > 0
            )
            if not len(active):
                return np.empty((0, kwargs.get("length", 32)), dtype=np.int64)
            subgraph = matrix[active][:, active]
            p = (
                guided_transition(subgraph, **guidance)
                if guidance
                else transition(subgraph)
            )
            neighbors, probs = csr_rows(p)
            rows = prepare_walk_rows(neighbors, probs)
            if self.cache_size:
                self.cache[key] = (active, neighbors, probs, rows)
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return active[sample_neighbors(neighbors, probs, rows=rows, **kwargs)]


def csr_rows(matrix):
    """Split a CSR matrix into per-row column and value lists.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Sparse matrix.

    Returns
    -------
    tuple of list
        ``(neighbors, probs)``: column indices and values of each row.
    """
    matrix = matrix.tocsr()
    bounds = zip(matrix.indptr[:-1], matrix.indptr[1:], strict=True)
    neighbors, probs = [], []
    for start, stop in bounds:
        neighbors.append(matrix.indices[start:stop].tolist())
        probs.append(matrix.data[start:stop].tolist())
    return neighbors, probs


def sample_neighbors(
    neighbors,
    probs,
    k=32,
    length=32,
    seed=0,
    start_policy="coverage",
    epsilon=0.05,
    reverse=False,
    rows=None,
):
    """Sample ``k`` non-backtracking walks over adjacency lists.

    Coverage starts favour rarely visited states until coverage stops
    growing by ``epsilon``, then fall back to uniform starts.

    Parameters
    ----------
    neighbors : list of list of int
        Neighbor state indices for each state.
    probs : list of list of float
        Transition weights aligned with ``neighbors``.
    k : int, optional
        Number of walks to sample (default: 32).
    length : int, optional
        Number of states per walk (default: 32).
    seed : int, optional
        Seed for the random generator (default: 0).
    start_policy : str, optional
        Start-state policy, ``"uniform"`` or ``"coverage"``
        (default: "coverage").
    epsilon : float, optional
        Coverage-change threshold below which coverage starts fall back to
        uniform starts (default: 0.05).
    reverse : bool, optional
        If True, append the reversal of every walk (default: False).
    rows : WalkRows, optional
        Precomputed output of ``prepare_walk_rows`` (default: None).

    Returns
    -------
    numpy.ndarray
        Walks of state indices with shape ``(num_walks, length)``.
    """
    if k < 1 or length < 1 or start_policy not in {"uniform", "coverage"}:
        raise ValueError("Invalid walk budget, length or start policy")
    if not neighbors:
        return np.empty((0, length), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if rows is None:
        rows = prepare_walk_rows(neighbors, probs)
    visits = np.zeros(len(neighbors), dtype=np.float64)
    previous = 0.0
    paths = []
    for _ in range(k):
        if start_policy == "coverage" and visits.sum() > 0:
            coverage = float((visits > 0).mean())
            if abs(coverage - previous) < epsilon:
                start = int(rng.integers(0, len(neighbors)))
            else:
                weights = 1 / (1 + visits)
                start = int(
                    rng.choice(len(neighbors), p=weights / weights.sum())
                )
            previous = coverage
        else:
            start = int(rng.integers(0, len(neighbors)))
        path = simulate_nbrw_sparse(
            neighbors, probs, start, length, rng, rows=rows
        )
        np.add.at(visits, path, 1)
        paths.append(path)
    paths = np.asarray(paths, dtype=np.int64)
    return np.concatenate((paths, paths[:, ::-1])) if reverse else paths


class WalkRows(list):
    """Per-state ``(neighbors, weights)`` rows plus cached kernel arrays."""

    csr = None


def prepare_walk_rows(neighbors, neigh_probs):
    """Per-state neighbor lists with unusable weights already zeroed.

    NaN, infinite and nonpositive weights are unavailable transitions.
    Preparing once per graph avoids repeating this on every walk step.

    Parameters
    ----------
    neighbors : list of list of int
        Neighbor state indices for each state.
    neigh_probs : list of list of float
        Transition weights aligned with ``neighbors``.

    Returns
    -------
    WalkRows
        One ``(neighbors, weights)`` pair per state with cleaned weights.
    """
    rows = WalkRows()
    for nbrs, row in zip(neighbors, neigh_probs, strict=True):
        raw = np.asarray(row, dtype=np.float64)
        clean = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
        rows.append(([int(n) for n in nbrs], clean.tolist()))
    return rows


def _numpy_sum(values):
    """Sum floats in exactly the order ``ndarray.sum`` uses.

    NumPy adds fewer than eight values sequentially and otherwise keeps eight
    running partial sums (pairwise summation). Reproducing that order keeps
    sampled walks bit-identical to ``Generator.choice`` without its overhead.

    Parameters
    ----------
    values : list of float
        Values to sum.

    Returns
    -------
    float
        Sum of ``values``.
    """
    n = len(values)
    if n < 8:
        total = 0.0
        for value in values:
            total += value
        return total
    if n > 128:
        # Longer blocks recurse; defer to NumPy itself.
        return float(np.sum(values))
    r = list(values[:8])
    stop = n - n % 8
    for i in range(8, stop, 8):
        for j in range(8):
            r[j] += values[i + j]
    total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    for i in range(stop, n):
        total += values[i]
    return total


def simulate_nbrw_sparse(
    neighbors: list[list[int]],
    neigh_probs: list[list[float]],
    start: int,
    length: int,
    rng: np.random.Generator,
    rows=None,
) -> list[int]:
    """Non-backtracking weighted walk of ``length`` states.

    Each step draws exactly as ``rng.choice(len(nbrs), p=probs)`` does: one
    uniform sample against the normalized cumulative weights. Plain-float
    arithmetic reproduces NumPy's results bit for bit while avoiding its
    per-call overhead on these short rows. Pass ``rows`` from
    ``prepare_walk_rows`` when sampling several walks on one graph.

    Parameters
    ----------
    neighbors : list of list of int
        Neighbor state indices for each state.
    neigh_probs : list of list of float
        Transition weights aligned with ``neighbors``.
    start : int
        Initial state index.
    length : int
        Number of states in the walk.
    rng : numpy.random.Generator
        Random generator supplying one uniform draw per step.
    rows : WalkRows, optional
        Precomputed output of ``prepare_walk_rows`` (default: None).

    Returns
    -------
    list of int
        Visited state indices, starting with ``start``.
    """
    if rows is None:
        rows = prepare_walk_rows(neighbors, neigh_probs)
    if fast_walks.available and isinstance(rows, WalkRows):
        # Compiled kernel with the identical arithmetic and random draws.
        if rows.csr is None:
            rows.csr = fast_walks.walk_csr(rows)
        return fast_walks.walk(rows.csr, start, length, rng)
    current = int(start)
    prev = -1
    path = [current]
    for _ in range(length - 1):
        nbrs, clean = rows[current]
        probs = (
            [0.0 if n == prev else p for n, p in zip(nbrs, clean, strict=True)]
            if prev >= 0
            else clean
        )
        total = _numpy_sum(probs)
        if not np.isfinite(total) or total <= 0.0:
            # Backtracking is allowed when it is the only usable move.
            probs = clean
            total = _numpy_sum(probs)
            if not np.isfinite(total) or total <= 0.0:
                if nbrs:
                    probs, total = [1.0 / len(nbrs)] * len(nbrs), 1.0
                else:
                    # Defensive only: graph construction normally inserts a
                    # self-loop for isolated states.
                    path.append(current)
                    prev = current
                    continue
        cdf = list(accumulate(p / total for p in probs))
        last = cdf[-1]
        cdf = [c / last for c in cdf]
        nxt = nbrs[bisect_right(cdf, rng.random())]
        prev, current = current, nxt
        path.append(current)
    return path
