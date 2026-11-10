"""Optional compiled non-backtracking walk kernel (requires ``numba``).

The kernel reproduces ``sampling.simulate_nbrw_sparse`` bit for bit: it
draws one ``Generator.random()`` per step from the caller's NumPy
generator, and repeats NumPy's floating-point operation order (sequential
sums below eight values, pairwise summation above, sequential cumulative
sums). Without ``numba`` callers use the pure-Python implementation.
"""

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised when numba is absent
    njit = None

__all__ = ["available", "walk_csr", "walk"]

available = njit is not None


def walk_csr(rows):
    """Flatten ``prepare_walk_rows`` output into CSR arrays for the kernel.

    Parameters
    ----------
    rows : list of tuple
        Per-state ``(neighbors, weights)`` pairs from ``prepare_walk_rows``.

    Returns
    -------
    tuple
        ``(indptr, indices, weights, width)`` where ``width`` is the largest
        row length (at least 1).
    """
    lengths = np.fromiter((len(nbrs) for nbrs, _ in rows), np.int64, len(rows))
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = np.fromiter(
        (n for nbrs, _ in rows for n in nbrs), np.int64, int(indptr[-1])
    )
    weights = np.fromiter(
        (w for _, row in rows for w in row), np.float64, int(indptr[-1])
    )
    width = int(lengths.max()) if len(lengths) else 0
    return indptr, indices, weights, max(width, 1)


if available:

    @njit(cache=True)
    def _block_sum(values, start, n):
        # NumPy's pairwise_sum base case (blocks of at most 128 values).
        if n < 8:
            total = 0.0
            for i in range(start, start + n):
                total += values[i]
            return total
        r0 = values[start]
        r1 = values[start + 1]
        r2 = values[start + 2]
        r3 = values[start + 3]
        r4 = values[start + 4]
        r5 = values[start + 5]
        r6 = values[start + 6]
        r7 = values[start + 7]
        stop = n - n % 8
        for i in range(8, stop, 8):
            base = start + i
            r0 += values[base]
            r1 += values[base + 1]
            r2 += values[base + 2]
            r3 += values[base + 3]
            r4 += values[base + 4]
            r5 += values[base + 5]
            r6 += values[base + 6]
            r7 += values[base + 7]
        total = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        for i in range(stop, n):
            total += values[start + i]
        return total

    @njit(cache=True)
    def _pairwise_sum(values, start, n):
        """NumPy's float64 pairwise sum, with its recursion made explicit.

        Numba cannot reliably load cached recursive functions, so the
        recursive halving is replayed with a stack in the same order.
        """
        if n <= 128:
            return _block_sum(values, start, n)
        frames = np.empty((128, 3), dtype=np.int64)
        partial = np.empty(128, dtype=np.float64)
        frames[0, 0], frames[0, 1], frames[0, 2] = start, n, 0
        top, filled = 1, 0
        while top > 0:
            top -= 1
            first, size, phase = frames[top, 0], frames[top, 1], frames[top, 2]
            if size <= 128:
                partial[filled] = _block_sum(values, first, size)
                filled += 1
            elif phase == 0:
                half = size // 2
                half -= half % 8
                frames[top, 0], frames[top, 1], frames[top, 2] = first, size, 1
                frames[top + 1, 0], frames[top + 1, 1] = (
                    first + half,
                    size - half,
                )
                frames[top + 1, 2] = 0
                frames[top + 2, 0], frames[top + 2, 1], frames[top + 2, 2] = (
                    first,
                    half,
                    0,
                )
                top += 3
            else:
                partial[filled - 2] = partial[filled - 2] + partial[filled - 1]
                filled -= 1
        return partial[0]

    @njit(cache=True)
    def _walk(indptr, indices, weights, width, start, length, rng):
        path = np.empty(length, dtype=np.int64)
        probs = np.empty(width, dtype=np.float64)
        cdf = np.empty(width, dtype=np.float64)
        current = start
        previous = -1
        path[0] = current
        for step in range(1, length):
            lo = indptr[current]
            n = indptr[current + 1] - lo
            for j in range(n):
                probs[j] = (
                    0.0 if indices[lo + j] == previous else weights[lo + j]
                )
            total = _pairwise_sum(probs, 0, n)
            if not (np.isfinite(total) and total > 0.0):
                # Backtracking is allowed when it is the only usable move.
                for j in range(n):
                    probs[j] = weights[lo + j]
                total = _pairwise_sum(probs, 0, n)
                if not (np.isfinite(total) and total > 0.0):
                    if n == 0:
                        path[step] = current
                        previous = current
                        continue
                    for j in range(n):
                        probs[j] = 1.0 / n
                    total = 1.0
            running = 0.0
            for j in range(n):
                running = (
                    probs[j] / total if j == 0 else running + probs[j] / total
                )
                cdf[j] = running
            last = cdf[n - 1]
            for j in range(n):
                cdf[j] = cdf[j] / last
            draw = rng.random()
            index = 0
            while index < n and cdf[index] <= draw:
                index += 1
            previous = current
            current = indices[lo + index]
            path[step] = current
        return path


def walk(csr, start, length, rng):
    """Sample one walk with the compiled kernel.

    Parameters
    ----------
    csr : tuple
        ``(indptr, indices, weights, width)`` arrays from ``walk_csr``.
    start : int
        Initial state index.
    length : int
        Number of states in the walk.
    rng : numpy.random.Generator
        Random generator supplying one uniform draw per step.

    Returns
    -------
    list of int
        Visited state indices, starting with ``start``.
    """
    indptr, indices, weights, width = csr
    return _walk(
        indptr, indices, weights, width, int(start), int(length), rng
    ).tolist()
