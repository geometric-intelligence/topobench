"""CPU structural encodings cached by TopoBench preprocessing."""

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from topobench.data.utils.trawl.sampling import (
    csr_rows,
    prepare_walk_rows,
    simulate_nbrw_sparse,
)


def transition(matrix):
    """Row-normalize positive weights, retaining isolated states.

    Non-finite and nonpositive weights are dropped; empty rows receive a
    self-loop.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Square weighted adjacency matrix.

    Returns
    -------
    scipy.sparse.spmatrix
        Row-stochastic transition matrix.
    """
    matrix = matrix.astype(np.float64).tocsr(copy=True)
    matrix.data = np.where(
        np.isfinite(matrix.data) & (matrix.data > 0), matrix.data, 0
    )
    matrix.eliminate_zeros()
    empty = np.asarray(matrix.sum(axis=1)).ravel() == 0
    matrix = matrix + sp.diags(empty.astype(float))
    return sp.diags(1 / np.asarray(matrix.sum(axis=1)).ravel()) @ matrix


def chung_laplacian(matrix):
    """Symmetric directed Laplacian with a power-iterated stationary measure.

    Laziness makes iteration converge on bipartite Hasse graphs. Disconnected
    components retain their uniform-initialization mass.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Square weighted adjacency matrix.

    Returns
    -------
    scipy.sparse.spmatrix
        Symmetric Chung Laplacian of shape ``(n, n)``.
    """
    p = transition(matrix)
    n = p.shape[0]
    pi = np.full(n, 1 / n)
    for _ in range(1000):
        updated = (pi + p.T @ pi) / 2
        if np.max(np.abs(updated - pi)) < 1e-12:
            pi = updated
            break
        pi = updated
    pi = np.maximum(pi, 1e-15)
    normalized = sp.diags(np.sqrt(pi)) @ p @ sp.diags(1 / np.sqrt(pi))
    return sp.eye(n) - (normalized + normalized.T) / 2


def guided_transition(matrix, gamma=0.2, diffusion_t=0.1, dense_limit=2048):
    """Heat-kernel-guided weights on existing transitions only.

    Exact dense decomposition is deliberately bounded: never silently change
    an experiment to an approximation when its graph exceeds the limit.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Square weighted adjacency matrix.
    gamma : float, optional
        Nonnegative exponent on the heat-kernel magnitude; 0 disables
        guidance (default: 0.2).
    diffusion_t : float, optional
        Nonnegative heat-kernel diffusion time (default: 0.1).
    dense_limit : int, optional
        Maximum number of states for the dense eigendecomposition
        (default: 2048).

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-stochastic guided transition matrix.

    Raises
    ------
    ValueError
        If ``gamma`` or ``diffusion_t`` is negative, or guidance is enabled
        and the matrix exceeds ``dense_limit``.
    """
    if gamma < 0 or diffusion_t < 0:
        raise ValueError(
            "Guidance gamma and diffusion time must be nonnegative"
        )
    p = transition(matrix).tocsr()
    if gamma == 0:
        return p
    if matrix.shape[0] > dense_limit:
        raise ValueError(
            "Heat guidance exceeds dense_limit; disable guidance or explicitly raise the limit"
        )
    eigenvalues, vectors = scipy.linalg.eigh(chung_laplacian(matrix).toarray())
    heat = (vectors * np.exp(-diffusion_t * eigenvalues)) @ vectors.T
    coo = p.tocoo()
    coo.data *= np.abs(heat[coo.row, coo.col]) ** gamma
    return transition(coo.tocsr())


def positional_encodings(
    matrix,
    local=True,
    rw_steps=8,
    rw_samples=32,
    heat_times=(),
    electrostatic_betas=(),
    laplacian_dim=0,
    seed=0,
    dense_limit=2048,
):
    """Local degree, empirical nonbacktracking RWSE, heat, potential and LapPE.

    Per-relation inactive states receive zeros. LapPE signs are canonicalized;
    repeated eigenvalues still admit basis rotations across numerical backends.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Square weighted adjacency matrix of one relation.
    local : bool, optional
        Whether to include log-degree and neighbor-count channels
        (default: True).
    rw_steps : int, optional
        Number of nonbacktracking random-walk steps; 0 disables RWSE
        (default: 8).
    rw_samples : int, optional
        Number of sampled walks per active state (default: 32).
    heat_times : sequence of float, optional
        Nonnegative heat-kernel diagonal times (default: ()).
    electrostatic_betas : sequence of float, optional
        Positive regularization shifts for degree-charge potentials
        (default: ()).
    laplacian_dim : int, optional
        Number of Laplacian eigenvector channels (default: 0).
    seed : int, optional
        Random seed for walk sampling (default: 0).
    dense_limit : int, optional
        Maximum number of states for spectral encodings (default: 2048).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(n, d)`` with the selected encodings
        concatenated column-wise.

    Raises
    ------
    ValueError
        If dimensions, sample counts, heat times or betas are invalid, or a
        spectral encoding exceeds ``dense_limit``.
    """
    n = matrix.shape[0]
    if rw_steps < 0 or rw_samples < 1 or laplacian_dim < 0:
        raise ValueError("Invalid positional encoding dimensions/sample count")
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    active = np.asarray((matrix + matrix.T).sum(axis=1)).ravel() > 0
    parts = []
    if local:
        parts.append(
            np.stack(
                (
                    np.log1p(degree),
                    np.asarray(matrix.getnnz(axis=1), dtype=float),
                ),
                axis=1,
            )
        )
    p = transition(matrix).tocsr()
    if rw_steps:
        neighbors, probs = csr_rows(p)
        rng = np.random.default_rng(seed)
        rows = prepare_walk_rows(neighbors, probs)
        rw = np.zeros((n, rw_steps))
        for i in np.flatnonzero(active):
            for _ in range(rw_samples):
                path = simulate_nbrw_sparse(
                    neighbors, probs, i, rw_steps + 1, rng, rows=rows
                )
                rw[i] += np.asarray(path[1:]) == i
        parts.append(rw / rw_samples)
    if heat_times or electrostatic_betas or laplacian_dim:
        if n > dense_limit:
            raise ValueError("Spectral encoding exceeds dense_limit")
        values, vectors = scipy.linalg.eigh(chung_laplacian(matrix).toarray())
        for t in heat_times:
            if t < 0:
                raise ValueError("Heat times must be nonnegative")
            parts.append(((vectors**2) @ np.exp(-t * values))[:, None])
        charge = degree / max(degree.sum(), 1)
        for beta in electrostatic_betas:
            if beta <= 0:
                raise ValueError(
                    "Electrostatic regularization must be positive"
                )
            parts.append(
                (vectors @ ((vectors.T @ charge) / (values + beta)))[:, None]
            )
        if laplacian_dim:
            selected = vectors[:, values > 1e-6][:, :laplacian_dim].copy()
            for col in range(selected.shape[1]):
                pivot = np.argmax(np.abs(selected[:, col]))
                selected[:, col] *= 1 if selected[pivot, col] >= 0 else -1
            parts.append(
                np.pad(
                    selected, ((0, 0), (0, laplacian_dim - selected.shape[1]))
                )
            )
    result = np.concatenate(parts, axis=1) if parts else np.zeros((n, 0))
    result[~active] = 0
    return np.nan_to_num(result).astype(np.float32)
