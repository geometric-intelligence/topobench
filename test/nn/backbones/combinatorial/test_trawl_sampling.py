"""Tests for TRAWL's seeded random-walk sampling."""

import numpy as np
import pytest
import scipy.sparse as sp

from topobench.nn.backbones.combinatorial.trawl_utils import fast_walks
from topobench.nn.backbones.combinatorial.trawl_utils.sampling import (
    WalkSampler,
    _numpy_sum,
    guided_transition,
    prepare_walk_rows,
    simulate_nbrw_sparse,
    transition,
    walk_seed,
)


def test_transition_is_row_stochastic_with_isolated_self_loops():
    matrix = sp.csr_matrix([[0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0, 0, 0]])
    p = transition(matrix).toarray()
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[2, 2] == 1.0


def test_guided_transition_respects_dense_limit():
    ring = sp.diags([1.0, 1.0], [1, -1], shape=(5, 5)).tocsr()
    guided = guided_transition(ring, gamma=0.2)
    np.testing.assert_allclose(guided.toarray().sum(axis=1), 1.0)
    assert guided.nnz == transition(ring).nnz
    with pytest.raises(ValueError, match="dense_limit"):
        guided_transition(ring, gamma=0.2, dense_limit=4)


def test_walk_budget_nonbacktracking_and_reversal():
    matrix = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
    paths = WalkSampler(0)(matrix, k=8, length=10, reverse=True)
    assert paths.shape == (16, 10)
    np.testing.assert_array_equal(paths[8:], paths[:8, ::-1])
    assert np.all(paths[:8, 2:] != paths[:8, :-2])
    np.testing.assert_array_equal(
        paths, WalkSampler(0)(matrix, k=8, length=10, reverse=True)
    )


def test_walk_seed_is_collision_free():
    seeds = {
        walk_seed(40, identity, step, view)
        for identity in range(200)
        for step in range(3)
        for view in range(3)
    }
    assert len(seeds) == 200 * 3 * 3


def test_transition_cache_respects_masked_edges():
    full = sp.csr_matrix(np.ones((3, 3)) - np.eye(3))
    masked = sp.csr_matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    sampler = WalkSampler(cache_size=1)
    sampler(full, k=8, length=8, seed=4)
    paths = sampler(masked, k=8, length=8, seed=4)
    assert set(paths.flatten()) == {0, 1}
    assert len(sampler.cache) == 1


def test_sampler_on_large_sparse_graph():
    n = 10000
    rows = np.arange(n)
    matrix = sp.coo_matrix(
        (
            np.ones(2 * n),
            (
                np.concatenate([rows, rows]),
                np.concatenate([(rows + 1) % n, (rows - 1) % n]),
            ),
        ),
        shape=(n, n),
    ).tocsr()
    paths = WalkSampler(0)(matrix, k=32, length=32)
    assert paths.shape == (32, 32)
    assert ((paths >= 0) & (paths < n)).all()
    with pytest.raises(ValueError, match="dense_limit"):
        WalkSampler(0)(matrix, k=2, length=2, guidance={"gamma": 0.2})


def test_fast_walk_sum_matches_numpy():
    generator = np.random.default_rng(0)
    for _ in range(5000):
        size = int(generator.integers(0, 140))
        values = generator.random(size) * 10.0 ** generator.integers(-9, 9)
        assert _numpy_sum(values.tolist()) == float(np.sum(values))


def test_fast_walks_match_generator_choice():
    def reference(neighbors, weights, start, length, rng):
        # The original per-step implementation using Generator.choice.
        current, previous, path = start, -1, [start]
        for _ in range(length - 1):
            nbrs = neighbors[current]
            raw = np.asarray(weights[current], dtype=np.float64)
            probs = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
            probs[[i for i, n in enumerate(nbrs) if n == previous]] = 0.0
            if probs.sum() <= 0.0:
                probs = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
            if probs.sum() <= 0.0:
                probs = np.full(len(nbrs), 1.0 / len(nbrs))
            probs /= probs.sum()
            previous, current = (
                current,
                int(nbrs[int(rng.choice(len(nbrs), p=probs))]),
            )
            path.append(current)
        return path

    generator = np.random.default_rng(1)
    for _ in range(300):
        size = int(generator.integers(2, 30))
        neighbors = [
            [
                int(n)
                for n in generator.choice(size, int(generator.integers(1, 14)))
            ]
            for _ in range(size)
        ]
        weights = [
            (
                generator.random(len(n)) * generator.choice([1, 1, 0], len(n))
            ).tolist()
            for n in neighbors
        ]
        seed, start = (
            int(generator.integers(1e9)),
            int(generator.integers(size)),
        )
        fast_rng, slow_rng = (
            np.random.default_rng(seed),
            np.random.default_rng(seed),
        )
        assert simulate_nbrw_sparse(
            neighbors, weights, start, 32, fast_rng
        ) == reference(neighbors, weights, start, 32, slow_rng)
        assert fast_rng.bit_generator.state == slow_rng.bit_generator.state


def test_compiled_walks_match_pure_python():
    if not fast_walks.available:
        pytest.skip("numba is not installed")
    generator = np.random.default_rng(3)
    for _ in range(300):
        size = int(generator.integers(1, 30))
        neighbors = [
            [
                int(n)
                for n in generator.choice(size, int(generator.integers(0, 20)))
            ]
            for _ in range(size)
        ]
        weights = [
            (
                generator.random(len(n))
                * generator.choice([1.0, 1.0, 0.0, -1.0, np.nan], len(n))
            ).tolist()
            for n in neighbors
        ]
        rows = prepare_walk_rows(neighbors, weights)
        seed, start = (
            int(generator.integers(1e9)),
            int(generator.integers(size)),
        )
        fast, slow = np.random.default_rng(seed), np.random.default_rng(seed)
        # A plain list forces the pure-Python path.
        assert simulate_nbrw_sparse(
            neighbors, weights, start, 33, fast, rows=rows
        ) == simulate_nbrw_sparse(
            neighbors, weights, start, 33, slow, rows=list(rows)
        )
        assert fast.bit_generator.state == slow.bit_generator.state
