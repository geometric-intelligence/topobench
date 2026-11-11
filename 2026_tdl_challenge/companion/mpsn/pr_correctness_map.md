# MPSN — correctness map (paper → code)

**Model.** Message Passing Simplicial Networks (MPSN) — Bodnar, Frasca, Wang, Otter,
Montúfar, Liò, Bronstein, *Weisfeiler and Lehman Go Topological: Message Passing
Simplicial Networks*, ICML 2021 — [arXiv:2103.03212](https://arxiv.org/abs/2103.03212).

Every construct of the paper's Definition 4 and message-passing equations (1)–(6) maps
to an explicit location in `topobench/nn/backbones/simplicial/mpsn.py`. Line numbers are
exact for the submitted file; the docstrings themselves also cite the equations inline.

### Definition 4 — the four adjacencies of a simplex σ

| Paper (Def. 4) | Meaning | Code |
|---|---|---|
| Boundary 𝓑(σ) | faces of rank *k−1* | derived in `_build_adjacency_index`, boundary pairs — **L221–222** |
| Co-boundary 𝓒(σ) | co-faces of rank *k+1* | `_build_adjacency_index`, co-boundary pairs — **L224–225** |
| Lower N↓(σ) | same rank, share a **face** δ | `_ordered_pairs` (**L135**) + `_build_adjacency_index` — **L227** |
| Upper N↑(σ) | same rank, share a **co-face** δ | `_ordered_pairs` (**L135**) + `_build_adjacency_index` — **L227** |

All four are derived **only** from the boundary matrices B₁, B₂ (no orientation input);
the orientation-invariant ("absolute value") variant used for graph-lifted benchmarks is
implemented by dropping signs in `_coo_indices` — **L114–118**. Docstring: **L10–L25, L45–L50**.

### Equations (1)–(4) — the four message types

Each adjacency owns a dedicated message MLP `M_•`; messages are summed (paper's ⊕ = sum),
implemented by `index_add` inside `_pair_message` (**L312**).

| Paper | Message MLP (declared) | Computed in `forward` |
|---|---|---|
| Eq. (1) boundary msg `m_𝓑` | `msg_1_bnd` L282, `msg_2_bnd` L289 | edges **L396–398**, faces **L417–419** |
| Eq. (2) co-boundary msg `m_𝓒` | `msg_0_cob` L277, `msg_1_cob` L283 | nodes **L385–387**, edges **L400–402** |
| Eq. (3) lower msg `m_↓` (uses shared face) | `msg_1_low` L284, `msg_2_low` L290 | edges **L404–406**, faces **L421–423** |
| Eq. (4) upper msg `m_↑` (uses shared co-face) | `msg_0_up` L278, `msg_1_up` L285 | nodes **L389–391**, edges **L407–409** |

The shared simplex δ (the face for lower, the co-face for upper) is passed as the third
argument to `_pair_message` and concatenated into the message MLP input — this is the
paper's use of the shared simplex feature in Eqs. (3)–(4).

### Equation (5) — the update

`h_σ = U( x_σ , ⊕ m_𝓑 , ⊕ m_𝓒 , ⊕ m_↓ , ⊕ m_↑ )`, a per-rank MLP over the concatenation
of the simplex's own feature and its aggregated incoming messages.

| Rank | Update MLP | Applied |
|---|---|---|
| nodes (rank 0) | `upd_0` (3·h → h) L279 | **L392** — `cat([x0, m0_cob, m0_up])` |
| edges (rank 1) | `upd_1` (5·h → h) L286 | **L411–413** — `cat([x1, m1_bnd, m1_cob, m1_low, m1_up])` |
| faces (rank 2) | `upd_2` (3·h → h) L291 | **L424** — `cat([x2, m2_bnd, m2_low])` |

Input-arity of each update MLP encodes the paper's per-rank adjacency structure in a
2-complex: **vertices have no boundary** (rank 0 uses only co-boundary + upper — L276),
and **faces have no co-face** (rank 2 uses only boundary + lower — L288).

### Equation (6) — readout

Per-rank multiset aggregation over the final simplex features. Delegated to the TopoBench
readout module (`PropagateSignalDown`, set in `configs/model/simplicial/mpsn.yaml`), per
the framework's standard composition. Docstring: **L42–L43**.

### Input contract

`MPSN.forward(x_all, incidence_1=B₁, incidence_2=B₂)` — **L486–507**; the full adjacency
index is built once from B₁, B₂ and shared across layers — **L507**.

### Verified conformance

- `pytest test/nn/backbones/simplicial/test_mpsn.py` → **10 passed**, against the real
  `SimplicialCliqueLifting` (toponetx) and `NNModuleAutoTest`.
- Backbone line coverage **97 %** (118 stmts, 3 missed) — above the ≥ 93 % bar.
