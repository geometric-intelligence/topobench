# SheafTSP — Sheaf-Topological Signal Processing

**Track 2 (TNN) submission for the TDL Challenge 2026.**
Team: Fernando Espinosa.

## Model

SheafTSP is a spectral convolutional network built on a learned, weighted
sheaf Laplacian over the 1-cell adjacency structure of a lifted cell
complex. Each layer learns orientation-equivariant SO(d) restriction maps
(antisymmetrized Cayley generator, so `R_vu = R_uv^-1` by construction),
weights edges by a Gaussian kernel on the transport-consistency distance
`||s_i - R_e s_j||`, assembles the degree-normalized sheaf Laplacian
`L̂ = D^{-1/2} δᵀKδ D^{-1/2}` (spectrum in [0,2]), and diffuses with a
PPR-initialized scalar polynomial filter on the sheaf lazy walk
`P = I - L̂/2` (K = 10 hops). Alongside the diffusion machinery, an
unnormalized counting pathway injects the exact per-node triangle signal
`t_v = |B1||B2|·1` — endogenous to the lifted complex and exact under the
all-3-cliques lifting — through a sum-linear route to the prediction.
Transports are regularized by a bounded kernel-alignment term (λ = 0.01,
signal and bandwidth detached).

Official 72-run grid (12 GraphUniverse settings × 3 seeds × 2 tasks):
**community detection 0.4735, triangle-count MSE/triangle 0.0108.**
Per-run values: `2026_tdl_challenge/outputs/sheaf_tsp_full_grid/results.json`.

## Documents in this directory

| Document | Contents |
|---|---|
| [`sheaf_tsp_overview.html`](sheaf_tsp_overview.html) | Why the model is built the way it is — the hypotheses, each equation of the architecture, the design principles, and the grid results. **Start here.** |
| [`architecturediagram.html`](architecturediagram.html) | Flow diagrams of the end-to-end dataflow and the layer internals, plus the submission configuration table. |
| [`sheaf_tsp_explained.ipynb`](sheaf_tsp_explained.ipynb) | An executable walkthrough that runs every computation on a toy graph, verifies the claimed properties (orthogonality, equivariance, exact counting), and ends at the evaluation harness. |

The complete experiment log — every study, command, number, and
verdict, negative results included — lives in the development
repository alongside this integration.

## Submission files

```
topobench/nn/backbones/cell/sheaf_tsp.py                          backbone
topobench/nn/wrappers/cell/sheaf_tsp_wrapper.py                   wrapper
configs/model/cell/sheaf_tsp.yaml                                 model config
configs/transforms/model_defaults/sheaf_tsp.yaml                  lifting defaults
topobench/transforms/liftings/graph2cell/clique_cell_lifting.py   all-3-cliques lifting
configs/transforms/liftings/graph2cell/clique_cell.yaml           lifting config
topobench/transforms/data_manipulations/triangle_degree.py        verification transform
test/nn/backbones/cell/test_sheaf_tsp.py                          backbone tests
test/nn/wrappers/cell/test_sheaf_tsp_wrapper.py                   wrapper tests
test/transforms/liftings/graph2cell/test_clique_cell_lifting.py   lifting tests
test/transforms/data_manipulations/test_TriangleDegree.py         transform tests
```

## Usage

```bash
pytest test/nn/backbones/cell/test_sheaf_tsp.py -v

# quick run (macOS: MPS lacks sparse support, use CPU)
python -m topobench model=cell/sheaf_tsp dataset=graph/graphuniverse_inductive \
    trainer.accelerator=cpu trainer.devices=1 trainer.max_epochs=50

# official evaluation: MODEL_CONFIG = "cell/sheaf_tsp" is set in
# 2026_tdl_challenge/run_evaluation.ipynb (the one permitted line)
```

## Configuration reference (`sheaf_tsp.yaml` defaults)

| Knob | Default | Meaning |
|---|---|---|
| `n_layers` / `stalk_dim` | 3 / 2 | depth; SO(2) transports (≅ learned magnetic Laplacian) |
| `filter_basis` / `filter_order` | `ppr` / 10 | scalar PPR coefficients on `P = I − L̂/2`; alternatives: `monomial`, `chebyshev` |
| `kernel_distance` | `transport` | kernel on `‖s_i − R_e s_j‖`; alternative: `feature` |
| `reg_form` | `alignment` | bounded kernel-alignment transport regularizer; alternative: `dirichlet` |
| `rotation_param` | `cayley` | SO(d) projection; alternative: `exp` (surjective) |
| `dropout` / `mlp_dropout` | 0.0 / 0.5 | signal-path dropout corrupts counting; regularize the transport MLP instead |
| `count_source` / `tri_warm` | `incidence` / 0.1 | endogenous count signal; warm-start channel scale |
| lifting | `clique_cell` | all 3-cliques as 2-cells, so the lifting is canonical and permutation-invariant |

## Collaboration

This project is a collaboration with Claude Fable (Anthropic). I am but
an ML engineer with an interest in mathematics, and I lean on a lot of
assistance to execute ideas in this field. Claude did much of the
mathematical development, the implementation, and the experiment
campaign; every design decision we kept is backed by a measurement in
the project's experiment log.

## References

- Tandon, Gould, Bhatia, Dominici, Ribeiro & Battiloro, "Consistent
  Geometric Deep Learning via Hilbert Bundles and Cellular Sheaves"
  (2026, arXiv:2605.06395) — framework: kernel-weighted sheaf Laplacian,
  polynomial sheaf filter, transport regularizer
- Bodnar et al., "Neural Sheaf Diffusion" (2022) — sheaf diffusion baseline
- Bamberger et al., "Bundle Neural Networks for message diffusion on graphs" (2024) — learned orthogonal bundle maps on graphs
- Chen, Chen, Villar & Bruna, "Can GNNs Count Substructures?" (NeurIPS
  2020) — expressivity context for the counting task
- Zhang et al., "MagNet" (NeurIPS 2021) — magnetic Laplacian family
  (the d = 2 specialization of our transports)
- Hansen & Ghrist, "Toward a Spectral Theory of Cellular Sheaves" (2019)
  — cellular sheaves, sheaf Laplacian, global sections
