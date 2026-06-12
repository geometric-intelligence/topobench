# Track 2 — SheafHyperGNN Submission

## Track

Track 2 — Topological Neural Networks (TNNs)

## Team Name

s/pairwise/ho

## Model

Sheaf Hypergraph Networks (SheafHyperGNN)

## Status

Draft / work in progress

## Summary

This draft PR develops a TopoBench-native implementation of SheafHyperGNN from
"Sheaf Hypergraph Networks" (Duta et al., NeurIPS 2023) for the 2026 TDL
Challenge.

The model generalises standard hypergraph convolutions by equipping every node
and hyperedge with a *d*-dimensional stalk and learning per-pair restriction
maps that define a cellular sheaf over the hypergraph. The resulting sheaf
Laplacian replaces the standard hypergraph Laplacian in the diffusion operator,
enabling richer structural representations without additional data requirements
beyond the incidence matrix.

## Planned Implementation

- [x] Inspect official implementation and paper equations; confirm feasibility.
- [ ] Add SheafHyperGNN backbone under `topobench/nn/backbones/hypergraph/sheaf_hypergnn.py`.
- [ ] Add Hydra config under `configs/model/hypergraph/sheaf_hypergnn.yaml`.
- [ ] Add unit tests.
- [ ] Update `test/pipeline/test_pipeline.py`.
- [ ] Run the official GraphUniverse evaluation notebook.
- [ ] Add generated `results.json`.

## Reference

Duta, I., Cassarà, G., Silvestri, F., & Liò, P.
"Sheaf Hypergraph Networks." *NeurIPS 2023.*

Paper: https://arxiv.org/abs/2309.17116

Official implementation: https://github.com/IuliaDuta/sheaf_HNN
