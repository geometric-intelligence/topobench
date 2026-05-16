# Track 2 Submission: E(n)-Equivariant Topological Neural Networks

Participant: Gaurav Khanal  
Track: Track 2 — Topological Neural Networks  
Model: E(n)-Equivariant Topological Neural Networks  
Status: Draft / work in progress

This draft PR reserves and develops an implementation of E(n)-Equivariant Topological Neural Networks (ETNN) for TopoBench.

Planned implementation scope:

- Add a TopoBench-native ETNN backbone.
- Target the combinatorial-complex domain where possible.
- Use sparse relation-index message passing.
- Avoid dense pairwise coordinate tensor construction.
- Use native PyTorch reductions where appropriate.
- Add Hydra model configuration.
- Add unit tests for forward pass, tensor shapes, and equivariance behavior.
- Add pipeline integration test.
- Run the official GraphUniverse evaluation notebook and include `results.json`.

Reference:

E(n) Equivariant Topological Neural Networks.
