# Principles for Building Beautiful Research-Challenge Code

> This file is the contract. Every line of new code in this fork must pass
> these principles before it ships. When in doubt, re-read this. Pull
> requests that violate the principles will be revised before merge — even
> if the model trains.

Goal:
Build challenge code like a tiny scientific instrument — not a hackathon repo.

The standard is not "works on leaderboard."
The standard is:

* mathematically honest
* easy to verify
* elegant under pressure
* obvious to extend
* impossible to misunderstand

Think:

* tinygrad
* sqlite
* xv6
* well-written research infra

Especially in topological/geometric ML, code quality itself becomes part of the scientific contribution.

---

# 1. Abstractions Are Sacred

The biggest failure mode in research code:
ideas leaking across boundaries.

Bad:

* topology logic inside dataloaders
* dataset-specific hacks inside model code
* training assumptions inside lifting operators
* giant utility files touching everything

Good:

```python
complex = lift(graph)
embeddings = model(complex)
loss = objective(embeddings)
```

Every layer should have:

* one responsibility
* one vocabulary
* one abstraction level

If changing a lifting method requires editing training code, your boundaries are wrong.

The maintainer should feel:

> "Of course this belongs here."

---

# 2. Make the Math Visible in the Code

Research code should preserve the ontology of the math.

Bad:

```python
x2 = process(x)
```

Good:

```python
boundary_matrix = build_boundary_matrix(complex)
laplacian_1 = build_hodge_laplacian(boundary_matrix)
edge_embeddings = laplacian_1 @ edge_embeddings
```

The implementation should teach the paper.

A reviewer should be able to map:
paper → code
without archaeology.

Especially for TDL:

* simplices should remain simplices
* cochains should remain cochains
* filtrations should remain filtrations

Do not flatten everything into anonymous tensors too early.

---

# 3. Data Flow > Control Flow

tinygrad-style principle.

Bad:

```python
for batch in loader:
    x = preprocess(batch)
    x = model(x)
    x = pool(x)
    x = classifier(x)
```

Better:

```python
prediction = (
    graph
    | lift_to_complex
    | topo_encoder
    | persistence_pool
    | classifier
)
```

Structure code as transformations on mathematical objects.

You want the architecture to be visually understandable.

---

# 4. Shapes Are Types

In geometric/topological ML:
shape carries meaning.

Bad:

```python
Tensor(shape=[128, 64])
```

Good:

```python
NodeFeatures[B, N, D]
EdgeFeatures[B, E, D]
TriangleFeatures[B, T, D]
```

Or at minimum:

```python
# x_edges: [batch, num_edges, hidden_dim]
```

You should never wonder:

* what dimension means what
* whether tensors are node- or edge-level
* whether orientation is preserved

Most subtle topology bugs come from hidden shape semantics.

---

# 5. Optimize for Proof-of-Correctness

The best research code feels obviously correct.

That means:

* tiny functions
* explicit invariants
* assertions everywhere
* deterministic seeds
* reproducible pipelines

Bad:

```python
return x.transpose(1, 2) @ y
```

Good:

```python
assert boundary.shape[0] == num_edges
assert boundary.shape[1] == num_nodes

laplacian_0 = boundary.T @ boundary
```

The reader should trust the code without mentally simulating the entire repo.

---

# 6. Tests ARE the Specification

sqlite-level quality comes from treating tests as the real spec.

Especially in topology:
you already know many invariants.

Examples:

```python
def test_boundary_squared_zero():
    assert torch.allclose(B1 @ B2, 0)

def test_permutation_invariance():
    pred1 = model(graph)
    pred2 = model(permuted_graph)
    assert close(pred1, pred2)

def test_filtration_monotonicity():
    assert filtration[t] <= filtration[t + 1]
```

Your tests should encode:

* algebraic truths
* geometric truths
* stability assumptions
* equivariance
* orientation consistency

The tests should communicate:

> "This person understands the mathematical object."

---

# 7. Tiny Synthetic Examples First

Before benchmarks:
build toy worlds.

Example:

```python
V = {0,1,2}
E = {(0,1), (1,2), (0,2)}
```

Manually verify:

* boundary matrices
* Laplacians
* persistence diagrams
* orientations
* filtrations

If your method cannot be explained on:

* a triangle
* a tetrahedron
* a 5-node graph

then it is not mature.

Most great systems emerge from obsession with tiny examples.

---

# 8. Remove 30–50% of the Code

Almost every first implementation is bloated.

Delete:

* premature abstractions
* dead helpers
* generic registries
* unnecessary configs
* old experiments
* speculative flexibility

Bad:

```python
TopoLiftingFactoryRegistryManager
```

Good:

```python
lift_graph_to_complex(graph)
```

Research elegance usually means:

* fewer files
* fewer concepts
* fewer indirections

Every abstraction must earn its existence.

---

# 9. Benchmarking Code Should Feel Boring

The method should be novel.
The experimental infrastructure should feel industrial.

Good:

```bash
python train.py --config configs/cora.yaml
```

Bad:

```bash
python final_run_v3_real.py
```

Requirements:

* deterministic
* reproducible
* simple
* transparent

No mystery preprocessing.
No hidden scripts.
No manual steps.

The more novel the method,
the more conservative the surrounding engineering should be.

---

# 10. Respect the Geometry

Most weak TDL implementations secretly destroy the topology.

Bad:

```python
x = torch.cat([node_features, edge_features], dim=-1)
x = mlp(x)
```

Good:

```python
x_nodes = node_encoder(nodes)
x_edges = edge_encoder(edges)

x_edges = hodge_laplacian @ x_edges
```

The architecture should preserve:

* grading
* orientation
* adjacency structure
* geometric meaning

Do not reduce everything to generic MLP pipelines.

---

# 11. One Experiment = One Commit

Your git history should read like a scientific notebook.

Good:

```bash
feat: add persistence pooling
fix: correct boundary orientation
perf: vectorize filtration construction
```

Bad:

```bash
misc fixes
```

Benefits:

* reproducibility
* reversibility
* cleaner final PR
* easier debugging
* easier ablations

---

# 12. Documentation Should Explain Design Decisions

Not:

> "Computes lifting."

But:

> "Constructs a clique-complex lifting while controlling simplex explosion through thresholded filtration."

Good docs explain:

* mathematical intent
* complexity
* memory tradeoffs
* invariants
* failure modes

The reader should understand:
WHY this exists,
not just WHAT it does.

---

# 13. Avoid Stateful Magic

Research code becomes unreadable when everything mutates implicitly.

Bad:

```python
self.cache.update(...)
global_state["complex"] = ...
```

Good:

```python
new_complex = update_filtration(complex)
```

Prefer:

* pure functions
* explicit inputs
* explicit outputs
* immutable intermediate structures

A reader should be able to mentally execute the code locally.

---

# 14. Beautiful Naming Matters

Naming is part of the science.

Bad:

```python
x1
tmp2
process_features
```

Good:

```python
boundary_operator
persistent_features
edge_coboundary
```

Good naming compresses understanding.

---

# 15. The Final PR Should Feel Distilled

Your exploration process can be chaotic.
The final artifact should not reveal that chaos.

The final repo should feel:

* compressed
* intentional
* inevitable

Like:

> this was always the cleanest possible implementation.

---

# 16. The Meta-Principle

Beautiful research code is not about cleverness.

It is about honesty.

Every line exists because it is necessary.
Every abstraction earns its place.
Every test proves something meaningful.
Every tensor preserves mathematical intent.

The best research PRs feel calm.

Not flashy.
Not "smart."
Just deeply coherent.
