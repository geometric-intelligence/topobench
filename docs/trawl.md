# TRAWL in TopoBench

TRAWL lifts input data to cells, computes structural encodings, samples random
walks over the cells, and encodes the walk sequences with configurable neural
layers. It uses TopoBench's loaders, transform cache, collator, `TBModel`,
losses, evaluators, optimizers, Lightning trainer, callbacks and Hydra sweeps.
Every behavior is selected by configuration; no code branches on dataset names.

## Repository layout

TRAWL follows the backbone / wrapper / readout split of other TopoBench
models; the runner, `TBModel` and the config resolvers contain no
TRAWL-specific code.

- `nn/backbones/combinatorial/trawl.py`: the `TRAWL` backbone (learnable
  projections, sequence encoders, pooling and fusion).
- `nn/backbones/combinatorial/trawl_utils/`: sequence layers (Mamba, SISA,
  GRU, transformer, MLP, graph adapters) in `layers.py`; CPU walk sampling and
  guidance in `sampling.py`, with an optional `numba` kernel in
  `fast_walks.py`.
- `nn/wrappers/combinatorial/trawl_wrapper.py`: `TRAWLWrapper` samples walks
  for every complex in a batch and assembles the model output.
- `nn/readouts/trawl.py`, `nn/encoders/trawl.py`: the task head
  `TRAWLReadout` and the feature adapter `TRAWLFeatureEncoder`.
- `transforms/data_manipulations/trawl.py`: builds walk states, relations and
  TopoBench's HOPSE positional/structural encodings after any lifting.
- `model/pretraining/trawl.py`: optional self-supervised pretraining.
- Configs: `configs/model/{graph,cell,simplicial,hypergraph,combinatorial}/trawl.yaml`,
  `configs/transforms/model_defaults/trawl.yaml` (plus
  `model_dataset_defaults/trawl_<dataset>.yaml` for datasets with their own
  defaults), `configs/pretraining/trawl*.yaml` and
  `configs/experiment/trawl_*.yaml`.

Generic, opt-in features used by the recipes live in shared modules:
`evaluator/checkpoint.py` and `callbacks/model_checkpoint.py` (top-K
checkpoint evaluation), `dataloader/samplers.py` (seeded epoch order) and the
`seeded_stratified` and `imported` split types.

## Start here

```bash
python -m topobench model=cell/trawl dataset=graph/PROTEINS
python -m topobench model=cell/trawl dataset=graph/NCI1 model.backbone.architecture=sisa
python -m topobench model=simplicial/trawl dataset=simplicial/mantra_betti_numbers
python -m topobench model=graph/trawl dataset=graph/cocitation_cora
```

Like other TopoBench models, the model config's domain selects the lifting:
`cell/trawl` lifts graphs with the default cycle lifting, `simplicial/trawl`
with the clique lifting, `hypergraph/trawl` and `combinatorial/trawl` with
their default liftings, and data already in the model's domain is not lifted.
`graph/trawl` walks the original graph without higher cells. The TRAWL
transform always runs last (`configs/transforms/model_defaults/trawl.yaml`).
To use another lifting, override the transforms as for any other model and
set `model.backbone.max_rank` to the highest rank you want.

Pure-PyTorch Mamba is the default and runs on CPU. The official kernel is
selected with `model.backbone.layer_options.mamba.backend=mamba_ssm` (validated
with `mamba-ssm==2.2.6.post3`, `transformers==4.44.2`); selection never falls
back silently.

## Relations and walks

`transforms.trawl.graph` chooses the walk graph:

- `hasse`: immediate incidences only; every step changes rank.
- `augmented_hasse` (default): any selected incidences and same-rank adjacencies.
- `cell_overlap`: cells in `overlap_ranks` connect when they share a vertex.

`transforms.trawl.neighborhoods` uses TopoBench names (`up_incidence-0`,
`down_adjacency-1`, `2-up_adjacency-0`, ...); the default is `up_incidence-r`
for every rank below `max_rank` (`up_adjacency-0` for `graph/trawl`), traversed
in both directions. `model.backbone.num_neighborhoods` must match the number of
relations. `experiment=trawl_adjacency` and `experiment=trawl_mixed` are
examples of other relation sets.

Walk options live under `model.backbone_wrapper.walks`: `k`, `length` (default 32/32),
`start_policy` (`coverage` favours rarely visited start states until coverage
plateaus), `epsilon`, `reverse` (adds reversed copies), and `guidance`
(`gamma`, `diffusion_t`, `dense_limit`). Walks are non-backtracking unless a
state has no other exit.

- `walk_scope=union` samples K walks over all relations together; `separate`
  samples K per relation and fuses them (`fusion=mean|concat|learned|attention`,
  `encoder_sharing=shared|independent`).
- `model.backbone_wrapper.walk_refresh=train` resamples every training step;
  `fixed` keeps one sample.
- `model.backbone_wrapper.eval_views` averages several walk samples during
  validation and testing.

## Encodings

`transforms.trawl.encodings` adds per-state positional/structural encodings
computed on each relation's graph (or on the union of relations with
`encoding_scope=union`). Apart from `local` (`log(1 + degree)` and neighbor
count), every key names one of TopoBench's HOPSE encodings and holds its
arguments: `LapPE`, `RWSE`, `ElectrostaticPE` and `HKdiagSE`. They are computed
on the unweighted, active part of each relation; states outside a relation get
zeros. The default is

```yaml
encodings:
  local: true
  LapPE: {max_pe_dim: 8}
  RWSE: {max_pe_dim: 8}
  ElectrostaticPE: {}
  HKdiagSE: {kernel_param_HKdiagSE: [1, 6]}
```

`walks.guidance` reweights transitions by `|exp(-tL)_ij|^γ`, with L the
symmetric Chung Laplacian of the weighted relation. It uses an exact dense
eigendecomposition limited by `dense_limit=2048`; exceeding it raises an error
rather than switching to an approximation. `graph/trawl` therefore disables
guidance (`gamma: 0`), since node-level graphs are usually larger.

Optional categorical colors come from `color_key` (set `model.backbone.num_colors`
to cover them); `color_refinement=true` adds encodings for color-pair subgraphs.
Absent ranks stay empty and empty relations get zero encodings. Unvisited cells
fall back to their projected input features; with the default budget of 32
walks of length 32 per complex, most nodes of a large graph are unvisited in
each step, so raise `walks.k` for node-level tasks.

## Layers and outputs

`architecture` selects `mamba`, `sisa`, `hybrid` (M–S–M–S–M at depth 5),
`transformer`, `gru` or `mlp`; `layer_options` configures each kind. For any
other stack, set `model.backbone.layers` to a list of layer configs
(`experiment=trawl_custom_layers`). A layer may be a Hydra `_target_` mapping
`[walk, time, hidden]` to the same shape, or `kind: graph` wrapping a PyG module
that runs on each walk as a path graph.

The wrapper returns contextual cell features `x_r`/`batch_r` for any TopoBench
readout, plus walk and graph embeddings. `TRAWLReadout` supports graph and node
tasks with `aggregation=embedding` (head on the pooled graph embedding) or
`walk_logits` (head per walk, logits averaged). Time pooling is `mean`, `max` or
`mean_max`; `occurrence_pooling` and `graph_readout=walks|cells` control how
walk states become cell and graph embeddings.

Input widths are inferred from the training data before the optimizer is built.
To use a native feature encoder, wrap it in
`topobench.nn.encoders.trawl.TRAWLFeatureEncoder` and set
`model.backbone.in_channels`.

## Pretraining

```bash
python -m topobench model=cell/trawl dataset=graph/PROTEINS pretraining=trawl
```

Self-supervised pretraining reconstructs masked cell features and can also
predict colors and masked connections (`pretraining.objectives`). It uses only
the training split, never labels, and hides encodings that would leak masked
structure. The best validation encoder is restored before supervised training;
`reset_head` reinitializes the head. Resume with
`pretraining.ckpt_path=.../pretraining/last.ckpt`; `pretraining.min_delta` sets
the improvement needed by both checkpointing and early stopping. The runner
calls the configured procedure through the generic `pretraining` hook (see
`topobench/model/pretraining/__init__.py`); `pretraining=null` (default)
disables it.

## Recipes

The `configs/experiment/trawl_*.yaml` configs are ordinary compositions of
the options above:

```bash
python -m topobench experiment=trawl_proteins_mamba
python -m topobench experiment=trawl_proteins_hybrid
python -m topobench experiment=trawl_nci1_hybrid    # also trawl_nci1_sisa, trawl_nci1_mamba
python -m topobench -m experiment=trawl_proteins_hybrid seed=40,41,42,43,44
```

`trawl_recipe.yaml` holds the shared PROTEINS/NCI1 settings: cycles up to six
nodes, the default HOPSE encodings, hidden width 224, one-logit BCE with
walk-logit averaging, seeded 50/25/25 stratified splits, batch 2 with gradient
accumulation, SSL pretraining (`pretraining=trawl_recipe`), three evaluation
views and `deterministic=strict`. `trawl_proteins_hybrid` adds top-5
checkpoint weight averaging.

These recipes follow the original TRAWL experiments rather than TopoBench's
standard protocol (e.g. `split_type=random` with a fixed `data_seed`), so their
scores are not directly comparable with other models' results.

## Reproducibility and performance

Walks, splits, epoch order, pretraining masks and initialization are seeded.
`deterministic=strict` also enables PyTorch's deterministic kernels (raising on
any without one), passes the setting to every Lightning trainer, and sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (launchers should export it before CUDA
starts). Under this mode two runs on separate GPUs produced bit-identical
checkpoints, metric histories and evaluations, for about 9% extra wall time.
Exact repeats assume the same software stack, GPU model, device count and
evaluation batch size.

Speedups that leave results unchanged: an exact reimplementation of NumPy's
weighted choice (compiled when `numba` is installed, `pip install topobench[trawl]`),
cached transition distributions and evaluation walks, host-side walk bookkeeping
with non-blocking batch transfer (`TBModel` passes host copies of the wrapper's
`host_fields`), pinned memory and `evaluator.validate_args=false`.

Memory: activations scale with the number of graphs in a batch times the walk
budget. The default config uses the sequential Mamba scan and activation
checkpointing (`checkpoint_layers`), which keeps a few MiB of activations per
graph; `scan: parallel` without checkpointing is faster but needs hundreds of
MiB per graph.

## Splits and evaluation

- `split_type=seeded_stratified` performs two stratified splits with
  `random_state=data_seed`; `split_type=imported` reads `train`/`valid`/`test`
  index arrays from `split_file`.
- `dataset.dataloader_params.order_seed` shuffles the training set with
  `EpochRandomSampler` (`dataloader/samplers.py`): epoch `e` uses a permutation
  seeded by `order_seed + e + 1`, independent of other random draws. Without
  it, the usual seeded shuffle is used. Under several processes, evaluation
  uses `UnpaddedDistributedSampler`, which visits every sample exactly once
  instead of padding uneven shards with repeats.
- `checkpoint_selection.strategy` is `best`, `weight_average` or
  `logit_ensemble` (`evaluator/checkpoint.py`): the latter two average the
  weights, or the logits, of the `checkpoint_selection.top_k` best
  checkpoints. The recipes rank them with `RankedModelCheckpoint`
  (`callbacks/model_checkpoint.py`), which evicts the latest of tied
  checkpoints and indexes the top K in `checkpoint_index.json`, so a run
  directory can be moved and replayed with `train=false ckpt_path=...` (a
  supplied checkpoint is never deleted by `delete_checkpoint_after_test`).
  With `save_top_k: 1` it behaves like Lightning's `ModelCheckpoint`, which
  stays the default for other models.
- One-logit binary outputs are scored as two-class probabilities, so AUROC does
  not depend on the evaluation batch size. Plateau schedulers step only on
  validated epochs.
- With `checkpoint_selection.report=true` (set by the recipes), a run writes
  `evaluation_report.json` with the selected checkpoints and final metrics.

## Development checks

```bash
python -m pytest -q -k "trawl or TRAWL" test
python -m ruff check topobench
```

Tests mirror the source tree (e.g. `test/nn/backbones/combinatorial/test_trawl.py`,
`test/nn/wrappers/combinatorial/test_trawl_wrapper.py`,
`test/pipeline/test_trawl_pipeline.py`).
`test/nn/backbones/combinatorial/fixtures/trawl_reference.npz` holds SISA
outputs from the original implementation.
