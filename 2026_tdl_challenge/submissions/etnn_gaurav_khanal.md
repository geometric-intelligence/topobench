# Track 2 Submission: E(n)-Equivariant Topological Neural Networks

Participant: Gaurav Khanal
Track: Track 2 - Topological Neural Networks
Team: E(n)igma
Model config: `combinatorial/etnn`
Status: Full GraphUniverse evaluation completed

## Scope

This submission adds a TopoBench-native combinatorial ETNN backbone for lifted
GraphUniverse inputs. The implementation follows the topological feature-update
part of E(n)-Equivariant Topological Neural Networks (ETNNs): rank-wise cell
states are updated by aggregating relation-specific messages over combinatorial
complex neighborhoods.

The original ETNN formulation combines:

- combinatorial-complex neighborhood message passing over cells;
- geometric invariants computed from node coordinates;
- an E(n)-equivariant coordinate update.

GraphUniverse does not provide physical coordinates. For that reason, this
implementation is intentionally coordinate-free: it implements the ETNN
combinatorial message-passing core over TopoBench neighborhoods, but it does not
claim to implement the full coordinate-update component of ETNN on these
datasets. This adaptation keeps the coordinate extension point explicit while
remaining compatible with TopoBench's graph-to-combinatorial lifting pipeline.

## Design Choices

- `GraphTriangleInducedCC` lifts graph datasets into combinatorial complexes
  with rank-0, rank-1, and rank-2 cells.
- `AllCellFeatureEncoder` projects selected ranks to a shared hidden dimension.
- The ETNN backbone uses the configured TopoBench neighborhoods as typed
  relation functions:
  - same-rank adjacencies for rank 0, rank 1, and rank 2 cells;
  - bidirectional node-edge incidences;
  - bidirectional edge-face incidences.
- Each relation has a separate message MLP, and each destination rank has a
  rank-specific update MLP.
- Sparse neighborhood tensors are converted to explicit sender-receiver edge
  indices with rows treated as receivers and columns as senders.
- Empty-rank sparse placeholders are filtered/compacted so lifted batches with
  no rank-2 cells remain valid.
- Tensor allocation follows the active feature device, so the same code runs on
  CPU and CUDA devices.

## Evaluation

The official GraphUniverse evaluation completed all 72 runs.

Metadata:

- study id: `2026-06-12_10-14-24`
- model config: `combinatorial/etnn`
- seeds: `42`, `43`, `44`
- output directory: `2026_tdl_challenge/outputs/etnn`

Headline in-distribution metrics from `results.json`:

| Task | Metric | Mean +- std | Runs |
| --- | --- | ---: | ---: |
| Community detection | Accuracy | `0.4534 +- 0.1326` | 36 |
| Triangle counting | MSE / total triangles | `0.1213 +- 0.1137` | 36 |

Homophily slices:

| Task | h_lo | h_mid | h_hi |
| --- | ---: | ---: | ---: |
| Community accuracy | `0.3195 +- 0.0047` | `0.4205 +- 0.0330` | `0.6203 +- 0.0610` |
| Triangle MSE / triangles | `0.0259 +- 0.0301` | `0.1141 +- 0.0411` | `0.2239 +- 0.1307` |

## Tests

The implementation includes unit and integration coverage for:

- rank-wise output shapes;
- no-coordinate GraphUniverse compatibility;
- `TuneWrapper` compatibility;
- sparse neighborhood direction;
- empty-rank sparse placeholder compaction;
- Hydra composition with graph-to-combinatorial lifting;
- optional one-epoch pipeline smoke test.

Verified locally before submission:

```bash
uv run pytest test/nn/backbones/combinatorial/test_etnn.py test/pipeline/test_etnn_pipeline.py -q
```

Result:

```text
8 passed, 1 skipped, 1 warning
```

The skipped test is an explicit one-epoch pipeline smoke test that may
download/process data and is intended for manual execution when the environment
is available.

## References

- Claudio Battiloro, Ege Karaismailoglu, Mauricio Tec, George Dasoulas,
  Michelle Audirac, Francesca Dominici. E(n) Equivariant Topological Neural
  Networks. arXiv:2405.15429.
- Official ETNN implementation:
  https://github.com/NSAPH-Projects/topological-equivariant-networks
