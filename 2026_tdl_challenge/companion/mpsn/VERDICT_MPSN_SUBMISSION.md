# AUDIT VERDICT — MPSN Track-2 submission (TDL Challenge 2026)

Generated programmatically from hash-sealed evidence on
2026-07-05 16:00 UTC. First executed instance of the
operational audit architecture (charter → classify → calibrate → read → compose →
verdict); self-applied, with independence impairment disclosed per IIA 1130.

## Charter (frozen before the confirmatory run)
Scope: MPSN implementation + capability claim ("implements Def. 4 faithfully AND
exhibits >1-WL triangle capability on GraphUniverse"). Access: white-box.
Preregistered criteria H1–H4 stored verbatim in `audit_results.json`.

## Classify
- L1 (statistical) fires — output distributions vs baselines. ✔ exercised.
- L2 (geometric) fires (embedding-bearing model). ✔ exercised at a declared audit
  scale: sliced-W2 drift, H1-bottleneck fracture, LOF collapse — all with
  self-calibrated within-train nulls.
- L3 (logical) fires — challenge spec as φ_required. ✔ exercised.
- L4 (causal/identification) fires — capability claim under confounding. ✔ exercised
  (confound-as-treatment design + WL-hard probe as intervention + placebo-outcome
  refutation bounding the generic-capacity component).

## V — Verdict
**pass(L1, L2-at-audit-scale, L3, L4-identified-and-refuted).** O4 cleared: the
geometric reference is now exercised. Certificate still awaits the external stamps
in Γ (independent re-run; organizer evaluation).

## F — Findings (from sealed evidence)
- H1 identifiability: PASS — Δ>0 with 95% CI excluding 0 in 6/6 regimes
  (lowest-confound R3: Δ=+0.53
  [+0.39,+1.05]; highest R4:
  Δ=+0.06 [+0.04,+0.10]).
- H2 monotonicity: PASS — Spearman(ρ,Δ)=-0.89, exact permutation
  p=0.0167 (6 regimes).
- H3 probe certification: PASS — Δ=+1.00
  [+1.00,+1.01]; GNN and degree R²≈0.
- H4 feature robustness: PASS — real 15-dim features do not rescue the 1-WL GNN.
- Capacity falsification: probe Δ CI-lower ≥ +0.996 across a 4× width / 2× depth
  GNN sweep — a provable 1-WL capability limit, not under-parameterisation.
- Q1 structural sensitivity: crossover — higher-order gain +0.041
  [+0.011,+0.071] under severe heterophily vs -0.157
  [-0.219,-0.090] under strong homophily (finding **against** the
  submitted model, reported; budget diagnostic: partly optimization cost, partly genuine).
- Gate attribution: **negative result** — G1 fails both seeds across three design
  iterations; retained value: opt-in fusion default (uniform init demonstrably harmful).
- L2 geometric (`hardening_results.json`): in-distribution PASS — drift
  0.79x, fracture
  0.87x, LOF
  0.58x of self-calibrated
  95% nulls. Probe OOD: drift 2.90x,
  LOF 1.70x, H1 fracture
  1.08x — the shift is
  metric/density, not small-scale cyclic.
- L4 placebo (degree): STRICT FAIL, informative — generic floor
  Δ=+0.023
  [+0.012,+0.036];
  probe/floor ≈ 44x,
  smallest on-bench Δ/floor ≈ 2.8x.
- MPSN capacity on probe: Δ CI-lower > 0.99 at hidden 16/32/64.
- Conformance (L3): 10 passed  (test/nn/backbones/simplicial/test_mpsn.py); backbone coverage
  97% (118 statements, 3 missed); readout GatedCrossRankReadout: 6 passed, coverage 100%;
  labels verified by two independent methods on every graph.

## T — Thresholds & calibration
Preregistered, CI-based (95% nested percentile bootstrap over seed×graph); H2 via exact
permutation (n=6 ⇒ min p=1/720); probe window [0.90, 1.10]; |R²|≤0.05 nulls.
Multiplicity (supplementary, criteria unchanged): Holm across 6 regimes —
H1 all<.05: True; H4 all<.05: True.

## N — No-go theorems terminating the verdict
L1 Incompleteness (bootstrap CIs, not tail-exhaustive); L3 Extended Rice (behavioural
spec class only); L4 Counterfactual Incompleteness (identification via designed probe;
sensitivity class not E-value-quantified).

## O — Obstructions
O4: L2 geometric reference not exercised (persistence drift / fracture / LOF on the
learned simplicial embeddings) — flagged, not hidden. O1–O3 clear (single version,
coherent thresholds, no verdict incompatibility).

## Λ — What this audit cannot say
- L2 residual (Geometric Incompleteness): exercised at ONE declared audit scale on
  ONE regime; scalar summaries are not bi-Lipschitz classifying — untested scales and
  dimensions remain, and probe H1 near-null shows scale-sensitivity of the fracture check.
- L4 residual: the placebo bounds the generic component (+0.023) but is a point
  comparison, not a sensitivity class; no E-value analogue exists for this claim form.
- Self-audit: harness, generator wrappers, probe and verdict share one author —
  independence impaired (IIA 1130); disclosed, not fabricated.
- Scale: 2 seeds, small graph samples; constant features isolate topology by design
  (MPSN capacity now swept 16/32/64 on the probe).
- Byte-level hashes vary across re-runs (timestamps inside evidence files); *numeric*
  results are seed-deterministic. Acceptance (challenge bar, PMLR review) is external.

## Γ — Cycle plan
- T1: commit official `results.json` (run_evaluation.ipynb, pinned env) → re-verify.
- T3: independent re-run by co-author (closes the 1130 impairment) or second
  implementation reproducing probe Δ.
- T5: L2 exercised at audit scale on one regime — extend per-regime and across
  scales before any claim upgrade.
- T7: organizer evaluation = external stamp; verdict upgrade only then.

## Evidence manifest (SHA-256)
- `audit_results.json` — `a245bbc6c8bb53eb…`
- `ablation_results.json` — `4217c5e949b5c670…`
- `structural_sensitivity.json` — `403c971edb3bef62…`
- `gate_diagnostics.json` — `84c8181ede54ba2a…`
- `mpsn_confirmatory_r2.png` — `cb4d381d83e89981…`
- `mpsn_confirmatory_delta_vs_rho.png` — `85e79c5819c9a0b7…`
- `mpsn_capacity_ablation.png` — `1a0c056cb77a4364…`
- `mpsn_structural_sensitivity.png` — `5567edebfd4b57a1…`
- `mpsn_gate_diagnostics.png` — `e258e9e134716d0d…`
- `hardening_results.json` — `cfe9d8ad8d5b16bd…`
- `mpsn_l2_geometric.png` — `22af271bc3bc5edb…`
