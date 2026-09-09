# R5B — Deviations and Scoping Notes

Engineering deviations from the literal plan text, with rationale. None is
an owner ratification; every scientific default remains
`proposed_protocol_default`.

## DEV-R5B-1 — The five-day REAL control-flow verification is blocked-honest

**Plan**: PHASED R5B deliverable 12 — "five-day real control-flow
verification + synthetic feature-formula fixtures — no full-development
feature materialization."
**Deviation**: the real half cannot run before the owner's
`VerificationAuthorizationRef` (decisions 21/R-5) exists — identical in
kind to DEV-R5-6 (the R5 real-slice ladder half). What ships now: the
complete policy-gated real path (`read_mbp1_partition_frame` /
`build_mbp1_source_artifact_from_paths` — authorize-before-path, schema
validation before rows decode, refusal without a policy, fail-before-path
on denied days) and the synthetic E2E that proves the identical
control-flow shape (S05 materialize→persist→join, S09 controlled study in
the 0-valid-fold safe-failure state). The real pipeline entry
(`pipeline_baseline_verification_entry`) does NOT wire an
`mbp1_evidence_source`; a real MBP-1 plan therefore refuses at S00 with
the exact missing-seam reason — fail-closed, never fabricated. Completing
the real wiring (anchors from the published baseline v2 dataset + the
authorized mbp1 partitions) is acceptance-time work under the owner's
authorization, recorded as an open blocker in `GATE_SUMMARY.md`.

## DEV-R5B-2 — Module placement of the controlled-study workflow

**Plan**: the R5B file list names the six `features/mbp1_*.py` modules
"+ their test suites; dashboard additions…"; deliverable 10 (the
controlled Baseline vs Baseline+MBP-1 study workflow) names no module.
**Deviation**: the workflow lives in
`ifvg/ml/controlled_feature_study.py` — beside the ladder it
parametrizes, in the lane that owns model protocols (the §18 module count
"ml/ ×14 + fixtures" gains one). The alternative (embedding it in
`supervised_ladder.py`) would have mixed the persistence/identity layer
into the pure runner.

## DEV-R5B-3 — Derived-identity structure evolution (pre-acceptance)

`BundleFeatureViewPayload` gained `mbp1_feature_artifact_id` (None for
non-MBP-1 bundles) and the ladder id now binds a typed `feature_source`
({frozen_tier} | {resolved_bundle + names + evidence_ref}) instead of a
bare tier string — after the adversarial round (finding F2), the bundle
path's `evidence_ref` carries the exact MBP-1 feature artifact id, so two
challenger ladders over different evidence artifacts can never share one
`ladder_id`. Both are required for honest identity (the same view+bundle
over different MBP-1 evidence, or the same view under tier-vs-bundle
parametrization, must never share an id). Consequence: derived in-memory identities
(bundle-view ids, ladder ids, and therefore the S05/S09/S10 stage-result
ids re-derived from them) differ from R5-era values; a retry of an R5-era
pipeline across this boundary legitimately RE-EXECUTES those stages
(reuse is proven by identity — an id mismatch re-runs, it never lies).
No immutable persisted artifact is mutated; no store contract changed
shape. Pre-acceptance schema evolution under the authoring-vs-acceptance
model (V3 P0-8).

## DEV-R5B-4 — Decision-log numbering

The plan's docs list reserves D-039…D-045; the R5-era reservation note in
`docs/DECISIONS.md` promised "the MBP-1 activation block … lands with
R5B". That entry landed as **D-046** — the next number after the reserved
block — rather than overwriting a landed decision. Additive; the
reservation note now points at it.

## DEV-R5B-5 — Dashboard mount point

`FRONTEND_UX_CONTRACT.md` §35 mandates the R5B panels (availability,
coverage, bundle selection, comparison, missingness, stage-window
drill-down, persistent `research_only_offline` labeling) but no mount
point, and §36's FUX-IA-002 fixes the Experiments sub-nav at five entries.
The panels render as the "MBP-1 Order Flow (research-only offline)"
expander on the pipeline surface (`render_pipeline_run` tail; exact-ID
auto-fill from the selected run's persisted stage evidence) plus the
Configure-phase integration (bundle selection after activation; the
logistic-only model restriction with the tier-lock caption; the boundary
badge). The fixed sub-nav and the M0–M3 panel are untouched.

## DEV-R5B-6 — Engineering definitions inside `ifvg_order_flow_mbp1_formula_v1`

The plan fixes the metric NAMES (DT §6.2) but not every numerical
formula. The registered formula version pins: CKS OFI over consecutive
ADMITTED pairs (window-scoped — the pre-window predecessor is excluded);
aggressor fractions from Databento trade `side` ('B' buy / 'A' sell;
NaN when the window has no trades); depletion/replenishment as
price-step-or-size-change book transitions per side; absorption =
traded size / (1 + |Δmid| in ticks); intensities over the
anchor-to-anchor span in seconds; snapshot metrics from the LAST admitted
event's top of book with zero-denominator imbalances as formula-NaN
(window VALID). Sequence-gap semantics: positive jump > 1 = gap, vendor
reset (decrease) = not a gap; day coverage = 1 − gap-ns/span-ns with the
`MIN_DAY_COVERAGE_FRACTION = 0.95` engineering default. Typed-missing
precedence: day → anchor → boundary → content (DECISIONS_TAKEN #49).
Changing ANY of these mints a new formula version and a new resolved
block id.

> **SUPERSEDED by R5B.1 (2026-08-28, owner planning decision Q1):** the
> sequence-gap semantics above are WITHDRAWN and unrepresentable
> (`gap_semantics` is the Literal `mbp1_source_coverage_declared_evidence_v2`;
> raw sequence jumps are diagnostics only). Coverage is evidence-based under
> policy v2 — see `../R5B.1/` (D-048; DECISIONS_TAKEN #67–#75). The formula
> version moved to `ifvg_order_flow_mbp1_formula_v2` (block v3). This entry
> is retained as the R5B historical record only.

## Scoping notes (not deviations)

- The R5-era tests that pinned the planned state
  (`test_planned_blocks_fail_closed_everywhere`,
  `test_mbp1_bundles_refuse_before_r5b`,
  `test_mbp1_study_plan_refuses_before_r5b`, the two Configure AppTests)
  were release-gated by TEST_MATRIX §3.8's two-sided "MBP-1 activation
  state" row and flipped to their R5B halves in the same change; the
  activation-event proof now replays `with_activated_block` over the
  exported pre-activation state and asserts it reproduces the published
  registry exactly.
- The controlled study pins `ifvg_context_logistic_l2_v1` on both arms
  because the CatBoost fold runner is tier-locked inside the frozen M0–M3
  lane (never modified in v1) — enforced at the ladder, the stage-plan
  readiness (launch-blocking), and the UI, all with the same reason text.
- `docs/ML_TRAINING_WORKBENCH.md` remains untouched and uncommitted
  (user-owned dirty file; no R5B content belongs there).
