# R6.1 — Deviations and Scoping Notes

Written before the adversarial round and reconciled after it
(`ADVERSARIAL_REVIEW_RESOLUTION.md`; the reviewers challenged the first
draft's header, DEV-R6.1-8 and DEV-R6.1-10 — F16 / S8); every entry states
what the CODE does now.

## DEV-R6.1-1 — S07 stays a plan prerequisite of S08/S09 (frozen dependency table)

Plan D14 describes a descriptive KMeans study as needing S05/S06/S08/S09a/S10
only. The frozen 16-stage contract (`STAGE_DEPENDENCIES`: S08 ← S07; S09 ←
S05, S07, S08) is unchanged, so an operator plan with a regime study still
carries `07_derive_labels` (+ a label policy); a descriptive study simply
never consumes the labels (S09a runs; no supervised protocol is pinned —
`model_protocol_id` may be `None` when a regime study is present, the one
validator relaxation). The label-free `build_candidate_folds_from_schedule`
is the S08 builder whenever derived labels are absent (defensive path;
unit-tested) and the CLI/unit fold-schedule paths use it directly.
DECISIONS_TAKEN #77.

## DEV-R6.1-2 — Semantic-id, assessment-id, bundle-view-id, simulation-id evolution (pre-acceptance)

`PipelineSemanticSpecPayload.regime_study` (default `None`; every semantic id
moves), `regime_capability_assessment_id` (per-fold stability, transition
fields, the renamed gate), `bundle_feature_view_id`
(`regime_fold_feature_artifact_id`), `feature_block_registry_hash` (panel
registration), every bundle-path ladder id (`comparison_row_id`, fold-feature
evidence ref), account/portfolio simulation ids, `SimulationProtocol`,
charter `search_id` (D15 policy fields, default `none_v0`). Fit ids and
`resolved_regime_protocol_id` are proven unchanged (golden R6 fit id; the
candidate protocol id `a9b7888ad1ff…` of the R6 smoke reproduces). No
persisted real artifact exists outside tmp/smoke roots; nothing immutable was
mutated (DEV-R5B-3 / DEV-R6-6 precedent). DECISIONS_TAKEN #78.

## DEV-R6.1-3 — Regime fixture shape: a 55-day candidate view under the 3-day synthetic allowlist

`build_pipeline_fixture(regime_study=…)` replaces the 24-candidate mini view
with ML fixture 2 (600 known-cluster candidates over 55 synthetic business
days from 2026-01-05) so the frozen 40/5/5/2 schedule yields real folds and
real fits inside the pipeline E2E, while the charter allowlist stays the
three synthetic days (verification scope + synthetic marker; S00 never
compares the candidate view's days to the allowlist on the synthetic
branch). The panel shapes write a synthetic VERIFIED replay-chart artifact
under the tmp root and serve it through `load_verified_replay_chart_artifact`
as the `context_bar_source`. Recorded as a fixture-shape deviation; the real
five-day mini-run remains the owner-blocked real slice (plan §12 blocker 1).

Amended after the adversarial round (reviews S1 / F4): the fixture seam no
longer ignores the requested chart id — S04 writes a real-shaped synthetic
chart per child and `_synthetic_replay_chart_seam` loads exactly the
requested id through `load_verified_replay_chart_artifact(expected_pair=…)`
(`foreign_context_bar_source` reproduces the reviewer's probe and is refused
by S05); the synthetic panel bars plant regimes PER TRADING DAY
(`regime_structured_bars_1m` / `REGIME_DAY_SCALES`) because the previous
intraday-planted bars failed `minimum_cluster_occupancy` (1.6 %), which made
FEATURE_ELIGIBLE unlawful on the panel grain and a supervised panel E2E
impossible — occupancy is now 61 / 9 / 30 % and every gate passes. Every
synthetic panel / replay-chart / panel-protocol id re-mints (synthetic only;
the smoke's panel run and the draft proposals re-mint with it).

## DEV-R6.1-4 — Single-threaded regime kernel for bit-reproducible artifacts

Multithreaded BLAS/OpenMP reductions made centroid distances differ at the
ULP level between otherwise identical runs, forking the persisted assignment
tables (and every artifact hashing them) without any scientific change.
`run_regime_protocol` executes under `threadpool_limits(limits=1)`
(threadpoolctl, a scikit-learn dependency). This is an execution-environment
control, not a protocol field: no protocol/fit identity moves; the double-run
pipeline test proves byte-identical stage results. DECISIONS_TAKEN #79.

## DEV-R6.1-5 — The session of a completed bar is the session in force at its final instant

The plan's shorthand `classify_session(bar_close, scheme).session` would put
a bar closing exactly at 08:00 ET into the NY session although it contains
no NY-session trading. The materializer classifies at `close − 1 µs`
(`SESSION_CLASSIFICATION_INSTANT_POLICY = session_of_bar_final_instant_v1`) and
measures `cbp_bar_position_in_session` from the close instant against that
session's [open, close): the first NY bar closes 08:05, position 300/21600;
the last closes 14:00, position 1.0; an early close leaves the value < 1
(test-pinned, DST-aware). DECISIONS_TAKEN #80.

## DEV-R6.1-6 — Refusal exception names carry the `Error` suffix

The plan names `RegimeStatusRefusal` / `OwnerDecisionRefusal`; the repo's
ruff rule N818 requires the `Error` suffix, so the classes are
`RegimeStatusRefusalError` / `OwnerDecisionRefusalError` /
`RegimeActivationRefusalError` / `RegimeSupervisedStudyRefusalError`. The
refusal TEXTS are the plan's ("nothing here promotes", …).

## DEV-R6.1-7 — `hard_id_encoding` semantics and the model-facing local id

The fit-local id column `ctx_regime_<p12>_local_id` is always materialized
in the fold-feature artifact; it is a MODEL feature only under
`hard_id_encoding = "fit_local_categorical_v1"` (the stamped default
`regime_feature_hard_id_encoding_default = none` keeps it out of the model
features; distances/margin are always model features). ONE vocabulary
(`regime_fold_features.py`: `HARD_ID_ENCODING_NONE` / `HARD_ID_ENCODING_CATEGORICAL`
/ `HARD_ID_ENCODINGS` / `HardIdEncoding`) is shared by `RegimeStudyRequest`,
the fold-feature artifact and S09b (the first draft's `categorical` alias
and the `!= "none"` remap are gone — review F11; the supervised zero-fitting
runs exercise `fit_local_categorical_v1` end to end). The regime block's
resolution is minted per protocol/activation (`ctx_regime_<p12>_*` names),
so `B7_CORE_REGIME` resolves only under an activation. Both study arms share
one static `view_id` (the base bundle frame); D13's `comparison_row_id` is
what pairs them. DECISIONS_TAKEN #81, #100.

## DEV-R6.1-8 — Reused children for stratified reports: re-derived tables are adopted ONLY by verified reproduction

A child whose immutable replay already exists is REUSED without executed-trade
tables (R5 design). When a regime study requests stratified reporting, S02
re-derives such a child through the wired runner; the re-derived tables are
adopted as this run's evidence ONLY when a persisted costed evaluation for
THIS cost policy exists and the re-derived metrics reproduce it byte-for-byte
(the row stays `reused`, `replay_invocations = 1`, "verified reuse by
reproduction … reproduced its persisted costed evaluation"). With no such
evaluation (a different cost policy, or a replay whose evaluation was never
published) the tables are NOT adopted: the row stays `reused` with
"reproduction unverifiable; the re-derived tables are not this run's
evidence", S03 evaluates no gates for it and publishes no evaluation from
them, and the child is out of the stratified reports (`children_skipped`).
The first draft adopted the tables unconditionally on that branch while
claiming reproduction (review S4) — corrected;
`test_s02_reused_children_are_adopted_only_by_verified_reproduction`. A prior
attempt's S14 report ids are otherwise recovered from its stage sidecar
(verified reload). The two-pass workflow (descriptive run → owner evidence →
frozen model-bearing run over the SAME charter and cost policy) therefore
works over one store without a store scan. DECISIONS_TAKEN #82 (amended), #98.

## DEV-R6.1-9 — S12/S13 record the exact persisted account-simulation ids

`make_prop_simulator(on_simulation_persisted=…)` reports every persisted
`account_simulation_id` (firm label, mode) to the pipeline, which writes an
attempt-invariant `account_simulations.json` stage sidecar and recovers it
from the prior attempt on reuse — the stratified-prop reports therefore read
exact ids, never a store listing. DECISIONS_TAKEN #83.

## DEV-R6.1-10 — Modeled classes are S09c deliverables; S14 records them under `delivered_by`

`feature_only` / `cohort_model` produce the persisted `RegimeControlledStudy`
/ `RegimeCohortModelStudy` artifacts at S09c (every fit of the run happens
there). S14 does not build duplicate report envelopes for them: the report
stage passes the exact S09c study ids (this attempt's results, else the run's
OWN verified S09 stage record — never a listing) as
`StratificationInputs.delivered_by`; the service verifies each id by exact
reload and records it in `StratificationOutcome.delivered_by` (persisted in
the S14 record and the report index). A modeled class that was REQUESTED but
not delivered is a typed refusal ("… this run recorded none for it …;
nothing here promotes"); a class not requested has no entry. The first
draft recorded the delivered classes as refusals (review F6) — corrected.
The plan's class table is honored (paired deltas at S09c/S10; status-gated
activation); a report-envelope form for the modeled classes remains a
hardening candidate. DECISIONS_TAKEN #86 (amended), #92.

## DEV-R6.1-11 — `panel_source_bar_incomplete` extends the typed-null vocabulary (review F12)

`PANEL_ASSIGNMENT_MISSING_REASONS` carries `panel_source_bar_incomplete`
(the panel→candidate PIT rule maps a panel row invalidated by
`source_bar_incomplete` — final ruling 3 — onto it) beside the plan's
`panel_warmup` / `no_completed_panel_bar` / `panel_gap` / `panel_stale` /
`coverage_gap`. No UI surface enumerates the reasons (`study_status.py`
registers no per-reason empty state; the assignment view renders whatever
`missing_reason` values the artifact carries), so no UI change was needed.

## DEV-R6.1-12 — §9.1 test-name mapping (review F5)

The plan's `test_decision_25_algorithm_snapshot_is_authorized` is delivered
as the three cases `algorithm_key` / `algorithm_parameters_hash` /
`algorithm_parameters` of the parametrized
`test_decision_25_28_29_30_values_are_verified_against_registry_protocol_assessment`
(`tests/agents/ifvg_search/test_owner_decisions.py`). The two supervised-study
tests whose names exceeded their assertions were renamed/split
(`…_across_distinct_bundles`; the genuinely distinct-view proof is
`test_challenger_arm_with_a_distinct_view_id_pairs_only_on_comparison_row_id`),
and the controlled regime study is unit-tested over
`DEFAULT_BUNDLE_LADDER_PROTOCOLS` (prevalence + logistic + the CatBoost
bundle rung); the cohort fixture keeps the two linear rungs for runtime.

## DEV-R6.1-13 — `threadpoolctl` is imported lazily and stays undeclared (review S9)

`pyproject.toml` is untouched (no new declared dependency in this release).
`run_regime_protocol` imports `threadpoolctl` lazily and raises
`RuntimeError(THREADPOOLCTL_MISSING_MESSAGE)` (names the dependency and the
reason — bit-reproducible assignment tables) when it is missing; the module
no longer imports it at import time. Declaring `threadpoolctl>=3.1`
explicitly is an owner call for a later release.

## DEV-R6.1-14 — The D15 writer streams literally; the store gained a sidecar-producer protocol (reviews S12 / F16)

`search/store.py`: `ProducedSidecar` / `SidecarProducer` /
`write_produced_sidecar`; `save_envelope_immutable(sidecar_producer=)` runs
the producer INTO the temporary publication directory, re-hashes every
produced file by streaming, refuses any bookkeeping disagreement, stray,
ghost, reserved or non-whitelisted name (nothing published, temp directory
discarded), and merges produced sidecars into the manifest sorted with
`extra_files` (byte-identical manifests to the in-memory path);
`save_or_reuse_envelope` on reuse runs the producer into a scratch directory
and requires every produced file to reproduce the stored manifest hash.
`propsim/event_detail.py` builds one path block at a time column-wise, writes
each ZSTD partition straight to the directory, hashes by streaming, and
releases the block; the row preflight is unchanged and the cumulative byte
budget refuses at the offending block (later blocks never built). The only
whole-artifact state is a 32-byte-per-row `event_id` uniqueness index.
Simulation ids and manifests are unchanged (byte-identical partitions).
DECISIONS_TAKEN #88 (amended), #93.

## DEV-R6.1-15 — Owner-decision governance residuals after the adversarial round (reviews S2 / S3 / S5 / S11 / F10)

- `run_scope` is NOT recorded inside `RegimePromotionDecision` (identity-bearing;
  would have moved S10's decision ids mid-round). Enforcement is structural
  instead: `assert_run_scope_lawful_for_root` mirrors the charter's P0-4
  namespace rule (a `search` segment that is not `search_test` is the
  research namespace), `persist_regime_promotion(run_scope="synthetic_fixture")`
  refuses there, and synthetic-provenance owner artifacts are refused at
  persist AND at load (`load_owner_decision`) in that namespace — a
  synthetic-backed FEATURE_ELIGIBLE cannot exist in a research root because
  its owner artifact is unloadable there. The CLI `chain` annotates every
  ratified row with `owner_evidence: verified (<provenance>) | refused: <reason>`.
- `decided_at >= approved_at` is enforced transitively (the artifact validator
  enforces `effective_from >= approved_at`; effectivity refuses
  `decided_at < effective_from` as "not yet effective"), not by a separate
  check; the store enforces the monotone chain (`decided_at >= previous`).
- Deleting `SUPERSESSIONS.jsonl` AND `SUPERSESSIONS.head` together restores
  the pre-supersession state: without a store listing no loader can know a
  supersession existed — the DEV-R6-8 trust boundary (integrity, not
  authenticity; signing is the hardening candidate). Every single-file
  tamper (edited / deleted / reordered line, missing head, dangling line)
  fails closed.
- `MODEL_FEATURE` stays representable at the contract (ladder / role tests)
  but is unpersistable through every V1 path
  (`MODEL_FEATURE_PROMOTION_REFUSAL`, one constant shared by the store and
  the CLI).
- `--decided-at` is removed from the CLI; `decided_at` derives from verified
  artifacts (the previous decision's `decided_at`, or the owner artifact's
  `effective_from` for ratified statuses, whichever is later;
  `decided_at_source` reported), so `promote --to stratification_ready`
  reproduces S10's exact decision id (a reuse by design).
  DECISIONS_TAKEN #85 (amended), #90, #91.

## DEV-R6.1-16 — The event-regime summary budget fails S14 typed; cross-partition uniqueness by construction (review F3)

`account_event_regime_summary_budget_v1` (5,000,000 rows / 268,435,456 bytes)
raises the typed `EventRegimeSummaryBudgetError` before publication and the
service lets it propagate — a capacity anomaly fails S14 loudly rather than
becoming a per-class refusal. Cross-partition `event_id` uniqueness is
guaranteed by per-partition uniqueness plus `path_instance_id` disjointness
across partitions (memory-bounded), not by a global id set. Every
`RegimeStratifiedReport` id moves (the `stratified_prop` body gains
`summary_budget` / `summary_rows`; the envelope gains four summary extras —
final ruling 7: extras, never identity inputs). DECISIONS_TAKEN #92.

## DEV-R6.1-17 — S08's authorized trading days are the candidate view's observed days (review F7)

`authorized_trading_days` = the charter allowlist ∩ the candidate view's
observed trading days: `_stage_s08_folds` derives the days from the
candidate view whenever a regime study is planned and builds the labeled
folds from those SAME days; `s08_regime_folds` derives the schedule from the
same source and records `trading_days_source = candidate_view_observed_days`
and `days_without_labels` in the S08 sidecar. A label builder that drops a
day therefore yields a typed fact (a fold with fewer labeled rows), not a
window-mismatch stage failure
(`test_s08_derives_the_schedule_once_from_the_candidate_view_days`).
DECISIONS_TAKEN #97.

## DEV-R6.1-18 — S05 chart binding / selection; S14's trade tables are S02's this-run tables (reviews S1, F13, F14)

- S04 produces one replay chart per child. S05 selects the panel's chart by
  the declared policy `panel_chart_lowest_core_replay_id_v1` (the chart of
  the lowest core replay id; no chart / more than one chart on that child
  refuses), asks the seam for exactly that id, and refuses (typed; S05 fails
  closed; nothing persisted) when the returned artifact's id or pair differs
  from the request; the materialized panel's `replay_chart_artifact_id` is
  re-checked and the S05 sidecar records the LOADED chart id, the selection
  policy, `panel_source_core_replay_id` and `replay_chart_source_pair_sha256`
  (`test_s05_refuses_a_seam_that_serves_another_verified_chart`).
- S14's panel PIT event assigner verified-loads everything by exact id (the
  OOS assignment → its protocol → the panel envelope + frame → every fit's
  assignment sidecar; no in-memory run object). The executed-trade tables are
  the ONE input that is not a persisted artifact — a core replay persists
  identity only (`CoreStrategyReplayPayload` carries no tables or stream
  hash) — so `build_reports` consumes S02's this-run tables (fresh, or
  re-derived and adopted only by verified reproduction, DEV-R6.1-8). A
  persisted trade-table artifact remains a hardening candidate.
  DECISIONS_TAKEN #96, #99.

## Scoping notes (not deviations)

- The R6 panel tests migrated to `BP0_CONTEXT_BAR_PANEL` + `cbp_*` inputs
  (grain/bundle-key coherence refuses a candidate bundle under the panel
  grain); the R6 evidence folder is never edited.
- The owner-decision proposals under `DRAFT_OWNER_DECISION_PROPOSALS/` are
  generated from SYNTHETIC runs; the panel protocol ids bind synthetic panel
  artifact ids — the decision VALUES are what the owner ratifies; the real
  ids are minted by the real run (the candidate protocol id is already the
  real one: it carries no artifact id).
- The three owner-decision drafts: `candidate_stage_row.md` and
  `context_bar_panel_300s.md` were REGENERATED after the fix round by
  `scripts/ifvg_regime_promotion.py propose` over the browser-smoke store
  (commit `6c0b60a`; synthetic ids of the smoke runs); the browser-smoke
  harness runs the 300 s panel study only, so `context_bar_panel_900s.md`
  keeps its pre-fix-round synthetic ids with an explicit header note (its
  VALUES equal the 300 s draft except `panel_interval_seconds = 900`; the
  real 900 s ids are minted by the real run).
- The browser smoke serves the candidate, panel and candidate-supervised
  runs (`r61_smoke_app.py`); the supervised PANEL run is proven by the
  pipeline E2E (`test_model_bearing_panel_run_executes_s09b_s09c_pit_on_the_frozen_authority`)
  and the parametrized zero-fitting test, not by a served view.
- Cross-pipeline child reuse without tables and without a prior S14 record
  of the same pipeline leaves the affected children out of the descriptive
  reports (recorded in `children_skipped`) — the reproduction path above
  covers the two-pass workflow; a runner that cannot reproduce fails the
  child closed.
