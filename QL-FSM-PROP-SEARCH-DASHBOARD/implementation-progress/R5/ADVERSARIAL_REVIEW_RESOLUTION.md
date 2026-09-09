# R5 — Adversarial review resolutions

Every finding dispositioned; fixes verified by re-run (FINAL suite counts
in `GATE_SUMMARY.md`). Format: finding → disposition → evidence.

## Contract lens (Reviewer 1)

- **M-1 (label identity collision / order fork) → FIXED.** The id now
  hashes SORTED `(candidate_id, target)` PAIRS — candidate↔target bound,
  order-invariant (`pipeline.py::_stage_s07_labels`). The reviewer's
  reproduction is impossible by construction (pairs differ ⇒ hash
  differs; reorder ⇒ same sorted pairs ⇒ same hash).
- **M-2 (comparison consumption dead code) → FIXED.** The Monitor now
  consumes the persisted S14 `ComparisonResultEnvelope`s through
  `load_comparison_results_for_search` (exact-ID loads): per-kind
  `match_basis` + Jaccard rows and the lineage-uniqueness evidence links
  render in a dedicated panel. AppTest-witnessed
  (`test_monitor_consumes_the_persisted_comparison_results` — asserts
  the exact-basis rows AND the truthfully `not_comparable` trade kind).
- **M-3 (§30.4 monitor fields) → FIXED + recorded.** The Monitor renders
  elapsed (attempt start/end), a remaining-stages note, the configured
  worker count with the truthful sequential-execution caption, the
  running child (ordinal + short id) when one exists, and the warnings
  list ("none recorded" when empty). AppTest-witnessed
  (`test_monitor_renders_the_section304_operational_fields`). The one
  structurally unavailable field — a live "current fold" — is recorded
  as DEV-R5-11 (fold execution is one in-memory call; a live indicator
  would be fabricated).
- **M-4 (ruff record) → FIXED before this resolution round completed.**
  The two F401s were introduced and removed in the same working session
  (a post-run edit trimmed the last uses; `ruff --fix` applied); the
  FINAL gate re-runs both commands and `GATE_SUMMARY.md` records the
  re-run results as the release evidence. The TEST_RESULTS FINAL row is
  the re-run, per its own rule.
- **m-5 → FIXED.** `_persist_policy_set_envelopes` runs unconditionally
  at S12 start; the stage's outputs always name persisted envelopes.
- **m-6 → FIXED.** The gate cache key embeds the pipeline id
  (`gate_results_{pipeline_id}`); another run's gates can never enable
  this run's activation control (the server-side refusal was already
  fail-closed).
- **m-7 → ACCEPTED (no change).** The guarded assignment is strictly
  safer (never overwrites an earlier reason with None); the divergent
  input is not constructible; explanation strings byte-identical
  (reviewer-verified).
- **m-8 → NO CHANGE (recorded).** `FailureReason.REPLAY` for prop
  runtime failures matches `run_search`'s R3 seam exactly — cross-lane
  consistency wins; the "prop simulation failed:" prefix carries the
  truth. Scoping note added to DEVIATIONS.
- **m-9 → FIXED.** `ladder_id` now binds the label content hash (sorted
  candidate→target pairs) and the fold-set hash (per-fold train/test id
  lists) alongside view/tier/calibration/rung hashes.
- **m-10 → FIXED.** S00's wiring closure requires `candidate_view_source`
  whenever S07 is planned (not only S05).
- **m-11 → FIXED.** Stage-result publication is inside the sanitized
  per-stage failure path: a store refusal (e.g. cross-attempt sidecar
  divergence) marks the stage FAILED with sanitized evidence and halts
  cleanly instead of crashing the runner mid-state.
- **m-12 → FIXED.** The spec validator refuses `feature_bundle_ids` when
  no feature stage is planned — an inert bundle claim can never ride the
  semantic identity.
- **m-13 → ACCEPTED (recorded).** The enumeration defense is structural
  (ids-only entry; no multi-value surface exists) plus the direct-call
  tests; the single-value call site documents intent. Scoping note in
  DEVIATIONS.
- **m-14 → FIXED (test added).** Direct leak probe
  (`test_fold_local_fit_statistics_are_blind_to_test_rows`): perturbing
  a TEST row's numeric feature leaves every fitted statistic (imputer
  medians, indicator features, scaler mean/scale, coefficients)
  byte-identical.
- **m-15 → RECORDED.** DEV-R5-12 documents the §30.1 wizard/pipeline
  split and the field-persist rule that keeps it coherent.
- **m-16 → FIXED.** S00's real branch now compares
  `spec.date_allowlist`/`allowlist_hash` against the run payload
  (fail-before-path) in addition to the existing bindings, and the
  expected seed accepts the runner-supplied
  `wiring.expected_seed_snapshot_id` (safety F3's fix) before falling
  back to the envelope value.
- **m-17 → FIXED.** The Preview runtime row renders per-phase NUMBERS
  (children × days × measured per-day replay cost; prop sim count;
  ML in-memory) as operational annotations.
- **Cosmetics → FIXED.** DEV-R5-6 count corrected (47); the prevalence
  rung gained the in-lane duplicate-row guard (refuted-suspicion-7
  footnote).

## Safety lens (Reviewer 2)

- **F1 → FIXED.** `zero_forbidden_counters` is DERIVED: the per-child
  containment records any tripped access assertion
  (`forbidden_access_detected`) and the gate reads it — test-witnessed
  (`test_tripped_access_assertion_fails_the_derived_counter_gate`: a
  tripping drive fails the child, the gate, the report, and publication).
  The persisted report can never contradict the audit it names.
- **F2 → FIXED.** Synthetic-branch stamps carry `real_date_count=0`,
  the empty-allowlist hash, `synthetic_fixture_ids`, and a separate
  `synthetic_date_count` — E2E-witnessed
  (`test_pipeline_result_prepared_not_published_with_stamps`).
- **F3 → FIXED (with F/m-16).** `PipelineWiring.expected_seed_snapshot_id`
  lets the runner supply the seed it will actually load; S00 prefers it.
  The authorization ref remains sourced from the persisted envelope —
  that envelope IS the design's owner-action trust anchor (reviewer's
  own refuted-suspicion 1), corroborated by the independent charter
  cross-checks.
- **F4 → DEFERRED (recorded).** DEV-R5-10 stands; the search-shim
  factory-signature alignment lands with the R5B shim work.
- **F5 → FIXED.** Sidecar names must fullmatch
  `[A-Za-z0-9][A-Za-z0-9._-]*` (no separators, no leading dot, no
  drive-relative or ADS colons).
- **F6 → FIXED.** The S12 note reads "resolved AccountPolicySetEnvelopes
  persisted…" — no "real"-firm ambiguity.
- **F7 → FIXED.** `full_pipeline_not_run` derives from the run scope.
- **F8 → NO CHANGE (refuted in practice).** The badge, the ack, and the
  assembled charter's access policy all derive from the same draft
  field (`validation.run_scope`) — no divergence is constructible
  (corroborated by contract reviewer's refuted-suspicion 11).
- **F9 → FIXED.** `sanitize_failure_message` drops any drive-letter
  rooted path line (`[A-Za-z]:[\\/]`), not only `C:`.
- **F10 → PARTIALLY FIXED.** (b) the `ComparisonResultEnvelope`
  docstring now states both reference forms (study-cell envelope vs the
  search-lane derivation id); (d) the launch docstring names both
  handler-confined spawn sites. (a)/(c) recorded as accepted
  observations.

## Post-fix verification

All fix-affected suites re-run green (56 tests across pipeline run/
contracts/job-shim/tab/logistic lanes) plus the orchestrator/prop-seam
regression (22) proving the failure-sanitizer change altered no existing
explanation. The FINAL full-repo suite, ruff, and diff-check results are
recorded in `GATE_SUMMARY.md`.

## R5-FIX addendum (post-gate owner findings, 2026-08-21/22)

The owner's gate review rejected two of the dispositions above and the
smoke evidence they rode on; `R5_FIX_REPORT.md` carries the full
finding→change map. Superseded dispositions:

- **F3 / m-16 (expected seed) → SUPERSEDED by R5-FIX finding 7.** The
  original fix let the runner SUPPLY `expected_seed_snapshot_id` — still a
  caller-provided string, with a tautological fallback to the run
  envelope's own value. Now: the real verification branch REFUSES without
  `PipelineWiring.loaded_seed_snapshot_id_source`; the source performs the
  VERIFIED seed-snapshot store load and returns the loaded envelope's
  content-derived id, which is what `validate_verification_run` checks
  against the owner authorization (DECISIONS_TAKEN #44;
  `test_real_scope_seed_requires_the_loaded_artifact_source`,
  `test_executor_seed_source_returns_the_loaded_artifacts_id`).
- **F10(b) (comparison reference forms) → SUPERSEDED by R5-FIX finding 6.**
  The docstring documented two reference domains but enforced neither.
  Now: `ComparisonResult.subject` is a discriminated union
  (`StudyCellComparisonSubject` | `SearchDerivationComparisonSubject`,
  DECISIONS_TAKEN #45) — a bare 64-hex string is no longer a lawful
  subject
  (`test_comparison_result_subject_is_a_typed_discriminated_reference`).

Additionally brought forward by the same gate review: the production
runner registry's `tests.*` values (finding 3 → DECISIONS_TAKEN #43), the
ladder table's Arrow-unsafe mixed column (finding 1), the zero-row parity
copy (finding 5), the deprecated `use_container_width` API (finding 8),
the smoke harness's untied reuse marker (finding 4), and the stale
evidence documents (findings 2/9) — all resolved in the R5-FIX commit and
re-evidenced by the re-run browser smoke (zero tracebacks, zero
deprecation warnings; `browser-smoke/MANIFEST.json`).
