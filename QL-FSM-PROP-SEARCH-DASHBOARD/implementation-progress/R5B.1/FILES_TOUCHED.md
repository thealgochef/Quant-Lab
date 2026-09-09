# R5B.1 — Files Touched

Reconciled against `git status --short` before the release commit
(HEAD = `179a2c9`, R6), including the adversarial-fix round. The R5B.1
commit is PATH-SCOPED to exactly the files below: the worktree
concurrently carries the R6.1 regime-lane files (`ml/regime_*.py`,
`fold_schedules.py`, `ml/fold_set_artifact.py`, `features/arrow_tables.py`,
`features/context_bar_panel_*.py`, `search/owner_decisions.py`,
`bundle_feature_view.py`, the fold-hash delegation in
`supervised_ladder.py` / `controlled_feature_study.py`, their tests and
fixtures), which are NOT part of this commit (review S8).

## New source modules (2)

| File | Content |
|---|---|
| `src/.../ifvg/features/mbp1_coverage_evidence.py` | coverage policy v2 (`mbp1_source_coverage_declared_evidence_v2`): `Mbp1CoverageEvidenceKind`, `Mbp1EvidenceScopeLevel`, `Mbp1EvidenceScope` (dataset / publisher / channel / instrument partition, physical partition key, UTC date, verified expected span), `Mbp1DeclaredGapInterval` (start/end evidence kinds, recovery kind, `contributing_evidence_kinds`), `Mbp1PartitionGapManifest{Payload,Envelope}`, `Mbp1CompletenessCompilationReport{Payload,Envelope}` (source inventory + owner review; `positive_completeness_authorized`), `compile_mbp1_partition_gap_manifest` (the ONLY producer of positive completeness; `source_document_sha256 ∈ evidence_refs` and never the report id), `Mbp1RecoveryBoundary`, `Mbp1DatasetConditionRecord` → the five `vendor_*` states, `Mbp1PartitionEvidence` (manifest + report + boundaries + conditions; scope-checked; completeness requires the report), `Mbp1SequenceJumpDiagnostics` / `Mbp1TsRecvGapDiagnostics` (diagnostics only), the bad-book interval builder (trusted start; documented recovery or fail closed to the partition end; channel scope only behind a verified map), `merge_intervals` / clipping / `physical_expected_span`, `authorized_session_span_ns`, the `MBP1_PROPOSED_DEFAULTS`-backed policy constants, the evidence stores (`mbp1_gap_manifests`, `mbp1_completeness_reports`) with `load_verified_partition_evidence`, `assert_evidence_date_representable`; identity pairs registered |
| `src/.../ifvg/features/mbp1_coverage_diagnostic.py` | the bounded real-data diagnostic: `Mbp1PartitionDiagnosticRow` (row/flag counts, sequence-jump and `ts_recv`-gap distributions, clipped-row counts, evidence-present facts, tri-state `open_uncertainty_to_partition_end`, completeness status), `Mbp1CoverageDiagnosticReport{Payload,Envelope}` (structural `completeness_inferred_from_sequence_continuity=False`; `partition_evidence_manifest_ids`), `assert_verification_namespace` (`search_test/v1` as the LAST two path components), `assert_diagnostic_authorized` (the R1 real-slice gate: `VerificationReplayPolicy`, verified `VerificationRunEnvelope` + `VerificationAuthorizationRef`, allowlist hash equality, coverage-matrix verified load, the one canonical program allowlist), `load_partition_evidence_manifest` (store-verified manifests + typed records only), `build_/persist_coverage_diagnostic_report`; store `mbp1_coverage_diagnostics`; identity pair registered |

## New scripts (1)

`scripts/ifvg_mbp1_coverage_diagnostic.py` — the diagnostic CLI: refuses
before any source path is constructed without the owner's verified
authorization (`PermissionError` → `SystemExit`), `--evidence-json` takes
store IDS + typed records (never a path), `--synthetic-source-artifact-id`
runs the report shape over an already-persisted synthetic artifact only;
importing launches nothing.

## Modified source (9)

| File | Change |
|---|---|
| `src/.../ifvg/features/feature_blocks.py` | the R5B.1 re-resolution as a SECOND versioned event: pure `with_reresolved_block` (bumps `block_version` by exactly one; refuses non-AVAILABLE blocks, wrong successor versions, and an unchanged resolved id), `mbp1_coverage_v2_resolution_payload` (v3; formula/materializer v2), `MBP1_COVERAGE_V2_ENVELOPE`, the R5B state exported as `PRE_R5B1_FEATURE_BLOCK_REGISTRY` / `PRE_R5B1_RESOLUTION_REGISTRY`; the R5B activation payload keeps the HISTORICAL v1 versions (`MBP1_FORMULA_VERSION_V1` / `MBP1_MATERIALIZER_VERSION_V1`); import-time invariants for both events (v2 available at R5B, v3 available now, both registry-hash changes) |
| `src/.../ifvg/features/mbp1_arrow_schemas.py` | the normalized event schema RETAINS `publisher_id` + `flags` (DBN bit 4 = `F_MAYBE_BAD_BOOK`) — a schema-hash change that re-mints every source artifact identity |
| `src/.../ifvg/features/mbp1_coverage.py` | the coverage report is evidence-based: per-day `partition_count`, `completeness_status`, `dataset_condition_status`, `declared_gap_count`, `open_uncertainty_to_partition_end`, the DIAGNOSTIC `sequence_positive_jump_count` (replacing `sequence_gap_count`), `coverage_policy_id` (Literal v2); rows come from `day_coverage_views` |
| `src/.../ifvg/features/mbp1_feature_materializer.py` | the typed reason order `no_mbp1_partition` → `coverage_evidence_unavailable` (`completeness_unknown`) → `coverage_below_threshold` → `stage_outside_coverage` → `same_timestamp_order_unavailable` → `coverage_evidence_unavailable` (uncovered session sub-span) → `declared_source_gap` → `instrument_roll_boundary` → `minimum_event_count_not_met`; windows never widened or imputed; `_validate_window_specs` refuses a superseded (v1) resolution and the artifact stamps versions FROM the block (review F8) |
| `src/.../ifvg/features/mbp1_source_artifact.py` | `Mbp1PartitionCoverage` gains the policy-v2 fields (evidence scope / provenance / declared intervals / completeness + condition status / diagnostics / `flag_counts` / content refs / `physical_expected_span_ns` / `union_gap_ns` / clip counters); `Mbp1DayCoverageView` + `day_coverage_views` (duration-weighted multi-partition days; `uncovered_session_intervals`); `_partition_slice` half-open, `_clipped_spans` disjointness refusal (F3), the stray-event refusal (F2), the positive-claim content binding (F1/S3), `assert_evidence_provenance_permitted` (synthetic provenance only under the synthetic marker), `_sequence_gap_intervals` → sequence-jump diagnostics; `read_mbp1_partition_frame` clips to `[18:00 ET D-1, 17:00 ET D)` and `DEVELOPMENT_CUTOFF_UTC` BEFORE normalization/hashing (S1) with the raw file sha256 riding the partition row |
| `src/.../ifvg/features/mbp1_source_contract.py` | `gap_semantics` Literal `mbp1_source_coverage_declared_evidence_v2`, `sequence_jump_semantics = sequence_jump_diagnostic_only_v2`, `coverage_policy` gains `completeness_evidence_required=True` + ORDERED `evidence_kinds_accepted` / `recovery_boundary_kinds_accepted`; `MBP1_FORMULA_VERSION` / `MBP1_MATERIALIZER_VERSION` move to v2 with the v1 names kept; the stamped `MBP1_PROPOSED_DEFAULTS` table (seven R-6-family values, `proposed_protocol_default`) |
| `src/.../ifvg/search/identities.py` | audit-enumeration import list gains `mbp1_coverage_evidence` and `mbp1_coverage_diagnostic` |
| `src/.../ifvg/search/pipeline.py` | S05 `_ensure_mbp1_evidence` calls `assert_evidence_provenance_permitted(..., synthetic_scope=context.synthetic)` before the artifact is trusted |
| `src/.../ifvg/search/store.py` | three new store names (`mbp1_gap_manifests`, `mbp1_completeness_reports`, `mbp1_coverage_diagnostics`) |

## Modified scripts (1)

`scripts/ifvg_mbp1_panels.py` — the coverage view renders the
evidence-based facts (completeness status, declared intervals, open
uncertainty, dataset condition, "Sequence jumps (diagnostic)"), and the
new `_render_coverage_policy_stamps` table surfaces `MBP1_PROPOSED_DEFAULTS`
as `proposed_protocol_default` rows (review F10). Still read-only
(source-scanned: no button/form/toggle/select).

## New tests (1) and modified tests / fixtures (7)

- `tests/agents/ifvg_search/test_mbp1_coverage_evidence.py` (18): the
  owner's six proofs, the three §9.1 MBP-1 rows, the final-closure rows
  (multi-UTC physical denominators, channel/publisher scope), the
  positive-completeness compiler, scope mismatch / empty span / head-tail,
  evidence store round trips, protected/sealed unrepresentability, the
  real-read clipping (synthetic parquet under tmp), synthetic-provenance
  refusal outside the synthetic scope, the diagnostic's gate matrix and
  fail-before-path, the evidence-manifest seam, the CLI refusal +
  synthetic shape.
- `tests/agents/ifvg_search/mbp1_fixture.py` — synthetic scope /
  interval / recovery boundary / completeness report / partition-evidence
  builders (`default_partition_evidence`, `canonical_content_sha256`).
- `tests/agents/ifvg_search/pipeline_fixture.py` — the pipeline's
  synthetic MBP-1 source carries partition-scope evidence under synthetic
  provenance.
- `tests/agents/ifvg_search/test_feature_blocks.py` — the v2
  re-resolution as a second versioned event (replay equality, hash
  change, B2/B3 ids move, B0/B1/B4 stable).
- `tests/agents/ifvg_search/test_identities.py` — the three new
  identity pairs enumerated canonically.
- `tests/agents/ifvg_search/test_mbp1_materializer.py` — giant sequence
  jump never lowers coverage; declared gap → `declared_source_gap` on
  exactly the intersecting windows; a day without partition evidence →
  `coverage_evidence_unavailable`; superseded resolution refused.
- `tests/agents/ifvg_search/test_mbp1_schemas_and_source.py` — sequence
  jumps and resets are diagnostics, never gaps.
- `tests/agents/test_ifvg_pipeline_tab.py` — the stamped coverage-policy
  defaults render; the coverage view shows evidence facts and never
  "Sequence gaps".

## Docs

- `docs/DECISIONS.md` — D-048 (MBP-1 coverage policy v2) + the
  reservation note (D-047 reserved for R6.1). Committed normally.
- `ARCHITECTURE.md` / `docs/README.md` / `docs/pipeline_state.yaml` —
  R5B.1 lane transforms via `R5B.1/stage_shared_docs.py` (HEAD + the
  R5B.1 transforms only; the post-commit `--apply-worktree` replay leaves
  the user's pre-existing hunks as the sole surviving worktree diff).
- `docs/ML_TRAINING_WORKBENCH.md` — untouched, uncommitted (user-owned).

## Implementation-progress files (this folder)

`PRE_R5B_1_BASELINE.md`, `FILES_TOUCHED.md` (this), `DEVIATIONS.md`,
`ACCESS_SAFETY_EVIDENCE.md`, `ADVERSARIAL_REVIEW.md`,
`ADVERSARIAL_REVIEW_RESOLUTION.md`, `TEST_RESULTS.md`, `GATE_SUMMARY.md`,
`stage_shared_docs.py`, raw outputs `_final_pytest.txt` /
`_final_pytest_keys_cleared.txt` / `_ruff_and_diffcheck.txt` /
`_surviving_shared_doc_diff.patch`, the source-review evidence
`R5B.1.patch` + `R5B.1.patch.sha256`; `../DECISIONS_TAKEN.md` entries
67–75 (+ the in-place "AMENDED by R5B.1" pointers on #48/#49 and the
"SUPERSEDED by R5B.1" note on `../R5B/DEVIATIONS.md` DEV-R5B-6).

## Never modified

All M0–M3 lane modules, all propsim modules, Strategy-Core, Trade-Lab,
every existing immutable artifact/catalog, the plan package, the R6 lane
modules (`ml/regime_*.py` at HEAD). No new package dependency.
