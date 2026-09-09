# R5B.1 — Access-Safety Evidence

Implementer's audit, corrected after the adversarial round; independently
corroborated by the safety-lens reviewer (`ADVERSARIAL_REVIEW.md`,
Reviewer 2 — protected/sealed zero-counter **AFFIRMED**: "no R5B.1 code
path, test, or review probe constructed, listed, stat'ed, opened, or read
any real source partition").

## Protected/sealed counters: ZERO

- **No real source path was ever constructed** during R5B.1 implementation,
  its tests, or the release-final full-suite runs. Every coverage-policy-v2
  suite runs on synthetic in-memory events (`tests/agents/ifvg_search/
  mbp1_fixture.py` — crafted events on the synthetic days `2026-01-13…15`)
  with store writes to pytest `tmp_path` roots only; the real-read clipping
  test (`test_real_read_clips_to_trading_day_session_and_development_cutoff`)
  reads a SYNTHETIC parquet it wrote under `tmp_path` through the real
  seam with a synthetic access policy.
- **Date literals in the R5B.1 set** (`grep` over the 14 new/modified code
  and test files): the only protected/sealed literals are REFUSAL PROBES in
  `test_mbp1_coverage_evidence.py` — `("2026-06-11", "2026-06-12",
  "2026-07-01", "2025-12-31")` in
  `test_protected_and_sealed_dates_are_unrepresentable_in_evidence` (each
  must be refused by `assert_evidence_date_representable` at scope and
  dataset-condition construction) and `allowlist=("2026-06-12",)` in
  `test_coverage_diagnostic_shape_and_fail_before_path` (the diagnostic
  report must refuse a sealed allowlist). Neither literal reaches a path,
  a policy, or a store; `2026-06-10` (the last exposed development day) is
  asserted representable. No source module carries a protected/sealed
  literal; `DEVELOPMENT_CUTOFF_UTC` is imported from `development_access`
  (`2026-06-10T21:00:00Z`), never restated.
- **Authorize-before-path is structural** in the one real-source seam:
  `read_mbp1_partition_frame` refuses `access_policy=None` BEFORE the path
  factory is invoked, reaches a path only through
  `access_policy.resolve_source_path`, validates the pinned mbp-1 schema on
  the parquet header before any row decodes, refuses deeper-book columns,
  reads through the pinned column projection, records every open in the
  policy's audit, and — NEW in R5B.1 (review S1) — clips the UTC-date file
  to the trading day's authorized session span `[18:00 ET D-1, 17:00 ET D)`
  and to `DEVELOPMENT_CUTOFF_UTC` BEFORE normalization or hashing, so the
  protected 18:00 ET tail of the last exposed day is structurally excluded
  from every artifact (`rows_outside_session_span` /
  `rows_after_development_cutoff` ride the partition row).
- **The bounded real-data diagnostic fails before any path** (reviews
  S2/S4/F9): `scripts/ifvg_mbp1_coverage_diagnostic.py` constructs its
  `IfvgCaptureConfig(data_dir=ROOT / "data/databento")` ONLY after
  `assert_diagnostic_authorized` has passed — a `VerificationReplayPolicy`
  over exactly the requested days, a persisted VERIFIED
  `VerificationRunEnvelope` whose `VerificationAuthorizationRef` binds the
  allowlist hash (`allowlist_hash == sha256(days) == approved_allowlist_hash`),
  the verification policy id, the `search_test/v1` namespace, a
  verified-loaded coverage-matrix artifact over exactly those days, and the
  ONE canonical program allowlist (`register_program_allowlist`); the store
  root's last two path components must be `("search_test", "v1")` (a
  substring is never a namespace). Every `PermissionError` surfaces as a
  `SystemExit`. The gate matrix is test-pinned (absent run / wrong policy /
  evil namespace / unbacked matrix / different allowlist / canonical-marker
  mismatch / forged hash / sealed allowlist), and the CLI refuses an
  unbacked persisted envelope. **No real run occurred** (DEV-R5B.1-4): no
  `VerificationRunEnvelope` / `VerificationAuthorizationRef` exists.
- **On disk** (implementer + reviewer): `find data -type f -newermt
  2026-08-25` → **0 files**; `data/ifvg_datasets/search` and `search_test`
  do not exist; no `mbp1_gap_manifests` / `mbp1_completeness_reports` /
  `mbp1_coverage_diagnostics` / `mbp1_source_artifacts` directory exists
  under `data/` (the only R5B-era `mbp1_source_artifacts` entry is the
  `%TEMP%\ifvg_r5b_smoke` scratch, outside governance).
- **Synthetic evidence is scope-bound**: `evidence_provenance ∈
  {owner_reviewed, synthetic_fixture, none}` rides every partition row;
  the pipeline's S05 (`assert_evidence_provenance_permitted`) and the real
  source builder (`build_mbp1_source_artifact_from_paths`) refuse
  `synthetic_fixture` provenance outside the synthetic marker — synthetic
  completeness can never certify a real partition.

## No full-development runs

No full-development replay, feature materialization, model fit,
configuration search, prop search, or operator pipeline ran. The largest
computations this release: the synthetic three-day pipeline E2E and the
synthetic controlled-study fixture (logistic fits only) — entirely
synthetic, tmp-rooted, seconds-to-minutes scale — plus the release-final
full-suite runs (twice, ~10 min each, `tmp_path` writes only).

## Positive completeness cannot be asserted (kickoff "a caller string is not evidence")

- The withdrawn "sequence jump > 1 = gap" rule is UNREPRESENTABLE
  (`Mbp1SourceContract.gap_semantics` Literal); raw sequence/`ts_recv`
  discontinuities are diagnostics only and can never widen OR narrow
  coverage.
- `evidenced_complete` exists only through `compile_mbp1_partition_gap_manifest`
  over a verified `Mbp1CompletenessCompilationReport` whose
  `verified_partition_refs` intersect the partition's content hashes; a
  provenance label without a manifest, a report whose
  `source_document_sha256` is its own id, or an `owner_review_decision_id`
  that is not a 64-hex decision hash is unrepresentable; the owner-review
  hash → persisted owner-decision artifact binding is the R6.1
  owner-evidence workstream (DEV-R5B.1-2).
- Dataset-condition records can only DOWNGRADE a scope (`degraded` /
  `pending` / `missing` → `completeness_unknown`); `F_MAYBE_BAD_BOOK`
  opens an interval that closes only at a documented recovery boundary
  and otherwise runs to the partition end (fail closed); a partition
  without partition-scope evidence types every window
  `coverage_evidence_unavailable`. Real MBP-1 research and R5B acceptance
  stay blocked until policy v2 passes on the real fixture (an owner action).

## Order-flow boundary (kickoff §9)

- MBP-1 remains the maximum representable depth: the deep-book guard runs
  at import over every schema/registry/feature namespace (the normalized
  schema gained `publisher_id` + `flags` only — top-of-book fields only),
  and the read seam still refuses deeper-book files and column-projects
  lawful ones (R5B review S1 — re-run green).
- `legacy_verified_replay_source` stays refused/unrepresentable; the R5B
  activation event is preserved byte-for-byte (`PRE_R5B1_*` exports; the
  activation payload keeps the historical v1 versions) and the v2
  re-resolution is a SECOND versioned event — no immutable artifact was
  mutated (DEV-R5B.1-3).

## UI safety

`scripts/ifvg_mbp1_panels.py` renders the evidence-based coverage facts and
the stamped `MBP1_PROPOSED_DEFAULTS` table (`proposed_protocol_default`,
owner decision R-6 family) — read-only: no button/form/toggle/select
control (the FUX source scans in `test_ifvg_study_scans.py` cover the
script); exact-ID loads only; errors sanitized; no launch path.

## Frozen lanes

None of the M0–M3 lane modules, no propsim module, no R6 regime-lane module,
and no Strategy-Core / Trade-Lab path changed in R5B.1 (Trade-Lab's 27
pre-existing user worktree entries untouched, newest mtime 2026-07-31;
Strategy-Core clean). The plan package is unmodified.
