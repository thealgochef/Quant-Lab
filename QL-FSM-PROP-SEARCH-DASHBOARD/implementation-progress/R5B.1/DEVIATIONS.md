# R5B.1 — Deviations and Scoping Notes

Reconciled after the adversarial round (`ADVERSARIAL_REVIEW_RESOLUTION.md`);
every entry states what the CODE does now.

## DEV-R5B.1-1 — The real read seam consumes the `day_utc_date` file only (no prev-partition composition)

`read_mbp1_partition_frame` reads ONE physical parquet per allowlist day
(`data/databento/NQ/<day>/mbp1.parquet`, authorized before its path exists)
and — new in this release (review S1) — clips its rows to the trading day's
authorized session span `[18:00 ET D-1, 17:00 ET D)` and to
`DEVELOPMENT_CUTOFF_UTC` BEFORE normalization or hashing. The previous UTC
file's 18:00 ET → midnight portion of the trading day is NOT composed:
the partition row records `relative_logical_partition_key =
"day_utc_date/mbp1"` and the file's own UTC date truthfully. Consequence:
with declared partition evidence, the un-read evening span is an
`uncovered_session_interval` whose windows type
`coverage_evidence_unavailable`; without evidence the day is
`completeness_unknown` anyway. Composition of `prev_utc_date/mbp1` is a
pre-condition of plan §12 blocker 3 (real MBP-1 research) and lands with
the owner-reviewed evidence workflow. DECISIONS_TAKEN #74.

## DEV-R5B.1-2 — `owner_review_decision_id` is a 64-hex decision hash, not yet a store-verified owner artifact

A positive completeness claim requires the compilation report's
`owner_review_decision_id` to be a 64-hex content hash (free text is
unrepresentable) and the report's `verified_partition_refs` to name the
partition content being certified; the real diagnostic loads evidence only
through `load_verified_partition_evidence` (store-verified manifests +
reports). Binding that hash to a persisted, verified owner-decision
artifact (with supersession/effectivity) is the R6.1 verified
owner-evidence workstream (`search/owner_decisions.py`); until it lands, the
MBP-1 owner-review evidence workflow has no persisted decision store —
recorded, not claimed. DECISIONS_TAKEN #73.

## DEV-R5B.1-3 — Pre-acceptance identity evolution (every MBP-1 identity re-mints)

The normalized event schema gained `publisher_id` + `flags`; the
`IFVG_ORDER_FLOW_MBP1_V1` block is re-resolved as a SECOND versioned event
(v3; formula/materializer v2; new registry hash; new B2/B3 bundle ids);
`Mbp1PartitionCoverage` gained the policy-v2 fields (evidence scope /
provenance / intervals / status / diagnostics / content refs / clip
counters); feature artifacts, coverage reports (`coverage_policy_id`), and
controlled-study ids move. No persisted real MBP-1 artifact exists inside
the repository (the only R5B-era `mbp1_source_artifacts` entry is the
`%TEMP%\ifvg_r5b_smoke` scratch, outside governance), so nothing immutable
was mutated (DEV-R5B-3 / DEV-R6-6 precedent). DECISIONS_TAKEN #71.

## DEV-R5B.1-4 — The bounded diagnostic's real run is an owner action and has not executed

`scripts/ifvg_mbp1_coverage_diagnostic.py` requires the persisted, verified
`VerificationRunEnvelope` + `VerificationAuthorizationRef` + coverage-matrix
artifact + canonical allowlist marker (the R1 gate). None exists; no real
run occurred. The synthetic path proves the report shape; the evidence seam
(`--evidence-json`) is exercised on store-verified synthetic manifests only.
Real MBP-1 research and R5B acceptance stay blocked until policy v2 passes
on the real fixture (plan §12 blocker 3).

## Scoping notes (not deviations)

- Dataset-condition records and recovery boundaries are typed records
  carried inside the evidence manifest (they have no store of their own);
  their scope is checked by the evidence contract (review F7).
- The ET session constants used by `authorized_session_span_ns` are
  duplicated from `IFVG_DOC_SESSION_SCHEME` (18:00 boundary, 17:00–18:00
  closed window) rather than read from the scheme object; a drift test is a
  hardening candidate (reviewer 1 "what holds" note).
- Panel/regime files touched concurrently in the worktree
  (`ml/regime_*.py`, `features/context_bar_panel_contract.py`,
  `fold_schedules.py`, `ml/fold_set_artifact.py`, …) belong to R6.1 and are
  NOT part of the R5B.1 commit (path-scoped `git add`; review S8).
