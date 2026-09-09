# R1 — Access-Safety Evidence

## Real-data operations performed during R1 implementation

Exactly two read-only operations against existing, already-authorized
immutable artifacts (no raw-source discovery, no new source reads):

1. Coverage-matrix computation: `load_accepted_v2_tables` over the accepted
   final-review dataset `143b510f8a73896072f44e08f331ef5156e85eb8e5124d25bdf441c4fb6b2ac7`
   (both pinned identities manifest-verified before load) plus the fsm-audit
   companion `7e55ee89…`'s `ifvg_audit_day_funnel.parquet`. Source-partition
   availability came from the dataset identity's `permitted_source_hashes`
   metadata — bytes were never re-read.
2. The window scan over the same loaded tables (pure pandas group-bys).

No other real market-data file was constructed, listed, stat-ed, opened, or
read by the implementation or its tests. All test I/O is confined to
pytest `tmp_path` directories and the synthetic three-day bar fixture.

## Protected/sealed counters

- June 11, 2026 and every date ≥ 2026-06-12: never constructed as a path,
  never listed, never opened. These strings appear in the new lane only
  inside refusal validators and negative tests
  (`VerificationReplayPolicy(("2026-06-11",))` → PermissionError BEFORE path
  construction; `DatePolicy` refuses ≥ 2026-06-11;
  `ArtifactProvenanceReadAdapter` refuses provenance dates > 2026-06-10).
- Development-audit protected counters: structurally zero — no
  `DevelopmentAccessAudit` recorded any PROTECTED_BUFFER/SEALED event during
  the run (no real replay executed; the coverage reads go through the
  manifest-verified artifact loader, not the source-date policy layer).

## Fail-before-path proofs (executed tests)

- 6th real date refused at policy construction (`test_policy_rejects_sixth_and_forbidden_dates_before_any_path`).
- Off-allowlist date inside the window refused + recorded in `denied_dates`.
- Protected/sealed dates refused at construction.
- Rotated allowlist refused program-wide by the canonical marker
  (`test_program_allowlist_marker_refuses_rotation`).
- Synthetic authorization marker refused for the real slice
  (`test_verification_run_refuses_synthetic_marker`).
- Profile/seed mismatch refused before ANY path construction with
  `audit.path_constructions == 0` asserted
  (`test_profile_seed_mismatch_is_refused_before_any_source_read`).

## Runs that did NOT occur

No full-development replay, no feature materialization, no model fit, no
configuration search, no prop search, no bootstrap research run, no operator
full-pipeline run, no real five-day vertical slice, no seed-snapshot
production replay. `docs/pipeline_state.yaml` records
`real_slice_executed: false`, `full_development_run_executed: false`,
`operator_full_run_executed: false`.
