# R1 — Adversarial Review (Findings)

Two independent, read-only reviewers were tasked to FALSIFY the R1
implementation against the authoritative package (contract-fidelity mandate;
safety/compatibility mandate). Combined verdict at review time:
**27 findings — 1 BLOCKER, 6 MAJOR, 20 MINOR** (one topic, the program-marker
root, was found by both). No protected/sealed access risk was proven by
either reviewer, and both independently confirmed: exact SC §2 axis
classification (all 44 fields), fail-closed policies (6th date / off-allowlist
/ protected / sealed / profile-seed mismatch all refused before path
construction), M0–M3 and propsim modules byte-untouched, Strategy-Core clean
at the pinned commit, no data-store file created or modified during
implementation (mtime-verified), MBP-10 unrepresentable outside the guard's
own refusal test.

## Contract-fidelity reviewer (17 findings: 1 B, 5 M, 11 m)

- **C1 | BLOCKER** — catalog lock loop caught only `FileExistsError`; on
  Windows a just-unlinked lock surfaces as `PermissionError` (delete-pending),
  escaping the retry loop and losing the writer's event (observed as an
  intermittent suite failure) — violating P0-18's no-loss property.
- **C2 | MAJOR** — `StudyCellSemanticPayload` never validated against the
  dimension registry (the claimed §3.1 payload-level row was hollow); the
  baseline cell carried a block key as its bundle key.
- **C3 | MAJOR** — the vertical slice used a zero-sentinel `core_replay_id`,
  never assembled the `ReplayInputBundle`/`CoreStrategyReplayIdentity`, and
  published nothing — R1 gate items not satisfiable by R1 code.
- **C4 | MAJOR** — `register_program_allowlist` was per-root: a different
  `store_root` evaded the one-canonical-allowlist marker (also safety S10).
- **C5 | MAJOR** — synthetic charters were not namespace-confined (P0-4).
- **C6 | MAJOR** — `validate_charter` lacked the GeneratedProfileCapability
  arm: a generated baseline failed *for being absent from the fixed registry*,
  contradicting §1.4.
- **C7–C17 | MINOR** — inconsistent baseline-ratification posture; axis-key
  binding unchecked in override expansion + incompatibilities never enforced;
  dependent composite member standalone-searchable; `delta_reports` inner
  immutable type dropped + no full config diff; placeholder equality tokens
  (everything-else-equal / charter list) uncomputed + DATA_LINEAGE
  missing-check exemption + model-gate required-difference absent;
  feature_only/cohort_model keyed on the logical model key; mutation-
  adversarial test lacked the contract's rigor; stale catalog lock never
  broken; snapshot/allowlist continuity unasserted; coverage gaps (slice
  composition, scoped source-tree fixture, deferred rows unrecorded);
  block-resolution schema hashes were descriptor stand-ins.

## Safety/compatibility reviewer (10 findings: 0 B, 1 M, 9 m)

- **S1 | MAJOR** — the provenance read adapter could reach the day-artifact
  WRITE path (rebuild on cache miss), stamping a new cache with the wide
  provenance allowlist hash (provenance-stamp forgery; no protected/sealed
  exposure — every date still inner-authorized).
- **S2–S10 | MINOR** — duck-typed trusted-branch attributes + empty-allowlist
  vacuity; store ids not 64-hex-constrained (drive-relative/ADS shapes);
  import-time bare `assert`s stripped under `python -O`; progress-doc
  wording claimed commits/files that did not yet exist; coverage evidence
  scripts not preserved (irreproducible); no output-namespace regression
  test; `DatePolicy` lacked the window lower bound; pickle seed sidecar
  deserialization ungated; program marker root not pinned (= C4).

Full reviewer reports (verbatim) are retained in the session task transcripts;
this file is the release-record summary. Resolution of every finding:
`ADVERSARIAL_REVIEW_RESOLUTION.md`.
