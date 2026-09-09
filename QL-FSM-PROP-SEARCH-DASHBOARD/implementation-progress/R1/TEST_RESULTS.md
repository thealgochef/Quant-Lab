# R1 — Test Results

All commands run from `C:\Users\gonza\Documents\Claude-Quant-Lab` on branch
`feature/ifvg-prop-robust-config-search-v1`.

| Check | Command | Result | Exit |
|---|---|---|---|
| Baseline full suite (pre-implementation) | `python -m pytest -q` | **1089 passed**, 3 warnings, 394.59s | 0 |
| R1 lane suite (pre-review) | `python -m pytest tests/agents/ifvg_search -q` | 119 passed, 1.9s | 0 |
| Interim full regression (pre-review) | `python -m pytest -q` | 1208 passed, 3 warnings, 395.89s | 0 |
| R1 lane suite (post-review-fixes, definitive) | `python -m pytest tests/agents/ifvg_search -q` | **137 passed**, 2.6s | 0 |
| Full regression (post-review-fixes, definitive) | `python -m pytest -q` | see `R1_PYTEST_FULL.txt` + `GATE_SUMMARY.md` (1089 pre-existing + 137 new expected) | 0 |
| Lint (post-review-fixes) | `ruff check src tests` | `All checks passed!` | 0 |
| Whitespace | `git diff --check` | clean (CRLF warnings only — repo-normal) | 0 |

## What the 119 new tests cover (TEST_MATRIX rows)

- §3.1: core-replay reuse/independence/sensitivity; membership separation;
  companion versioning; canonical semantic naming (incl. the study-independent
  record-id consequence); typed axis values incl. None/tuple round-trips;
  axis-registry fail-closed (inert traps, reducer-hardcoded, parent-fill
  `blocked_pending_owner_policy_review`); charter determinism + ceiling;
  store save→reload→assert/overwrite refusal/verified reuse/tamper detection;
  concurrent catalog publishers (8×25 events, zero loss), torn-final-line
  recovery, deterministic rebuild; feature-block partition + frozen tier
  bundles order-exact; MBP-10 guard; planned-block/bundle refusal; activation
  version-bump (mints resolved id, changes registry hash, dependent bundle id).
- §3.8: ReplayInputBundle content sensitivity (partition bytes, day-artifact
  manifests); QL replay-source identity sensitivity (field-level); portability
  + reorder-only normalization; StudyCell annotation exclusion; identity
  self-field exclusion (projection audit over all 12 registered pairs);
  GeneratedProfileCapability (5 paths); computation-path-scoped authorization
  (10 parameterized modes); synthetic-marker refusal for the real slice;
  strategy-only cell without v3.
- §3.9: complete Payload/Envelope conversion + registry-key/resolved-id
  distinctness; legacy source-kind scoping (provenance-only; not a block
  source kind); physical partition identity (two partitions of one trading
  day never collide; unnormalizable orderings refused); access-audit
  separation (runtime audit never in identity); canonical-allowlist marker
  program-wide; deep-immutability mutation-adversarial (all registered
  payloads); window-trigger semantics registry + no-+inf construction.
- R0→R1 entry items (resolved synthetically per kickoff §3):
  (a) mid-chain start — snapshot-restart emission-identical to the continuous
  chain, both in-memory and through `build_ifvg_v2_capture` on-disk caches
  (`start_after_artifact`), with the `final_day_exhausts_dataset=False`
  requirement for snapshot-producing prefixes proven;
  (b) native-ID replay determinism — same synthetic 3-day chain replayed
  twice → byte-identical table content hashes for all 7 typed tables
  (walk exercises the funnel: setups born ≥ 1, taps, parent locks, opposing).
- Dual-drive audit neutrality: audit on/off core tables hash-equal +
  referential integrity + `ChildAuditNeutralityReport.passed`.
- Seed snapshots: immutable save/load round-trip, profile-bound refusal
  before source reads, mismatched-seed save refusal.
- Verification: `VerificationRunPayload` exact binding (authorization,
  allowlist hash, seed, coverage, pipeline id, profile) fail-before-path;
  nonresearch control-flow gates; exact report stamps; coverage-matrix
  builder honesty (uncovered paths named for synthetic coverage).

## Adversarial-review additions (post-fix; 18 new tests)

Slice composition (real identity assembly, publication, honest open gates,
canonical namespace binding, snapshot continuity, idempotent reuse), adapter
read-only guard, sandboxed seed unpickling, output-namespace Literal pin,
catalog stale-lock break, store 64-hex ids, generated-baseline charter arm,
synthetic-charter namespace confinement, uniform ratification posture,
axis-key binding, measurement-only dependent member, structural
everything-else-equal comparisons, model-gate required-difference, full
config diff, resolved-model-identity equalities, deep mutation walk, scoped
source-tree identity fixture, cell-payload registry refusals.

## Deferred TEST_MATRIX rows (explicitly deferred, not silently uncovered)

- §3.1 "UI provider surfaces registered values only" → R4 (no UI provider
  exists in R1; the axis registry itself refuses raw overrides).
- §3.8 "Planned spectral capability refusal — (registry checks in R1)" → the
  regime algorithm registry is authored in R5/R6 per PHASED_DELIVERY's file
  lists; its registry-refusal checks land with it (the parenthetical
  conflicts with the R6 file list; the file list governs).
- §3.9 legacy source-kind "one guard test per surface" → the feature-block
  surface guard exists now; dashboard/live/model-feature surface guards land
  with those surfaces (R4/R5/R5B).
- Execution-attempt identity (P0-3) → R5 (pipeline contracts).

## NOT run (by design; blocked or later-release)

- The real five-day baseline vertical slice (blocked on
  `VerificationAuthorizationRef` — R1 acceptance blocker).
- The seed-snapshot production replay (owner-authorized acceptance action).
- Any full-development run; any operator pipeline run.
- R2+ suites (orchestrator/lineage/prop/UI/ML) — later releases.
