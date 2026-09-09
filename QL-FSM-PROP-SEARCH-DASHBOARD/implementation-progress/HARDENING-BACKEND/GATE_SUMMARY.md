# HARDENING-BACKEND — Gate Summary

**Release:** HARDENING-BACKEND — Phase 2 backend hardening (plan §4: F-11 F-12
F-13 F-17 F-18 F-20) plus the code/contract halves of Phase 3 (§5: F-16 F-21
F-22) and Phase 4 (§6) of
`../R6.1-FIX-HARDENING-READINESS-PLAN/IMPLEMENTATION_PLAN.md` revision 3
(owner-approved 2026-09-01). Phase 1 (R6.1-FIX) is `0c8d528`.
**implementation_status: complete**
**acceptance_status: transitively_blocked_by_R1** (authoring-vs-acceptance
model, V3 P0-8)

> Per the kickoff commit-discipline rule: **implementation complete;
> acceptance blocked pending the owner's actions.** NOT performed by this
> release: the permanent verification-window selection and registration
> (plan §5.1 — the shortlist is evidence with `owner_selection = NOT
> PERFORMED`), the `SeedProductionAuthorizationRef` signature, the real
> seed-only replay, the `VerificationAuthorizationRef` signature, the real
> ≤5-day R1 verification run (§6) and the Full Authorized Development run
> (§7). The real verification store was NOT initialized as a namespace (the
> owner's explicit `scripts/ifvg_store_namespace.py init … --confirm`).
> `full_pipeline_not_run=true`; `data/` gained zero files.

> Session note: the implementation crossed one session rate limit
> (`_PROGRESS_CHECKPOINT.md` "Reset 1"): the midpoint full suite had already
> completed green; the two adversarial reviewers were relaunched with the
> same scope and both reported; the fix round ran as two disjoint lanes;
> the release-final suites ran twice over the combined tree; only then was
> the release-scoped commit made.

## Commits

- **`e56f937`** (`e56f9376a5b4ba269f7fa11cdc6e37b08200638f`; branch `feature/ifvg-prop-robust-config-search-v1`; parent
  `0c8d528` = R6.1-FIX; git tree `104900745106…`) — **61 files changed, +13,386 / −721**:
  26 added (7 src modules, 5 scripts, 14 test files incl. the `namespace_fixture.py` helper)
  + 35 modified (14 src, 2 scripts, 15 tests, `pyproject.toml`, `docs/DECISIONS.md` + the three
  staged shared docs) — see `FILES_TOUCHED.md`, reconciled against `git show --name-status`.
  **Message:** `HARDENING-BACKEND: semantic store namespace, immutable
  supersession chain with head witnesses, liveness-aware lock, measured
  capacity, warnings as errors, sequential truth; Phase 3/4 contracts authored`
  (+ the acceptance-blocked statement).
- Not pushed. Not merged.
- **Path-scoped**: the commit carries exactly the file list of
  `_commit_file_list.txt` + `docs/DECISIONS.md` + the three staged shared
  docs; `docs/ML_TRAINING_WORKBENCH.md` (user-owned) and the pre-existing
  untracked local files (data / reports / prompts / the `QL-*` evidence tree)
  are NOT in the commit.
- Shared-doc staging: `ARCHITECTURE.md`, `docs/README.md`,
  `docs/pipeline_state.yaml` staged as HEAD + HARDENING-BACKEND lane
  transforms ONLY (`stage_shared_docs.py`; dry-run verified against HEAD);
  post-commit `--apply-worktree` replayed the same transforms and **the surviving worktree diff
  on the four user-owned files is content-identical to `../R1/PRE_EXISTING_DIFF.patch`**
  (diff-of-diffs empty modulo `index` header lines; `_surviving_shared_doc_diff.patch`, 12,348 B).
- Source-review evidence: `HARDENING-BACKEND.patch` (`git format-patch --stdout
  0c8d528..e56f937`, 720,665 bytes, sha256
  `ba79837825f75855708f3b74631f2c18d6713cc082296684c3758b84a18f12b3`) +
  `HARDENING-BACKEND.patch.sha256`; **`HARDENING-BACKEND.bundle`** (the prerequisite-complete
  chain R5B.1 → R6.1 → R6.1-FIX → HARDENING-BACKEND = `179a2c9..e56f937`; the bundle
  requires `179a2c9d…` (R6) and carries the branch head; 684,480 bytes; `git bundle verify` OK;
  sha256 `81b97e200e7f6ed0fa0c1eed175738906bf2f432002454f24075e9a269992075`) +
  `HARDENING-BACKEND.bundle.sha256`.

## Implementation-progress files produced

`PRE_HARDENING_BASELINE.md`, `_DESIGN_SPEC.md`, `PHASE2_BACKEND_HARDENING.md`,
`PHASE3_SEED_PRODUCTION_AND_WINDOW.md` (+ `LOGICAL_WINDOW_COVERAGE_SCAN.json`,
`VERIFICATION_WINDOW_SHORTLIST.md`), `PHASE4_BOUNDED_VERIFICATION.md`,
`FILES_TOUCHED.md`, `DEVIATIONS.md` (DEV-HB-1…33), `TEST_RESULTS.md`,
`ACCESS_SAFETY_EVIDENCE.md`, `CAPACITY_AND_SCALING_REPORT.md` (=
`CAPACITY_BENCHMARKS.md` + `.json`, the raw driver logs),
`WARNING_POLICY_REPORT.md` + `WARNING_BASELINE.json`, `ADVERSARIAL_REVIEW.md`
(both reviewer reports verbatim; raw `_review_authority.md` /
`_review_capacity_pipeline.md`), `ADVERSARIAL_REVIEW_RESOLUTION.md` (20
dispositions), `stage_shared_docs.py`, the workstream notes (`_ws_B_NOTES.md`,
`_ws_C_NOTES.md`, `_ws_D_NOTES.md`, `_fix_E_NOTES.md`), raw outputs
(`_red_ws_*.txt`, `_red_fix_E.txt`, `_ws_*_pytest*.txt`,
`_ws_A_breakage_inventory.txt`, `_ws_A_regression.txt`,
`_fix_main_regression*.txt`, `_fix_E_pytest.txt`, `_midpoint_pytest.txt`,
`_final_pytest.txt`, `_final_pytest_keys_cleared.txt`,
`_ruff_and_diffcheck.txt`, `_surviving_shared_doc_diff.patch`,
`_capacity_benchmark_run*.txt`), `HARDENING-BACKEND.patch` + `.sha256`,
`HARDENING-BACKEND.bundle` + `.sha256`, `_PROGRESS_CHECKPOINT.md`,
`GATE_SUMMARY.md` (this) + `../DECISIONS_TAKEN.md` entries 115–125.

## Exact commands, exit codes, counts, timings

See `TEST_RESULTS.md`. **Release-final headline (the `e56f937` content): full repo 2102
passed, 0 failed as-is (23:04) and 2102 passed, 0 failed with the provider keys cleared (22:24);
+109 net new tests over R6.1-FIX (1993), 0 failures in either environment** — under
`filterwarnings = error` (no warnings summary exists in either raw output).
`ruff check src tests scripts` clean over the whole tree; `git diff --check`
clean over the tracked and the 26 intent-added new files; no `assert` in any
touched / new src module (`_ruff_and_diffcheck.txt`).

## Gate status (plan §4.7 + §4.1–§4.6, §5, §6 → evidence)

| Gate item | Status |
|---|---|
| Namespace migration / relocation evidence (§4.1) | **PASS** — `test_store_namespace.py` (9): explicit idempotent immutable init, verified load, identity not a path hash + relocation, path heuristic = defense in depth, unmarked stores carry no authority, CLI intent/confirm; `scripts/ifvg_store_namespace.py` smoke |
| Supersession rollback / crash evidence (§4.2) | **PASS** — `test_supersession_chain.py` (7) + the rewritten `test_owner_decisions.py`: orphan records have no authority, crash before the head update repaired idempotently, rollback / deletion / forgery / rewritten record refused, witness refuses missing / shorter / different heads; the witness rule at every real seam incl. the MBP-1 diagnostic (RA-01) and activation (RA-03) |
| Lock liveness evidence (§4.3) | **PASS** — `test_owner_decision_lock.py` (8) + `test_hardening_fix_round.py`: slow live holder never reclaimed, dead pid / PID reuse reclaimed after the heartbeat timeout, other host / malformed never reclaimed, a lost lock aborts before the head moves, release survives a transient reader and types a persistent failure (RA-02) |
| Capacity B1 / B2 numerical gates (§4.4) | **PASS** — `CAPACITY_BENCHMARKS.md`: 16/16 gates at 250k / 500k / 1M synthetic rows with native RSS, byte-identical repeats, projections at the registered maxima (B1 1.446 GiB @10M ≤ 4.4 GiB; B2 0.796 GiB @5M); gates stated for the benchmark shape (B-02, DEV-HB-32) |
| Warning policy (§4.5) | **PASS** — `WARNING_BASELINE.json`: project-owned warnings = 0; one exact third-party rule; both release-final suites ran under `filterwarnings = error` |
| Sequential-executor truth (§4.6) | **PASS** — `WorkerPolicy` accepts only the int 1 (typed `unsupported_worker_parallelism_v1`; 0 / 2 / 4 / 8 / True / "1" / 1.0 refused); receipts persist `effective_workers=1` / `execution_mode=sequential_children_v1`; the job shim refuses before job creation |
| Phase 3 contracts (§5) | **AUTHORED** — calendar, shortlist (the June proposal INELIGIBLE; the R1 store-day candidate provisional/ineligible; owner selection NOT performed), seed-production policy / authorization / run (synthetic proof; post-condition RA-07), unsigned packets |
| Phase 4 code (§6) | **AUTHORED** — typed preflight (17 reasons), `R1BaselineGateReport`, `BoundedReleaseControlFlowReport` (typed from persisted evidence incl. a real panel-grain regime run), the runner (refuses `fail_before_path` on the real store; proofs gathered by scratch probes) |
| Two full suites, as-is and with provider keys cleared, zero failures | **PASS** — 2102 / 0 and 2102 / 0 (`_final_pytest*.txt`); midpoint 2078 / 0 before the round (`_midpoint_pytest.txt`) |
| Ruff, `git diff --check`, exact golden-ID checks | **PASS** — `_ruff_and_diffcheck.txt`; `test_r61_fix_goldens.py` green in every run |
| Zero protected / sealed access and zero new real-data artifacts | **PASS** — `ACCESS_SAFETY_EVIDENCE.md`; `find data -type f -newermt "2026-09-02 00:00"` → 0 at baseline, after every workstream, after the fix round and at the gate; both reviewers' access verdicts AFFIRMED |
| Prerequisite-complete Git bundle and SHA-256 | **PASS** — `HARDENING-BACKEND.bundle` (`179a2c9..e56f937`; verify OK) + `.sha256` |
| One focused two-reviewer adversarial pass | **PASS** — 0 blockers; 2 major (RA-01, RA-02) + 6 medium + 12 minor, ALL 20 dispositioned: 17 FIXED (2 as variants), 2 ACCEPTED-documented (RA-11, B-09), 0 open |
| `implementation_status=complete`, `acceptance_status=transitively_blocked_by_R1`, `full_pipeline_not_run=true` | **STATED** — this file, `docs/pipeline_state.yaml` (`HARDENING_BACKEND`), D-050 |
| Commit boundary (one release-scoped commit parented by `0c8d528`; no push, no merge) | **PASS** — `e56f937` |

## Findings closed (plan §2 → code)

| Finding | Closed by |
|---|---|
| F-11 path-derived authority | `store_namespace.py` (+ every consumer seam; RA-01 / RA-09 in the round) |
| F-12 mutable supersession markers | `supersession_chain.py` + the witness rule (RA-03 in the round) |
| F-13 age-only lock reclaim | `owner_decision_lock.py` (RA-02 / RA-10 in the round) |
| F-17 capacity without RSS proof | the streaming writer, the external aggregation, `HARDENING_CAPACITY_POLICY_V1` |
| F-18 386 warnings | `filterwarnings = error` + `concat_schema_aligned`; project-owned = 0 |
| F-20 unsupported parallelism claim | `WorkerPolicy == 1`, receipts, the job shim |
| F-16 / F-21 / F-22 (owner-decision blockers) | contracts authored: seed production, seed-production authorization, the logical-day calendar + shortlist; the owner actions remain |

## Open blockers (in order)

1. **Owner window selection** (plan §5.1; decisions 21 / R-5) from the
   LOGICAL-day shortlist — the June proposal is ineligible as stated.
2. **`SeedProductionAuthorizationRef`** signature → the seed-only replay
   (`scripts/ifvg_seed_production.py run`) → owner review of the concrete seed.
3. **`VerificationAuthorizationRef`** signature; explicit `init` of the real
   verification store as a `test` namespace; the one program allowlist; the
   real ≤5-day run (`scripts/ifvg_bounded_verification.py run`) → R1
   acceptance and the dependent releases.
4. Hardening candidates recorded, not gates: a streaming simulation seam
   (`AccountSimulationRun` still materializes walk results, DEV-HB-11); a
   row-based event-detail flush guard as a versioned capacity policy
   (DEV-HB-32); the R1 seed sandbox / `pytz` canonicalization inside
   `save_seed_snapshot` (DEV-HB-18); the UI worker slider (DEV-HB-14); the
   plan's deferred cleanups (DEV-R6.1-FIX-11).

## Protected/sealed counters

**Zero.** See `ACCESS_SAFETY_EVIDENCE.md`.
