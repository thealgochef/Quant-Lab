# HARDENING-BACKEND-FIX — PROGRESS CHECKPOINT (resume-from-here file)

Authority: `IMPLEMENTATION_PLAN.md` (this folder; verbatim copy of
`HARDENING_BACKEND_FIX_IMPLEMENTATION_PLAN.md`, owner-approved). Compact corrective
release: the ten corrections of plan §§4–10, nothing else. One release-scoped commit
parented by HARDENING-BACKEND `e56f9376a5b4ba269f7fa11cdc6e37b08200638f`; evidence under
this folder; no push, no merge; `formal_acceptance_status: transitively_blocked_by_R1`.
Owner actions (window selection, seed-production signature, verification signature) are
NOT performed; no seed production, no real verification, no full pipeline.

## Workstreams (plan §§4–10) — implementation state

| WS | Plan | State |
|---|---|---|
| A | §4 token-safe lock reclamation (`file_mutex.py` new, `owner_decision_lock.py`) + atomic namespace init (`store_namespace.py`) | DONE — `_ws_A_regression.txt` in `_targeted_pytest.txt` |
| B | §5 public source-kind boundary (`trading_calendar.py`, `verification_window.py`, `seed_production.py`) | DONE |
| C | §6 regime OOS / fold provenance, native validation, schema identity | DONE |
| D | §7 central manifest validator, exact label / executed-trade evidence | DONE |
| E | §8 central seed canonicalization (`child_replay.py::save_seed_snapshot`) | DONE |
| F | §9 event-detail partition bound (`EVENT_DETAIL_BUDGET_V2`), benchmark shapes | DONE — `_capacity_benchmark.txt`, `CAPACITY_BENCHMARKS.md/.json` (PASS) |
| G | §10 complete supersession-chain proof at every real seam | DONE — the two WS-G failures fixed; rerun 43 passed (appended to `_targeted_pytest.txt`) |
| docs | `docs/DECISIONS.md` D-051 + reservation note | DONE |
| review | §13 two read-only reviewers, one pass → `FOCUSED_REVIEW.md` | see log |
| gate | §12.3 full suite ×2 (as-is / keys cleared), ruff, diff-check, goldens, benchmark, access proof, user-hunk check, exact reuse | see log |
| package | §14 commit, patch + sha256, bundle + sha256, `COMPLETION_REPORT.md` placeholders resolved | see log |

## Log

- 2026-09-03 01:20Z–02:54Z (previous session) — all seven workstreams implemented with their
  tests (54 new test functions); targeted regressions green (`_targeted_pytest.txt`); the
  focused benchmark PASS; red-first baseline against `e56f937` recorded
  (`_red_first_baseline_pytest.txt`: 68 failed / 131 passed / 5 collection errors); D-051
  recorded; `COMPLETION_REPORT.md` drafted with the placeholders FINAL_COMMIT,
  TESTS, REVIEW, FINAL_GATE; `review_diff.patch` prepared for the reviewers.

## Reset 1 (2026-09-03 ~02:55Z, context window) — resume record

The context window overflowed after the completion report was drafted and (apparently)
while the two reviewers were being launched: no `FOCUSED_REVIEW.md` and no reviewer output
existed on disk; the WS-G follow-up rerun announced in `_targeted_pytest.txt` had not been
appended; no full-suite run, no ruff / diff-check log, no commit, no patch / bundle.

- 03:10Z — fresh session verified the tree: HEAD still `e56f937`, nothing staged; the release
  diff = 50 files (+5,631 / −502; `git diff HEAD -- src scripts tests docs/DECISIONS.md`,
  sha256 `662bc4423bc3dad02ac54eef03ea50c0e9458bce09e359f958e16f4b77cd716b`); the four
  user-owned docs (`ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/README.md`,
  `docs/pipeline_state.yaml`) byte-identical to the baseline capture (patch sha256
  `b12b4200379a101e…`, same four file hashes); `find data -type f -newer <plan>` → 0; no
  python process running.
- 03:11Z — WS-G rerun: `test_owner_decisions.py` + `test_authorization.py` → 43 passed
  (appended to `_targeted_pytest.txt`). Ruff clean; `git diff --check` clean (pre-gate check).
- 03:15Z — Reviewer A (lock / namespace / chain seams / seed canonicalization) and Reviewer B
  (regime evidence / manifests / source kind / event-detail capacity) launched ONCE, read-only,
  ≤5 findings + ≤5 carry-forwards each, ≤1,500 words, over the diff above; outputs
  `_review_A.md` / `_review_B.md`. The exact-reuse evidence run (`_exact_reuse_pytest.txt`:
  pipeline_regime / pipeline_run / orchestrator / evidence-integrity / goldens) launched in
  parallel.
- 03:21Z — exact-reuse evidence GREEN (`_exact_reuse_pytest.txt`: pipeline_regime + pipeline_run +
  orchestrator + evidence-integrity + r61_fix_goldens → **71 passed** in 6:17, exit 0). Access / scope
  proofs collected: `data/` 0 new files, no namespace / mutex / head marker, Strategy-Core clean at
  `a4e3303`, Trade-Lab's 27 dirty files all pre-date the plan; `stage_shared_docs.py` (this folder)
  dry-run OK against HEAD, anchors present in the worktree; `../DECISIONS_TAKEN.md` #126–#133 appended.
- 03:45Z — BOTH REVIEWS DONE (`_review_A.md`: 0 blocker / 1 major RA-01 / 3 minor; `_review_B.md`: 0 blocker /
  1 major RB-01 / 2 medium / 2 minor; access affirmed). FIX ROUND (the one permitted round) applied to the
  tree by `patch_review_round.py` + `patch_review_round_tests.py` (job tmp): RA-01 typed foreign-token release,
  RA-02 `FileMutexError` + `lock_mutex_failed` / `store_namespace_initialization_failed`, RA-03 canonicalization
  coverage (keys, sets, extras, init=False, pandas Timestamp; datetime64 / NaT / ns refused), RA-04 orphan-free
  creation, RB-01 `label_identity_proof` + `assert_persistable_label_proof` at every study save, RB-02 fit-bearing
  folds refuse fit-less rows, RB-03 cleanup covers partial files + the manifest stage, RB-04 independent
  resident-batch observation + identity-bound reader bound, RB-05 consumed columns validated natively; D-051
  amended; `FOCUSED_REVIEW.md` written (9 FIXED, 5 carry-forwards). Ruff + diff-check clean. Direct targeted
  suites running (`_fix_round_pytest.txt`).
- 04:05Z — direct targeted suites: 295 passed / 1 failed (`_fix_round_pytest.txt`) — the failure was the
  RB-05 validation refusing the kernel's documented empty fit id for fit-less folds; `regime_fit_id`
  dropped from the consumed-column spec; rerun of the two OOS files 25 passed. Consumer regression over
  the fixed tree launched (`_exact_reuse_pytest.txt`, overwritten: pipeline_regime / pipeline_run /
  orchestrator / evidence-integrity / goldens / prop seam / stratification gate / external aggregation).
- 04:12Z — `COMPLETION_REPORT.md` review round folded in (file table, assertion evidence, identity note, §7,
  carry-forwards); release diff now 52 files (+6,333 / −536): 26 src+scripts (1 A), 25 tests (2 A),
  `docs/DECISIONS.md`; 60 new test functions; user-owned hunks still byte-identical (`b12b4200…`).
  Remaining placeholders: FINAL_COMMIT, TESTS, FINAL_GATE (+ FINAL_COMMIT in `FOCUSED_REVIEW.md`).
- 04:12Z — consumer regression over the fixed tree GREEN (`_exact_reuse_pytest.txt`: 93 passed, 6:11).
  `_ruff_and_diffcheck.txt` produced (ruff clean; diff-check clean; no production assert; 2195 tests
  collected = +93 over `e56f937`'s 2102). Benchmark RERUN launched ALONE (`_capacity_benchmark.txt`;
  the pre-round log kept as `_capacity_benchmark_before_review_round.txt`; free RAM 8.19 GiB at launch).
- 04:24Z — BENCHMARK RERUN (final tree): every numerical gate PASS (B1 1M peak RSS +0.326 GiB normal /
  +0.326 skewed / +0.328 dense; Python peak 35.8 MiB; observed resident writer batch 50,000 ≤ 50,000 for
  every shape and size; B2 1M +0.478 GiB; output hashes IDENTICAL to the pre-round run: 2c8f09db151d /
  86f9ef4338d0 / 13702335fd42 / 6c6bdee0c013) but `overall: FAIL` on the harness's ENVIRONMENTAL
  precondition only — minimum available RAM at run start 7.868 GiB (B1) / 7.956 GiB (B2) vs the ≥ 8 GiB
  gate: the host had 8.16 GiB free with the user's applications (Chrome, Code, Discord, memory
  compression) holding the rest; no suite was running. The pre-round run had PASSED the same gate at
  8.807 GiB (`_capacity_benchmark_before_review_round.txt`). Decision: run the two full suites now, retry
  the benchmark ONCE afterwards; report exactly. Full suites launched as one sequential chain
  (`_final_pytest.txt` then `_final_pytest_keys_cleared.txt`).
- 04:50Z — the first full-suite chain (tool-backgrounded) was KILLED by the harness at ~22 min (44 % of
  the as-is run; no python process left). Relaunched as a DETACHED Git Bash process
  (`run_final_gates.sh` in the job tmp dir: as-is → keys-cleared → benchmark retry attempt 2, then
  `gates.done`); pytest running since 04:50:06Z; a persistent monitor reports each stage's exit line.
  (Two launch mistakes on the way: `bash` resolved to the WSL launcher; then PowerShell split the
  `-lc` command so an interactive bash hung — both killed before anything ran.)
- 05:19Z — RELEASE-FINAL SUITE (as-is) GREEN: `_final_pytest.txt` **2195 passed, 0 failed in 1735.02 s
  (0:28:55)**, exit 0, no warnings summary (warnings-as-errors). Keys-cleared suite running in the same
  detached chain, then the benchmark retry.
- 05:49Z — KEYS-CLEARED SUITE GREEN: `_final_pytest_keys_cleared.txt` **2195 passed, 0 failed in 1777.70 s
  (0:29:37)**, exit 0 (both provider keys absent in the run's environment); no warnings summary.
  Benchmark retry attempt 2 running alone.
- 06:00Z — BENCHMARK RETRY (attempt 2, machine idle): every numerical gate PASS again (B1 1M +0.338 GiB,
  B2 +0.481 GiB; identical hashes) — `overall: FAIL` on the ≥ 8 GiB available-RAM precondition only
  (7.647 / 7.905 GiB; the owner's applications hold the host memory). Harness / policy NOT modified;
  reported exactly. Access re-check clean (`find data` → 0; no markers; no python process).
- 06:05Z — COMMIT **`a5eee1a`** (`a5eee1aa1e04127809eb57fec1e61331a1d692ef`; tree `7ded6e896f72…`; parent
  `e56f937`): 55 files (+6,410 / −537) = the 52 release files of `_commit_file_list.txt` + the three
  shared docs staged as HEAD + lane transforms (`stage_shared_docs.py`); `--apply-worktree` replayed;
  `_surviving_shared_doc_diff.patch` IDENTICAL to the baseline user-hunk capture (modulo index / @@
  lines); `HARDENING-BACKEND-FIX.patch` (447,911 B, sha256 95fbd946…d3af); `HARDENING-BACKEND-FIX.bundle`
  (`179a2c9..a5eee1a`, requires R6 `179a2c9d…`, carries the branch head; 788,937 B; sha256
  68f094be…a650; `git bundle verify` OK); both `.sha256` files check. Not pushed, not merged.
  `COMPLETION_REPORT.md` / `FOCUSED_REVIEW.md` placeholders resolved.
- 06:06Z — HARDENING-BACKEND-FIX CLOSED: implementation_status complete; backend_dev_complete_for_ui
  true; ui_implementation_may_begin true; formal_acceptance_status transitively_blocked_by_R1.
  Outstanding (not a code gap): the benchmark's formal `overall: PASS` artifact on an idle host. Next
  per the plan (owner actions, unchanged): window selection from `../HARDENING-BACKEND/
  VERIFICATION_WINDOW_SHORTLIST.md`, `SeedProductionAuthorizationRef`, the seed-only chain,
  `VerificationAuthorizationRef`, `init` of the real verification store, the program allowlist, the
  bounded run; the UI/UX implementation branch may rebase onto `a5eee1a` and begin.
- NEXT: nothing for this release; see the owner actions above.
