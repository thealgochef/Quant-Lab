# HARDENING-BACKEND — Test Results

All commands run from the repo root on Python 3.13.1 / pytest 9.0.2
(scikit-learn 1.7.0, SciPy 1.16.0, catboost 1.2.10, pyarrow 20.0.0, pandas
2.3.1, numpy 2.3.1, duckdb 1.4.4, pydantic 2.12.5, ruff 0.15.2 — unchanged;
no new package installed or declared; `psutil` not required). Every run
below executes under the registered warning policy (`pyproject.toml`
`filterwarnings = ["error", <one exact third-party rule>]`): a warning is a
failure.

## Midpoint full suite (the complete tree, BEFORE the adversarial round)

```text
python -m pytest -q -p no:cacheprovider                                   (environment AS-IS)
→ 2078 passed, 0 failed in 1728.66s (0:28:48)               exit=0
```

(raw: `_midpoint_pytest.txt`; started 2026-09-02T05:41Z on HEAD `0c8d528`
+ the complete working tree of every workstream; finished 06:09Z; zero
warnings surfaced — under `filterwarnings = error` any warning would have
failed a test.) **Baseline → midpoint: 1993 (R6.1-FIX, both environments)
→ 2078 passed (+85 net new tests), 0 failures.**

## Adversarial fix round (between the midpoint and the release-final suites)

| Lane | Suite | Result | Raw |
|---|---|---|---|
| main (RA-01/02/03/06/07/08/09/10, B-06, S08 `fold_summary.json`) | `test_hardening_fix_round.py` (14, new; red first inside the module's negative cases) | 14 passed | — |
| main | the affected suites: lock / namespace / chain / owner decisions / authorization / verification / slice / MBP-1 coverage evidence / pipeline contracts / job script / warning policy / seed production / evidence integrity / child replay / runner registry / search job / promotion CLI | **197 passed, 1 failed** → the one failure (`test_coverage_diagnostic_shape_and_fail_before_path`) re-pointed at the earlier namespace refusal; module re-run **18 passed** | `_fix_main_regression.txt` |
| main | `test_pipeline_run` + `test_pipeline_regime` + `test_orchestrator` + `test_ifvg_pipeline_tab` + goldens (the S08 sidecar and activation changes) | **91 passed** (5:44) | `_fix_main_regression_2.txt` |
| main | `test_store_namespace` + `test_pipeline_authority_seams` + `test_hardening_fix_round` (after the namespace-missing message change) | 26 passed | — |
| Fix-E (RA-04, RA-05, B-01, B-03 reader, B-04, B-05, B-07, B-08) | `test_bounded_verification` (13) + `test_bounded_verification_script` (2) + `test_pipeline_authority_seams` (3) + `test_event_detail_streaming` (9) + `test_stratified_prop_external_aggregation` (7) + `test_identities` (19) | **53 passed**; red first `_red_fix_E.txt` (8 failed / 2 passed before the code — the two passes are the B-07 / B-08 test-adequacy items whose code was already correct) | `_fix_E_pytest.txt` |

Collection after the round: `python -m pytest --collect-only -q` → **2102 tests
collected** (2078 at the midpoint + 24 fix-round tests).

## Release-final full suites (AFTER the adversarial round; the working tree that became the release commit)

Every HARDENING-BACKEND source / test / script file was present in the
working tree for both runs (the untracked new modules included), so the runs
exercise exactly the committed code. The only worktree content outside the
commit is the user-owned doc hunks (`ARCHITECTURE.md`, `docs/README.md`,
`docs/pipeline_state.yaml`, `docs/ML_TRAINING_WORKBENCH.md` — no test reads
them) and the pre-existing untracked local data / report files.

```text
python -m pytest -q -p no:cacheprovider                                   (environment AS-IS)
→ 2102 passed, 0 failed in 1384.92s (0:23:04)               exit=0

env -u POLYGON_API_KEY -u DATABENTO_API_KEY python -m pytest -q -p no:cacheprovider
→ 2102 passed, 0 failed in 1344.01s (0:22:24)               exit=0
```

**Both runs: 0 failed.** The provider-key tests are hermetic since R6.1, so
the as-is and keys-cleared counts coincide.

## Sequence

| Run | Result | Raw |
|---|---|---|
| Pre-hardening baseline (= R6.1-FIX release-final, tree identical to `0c8d528`) | 1993 passed, 0 failed, as-is and keys cleared | `../R6.1-FIX/_final_pytest*.txt` |
| Midpoint (complete tree, BEFORE the adversarial round) | **2078 passed, 0 failed**, 28:48 | `_midpoint_pytest.txt` |
| Release-final, as-is (the `e56f937` content) | **2102 passed, 0 failed**, 23:04 | `_final_pytest.txt` |
| Release-final, keys cleared (the `e56f937` content) | **2102 passed, 0 failed**, 22:24 | `_final_pytest_keys_cleared.txt` |

Midpoint → final: +24 tests from the adversarial fix round (14 main-lane
proofs, 3 pipeline authority seams, 5 bounded-verification additions, the
streaming path-count overrun test, the byte-budget test).

## Post-commit shared-doc verification

`stage_shared_docs.py --apply-worktree` replayed the three HARDENING-BACKEND
lane transforms into the worktree; the surviving worktree diff on the four
user-owned files (`ARCHITECTURE.md`, `docs/README.md`,
`docs/pipeline_state.yaml`, `docs/ML_TRAINING_WORKBENCH.md`) is
content-identical to `../R1/PRE_EXISTING_DIFF.patch` — the diff-of-diffs
with the git `index` header lines removed is EMPTY
(`_surviving_shared_doc_diff.patch`, 4 files, 12,348 bytes).

## Source-review evidence

`HARDENING-BACKEND.patch` = `git format-patch --stdout 0c8d528..e56f937`
(720,665 bytes); sha256 in `HARDENING-BACKEND.patch.sha256`
(`ba79837825f75855708f3b74631f2c18d6713cc082296684c3758b84a18f12b3`).
`HARDENING-BACKEND.bundle` = `git bundle create … 179a2c9..feature/ifvg-prop-robust-config-search-v1`
(the prerequisite-complete R5B.1 → R6.1 → R6.1-FIX → HARDENING-BACKEND chain;
prerequisite R6 `179a2c9`; 684,480 bytes; `git bundle verify` OK); sha256 in
`HARDENING-BACKEND.bundle.sha256`
(`81b97e200e7f6ed0fa0c1eed175738906bf2f432002454f24075e9a269992075`).

(raw: `_final_pytest.txt` / `_final_pytest_keys_cleared.txt`; each records
the HEAD at the time of the run (`0c8d528`, the tree that became the release
commit), the presence of the two provider keys, the start / finish instants
and the pytest exit code; under `filterwarnings = error` no warnings summary
can exist — none does.)

**Baseline → final: 1993 (R6.1-FIX, both environments) → 2102 passed (+109
net new tests), 0 failures.**

### Gate access re-check (after the final suites)

```text
find data -type f -newermt "2026-09-02 00:00" | wc -l        → 0
find data -name STORE_NAMESPACE.json | wc -l                 → 0
forbidden store directories under data/ (search, search_test, owner_decision_supersessions,
  seed_production_authorizations, r1_baseline_gate_reports, bounded_release_control_flow_reports) → 0
find data -name VERIFICATION_ALLOWLIST_MARKER.json | wc -l   → 0
Strategy-Core HEAD a4e3303179ac6a1088aecaaa3482934cf1aec4d7, dirty=0
Trade-Lab: 27 pre-existing dirty files, 0 newer than 2026-09-01
```

## Workstream suites (the per-workstream raw outputs)

| Workstream | Suite | Result | Raw |
|---|---|---|---|
| A (namespace / chain / lock / seams) | `test_store_namespace.py` (9) + `test_supersession_chain.py` (7) + `test_owner_decision_lock.py` (8) | 24 passed | this file's history; `_red_ws_A.txt` (the JSONL-log tests fail collection once `SUPERSESSIONS_FILE` is gone) |
| A | `test_owner_decisions.py` (30 rewritten) + `test_verification_slice.py` (7) + `test_child_replay.py` (10) | 47 passed | — |
| A | `test_authorization`, `test_charter`, `test_verification`, `test_study_providers`, `test_mbp1_coverage_evidence`, `test_child_audit_companion`, promotion CLI, `test_regime_store`, `test_identities` | 115 passed | `_ws_A_breakage_inventory.txt` (the pre-patch inventory: 11 failed + 5 errors, all the new required namespace/witness fields — every one patched) |
| A (consumer regression) | `test_pipeline_regime` (19) + `test_pipeline_run` (19) + `test_pipeline_evidence_integrity` (18) + `test_orchestrator` + `test_search_job_script` + `test_regime_stratification_gate` + `test_regime_supervised_studies` + `test_ifvg_pipeline_tab` + `test_ifvg_study_wizard` + goldens | **156 passed** (8:00) | `_ws_A_regression.txt` |
| B (capacity) | `tests/propsim` + the stratification suites + `test_stratified_prop_external_aggregation.py` (6) + `test_event_detail_streaming.py` (8) + `test_pipeline_regime.py` | **189 passed, 0 failed** | `_ws_B_pytest.txt`; red first `_red_ws_B.txt` (12 failed / 2 passed before the code) |
| C (warnings / sequential truth) | `test_hardening_warning_policy.py` (11) + pipeline contracts / run / job script + context-experiment engine + FSM audit contracts + dataset + v2 reconciliation + goldens + `test_ml_pipeline` + logistic | 117 passed / 2 failed at the time (the two failures were WS-A's concurrent field additions, since patched) | `_ws_C_pytest_core.txt`; directory runs `_ws_C_pytest_ifvg_search.txt` (484 passed; the 9 failed / 5 errors were WS-A/WS-D in-progress modules, all green in the midpoint suite), `_ws_C_pytest_data_infra_ifvg.txt` (212 passed), `_ws_C_pytest_propsim_ifvg_misc.txt` (539 passed); red first `_red_ws_C.txt` |
| D (Phase 3) | `test_trading_calendar.py` (5) + `test_verification_window.py` (4) + `test_seed_production.py` (9) + `test_identities` + `test_child_replay` + `test_verification` + `test_verification_slice` | **63 passed** | `_ws_D_pytest.txt`; red first `_red_ws_D.txt` (three collection errors before the code) |
| E (Phase 4) | `test_bounded_verification.py` (8) + `test_bounded_verification_script.py` (2) + `test_identities` (19) | 29 passed | — |

## Capacity benchmark (plan §4.4)

`python scripts/hardening_capacity_benchmark.py --out-dir …/HARDENING-BACKEND`
→ **B1 PASS (8/8 gates), B2 PASS (8/8 gates)**; attempt 2 recorded
(`CAPACITY_BENCHMARKS.md` / `.json`, `_capacity_benchmark_run.txt`);
attempt 1 (`_capacity_benchmark_run_attempt1.txt`) passed every numerical
gate with identical hashes but measured 7.896 GiB free RAM at the B1 start
while other suites ran (the ≥ 8 GiB precondition). No ceiling lowered.

## Real-store refusal proof (Phase 4 runner)

```text
python scripts/ifvg_bounded_verification.py preflight
→ {"status": "refused", "reason": "fail_before_path", "detail": "real verification execution is blocked: no persisted VerificationRunEnvelope exists for this store — the owner's VerificationAuthorizationRef (decisions 21/R-5) has not been granted (fail-before-path)"}   exit=2
find data -type f -newermt "2026-09-02 00:00" | wc -l → 0
```

## Namespace CLI smoke (a temp root)

`init` (intent) → `init --confirm` (initialized, class test) → `init
--namespace-class research --confirm` (refused `store_namespace_divergent`,
exit 2) → `show` (namespace + genesis witness); a `test` class under a
research-looking path refused `store_namespace_deployment_incoherent`; an
unmarked root `show` refused `store_namespace_missing` (also
`test_store_namespace.py::test_cli_…`).

## Lint and hygiene

```text
ruff check src tests scripts   → All checks passed!            (exit 0; whole tree, after every workstream)
git diff --check               → clean (tracked files)          (exit 0)
git add -N <new files>; git diff --check → clean                (exit 0)
grep '^\s*assert ' over every touched / new src module → none
```

(final raw: `_ruff_and_diffcheck.txt`, produced at the gate)

## Golden identities

`tests/agents/ifvg_search/test_r61_fix_goldens.py` (the seven identities of
`PRE_HARDENING_BASELINE.md`) — green in the WS-A regression, in WS-C's core
run, and in the midpoint full suite.

## Collection

`python -m pytest --collect-only -q` → **2078 tests collected** (1993 at the
R6.1-FIX baseline + 85 new).
