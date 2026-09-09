# HARDENING-BACKEND — Pre-implementation baseline

Recorded 2026-09-02T04:47Z before any HARDENING-BACKEND edit (plan
`../R6.1-FIX-HARDENING-READINESS-PLAN/IMPLEMENTATION_PLAN.md` revision 3,
owner-approved 2026-09-01; Phase 1 = R6.1-FIX is complete at `0c8d528`).
This release implements Phase 2 (backend hardening, §4), Phase 3 contract
authoring (§5: logical trading-day mapping and shortlist, seed-production
policy / authorization / run contracts, unsigned packets) and Phase 4 code
authoring (§6: preflight, R1 baseline gates, the bounded release-control-flow
report). No owner action is taken or simulated: the permanent allowlist is
NOT selected or registered, no `SeedProductionAuthorizationRef` or
`VerificationAuthorizationRef` is created, no real seed replay and no real
≤5-day verification run is performed.

| Fact | Value |
|---|---|
| Branch | `feature/ifvg-prop-robust-config-search-v1` |
| HEAD | `0c8d5287b601c53ce4c00c90adc7ab49b1e7dade` (R6.1-FIX; parent `6c0b60a1…` R6.1) |
| `git diff --stat HEAD -- src tests scripts pyproject.toml docs/DECISIONS.md` | **empty** — source, tests, scripts and the project config are byte-identical to the R6.1-FIX commit |
| Worktree modifications (user-owned, pre-existing since R1) | `ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/README.md`, `docs/pipeline_state.yaml` (the R1 `PRE_EXISTING_DIFF.patch` hunks; never committed by a release) |
| Untracked, user-owned (never committed) | `QL-FSM-PROP-SEARCH-DASHBOARD/`, local `data/ifvg_*` artifacts, `reports/`, `ifvg_search_runs/`, prompt/ledger documents |
| Python / pytest / ruff | 3.13.1 / 9.0.2 / 0.15.2 |
| pyarrow / pandas / numpy / duckdb / pydantic | 20.0.0 / 2.3.1 / 2.3.1 / 1.4.4 / 2.12.5 |
| `psutil` | not installed — the capacity monitor uses the native Win32 `GetProcessMemoryInfo` / POSIX `resource` readers (no new dependency) |
| Host | AMD Ryzen 7 7800X3D (16 logical), 31.15 GiB RAM total, 9.65 GiB free at baseline (Windows 11 Pro 10.0.26200) |
| Test baseline (the identical tree) | R6.1-FIX release-final: **1993 passed, 0 failed** as-is and with provider keys cleared (`../R6.1-FIX/_final_pytest*.txt`); 435 warnings = 429 third-party (sklearn/SciPy L-BFGS-B `disp`/`iprint` `DeprecationWarning` at `sklearn/linear_model/_logistic.py:456`) + 6 project-owned (5 × `dataset.py:813` pandas concat `FutureWarning`, 1 × `tests/agents/test_ifvg_context_experiment_engine.py:260` `FutureWarning`) |
| Lint baseline | `ruff check src tests scripts` clean; `git diff --check` clean (`../R6.1-FIX/_ruff_and_diffcheck.txt`) |
| Strategy-Core | installed pin `a4e3303179ac6a1088aecaaa3482934cf1aec4d7`; worktree clean; read-only |
| Trade-Lab | read-only; untouched |
| `find data -type f -newermt "2026-09-02 00:00"` | **0** files (no artifact newer than the R6.1-FIX close) |

Golden identities pinned at this HEAD (must not move through HARDENING-BACKEND;
`tests/agents/ifvg_search/test_r61_fix_goldens.py` stays green):

| Identity | Value |
|---|---|
| `resolved_regime_protocol_id` (kmeans_v1 over `B0_CORE`, `REGIME_INPUT_FEATURES`) | `a9b7888ad1ff801ad343422248bdf5b1951ddff9121a972ff23ca3814598159b` |
| R6 golden fold-0 `regime_fit_id` | `1e183cd722612c28c210396e0350ddc50edf3576c16dddbbb005b2a39360f7d0` |
| M0 frozen CatBoost `resolved_hash` | `b967af5e93eecb5be596005c4b41980bb3c5172d0f92c35c51cdcb43d605539e` |
| `core_replay_id` of the fixed test payload | `46b7148c6aee55c1cf62c2524a77a75f4b9b833f2ee423333a5f96ef07362249` |
| `account_simulation_id` of the fixed test payload | `96229ec06622182cf22408beb4ead9d64b8743b3729f652800a0dc07575976dc` |
| `feature_block_registry_hash()` | `4ebbe7ccbb16e897b1b6b208049f1570b226ce31aa5705c625259e5f0e514d5d` |
| `B0_CORE` resolved bundle id | `668f6fa60afe73f63d0036952874bfdd8319c328570041c25cbdcfdf7bc428cb` |

Registered capacity ceilings at this HEAD (the §4.4 projection targets):
`EVENT_DETAIL_BUDGET_V1` = 10,000,000 detail rows / 2 GiB / path block 250;
`ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1` = 5,000,000 summary rows / 256 MiB.

Authoritative inputs: `../../FINAL-IMPLEMENTATION-PLAN-DOCS/` (unchanged),
`../R6.1-CORRECTION-PLAN-DOCS-FINAL/` (unchanged), the plan above. Evidence
folders `../R5B.1/`, `../R6/`, `../R6.1/`, `../R6.1-FIX/` are never edited.
