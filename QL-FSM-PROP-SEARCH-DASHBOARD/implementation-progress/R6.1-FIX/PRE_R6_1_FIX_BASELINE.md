# R6.1-FIX — Pre-implementation baseline

Recorded 2026-09-01 before any R6.1-FIX edit (kickoff §1; plan
`../R6.1-FIX-HARDENING-READINESS-PLAN/IMPLEMENTATION_PLAN.md` revision 3,
owner-approved for implementation on 2026-09-01).

| Fact | Value |
|---|---|
| Branch | `feature/ifvg-prop-robust-config-search-v1` |
| HEAD | `6c0b60a1dddee89a058bfe015c55be5fbe2ceced` (R6.1; parent `f3f9ac26…` R5B.1) |
| `git diff --stat HEAD -- src tests scripts pyproject.toml docs/DECISIONS.md` | **empty** — source, tests and scripts are byte-identical to the R6.1 commit |
| Worktree modifications (user-owned, pre-existing since R1) | `ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/README.md`, `docs/pipeline_state.yaml` (the R1 `PRE_EXISTING_DIFF.patch` hunks; never committed by a release) |
| Untracked, user-owned (never committed) | `QL-FSM-PROP-SEARCH-DASHBOARD/`, local `data/ifvg_*` artifacts, `reports/`, `ifvg_search_runs/`, prompt/ledger documents |
| Python / pytest / ruff | 3.13.1 / 9.0.2 / 0.15.2 |
| pyarrow / pandas / numpy / duckdb / pydantic / scikit-learn / threadpoolctl | 20.0.0 / 2.3.1 / 2.3.1 / 1.4.4 / 2.12.5 / 1.7.0 / 3.6.0 |
| `psutil` | not installed (Phase 2 capacity work only) |
| Test baseline (the identical tree) | R6.1 release-final: **1931 passed, 0 failed** as-is and with provider keys cleared (`../R6.1/_final_pytest*.txt`, 386 warnings: 378 third-party SciPy/sklearn deprecations + 8 project-owned) |
| Lint baseline | `ruff check src tests scripts` clean; `git diff --check` clean (`../R6.1/_ruff_and_diffcheck.txt`) |
| Strategy-Core | installed pin `a4e3303179ac6a1088aecaaa3482934cf1aec4d7` (pyproject); read-only |
| Trade-Lab | read-only; untouched |

Golden identities pinned at this HEAD (must not move through R6.1-FIX):

| Identity | Value |
|---|---|
| `resolved_regime_protocol_id` (kmeans_v1 over `B0_CORE`, `REGIME_INPUT_FEATURES`) | `a9b7888ad1ff801ad343422248bdf5b1951ddff9121a972ff23ca3814598159b` |
| R6 golden fold-0 `regime_fit_id` (`test_regime_service.R6_GOLDEN_FOLD0_FIT_ID`) | `1e183cd722612c28c210396e0350ddc50edf3576c16dddbbb005b2a39360f7d0` |
| M0 frozen CatBoost `resolved_hash` (`test_catboost_bundle_model._M0_GOLDEN_RESOLVED_HASH`) | `b967af5e93eecb5be596005c4b41980bb3c5172d0f92c35c51cdcb43d605539e` |
| `core_replay_id` of the fixed test payload (see `test_r61_fix_golden_identities`) | `46b7148c6aee55c1cf62c2524a77a75f4b9b833f2ee423333a5f96ef07362249` |
| `account_simulation_id` of the fixed test payload (see `test_r61_fix_golden_identities`) | `96229ec06622182cf22408beb4ead9d64b8743b3729f652800a0dc07575976dc` |
| `feature_block_registry_hash()` | `4ebbe7ccbb16e897b1b6b208049f1570b226ce31aa5705c625259e5f0e514d5d` |
| `B0_CORE` resolved bundle id | `668f6fa60afe73f63d0036952874bfdd8319c328570041c25cbdcfdf7bc428cb` |

Authoritative inputs: `../../FINAL-IMPLEMENTATION-PLAN-DOCS/` (unchanged),
`../R6.1-CORRECTION-PLAN-DOCS-FINAL/` (unchanged), the R6.1-FIX plan above.
Evidence folders `../R5B.1/`, `../R6/`, `../R6.1/` are never edited.
