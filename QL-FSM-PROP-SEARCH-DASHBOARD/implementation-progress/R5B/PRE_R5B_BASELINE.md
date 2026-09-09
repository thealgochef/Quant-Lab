# Pre-R5B Baseline

Recorded 2026-08-26 before any R5B change.

- **Branch**: `feature/ifvg-prop-robust-config-search-v1`
- **HEAD**: `fb8062fe8512d65cdea8f85906c4447a31aba4eb` (R5-FIX)
- **Worktree**: the same four user-owned dirty files as every release since
  R1 — `ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`,
  `docs/README.md`, `docs/pipeline_state.yaml` (pre-existing hunks; never
  committed by release commits) — plus the long-standing untracked
  data/report/prompt entries recorded in `../R1/PRE_IMPLEMENTATION_BASELINE.md`.
- **Python**: 3.13.1 · pytest 9.0.2 · pluggy 1.6.0 (unchanged from R5).

## Baseline test run

`python -m pytest -q` (raw: `_baseline_pytest.txt`):

```text
2 failed, 1604 passed, 46 warnings in 554.31s (0:09:14)   exit=1
```

**The two failures are PRE-EXISTING and environment-dependent, not
regressions**: `tests/agents/test_data_infra.py::TestPolygonDataProvider::
test_connect_without_api_key_raises` and `tests/agents/
test_databento_provider.py::TestDatabentDataProvider::
test_connect_without_key_raises` both assert that provider construction
WITHOUT an API key raises — but this session's environment has
`POLYGON_API_KEY` and `DATABENTO_API_KEY` set (verified via `os.environ`),
so the constructors legitimately do not raise. The same 1,606-test suite
recorded **1606 passed** at R5-FIX in an environment without those
variables. R5B's gate criterion is therefore: **no new failures relative to
this baseline** (the two env-dependent tests are re-checked with the
variables cleared in `TEST_RESULTS.md`).

## Lint baseline

`ruff check src tests scripts` — clean at HEAD (R5-FIX final state).

## Authoritative package

`QL-FSM-PROP-SEARCH-DASHBOARD\FINAL-IMPLEMENTATION-PLAN-DOCS\` (unchanged;
SHA256SUMS.txt as recorded at R1). R5B scope authority:
`PHASED_DELIVERY.md` "Release 5B — Offline MBP-1 Feature Activation"
(the owner's 13 deliverables + gate), `CONTRACTS_AND_SCHEMAS.md` §10,
`DELTA_TAXONOMY.md` §6/§6.2/§7, `TEST_MATRIX.md` §3.8/§3.9/§3.10 R5B rows,
`FRONTEND_UX_CONTRACT.md` §35 (R5B) + §30.1, `OWNER_DECISIONS.md` R-6/P1-D.

## Carried obligations into R5B (from the R5 gate summary)

1. DEV-R5-10 — search-shim factory-signature/store-root alignment
   (safety F4).
2. The MBP-1 activation event itself (the owner's 13 R5B deliverables).
3. DECISIONS_TAKEN #41 — bundle-parametrized ladders "arrive with the
   R5B/R6 feature lanes".

## Strategy-Core / Trade-Lab

Read-only; not consulted for changes. No pin change this release.
