# R6 — Test Results

All commands run from the repo root on Python 3.13.1 / pytest 9.0.2.

## Release-final full suite

```text
python -m pytest -q -p no:cacheprovider
→ 2 failed, 1733 passed, 84 warnings in 602.58s (0:10:02)   exit=1
```

(raw: `_final_pytest.txt`; run detached via `cmd /c`, so the file's
footer states the pytest exit code explicitly)

**The two failures are the PRE-EXISTING environment-dependent pair from the
baseline** (`test_connect_without_api_key_raises` /
`test_connect_without_key_raises`): this session's environment carries
`POLYGON_API_KEY`/`DATABENTO_API_KEY`, so the "no key → raise" tests
legitimately do not raise. Proof they are environmental, not regressions:
re-run with both variables cleared → **2 passed**
(`_env_dependent_check.txt`). Same two failures, same tests, in the
pre-R6 baseline (= the R5B release-final run, `_baseline_pytest.txt`).

**Baseline → final: 1682→1733 passed (+51 net new tests), 0 new failures.**

## Sequence

| Run | Result | Raw |
|---|---|---|
| Pre-R6 baseline (= R5B release-final, same tree as HEAD `7f018f5`) | 2 failed (env pair), 1682 passed, 11:08 | `_baseline_pytest.txt` |
| Midpoint (pre-adversarial round; the pre-reset tree + the lint-only repair of `regime_contracts.py`) | 2 failed (env pair), 1711 passed, 11:32 | `_midpoint_pytest.txt` |
| Release-final (post-fix round: F1–F16 + S1–S9) | 2 failed (env pair), 1733 passed, 10:02 | `_final_pytest.txt` |

## Targeted suites (release-final states)

| Suite | Count |
|---|---|
| `tests/agents/data_infra/ifvg/test_regime_contracts.py` (P1-B grain, structural promotion + role ladders, default-on input permission, fail-closed policies, stamps, identity sensitivity) | **14 passed** |
| `tests/agents/data_infra/ifvg/test_regime_service.py` (determinism, content-bound fit ids, fold locality, unique keys, all-missing rows, alignment + exact tie-break + feature space, typed nulls, sample adequacy, OOS temporal facts, training-only bootstrap, k=2/winsorization, PIT panel assignment, the panel-grain fit path, planned refusals before any fit, stability report) | **20 passed** |
| `tests/agents/data_infra/ifvg/test_regime_store.py` (verify/reload/reuse, relocation, tamper, foreign-frame refusal, verify-before-publish, hardened sidecar loader, no-unpickle UI loader, structural promotion persistence, winsorized reload) | **9 passed** |
| `tests/agents/ifvg_search/test_identities.py` (projection audit incl. the 4 regime pairs) | 19 passed |
| `tests/agents/test_ifvg_pipeline_tab.py` (incl. the 8 R6 AppTests) | 30 passed |
| `tests/agents/test_ifvg_study_scans.py` (FUX source scans, now over both panel scripts) | 14 passed |
| Combined affected re-run after the fix round | **106 passed** (11.75s) |

## Lint and hygiene

```text
ruff check src tests scripts   → All checks passed!
git diff --check               → clean (CRLF advisories only, as every release)
```

## Live browser smoke (`browser-smoke/`)

`r6_smoke_app.py` (content-addressed scratch, evidence-tied reuse
discipline) served the real `render_pipeline_run` over a freshly built,
VALIDATED synthetic 16-stage run plus two persisted `kmeans_v1` regime runs
(healthy n=600; under-sampled n=170 under a distinct winsorized protocol),
their fits, assessments, first promotion decisions, and a
`CONTEXT_BAR_PANEL` protocol — **server log: 0 tracebacks, 0 deprecation
warnings** (`_smoke_server.log`). Eleven screenshots bound to commit
`7f018f57fe2e` (the parent tree; the release commit is the tree the smoke
ran on), source-tree digest `f27edb3c8a21…`, pipeline `0472b6cacdc9…`,
scratch key `46af2a58247ddc2f` (`browser-smoke/MANIFEST.json`): the
Regime Lane expander on the Configure surface; the registry with
planned-disabled entries + the mandatory spectral warning + the stamp
table; the exact-ID model card (identities, coverage, per-fold coverage,
fit identities, nominal occupancy, stability, per-cluster agreement,
centroid profiles, the OOS-timeline transition matrix, gates passed); the
exact-fit-id assignment view, stratification table, and OOS regime
timeline; the promotion role/status view; the under-sampled
insufficient-sample blocked state with gates failed; the cross-protocol
fit/decision refusals with their exact reasons; and the panel-grain
identity. Desktop viewport only — the four-viewport/keyboard matrix
remains the hardening gate, as designed.
