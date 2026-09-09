# R5B — Test Results

All commands run from the repo root on Python 3.13.1 / pytest 9.0.2.

## Release-final full suite

```text
python -m pytest -q
→ 2 failed, 1682 passed, 84 warnings in 668.02s (0:11:08)   exit=1
```

(raw: `_final_pytest.txt`)

**The two failures are the PRE-EXISTING environment-dependent pair from the
baseline** (`test_connect_without_api_key_raises` / 
`test_connect_without_key_raises`): this session's environment carries
`POLYGON_API_KEY`/`DATABENTO_API_KEY`, so the "no key → raise" tests
legitimately do not raise. Proof they are environmental, not regressions:
re-run with both variables cleared → **2 passed**
(`_env_dependent_baseline_check.txt`). Same two failures, same tests, in
the pre-R5B baseline.

**Baseline → final: 1604→1682 passed (+78 net new tests), 0 new failures.**

## Sequence

| Run | Result | Raw |
|---|---|---|
| Pre-R5B baseline (`python -m pytest -q`) | 2 failed (env pair), 1604 passed, 9:14 | `_baseline_pytest.txt` |
| Midpoint (pre-adversarial-fix round) | 3 failed (env pair + the FUX `json.loads` scan the new auto-fill tripped — fixed by moving the read into `study_providers`), 1668 passed, 11:44 | `_midpoint_pytest.txt` |
| Release-final (post-fix round) | 2 failed (env pair), 1682 passed, 11:08 | `_final_pytest.txt` |

## Targeted suites (release-final states)

| Suite | Count |
|---|---|
| `tests/agents/ifvg_search` + `tests/agents/data_infra/ifvg` (search + ML lanes, incl. all five MBP-1 suites and the controlled study) | **438 passed** (3:02) |
| `test_mbp1_schemas_and_source.py` | 22 passed |
| `test_mbp1_stage_windows.py` | 10 passed |
| `test_mbp1_materializer.py` | 23 passed |
| `test_bundle_feature_view.py` | 9 passed |
| `test_controlled_feature_study.py` | 8 passed (~1:45 — two-arm logistic fits over 60 synthetic days) |
| `test_pipeline_run.py` (incl. the 4 MBP-1 E2E tests) | 19 passed |
| `test_ifvg_pipeline_tab.py` (incl. 6 R5B AppTests) | 22 passed |
| `test_ifvg_study_scans.py` (FUX source scans) | 14 passed |
| `test_identities.py` (projection audit incl. the 4 new pairs) | 19 passed |

## Lint and hygiene

```text
ruff check src tests scripts   → All checks passed!
git diff --check               → clean (CRLF advisories only, as every release)
```

## Live browser smoke (`browser-smoke/`)

`r5b_smoke_app.py` (content-addressed scratch, R5-FIX evidence-tied reuse
discipline) served the real `render_pipeline_run` over a freshly built,
VALIDATED synthetic B2 16-stage run — **server log: 0 tracebacks, 0
deprecation warnings** (`_smoke_server.log`). Four screenshots bound to
commit `fb8062fe8512`, tree `e90a117170…`, pipeline `146b9ed73b27…`,
scratch key `f2cedfabb2c1429f` (`browser-smoke/MANIFEST.json`):
Configure with the MBP-1 expander; Monitor with 16/16 terminal stages and
the S05/S09 MBP-1 explanations; S11 Blocked + both boundary badges; the
exact stage-window drill-down for `pcand_0000` (all 9 windows, exact
cutoffs, admitted counts, zero ambiguity) and the auto-filled controlled
comparison with the exact safe-failure copy. Desktop viewport only — the
four-viewport/keyboard matrix remains the hardening gate, as designed.
