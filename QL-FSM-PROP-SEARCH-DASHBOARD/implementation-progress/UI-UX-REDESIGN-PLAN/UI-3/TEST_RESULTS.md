# UI-3 — Test results

All runs on the release tree over `eb85d09` (UI-2 release head); isolated `tmp_path` stores
only; no owner artifact signed, no seed produced, no real verification run, no browser evidence
(UI-6). Warnings are errors per `pyproject.toml` (`filterwarnings = error`).

## Red-first proofs (tests written before the implementation)

| Log | Scope | Result |
|---|---|---|
| `_red_A_D_pure.txt` | the four presentation registries (metric registry, roll-ups, help registry, labels) | 4 collection errors — the modules did not exist |
| `_red_E_L_common_scans.txt` | `ifvg_ui_common` primitives; the help / accessible-name / purity scans | 6 failed, 9 passed (the purity checks over the UI-1 / UI-2 modules already held) |
| `_red_FG_context.txt` | the Context Research presentation module, adapters, charts and renderer | collection error — `presentation.context_research` did not exist |
| `_red_H_results.txt` | the Results readings module and the three UI-3 Results AppTests | collection error — `presentation.results_presentation` did not exist |
| `_red_IJK_panels.txt` | the ladder dtype change and the five summary-first panel / ladder AppTests | 6 failed, 39 deselected |

## Regression over every script the help patch touched

| Log | Scope | Result |
|---|---|---|
| `_regression_L_part1.txt` | pipeline tab, results tab, results compare | 6 failed / 55 passed — exactly the six Results tests asserting the R4 `Analyst` / `Audit` labels (updated in H) |
| `_regression_L_part2.txt` | wizard, study tab, active runs, verification center, verifier, setup verifier, review, lab tab, study scans | **118 passed** |

## Targeted suites (29 files, one invocation)

`_targeted_pytest.txt` — **341 passed**, exit 0 (26.81 s), Ruff clean inside the same log:

presentation registries (metric registry 36, roll-ups 3, help registry 4, labels 3, results 3,
status vocabulary, run purpose, flows, review vocabulary, charter satisfiability), study status
(+ UI-2), `ifvg_ui_common` UI-3 (3), help scans (12), study scans (15), Context Research
presentation (12), context report adapters (9), lab tab (24), results tab (19), results compare
(5), pipeline tab (45), wizard, study tab, active runs, verification center, verifier tab, setup
verifier tab, verifier review, setup verifier charts.

## Targeted suites, run 2 (30 files — the UI-1 truthfulness suite added after the run-1 finding)

`_targeted_pytest_run2.txt` — **355 passed**, exit 0 (25.67 s) over the final tree.

## Lint and whitespace

`_ruff_and_diffcheck.txt` — `python -m ruff check src tests scripts`: `All checks passed!`
(exit 0); `git diff --check` over `src scripts tests docs QL-FSM-PROP-SEARCH-DASHBOARD`: clean
(exit 0; the `LF will be replaced by CRLF` lines are the repository's autocrlf notices, not
whitespace findings).

## Complete suite — ONE invocation over the FROZEN tree (detached)

`run_final_gates.sh` → `_final_pytest.txt`, `junit_full.xml`, `gates.done`.

Run 1 (`_final_pytest_run1_integrity_rollup.txt`, `junit_full_run1_integrity_rollup.xml`;
21:22Z → 21:53Z): 2,499 passed / 1 failed —
`test_ifvg_ui1_truthfulness.py::test_reconcile_banner_derives_from_evaluated_gates`, a genuine
defect of the UI-3 banner / readings change (see `DEVIATIONS.md`, gate-run note); fixed, the
targeted set extended, and the complete suite rerun untouched over the final tree.

Run 2 (the final tree): **2500 passed in 1816.72s (0:30:16)**, exit 0 — one invocation, 2026-09-04T21:56:54Z → 2026-09-04T22:27:16Z; POLYGON_API_KEY present: yes; DATABENTO_API_KEY present: yes; `junit_full.xml` beside the log.

## Not run (by design)

The capacity benchmark (no benchmarked byte changed); browser / viewport / keyboard evidence
(mandatory at UI-6; UI acceptance stays OPEN); the M0–M3 goldens are covered by the complete
suite (the computation lane is untouched).
