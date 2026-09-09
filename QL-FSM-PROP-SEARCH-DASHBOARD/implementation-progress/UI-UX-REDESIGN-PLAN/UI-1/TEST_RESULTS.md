# UI-1 — Test results

Environment: Windows 11 (10.0.26200), Python 3.13.1, pytest 9.0.2, Streamlit 1.54.0,
pydantic 2.12.5; `pyproject` `filterwarnings = error` (warnings are errors; the one narrowly
scoped third-party rule unchanged). Every run used `-p no:cacheprovider`.

## Red-first proofs

| Log | Content |
|---|---|
| `_red_A_presentation.txt` | the three presentation unit suites collected against the missing package: 3 collection errors (red) |
| `_red_DE.txt` | `test_study_providers_ui1.py` against the providers without the UI-1 symbols: collection error (red); the charter satisfiability tests appended to `test_charter.py` fail on the unpatched validator |

## Targeted suites (the 27 touched files) — `_targeted_pytest.txt`

**359 passed**, exit 0, no warnings summary. Per suite (as observed in the per-workstream runs):

| Suite | Result |
|---|---|
| `test_presentation_status_vocabulary.py` / `test_presentation_run_purpose.py` / `test_presentation_charter_satisfiability.py` | 37 passed |
| `test_study_status.py` (7) · `test_study_drafts.py` (6) · `test_study_presentation.py` (28) | passed |
| `test_study_providers.py` (12) · `test_study_providers_ui1.py` (4) | passed |
| `test_charter.py` (14) · `test_store_catalog.py` (9) · `test_pipeline_run.py` (20 incl. the namespace / state binding) | passed |
| `test_runner_registry.py` · `test_search_job_script.py` · `test_pipeline_job_script.py` · `test_authorization.py` · `test_orchestrator.py` | passed (unchanged backend seams the UI consumes) |
| `test_ifvg_study_tab.py` (9) · `test_ifvg_study_wizard.py` (31) · `test_ifvg_pipeline_tab.py` (40) | passed |
| `test_ifvg_active_runs_tab.py` (6) · `test_ifvg_results_tab.py` (13) · `test_ifvg_results_compare.py` (5) | passed |
| `test_ifvg_study_scans.py` (14) · `test_ifvg_ui1_truthfulness.py` (14) · `test_ifvg_context_report_adapters.py` (9) | passed |
| `test_ifvg_lab_tab.py` (24) · `test_ifvg_context_experiment_engine.py` (11) | passed |

## Complete suite, ONE invocation, environment as-is

| Run | Log | Result |
|---|---|---|
| 1 (2026-09-04 06:33Z → 07:01Z) — over a tree edited DURING the run (a test-module line removed at 06:34Z) | `_final_pytest_run1_edited_tree.txt`, `junit_full_run1_edited_tree.xml` | 2,362 passed, 1 failed (`test_capability_fallbacks_preserve_semantics`: `AppTest.from_function` read shifted source lines from disk — a harness artifact of the mid-run edit, see `DEVIATIONS.md`); superseded |
| 2 (2026-09-04 07:02Z → 07:30Z) — over the FINAL tree, untouched during the run | `_final_pytest.txt`, `junit_full.xml` | **2,363 passed, 0 failed** in 1,693.82 s (0:28:13), exit 0; no warnings summary; both provider keys present |

## Static gates — `_ruff_and_diffcheck.txt`

`python -m ruff check src tests scripts` → `All checks passed!` (exit 0);
`git diff --check -- src scripts tests docs QL-FSM-PROP-SEARCH-DASHBOARD` → clean (exit 0).

## Not run (by design / not applicable)

- The capacity benchmark: no benchmarked byte changed (the UI lane and the pure presentation
  package only).
- The credentials-cleared full suite: a HARDENING-BACKEND provider-independence gate; UI-1
  touches no provider seam (the environment as-is run is the plan's complete-suite gate).
- Browser / viewport / keyboard evidence: mandatory at UI-6; UI acceptance stays OPEN until then.
