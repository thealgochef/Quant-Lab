# UI-2 — Test results

Environment: Windows 11 (10.0.26200), Python 3.13.1, pytest 9.0.2, Streamlit 1.54.0,
pydantic 2.12.5; `pyproject` `filterwarnings = error` (warnings are errors; the one narrowly
scoped third-party rule unchanged). Every run used `-p no:cacheprovider`.

## Red-first proofs

| Log | Content |
|---|---|
| `_red_A_pure.txt` | the flows / drafts / review-vocabulary / status suites against the missing modules and states: 4 collection errors + the additive-key store test failing (red) |
| `_red_E_providers.txt` | `test_study_providers_ui2.py` against the providers without the UI-2 read models: collection error (red) |
| `_red_G_wizard.txt` | the rewritten wizard AppTests over the UI-1 wizard: 10 failed / 29 passed (the UI-2 behaviours red, the retained UI-1 behaviours green) |
| `_red_HI_history_review.txt` | the History lifecycle, the verifier review and the setup-review tests over the UI-1 surfaces: 8 failed / 20 passed (red) |

## Targeted suites (33 files) — `_targeted_pytest.txt`

**373 passed**, exit 0, no warnings summary. Per area:

| Area | Suites | Result |
|---|---|---|
| Presentation package | `test_presentation_flows.py` (8) · `test_presentation_review_vocabulary.py` (2) · `test_presentation_status_vocabulary.py` · `test_presentation_run_purpose.py` · `test_presentation_charter_satisfiability.py` | passed |
| Drafts / status / providers | `test_study_drafts.py` (13) · `test_study_status_ui2.py` · `test_study_status.py` · `test_study_providers_ui2.py` (10) · `test_study_providers_ui1.py` (4) · `test_study_providers.py` (11) · `test_study_presentation.py` | passed |
| Backend seams consumed | `test_seed_production_cli_ui2.py` (2) · `test_seed_production.py` (11) · `test_charter.py` · `test_pipeline_run.py` · `test_authority_chain_seams.py` · `test_pipeline_authority_seams.py` · `test_bounded_verification.py` · `test_verification_window.py` | passed (unchanged backend contracts) |
| AppTests | `test_ifvg_verification_center.py` (6) · `test_ifvg_study_tab.py` (9) · `test_ifvg_study_wizard.py` (39) · `test_ifvg_results_tab.py` (16) · `test_ifvg_verifier_review.py` (4) · `test_ifvg_setup_verifier_tab.py` (8) · `test_ifvg_verifier_tab.py` (7) · `test_ifvg_pipeline_tab.py` (40) · `test_ifvg_active_runs_tab.py` · `test_ifvg_ui1_truthfulness.py` · `test_ifvg_lab_tab.py` | passed |
| Scans / store | `test_ifvg_study_scans.py` (16; the center scanned, the owner-Q2 delete allowance proven typed, no new spawn site) · `test_ifvg_visual_review_store.py` (5) | passed |

The addendum in the same log (after the vocabulary-mapping fix): the seven presentation / status
suites over the FINAL tree — **56 passed**, exit 0.

## Complete suite, ONE invocation, environment as-is

| Run | Log | Result |
|---|---|---|
| 1 (2026-09-04 09:04Z → 09:34Z) — over the tree before the vocabulary-mapping fix | `_final_pytest_run1_vocabulary_gap.txt`, `junit_full_run1_vocabulary_gap.xml` | 2,414 passed, **1 failed** (`test_existing_vocabularies_map_additively_onto_ui_status`: the seven UI-2 empty states had no `UiStatus` mapping — a real omission, fixed; see `DEVIATIONS.md`); superseded |
| 2 (2026-09-04 09:36Z → 10:06Z) — over the FINAL tree, untouched during the run | `_final_pytest.txt`, `junit_full.xml` | **2,416 passed, 0 failed** in 1,782.47 s (0:29:42), exit 0; no warnings summary; both provider keys present |

## Static gates — `_ruff_and_diffcheck.txt` (over the final tree)

`python -m ruff check src tests scripts` → `All checks passed!` (exit 0);
`git diff --check -- src scripts tests docs QL-FSM-PROP-SEARCH-DASHBOARD` → clean (exit 0).

## Not run (by design / not applicable)

- The capacity benchmark: no benchmarked byte changed (the UI lane, the presentation package,
  the providers' read models and the seed CLI's operator seams only).
- The credentials-cleared full suite: a HARDENING-BACKEND provider-independence gate; UI-2
  touches no provider seam (the environment as-is run is the plan's complete-suite gate).
- Browser / viewport / keyboard evidence: mandatory at UI-6; UI acceptance stays OPEN until then.
- No owner artifact was signed, no seed produced, no real verification run: the Verification
  Center AppTests run over the isolated synthetic fixture store (`verification_center_fixture.py`)
  with test-fixture references — never the repository's real `search_test/v1` store.
