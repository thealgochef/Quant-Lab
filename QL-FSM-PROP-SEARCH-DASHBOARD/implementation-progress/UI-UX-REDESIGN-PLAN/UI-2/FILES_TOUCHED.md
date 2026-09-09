# UI-2 — Files touched

Release commit contents (staged by path; the user's pre-existing worktree hunks never enter it).

## Source — added (2)

| File | Role |
|---|---|
| `src/alpha_lab/agents/data_infra/ifvg/presentation/flows.py` | the goal-derived conditional flows (`StudyFlow`, `flow_for_goal`, `flow_for_draft_fields`, `goal_for_flow`, `prop_objective_selected`, `restore_step_index`, `step_position`) |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/review_vocabulary.py` | the owner-approved verdict labels / definitions over the preserved ledger keys; the `Unreviewed` UI sentinel |

## Source — modified (7)

| File | Change |
|---|---|
| `src/alpha_lab/agents/data_infra/ifvg/presentation/__init__.py` | docstring: the UI-2 modules |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/status_vocabulary.py` | the seven UI-2 empty states mapped onto `UiStatus` (found by the complete suite's first run; see `DEVIATIONS.md`) |
| `src/alpha_lab/agents/data_infra/ifvg/study_drafts.py` | schema 2 (additive `archived`, `archived_at_utc`, `current_step_key`); `archive_draft`, `restore_draft`, `delete_draft_permanently`, `is_empty_untitled_draft`, `bulk_archive_empty_untitled_drafts`, `find_duplicate_drafts`, `proposed_draft_name`; `list_drafts(include_archived=)`; `discard_draft` retired |
| `src/alpha_lab/agents/data_infra/ifvg/study_presentation.py` | `validate_prop_step`: a selected prop objective requires a verified contract (never rewritten) |
| `src/alpha_lab/agents/data_infra/ifvg/study_providers.py` | `verification_bundle_from_signed_ref` (run-independent; `verification_owner_bundle` delegates); the Verification Center read models: shortlist, inventory, center record + receipt pickup, seed authorization / snapshot / receipt states, the signed reference, run registration, the preflight, monitor rows, the evidence summary |
| `src/alpha_lab/agents/data_infra/ifvg/study_status.py` | the seven additive §31 states (`shortlist_unavailable`, `window_not_selected`, `seed_missing`, `final_authorization_unsigned`, `preflight_refused`, `draft_archived`, `draft_session_only`) |
| `src/alpha_lab/agents/data_infra/ifvg/visual_review_store.py` | the additive `not_applicable` verdict key |

## Scripts — added (1)

| File | Role |
|---|---|
| `scripts/ifvg_verification_center.py` | the Verification Center (six sections; no spawn seam; no Publish) |

## Scripts — modified (6)

| File | Change |
|---|---|
| `scripts/ifvg_seed_production.py` | `register-authorization`; `--receipt-out` on `run` / `register-authorization` |
| `scripts/ifvg_study_tab.py` | `REPO_ROOT`, `VERIFICATION_CENTER_ROOT`, roots; session-only Start cards (`stash_session_draft`); `Verify Implementation` delegates to the center |
| `scripts/ifvg_study_wizard.py` | session-only drafts, the Saved / Autosaved / Not-saved chip with autosave, the required name, the duplicate warning, the goal-derived flow (steps, breadcrumb, exact restore, effective payloads), the prop objective that blocks, flow-aware benchmarks, the archived-draft state |
| `scripts/ifvg_ui_common.py` | `SESSION_DRAFT_KEY` |
| `scripts/ifvg_results_tab.py` | History: read-only purpose / store filters, archive / restore / typed delete, the bulk archive, the run archive flag |
| `scripts/ifvg_verifier_tab.py` | the shared review form (per-case keys, `Unreviewed`, labels + definitions, explicit Save Review, `Unsaved` / `Saved`) for candidate and setup modes |

## Tests — added (8)

`tests/agents/ifvg_search/test_presentation_flows.py`,
`tests/agents/ifvg_search/test_presentation_review_vocabulary.py`,
`tests/agents/ifvg_search/test_study_status_ui2.py`,
`tests/agents/ifvg_search/test_study_providers_ui2.py`,
`tests/agents/ifvg_search/test_seed_production_cli_ui2.py`,
`tests/agents/ifvg_search/verification_center_fixture.py` (shared fixture),
`tests/agents/test_ifvg_verification_center.py`, `tests/agents/test_ifvg_verifier_review.py`.

## Tests — modified (8)

`tests/agents/ifvg_search/test_study_drafts.py` (rewritten), `test_study_status.py`;
`tests/agents/test_ifvg_study_wizard.py`, `test_ifvg_study_tab.py`, `test_ifvg_results_tab.py`,
`test_ifvg_setup_verifier_tab.py`, `test_ifvg_study_scans.py`, `test_ifvg_visual_review_store.py`.

## Docs (committed)

`docs/DECISIONS.md` (D-054); `ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`
as HEAD + the UI-2 lane transforms (`stage_shared_docs.py`).

## Dashboard package (untracked by convention, like every prior release's plan / evidence folder)

`FINAL-IMPLEMENTATION-PLAN-DOCS/FRONTEND_UX_CONTRACT.md` (UI-2 amendment of §§3.2, 7, 11, 14.4,
27, 29, 31, 35, 36); `implementation-progress/DECISIONS_TAKEN.md` (#137); this folder
(`_PROGRESS_CHECKPOINT.md`, `DEVIATIONS.md`, `FILES_TOUCHED.md`, `TEST_RESULTS.md`,
`COMPLETION_REPORT.md`, `stage_shared_docs.py`, `run_final_gates.sh`, the pytest / Ruff logs,
`junit_full.xml`, the patch and bundle with their `.sha256`).

Untouched by design: Strategy-Core; the fixed M0–M3 lane; every backend contract of
HARDENING-BACKEND / FIX / FIX.1 (the seed CLI gains operator seams only); `ifvg_pipeline_tab.py`,
`ifvg_active_runs_tab.py`, `ifvg_lab_tab.py`, `ifvg_verifier_charts.py`, `ifvg_mbp1_panels.py`,
`ifvg_regime_panels.py`, `ifvg_results_charts.py`, `ifvg_results_compare.py`; `data/`.
