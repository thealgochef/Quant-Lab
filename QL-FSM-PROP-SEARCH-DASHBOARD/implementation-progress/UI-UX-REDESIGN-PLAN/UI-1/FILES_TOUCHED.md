# UI-1 — Files touched

Release commit contents (staged by path; the user's pre-existing worktree hunks never enter it).

## Source — added (4)

| File | Role |
|---|---|
| `src/alpha_lab/agents/data_infra/ifvg/presentation/__init__.py` | the pure presentation package (plan §6 / §8) |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/run_purpose.py` | `RunPurpose`, `EvidenceClass`, `RunPurposeAnnotation`, `resolve_draft_purpose`, `namespace_state_for_store`, `resolve_purpose` |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/charter_satisfiability.py` | `StudyGoal`, `evaluate_charter_satisfiability`, `summarize_challenger_differences`, `STRATEGY_METRIC_NAMES` |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/status_vocabulary.py` | `UiStatus`, `STATUS_SPECS`, gate / reference / evidence rules, additive adapters |

## Source — modified (9)

| File | Change |
|---|---|
| `src/alpha_lab/agents/data_infra/ifvg/study_status.py` | `Start` / `Verify Implementation` routes; the additive §31 states (+ `pipeline_no_runs` re-keyed to `NO_RUNS`) |
| `src/alpha_lab/agents/data_infra/ifvg/study_drafts.py` | additive `purpose_annotation`; clones carry it (`derivation: cloned`) |
| `src/alpha_lab/agents/data_infra/ifvg/study_presentation.py` | the fifth research question; development evidence-date rules; sequential-V1 worker rule; satisfiability / readiness in the review validator; stale mode-5 copy removed |
| `src/alpha_lab/agents/data_infra/ifvg/study_providers.py` | `resolve_store_namespace`, `ArtifactScope`, typed `AuthorizationReadiness` (verification ref / owner bundle), bundle assembly from ready readiness, `locate_charter_store`, namespace-annotated `list_search_runs` / `list_pipeline_runs` |
| `src/alpha_lab/agents/data_infra/ifvg/search/charter.py` | fail-closed satisfiability in `validate_charter` (FSM ≥ 2 profiles, single ≤ 2, Universal ≥ 2 firms, real prop objective needs a firm contract — never rewritten) |
| `src/alpha_lab/agents/data_infra/ifvg/search/pipeline.py` | `store_namespace_id_or_none`, `publication_state_sha256`; gates record namespace id + state digest; `activate_pipeline_result(expected_store_namespace_id=…)` binding |
| `src/alpha_lab/agents/data_infra/ifvg/search/catalog.py` | additive catalog event kind `purpose` (+ index field) |
| `src/alpha_lab/agents/data_infra/ifvg/context_reporting.py` | reconciliation `passed` derived from evaluated report flags; `evaluated`, `gate_evaluations`, `unevaluated_reports` |
| `src/alpha_lab/agents/data_infra/ifvg/context_report_adapters.py` | per-gate / roll-up `status`, `evaluated`; informational access counters |

## Scripts — modified (7)

| File | Change |
|---|---|
| `scripts/ifvg_study_tab.py` | namespace radio removed; `Start` task cards; `Verify Implementation` readiness surface; `workspace_roots` / `roots_for_purpose` / `store_root_for_namespace_class`; `start_draft_from_card` |
| `scripts/ifvg_study_wizard.py` | purpose header + goal card + confirmation; `_resolve_for_draft`, `_satisfiability_for_draft`; Validation (evidence class, typed readiness, frozen warmup prefix, no worker control); Review (satisfiability card, challenger differences, readiness, unrewritten objective); `_assemble_charter` by the actual path; honest `_freeze_and_launch` (registry before spawn; state wait); purpose catalog annotation; configuration sentence |
| `scripts/ifvg_pipeline_tab.py` | purpose-derived store; Launch readiness + satisfiability; honest launch; no worker sliders; run-resolved store for Monitor / Resume / Publish / panels; namespace- and state-bound gate cache and activation; `not_configured` / `artifact_missing` states; execution-mode truth in Monitor |
| `scripts/ifvg_results_charts.py` | `direction_colorscale`; `metric_key` on the heatmap and firm-matrix builders |
| `scripts/ifvg_results_tab.py` | `metric_key` wiring; `no_runs`; per-run store; artifact-derived scope caption / badge; History rows show purpose / namespace / scope; rename into the run's store |
| `scripts/ifvg_active_runs_tab.py` | `no_runs`; per-run store; artifact-derived badge / scope; `artifact_missing` |
| `scripts/ifvg_lab_tab.py` | reconciliation banner derived from evaluated gates (UNAVAILABLE warning, unevaluated list) |

## Tests — added (5)

`tests/agents/ifvg_search/test_presentation_status_vocabulary.py`,
`tests/agents/ifvg_search/test_presentation_run_purpose.py`,
`tests/agents/ifvg_search/test_presentation_charter_satisfiability.py`,
`tests/agents/ifvg_search/test_study_providers_ui1.py`, `tests/agents/test_ifvg_ui1_truthfulness.py`.

## Tests — modified (10)

`tests/agents/ifvg_search/test_charter.py`, `test_pipeline_run.py`, `test_study_presentation.py`,
`test_study_status.py`; `tests/agents/test_ifvg_active_runs_tab.py`,
`test_ifvg_context_report_adapters.py`, `test_ifvg_pipeline_tab.py`, `test_ifvg_study_scans.py`,
`test_ifvg_study_tab.py` (rewritten), `test_ifvg_study_wizard.py` (rewritten).

## Docs (committed)

`docs/DECISIONS.md` (D-053); `ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`
as HEAD + the UI-1 lane transforms (`stage_shared_docs.py`).

## Dashboard package (untracked by convention, like every prior release's plan / evidence folder)

`FINAL-IMPLEMENTATION-PLAN-DOCS/FRONTEND_UX_CONTRACT.md` (UI-1 amendment of §§3.2, 7, 8.1, 14,
15, 30, 31, 35, 36); `implementation-progress/DECISIONS_TAKEN.md` (#136); this folder
(`_PROGRESS_CHECKPOINT.md`, `DEVIATIONS.md`, `FILES_TOUCHED.md`, `TEST_RESULTS.md`,
`COMPLETION_REPORT.md`, `stage_shared_docs.py`, `run_final_gates.sh`, the pytest / Ruff logs,
`junit_full.xml`, the patch and bundle with their `.sha256`).

Untouched by design: Strategy-Core; the fixed M0–M3 lane (`data_access.py`, the capture /
experiment / reporting modules other than the two truthfulness patches above); every backend
contract of HARDENING-BACKEND / FIX / FIX.1; `scripts/ifvg_verifier_*`, `ifvg_mbp1_panels.py`,
`ifvg_regime_panels.py`, `ifvg_results_compare.py`; `data/`.
