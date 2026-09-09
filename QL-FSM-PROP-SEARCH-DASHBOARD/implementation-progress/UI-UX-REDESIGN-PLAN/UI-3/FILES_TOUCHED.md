# UI-3 — Files touched

Release commit contents (staged by path; the user's pre-existing worktree hunks never enter it).

## Source — added (6)

| File | Role |
|---|---|
| `src/alpha_lab/agents/data_infra/ifvg/presentation/metric_registry.py` | the metric metadata registry: `MetricSpec` / `ReferenceSpec` / `MetricReading`, `describe`, `metric_keys`, `metric_keys_for_surface`, `gate_metric_key`, `format_value`, `evaluate_metric`, `evaluate_interval`, `evaluate_gate_flag`, `unavailable_reading` |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/rollups.py` | the deterministic section roll-ups: `RollupSection`, `SECTION_LABELS`, `SectionRollup`, `rollup_section` |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/help_registry.py` | `HelpEntry` / `HELP_REGISTRY` / `help_text` / `help_for_metric`; `GLOSSARY` / `GLOSSARY_TERMS` / `glossary_markdown`; `HELP_EXEMPTIONS` |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/labels.py` | the human-label registries, `label_for` / `technical_key_for`, `AvailabilityKind` / `AVAILABILITY_CHIPS` / `availability_chip` / `availability_for_*` |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/context_research.py` | the Context Research readings and roll-ups: `candidate_readings`, `sample_adequacy_readings`, `fold_chips`, `importance_top`, `execution_readings`, `coverage_readings`, `m3_status_chip`, `reconciliation_readings`, `decision_summary`, `compatibility_reasons` |
| `src/alpha_lab/agents/data_infra/ifvg/presentation/results_presentation.py` | the Results readings: `strategy_readings`, `prop_readings`, `worst_vector_value`, `column_guide` |

## Source — modified (2)

| File | Change |
|---|---|
| `src/alpha_lab/agents/data_infra/ifvg/presentation/__init__.py` | docstring: the UI-3 modules |
| `src/alpha_lab/agents/data_infra/ifvg/context_report_adapters.py` | additive keys read from what the frozen statistics persist: `reference_brier_score` / `count` in `metrics`, `calibration`, `oos_count`, `fold_summary` (candidate); `mean_realized_r`, `winning_trade_count` (execution); `access` (reconciliation) |

## Scripts — modified (13)

| File | Change |
|---|---|
| `scripts/ifvg_ui_common.py` | `DETAIL_LEVEL_LABELS`, `detail_levels` (`disclosure_level` delegates), `identity_reveal`, `status_chip_line`, `metric_card`, `rollup_card`, `glossary_expander`; help on the pagination controls |
| `scripts/ifvg_lab_tab.py` | Context Research: the decision summary first, the sample-adequacy card, registry metric cards, the named reliability diagonal, coverage / net R on separate axes, fold chips, intervals, top-10 importance, the roll-up-keyed reconciliation banner, raw JSON under Technical identity & audit, run-compatibility reasons in words, render tags, the glossary; help on every widget |
| `scripts/ifvg_lab_charts.py` | `build_coverage_figure` over the adapter's threshold rows (separate axes; legacy shape kept), `build_reliability_figure` |
| `scripts/ifvg_results_tab.py` | the Selected configuration block (roll-ups + registry metric cards), registry captions on the metric pickers, the explorer column guide, `_vector_value` delegating to the pure helper; help on every widget; the rename input's accessible name |
| `scripts/ifvg_pipeline_tab.py` | `_ladder_frame` (AUC Float64 + `AUC reason` + `Model`), `_ladder_definitions` caption; help on every widget |
| `scripts/ifvg_mbp1_panels.py` | summary-first (readiness resolved from the selected run), Research details, Advanced diagnostics with the manual exact-id inputs (same keys), registries and stamps; availability chips and human names in the tables |
| `scripts/ifvg_regime_panels.py` | summary-first (`_render_summary`), `_advanced_inputs`, `_render_model_card` over the ids returning facts, `_render_promotion_view` returning the payload; the read-only rule kept |
| `scripts/ifvg_study_tab.py` | help on the Verification Center card button |
| `scripts/ifvg_study_wizard.py` | help on every widget; the gate inputs' accessible names + `help_for_metric(gate_metric_key(name))` |
| `scripts/ifvg_active_runs_tab.py` | help on every widget |
| `scripts/ifvg_results_compare.py` | help on every widget |
| `scripts/ifvg_verification_center.py` | help on the four remaining buttons |
| `scripts/ifvg_verifier_tab.py` | help on every widget (UI-4 regroups the controls; the entries stay) |

## Tests — added (8)

`tests/agents/ifvg_search/test_presentation_metric_registry.py`,
`tests/agents/ifvg_search/test_presentation_rollups.py`,
`tests/agents/ifvg_search/test_presentation_help_registry.py`,
`tests/agents/ifvg_search/test_presentation_labels.py`,
`tests/agents/ifvg_search/test_presentation_results.py`,
`tests/agents/test_ifvg_ui_common_ui3.py`, `tests/agents/test_ifvg_help_scans.py`,
`tests/agents/test_ifvg_context_presentation.py`.

## Tests — modified (2)

`tests/agents/test_ifvg_results_tab.py` (the detail-level labels; three UI-3 tests),
`tests/agents/test_ifvg_pipeline_tab.py` (the ladder dtypes; five UI-3 panel / ladder tests).

## Docs (committed)

`docs/DECISIONS.md` (D-055); `ARCHITECTURE.md`, `docs/README.md`, `docs/pipeline_state.yaml`
as HEAD + the UI-3 lane transforms (`stage_shared_docs.py`).

## Dashboard package (untracked by convention, like every prior release's plan / evidence folder)

`FINAL-IMPLEMENTATION-PLAN-DOCS/FRONTEND_UX_CONTRACT.md` (UI-3 amendment of §§5.1, 5.6 (new), 17,
30.4, 32.1, 35, 36); `implementation-progress/DECISIONS_TAKEN.md` (#138); this folder
(`_PROGRESS_CHECKPOINT.md`, `DEVIATIONS.md`, `FILES_TOUCHED.md`, `TEST_RESULTS.md`,
`COMPLETION_REPORT.md`, `stage_shared_docs.py`, `run_final_gates.sh`, the pytest / Ruff logs,
`junit_full.xml`, the patch and bundle with their `.sha256`).

Untouched by design: Strategy-Core; the fixed M0–M3 lane (`context_reporting.py`,
`context_statistics.py`, the experiment service); every backend contract of HARDENING-BACKEND /
FIX / FIX.1; `study_providers.py`, `study_drafts.py`, `study_presentation.py`, `study_status.py`,
`search/*`; `ifvg_verifier_charts.py`, `ifvg_results_charts.py`; `data/`.
