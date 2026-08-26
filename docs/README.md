# Quant-Lab documentation index

Updated: 2026-06-04.

Use this index to avoid treating old reports as current architecture. The current production-aligned path is **dashboard-utility training through Strategy-Core v3**. Bundle presence/checksum verification is deferred until the local data/model zip is available.

## Current / canonical docs

| Doc | Status | Use for |
|---|---|---|
| `../ARCHITECTURE.md` | **Canonical current architecture** | Repo purpose, v3 semantics, current workflows, generated outputs. |
| `ML_TRAINING_WORKBENCH.md` | **Current workflow guide** | Streamlit ML tab, dashboard-utility build/train/save path. |
| `pipeline_state.yaml` | **Current machine-readable summary** | Quick state for agents/scripts; v3 fields and known gaps. |
| `DECISIONS.md` | **Decision log** | Architectural decisions and why they changed. |
| `../../Strategy-Core/README.md` | **Shared engine truth** | Strategy-Core v3 constants/semantics/tests. |
| `../../Strategy-Core/V3_COMPATIBILITY_MATRIX.md` | **Cross-repo matrix** | Quant-Lab / Strategy-Core / Trade-Lab compatibility state. |
| `../../Strategy-Core/MIGRATION.md` | **Remaining migration work** | Trade-Lab repoint sequence and deferred bundle checks. |

## Current source files that define behavior

| Path | Why it matters |
|---|---|
| `../scripts/ml_training_tab.py` | Orchestrates dataset build, walk-forward training, model save, and `strategy.json` emission. |
| `../scripts/run_dashboard_session_experiment.py` | Non-Streamlit CLI for Databento-backed dashboard-utility session experiments. |
| `../src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` | Builds dashboard-utility datasets from local parquet; feeds Strategy-Core. |
| `../src/alpha_lab/agents/data_infra/ml/engine_decision.py` | Adapter into Strategy-Core v3 decisions/features/outcomes. |
| `../src/alpha_lab/agents/data_infra/ml/strategy_contract.py` | Emits Strategy-Core-stamped `strategy.json`. |
| `../src/alpha_lab/agents/data_infra/ml/config.py` | Config, session-experiment presets, and cache hash; includes engine version in utility cache identity. |
| `../src/alpha_lab/agents/data_infra/tick_store.py` | DuckDB/parquet data access. |

## Pruned stale docs

Older historical reports and scaffold prompt documents were removed from the working tree to reduce stale-doc noise. Current project state should be reconstructed from the canonical docs above plus code/tests, not from old phase reports.

Pruned families:

- old knowledge-transfer handoff notes
- old backtest-finding summaries
- the old levels-probe report
- standalone agent prompt scaffold docs
- the duplicate architecture redirect that used to live in this docs directory

If old audit context is needed, recover it from Git history and verify against current code before citing it.

## Retained compatibility/export tooling

The old 3-feature exporter still exists:

```text
../scripts/train_dashboard_model.py -> ../data/models/dashboard_3feature_v1.cbm
```

Treat it as retained compatibility tooling, not the canonical Strategy-Core v3 bundle path. A model is v3-compatible only if its `strategy.json` validates against `strategy_core_engine_v3` and bundle files/checksums are verified.

## Documentation maintenance rule

When code behavior changes, update the matching docs in the same change:

- Strategy/session/label/feature semantics -> `../ARCHITECTURE.md`, this index, `ML_TRAINING_WORKBENCH.md`, and Strategy-Core docs/matrix if cross-repo.
- Training UI/workflow changes -> `ML_TRAINING_WORKBENCH.md`.
- Cache/model-bundle output changes -> `../ARCHITECTURE.md`, `pipeline_state.yaml`, and `AGENTS.md` / `CLAUDE.md` if agent start points change.
- Do not recreate historical reports as current-state docs. If old audit context is needed, extract only the still-valid lessons into the canonical docs after code/test verification.

## IFVG robust FSM configuration search & prop realization lane

`ifvg_prop_robust_config_search_v1` (additive; zero Strategy-Core changes in
v1). Authority: the approved plan package under
`QL-FSM-PROP-SEARCH-DASHBOARD/FINAL-IMPLEMENTATION-PLAN-DOCS/`. Code:
`src/alpha_lab/agents/data_infra/ifvg/{search,study,features}/`; suites:
`tests/agents/ifvg_search/`. Decisions D-039 through D-046 (D-044's
supervised-ladder half and D-045's pipeline half landed with R5; the
MBP-1 activation D-046 landed with R5B; D-044's KMeans regime half lands
with R6). R1, R2 (multi-child search, lineage, exact deltas, verifier
integration, `scripts/ifvg_search_job.py`), R3 (prop lifecycle:
fidelity-first trade paths, typed calendars, field-level contract
evidence, the full account walk, portfolio/stress/simulation identities,
and prop-gate/worst-firm frontier wiring — `alpha_lab.propsim` lifecycle
modules + `tests/propsim/`), and R4 (trader workspace UI: the Experiments
sub-navigation, eight-step five-mode wizard with disk drafts, registry-
gated launch, Active Runs monitor, Results/History/comparison/insights/
account-timeline surfaces — `scripts/ifvg_study_*.py`,
`ifvg_active_runs_tab.py`, `ifvg_results_*.py`, `ifvg_ui_common.py` +
`ifvg/study_{status,presentation,drafts,providers}.py`), and R5 (pipeline
runner + MBP-1 contract readiness + supervised model ladder:
`ifvg/search/pipeline.py` with the semantic/attempt split and 16-stage
executors, `ifvg/ml/` ladder + decision/calibration registries + core
drift builders, `ifvg/features/bundle_feature_view.py`, the real
baseline-verification executors in `ifvg/search/executors.py`, and the
Full Pipeline Run surface `scripts/ifvg_pipeline_tab.py` +
`ifvg_pipeline_job.py`), and R5B (offline MBP-1 feature activation,
research-only: `ifvg/features/mbp1_{arrow_schemas,source_artifact,
stage_windows,feature_materializer,coverage,feature_join}.py`, the
versioned `IFVG_ORDER_FLOW_MBP1_V1` activation in `feature_blocks.py`,
the controlled Baseline vs Baseline+MBP-1 study
`ifvg/ml/controlled_feature_study.py`, and the MBP-1 dashboard panels
`scripts/ifvg_mbp1_panels.py`) implementations are complete;
**R1 acceptance is blocked pending the owner-approved verification fixture**
(`VerificationAuthorizationRef`; owner decisions 21/R-5) — no real five-day
slice runs until it exists, and no research interpretation attaches to any
verification output (`verification_only=true`, `full_pipeline_not_run=true`).
