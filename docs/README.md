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
`tests/agents/ifvg_search/`. Decisions D-039, D-040, D-042, D-043 (D-041,
D-044, D-045 reserved for R3/R5-R6/R4). R1 implementation is complete;
**R1 acceptance is blocked pending the owner-approved verification fixture**
(`VerificationAuthorizationRef`; owner decisions 21/R-5) — no real five-day
slice runs until it exists, and no research interpretation attaches to any
verification output (`verification_only=true`, `full_pipeline_not_run=true`).
