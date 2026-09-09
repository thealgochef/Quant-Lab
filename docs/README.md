# Quant-Lab documentation index

Updated: 2026-09-08.

Use this index to avoid treating old reports as current architecture. The current production-aligned path is **dashboard-utility training through Strategy-Core v3**. Bundle presence/checksum verification is deferred until the local data/model zip is available.

## Current / canonical docs

| Doc | Status | Use for |
|---|---|---|
| `../ARCHITECTURE.md` | **Canonical current architecture** | Repo purpose, v3 semantics, current workflows, generated outputs. |
| `ML_TRAINING_WORKBENCH.md` | **Current workflow guide** | Streamlit ML tab, dashboard-utility build/train/save, exact-source IFVG R5–R6 research and separate authorization. |
| `pipeline_state.yaml` | **Current machine-readable summary** | Quick state for agents/scripts; v3 fields and known gaps. |
| [Saved strategy-search approval](IFVG_STRATEGY_SEARCH_APPROVAL.md) | **Current approval and execution contract** | Enable Run for an exact approved strategy-only configuration without launching it. |
| [Focused IFVG workspace report](../QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/UI-UX-REDESIGN-PLAN/FOCUSED_WORKSPACE/COMPLETION_REPORT.md) | **Current presentation implementation** | Page routing, My studies, startup Developer boundary, Replay selection and test evidence. |
| `reports/ifvg_browser_acceptance/20260907/DESKTOP_FLOW_REPORT.md` (local repository root) | **Saved local browser evidence** | Historical desktop acceptance report; generated reports are ignored and are not shipped with a clean clone. |
| `DECISIONS.md` | **Decision log** | Architectural decisions and why they changed. |
| `IFVG_CONTEXT_CAPTURE_V3.md` | **Current measurement contract** | Separate IFVG v3 context tables, fixed nonsealed access, immutable identity, and future M0-M3 inputs. |
| `../../Strategy-Core/README.md` | **Shared engine guide** | Code pointers for Strategy-Core contracts, mechanics and tests. Verify the checkout used by the run. |
| `../../Strategy-Core/V3_COMPATIBILITY_MATRIX.md` | **Historical compatibility matrix** | Recorded migration evidence; current compatibility requires code and bundle verification. |
| `../../Strategy-Core/MIGRATION.md` | **Historical migration checklist** | Migration rationale and recorded checks; not current Trade-Lab certification. |

## Current source files that define behavior

| Path | Why it matters |
|---|---|
| `../scripts/dashboard.py` | Executes the selected page: IFVG Lab, ML Training, Dashboard Compatibility, Strategy Analysis, or startup-enabled Developer. |
| `../scripts/ifvg_workspace.py` and `../scripts/ifvg_research_*.py` | Current research-facing IFVG workflow; replaces the former top-level IFVG tabs. |
| `../src/alpha_lab/agents/data_infra/ifvg/search/research_runs.py` | Exact-source research preflight, new namespace-bound approval, group lifecycle and registered worker dispatch. |
| `../src/alpha_lab/agents/data_infra/ifvg/search/research_regimes.py` | Read-only eligibility review and explicit evidence-bound owner decision for linked B0→B7 feature research. |
| `../src/alpha_lab/agents/data_infra/ifvg/search/research_subject.py`, `research_data.py`, `research_mbp1.py` | Saved effective configuration binding, source/context preparation and neutrality, configured-R labels and MBP-1 source evidence. |
| `../src/alpha_lab/agents/data_infra/ifvg/search/research_artifacts.py` and `../src/alpha_lab/agents/data_infra/ifvg/ml/research_evidence.py` | Scoped cohorts/labels and durable inputs, models, OOS predictions and verified reload without fitting. |
| `../src/alpha_lab/agents/data_infra/ifvg/ml/regime_execution_cache.py` | Opt-in completed real S09a KMeans/bootstrap/assignment reuse; incomplete attempts recompute. |
| `../src/alpha_lab/agents/data_infra/ifvg/replay_chart_store.py` | Exact custom-store research charts/forward-source verification and scoped geometry; configured labels are separate annotations. |
| `../src/alpha_lab/agents/data_infra/ifvg/presentation/workspace_mode.py` | Startup-only Developer flag and route-scoped technical presentation. |
| `../scripts/ml_training_tab.py` | Orchestrates dataset build, walk-forward training, model save, and `strategy.json` emission. |
| `../scripts/run_dashboard_session_experiment.py` | Non-Streamlit CLI for Databento-backed dashboard-utility session experiments. |
| `../src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` | Builds dashboard-utility datasets from local parquet; feeds Strategy-Core. |
| `../src/alpha_lab/agents/data_infra/ml/engine_decision.py` | Adapter into Strategy-Core v3 decisions/features/outcomes. |
| `../src/alpha_lab/agents/data_infra/ml/strategy_contract.py` | Emits Strategy-Core-stamped `strategy.json`. |
| `../src/alpha_lab/agents/data_infra/ml/config.py` | Config, session-experiment presets, and cache hash; includes engine version in utility cache identity. |
| `../src/alpha_lab/agents/data_infra/tick_store.py` | DuckDB/parquet data access. |
| `../src/alpha_lab/agents/data_infra/ifvg/context_contracts.py` | Separate normalized v3 table contracts and exact-link validation. |
| `../src/alpha_lab/agents/data_infra/ifvg/verification.py` | Fixed allowlist, one-replay verification, reports, and immutable v3 save. |

## Completed study audits

The exact-source R5–R6 workflow adds generated artifact families
`research_subjects`, `research_groups`, `research_approvals`,
`research_context_companions`, `research_cohorts`, `research_labels`,
`research_model_inputs`, `research_model_runs`, `research_regime_executions` and
`research_replay_charts` under the selected search store. Research chart discovery
uses that store's `research_replay_chart_catalog.json`.
Existing replay/context artifacts and prior approvals remain immutable. Current
source code and exact verified manifests establish behavior and saved evidence;
documentation or old implementation/verification reports alone do not establish
completion of a real feature/model research run.

The local `reports/parent_staleness_audit/20260908/REVIEW.md` (repository root)
records the completed study's corrected evaluation and UI trade reviews.
Its `files.jsonl`, `json_fields.jsonl`, `parquet_columns.jsonl` and `trades.jsonl`
provide exhaustive file, JSON-field, column and execution check inventories.
Search evaluation corrections use new immutable costed-evaluation IDs under
`post_warmup_zero_peak_v2`; original run artifacts remain forensic evidence.

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

## IFVG implementation and retained plan package

The current research UI is `scripts/ifvg_workspace.py` and the
`ifvg_research_*.py` pages. The research runner is registered through
`search/runner_registry.py`; `search/research_runs.py` freezes exact subjects,
checks sources and authorization, and coordinates per-subject execution. The
S00–S10/S14/S15 preset uses the existing R5 ladder, controlled R5B comparisons and
fold-local R6 stages. Read the workbench guide for supported options and blockers.
S11 model-guided replay, full prop realization and combined MBP/regime features
have separate requirements; research-run completion does not establish their
availability or scientific improvement.

The approved package in
`QL-FSM-PROP-SEARCH-DASHBOARD/FINAL-IMPLEMENTATION-PLAN-DOCS/` is retained as
versioned design and decision history. It contains contracts cited in code/tests;
its authority records and checksums must not be silently rewritten to make an
old plan look current. The phase reports under `implementation-progress/` record
their revision's implementation and verification evidence, not present launch
readiness. The kickoff-prompt link in the package README and the retired UI-map
citation in its FSM plan are historical references. Use the current workbench
instead of those earlier UI instructions.

`IFVG_PLUGIN_DESIGN_ict_amended_ml_research_revised.md` is also retained as
historical design and owner-ruling provenance. It is not a current behavior
guide. Current context measurements are documented in `IFVG_CONTEXT_CAPTURE_V3.md`
and `ifvg/IFVG_CONTEXT_FORMULA_V2_CONTRACT.md` and implemented by the Core observer
and Quant-Lab companion readers.

## Local artifacts and repository housekeeping

`.gitignore` excludes IFVG data stores, job state, user profiles, drafts, review
ledgers, search outputs and generated reports/test workspaces. These are retained
on the local machine; ignoring them does not archive or back them up. In
particular, keep completed studies, approvals, market data and audit evidence.
Do not use `git clean -X` to remove repository noise. Reusable source fixtures
belong under `tests/`, and curated documents remain trackable.

The September cleanup removes the two completed root Codex prompts, the duplicate
UI-1 root patch (the original remains in the UI-1 evidence package), and the
obsolete `docs/ifvg/IFVG_LAB_UI_RESTORATION_MAP.md`. The current workbench replaces
the old three-tab UI map. Historical audit archives, governing contracts and
owner-decision records remain intact. The local agent entry point `CLAUDE.md`
now points to `AGENTS.md` rather than repeating stale engine/workflow claims.
