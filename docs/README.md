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
`scripts/ifvg_mbp1_panels.py`; R5B.1 replaced the withdrawn
sequence-jump gap rule with the evidence-based coverage policy v2 —
`ifvg/features/mbp1_coverage_{evidence,diagnostic}.py`, block re-resolution
v3, `scripts/ifvg_mbp1_coverage_diagnostic.py`), and R6 (the V1 KMeans regime lane:
`ifvg/ml/regime_{contracts,algorithms,preprocessing,service,alignment,
diagnostics,store}.py`, ML fixtures 2 + 4-KMeans, and the Regime Lane
panel `scripts/ifvg_regime_panels.py` — kmeans_v1 only; the
GMM/minibatch/spectral/Nyström implementations are the post-V1
regime-expansion release), and R6.1 (the regime-lane correction: the
5m/15m context-bar panel materializer + `IFVG_CONTEXT_BAR_PANEL_V1` /
`BP0_CONTEXT_BAR_PANEL`, fold schedules + fold-set artifacts, the verified
observation seam + executor, the descriptive OOS-assignment and fold-local
feature artifacts, the regime study inside the 16-stage pipeline
(`ifvg/ml/regime_study.py`, `ifvg/search/pipeline_regime.py`), verified
owner-decision evidence `ifvg/search/owner_decisions.py` + the
`ifvg_regime_promotion.py` CLI, the five stratification classes
`ifvg/ml/regime_strat*.py`, the bundle-aware CatBoost rung
`ifvg/ml/catboost_bundle_model.py` + D13 comparison rows, the D15 prop-event
detail `alpha_lab/propsim/event_detail.py`, per-fold stability + grain
transition policies, and the regime surfaces of the pipeline tab), and
R6.1-FIX (verified per-fit assignment evidence with an enforced sidecar
schema, fold-feature source refs, the typed `candidate_as_of_missing`
reason, raw thin-regime net-R accounting, the exact label artifact id, the
immutable `executed_trade_tables` store consumed by S02/S14, the typed
sidecar probe with fail-closed prior-stage recoveries, and the
`PipelineWiringError` / MBP-1 scope-equality / enum-copy corrections), and
HARDENING-BACKEND (the semantic store namespace, the immutable supersession
record chain with head witnesses, the liveness-aware owner-decision lock,
the streaming event-detail writer and the external DuckDB event-regime
summary under measured capacity gates, the warnings-as-errors policy, the
sequential-execution truth of the V1 executor, the logical trading-day
calendar with the rebuilt verification-window shortlist, the seed-production
authorization/run contracts, and the bounded-verification preflight and
reports), and HARDENING-BACKEND-FIX (the compact backend correction: token-safe
stale-lock reclamation, atomic recoverable namespace initialization, the public
source-kind boundary, exact regime provenance with native validation, fail-closed
manifests with exact label / executed-trade evidence, central seed
canonicalization, the bounded event-detail partition, and the complete
authority-chain proof at every real seam), and UI-1 (Phase 1 of the
owner-approved UI/UX redesign: the presentation-only run purpose that
derives scope / namespace / evidence / authorization class, the removal
of the namespace selector, `Start` and `Verify Implementation` routes,
charter satisfiability before freeze, typed authorization readiness, the
honest launch outcome, namespace-bound publication, the sequential-V1
runtime truth, direction-aware colorscales and evaluated-gate banners —
`ifvg/presentation/{run_purpose,charter_satisfiability,status_vocabulary}.py`),
and UI-2 (Phase 2: the complete Verification Center — fixture, seed,
final authorization, review / run, monitor — with no spawn seam, the
goal-derived conditional flows, session-only drafts with archive /
restore / typed delete, the explicit reviewer verdicts and the seed CLI
receipt seams — `ifvg/presentation/{flows,review_vocabulary}.py`,
`scripts/ifvg_verification_center.py`)
implementations are complete;
**R1 acceptance is blocked pending the owner-approved verification fixture**
(`VerificationAuthorizationRef`; owner decisions 21/R-5) — no real five-day
slice runs until it exists, and no research interpretation attaches to any
verification output (`verification_only=true`, `full_pipeline_not_run=true`).
