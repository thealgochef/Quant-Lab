# Alpha Signal Research Lab - Agent Notes

Updated: 2026-09-08.

## Start points

- This repo is a Python ML training workbench for NQ/ES futures (Streamlit workbench + research CLIs). The old live-dashboard prototype (`src/alpha_lab/dashboard/`, `dashboard-ui/`) was deleted in the 2026-07 cleanup — it had no tests, no importers outside three frozen scripts, and pre-v3 session semantics; recover from git history if ever needed.
- Current canonical architecture doc: `ARCHITECTURE.md`.
- Docs index / stale-doc classification: `docs/README.md`.
- Streamlit ML tab workflow: `docs/ML_TRAINING_WORKBENCH.md`.
- Session experiment CLI: `scripts/run_dashboard_session_experiment.py`.
- Strategy-Core v3 cross-repo matrix: `../Strategy-Core/V3_COMPATIBILITY_MATRIX.md`.
- For model-training changes, start with `scripts/ml_training_tab.py` and `src/alpha_lab/agents/data_infra/ml/`, not the older generic multi-agent scaffold.
- IFVG exact-source R5–R6 research: `scripts/ifvg_research_pipeline.py`,
  `src/alpha_lab/agents/data_infra/ifvg/search/research_runs.py`, and
  `src/alpha_lab/agents/data_infra/ifvg/ml/research_evidence.py`.

## Commands

Run from the repo root (`C:\Users\gonza\Documents\Claude-Quant-Lab`, Windows; system Python 3.13 — no venv activation step):

```bash
python -m pytest -q
python -m ruff check src tests scripts    # matches .github/workflows/ci.yml
streamlit run scripts/dashboard.py
```

Choose tests according to the change's actual impact. Do not run the full
regression suite by default, especially for isolated UI or presentation changes.
Run focused tests for the affected behavior and dependencies, plus applicable
lint checks. Run the full suite only when it is necessary to establish correctness
and targeted checks cannot cover the affected behavior. Once relevant checks pass,
do not broaden testing without a concrete unresolved correctness concern.

Project metadata requires Python `>=3.13` (`pyproject.toml`); `.python-version` pins `3.13.1`; CI runs 3.13. Do not hardcode other interpreter versions in docs or scripts.

## Data flow

- Databento batch import: `python scripts/process_batch_download.py <zip>`.
- Imported tick files are expected at `data/databento/{symbol}/{YYYY-MM-DD}/mbp10.parquet`, `mbp1.parquet`, or `trades.parquet`.
- The Streamlit ML tab reads local parquet only; it does not call the Databento API.
- Per-date ML caches are mode-specific:
  - Extrema: `ml_features_{config_hash}.parquet`
  - Dashboard utility: `ml_utility_{config_hash}.parquet`
- The dashboard-utility hash includes Strategy-Core engine/source semantics, so v1/v2/v3 structural changes do not reuse stale caches.

## Current Strategy-Core v3 contract

Dashboard-utility mode is the production-aligned research path and is single-sourced to Strategy-Core v3:

- `engine_version`: `strategy_core_engine_v3`
- Bars/features: trade-price/trade-print, NQ 0.25 grid, default `147t` touch bars.
- Sessions: ET `asia` 19:00→02:45, `london` 03:00→08:00, `ny` 09:00→17:00; 18:00 ET trading-day boundary.
- Levels: PDH/PDL from the full prior trading day; Asia/London levels from current-day sessions.
- Touch guard: level `available_from` is enforced; pre-availability self-touches do not consume zones.
- Label entry: realistic decision-time price at `touch_close + 5m`; not level price at touch.
- Cutoffs: flatten/no-new-decision at 16:40 ET; forward cutoff 17:00 ET.
- Gate: `tradeable_reversal`, `eligible_session="ny"`, confidence `0.70`.
- Research session experiments: train/evaluate all sessions by default, production-gate NY by default; scope is persisted into `evaluation.json`, `metadata.json`, and `strategy.json` as audit metadata.

Trade-Lab consumes Strategy-Core v3 through its adapter seam (repointed in the W1 one-surface migration); batch↔serving per-touch parity is gated by the W3b harness in Trade-Lab plus this repo's contract/no-drift acceptance tests.

## Evaluation rules

- Quality gates must use concatenated out-of-sample walk-forward fold predictions, not predictions from the final refit model.
- RFECV runs once before the walk-forward loop; the selected features must be reused by every fold model and the final saved model.
- Label purging removes training rows whose forward labeling window crosses into the test period.
- The saved runtime model is refit on all labeled rows after evaluation.
- Session experiment filters are applied after broad dataset generation: fold training, OOS metrics, and final refit use the configured training/evaluation scope.
- Preserve CatBoost native missing-value handling; do not blanket `fillna(0.0)` unless intentionally matching a separate runtime approximation path.

## Runtime/output artifacts

A Streamlit save writes:

```text
models/{model_name}/model.cbm
models/{model_name}/metadata.json
models/{model_name}/evaluation.json
models/{model_name}/strategy.json
models/{model_name}/oos_predictions.parquet  # when OOS rows are available
```

Treat `models/`, `catboost_info/`, `*.cbm`, cached `*.parquet`/`*.csv`, local imported data, and scratch chart HTML as generated local outputs unless a task explicitly says otherwise.

Repository ignore rules also cover local IFVG stores, job state, saved profiles,
drafts, review ledgers, and generated report/test workspaces. Ignoring them does
not make them disposable: preserve previous studies, approvals and audit evidence
on disk. Do not use `git clean -X` as repository housekeeping. Keep reusable test
fixtures under `tests/`, outside the ignored local stores. Archived documentation
and approved IFVG contracts remain versioned; local agent permissions do not.

IFVG source-based research also writes immutable generated families under its
selected search store: `research_subjects`, `research_groups`,
`research_approvals`, `research_context_companions`, `research_cohorts`,
`research_labels`, `research_model_inputs`, `research_model_runs`,
`research_regime_executions`, and `research_replay_charts`. Preserve
historical replay/context artifacts and approvals. Each subject binds the exact
saved child and effective configuration sections; profile names and documentation
are not substitutes for those hashes. Targets/costs remain per subject, warmup is
excluded, and model inputs persist before fitting. Inspect these artifacts through
verified readers, including incomplete runs; do not infer success from file
presence. Real launches require the new exact-plan research authorization, which
does not inherit strategy-search scope. Implementation tests/preflight readiness
do not establish a completed real research run.

MBP completeness receipts require an explicit reviewed import; dry-run preview
saves nothing and cannot infer missing completeness. Regime feature eligibility
is a separate, review-ID-bound owner decision over saved assessment gates,
bootstrap evidence and sample floors. It configures a linked B0→B7 study but
grants no fitting until that new research plan is separately authorized.
Linked regime studies preserve the exact precursor numerical protocol. Research
candidate charts use their selected store and separately verified configured
labels; legacy fixed-1R annotations do not define research labels. MBP source
loading stays bounded to one day. Completed real S09a executions may be reused;
incomplete S09a fitting attempts recompute. Real folds purge from the first logical
test-day boundary even if that day has no candidates. Descriptive reports retain
all scoped trades with selection-gate flags; frontier promotion floors stay fixed.

The old `data/models/dashboard_3feature_v1.cbm` exporter is retained compatibility tooling, not automatically a v3 bundle. Canonical bundle location/checksum verification is deferred until the incoming local zip is available.

## Documentation maintenance

IFVG search evaluation corrections (2026-09-08): exclude stamped warmup trades
before metrics/gates and include the initial zero-equity drawdown peak. Current
costed evaluations use `metrics_policy_id=post_warmup_zero_peak_v2`. Legacy
artifacts remain immutable; `scripts/audit_ifvg_search.py` publishes scoped
replacement reports and backs up the operational state before updating its
frontier pointer. Search trade reviews join exact core/v2/input-bar references
and append only to the existing visual review ledger.

When code behavior changes, update docs in the same change. At minimum:

- Strategy/session/feature/label semantics -> `ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/pipeline_state.yaml`, and Strategy-Core docs/matrix.
- UI/workflow changes -> `docs/ML_TRAINING_WORKBENCH.md`.
- Model-bundle/cache output changes -> `ARCHITECTURE.md`, `docs/README.md`, and this file.
