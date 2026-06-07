# Alpha Signal Research Lab - Current Repo Context

Updated: 2026-06-04.

## Current purpose

Quant-Lab is a local ML training workbench for NQ/ES futures. It has two active training modes:

1. **Extrema Rebound/Crossing** — binary tick-extrema research classifier.
2. **Dashboard Utility** — 3-class level-touch classifier built through **Strategy-Core v3**.

The dashboard-utility path is the production-aligned research path, but Trade-Lab is not yet v3-compatible. Do not call a v3 model live/paper-ready until Trade-Lab is repointed and end-to-end parity is proven.

## Read first

1. `docs/README.md` — docs inventory and stale/historical classification.
2. `ARCHITECTURE.md` — current repo architecture and v3 semantics.
3. `docs/ML_TRAINING_WORKBENCH.md` — Streamlit ML tab workflow.
4. `docs/pipeline_state.yaml` — machine-readable current state.
5. `../Strategy-Core/README.md` and `../Strategy-Core/V3_COMPATIBILITY_MATRIX.md` — shared engine truth and cross-repo gaps.

## Primary path

- `scripts/ml_training_tab.py` — Streamlit UI with mode selector, dataset build, training, evaluation, save.
- `scripts/run_dashboard_session_experiment.py` — CLI runner for repeatable dashboard-utility session-scope experiments.
- `scripts/process_batch_download.py` — Databento ZIP to per-date parquet.
- `src/alpha_lab/agents/data_infra/tick_store.py` — DuckDB query layer.
- `src/alpha_lab/agents/data_infra/ml/config.py` — config models and dataset cache hash.
- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` — utility dataset builder.
- `src/alpha_lab/agents/data_infra/ml/engine_decision.py` — Strategy-Core v3 adapter.
- `src/alpha_lab/agents/data_infra/ml/strategy_contract.py` — emits `strategy.json` from Strategy-Core constants.
- `src/alpha_lab/agents/data_infra/ml/model_trainer.py` / `model_evaluator.py` / `walk_forward.py` — shared training and evaluation infrastructure.

## Strategy-Core v3 dashboard-utility contract

- `engine_version`: `strategy_core_engine_v3`.
- Bars/features: trade-price/trade-print on 0.25 grid; default `147t` touch bars.
- Sessions: ET `asia` 19:00→02:45, `london` 03:00→08:00, `ny` 09:00→17:00; 18:00 ET boundary.
- Levels: PDH/PDL from full prior trading day; Asia/London current-session levels.
- Availability guard: enforced before touches can consume a zone.
- Touches: merged zones, bar-range intersection, first touch per zone/trading day.
- Labels: 3 classes (`tradeable_reversal`, `trap_reversal`, `aggressive_blowthrough`), MAE-first, default TP 15 / SL 30 / trap MFE 5.
- Entry: realistic decision-time trade price at `touch_close + 5m`.
- Cutoffs: flatten/no-new-decision at 16:40 ET; forward cutoff 17:00 ET.
- Gate: `tradeable_reversal`, `eligible_session="ny"`, confidence `0.70`.
- Session experiments: train/evaluate all sessions by default, production-gate NY by default; persisted as audit metadata in saved bundles/contracts.

## Secondary / historical paths

- `src/alpha_lab/experiment/`, `scripts/experiment_tab.py`, and `scripts/train_dashboard_model.py` are retained compatibility/export tooling.
- `data/models/dashboard_3feature_v1.cbm` is a legacy 3-feature artifact path, not automatically a v3 bundle.
- Older historical reports and scaffold prompt documents were pruned from the working tree. Use Git history only for audit context, not current implementation truth.

## Generated outputs

These are outputs, not source-of-truth code/docs:

- `models/`
- `catboost_info/`
- `*.cbm`
- cached parquet/csv files under `data/`
- imported local Databento parquet
- scratch chart HTML outputs

## Documentation rule

When behavior changes, update docs in the same change. Strategy/session/label/feature changes must also update Strategy-Core docs or the v3 compatibility matrix if they affect cross-repo semantics.
