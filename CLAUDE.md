# Alpha Signal Research Lab - Current Repo Context

## Current Purpose

The repo is a Streamlit ML training workbench for NQ futures with two training modes:

1. **Extrema Rebound/Crossing** — binary tick-level classifier (research)
2. **Dashboard Utility** — 3-class level-touch classifier aligned to Trading-Dashboard execution

Both modes share walk-forward evaluation, CatBoost training, and quality gate infrastructure. The mode selector is at the top of the ML Training tab.

Additionally, a retained standalone exporter (`scripts/train_dashboard_model.py`) produces the canonical downstream `.cbm` artifact.

## Primary Path

- `scripts/ml_training_tab.py` — Streamlit UI with mode selector, dataset build, training, evaluation, save
- `scripts/process_batch_download.py` — Databento ZIP to per-date parquet (with ingest validation)
- `src/alpha_lab/agents/data_infra/tick_store.py` — DuckDB query layer with MBP-10 support
- `src/alpha_lab/agents/data_infra/ml/` — all ML pipeline modules (config, builders, labeling, features, trainer, evaluator, walk-forward)
- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` — utility-mode dataset builder
- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_labeling.py` — utility-mode 3-class labeling

Important behavior:

- ML training quality gates use concatenated out-of-sample fold predictions
- RFECV runs once before the walk-forward loop; same features for all folds and final model
- Label purging removes training rows whose forward labeling window crosses into the test period
- Feature cache files are keyed to config hash (training mode + all relevant parameters)
- The saved model is refit on all labeled rows after evaluation
- Saved artifacts include `training_mode` in evaluation.json metadata

## Extrema Runtime Detector (EXPERIMENTAL)

- `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py`
- Marked `_EXPERIMENTAL = True` with runtime warning
- Uses bar-level feature approximations that differ from tick-level training features
- NOT execution-faithful — for research exploration only

## Secondary Path

- `src/alpha_lab/experiment/` — 6-phase experiment pipeline (levels, events, labeling, features, training, diagnostics)
- `scripts/experiment_tab.py` — Streamlit diagnostics tab
- `scripts/train_dashboard_model.py` — canonical 3-feature exporter
- Downstream artifact contract: `data/models/dashboard_3feature_v1.cbm`

## Cross-Repo Contract (Trading-Dashboard)

Trading-Dashboard (`C:\Users\gonza\Documents\Trade-Dashboard`) consumes:
- 3 features: `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`
- 3 classes: tradeable_reversal (0), trap_reversal (1), aggressive_blowthrough (2)
- Resolution ordering: MAE-first (conservative) — aligned in both repos
- Dashboard validates model contract on upload (feature count, class count, feature names)

## Generated Outputs

These are outputs, not source-of-truth code:

- `models/` — saved model bundles
- `catboost_info/` — CatBoost scratch
- `*.cbm` — model files
- cached parquet/csv files under `data/`
- scratch chart HTML outputs

## Read First

1. `ARCHITECTURE.md`
2. `docs/pipeline_state.yaml`
3. `docs/DECISIONS.md`

Then read `scripts/ml_training_tab.py` for the full training flow.
