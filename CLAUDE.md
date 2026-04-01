# Alpha Signal Research Lab - Current Repo Context

## Current Purpose

The repo's primary purpose is a Streamlit extrema-model training workbench:

- build extrema datasets from local Databento parquet files
- run out-of-sample walk-forward evaluation
- refit and save runtime CatBoost model bundles
- consume those bundles through the ML extrema detector path

The older 3-class dashboard model path is still supported, but it is now a secondary compatibility/export surface for ML-Trading-Dashboard.

## Primary Path

- `scripts/ml_training_tab.py`
- `scripts/process_batch_download.py`
- `src/alpha_lab/agents/data_infra/tick_store.py`
- `src/alpha_lab/agents/data_infra/ml/`
- `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py`

Important behavior:

- ML training quality gates must use concatenated out-of-sample fold predictions.
- The saved extrema runtime model is refit on all labeled rows after evaluation.

## Secondary Path

- `src/alpha_lab/experiment/`
- `scripts/experiment_tab.py`
- `scripts/train_dashboard_model.py`
- downstream artifact contract: `data/models/dashboard_3feature_v1.cbm`

Keep this path stable for downstream consumers, but do not treat it as the repo's main story.

## Generated Outputs

These are outputs, not source-of-truth code:

- `models/`
- `catboost_info/`
- `*.cbm`
- cached parquet/csv files under `data/`
- scratch chart HTML outputs

## Read First

1. `ARCHITECTURE.md`
2. `docs/pipeline_state.yaml`
3. `docs/DECISIONS.md`

Then read the primary workflow entrypoints before diving into the broader multi-agent code.
