# Architecture - Alpha Signal Research Lab

## Repo Purpose Now

Alpha Signal Research Lab is a futures research repo with one primary workflow and one retained secondary compatibility path.

- Primary workflow: Streamlit extrema-model training and runtime model bundle generation.
- Secondary workflow: retained 3-class dashboard compatibility/export tooling for ML-Trading-Dashboard.

The multi-agent architecture still exists, but it is no longer the clearest first explanation of the repo.

## Primary Workflow

The main application path is the Streamlit extrema workflow:

1. `scripts/process_batch_download.py`
   Processes downloaded Databento files into the local parquet layout used by the repo.
2. `scripts/ml_training_tab.py`
   Main training UI for dataset build, walk-forward evaluation, and model save.
3. `src/alpha_lab/agents/data_infra/tick_store.py`
   Local parquet query layer for tick data.
4. `src/alpha_lab/agents/data_infra/ml/dataset_builder.py`
   Builds labeled extrema datasets.
5. `src/alpha_lab/agents/data_infra/ml/extrema_detection.py`
   Detects extrema candidates.
6. `src/alpha_lab/agents/data_infra/ml/labeling.py`
   Assigns rebound or crossing labels.
7. `src/alpha_lab/agents/data_infra/ml/features_*.py`
   Extracts PL, MS, and optional signal features.
8. `src/alpha_lab/agents/data_infra/ml/walk_forward.py`
   Creates walk-forward splits.
9. `src/alpha_lab/agents/data_infra/ml/model_trainer.py`
   Runs optional RFECV and fits the final runtime CatBoost model on the full labeled set.
10. `src/alpha_lab/agents/data_infra/ml/model_evaluator.py`
    Scores true out-of-sample fold predictions and computes quality gates.
11. `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py`
    Loads saved runtime model bundles for downstream signal generation.

### Primary Workflow Contract

- The walk-forward metrics shown in the training UI are based on true out-of-sample fold predictions.
- The saved runtime model is refit on all labeled rows after evaluation.
- Saved runtime bundles live under `models/<run>/` and contain `model.cbm`, `metadata.json`, and `evaluation.json`.

## Secondary Compatibility / Export Path

The older 3-class path remains in the repo for downstream dashboard consumption:

- `src/alpha_lab/experiment/`
  Self-contained research and diagnostics pipeline for the retained 3-class model world.
- `scripts/experiment_tab.py`
  Streamlit diagnostics and review surface for those results.
- `scripts/train_dashboard_model.py`
  Canonical exporter for the downstream dashboard model artifact.

### Secondary Workflow Contract

- Canonical downstream artifact: `data/models/dashboard_3feature_v1.cbm`
- Purpose: feed ML-Trading-Dashboard
- Status: supported, but secondary

## Shared and Supporting Surfaces

- `scripts/dashboard.py`
  Hosts both the primary ML training tab and the secondary dashboard compatibility tab.
- `src/alpha_lab/dashboard/`
  FastAPI runtime and related backend surfaces.
- `dashboard-ui/`
  React frontend assets.
- `src/alpha_lab/agents/`
  Broader multi-agent research architecture, still present and still relevant for runtime and orchestration.

## Generated Outputs vs Source Code

Generated outputs should not be treated as source-of-truth code:

- `models/`
- `catboost_info/`
- `*.cbm`
- cached parquet/csv outputs under `data/`
- scratch chart HTML files

Source-of-truth code lives in `scripts/`, `src/`, `tests/`, `config/`, and current docs.

## Recommended Read Order

1. `scripts/ml_training_tab.py`
2. `src/alpha_lab/agents/data_infra/ml/`
3. `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py`
4. `scripts/train_dashboard_model.py`
5. `src/alpha_lab/experiment/`
