# Alpha Signal Research Lab - Agent Handoff Guide

## Start Here

This repo is not scaffold-only.

The current repo has one primary workflow and one retained secondary path:

- Primary workflow: Streamlit extrema-model training and runtime model generation.
- Secondary workflow: the older 3-class dashboard compatibility/export path kept for ML-Trading-Dashboard.

## Primary Workflow

Start here when you are reasoning about the repo's main purpose:

- UI: `scripts/ml_training_tab.py`
- Shared Streamlit host: `scripts/dashboard.py`
- Data prep: `scripts/process_batch_download.py`
- Core training modules: `src/alpha_lab/agents/data_infra/ml/`
- Runtime consumer: `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py`

The primary workflow builds datasets from local Databento parquet files, evaluates true out-of-sample walk-forward folds, then refits and saves a runtime CatBoost bundle.

## Secondary Compatibility / Export Path

Keep this path working, but do not treat it as the repo's competing main architecture:

- Research and diagnostics: `src/alpha_lab/experiment/`
- Streamlit diagnostics tab: `scripts/experiment_tab.py`
- Canonical downstream export: `scripts/train_dashboard_model.py`
- Canonical artifact contract: `data/models/dashboard_3feature_v1.cbm`

This path exists to feed ML-Trading-Dashboard.

## Working Rules

1. Treat `scripts/ml_training_tab.py` and `src/alpha_lab/agents/data_infra/ml/` as the main path.
2. Treat `src/alpha_lab/experiment/` and `scripts/train_dashboard_model.py` as compatibility/export, not as a second primary workflow.
3. Quality gates in the ML training tab must come from true out-of-sample fold predictions.
4. The final extrema runtime model is refit on all labeled rows after evaluation.
5. Preserve the `data/models/dashboard_3feature_v1.cbm` contract unless a user explicitly asks to change the downstream interface.
6. Generated outputs are not source-of-truth code. `models/`, `catboost_info/`, `*.cbm`, cached parquet/csv files, and scratch chart HTML belong outside normal code review.

## What Is Still True

- The multi-agent architecture still exists in `src/alpha_lab/agents/`.
- The validation, execution, and monitoring surfaces are still part of the repo.
- FastAPI and React dashboard code still lives in `src/alpha_lab/dashboard/` and `dashboard-ui/`.

What changed is priority: those surfaces no longer explain the repo better than the Streamlit extrema workflow does.

## Recommended Reading Order

1. `scripts/ml_training_tab.py`
2. `src/alpha_lab/agents/data_infra/ml/config.py`
3. `src/alpha_lab/agents/data_infra/ml/dataset_builder.py`
4. `src/alpha_lab/agents/data_infra/ml/labeling.py`
5. `src/alpha_lab/agents/data_infra/ml/model_trainer.py`
6. `src/alpha_lab/agents/data_infra/ml/model_evaluator.py`
7. `src/alpha_lab/agents/data_infra/ml/walk_forward.py`
8. `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py`
9. `scripts/train_dashboard_model.py`
10. `src/alpha_lab/experiment/training.py`

## State Tracking

Use these files first when orienting:

- `docs/pipeline_state.yaml`
- `docs/DECISIONS.md`
- `ARCHITECTURE.md`

## Verification Commands

Use Python 3.13 on this machine:

```bash
"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m pytest tests/agents/test_ml_pipeline.py tests/agents/test_ml_features.py tests/agents/test_ml_extrema.py -v
ruff check src/ tests/
```
