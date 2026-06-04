# Alpha Signal Research Lab - Agent Notes

## Start Points
- This repo is a Python ML training workbench for NQ futures plus a dashboard runtime; the primary training path is `scripts/dashboard.py` -> `scripts/ml_training_tab.py` -> `src/alpha_lab/agents/data_infra/ml/`.
- High-level documentation for the Streamlit ML tab lives at `docs/ML_TRAINING_WORKBENCH.md`.
- The older multi-agent scaffold under `src/alpha_lab/agents/` still exists, but for model-training changes start with `scripts/ml_training_tab.py`, not the generic agent classes.
- Training modes live in `MLPipelineConfig.training_mode`: `extrema_rebound_crossing` is the binary tick-extrema research mode; `dashboard_utility` is the 3-class level-touch model aligned to Trading-Dashboard.

## Commands
- Use Python 3.13 at `"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe"`; dependencies are installed there even though `pyproject.toml` allows `>=3.11`.
- Install backend deps if needed with `"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m pip install -e ".[dev]"`.
- Run all backend tests with `"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m pytest tests/ -v`; collection is 669 tests as of 2026-04-27.
- Run focused backend tests with normal pytest node ids, for example `"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m pytest tests/agents/test_ml_pipeline.py::TestModelTrainer::test_basic_training -v`.
- Backend lint/typecheck commands are `"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m ruff check src tests scripts` and `"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m mypy src`.
- Run the Streamlit workbench with `streamlit run scripts/dashboard.py`.
- React dashboard commands run from `dashboard-ui/`: `npm install` if `node_modules/` is missing, `npm run dev`, `npm run build`, and `npm test`.
- The React Vite dev server uses port 3000 and proxies `/api` plus `/ws` to `localhost:8000`; start the FastAPI backend with `PYTHONPATH=src "/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m alpha_lab.dashboard.api`.

## Data Flow
- Databento batch import is `"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" scripts/process_batch_download.py <zip>`; without an argument it auto-finds the newest `GLBX-*.zip` in Downloads.
- Imported tick files are expected at `data/databento/{symbol}/{YYYY-MM-DD}/mbp10.parquet`, `mbp1.parquet`, or `trades.parquet`; the Streamlit UI currently offers `NQ`/`ES`, reads local Parquet, and does not call the Databento API.
- Per-date ML caches are mode-specific: extrema uses `ml_features_{config_hash}.parquet`; dashboard-utility uses `ml_utility_{config_hash}.parquet`. The hash includes training mode and dataset-relevant config, so config changes create separate cache files.
- The ML-tab dashboard-utility builder is self-contained from local tick Parquet; retained experiment/export scripts still use `data/experiment/events.parquet` and `data/experiment/feature_matrix.parquet`.

## Evaluation Rules
- Quality gates must use concatenated out-of-sample walk-forward fold predictions, not predictions from the final refit model.
- RFECV runs once before the walk-forward loop; the selected features must be reused by every fold model and the final saved model.
- Label purging removes training rows whose forward labeling window crosses into the test period.
- The saved runtime model is refit on all labeled rows after evaluation.
- Preserve CatBoost's native missing-value handling in training; do not blanket `fillna(0.0)` unless you are intentionally matching a runtime approximation path.

## Dashboard Contract
- Keep the retained compatibility path working: `src/alpha_lab/experiment/`, `scripts/experiment_tab.py`, and `scripts/train_dashboard_model.py`.
- The canonical downstream artifact is `data/models/dashboard_3feature_v1.cbm`; the FastAPI dashboard auto-load prefers this file from `data/models/`.
- Trading-Dashboard at `C:\Users\gonza\Documents\Trade-Dashboard` consumes exactly these features in order: `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`.
- The 3 dashboard classes are `tradeable_reversal` (0), `trap_reversal` (1), and `aggressive_blowthrough` (2); resolution ordering is MAE-first.
- `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py` is explicitly experimental and has a known train/serve domain mismatch; do not treat it as production-ready.

## Runtime And Outputs
- Root `.env.example` documents `POLYGON_API_KEY` and `DATABENTO_API_KEY`; dashboard live settings use `DASHBOARD_`-prefixed env vars from `src/alpha_lab/dashboard/config/settings.py`, such as `DASHBOARD_DATABENTO_API_KEY`, `DASHBOARD_DATA_SOURCE`, `DASHBOARD_DATABASE_URL`, and `DASHBOARD_MODEL_DIR`.
- The live dashboard defaults to Databento and will continue without PostgreSQL persistence if DB init fails, but Databento streaming requires `DASHBOARD_DATABENTO_API_KEY`.
- Treat `models/`, `catboost_info/`, `*.cbm`, cached `*.parquet`/`*.csv`, `data/raw/`, `data/processed/`, and scratch chart HTML files as generated local outputs unless a task explicitly says otherwise.
- If architecture is unclear, read `ARCHITECTURE.md`, `docs/pipeline_state.yaml`, and `docs/DECISIONS.md`; they are concise and aligned with the current dual-mode training workflow.
