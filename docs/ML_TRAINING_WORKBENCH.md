# Streamlit ML Training Workbench

The ML Training Workbench is the primary in-repo UI for building local ML datasets, running walk-forward CatBoost evaluation, and saving runtime model bundles. It is mounted by `scripts/dashboard.py` in the `🧠 ML Training` tab and rendered by `render_ml_training_tab()` in `scripts/ml_training_tab.py`.

The workbench is local-data only: it discovers and reads Databento Parquet files from disk through `TickStore`. It does not call the Databento API.

## Code map

- `scripts/dashboard.py` - creates the Streamlit tab and calls `render_ml_training_tab()`.
- `scripts/ml_training_tab.py` - UI state, configuration assembly, dataset build orchestration, walk-forward training, metrics display, and model save.
- `src/alpha_lab/agents/data_infra/tick_store.py` - DuckDB-backed reader over `data/databento/{symbol}/{date}/` Parquet files.
- `src/alpha_lab/agents/data_infra/ml/config.py` - shared Pydantic config objects (`MLPipelineConfig` and sub-configs).
- `src/alpha_lab/agents/data_infra/ml/dataset_builder.py` - binary extrema dataset builder.
- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` - 3-class dashboard-utility dataset builder.
- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_labeling.py` - dashboard-utility label semantics.
- `src/alpha_lab/agents/data_infra/ml/walk_forward.py` - rolling/expanding time-series split generation.
- `src/alpha_lab/agents/data_infra/ml/model_trainer.py` - CatBoost training, optional RFECV, model save/load.
- `src/alpha_lab/agents/data_infra/ml/model_evaluator.py` - OOS aggregate metrics, confidence intervals, permutation tests, threshold and calibration summaries.
- `src/alpha_lab/experiment/` - retained compatibility source used only where needed by the utility approach-feature query and older dashboard exporter.

## Configuration flow

All major UI selections are copied into `MLPipelineConfig` before dataset build or training:

- `training_mode`: `extrema_rebound_crossing` or `dashboard_utility`.
- `instrument` / `tick_size`: selected symbol (`NQ` or `ES`) and tick size (`0.25`).
- `walk_forward`: train window, test window, gap.
- `model`: CatBoost iterations, depth, loss, and RFECV setting.
- `dashboard_utility`: TP/SL, bar type, interaction window, approach-feature settings.
- `features`: extrema feature settings; the UI currently disables signal-detector features for extrema builds.

`MLPipelineConfig.dataset_config_hash()` hashes settings that affect dataset generation. Build caches are keyed by this hash so mode/config changes create separate cached Parquet files instead of reusing stale features. Walk-forward and CatBoost settings do not affect dataset cache identity.

The Streamlit tab also stores a UI-level dataset identity in session state. When mode, dates, label, walk-forward windows, or utility-specific settings change, it clears stale `ml_dataset`, `ml_build_config`, `ml_training_result`, and `ml_train_config` values.

## Training modes

### Extrema Rebound/Crossing

Purpose: binary research classifier over tick-level extrema.

- Data source: local tick Parquet under `data/databento/{symbol}/{YYYY-MM-DD}/`.
- Builder: `ExtremaDatasetBuilder` via `build_training_dataset()`.
- Candidate events: extrema detected from tick prices using `ExtremaConfig`.
- Labels: rebound/crossing labels from `labeling.py` with UI-selected threshold:
  - `label_20t`: 20 ticks / 5 points
  - `label_40t`: 40 ticks / 10 points
  - `label_60t`: 60 ticks / 15 points
- Features: `pl_*` price-level microstructure and `ms_*` momentum columns; `sig_*` is supported by the lower-level builder but disabled by this UI path.
- Training loss: CatBoost binary `Logloss`.
- Positive class for evaluation: rebound (`1`).

### Dashboard Utility (3-class)

Purpose: level-touch model aligned to Trading-Dashboard execution semantics.

- Data source: same local tick Parquet layout. This builder is self-contained and does not require prebuilt `data/experiment/events.parquet` for the workbench path.
- Builder: `build_utility_dataset()`.
- Candidate events: first touches of merged key-level zones from prior/current session levels (`PDH`, `PDL`, Asia high/low, London high/low).
- Bars: configured as `987t`, `2000t`, `147t`, or `1m`; tick bars are preferred for touch detection fidelity.
- Labels: `label_encoded` from `dashboard_utility_labeling.py`:
  - `0`: `tradeable_reversal`
  - `1`: `trap_reversal`
  - `2`: `aggressive_blowthrough`
- Resolution ordering: MAE is checked before MFE on each forward bar, matching the dashboard contract.
- Features: canonical live interaction features `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`; optionally live-computable `app_*` approach features.
- Training loss: CatBoost `MultiClass`.
- Positive class for binary quality metrics: `tradeable_reversal` (`0`), converted to positive during OOS evaluation.

The retained compatibility/export path (`src/alpha_lab/experiment/`, `scripts/experiment_tab.py`, `scripts/train_dashboard_model.py`) still exists for the canonical downstream dashboard artifact. The workbench utility builder is the newer self-contained training path inside the ML tab.

## UI sections

### Data/config section

The top of the tab controls:

- Symbol and data directory. Default data root is `data/databento`.
- Date range. Available dates are auto-detected from local folders containing `mbp10.parquet`, `mbp1.parquet`, or `trades.parquet`.
- Walk-forward windows: train days, test days, and gap days. The UI estimates fold count from the detected data span.
- CatBoost settings: iterations and tree depth.
- RFECV:
  - Extrema mode uses the explicit `RFECV feature selection` checkbox.
  - Utility mode enables RFECV when approach features are included; the generic checkbox is not used for that mode.
- Mode-specific settings:
  - Extrema: label threshold.
  - Utility: TP, SL, bar type, interaction window, approach-feature toggle/window.

### Step 1: Local Data

Step 1 scans the selected data directory for local Databento-derived Parquet files:

```text
data/databento/{symbol}/{YYYY-MM-DD}/mbp10.parquet
data/databento/{symbol}/{YYYY-MM-DD}/mbp1.parquet
data/databento/{symbol}/{YYYY-MM-DD}/trades.parquet
```

`TickStore` resolves files in priority order: `mbp10` > `mbp1` > `trades`. If no local data is found, the UI points the user to Databento batch download processing via `scripts/process_batch_download.py`.

This step never downloads data. It only checks the filesystem.

### Step 2: Build Dataset

Step 2 builds one labeled row per model event and keeps the result in Streamlit session state as `ml_dataset` with its build config as `ml_build_config`.

Extrema mode:

1. For each selected date, register that date in a fresh `TickStore`.
2. Query tick feature rows with `query_tick_feature_rows()`.
3. Detect extrema, label them, and extract `pl_*` / `ms_*` features.
4. Cache the date result as `ml_features_{config_hash}.parquet`.

Utility mode:

1. For each selected date, build the configured tick/time bars from local ticks.
2. Compute key levels and merge nearby levels into zones.
3. Detect first touches and label each touch with TP/SL utility semantics.
4. Compute the three `int_*` interaction features and optional live `app_*` approach features.
5. Cache the date result as `ml_utility_{config_hash}.parquet`.

The `Clear Cache` button deletes `ml_features_*.parquet` files for selected dates. It is primarily aimed at extrema caches; utility caches use the `ml_utility_*.parquet` prefix and are separated by config hash.

The dataset preview and counters are mode-specific:

- Extrema: total extrema, labeled rows, rebound/crossing counts, feature preview.
- Utility: touch events, three class counts, interaction/approach feature counts, feature preview.

### Step 3: Train Model

Training is coordinated by `run_walk_forward_training()`:

1. Select feature columns by mode (`pl_`/`ms_`/`sig_` for extrema, `int_`/`app_` for utility).
2. Drop rows without the selected label column.
3. Build chronological walk-forward splits using `WalkForwardSplitter`.
4. Require at least two folds; otherwise the UI reports the date-span/window mismatch.
5. Run RFECV once before the fold loop when enabled and there are at least two valid preliminary CV folds.
6. Reuse the selected feature subset for every fold model and the final saved model.
7. Purge training rows whose forward labeling window could cross into the test period.
8. Fit one CatBoost model per valid fold and collect true OOS predictions.
9. Evaluate by concatenating OOS fold predictions, not by scoring the final refit model.
10. Refit the final runtime CatBoost model on all labeled rows using the same selected features.

CatBoost is allowed to handle missing values natively. Do not add blanket `fillna(0.0)` in this training path unless intentionally matching a separate runtime approximation path.

## Metrics and outputs shown in the UI

The UI reports metrics from the concatenated OOS fold population:

- Precision, recall, F1, ROC-AUC, sample count.
- Bootstrap confidence intervals for precision/F1.
- Permutation p-value and Cohen's d.
- Fold counts, skipped single-class folds, and per-fold metrics.
- Confusion matrix, specificity, false-positive rate, and predicted positive rate.
- RTH coverage for OOS samples (NY 09:30-16:15 ET).
- Label-purged row count.
- Feature stability from cross-fold feature-importance rank correlation.
- Feature importance from the final refit runtime model; this is not an OOS metric.
- Threshold/coverage and calibration tables when probabilities are available.
- Simulated trade utility from OOS predictions at 15/15 and 15/30 TP/SL assumptions.

Quality gates are also based on OOS predictions:

- Precision >= 0.55
- Permutation p < 0.05
- Fold precision std < 0.15
- ROC-AUC > 0.55
- Brier score < 0.25
- Test samples >= 200

For utility mode, aggregate quality metrics are binary views of the 3-class model where `tradeable_reversal` is treated as the positive/executable class.

## Saved artifacts

Saving from the workbench writes a model bundle under `models/{model_name}/`:

```text
models/{model_name}/model.cbm
models/{model_name}/metadata.json
models/{model_name}/evaluation.json
```

- `model.cbm`: final CatBoost model refit on all labeled rows.
- `metadata.json`: selected features, feature importances, train metrics, model config.
- `evaluation.json`: all UI evaluation metrics plus full pipeline config and date range.

For the Trading-Dashboard compatibility contract, the canonical downstream artifact remains:

```text
data/models/dashboard_3feature_v1.cbm
```

The downstream dashboard consumes exactly these three features, in this order:

1. `int_time_beyond_level`
2. `int_time_within_2pts`
3. `int_absorption_ratio`

If training a dashboard-utility model for that runtime, preserve this feature contract. The retained exporter `scripts/train_dashboard_model.py` and compatibility path still exist for producing/promoting the canonical artifact expected by the FastAPI dashboard and external Trading-Dashboard.

## Gotchas

- The ML tab is local-Parquet only; missing data must be imported before opening the workbench.
- Dataset caches are config-hashed. If behavior changes without config changes, clear relevant caches manually.
- Quality gates must stay tied to concatenated OOS fold predictions, never final-refit predictions.
- RFECV must run once before the walk-forward loop; do not select different features per fold.
- Label purging is required to prevent forward-window leakage into test periods.
- CatBoost native NaN handling is part of the training contract.
- `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py` is experimental and has a known train/serve domain mismatch; do not treat it as production-ready for dashboard-utility deployment.
