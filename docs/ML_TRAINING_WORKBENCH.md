# Streamlit ML Training Workbench

Updated: 2026-06-05.

The ML Training Workbench is the primary Quant-Lab UI for building local ML datasets, running walk-forward CatBoost evaluation, and saving runtime model bundles. It is mounted by `scripts/dashboard.py` and implemented in `scripts/ml_training_tab.py`.

The workbench is local-data only: it discovers and reads Databento-derived Parquet files from disk through `TickStore`. It does not call the Databento API.

---

## Current code map

| Path | Role |
|---|---|
| `scripts/dashboard.py` | Streamlit host. |
| `scripts/ml_training_tab.py` | UI state, configuration assembly, dataset build orchestration, walk-forward training, metrics display, save. |
| `src/alpha_lab/agents/data_infra/tick_store.py` | DuckDB-backed reader/query layer over local Databento parquet. |
| `src/alpha_lab/agents/data_infra/ml/config.py` | Pydantic configs; dataset cache hash includes Strategy-Core engine/version semantics. |
| `src/alpha_lab/agents/data_infra/ml/dataset_builder.py` | Extrema-mode binary dataset builder. |
| `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` | Dashboard-utility dataset loop; bars, levels, cache writes. |
| `src/alpha_lab/agents/data_infra/ml/engine_decision.py` | Strategy-Core v3 adapter for zones, touches, features, and honest outcomes. |
| `src/alpha_lab/agents/data_infra/ml/strategy_contract.py` | Emits `strategy.json` from Strategy-Core constants and `ENGINE_VERSION`. |
| `src/alpha_lab/agents/data_infra/ml/walk_forward.py` | Rolling/expanding time-series split generation. |
| `src/alpha_lab/agents/data_infra/ml/model_trainer.py` | CatBoost training and optional RFECV. |
| `src/alpha_lab/agents/data_infra/ml/model_evaluator.py` | OOS metrics, bootstrap CIs, permutation tests, calibration summaries. |
| `scripts/run_dashboard_session_experiment.py` | CLI runner for Databento-backed dashboard-utility session experiments. |

---

## Configuration flow

All major UI selections are copied into `MLPipelineConfig` before dataset build or training:

- `training_mode`: `extrema_rebound_crossing` or `dashboard_utility`.
- `instrument` / `tick_size`: selected symbol and tick size; NQ trade grid is 0.25.
- `walk_forward`: train window, test window, gap.
- `model`: CatBoost iterations, depth, loss, and RFECV setting.
- `dashboard_utility`: TP/SL, bar type, interaction window, approach-feature settings.
- `session_experiment`: research-only training/evaluation/gate session scope.
- `features`: extrema feature settings.

`MLPipelineConfig.dataset_config_hash()` hashes settings that affect dataset generation. For dashboard-utility, the hash also includes Strategy-Core's engine version and price/label source semantics, so v1/v2/v3 caches cannot be silently reused across structural engine changes.

Dashboard-utility's default bar type is `147t`. The Streamlit dropdown is intentionally ordered with `147t` first so the UI default matches `DashboardUtilityConfig` and the emitted Strategy-Core v3 contract.

Dashboard-utility session experiments are typed config, not loose UI toggles:

```yaml
session_experiment:
  training_sessions: [asia, london, ny]
  evaluation_sessions: [asia, london, ny]
  production_gate_sessions: [ny]
  report_session_breakdowns: true
```

Presets are available in the UI and CLI: `all_to_ny`, `ny_only`, `asia_only`, `london_only`, `asia_london_only`, and `all_sessions_all_gates`. Dataset generation remains broad/cacheable; the session scope is applied during fold training, OOS evaluation, final refit, and gate reporting.

---

## Training modes

### Extrema Rebound/Crossing

Purpose: binary research classifier over tick-level extrema.

- Data source: local parquet under `data/databento/{symbol}/{YYYY-MM-DD}/`.
- Builder: `ExtremaDatasetBuilder` via `build_training_dataset()`.
- Candidate events: extrema detected from tick prices using `ExtremaConfig`.
- Labels: rebound/crossing labels from `labeling.py` with UI-selected thresholds.
- Features: `pl_*` price-level microstructure and `ms_*` momentum columns; `sig_*` support exists lower-level but is not the UI's primary path.
- Loss: CatBoost binary `Logloss`.
- Runtime caveat: `ml_extrema_classifier.py` is experimental and not execution-faithful.

### Dashboard Utility — Strategy-Core v3 path

Purpose: 3-class level-touch classifier for the execution problem that Trade-Lab must eventually reproduce.

Current production-aligned behavior:

- Data source: local tick parquet. The workbench path is self-contained and does **not** require prebuilt `data/experiment/events.parquet`.
- Builder: `build_utility_dataset(..., use_engine=True)`.
- Engine: Strategy-Core v3 via `engine_decision.process_single_date_engine()`.
- Bars: trade-price tick bars on the 0.25 grid; `147t` is the default decision/touch bar.
- Sessions: ET `asia` 19:00→02:45, `london` 03:00→08:00, `ny` 09:00→17:00; 18:00 ET trading-day boundary.
- Levels: PDH/PDL from the full prior trading day; Asia/London session high/low from the current trading day.
- Availability: each level carries `available_from`; touches before the level is knowable are skipped and do not consume first-touch.
- Touches: merged zones within 3.0 points; first bar-range intersection per zone/trading day.
- Labels: `tradeable_reversal`, `trap_reversal`, `aggressive_blowthrough`; MAE-first; default TP 15 / SL 30 / trap MFE min 5.
- Honest entry: label/outcome entry is the realistic trade price at `touch_close + 5m`, not the level price at touch time.
- Cutoffs: no new decision at/after 16:40 ET; forward cutoff 17:00 ET.
- Features: 3 trade-print interaction features plus optional live-computable `app_*` approach subset.
- Loss: CatBoost `MultiClass`.
- Positive class for binary quality metrics: `tradeable_reversal` (`0`).

---

## UI steps

### Step 1: Local Data

Scans the selected data directory for:

```text
data/databento/{symbol}/{YYYY-MM-DD}/mbp10.parquet
data/databento/{symbol}/{YYYY-MM-DD}/mbp1.parquet
data/databento/{symbol}/{YYYY-MM-DD}/trades.parquet
```

`TickStore` resolves available local files and registers dates. If no local data is found, import a Databento batch zip first with `scripts/process_batch_download.py`.

### Step 2: Build Dataset

Extrema mode:

1. Register each selected date in a fresh `TickStore`.
2. Query tick feature rows.
3. Detect extrema, label them, and extract `pl_*` / `ms_*` features.
4. Cache per-date results as `ml_features_{config_hash}.parquet`.

Dashboard-utility mode:

1. Build trade-price bars for each selected trading date.
2. Compute current/prior levels and attach v3 availability timestamps.
3. Convert bars/levels/ticks into Strategy-Core neutral types.
4. Build zones, detect touches, and resolve honest decision-time outcomes through Strategy-Core.
5. Compute trade-print interaction features and optional live `app_*` features.
6. Cache per-date results as `ml_utility_{config_hash}.parquet`.

The `Clear Cache` UI is safest for stale extrema caches; utility caches are separated by `ml_utility_*` and config/engine hash.

### Step 3: Train Model

`run_walk_forward_training()`:

1. Select feature columns by mode (`pl_`/`ms_`/`sig_` for extrema, `int_`/`app_` for utility).
2. Drop rows missing the selected label.
3. Build chronological walk-forward splits.
4. Require at least two valid folds.
5. Run RFECV once before the fold loop when enabled and valid.
6. Use the same selected feature subset for every fold model and the final refit model.
7. Purge training rows whose forward label window could cross into test.
8. Restrict fold training rows to `session_experiment.training_sessions`.
9. Restrict OOS fold metrics to `session_experiment.evaluation_sessions`.
10. Fit one CatBoost model per valid fold and collect true OOS predictions.
11. Evaluate by concatenating OOS fold predictions, not by scoring the final refit model.
12. Refit the final runtime CatBoost model on the configured training sessions only.

CatBoost native NaN handling is part of the training contract. Do not add blanket `fillna(0.0)` unless intentionally matching a separate runtime approximation path.

For dashboard-utility, purge metadata is exact when possible: if rows contain `label_window_end`, training rows are purged by `label_window_end < test_start`; old utility caches that lack the column derive the v3 label horizon from the trading day and Strategy-Core cutoff before falling back to heuristic timestamp purging. Old caches with null/unknown `session` values are also repaired from Strategy-Core timestamp classification before production-gate OOS reporting.

---

## Metrics and quality gates

The UI reports metrics from concatenated OOS fold predictions:

- Precision, recall, F1, ROC-AUC, sample count.
- Bootstrap confidence intervals for precision/F1.
- Permutation p-value and Cohen's d.
- Fold counts, skipped single-class folds, and per-fold metrics.
- Confusion matrix, specificity, false-positive rate, predicted positive rate.
- Label-purged row count.
- Feature stability from cross-fold feature-importance rank correlation.
- Feature importance from the final refit model (not an OOS metric).
- Threshold/coverage and calibration tables when probabilities are available.
- Utility summaries for OOS predictions; interpret these as model diagnostics, not production PnL proof.
- Dashboard-utility production-gate OOS diagnostics: configured gate sessions, defaulting to `session == ny`, and `P(tradeable_reversal) >= 0.70` trade count, precision, coverage, eligible-session coverage, and idealized 15/30 expectancy.
- Session-filtered OOS diagnostics so the NY execution population can be separated from aggregate all-session metrics.
- Optional `oos_predictions.parquet` with fold, timestamp, session, raw class, prediction, probabilities, a configured runtime-session gate flag, and the `0.70/ny` gate flag for offline error analysis.

Quality gates:

- Precision >= 0.55
- Permutation p < 0.05
- Fold precision std < 0.15
- ROC-AUC > 0.55
- Brier score < 0.25
- Test samples >= 200

For utility mode, aggregate quality metrics are binary views of the 3-class model where `tradeable_reversal` is treated as the positive/executable class.

Failed quality gates block `save_trained_model()` by default. A saved bundle with failed gates requires an explicit override (`allow_failed_gates=True`), and that override is recorded in `evaluation.json`; such bundles are smoke/research artifacts unless later evidence justifies promotion.

---

## Saved artifacts

Saving from the workbench writes a model bundle under `models/{model_name}/`:

```text
models/{model_name}/model.cbm
models/{model_name}/metadata.json
models/{model_name}/evaluation.json
models/{model_name}/strategy.json
models/{model_name}/oos_predictions.parquet  # when OOS rows are available
```

- `model.cbm`: final CatBoost model refit on all labeled rows.
- `metadata.json`: selected features, feature importances, train metrics, model config.
- `evaluation.json`: UI evaluation metrics, full pipeline config, date range, `session_experiment`, and `session_filter` metadata.
- `strategy.json`: runtime strategy semantics stamped with `contract_version` and `engine_version`; also records `research_session_experiment` for audit.
- `oos_predictions.parquet`: OOS fold-level prediction rows for post-training diagnostics and gate/error slicing.

The CLI path for repeatable experiments is:

```bash
python scripts/run_dashboard_session_experiment.py --preset ny_only --dry-run
python scripts/run_dashboard_session_experiment.py \
  --preset all_to_ny \
  --include-approach-features \
  --approach-window 90 \
  --start 2025-07-09 \
  --end 2025-07-15
```

Run a dry run first to prove date discovery and session scope, then a bounded smoke train before full-range training. Use `--save` only when an artifact is needed; weak models still require `--allow-failed-gates` and remain research-only.

The old retained exporter still writes:

```text
data/models/dashboard_3feature_v1.cbm
```

That file is not automatically a Strategy-Core v3 bundle. The deferred bundle task is to identify the canonical v3 bundle location and verify file presence/checksums once the incoming data/model archive is available.

---

## Gotchas

- Trade-Lab is not v3-compatible yet. Do not claim runtime readiness just because Quant-Lab can train/emit v3 contracts.
- Historical docs and reports may mention `strategy_core_engine_v1/v2`, `ny_rth`, 09:30→16:15, 15:55 flatten, or book-mid features. Those are superseded for v3 unless a historical audit is explicitly being discussed.
- Utility mode defaults must remain contract-aligned: `147t`, trade-price bars, 5-minute decision offset, `ny` eligible session, and `0.70` confidence gate.
- Session experiment defaults must remain fail-safe: train/evaluate all sessions but production-gate NY only; persist scope into `evaluation.json`, `metadata.json`, and `strategy.json`.
- Utility dataset caches are engine/config-keyed; behavior changes without a cache-key change require manual cache clearing.
- Quality gates must stay tied to OOS fold predictions.
- RFECV must run once before walk-forward, not per fold.
- Label purging remains required for walk-forward leakage control.
- No model profitability, robustness, or live readiness claim is valid without a current v3 backtest/paper-trading evidence chain.
