# Architecture - Alpha Signal Research Lab

## Repo Purpose

Alpha Signal Research Lab is a futures ML research and training workbench for NQ contracts. It has two training modes and one retained compatibility export path:

1. **Extrema Rebound/Crossing mode** (research) — binary classifier on tick-level price extrema
2. **Dashboard Utility mode** (production-aligned) — 3-class level-touch classifier aligned to Trading-Dashboard execution semantics
3. **Retained compatibility export** — standalone `train_dashboard_model.py` exporter for the canonical 3-feature downstream artifact

The multi-agent architecture still exists but is not the primary workflow.

## Training Modes

The Streamlit UI (`scripts/ml_training_tab.py`) provides a mode selector:

### Mode 1: Extrema Rebound/Crossing (Research)

Pipeline: tick data -> extrema detection -> binary labeling -> PL/MS feature extraction -> walk-forward CatBoost training -> OOS evaluation -> final refit -> model save

Key modules:
- `src/alpha_lab/agents/data_infra/ml/extrema_detection.py` — scipy peak-finding on tick prices
- `src/alpha_lab/agents/data_infra/ml/labeling.py` — binary rebound/crossing labels at configurable tick thresholds
- `src/alpha_lab/agents/data_infra/ml/features_microstructure.py` — PL features from MBP-10 book data (snapshot + dynamics)
- `src/alpha_lab/agents/data_infra/ml/features_momentum.py` — MS features (RSI, MACD, velocity, volatility)
- `src/alpha_lab/agents/data_infra/ml/dataset_builder.py` — orchestrates tick -> extrema -> labels -> features
- `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py` — **EXPERIMENTAL** runtime detector (known train/serve domain mismatch)

### Mode 2: Dashboard Utility (Production-Aligned)

Pipeline: experiment events.parquet -> utility labeling (+TP/-SL) -> 3 dashboard features from tick interaction window -> walk-forward CatBoost MultiClass -> OOS evaluation -> final refit -> model save

Key modules:
- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` — builds 3-feature dataset from level-touch events
- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_labeling.py` — 3-class labels aligned to dashboard execution (MAE-first, configurable TP/SL)
- Reuses `src/alpha_lab/experiment/event_detection.py` for touch events and `src/alpha_lab/experiment/labeling.py` for forward-bar loading

Output: CatBoost `.cbm` with 3 features (`int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`) and 3 classes (tradeable_reversal, trap_reversal, aggressive_blowthrough). Directly compatible with Trading-Dashboard's `validate_model_contract()`.

## Shared Training Infrastructure

Both modes use:
- `src/alpha_lab/agents/data_infra/ml/config.py` — Pydantic config with `training_mode` field, `DashboardUtilityConfig`, `ModelConfig` (supports `loss_function: MultiClass`)
- `src/alpha_lab/agents/data_infra/ml/walk_forward.py` — time-series walk-forward splitter with gap enforcement
- `src/alpha_lab/agents/data_infra/ml/model_trainer.py` — CatBoost training with optional RFECV, supports binary and multiclass
- `src/alpha_lab/agents/data_infra/ml/model_evaluator.py` — OOS evaluation with block bootstrap CI, Brier score, probability-based Cohen's d, permutation test
- `src/alpha_lab/agents/data_infra/tick_store.py` — DuckDB-backed parquet query layer with look-ahead prevention

## Data Flow

```
Databento ZIP -> process_batch_download.py -> per-date Parquet (with ingest validation)
                                                ↓
                                          TickStore (DuckDB)
                                                ↓
                            ┌───── Mode 1 (Extrema) ──────┐
                            │  detect_extrema              │
                            │  label rebound/crossing      │
                            │  extract PL/MS features      │
                            └──────────────────────────────┘
                                        OR
                            ┌───── Mode 2 (Utility) ──────┐
                            │  load experiment events      │
                            │  label with TP/SL thresholds │
                            │  compute 3 dashboard features│
                            └──────────────────────────────┘
                                        ↓
                              walk-forward splits
                              (with label purging)
                                        ↓
                              per-fold CatBoost train
                              (RFECV once before loop)
                                        ↓
                              OOS fold evaluation
                              (block bootstrap, Brier, utility metrics)
                                        ↓
                              full-data refit -> model.cbm
```

## Quality Gates

Current gates in `check_quality_gates()`:
- Precision >= 0.55
- Permutation p < 0.05 (plus-one corrected, 500 permutations)
- Fold stability (std < 0.15)
- ROC-AUC > 0.55
- Brier score < 0.25
- Test samples >= 200

Additional metrics surfaced in UI:
- RTH coverage fraction (warns if < 50%)
- Label-purged row count
- Feature stability (Spearman rank-correlation across folds)
- Trade utility (expectancy at 15/15 and 15/30, profit factor)

## Retained Compatibility / Export Path

Standalone exporter for the canonical downstream artifact:

- `scripts/train_dashboard_model.py` — purged walk-forward training on 3 features, exports `data/models/dashboard_3feature_v1.cbm`
- `src/alpha_lab/experiment/` — self-contained 6-phase research pipeline (key_levels, event_detection, labeling, features, training, diagnostics)
- `scripts/experiment_tab.py` — Streamlit diagnostics surface

### Downstream Contract

- Artifact: `data/models/dashboard_3feature_v1.cbm`
- Features: `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`
- Classes: tradeable_reversal (0), trap_reversal (1), aggressive_blowthrough (2)
- Consumer: Trading-Dashboard (`C:\Users\gonza\Documents\Trade-Dashboard`)

## Feature Cache

Cached datasets use config-keyed filenames: `ml_features_{hash}.parquet` where hash = SHA256 of (training_mode + extrema + labeling + features + dashboard_utility + tick_size). Changing any config parameter auto-invalidates stale cache.

## Known Limitations

1. **ml_extrema_classifier.py train/serve mismatch** — marked `_EXPERIMENTAL`. Approximates tick-level features from bars with 0.0 fills and placeholder values. Not execution-faithful.
2. **Entry price semantic drift** — training labels use level_price as reference; dashboard trades enter at market_price. Prediction "correct" may not equal "profitable."
3. **Execution population** — training includes all sessions; dashboard executes only NY RTH tradeable_reversal predictions.

## Generated Outputs (Not Source Code)

- `models/` — saved model bundles
- `catboost_info/` — CatBoost scratch logs
- `*.cbm` — model binaries
- `data/` — cached parquet/csv outputs
- scratch chart HTML files

## Recommended Read Order

1. `scripts/ml_training_tab.py` (mode selector, full training flow)
2. `src/alpha_lab/agents/data_infra/ml/config.py` (all config models)
3. `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` (utility mode)
4. `src/alpha_lab/agents/data_infra/ml/model_evaluator.py` (evaluation)
5. `scripts/train_dashboard_model.py` (canonical export)
6. `src/alpha_lab/experiment/` (experiment pipeline)
