# Alpha Signal Research Lab - Agent Handoff Guide

## Start Here

This repo is an ML training workbench for NQ futures with two training modes:

1. **Extrema Rebound/Crossing** — binary tick-level classifier (research)
2. **Dashboard Utility** — 3-class level-touch classifier aligned to Trading-Dashboard

Both modes share walk-forward evaluation, CatBoost training, and quality gate infrastructure.

## Primary Workflow

The Streamlit UI has a mode selector at the top. Both modes go through the same training flow:

- **UI**: `scripts/ml_training_tab.py` (mode selector, dataset build, train, evaluate, save)
- **Host**: `scripts/dashboard.py`
- **Data prep**: `scripts/process_batch_download.py` (with ingest quality checks)
- **Core ML modules**: `src/alpha_lab/agents/data_infra/ml/`
- **Config**: `src/alpha_lab/agents/data_infra/ml/config.py` (training_mode, DashboardUtilityConfig, ModelConfig)

### Extrema Mode Modules
- `dataset_builder.py`, `extrema_detection.py`, `labeling.py`, `features_microstructure.py`, `features_momentum.py`

### Dashboard Utility Mode Modules
- `dashboard_utility_builder.py`, `dashboard_utility_labeling.py`
- Reuses `src/alpha_lab/experiment/event_detection.py` and `src/alpha_lab/experiment/labeling.py`

### Shared Modules
- `walk_forward.py`, `model_trainer.py`, `model_evaluator.py`, `tick_store.py`, `config.py`

### Experimental Runtime Detector
- `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py`
- Marked `_EXPERIMENTAL = True` with runtime warning
- Known train/serve domain mismatch — bar-level approximations with 0.0 fills
- NOT for production use

## Retained Compatibility / Export Path

Keep this path working as a standalone exporter:

- Research pipeline: `src/alpha_lab/experiment/`
- Diagnostics tab: `scripts/experiment_tab.py`
- Canonical exporter: `scripts/train_dashboard_model.py`
- Artifact contract: `data/models/dashboard_3feature_v1.cbm`

## Cross-Repo Contract (Trading-Dashboard)

Trading-Dashboard at `C:\Users\gonza\Documents\Trade-Dashboard` consumes:
- 3 features: `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`
- 3 classes: tradeable_reversal (0), trap_reversal (1), aggressive_blowthrough (2)
- Resolution ordering: MAE-first (conservative) — aligned in both repos
- Dashboard validates model contract on upload (feature count, class count, feature names)
- Dashboard PositionMonitor supports configurable slippage (default 0.50 pts in runners)

## Working Rules

1. Both training modes go through `scripts/ml_training_tab.py`
2. Quality gates come from true out-of-sample fold predictions with label purging
3. RFECV runs once before the walk-forward loop (same features for all folds + final model)
4. Feature cache files are keyed to config hash (changing any parameter invalidates cache)
5. The final model is refit on all labeled rows after evaluation
6. Saved artifacts include `training_mode` in metadata
7. Dashboard-utility mode output is directly compatible with Trading-Dashboard contract
8. Preserve `data/models/dashboard_3feature_v1.cbm` contract for standalone export path

## Key Technical Details

- **Bootstrap**: Block bootstrap (block_size=10) respecting time-series autocorrelation
- **Brier score**: Computed and enforced as quality gate (< 0.25)
- **Cohen's d**: Computed on predicted probabilities (not binary predictions)
- **Permutation test**: Plus-one corrected, 500 permutations
- **RTH coverage**: Fraction of OOS samples in NY RTH, warned if < 50%
- **Feature stability**: Spearman rank-correlation of importance across folds
- **Trade utility**: Expectancy and profit factor at configurable TP/SL geometries
- **Label purging**: Forward-window leakage prevented via time-based purge buffer
- **Ingest validation**: Timestamp monotonicity, price sanity, duplicate detection

## Verification Commands

```bash
# Run all tests (669 tests expected)
"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m pytest tests/ -v

# Run ML-specific tests
"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m pytest tests/agents/test_ml_pipeline.py tests/agents/test_ml_features.py tests/agents/test_ml_extrema.py -v

# Run Streamlit UI
streamlit run scripts/dashboard.py
```

## State Tracking

- `docs/pipeline_state.yaml` — current phase, module inventory, cross-repo alignment
- `docs/DECISIONS.md` — all architectural decisions with rationale (D-001 through D-028)
- `ARCHITECTURE.md` — full architecture with data flow diagram

## Recommended Reading Order

1. `scripts/ml_training_tab.py` (start here — mode selector, full flow)
2. `src/alpha_lab/agents/data_infra/ml/config.py` (all config models)
3. `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` (utility mode)
4. `src/alpha_lab/agents/data_infra/ml/model_evaluator.py` (evaluation)
5. `src/alpha_lab/agents/data_infra/ml/model_trainer.py` (CatBoost training)
6. `scripts/train_dashboard_model.py` (canonical export)
7. `src/alpha_lab/experiment/` (experiment pipeline)
