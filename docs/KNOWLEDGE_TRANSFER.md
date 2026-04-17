# Knowledge Transfer — Quant-Lab + Trading-Dashboard Audit & Remediation

**Date**: 2026-04-04
**Scope**: Complete cross-repo audit, bug remediation, and Phase 3 dashboard-utility mode implementation

---

## 1. What Was Done

### Cross-Repository Audit

A comprehensive code-verified audit of both Quant-Lab and Trading-Dashboard was performed, examining every critical module in both repos. The audit covered:

- ML pipeline integrity (training, evaluation, feature extraction, labeling)
- Cross-repo contract alignment (features, classes, resolution ordering, entry price semantics)
- Execution realism (TP/SL exit mechanics, slippage, outcome tracking)
- Statistical rigor (bootstrap methodology, permutation tests, calibration)
- Metric correctness (what metrics can be trusted, which are misleading)

### Bug-Level Remediations (All Committed)

**Quant-Lab** (commits `a441c8c`, `372d93d`, plus subsequent work):

| Fix | What Changed |
|---|---|
| Config-keyed cache | `ml_features_{hash}.parquet` — changing any config invalidates cache |
| RFECV alignment | Runs once before walk-forward loop; same features for all folds + final model |
| fillna(0.0) removed | CatBoost handles NaN natively now |
| Default label_column fixed | `"label_7t"` -> `"label_20t"` |
| Bootstrap/permutation counts | 1000/500 (was 200/100) |
| Block bootstrap | Moving-block (block_size=10) replaces IID resampling |
| Cohen's d on probabilities | Uses predicted probabilities, not binary predictions |
| Brier score quality gate | Added to EvaluationResult + quality gate (< 0.25) |
| Min sample gate raised | 50 -> 200 |
| evaluate_walk_forward deprecated | DeprecationWarning emitted; in-sample-only docstring |
| Spread tick-size fixed | Uses `tick_size` parameter, not hardcoded 0.25 |
| Experiment docstring fixed | "MFE >= 10" -> "MFE >= 25" |
| RSI/precompute cleanup | Dead `precompute_ms_indicators()` removed |
| Label purging | Forward-window leakage prevented via time-based purge buffer |
| RTH coverage reporting | Fraction of OOS samples in NY RTH, warned if < 50% |
| Feature stability | Spearman rank-correlation across folds |
| Trade utility metrics | Expectancy and profit factor at configurable TP/SL |
| MBP depth dynamics | Time-dynamic features (bid/ask vol trend, imbalance volatility, spread max) |
| Ingest validation | Timestamp monotonicity, price sanity, duplicate detection in batch download |
| ml_extrema_classifier marked | `_EXPERIMENTAL = True` with runtime warning |
| Pydantic sort_keys compat | `json.dumps(model_dump(), sort_keys=True)` for older Pydantic |

**Trading-Dashboard** (commit `e440ba4`, plus subsequent work):

| Fix | What Changed |
|---|---|
| Resolution ordering aligned | OutcomeTracker now MAE-first (conservative), matching training labels |
| Slippage support | `slippage_points` parameter in PositionMonitor (default 0, configurable) |
| Slippage wired into runners | `--slippage` CLI arg in sweep and comparison runners (default 0.50 pts) |
| Model contract validation | Feature count + class count + feature names checked on activation |
| Entry-vs-level offset analytics | `entry_vs_level_offset` field + summary stats in prediction analytics |
| Drift monitor module | `drift_monitor.py` — rolling feature distribution monitoring with z-score alerts |

### Phase 3: Dashboard-Utility Training Mode

The largest architectural addition. The Streamlit UI now has a mode selector:

**New files:**
- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` — builds 3-feature dataset from level-touch events
- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_labeling.py` — 3-class utility labeling with configurable TP/SL

**How it works:**
1. User selects "Dashboard Utility (3-class)" mode in Streamlit
2. Configures TP/SL points (default 15/30)
3. Builds dataset from `data/experiment/events.parquet` (requires experiment Phase 1+2)
4. 3 features computed from tick interaction window: `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`
5. 3-class labels: tradeable_reversal (MFE >= TP before MAE >= SL), trap_reversal (stopped + MFE >= 5), aggressive_blowthrough (stopped + MFE < 5)
6. MAE-first resolution ordering (matches both experiment path and Trading-Dashboard)
7. Trains CatBoost with `loss_function="MultiClass"`
8. Evaluates with binary metrics (tradeable_reversal = positive class)
9. Saves model with `training_mode: "dashboard_utility"` in metadata
10. Output `.cbm` is directly compatible with Trading-Dashboard's `validate_model_contract()`

---

## 2. Architecture Summary

### Quant-Lab Pipeline

```
Databento ZIP -> process_batch_download.py -> per-date Parquet
                                                ↓
                                          TickStore (DuckDB)
                                                ↓
                    Mode 1 (Extrema)         OR         Mode 2 (Utility)
                    ├ detect_extrema                    ├ load experiment events
                    ├ label rebound/crossing             ├ label with TP/SL
                    └ extract PL/MS features             └ compute 3 dashboard features
                                                ↓
                              walk-forward splits (with label purging)
                                                ↓
                              RFECV (once, if enabled)
                                                ↓
                              per-fold CatBoost train + OOS evaluation
                                                ↓
                              quality gates + utility metrics + calibration
                                                ↓
                              full-data refit -> model.cbm + metadata
```

### Cross-Repo Contract

| Aspect | Quant-Lab | Trading-Dashboard | Status |
|---|---|---|---|
| Features | 3: int_time_beyond_level, int_time_within_2pts, int_absorption_ratio | Same 3, same order | ALIGNED |
| Classes | 0=reversal, 1=trap, 2=blowthrough | Same encoding | ALIGNED |
| Resolution order | MAE-first (conservative) | MAE-first (conservative) | ALIGNED |
| Feature proximity | 0.50 pts for absorption | 0.50 pts (`LEVEL_PROXIMITY`) | ALIGNED |
| Zone merge | 3.0 pts | 3.0 pts (`PROXIMITY_THRESHOLD`) | ALIGNED |
| Entry reference | level_price in labels | current_price for trades, level_price for outcome tracking | KNOWN DRIFT |
| Execution population | All sessions in training | Only NY RTH tradeable_reversal | KNOWN DRIFT |

### Quality Gates (Current)

| Gate | Threshold |
|---|---|
| Precision | >= 0.55 |
| Permutation p-value | < 0.05 |
| Fold stability (std) | < 0.15 |
| ROC-AUC | > 0.55 |
| Brier score | < 0.25 |
| Test samples | >= 200 |

Additional surfaced metrics: RTH coverage, label-purged count, feature stability, expectancy (15/15 and 15/30), profit factor.

---

## 3. Known Remaining Gaps

These are architectural/semantic, not bugs:

| ID | Gap | Impact |
|---|---|---|
| **A1** | ml_extrema_classifier.py train/serve mismatch | Bar-level approximations serve a tick-trained model. Marked experimental. Not used by Dashboard. |
| **A2** | Entry price semantic drift | Labels use level_price; trades enter at market_price. "Correct" != "profitable." |
| **A3** | Execution population mismatch | Training includes all sessions; Dashboard executes only NY RTH. RTH fraction now reported. |
| **A4** | OutcomeTracker reference frame | Measures from level_price, not entry_price. By design but drift visible via entry_vs_level_offset. |
| **A5** | Drift monitor not yet wired | Module created but not integrated into live pipeline service. |

---

## 4. Key Files Reference

### Quant-Lab

| Path | Purpose |
|---|---|
| `scripts/ml_training_tab.py` | Main Streamlit UI — mode selector, build, train, evaluate, save |
| `scripts/dashboard.py` | Streamlit host (tabs: ML Training, Experiment) |
| `scripts/process_batch_download.py` | Databento ZIP -> parquet (with ingest validation) |
| `scripts/train_dashboard_model.py` | Canonical 3-feature exporter |
| `src/alpha_lab/agents/data_infra/ml/config.py` | All config models (training_mode, DashboardUtilityConfig, ModelConfig) |
| `src/alpha_lab/agents/data_infra/ml/dataset_builder.py` | Extrema dataset builder |
| `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` | Utility dataset builder |
| `src/alpha_lab/agents/data_infra/ml/dashboard_utility_labeling.py` | Utility 3-class labeling |
| `src/alpha_lab/agents/data_infra/ml/model_trainer.py` | CatBoost training (binary + multiclass) |
| `src/alpha_lab/agents/data_infra/ml/model_evaluator.py` | OOS evaluation (block bootstrap, Brier, Cohen's d) |
| `src/alpha_lab/agents/data_infra/ml/walk_forward.py` | Walk-forward splitter |
| `src/alpha_lab/agents/data_infra/tick_store.py` | DuckDB tick query layer |
| `src/alpha_lab/agents/data_infra/ml/features_microstructure.py` | PL features (snapshot + dynamics) |
| `src/alpha_lab/agents/data_infra/ml/features_momentum.py` | MS features (RSI, MACD, velocity) |
| `src/alpha_lab/experiment/` | 6-phase experiment pipeline |
| `docs/DECISIONS.md` | All architectural decisions (D-001 through D-028) |
| `docs/pipeline_state.yaml` | Current phase, module inventory, cross-repo alignment |
| `docs/BACKTEST_FINDINGS.md` | Historical backtest results and analysis |

### Trading-Dashboard

| Path | Purpose |
|---|---|
| `backend/src/alpha_lab/dashboard/model/__init__.py` | FEATURE_COLUMNS, CLASS_NAMES constants |
| `backend/src/alpha_lab/dashboard/model/model_manager.py` | Model loading, versioning, contract validation |
| `backend/src/alpha_lab/dashboard/model/prediction_engine.py` | CatBoost inference |
| `backend/src/alpha_lab/dashboard/model/outcome_tracker.py` | MFE/MAE resolution (MAE-first) |
| `backend/src/alpha_lab/dashboard/engine/feature_computer.py` | 3-feature computation from ticks |
| `backend/src/alpha_lab/dashboard/engine/level_engine.py` | PDH/PDL/session level computation |
| `backend/src/alpha_lab/dashboard/engine/touch_detector.py` | Level touch detection |
| `backend/src/alpha_lab/dashboard/trading/position_monitor.py` | TP/SL enforcement (slippage support) |
| `backend/src/alpha_lab/dashboard/trading/trade_executor.py` | Trade entry at market price |
| `backend/src/alpha_lab/dashboard/engine/drift_monitor.py` | Feature drift detection |
| `backend/run_tp_sl_sweep.py` | TP/SL parameter sweep backtest |
| `backend/run_strategy_comparison.py` | Multi-strategy comparison |
| `backend/run_prediction_analytics.py` | Prediction-level analysis with offset stats |

---

## 5. Verification

### Tests
```bash
# Quant-Lab: 669 tests expected
"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m pytest tests/ -v

# Streamlit UI
streamlit run scripts/dashboard.py
```

### Smoke Test (Utility Mode)
```bash
# Requires data/experiment/events.parquet from experiment Phase 1+2
python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import build_utility_dataset
from alpha_lab.agents.data_infra.ml.config import MLPipelineConfig, DashboardUtilityConfig
config = MLPipelineConfig(training_mode='dashboard_utility', dashboard_utility=DashboardUtilityConfig(tp_points=15.0, sl_points=30.0), instrument='NQ')
df = build_utility_dataset(Path('data/experiment/events.parquet'), Path('data/databento'), config)
print(df.shape, df.columns.tolist())
"
```

---

## 6. Decision Log Summary (New Decisions)

| ID | Decision | Date |
|---|---|---|
| D-020 | Config-keyed feature cache | 2026-04-03 |
| D-021 | RFECV runs once before walk-forward loop | 2026-04-03 |
| D-022 | CatBoost NaN handling preserved (fillna removed) | 2026-04-03 |
| D-023 | Cross-repo resolution ordering aligned (MAE-first) | 2026-04-03 |
| D-024 | Block bootstrap for time-series CIs | 2026-04-03 |
| D-025 | Brier score quality gate | 2026-04-03 |
| D-026 | Dashboard-utility training mode | 2026-04-04 |
| D-027 | Label purging in primary pipeline | 2026-04-04 |
| D-028 | ml_extrema_classifier marked experimental | 2026-04-04 |

---

## 7. What To Do Next

1. **Run the Streamlit UI** — verify both modes work end-to-end with real data
2. **Train a utility-mode model** — validate the 3-class output passes Trading-Dashboard contract validation
3. **Run a backtest** — use `run_tp_sl_sweep.py --slippage 0.50` with the new model to see realistic P&L
4. **Wire drift monitor** — integrate `drift_monitor.py` into the live pipeline service
5. **Consider RTH-only training** — if RTH coverage is low, filter training events to RTH-only for tighter population alignment
6. **Decide on entry-price semantics** — explicitly document whether "prediction correct" means level-reaction or trade-profitability, and make this visible in the UI
