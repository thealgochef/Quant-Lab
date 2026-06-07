# Architecture — Quant-Lab

Updated: 2026-06-04.

Quant-Lab is the research/training workbench for NQ/ES futures ML models. Its current production-aligned path is the **dashboard-utility** workflow, which is now single-sourced to **Strategy-Core v3** for the decision semantics that must match runtime execution.

> Current compatibility state: Quant-Lab emits `strategy_core_engine_v3` contracts and builds dashboard-utility datasets through Strategy-Core. Trade-Lab is **not yet v3-compatible**; see `../Strategy-Core/V3_COMPATIBILITY_MATRIX.md`.

---

## Canonical read order

1. `docs/README.md` — docs inventory and stale/historical classification.
2. `ARCHITECTURE.md` — this current architecture summary.
3. `docs/ML_TRAINING_WORKBENCH.md` — Streamlit workflow details.
4. `docs/pipeline_state.yaml` — machine-readable current-state summary.
5. `../Strategy-Core/README.md` and `../Strategy-Core/V3_COMPATIBILITY_MATRIX.md` — shared engine and cross-repo contract truth.
6. `scripts/ml_training_tab.py` — orchestration of build/train/save.
7. `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py`, `engine_decision.py`, and `strategy_contract.py` — v3 utility path.

Older historical reports and scaffold prompt documents were pruned from the working tree. Reconstruct current state from the canonical docs above plus current code/tests; use Git history only for audit context.

---

## Workflows

### 1. Extrema Rebound/Crossing mode — research only

Purpose: binary classifier over tick-level extrema.

Pipeline:

```text
local Databento parquet
  -> ExtremaDatasetBuilder
  -> extrema detection
  -> rebound/crossing labels
  -> PL/MS feature extraction
  -> walk-forward CatBoost binary evaluation
  -> final refit model bundle
```

Key modules:

- `src/alpha_lab/agents/data_infra/ml/dataset_builder.py`
- `src/alpha_lab/agents/data_infra/ml/extrema_detection.py`
- `src/alpha_lab/agents/data_infra/ml/labeling.py`
- `src/alpha_lab/agents/data_infra/ml/features_microstructure.py`
- `src/alpha_lab/agents/data_infra/ml/features_momentum.py`

Runtime caveat: `src/alpha_lab/agents/signal_eng/detectors/tier3/ml_extrema_classifier.py` is explicitly experimental and has a known train/serve domain mismatch.

### 2. Dashboard Utility mode — production-aligned research path

Purpose: 3-class level-touch classifier whose semantics are intended to be reproducible by Trade-Lab once Trade-Lab is repointed to Strategy-Core v3.

Pipeline:

```text
local Databento parquet
  -> TickStore / DuckDB bar + tick queries
  -> dashboard_utility_builder.py
  -> Strategy-Core v3 decision layer
       build_zones
       detect_touches with available_from guard
       resolve_honest_outcome
       v3 feature formulas
  -> walk-forward CatBoost MultiClass evaluation
  -> final refit model bundle
  -> strategy.json emitted from Strategy-Core constants
```

Key modules:

- `src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py` — self-contained date loop, bars, levels, cache writes.
- `src/alpha_lab/agents/data_infra/ml/engine_decision.py` — adapter from Quant-Lab dataframes/tick queries to Strategy-Core neutral types.
- `src/alpha_lab/agents/data_infra/ml/strategy_contract.py` — emits `strategy.json` from Strategy-Core constants/version stamps.
- `src/alpha_lab/agents/data_infra/ml/config.py` — config models and dataset cache hash; hash includes Strategy-Core engine version and price/label semantics.
- `src/alpha_lab/agents/data_infra/tick_store.py` — local DuckDB-backed parquet query layer.
- `scripts/run_dashboard_session_experiment.py` — CLI entrypoint for repeatable session-scope experiments.

Session-scope experiments are explicit research config. The default is to train/evaluate on `asia`, `london`, and `ny`, while production-gate diagnostics remain NY-only. Presets such as `ny_only`, `asia_only`, `london_only`, `asia_london_only`, and `all_sessions_all_gates` apply after dataset generation so caches stay reusable while fold training, OOS metrics, final refit, and gate reporting stay auditable.

### 3. Retained legacy compatibility/export path

The older `src/alpha_lab/experiment/`, `scripts/experiment_tab.py`, and `scripts/train_dashboard_model.py` path is retained as historical/compatibility tooling. It is **not** the canonical Strategy-Core v3 bundle path.

Legacy artifact:

```text
data/models/dashboard_3feature_v1.cbm
```

Do not treat that legacy 3-feature artifact as v3-compatible unless its accompanying `strategy.json` validates against `strategy_core_engine_v3` and the bundle files/checksums are verified.

---

## Strategy-Core v3 semantics used by dashboard-utility mode

| Area | Current behavior |
|---|---|
| Engine stamp | `strategy_core_engine_v3` |
| Contract stamp | `trade_lab_contract_v1` |
| Bars | Trade-price tick bars on 0.25 grid; default touch bar `147t`. |
| Sessions | ET-native: 18:00 trading-day boundary; `asia` 19:00→02:45, `london` 03:00→08:00, `ny` 09:00→17:00; gaps classify as `none`. |
| Levels | `PDH/PDL` = full prior trading-day high/low over `[18:00, 18:00)` ET. Session levels: Asia high/low and London high/low. |
| Availability guard | Enforced. A level cannot be touched before the session that defines it has closed. Merged-zone availability is the max constituent availability. |
| Touches | Merged zones within 3.0 points; representative = mean; first bar whose `[low, high]` intersects the zone representative; first touch per zone/day. |
| Features | `int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio` from trade prints; live runtime approach subset is `app_large_trade_vol_pct`, `app_avg_trade_size`, `app_max_spread`. |
| Labels | `tradeable_reversal=0`, `trap_reversal=1`, `aggressive_blowthrough=2`; MAE-first same-bar priority; TP 15, SL 30, trap MFE min 5 by default. |
| Honest entry | Decision can fire only after the post-touch feature window: `touch_close + 5m`. Label/outcome entry is the realistic trade price at that decision instant, not the level price at touch time. |
| Cutoffs | Drop new decisions at/after 16:40 ET; forward label cutoff is 17:00 ET. |
| Inference gate | Default contract gate is `tradeable_reversal`, `eligible_session="ny"`, `confidence_gate=0.70`. |

---

## Data and generated outputs

Local data layout:

```text
data/databento/{symbol}/{YYYY-MM-DD}/mbp10.parquet
data/databento/{symbol}/{YYYY-MM-DD}/mbp1.parquet
data/databento/{symbol}/{YYYY-MM-DD}/trades.parquet
```

Per-date training caches:

```text
ml_features_{config_hash}.parquet   # extrema mode
ml_utility_{config_hash}.parquet    # dashboard-utility mode
```

Saved Streamlit model bundle:

```text
models/{model_name}/model.cbm
models/{model_name}/metadata.json
models/{model_name}/evaluation.json
models/{model_name}/strategy.json
models/{model_name}/oos_predictions.parquet  # when OOS rows are available
```

Saved `evaluation.json` and `metadata.json` include `session_experiment` and `session_filter`; emitted `strategy.json` includes `research_session_experiment` but still advertises `supported_by_runtime=false` until Trade-Lab v3 parity is proven.

Generated/local outputs, not source-of-truth docs/code:

- `models/`
- `catboost_info/`
- `*.cbm`
- cached parquet/csv files under `data/`
- scratch chart HTML files
- local imported Databento data

The roadmap item "identify canonical data/model bundle location and verify file presence/checksums" is deliberately deferred until AlgoChef's local data zip is available.

---

## Verification expectations

- For code changes, run focused tests and update relevant docs in the same change.
- For dashboard-utility semantics, run Strategy-Core tests and Quant-Lab contract/no-drift tests before claiming v3 alignment.
- For any model/backtest claim, report date range, data source, fees/slippage assumptions, trade count, return/expectancy, drawdown, and limitations.
- Do not claim Trade-Lab runtime readiness until Trade-Lab is repointed to Strategy-Core v3 and end-to-end parity is proven.
