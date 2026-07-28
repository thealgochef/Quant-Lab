# W3-CONFIG RECON — store inventory + current training/label/feature defaults

**Purpose:** owner inputs for picking the W3 training window and gate days. Read-only recon; nothing committed, no pipeline runs.
**Date:** 2026-06-12
**Repos (verified at recon time):** QL `12c50a5` · SC `ecbc15e` · TL `5a8d28a` (branch `platform-refactor`)
**Paths:** QL = `C:\Users\gonza\Documents\Claude-Quant-Lab`, SC = `C:\Users\gonza\Documents\Strategy-core`, TL = `C:\Users\gonza\Documents\Trade-Lab`. File:line references are relative to the owning repo root.

---

## 1. Store inventory — `data/databento/NQ` (QL repo)

Enumerated `C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento/NQ` (dated `YYYY-MM-DD` directories).

| Metric | Value |
|---|---|
| Total day directories | **312** |
| First day | **2021-12-02** |
| Last day | **2026-02-22** (a Sunday partial-evening dir; last weekday dir is 2026-02-20) |
| Contiguous spans (split at >3-calendar-day gaps) | **2021-12-02 → 2022-03-10** (85 dirs) and **2025-06-02 → 2026-02-22** (227 dirs) |
| Gaps > 3 calendar days | **One: 2022-03-10 → 2025-06-02 (1,180 calendar days)** — the entire 2022-03 → 2025-05 period is absent |

Notes on directory semantics:
- The store includes **Sunday dirs** (e.g. 2026-01-25, 2026-02-01, 2026-02-08, 2026-02-15, 2026-02-22). Per SC `docs/PLATFORM_REFACTOR_PROGRESS.md:217-218`, calendar-prev files exist so the Globex window `[prev 18:00 ET, day 18:00 ET)` is fully captured — Sunday dirs are the evening-session calendar-prev for Mondays, not standalone trading days.
- 2026-01-19 (MLK) and 2026-02-16 (Presidents' Day) are present with non-zero mbp10 (shortened-session holidays; data exists).

**mbp10.parquet health — 2 bad days (of 312):**
| Day | Problem | Dir contents |
|---|---|---|
| **2025-11-20** (Thursday) | `mbp10.parquet` MISSING | directory is **empty** — a true weekday hole inside the 2025-06→2026-02 span |
| **2026-02-14** (Saturday) | `mbp10.parquet` MISSING | only `ohlcv_1m_20260214_20260224.parquet` — a stray non-trading-day dir |

All other 310 days have a present, non-zero `mbp10.parquet`.

**Last 30 available trading days** (weekday dirs with valid non-zero `mbp10.parquet`; weekend partial dirs excluded):

```
2026-01-12 Mon    2026-01-13 Tue    2026-01-14 Wed    2026-01-15 Thu    2026-01-16 Fri
2026-01-19 Mon*   2026-01-20 Tue    2026-01-21 Wed    2026-01-22 Thu    2026-01-23 Fri
2026-01-26 Mon    2026-01-27 Tue    2026-01-28 Wed    2026-01-29 Thu    2026-01-30 Fri
2026-02-02 Mon    2026-02-03 Tue    2026-02-04 Wed    2026-02-05 Thu    2026-02-06 Fri
2026-02-09 Mon    2026-02-10 Tue    2026-02-11 Wed    2026-02-12 Thu    2026-02-13 Fri
2026-02-16 Mon*   2026-02-17 Tue    2026-02-18 Wed    2026-02-19 Thu    2026-02-20 Fri
```
`*` = exchange holiday with shortened Globex session (data present; flag for gate-day selection).

---

## 2. The nine replay days (D-window Gate A/B characterization)

From SC `docs/PLATFORM_REFACTOR_PROGRESS.md:216-217`, verbatim:

> **Dates (9 CONSECUTIVE trading days):** 2025-07-10, 2025-07-11, 2025-07-14, 2025-07-15, 2025-07-16, 2025-07-17, 2025-07-18, 2025-07-21, 2025-07-22.

The D-window Gate A/B entry (`PLATFORM_REFACTOR_PROGRESS.md:1242-1245`) references this same set: "**GATE A: EXACT streaming==batch per-touch parity** over the 9 real store days (42 touches: 35 resolved {tradeable 25, blowthrough 7, trap 3} + drops {flatten 7} …) **GATE B characterization** (real bundle through the registry, trades-only, live-like): 29 predictions, 29/29 …".

Two contextual facts the owner should know:
- The harness reads those days from the **Trade-Dashboard** store: `Trade-Dashboard/data/databento/NQ/<DATE>/mbp10.parquet` (`PLATFORM_REFACTOR_PROGRESS.md:211`), not the QL store. However, **all nine days are also present in the QL store** (verified, non-zero mbp10).
- Each day's calendar-prev file is present, so the full Globex window is captured and the prior-day reseed uses the real preceding trading day (`PLATFORM_REFACTOR_PROGRESS.md:217-220`).

---

## 3. Current training defaults (QL)

Primary config: `src/alpha_lab/agents/data_infra/ml/config.py` (pydantic models). Defaults verified at `12c50a5`.

### Core mode/scope
| Knob | Default | Where |
|---|---|---|
| `training_mode` | `"extrema_rebound_crossing"` — note: NOT dashboard_utility; production-contract runs set `dashboard_utility` explicitly | `config.py:353-356` |
| Session preset | **`all_to_ny`**: train [asia, london, ny], evaluate [asia, london, ny], production-gate **[ny]** | `config.py:255-260` (presets: `ny_only` :261-266, `asia_only` :267-272, `london_only` :273-278, `asia_london_only` :279-284, `all_sessions_all_gates` :285-291) |
| `SessionExperimentConfig` field defaults | training `["asia","london","ny"]` :222-225 · evaluation `["asia","london","ny"]` :226-229 · production_gate `["ny"]` :230-233 · report_session_breakdowns `True` :234-237 | `config.py` |
| `bar_type` | `"147t"` (valid: `147t`, `987t`, `2000t`, `1m`) | `config.py:333-336` |
| `instrument` / `tick_size` | `"NQ"` / `0.25` | `config.py:380` / `config.py:375` |

### Walk-forward fold schemes (three exist — pick deliberately)
| Scheme | Parameters | Where |
|---|---|---|
| `WalkForwardConfig` (pipeline config) | `train_days=60` (calendar), `test_days=20`, `gap_days=1`, `expanding=False` (rolling) | `config.py:146, :151, :156, :161` |
| `experiment/training.py` | `DEFAULT_TRAIN_DAYS=40` (trading), `DEFAULT_TEST_DAYS=5`, `DEFAULT_STEP_DAYS=5` | `src/alpha_lab/experiment/training.py:39-41` |
| `scripts/train_dashboard_model.py` (purged walk-forward) | `TRAIN_DAYS=40` (trading), `TEST_DAYS=5`, `STEP_DAYS=5`, `PURGE_DAYS=2` | `scripts/train_dashboard_model.py:42-45` |

### Eligibility / min-sample guards
| Guard | Value | Where |
|---|---|---|
| `MIN_TRAIN_EVENTS` (skip fold if fewer training events) | `30` | `src/alpha_lab/experiment/training.py:42` (enforced :163-168) and `scripts/train_dashboard_model.py:46` |
| Empty-test-fold skip | skip fold if 0 test events | `scripts/train_dashboard_model.py:97-99` |
| Session filtering | rows filtered to `training_sessions` before fit, `evaluation_sessions` for OOS | `scripts/ml_training_tab.py:222-230` |

### Model + selection + evaluation
| Knob | Default | Where |
|---|---|---|
| `model_type` | `"catboost"` | `config.py:170` |
| `iterations` / `depth` / `learning_rate` | `1000` / `6` / `0.03` | `config.py:174 / :179 / :185` |
| `loss_function` / `auto_class_weights` | `"Logloss"` (MultiClass for 3-class) / `"Balanced"` | `config.py:190 / :194` |
| `random_seed` | `42` (trainer hardcoded; also evaluator) | `src/alpha_lab/agents/data_infra/ml/model_trainer.py:155`; `model_evaluator.py:53` |
| RFECV | `rfecv_enabled=True`, `rfecv_min_features=5`, `early_stopping_rounds=50` | `config.py:198 / :202 / :207` |
| `confidence_gate` (inference) | `0.70` | `scripts/ml_training_tab.py:453` |
| Evaluation | `n_bootstrap=1000`, `n_permutations=500` | `scripts/ml_training_tab.py:303` |

Note: the recent production bundle `models/NQ_20260603_233847` was trained with **iterations=800, depth=4** (its `strategy.json` provenance block) — overrides, not the config defaults above.

### Dashboard-utility sub-config (the label/feature knobs a contract run consumes)
| Knob | Default | Where |
|---|---|---|
| `tp_points` | `15.0` | `config.py:308-312` |
| `sl_points` | `30.0` | `config.py:313-317` |
| `trap_mfe_min` | `5.0` | `config.py:318-322` |
| `interaction_window_minutes` | `5` | `config.py:323-327` |
| `level_proximity_pts` | `0.50` | `config.py:328-332` |
| `include_approach_features` | `False` (docstring says "27 approach-window … features" — the live list is 8; stale docstring) | `config.py:337-342` |
| `approach_window_minutes` | `90` | `config.py:343-347` |

---

## 4. Label policy values — production contract path vs test fixtures

The known fixture-vs-production divergence (sl=30/16:15 vs sl=15/17:xx), made exact.

### Production contract path (what the builder emits today)
The contract builder `src/alpha_lab/agents/data_infra/ml/strategy_contract.py:212-228` emits `label_policy` as:
- `tp_points` / `sl_points` / `trap_mfe_min` / `forward_bar_type` ← run config (`du.tp_points` etc., `strategy_contract.py:223-226`) — config defaults **15.0 / 30.0 / 5.0 / 147t** (`config.py:309/:314/:319/:334`)
- `entry_reference` ← `k.LABEL_ENTRY_REFERENCE` = **`"realistic_at_decision"`** (SC `src/strategy_core/constants.py:259`)
- `decision_offset_minutes` ← `du.interaction_window_minutes` = **5** (`strategy_contract.py:222`; SC single-sources `DECISION_OFFSET_MINUTES = DEFAULT_INTERACTION_WINDOW_MINUTES`, `constants.py:265`)
- `forward_cutoff` ← `k.LABEL_FORWARD_CUTOFF` = **`"17:00_US/Eastern_ny_close"`** (SC `constants.py:279`, built from `RTH_END = time(17, 0)` at `constants.py:152`)
- Executor flatten gate (no new entries): **`FLATTEN_TIME = time(16, 40)`** ET (SC `constants.py:273`; v3 changed 15:55 → 16:40)
- `resolution` = `"mae_first"`, `no_resolution_dropped` = `True` (SC `constants.py:247 / :274`)

**What the served production bundle actually contains** — `models/NQ_20260603_233847/strategy.json` `label_policy`: `tp_points: 15.0`, **`sl_points: 15.0`**, `forward_cutoff: "17:00_US/Eastern_ny_close"`, `entry_reference: "realistic_at_decision"`, `decision_offset_minutes: 5`. The **sl=15** is a **per-run override** (the config default is 30.0) — same override as `scripts/audit_NQ_20260602/reproduce_oos.py:63-64` (`tp_points=15.0, sl_points=15.0`; also `approach_window_minutes=15` at :70). So "production today" = sl 15 by trained-bundle convention, while the untouched config default remains sl 30.

### Test fixtures
| Parameter | Production (bundle / SC constant) | TL serving fixture `backend/tests/fixtures/strategy.json` | QL contract-test fixture `tests/agents/test_strategy_contract_nodrift.py` |
|---|---|---|---|
| `tp_points` | 15.0 (bundle; `config.py:309`) | **15.0** (:38) | 15.0 (:65) |
| `sl_points` | **15.0** (bundle; config default 30.0 at `config.py:314`) | **30.0** (:39) ⚠ | **30.0** (:66) ⚠ |
| `forward_cutoff` | **`17:00_US/Eastern_ny_close`** (`constants.py:279`) | **`16:15_US/Eastern_rth_close`** (:42) ⚠ | (not asserted as a literal in the fixture config) |
| `entry_reference` | **`realistic_at_decision`** (`constants.py:259`) | **`level_representative_price`** (:36) ⚠ — the superseded v1 anchor | n/a |
| `decision_offset_minutes` | 5 | 5 (:37) | 5 (`interaction_window_minutes`, :62) |
| `trap_mfe_min` | 5.0 | 5.0 (:40) | 5.0 (:67) |
| `forward_bar_type` | `147t` | `147t` (:41) | `147t` (:61) |
| `approach_window_minutes` | 90 default / 15 in the audit-trained bundle | 30 (section `feature_windows`, :117) | 30 (:63) |
| flatten time | 16:40 ET (`constants.py:273`) | n/a (not in contract) | n/a |

So the divergence in one line: **TL's serving fixture is a v1-flavored contract (sl=30, cutoff 16:15 rth_close, entry level_representative_price); the production v3 bundle serves sl=15, cutoff 17:00 ny_close, entry realistic_at_decision.** The old 16:15 cash-RTH close is explicitly superseded in SC (`constants.py:276-277`: "v3: RTH_END is now the 17:00 ET ny-session close (not the old 16:15 cash-RTH close)").

Other fixtures sharing the QL sl=30 values: `tests/agents/test_decision_repoint_parity.py:110-127`, `tests/agents/test_strategy_contract_repoint.py:44-49`, `tests/agents/test_ml_training_reporting.py:397-399`. Intentionally-small barriers (not a divergence): `tests/agents/test_stream_labeling_e2e.py:84-85` uses tp=2.0/sl=2.0 so a 1-hour slice resolves.

---

## 5. Feature set vs stub

### Default selected-features path
- **Selection mechanism:** RFECV over walk-forward splits when `rfecv_enabled=True` (default; `config.py:198`), run in `src/alpha_lab/agents/data_infra/ml/model_trainer.py:66-76`; falls back to all features if RFECV keeps fewer than `rfecv_min_features=5`.
- **Persistence:** the chosen list is saved as `selected_features` in the bundle's `metadata.json` (`model_trainer.py:179`).
- **Contract fallback:** when no `selected_features` are passed, the contract builder defaults to `LIVE_INTERACTION_FEATURES` (3 features) — `strategy_contract.py:147-149`.

### Full default feature-name list (11 = 3 interaction + 8 approach)
`LIVE_INTERACTION_FEATURES` — `config.py:28-32`:
`int_time_beyond_level`, `int_time_within_2pts`, `int_absorption_ratio`

`LIVE_APPROACH_FEATURES` — `config.py:17-26`:
`app_large_trade_vol_pct`, `app_trade_count`, `app_volume_acceleration`, `app_avg_trade_size`, `app_avg_tob_imbalance`, `app_max_spread`, `app_volatility_recent`, `app_volatility_ratio`

`LIVE_ALL_FEATURES = LIVE_INTERACTION_FEATURES + LIVE_APPROACH_FEATURES` — `config.py:35`. Reminder: `include_approach_features` defaults to **False** (`config.py:337-338`), so approach features must be opted into per run.

**What the served production bundle uses** (`models/NQ_20260603_233847/strategy.json` `feature_set.names`, RFECV-selected, 5): `int_time_within_2pts`, `int_absorption_ratio`, `app_avg_trade_size`, `app_large_trade_vol_pct`, `app_max_spread` (dropped `int_time_beyond_level`).

### Which features consume `quotes_in_window`, and stub status
`quotes_in_window` is the **Strategy-Core PlatformContext** quote accessor (PLAN §9.10), and it is **still a stub returning `()`**:
- Stub: SC `src/strategy_core/runtime/context.py:103-109` — `return ()` with `TODO(§9.10): back this with a bounded quote buffer so app_max_spread has live data when a quote-consuming plugin is wired.` (docstring confirmation at `context.py:19-20`).
- Of the 11 features, **two are quote-derived**: `app_max_spread` (max ask−bid at L0) and `app_avg_tob_imbalance` (bid_sz/(bid_sz+ask_sz) at L0) — `config.py:22-23`.
- **`app_max_spread`** is the only quote consumer with an engine formula: SC `src/strategy_core/decisions/features.py:206` — `def app_max_spread(quotes: Sequence[Quote], tick_size: float) -> float`. It is **in the served bundle's 5-feature set**, so any future plugin-path live computation through `PlatformContext.quotes_in_window` would see `()` until §9.10 is implemented.
- **`app_avg_tob_imbalance`** has **no engine formula at all** (QL `src/alpha_lab/agents/data_infra/ml/engine_decision.py:469-476`: only 3 approach features have engine scalar functions — `app_avg_trade_size`, `app_large_trade_vol_pct`, `app_max_spread`; "the other LIVE_APPROACH_FEATURES (acceleration, imbalance, volatility) have no engine formula yet"). The QL engine adapter returns only the 3 engine-backed approach features (`engine_decision.py:523-524`).
- The QL **research/training path is NOT stubbed**: it feeds real L0 quotes from the store via `_query_quotes` (DuckDB over mbp10) — `engine_decision.py:562-578` (returns `ts_event, bid_px_00, ask_px_00`; empty DataFrame when no view/no rows). So training-time `app_max_spread` values are real; the stub gap is on the SC live/plugin serving path only.

---

## 6. Compute feel — sample parquet sizes (no pipeline run; pyarrow metadata only)

| Day | mbp10.parquet rows | size | cols | row groups | ml_features rows |
|---|---|---|---|---|---|
| **2022-02-15** (old span) | 9,248,293 | 399.2 MB | 74 | 9 | 5,776 (0.9 MB) |
| **2026-02-18** (in last 30) | 19,107,919 | 905.3 MB | 74 | 19 | 8,203 (1.3 MB) |

A recent day is roughly **2× a 2022 day** in both rows and bytes (~19.1M rows / ~0.9 GB vs ~9.2M / ~0.4 GB). The recent day dir also carries derived artifacts from prior runs: `ml_features_<hash>.parquet` (8,203 rows each), six `ml_utility_<hash>.parquet` variants (3–4 rows each, including `d8e239c7` — the v3 hash), `ohlcv_1m.parquet` (1,031 rows), `ohlcv_1m_session.parquet` (1,091 rows).

Rough planning arithmetic: a 30-trading-day recent window ≈ 30 × ~19M ≈ **~570M mbp10 rows / ~27 GB scanned**; a 60-day window ≈ ~1.1B rows / ~55 GB.

---

*End of recon. Generated read-only at QL `12c50a5` · SC `ecbc15e` · TL `5a8d28a`; intentionally uncommitted.*
