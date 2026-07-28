# RECON — QL training config: UI surface vs. the D-036 arg set (cache tag `7850272e`)

**Mode:** read-only. No edits/commits/branch changes. Only write is this file (untracked).
**Repos (branch `platform-refactor`):** QL `C:\Users\gonza\Documents\Claude-Quant-Lab`, TL `C:\Users\gonza\Documents\Trade-Lab`.
**Target:** dataset cache tag `7850272e` and bundle `NQ_W3_20260613T055600Z`.

---

## A — Does a QL training UI exist?

**YES.** A Streamlit "ML Training Workbench" tab can configure AND launch a full
dataset-build → train → save-bundle run, and it converges on the **same config path**
the CLI harness uses.

| Surface | What | Location |
|---|---|---|
| UI tab | `render_ml_training_tab()` (Streamlit) | `scripts/ml_training_tab.py:1851` |
| Mounted in | main dashboard `tab_ml` → `render_ml_training_tab()` | `scripts/dashboard.py` (ML Training tab) |
| Build button | "Build Dataset" → `build_utility_dataset(...)` | `ml_training_tab.py:2124`, calls at `:2174` |
| Train button | "Train Model" → `run_walk_forward_training(...)` | `ml_training_tab.py:2302`, calls at `:2365` |
| Save button | "Save Model" → `save_trained_model(...)` | `ml_training_tab.py:2695`, calls at `:2702` |
| Launch | `streamlit run scripts/dashboard.py` | — |

**Same config path? YES — byte-for-byte for the hash-affecting fields.** The UI's
`build_config` (`ml_training_tab.py:2150-2168`) constructs `MLPipelineConfig` with
`training_mode="dashboard_utility"`, a `DashboardUtilityConfig` built from the SAME six
widget fields the CLI resolver sets (`run_dashboard_session_experiment._resolve_config`,
`run_dashboard_session_experiment.py:127-152`), `tick_size=0.25`, and
`instrument=symbol`. Both paths leave `extrema`/`labeling`/`features` at `default_factory`
defaults and leave `DashboardUtilityConfig.trap_mfe_min`/`level_proximity_pts` at their
defaults. The dataset cache is keyed by `MLPipelineConfig.dataset_config_hash()`
(`config.py:384`), so an identically-filled config yields an identical tag regardless of
which surface built it. → **No divergence in the hash inputs.**

The CLI is the explicit non-Streamlit twin of this UI — its own docstring says so:
"This is the non-Streamlit path for the same configuration surfaced in the ML Training
Workbench" (`run_dashboard_session_experiment.py:4-5`).

**Trade-Lab does NOT launch QL training.** TL's API (`backend/src/trade_lab/api/app.py`)
only serves health/status/live/models/replay/ws — inference & replay, no training/dataset
route. TL's only QL-training reference is the **W3b parity harness**
(`backend/scripts/w3b/window.py:110-148`), which imports QL's CLI purely to *re-resolve and
validate* the D-036 config (asserts the resolved tag `== 7850272e`, `window.py:131-136`) —
it reproduces config for parity testing, it does not launch a train.

---

## B — The fingerprint: what feeds `dataset_config_hash()`?

`MLPipelineConfig.dataset_config_hash()` (`config.py:384-424`) SHA-256s this payload:

**IN the dataset hash (the cache-tag-determining set):**

| # | Hash input | Source | D-036 effective value |
|---|---|---|---|
| 1 | `training_mode` | `MLPipelineConfig.training_mode` | `dashboard_utility` |
| 2 | `extrema.model_dump()` (all `ExtremaConfig` fields) | defaults (`config.py:61-88`) | all default |
| 3 | `labeling.model_dump()` (all `LabelingConfig` fields) | defaults (`config.py:91-107`) | all default |
| 4 | `features.model_dump()` (all `FeatureConfig` fields) | defaults (`config.py:110-139`) | all default |
| 5 | `dashboard_utility.model_dump()` — **all 8 fields** | mixed (see C) | tp=15, sl=15, trap_mfe_min=5.0, int_win=5, level_prox=0.50, bar=147t, approach=True, app_win=15 |
| 6 | `tick_size` | hardcoded `0.25` (resolver `:150` / UI `:2166`) | 0.25 |
| 7 | `BAR_PRICE_SOURCE` | `strategy_core.constants` (env, not config) | repo constant |
| 8 | `LABEL_ENTRY_REFERENCE` | `strategy_core.constants` (env, not config) | repo constant |
| 9 | `PLATFORM_VERSION` | `strategy_core` (env, not config) | repo constant |
| 10 | literal `decision_pipeline=sc_runtime_stream_v1` | hardcoded (`config.py:422`) | fixed |

**NOT in the dataset hash (model/fold/eval/identity-only):**

- `walk_forward` (`WalkForwardConfig`: train/test/gap days) — explicitly "do NOT affect the cached feature matrix" (`config.py:386-388`).
- `model` (`ModelConfig`: iterations, depth, learning_rate, loss_function, auto_class_weights, rfecv_enabled, seed) — none enter the payload.
- `session_experiment` (`SessionExperimentConfig`) — explicitly outside; "Changing session scope should not invalidate … `ml_utility_*` caches" (`config.py:213-220`).
- `instrument` / `--symbol` — **NOT hashed** (only `tick_size` is). The symbol selects which day folder is read; it does not change the tag.
- The W3a riders `--fold-scheme`, `--fold-*-days`, `--min-train-events`, `--pin-features`, `--rfecv` — fold/feature-selection only; not hashed.

**Defaults `_resolve_config` fills in that are NOT in `_EXP_ARGV`** (and whether hash-affecting):

| Filled default | Value | Source | Hash-affecting? |
|---|---|---|---|
| `tick_size` | 0.25 | hardcoded `:150` | **YES** |
| `training_mode` | `dashboard_utility` | hardcoded `:129` | **YES** |
| `loss_function` | `MultiClass` | hardcoded `:148` | no (model) |
| `DashboardUtilityConfig.trap_mfe_min` | 5.0 | class default `config.py:318` | **YES** (via model_dump) |
| `DashboardUtilityConfig.level_proximity_pts` | 0.50 | class default `config.py:328` | **YES** (via model_dump) |
| `extrema`/`labeling`/`features` | all class defaults | `default_factory` | **YES** (all enter payload at defaults) |
| `learning_rate` | 0.03 | `ModelConfig` default `config.py:184` | no (model) |
| `auto_class_weights` | `Balanced` | `ModelConfig` default `config.py:193` | no (model) |
| `random_seed` | **42 (hardcoded)** | `model_trainer.py:155` | no (model) |
| `--train/test/gap-days` | 30 / 7 / 1 | argparse defaults `:74-76` | no (and ignored under purged-days) |
| `--data-dir` | `QL/data/databento` | `_DEFAULT_DATA_DIR` | no (selects rows) |
| session scope | `all_to_ny` preset (asia,london,ny / …/ ny) | preset `:60` | no |

---

## C — D-036 knob-by-knob: reachable from the UI?

Effective D-036 config = `_EXP_ARGV` (`window.py:61-82`) + the resolver/class defaults from B.
**Hash-affecting rows are bold (load-bearing for tag `7850272e`).**

| knob | D-036 value | feeds dataset hash? | source | UI-settable? | UI default = D-036? |
|---|---|---|---|---|---|
| **training_mode** | dashboard_utility | **YES** | resolver hardcode `:129` / UI radio `:1856` | yes (radio "Dashboard Utility (3-class)") | no (must pick; default is Extrema) |
| **dashboard_utility.tp_points** | 15 | **YES** | `--tp 15` / UI `:1940` | yes (slider 5–50) | **yes** (default 15) |
| **dashboard_utility.sl_points** | 15 | **YES** | `--sl 15` / UI `:1941` | yes (slider 5–50) | **no** (UI default 30) |
| **dashboard_utility.bar_type** | 147t | **YES** | `--bar-type 147t` / UI `:1942` | yes (selectbox) | **yes** (index 0 = `147t`, list `:41`) |
| **dashboard_utility.interaction_window_minutes** | 5 | **YES** | `--interaction-window 5` / UI `:1948` | yes (slider 1–15) | **yes** (default 5) |
| **dashboard_utility.include_approach_features** | True | **YES** | `--include-approach-features` / UI `:1955` | yes (checkbox) | **no** (default unchecked) |
| **dashboard_utility.approach_window_minutes** | 15 | **YES** | `--approach-window 15` / UI `:1962` | yes (slider 15–120, shown when approach on) | **no** (UI default 90) |
| **dashboard_utility.trap_mfe_min** | 5.0 | **YES** | class default (not in argv, not in UI) | no = fixed default | **yes** (both use 5.0) |
| **dashboard_utility.level_proximity_pts** | 0.50 | **YES** | class default (not in argv, not in UI) | no = fixed default | **yes** (both use 0.50) |
| **tick_size** | 0.25 | **YES** | hardcode `:150` / UI `:2166` | no = fixed | **yes** (0.25) |
| **extrema/labeling/features** | all defaults | **YES** | `default_factory` (neither surface exposes them) | no = fixed defaults | **yes** (both default) |
| **BAR_PRICE_SOURCE / LABEL_ENTRY_REFERENCE / PLATFORM_VERSION** | repo constants | **YES** | `strategy_core` (environmental) | n/a (not config) | yes (same repo state) |
| **decision_pipeline literal** | sc_runtime_stream_v1 | **YES** | hardcode `config.py:422` | n/a | yes (fixed) |
| symbol / instrument | NQ | no | `--symbol` / UI `:1887` | yes (selectbox) | yes (default NQ) |
| start / end | 2025-11-21 → 2026-02-13 | no (selects rows) | `--start/--end` / UI `:1897-1906` | yes (date pickers) | no (defaults to full available range) |
| session preset / scope | all_to_ny | no | `--preset` / UI `:2007` | yes (selectbox) | yes (default all_to_ny) |
| model.iterations | 1000 | no | `--iterations 1000` / UI `:1978` | yes (slider 100–2000) | no (UI default 500) |
| model.depth | 6 | no | `--depth 6` / UI `:1986` | yes (slider 3–10) | yes (default 6) |
| model.learning_rate | 0.03 | no | class default | no = fixed default | yes (0.03) |
| model.loss_function | MultiClass | no | resolver hardcode `:148` / UI hardcode `:2334` | no = fixed (utility mode) | yes |
| model.auto_class_weights | Balanced | no | class default `config.py:193` | no = fixed default | yes |
| model.random_seed | 42 | no | **hardcoded** `model_trainer.py:155` | no = fixed constant (never UI-exposed, never randomized) | yes (always 42) |
| **fold scheme** | purged-days | no | `--fold-scheme purged-days` | **NO — not in UI** | UI is always calendar `WalkForwardSplitter` |
| fold-train/test/step/purge-days, min-train-events | 40/5/5/2, 30 | no | `--fold-*-days`, `--min-train-events` | **NO — not in UI** | n/a |
| **pin-features** (exact 5) | int_time_within_2pts, int_absorption_ratio, app_avg_trade_size, app_large_trade_vol_pct, app_max_spread | no | `--pin-features …` | **NO — not in UI** | n/a |
| model.rfecv_enabled | False (overridden by pin) | no | `--rfecv` absent → False | yes-ish: UI forces `rfecv_enabled = ml_approach` (`:2333`) → **True** when approach on | **no** (UI couples RFECV to approach-features) |

---

## D — The decisive answers

### D.1 — Can a UI-configured run produce dataset tag `7850272e`? → **YES.**

Every hash-affecting knob is either UI-settable to the D-036 value or a fixed default
already equal to D-036:

- The three UI defaults that differ — **`sl_points` (30→15)**, **`include_approach_features`
  (off→on)**, **`approach_window_minutes` (90→15)** — are all UI-settable (sliders/checkbox),
  and 15 is inside the approach-window slider's 15–120 range.
- `tp_points`/`interaction_window`/`bar_type` UI defaults already equal D-036.
- `trap_mfe_min`, `level_proximity_pts`, `tick_size`, and `extrema/labeling/features` are
  fixed defaults identical on both surfaces.
- `strategy_core` constants are environmental (same repo → same values).

So setting the radio to "Dashboard Utility (3-class)", TP=15, **SL=15**, bar=147t, interaction=5,
**check Include approach features**, **approach window=15** reproduces tag `7850272e`. The UI
build path constructs the identical `MLPipelineConfig` hash input as the CLI resolver.
(Symbol/dates/session/model knobs don't affect the tag.)

### D.2 — Can a UI-configured run reproduce the BUNDLE? → **NO.**

The dataset tag matches (D.1), and the model HPs that the bundle pins are individually
reachable or fixed: **iterations 1000** (settable), **depth 6** (settable/default),
**lr 0.03** (fixed default), **balanced** (fixed default), **MultiClass** (fixed in utility
mode). **Seed is NOT a blocker** — `random_seed` is **hardcoded to 42** in
`ExtremaModelTrainer._build_classifier` (`model_trainer.py:155`); it is never UI-exposed and
never randomized, so it is bit-deterministic on both surfaces (the legacy `random_seed=42`
in `train_dashboard_model.py:192,274` is a *different, unused-for-D-036* script: iters=200,
depth=4).

The bundle is still **not reproducible from the UI** because two D-036 training knobs have
**no UI control**, and the saved final model depends on one of them:

1. **`--pin-features` is unreachable (the load-bearing blocker).** The UI's train call passes
   `pinned_features=None` (`ml_training_tab.py:2365-2369`). The saved bundle's model is
   `ExtremaModelTrainer(...).train(features[selected_features], …)` over `selected_features`
   (`ml_training_tab.py:647-651`). With no pin, `selected_features` = RFECV output or all
   ~30 `int_*`/`app_*` columns — **not** the exact 5 contract features in order. Worse, the UI
   **forces `rfecv_enabled = ml_approach`** (`:2333`), so enabling approach features (required
   for the tag) also turns RFECV **on**, guaranteeing a dynamically-selected feature set
   rather than the pinned 5. → different model weights AND a different `strategy.json`
   `feature_set`.
2. **`--fold-scheme purged-days` + fold params are unreachable.** The UI always uses the
   calendar `WalkForwardSplitter` (`ml_training_tab.py:299-300`). This does **not** change the
   saved final model's weights (the final refit uses `final_train_indices` = all
   training-session rows, fold-independent — `:287, :639-651`), but it **does** change the OOS
   metrics and the confidence-gate result written into the bundle's `evaluation.json`. So even
   the bundle's recorded gate/metrics would differ from D-036.

**Biggest blocker:** the missing `--pin-features` control (compounded by the UI hard-coupling
RFECV to the approach-features checkbox) — the UI cannot train the exact 5-feature model the
bundle pins.

---

## E — The real reproduce-D-036 recipe (CLI)

QL training for D-036 is **CLI/script-only** for the bundle. Interface:
`scripts/run_dashboard_session_experiment.py` (the resolver the W3b harness re-uses and asserts
against, `window.py:127-136`).

**Gotcha:** run with `PYTHONPATH=src` from the QL repo root (per project memory; `scripts/` and
`src/` must both be importable — the harness imports `ml_training_tab` and `alpha_lab.*`).

**Dataset tag `7850272e` only** (minimal hash-affecting args; build the cache, no model):
```
# from C:\Users\gonza\Documents\Claude-Quant-Lab  (PYTHONPATH=src)
python scripts/run_dashboard_session_experiment.py \
  --symbol NQ --bar-type 147t \
  --tp 15 --sl 15 --interaction-window 5 \
  --include-approach-features --approach-window 15 \
  --start 2025-11-21 --end 2026-02-13 --dry-run   # drop --dry-run to build
```
(`--preset`, `--symbol`, `--start/--end`, and all model/fold args do NOT change the tag; the
tag is fixed by `training_mode=dashboard_utility` + those `DashboardUtilityConfig` values +
`tick_size=0.25` + the `strategy_core` constants.)

**Full bundle `NQ_W3_20260613T055600Z`** (the ratified `_EXP_ARGV` + save flags):
```
python scripts/run_dashboard_session_experiment.py \
  --preset all_to_ny --symbol NQ --bar-type 147t \
  --start 2025-11-21 --end 2026-02-13 \
  --tp 15 --sl 15 --interaction-window 5 \
  --include-approach-features --approach-window 15 \
  --fold-scheme purged-days --fold-train-days 40 --fold-test-days 5 \
  --fold-step-days 5 --fold-purge-days 2 --min-train-events 30 \
  --pin-features int_time_within_2pts,int_absorption_ratio,app_avg_trade_size,app_large_trade_vol_pct,app_max_spread \
  --iterations 1000 --depth 6 \
  --save --model-name NQ_W3_20260613T055600Z --allow-failed-gates
```
- **Required (beyond the dataset args):** `--fold-scheme purged-days` + the four `--fold-*-days`
  + `--min-train-events`, `--pin-features <5>`, `--iterations 1000`, `--depth 6`, `--save`,
  `--model-name`. `--allow-failed-gates` is required because the W3 model fails its quality
  gates (research-only, negative OOS edge).
- **Defaulted (correct without flags):** `--learning-rate` (n/a — fixed 0.03), class weighting
  (Balanced), `loss_function` (MultiClass), `random_seed` (42, hardcoded). `--rfecv` must be
  *absent* (pin overrides it anyway).

---

## VERDICT

1. **A UI exists** — the Streamlit "ML Training Workbench" (`ml_training_tab.py:1851`,
   `streamlit run scripts/dashboard.py`) can build → train → save, and it builds config through
   the **same** `MLPipelineConfig`/`DashboardUtilityConfig` → `dataset_config_hash()` path as the
   CLI. TL launches no QL training.
2. **Dataset tag `7850272e`: REPRODUCIBLE from the UI** — every hash-affecting knob is
   UI-settable or a matching fixed default; the operator must just change three non-default
   widgets (SL 30→15, check approach features, approach window 90→15).
3. **Bundle `NQ_W3_20260613T055600Z`: NOT reproducible from the UI** — the UI cannot pin the
   exact 5 features or select the purged-trading-day fold scheme, and it hard-couples RFECV to the
   approach-features checkbox. (Seed is *not* the problem — it's hardcoded to 42 everywhere.)
4. **Single biggest blocker:** no `--pin-features` control in the UI — so the UI-trained model is
   RFECV/all-feature, not the bundle's exact 5-feature pinned model. Reproducing the bundle
   requires the CLI (`run_dashboard_session_experiment.py …`, `PYTHONPATH=src`).
