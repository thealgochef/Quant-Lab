# RECON — UI training: what already exists vs. what's needed to reproduce the CLI D-036 bundle

**Mode:** read-only. No edits/commits/branch changes. Only write is this file (untracked).
**Repo:** QL `C:\Users\gonza\Documents\Claude-Quant-Lab`, branch `platform-refactor`.
**Target bundle:** `NQ_W3_20260613T055600Z` (cache tag `7850272e`) — the CLI recipe in the prompt
(`run_dashboard_session_experiment.py` + `_EXP_ARGV` `window.py:61-82` + save flags).

**Headline:** the training/save core already honors **all four** target behaviors for the CLI.
Three of the four gaps are **UI-wiring-only**; the fourth (**save-over-failed-gates**) is **already
exposed in the UI**. **No core change is required** to reproduce the bundle from the UI.

---

## A — What the UI already exposes (baseline, "no new work")

All controls below already feed the same `MLPipelineConfig`/`DashboardUtilityConfig`/`ModelConfig`
the CLI builds (UI build at `ml_training_tab.py:2150-2168`, train cfg at `:2314-2344`).

| CLI arg | UI control | file:line |
|---|---|---|
| (training mode = dashboard_utility) | radio "Dashboard Utility (3-class)" | `ml_training_tab.py:1856` |
| `--symbol NQ` | selectbox | `:1887` |
| `--bar-type 147t` | selectbox (list `_DASHBOARD_UTILITY_BAR_TYPES` `:41`) | `:1942` |
| `--start` / `--end` | date_input × 2 | `:1897` / `:1906` |
| `--tp 15` | slider 5–50 | `:1940` |
| `--sl 15` | slider 5–50 | `:1941` |
| `--interaction-window 5` | slider 1–15 | `:1948` |
| `--include-approach-features` | checkbox | `:1955` |
| `--approach-window 15` | slider 15–120 (shown when approach on) | `:1962` |
| `--preset all_to_ny` | selectbox | `:2007` |
| `--iterations 1000` | slider 100–2000 (default 500) | `:1978` |
| `--depth 6` | slider 3–10 (default 6) | `:1986` |
| `--save` | "Save Model" button | `:2695` |
| `--model-name <name>` | text_input | `:2676` |
| `--allow-failed-gates` | checkbox (see Gap 4) | `:2685` |
| (`--data-dir`) | text_input | `:1888` |

**Not exposed (the gaps below):** `--pin-features`, `--fold-scheme purged-days` + the four
`--fold-*-days` + `--min-train-events`. (`--rfecv` is exposed but mis-coupled — Gap 2.)

---

## B — Gap 1: pinned features

**Core already honors it (CLI path proves it):**
- CLI parses → `pinned_features = _parse_sessions(args.pin_features)` (`run_dashboard_session_experiment.py:169`).
- CLI passes it into the same training entry the UI uses:
  `run_walk_forward_training(dataset, config, "label_encoded", day_folds=day_folds, pinned_features=pinned_features)` (`:198-204`).
- Inside `run_walk_forward_training`, a pinned list becomes `selected_features` **exactly**, with
  RFECV **bypassed** (`ml_training_tab.py:359-369`): `selected_features = feature_cols` →
  `if pinned_features is not None:` validates the list is a subset and sets
  `selected_features = list(pinned_features)`; the RFECV branch is the `elif` (`:370`), so it never runs.
- Both per-fold and final models train on that `selected_features` (`:423`, and the saved final model
  at `:647-651`). So a real pinned list reproduces the exact 5-feature model + the `strategy.json` feature_set.

**Where the UI blocks it:** the UI train call passes neither kwarg —
`run_walk_forward_training(st.session_state["ml_dataset"], train_config, label_col)`
(`ml_training_tab.py:2365-2369`), so `pinned_features` defaults to `None` (signature `:214-220`).
**Passing a real list through that one call site is sufficient** — the parameter already exists; no
core change.

**Feature universe for a multiselect is available post-Build:** the dataset is held in
`st.session_state["ml_dataset"]` after the Build step (`:2183`), and its feature columns are exactly
`[c for c in dataset.columns if c.startswith(("int_", "app_"))]` — already computed inside training
(`:241`) and already displayed at the dataset preview (`feature_cols_display`, `:2235-2243`). A Train-step
multiselect can be populated from that same derivation (and the core validates the subset at `:361-369`,
so an out-of-universe pick raises cleanly).

**VERDICT: UI-wiring-only.** Touch: `ml_training_tab.py` — add a multiselect near the CatBoost
controls (`~:1976-1992`), and add `pinned_features=<list or None>` to the call at `:2365-2369`.

---

## C — Gap 2: RFECV coupled to the approach checkbox

- The coupling is **UI-side only**: `rfecv_enabled=ml_approach` in the train config (`ml_training_tab.py:2333`).
- `rfecv_enabled` is an **independent** core parameter (`ModelConfig.rfecv_enabled`, `config.py:197`); the
  core reads it at `:370` and it can be `False` while approach features are on. Nothing in the core ties
  RFECV to approach features.
- **Subsumed by Gap 1:** once `pinned_features` is passed, RFECV is bypassed regardless of
  `rfecv_enabled` (the `if/elif` at `:359-370`), so the mis-coupling becomes moot for D-036. For
  cleanliness you'd still replace `:2333` with an explicit RFECV checkbox (or set it `False` when a pin
  list is present).

**VERDICT: UI-wiring-only.** Touch: `ml_training_tab.py:2333` (decouple), optional checkbox at `~:1987`.

---

## D — Gap 3: fold scheme (purged-days vs calendar)

**Core already supports the choice (CLI path proves it):**
- CLI builds the dict `day_folds = {train_days, test_days, step_days, purge_days, min_train_events}`
  when `--fold-scheme purged-days` (`run_dashboard_session_experiment.py:160-168`) and passes it as
  `day_folds=` (`:202`).
- The splitter is selected **inside** `run_walk_forward_training`:
  `if day_folds is not None: splits = _purged_trading_day_splits(valid, **day_folds)` **else**
  `splitter = WalkForwardSplitter(config.walk_forward)` (`ml_training_tab.py:296-300`).
- `_purged_trading_day_splits` (`:168-211`) is the purged 40/5/5/2 + min-train-events scheme.

**Why the UI is calendar:** it never passes `day_folds`, so it falls to the `else` branch (`:299-300`).
That `if/else` is the **only** calendar hardcode in the UI training path. (The other `WalkForwardSplitter`
call sites — `honest_edge_audit.py:639`, `audit_NQ_20260602/reproduce_oos.py:98` — are standalone audit
scripts, not the Workbench.) To reach purged-days the UI just needs to pass a `day_folds` dict.

**Note (scope of effect):** the fold scheme does **not** change the saved final model weights (the final
refit uses `final_train_indices` = all training-session rows, fold-independent — `:287`, `:639-651`); it
changes the per-fold **OOS metrics + the confidence-gate result** written to `evaluation.json`. So it is
required for a byte-identical *bundle*, not for the *model weights*.

**VERDICT: UI-wiring-only.** Touch: `ml_training_tab.py` — add a fold-scheme selectbox + 5 numeric inputs
(`~:1976-1992`), build the `day_folds` dict, and add `day_folds=<dict or None>` to the call at `:2365-2369`.

---

## E — Gap 4: Save over failed quality gates

**Already supported AND already exposed in the UI — no work needed.**
- Core gate: `save_trained_model` calls `check_quality_gates(eval_result)` (`ml_training_tab.py:1527`;
  gate def `:1416-1474`, `all_passed` `:1473-1474`) and raises **unless** overridden:
  `if not quality_gates["all_passed"] and not allow_failed_gates: raise ValueError(...)` (`:1529-1536`).
- The `allow_failed_gates` kwarg **is** the `--allow-failed-gates` equivalent — the CLI passes
  `args.allow_failed_gates` into the same function (`run_dashboard_session_experiment.py:235`).
- The UI already wires it: computes `gates_result = check_quality_gates(ev)` (`:2552`), shows an
  "Allow save despite failed quality gates" checkbox when gates fail (`:2684-2693`), gates the Save
  button on it (`disabled=not gates_result["all_passed"] and not allow_failed_gates`, `:2698`), and
  passes `allow_failed_gates=allow_failed_gates` into `save_trained_model` (`:2709`).

**VERDICT: already done.** No file change required.

---

## F — Consolidated verdict

| gap | core already supports it? | UI-only or core-change | files / functions touched | rough size |
|---|---|---|---|---|
| 1 — pin-features | **Yes** — pinned wins, RFECV bypassed (`ml_training_tab.py:359-369`); CLI passes it (`run_dashboard_session_experiment.py:169,198-204`) | **UI-wiring-only** | `ml_training_tab.py`: multiselect `~:1976-1992` + `pinned_features=` at `:2365-2369` | small (1 widget + 1 kwarg) |
| 2 — RFECV coupling | **Yes** — `rfecv_enabled` independent (`config.py:197`), read at `:370`; pin makes it moot | **UI-wiring-only** (subsumed by Gap 1) | `ml_training_tab.py:2333` (+ optional checkbox `~:1987`) | trivial |
| 3 — fold scheme purged-days | **Yes** — `day_folds` branch + `_purged_trading_day_splits` (`ml_training_tab.py:296-300, 168-211`); CLI passes it (`:160-168, 202`) | **UI-wiring-only** | `ml_training_tab.py`: selector + 5 fields `~:1976-1992` + `day_folds=` at `:2365-2369` | small (5 fields + dict + 1 kwarg) |
| 4 — save over failed gates | **Yes** — `allow_failed_gates` in `save_trained_model` (`:1529`) | **Already exposed in UI** (`:2685`, `:2709`) | none | none |

**Is reproducing the CLI bundle from the UI entirely UI-wiring?** **Yes.** The core already performs all
four behaviors for the CLI through the *same* `run_walk_forward_training` / `save_trained_model` functions
the UI calls — the UI simply hardcodes two kwargs to their defaults (`pinned_features=None`, no `day_folds`),
mis-couples one independent flag (`rfecv_enabled=ml_approach`), and already exposes the fourth. **No
training/save core change is needed.**

**Single biggest item:** the **pin-features multiselect (Gap 1)** — it is the only gap that changes the
saved *model weights* and the contract feature_set, and wiring it also neutralizes Gap 2. Gap 3 is the
second piece needed for a byte-identical bundle (it fixes the recorded OOS/gate metadata, not the weights).

**Caveats to "reproduce" (not gaps, just preconditions):** the operator must also set the non-default
*values* already reachable today — SL 30→15, check approach features, approach-window 90→15, iterations
500→1000, the exact date window 2025-11-21→2026-02-13, and `model-name = NQ_W3_20260613T055600Z`. Byte-identity
further assumes the same `data_dir`/available dates and CatBoost determinism (seed is hardcoded to 42 in
`model_trainer.py:155`).

---

## VERDICT

Reproducing the CLI D-036 bundle from the UI is **entirely UI-wiring** — the training/save core already
honors pin-features, independent RFECV, purged-day folds, and gate-override for the CLI via the very same
functions the Workbench calls. Gap 4 is **already in the UI**; Gaps 1–3 are small additions to
`ml_training_tab.py` (widgets + threading two existing kwargs — `pinned_features=` and `day_folds=` — into
the `run_walk_forward_training` call at `:2365-2369`). **No core change.** The single load-bearing item is
the **pin-features multiselect** (it sets the model weights + feature_set and moots the RFECV coupling);
the purged-day-fold controls are the second piece for a byte-identical bundle. Everything else is just
setting values the UI can already reach.
