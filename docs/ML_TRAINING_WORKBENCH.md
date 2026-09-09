# Streamlit ML Training Workbench

Updated: 2026-09-08.

The ML Training Workbench is the primary Quant-Lab UI for building local ML datasets, running walk-forward CatBoost evaluation, and saving runtime model bundles. It is mounted by `scripts/dashboard.py` and implemented in `scripts/ml_training_tab.py`.

The workbench is local-data only: it discovers and reads Databento-derived Parquet files from disk through `TickStore`. It does not call the Databento API.

## Opening a workspace

Completed IFVG searches evaluate post-warmup trades only. The parent-staleness
audit on 2026-09-08 corrected warmup leakage and the initial drawdown peak. Its
baseline now fails the 20-traded-day minimum (18 days); 240, 360 and 480 pass the
configured gates. Results always show the 95% expectancy interval, even when it
is not a required gate. A passing gate set does not establish a positive edge.
Time under water counts distinct days with executions below the net-R peak;
inactive dates are excluded. Time-block consistency is majority-sign agreement
across chronological groups of traded days, not necessarily agreement with the
overall P&L sign.

Use **Review this configuration's trades** from Results, or choose **Study
executions** in Trade review, to inspect that exact run's bars and executions.
Warmup trades are hidden by default and explicitly marked when included. The
chart uses New York time and offers the execution, parent and higher timeframes.
Save reviewer notes and labels through the existing **Review → Save Review** form.
These annotations remain separate from generated labels and execution outcomes.

The reproducible audit command is
`python scripts/audit_ifvg_search.py SEARCH_ID --output REPORT_DIRECTORY`.
Add `--apply` to publish corrected evaluations and advance the completed study's
frontier pointer after verification. Original artifacts and a backup of the state
are retained. Corrected candidate, decision and trade reports plus an explicit
`evaluation_scope.json` live alongside the new costed evaluation; the original
dataset reports retain their full replay scope, including warmup.

Start with `streamlit run scripts/dashboard.py` from the repository root. The app
opens **IFVG Lab**. Select **ML Training** in the page navigation for the workflow
documented below. **Dashboard Compatibility** retains the experiment workspace;
**Strategy Analysis** owns its pipeline settings and analysis tabs. Only the
selected workspace runs its presentation code.

## IFVG Lab workflow

**My studies** is the IFVG landing page. It shows the research question, dates,
saved status, and next action for each study. Filters include archived studies.
Unresolved legacy scope is labeled; implementation verification is excluded from
the normal list. An unreadable study never becomes a completed result.

Each visible study card includes three plain-English **Strategy rules** bullets.
Open the study for a fuller walkthrough of trading hours, setup, confirmation,
entry, stop/target, and when to abandon or skip a trade. Terms are explained on
first use, and important configured limits keep their units. Comparison studies
show common rules once and explain **What this study changes** separately. In
results, **Rules for this configuration** follows the selected configuration.

Draft rules are labeled as previews and use the saved, applicable settings;
unconfigured drafts ask you to choose a strategy. Frozen studies use verified
settings, and completed configurations use their exact saved effective strategy
configuration. Missing or unsupported settings produce a partial or unavailable
explanation rather than substituting newer defaults. Context studies distinguish
the underlying strategy from the prediction question. The descriptions are
generated locally without AI calls, replaying trades, or changing saved evidence.

Choose **New study**, then **Evaluate**, **Compare**, or **Search**. **More study
types** exposes feature/model research, prop feasibility, cross-firm strategy
research, full workflows, and the existing context feature lane. Configure only
the applicable steps; advanced settings remain available within those steps.
Opening a new form does not create a draft. In the guided strategy forms,
**Save draft** or a valid **Next**
saves it; subsequent changes to a saved form autosave. Duplicate questions and
configurations receive an advisory. Names must be descriptive rather than paths
or technical identifiers.

Continue a draft, explicitly run a ready study, or inspect a running study's saved
progress. Draft cards also offer **Details and actions** for rename, clone and
archive. A selected running detail view refreshes every five seconds. **Stop after
current work** requests safe cancellation. Failed/interrupted studies retain
supported resume actions; backend readiness and authorization checks still govern
execution. A full progress bar alone does not establish completion. If a workflow
never saved its progress, restore that evidence or clone its saved settings.

An explicitly approved strategy search can be saved with **Run study** enabled
without starting it. Approval follows the exact saved configurations, dates,
thresholds, costs and seed; changing those settings requires matching approval.
Only clicking Run freezes and launches the study. See
[saved strategy-search approval](IFVG_STRATEGY_SEARCH_APPROVAL.md) for the operator
record and execution contract.

### Feature and model studies from exact saved children

Under **New study → More study types**, choose **Feature and model study**.
This opens the source-based research form rather than a new strategy charter.

1. Select one or more **Saved strategy configurations**. The form displays each
   exact child's target in R, round-trip cost, evaluation dates and warmup count.
   **Exact source references** shows the saved replay, profile-section hash and
   v2 dataset references. Names are display labels; the full saved references
   identify the source. No child is selected automatically.
2. Set the evaluation window within the common saved evaluation dates. Targets
   and costs are inherited independently for each child. The research population
   includes all candidates in that window and excludes stamped warmup rows.
   Preflight displays the exact outcome cutoff: 17:00 ET on the final evaluation
   date, capped by the protected data boundary.
3. Choose core (B0), core plus structure (B1), or core plus structure/liquidity
   (B4) and the headline model. Results identify the frozen headline protocol
   and open its saved report first; all model rungs remain available for review.
   This review order does not rank performance. Optional MBP-1 compares B0 with B2 or B1 with B3;
   B4 has no combined MBP comparison in this preset. Source completeness without
   supporting evidence remains unknown. **Import reviewed MBP-1 completeness
   evidence** accepts a JSON receipt containing its source declaration and owner
   review. **Verify receipt** checks the exact subject, window and physical file
   hash without saving. **Import verified receipt** saves that evidence explicitly
   and refreshes preflight; it does not start research.
4. **Include market regime research** defaults to descriptive numeric 5-minute
   KMeans with three clusters and 50 requested bootstrap refits. Regimes are fit
   within each training fold. **Regime observations** also offers candidate entry
   decisions. Choose numeric bundle inputs for that grain; its registered minimum
   remains 150 training observations per fold. The form freezes the requested
   refits and optional training-fold percentile clipping. The configured-R model labels inherit each source
   target/cost; they are separate from historical version 1 captured labels.
5. Review the exact preflight: subjects, work cells, source/context preparation
   or reuse, warnings, blockers and activities. Unresolved evidence blocks
   authorization. The research plan runs source preparation, feature measurement,
   labels, folds, model fitting, predictions and artifact verification
   (S00–S10, S14, S15).
6. Enter a reviewer name, review the exact authorization statement, check
   **I authorize this exact research plan**, then click **Authorize and start
   research**. Changing the plan resets its approval controls. This creates a new
   namespace-bound research authorization; prior strategy-search approval cannot
   authorize it. Viewing the form or preflight starts no work.

The saved group appears in **My studies → Feature and model studies**. Inspect
each subject/cell independently; targets, costs and populations are not pooled.
**Refresh research progress** reads persisted status. **Stop after current work**
requests safe cancellation; **Resume authorized research** rechecks the frozen
authority and dependencies. A worker-start failure leaves the authorized group
available for inspection and resume.
During context capture, the selected cell's completed/total day count and current
trading date refresh every five seconds. The source section reports whether the
exact companion was reused or a context replay was performed.
Completed regime fitting/assessment executions can be verified and reused;
incomplete regime fitting attempts recompute on retry. MBP processing reads one
day at a time and verifies source hashes on each reload.

Incomplete cells retain their stage details and recorded artifact references.
**Research evidence status** is separate from worker completion: it displays
out-of-sample rows, valid folds, regime gate outcomes and insufficiency reasons.
Operational-only status is labeled until the saved report can be verified.
An evaluable result does not imply positive model lift.
**Trades and cohorts** shows the verified candidate population, excluded warmup
and out-of-window counts, configured labels, and scoped executed-trade rows with
exact identities. Executions join the included candidate IDs and must resolve
before the outcome cutoff; censored/unresolved exclusions remain visible.
**Open saved execution chart** selects that original trade;
chart labels keep their original captured policy. Select a candidate and click
**Inspect configured research chart**, or open **Research chart**, to inspect the
exact cell's saved forward-price evidence and configured research target/stop
annotations. This reader uses the selected search store's chart and source
references; it does not depend on the legacy global chart catalog. **Artifacts** shows saved
candidate features, labels and fold assignments even when fitting failed or no
valid folds could be built. Table previews are limited to 500 rows.
**Order flow** and **Market regimes** retain the full existing coverage,
missingness, comparison and assessment panels, using the selected cell's exact
references. Missing/corrupt evidence remains visible and cannot support a result.

A fresh descriptive result can proceed through **Market regimes → Review regime
feature eligibility**. Review the saved protocol inputs, eligibility gates,
requested/applied bootstrap counts and sample requirements. If the exact evidence
passes, enter a reviewer name, check the assessment-specific approval and click
**Approve regime feature use**. The action saves the owner decision and opens a
linked configuration; it performs no fitting. Failed or insufficient gates remain
blocked, and changes to the reviewed evidence reset the approval controls.

A linked regime feature study compares B0 with B7 using the exact regime decision,
owner authorization and assessment. The form verifies these frozen references;
missing authority blocks the follow-up rather than falling back to a descriptive
run. It retains the exact precursor grain, stage, numeric inputs, clipping,
bootstrap budget and cluster protocol instead of choosing new defaults.
Descriptive regime results include scoped trades that failed strategy-selection
gates and report those flags; the existing frontier requirements, including its
30-trade minimum, still govern promotion. The approved feature-only path does not include combined MBP/regime studies,
model-gated strategy replays, prop simulation, live execution or promotion into
production. These workflow instructions and implementation tests do not establish
that a real research study has completed or demonstrated model skill.

### Strategy results and trade review

Completed strategy results lead with the study verdict, limitations, net expectancy,
profit factor, maximum drawdown, and trade count. Model and prop studies use their
question-specific metrics. Detailed comparisons, calibration, folds, sensitivity,
and applicable prop scenarios are available on demand. Quantitative comparisons
require compatible evidence. Missing evidence remains unavailable; a proposed
threshold is never presented as an approved research standard.

Open **Trade review** directly, or follow an available exact supporting-case link.
Choose an entry opportunity/trade or a strategy setup, including setups without a
candidate. Empty filters and missing charts stop the selection rather than opening
another case. Chart options include overlays and point-in-time review; a bar table
and CSV provide an alternative to chart interaction. Actual execution, hypothetical
labels, model probabilities, and reviewer judgment remain separate. A review starts
unreviewed and is written only by **Save Review**. Supported study actions include
rename, clone, archive, and restore; frozen scientific settings remain immutable.
Saving updates the visible status immediately; subsequent edits become Unsaved.
Filters synchronize the selected case with its evidence. Model-result verdicts
identify their evidence section, so a data-integrity pass does not imply model skill.

The persistent **Exploratory research** label applies throughout IFVG. Technical
diagnostics are available only when starting the app with Developer mode enabled:

```powershell
$env:QUANT_LAB_DEVELOPER_MODE = '1'
streamlit run scripts/dashboard.py
```

Restart without that environment setting to remove the Developer route. Developer
contains Verification Center, data/system health, research diagnostics, and trade
diagnostics. Unknown health evidence never passes. Developer mode grants no
authorization, and ordinary research pages still hide technical panels. Research
workflows whose executor or label policy is unavailable remain blocked; preparing
or authorizing those dependencies is separate from this presentation change.

This workflow supersedes the earlier Experiments / Replay-Verifier / Data & Audit
navigation and always-visible technical disclosure descriptions. The browser retry
completed desktop flow checks and fixed issues found in actual use. The user removed
mobile responsiveness from acceptance. See the [desktop flow report](../reports/ifvg_browser_acceptance/20260907/DESKTOP_FLOW_REPORT.md)
for screenshots, tested actions and remaining live execution limits. Research runs
remain blocked where authorization, verified contracts or labels are unavailable.

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

### IFVG v2 correctness lab is not a training mode

The dashboard also exposes an IFVG v2 correctness surface, but it is
deliberately separate from `MLPipelineConfig`, walk-forward training, and model
save. Its active entrypoint displays:

- resolved profile name, qualification mode, runnability, active guards,
  overrides, section/evaluation hashes, sessions, anchor, and resolver;
- candidate counts, ordered block reasons, and counterfactual label
  distributions;
- eligible-decision counts and guard reconciliation;
- executed-trade performance only from validated resolved execution rows;
- invariant, count/FK, artifact-hash, and source-access audits.

The IFVG v2 tab does not expose search, tuning, feature selection, CatBoost,
validation/sealed evaluation, or custom recapture controls. Legacy saved IFVG
runs are shown only as non-executable candidate studies. The fixed verifier is:

```bash
python scripts/run_ifvg_repair_verification.py
```

It uses the explicit nonsealed January allowlist and writes once to
`data/ifvg_datasets/v2/<dataset-id>/exploration/`; an existing dataset identity
is never overwritten. This verification makes no profitability or edge claim.

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
- Optional `oos_predictions.parquet` with fold, timestamp, session, raw class, prediction, probabilities, a configured runtime-session gate flag, and the `0.70/ny` gate flag for offline error analysis. Since PROP-SIM P1 fresh saves also carry per-row labeler outcome columns — `max_mfe_pts`, `max_mae_pts`, `entry_price` (threaded from the training frame, never recomputed; NaN when the source frame/cache predates the column) and `resolution_type` (the ratified label mapping: tradeable_reversal → tp_hit, trap/blowthrough → sl_hit) — the prop-firm walker's inputs. Existing bundles are NOT retrofitted.

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
