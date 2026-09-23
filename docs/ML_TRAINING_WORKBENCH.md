# Streamlit ML Training Workbench

Updated: 2026-09-22.

The research API also supports `geometry_comparison=True` with `B0_CORE`,
`mbp1_comparison=False` and no regime request. This freezes the comparison
`B0_CORE` versus `B0_GEOMETRY_CORE_ATR14_V1`; each arm runs the registered training
prevalence, logistic and bundle-aware CatBoost protocols. Prepare and verify each
saved child's own context once, inspect actual purged/embargoed fold populations,
validate the fixed additions, then freeze the exact request before starting
`scripts/ifvg_research_job.py`. Separate one-child groups allow the bounded batch
to continue after a child failure. Completed arms are verified and reused on
resume. The existing UI presets and production activation scope are unchanged.
Geometry labels distinguish the actual Core ATR14 bundle from historical ATR20
preparation, which was superseded before fitting; both retain their technical keys.
See `IFVG_GEOMETRY_CORE_ATR14_RESEARCH.md`. The historical batch delivery location
was `reports/ifvg_geometry_batch/20260909/`; those local outputs are not part of a
clean checkout. Use an available preserved study delivery for its evidence.

The ML Training Workbench is the primary Quant-Lab UI for building local ML datasets, running walk-forward CatBoost evaluation, and saving runtime model bundles. It is mounted by `scripts/dashboard.py` and implemented in `scripts/ml_training_tab.py`.

The workbench is local-data only: it discovers and reads Databento-derived Parquet files from disk through `TickStore`. It does not call the Databento API.

## Opening a workspace

Open **My studies → New study → Evaluate → Configuration** to modify a single
strategy configuration. The existing Configuration step selects one registered
value per setting; it does not add default comparisons or a Cartesian search.
There is no home-screen replication card or preset picker. Dates, warmup, seed
and gates remain in their existing study steps.
Parent/opposing distance **160 ticks** and reaction window **20 parent bars** are
now selectable alongside the existing timeout choices. Whole-setup lifetime
**180/240 processed 1m bars** and the two parent policies appear only when the
loaded Core actually implements them. `None` means unbounded and remains distinct
from “Keep baseline.”

For exact IFSM replication, run `python scripts/run_ifsm_research_ui.py` and open
`http://localhost:8502`. This separate process loads the preserved, hash-verified
research Core with the active-selected-HTF repair, uses current Quant-Lab UI code,
and writes only to `data/ifsm_ui_replication/`. It neither installs a package nor
repoints the normal dashboard or running geometry work. Each new study retains
its actual source identity and requires its own exact approval; selecting values
never starts a replay or inherits earlier approvals. See
[IFSM UI replication](IFSM_UI_REPLICATION.md) for every tested change and result.

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

Under **Study Configuration → Session Policy → Enabled entry sessions**, choose
**Asia only**, **London only**, **New York only**, or
**NY from 7am to 10:30am (Eastern Time)** alongside the existing default option.
The custom NY window allows new entry confirmations from **7:00am inclusive to
10:30am exclusive, New York time**, with daylight saving time handled automatically.
It is continuous through those hours, including time outside the standard doc-session
windows. Positions opened within the window continue under their configured exit
rules after 10:30am; other entry conditions still apply. Replay-chart document-session
bands follow that configuration's saved windows. The default sessions and
market-data session labels keep their existing definitions.

In a search, each selected entry-session option is a separate configuration;
selecting several compares those policies and automatically includes the registered
baseline. It does not combine their trading hours into one configuration. Inspect
the exact combinations in Review before approving and running the study.

For a parent-retest timeout backtest, choose **Search**, then under **Search axes
→ Staleness** select **Parent retest staleness timeout (1m bars)**. Available
values are **None (unbounded), 60, 90, 120, 240, 360, and 480**. Selecting
60/90/120/240 evaluates four challengers plus the unbounded baseline, with other
axes held at baseline. Each configuration requires a full sequential replay.
The clock counts processed one-minute bars in S1 after the current parent is
selected; a replacement parent restarts it. A retest at the configured limit
is still eligible, and expiry occurs on the following bar if still waiting.

Additional search choices are available under **Staleness** and **Risk
Admissibility**:

| Setting | Additional choices | Value used when empty with the standard baseline |
| --- | --- | --- |
| Opposing-FVG timeout | 60, 90, 120 one-minute bars | No timeout (unbounded) |
| HTF registry max age | 1, 2, 3, 4, 5 days | 15 days |
| Max executed trades per day | 1, 2, 3 | No daily trade cap |

The opposing-FVG timeout starts when the parent retest locks the setup and
expires on the first bar beyond the selected limit. HTF age counts calendar-day
differences between trading-day dates, including weekends; a zone at the age
limit remains, and older zones are removed. The execution cap counts trades
opened across all sessions of the trading day, resetting at 18:00 ET; it does
not count candidates or close an existing trade.

Every default option now displays its actual value and units instead of
"accepted doc-default baseline". **If left empty** beneath each menu shows the
value inherited from the selected baseline profile in both the normal
**Configure study** form and the Developer wizard. Empty means keep that
setting unchanged; it does not disable it or authorize a run. Other strategy
rules still apply when a particular timeout or cap is unbounded. An unavailable
baseline is labeled rather than replaced by an assumed default.

Selecting alternatives adds comparisons with the registered default, which is
included automatically. Selecting several settings tests all combinations:
the three opposing timeouts, five HTF ages and three daily caps together produce
96 configurations once their defaults are included (4 × 6 × 4). Settings left
empty keep their baseline values, except dependent fields explicitly controlled
by a composite option such as **Entry-near-parent gate**. Selection and saving
alone do not start a backtest.

Adding registered choices changes the registry hash included in exact strategy
approval. Previously approved plans need matching approval against the current
registry before launching again; saved results and approval records remain
immutable.

Continue a draft, explicitly run a ready study, or inspect a running study's saved
progress. Draft cards also offer **Details and actions** for rename, clone and
archive. A selected running detail view refreshes every five seconds. **Stop after
current work** requests a pause at the next work boundary: after the current
configuration and its saved metrics, between input-verification configurations,
or between finalization stages. A pending request stays visible until honored.
The screen reports completed, reused, failed, running, queued and stopped work
separately; attempted includes failed configurations. It labels input verification
separately from replay, and records timing for new attempts. Legacy missing times
remain explicitly unavailable.

An interrupted study stays paused until **Resume study** is clicked. Refreshing
or opening it does not launch work. Resume verifies the original identities,
reuses completed replay artifacts and saved metrics, and retries unfinished or
failed configurations. Missing legacy metrics can be recovered from verified
saved executions without replay. Changed source/data identities fail closed;
they require the original runtime or a separate study. Backend readiness and
authorization checks still govern execution. See
[study pause and recovery](IFVG_STUDY_PAUSE_RECOVERY.md). A full progress bar alone
does not establish completion. If a workflow
never saved its progress, restore that evidence or clone its saved settings.

Study preparation validates a shared day-artifact/seed chain once per worker,
instead of reconstructing the same bars and levels for every configuration.
Every new configuration still authorizes the exact paths and hashes both input
files for every approved date; changed bytes refuse reuse. Different artifact
settings or locations require a separate validation. Only verified input
references are retained, not decoded bars or strategy state. A resumed worker
starts with fresh validation. Both chronological replay drives and the final
input-integrity checks remain unchanged. This implementation carries a new
replay-source identity; historical source identities are never rewritten.

An explicitly approved strategy search can be saved with **Run study** enabled
without starting it. Approval follows the exact saved configurations, dates,
thresholds, costs and seed; changing those settings requires matching approval.
On **Configure study → Review**, the **Study approval** panel shows the exact
configuration combinations, date scope, costs and thresholds, plus the local
data metadata check. Resolve any listed evidence blockers, enter your reviewer
name, check **I approve this exact strategy study**, and select **Save study
approval**. This is one approval for the displayed strategy search; saving it
does not launch work. A settings change resets the confirmation. The panel
then confirms that the exact study is approved and **Run study** is available.
Only clicking Run freezes and launches the study. See
[saved strategy-search approval](IFVG_STRATEGY_SEARCH_APPROVAL.md) for the operator
record and execution contract.

### Feature and model studies from exact saved children

Under **New study → More study types**, choose **Feature and model study**.
This opens the source-based research form rather than a new strategy charter.

The older **Context feature study** and **IFVG Lab — Experiments** forms stop
before fitting when the selected pair lacks exact B0 stage/decision-bar sources.
Their message directs new work to **Feature and model study**. Saved context
studies remain readable. Explicit legacy partial-view reconstruction is limited
to tests and historical analysis; it is not a default or a source-validation
bypass for new studies.

Saved studies retain their original identities. Documentation-only Core updates
can be reused only with an immutable compatibility proof covering the pinned
source, current checkout/package and runtime environment, followed by exact Core
table reconciliation for a new context capture. Behavior-changing revisions remain
blocked. Historical dependency versions outside the saved evidence are unknown;
the new proof records that limitation rather than silently replacing the old pin.

Before treating a bounded acceptance as evaluable, inspect actual labeled fold
counts after scope, availability, setup grouping, purge and embargo. Candidate R6
still requires 150 training observations per fold; five-minute panel R6 requires
300 usable training bars and has separate stability/occupancy gates. Panel regimes
describe market conditions and map the latest completed bar to each candidate at
decision time. They do not provide more labeled trades. New fold preprocessing
keeps training-empty features unavailable throughout evaluation and persists that
decision for consistent saving/reloading.

When a regime group has fewer trades than its reporting floor, the UI states that
there are too few observations and shows the affected groups/threshold. This is
different from a missing or corrupt artifact. Passing the panel clustering gates
does not establish sufficient trade-by-regime samples or predictive feature lift.

The [bounded acceptance report](../reports/r5_r6_acceptance/20260908/ACCEPTANCE_REPORT.md)
records an actual parent-240 B0 ladder and five-minute KMeans run for February
23–June 10, 2026. Four supervised folds contain 37 held-out candidates; all seven
panel folds pass their descriptive gates. Reload reproduces saved predictions and
regime distances. Both fitted models score worse than training prevalence here.
Ten of 28 declared B0 fields are wholly empty because its historical feature view
read fields from the wrong record shape. This run therefore verifies the workflow
on the materialized inputs, not the complete intended B0 feature set. Candidate
R6, trade-by-regime reporting and MBP comparisons retain their sample/evidence
blockers. B0→B7 and S11 were not run; the report states the remaining requirements.

New B0 views use `ifvg_b0_selected_stage_projection_v2`: exact selected Core stage
records, verified geometry/event/ordinal/parent-clock parity and availability at
the candidate decision. Missing mapping evidence blocks a new study. Structural
entry-FVG nulls on retests remain valid and are reported separately from features
empty within a training fold. S06 persists the source and candidate-level repair
evidence. Fitted folds persist ordered raw/transformed feature schemas and verify
them with model reload. See [the field contract](IFVG_B0_PROJECTION_REPAIR.md) and
the local [controlled comparison report](../reports/b0_projection_repair/20260909/REPORT.md).
That comparison uses the same dates, folds, outcomes, costs, model settings and
seeds, and is exploratory evidence on an already-inspected period.

The controlled run completed with the same 37 OOS rows: Brier was 0.241401 for
prevalence, 0.375901 for logistic and 0.521333 for CatBoost. Both fitted models
worsened relative to their original partial-B0 scores (0.331452 and 0.458105).
All ten repaired fields are 143/143 populated and all eight predictive models
reload; neither software completion nor the repaired mapping establishes
sufficient evidence or predictive improvement. No tuning followed these results.

The independent [one-day MBP audit](MBP_ONE_DAY_CONVERSION_AUDIT.md) covers logical
February 23 and its February 22/23 physical files only. Conversion scalar/order
equivalence passes; completeness remains unknown and snapshot event redundancy is
unproved. Archived degradation warnings now reach coverage even alongside positive
local receipts. Other dates are not certified, and R5B remains blocked.

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


### HTF selection-cap experiment support (2026-09-12)

The registry now offers cap 2 as a pending value requiring exact strategy-search
approval; cap 1 stays the default. Evaluate One fixed settings expose this cap.
`search/htf_cap_experiment.py` creates the four fixed-profile drafts for the
240/90-wait comparison, preventing automatic default expansion. Per-timeframe
selection ranks both directions before taps and conflict/direction handling.
The new `selection_audit.py` companion preserves complete per-bar inventory
observations, pre-cap universes and HTF creations when the imported research
Core supports them, with strict coverage and tap reconciliation. These source
files enter replay identity; old artifacts and canonical tables stay immutable.

The task-local Core branch ports only active-selected-HTF physical tracking
after registry eviction, including schema-3 day seeds, and adds audit-only
selection observations. The installed/live Core pin is unchanged. Required
full replay uses the original 117-date bundle, four separately approved fixed
profiles and source-comparable controls; no other research policies apply.


### IFVG no-entry research (2026-09-14)

See `docs/IFVG_NO_ENTRY_DROUGHT.md` and the frozen task artifacts under
`../Claude-Quant-Lab-Research-Artifacts/archived-reports/ifvg_no_entry_drought_20260914/`. The isolated Core schema-4 day seed
preserves pending logical-close bars and the last minute decision. This repairs
seven partial-day deliveries without making bars available early, retains the
active-selected-HTF repair, and changes source/seed identities. Historical studies
and installed/live Core pins remain unchanged.

The separately approved research policy `htf_direction_selection_policy` retains
`mixed_direction_rank_v1` as default and adds `enabled_before_rank_v1`. The latter
filters disabled directions before HTF admission/ranking, while preserving physical
tracking, one setup/position and every unrelated setting. Its exact finite matrix
is B0/D0/B1/D1; there is no second mechanism or combined policy. The root registry
only exposes this field with a supporting research Core. Saved fixed drafts and
headless worker enumeration are the supported workflow.

New strategy-search v2 datasets optionally include manifest-bound
`entry_activity_report.json`: explicit evaluation calendar, actual-entry-day counts,
all consecutive zero-entry intervals/ties/censoring and adjacent elapsed/flat
intervals. Stored resolution-day economics and all original charter gates remain
separate; no activity statistic forces entries or supplies an acceptance threshold.
Historical datasets remain readable. This is same-sample research, with no fitting,
new data, June 11/holdout access or live promotion.

## One-hour / four-hour gap choice (September 18, 2026)

The IFSM research screen (python scripts/run_ifsm_research_ui.py) now exposes One-hour / four-hour gap invalidation in the existing Configuration step. Save a fixed policy for one configuration, or explicitly choose both for comparison. Old missing fields use the original wick rule. Worker environment and source verification use the same study engine. See IFVG_GAP_INVALIDATION_CHOICE.md for timing, save/reload, and restart behavior.

### IFSM research: daily close and corrected morning (September 18, 2026)

Use `python scripts/run_ifsm_research_ui.py` for the process-local research engine.
The study controls expose historical holding or the mandatory daily-close policy,
with a visible proposed 3:55 PM Chicago five-minute research buffer before the
owner's before-4:00-PM rule. Earlier scheduled market closes take priority.
The corrected **Morning - 7:00 AM to 10:30 AM Chicago time** preset changes actual
entry eligibility and worker settings. The old preset is visibly **Legacy morning
- 6:00 AM to 9:30 AM Chicago time (historical)**; use its explicit corrected-copy
action to create a new draft. Opening an old draft never migrates its meaning.
All-open-market and 7:00 AM–3:55 PM Chicago daytime presets are also selectable.
Closing a morning window does not close an existing position. Continuous market
context and protection remain active; daily/weekend entry locks apply separately.
The frozen 32-profile experiment, priced time-exit accounting, source-bound audit,
and operational limits are documented in `IFVG_DAILY_CLOSE_SESSIONS.md`.

### Local peer-review reports

Open `reports/IFVG_Daily_Close_Audit_Light/START_HERE.md` for the latest daily-close
study. Reports are local outputs, never committed. Keep only the findings,
settings, useful trade/equity/activity tables and charts needed for peer review.
Scripts, raw traces, runtime copies, working stores and duplicate ZIP extractions
do not belong in reports. Diagnostic CLIs use the sibling
`Claude-Quant-Lab-Research-Artifacts/` working directory by default.

The September 22 cleanup moved unique legacy study material and full audit
archives to `../Claude-Quant-Lab-Research-Artifacts/archived-reports/`, retaining
their original names. Use those preserved archives for detailed reconstruction;
the light folders contain the performance review. This is a storage change,
not a new replay, refit or revision to historical results.
