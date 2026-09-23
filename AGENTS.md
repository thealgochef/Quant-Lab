# Alpha Signal Research Lab - Agent Notes

Updated: 2026-09-22.

## Start points

- This repo is a Python ML training workbench for NQ/ES futures (Streamlit workbench + research CLIs). The old live-dashboard prototype (`src/alpha_lab/dashboard/`, `dashboard-ui/`) was deleted in the 2026-07 cleanup — it had no tests, no importers outside three frozen scripts, and pre-v3 session semantics; recover from git history if ever needed.
- Current canonical architecture doc: `ARCHITECTURE.md`.
- Docs index / stale-doc classification: `docs/README.md`.
- Streamlit ML tab workflow: `docs/ML_TRAINING_WORKBENCH.md`.
- Session experiment CLI: `scripts/run_dashboard_session_experiment.py`.
- Strategy-Core v3 cross-repo matrix: `../Strategy-Core/V3_COMPATIBILITY_MATRIX.md`.
- Main-branch Core pairing: `research/core/README.md`. Ordinary installation stays
  pinned to `a4e3303`; IFSM uses frozen `38825ed` and its verified source identity.
  Both are preserved in Core main. Do not substitute the latest Core main for a
  frozen study or change the installed dependency as a version-label cleanup.
  Keep the ordinary wheel's sibling source checkout equivalent to that package;
  merge newer Core main through a separate worktree when necessary.
- For model-training changes, start with `scripts/ml_training_tab.py` and `src/alpha_lab/agents/data_infra/ml/`, not the older generic multi-agent scaffold.
- IFVG exact-source R5–R6 research: `scripts/ifvg_research_pipeline.py`,
  `src/alpha_lab/agents/data_infra/ifvg/search/research_runs.py`, and
  `src/alpha_lab/agents/data_infra/ifvg/ml/research_evidence.py`.
- Saved-study reuse: `search/research_compatibility.py` proves the preserved source
  identity and binds current runtime evidence; schema-2 research context companions
  reference `research_core_compatibility`. Never rewrite historical identities.
- R6 new default: `median_impute_fold_empty_neutral_v2`; persist fold-local
  training-empty columns/schema and keep them unavailable during evaluation.
  Historical v1 protocols remain readable. Use `search/research_readiness.py`
  for actual labeled fold counts before bounded acceptance fitting.
- B0 mapping: `b0_projection.py`, contract `docs/IFVG_B0_PROJECTION_REPAIR.md`.
  New views require exact selected-stage audit evidence and decision-bar parity;
  `ifvg_b0_selected_stage_projection_v2` refuses unmapped advertised fields.
  Preserve retest entry-FVG structural nulls and historical partial-B0 artifacts.
  S06 persists `b0_projection_evidence.json`; new fitted supervised folds persist
  `feature_schema.json` with ordered raw/transformed names and dimensions.
  Logistic portable schema 2 retains schema-1 loading and frozen numeric behavior.
- Non-MBP geometry research: `features/geometry_core_atr14.py` and
  `search/research_geometry.py`; `B0_GEOMETRY_CORE_ATR14_V1` extends repaired B0 with
  three fixed ratios, retaining its raw features. The existing Core ATR14 convention
  uses a trailing arithmetic mean of 14 true ranges from completed TIME 1m bars,
  carrying source history across trading days and sessions. Exact source-bound
  artifacts live in `geometry_feature_artifacts`; validate coverage and freeze
  the exact batch plan before fitting. This is an offline bundle test, not proof
  isolating normalization or independent confirmation across overlapping children.
  No MBP, regime or activation scope is implied.

## Commands

Run from the repo root (`C:\Users\gonza\Documents\Claude-Quant-Lab`, Windows; system Python 3.13 — no venv activation step):

```bash
python -m pytest -q
python -m ruff check src tests scripts    # matches .github/workflows/ci.yml
streamlit run scripts/dashboard.py
```

Choose tests according to the change's actual impact. Do not run the full
regression suite by default, especially for isolated UI or presentation changes.
Run focused tests for the affected behavior and dependencies, plus applicable
lint checks. Run the full suite only when it is necessary to establish correctness
and targeted checks cannot cover the affected behavior. Once relevant checks pass,
do not broaden testing without a concrete unresolved correctness concern.

Project metadata requires Python `>=3.13` (`pyproject.toml`); `.python-version` pins `3.13.1`; CI runs 3.13. Do not hardcode other interpreter versions in docs or scripts.

## Data flow

- Databento batch import: `python scripts/process_batch_download.py <zip>`.
- Imported tick files are expected at `data/databento/{symbol}/{YYYY-MM-DD}/mbp10.parquet`, `mbp1.parquet`, or `trades.parquet`.
- The Streamlit ML tab reads local parquet only; it does not call the Databento API.
- Per-date ML caches are mode-specific:
  - Extrema: `ml_features_{config_hash}.parquet`
  - Dashboard utility: `ml_utility_{config_hash}.parquet`
- The dashboard-utility hash includes Strategy-Core engine/source semantics, so v1/v2/v3 structural changes do not reuse stale caches.

## Current Strategy-Core v3 contract

Dashboard-utility mode is the production-aligned research path and is single-sourced to Strategy-Core v3:

- `engine_version`: `strategy_core_engine_v3`
- Bars/features: trade-price/trade-print, NQ 0.25 grid, default `147t` touch bars.
- Sessions: ET `asia` 19:00→02:45, `london` 03:00→08:00, `ny` 09:00→17:00; 18:00 ET trading-day boundary.
- Levels: PDH/PDL from the full prior trading day; Asia/London levels from current-day sessions.
- Touch guard: level `available_from` is enforced; pre-availability self-touches do not consume zones.
- Label entry: realistic decision-time price at `touch_close + 5m`; not level price at touch.
- Cutoffs: flatten/no-new-decision at 16:40 ET; forward cutoff 17:00 ET.
- Gate: `tradeable_reversal`, `eligible_session="ny"`, confidence `0.70`.
- Research session experiments: train/evaluate all sessions by default, production-gate NY by default; scope is persisted into `evaluation.json`, `metadata.json`, and `strategy.json` as audit metadata.

Trade-Lab consumes Strategy-Core v3 through its adapter seam (repointed in the W1 one-surface migration); batch↔serving per-touch parity is gated by the W3b harness in Trade-Lab plus this repo's contract/no-drift acceptance tests.

## Evaluation rules

- Quality gates must use concatenated out-of-sample walk-forward fold predictions, not predictions from the final refit model.
- RFECV runs once before the walk-forward loop; the selected features must be reused by every fold model and the final saved model.
- Label purging removes training rows whose forward labeling window crosses into the test period.
- The saved runtime model is refit on all labeled rows after evaluation.
- Session experiment filters are applied after broad dataset generation: fold training, OOS metrics, and final refit use the configured training/evaluation scope.
- Preserve CatBoost native missing-value handling; do not blanket `fillna(0.0)` unless intentionally matching a separate runtime approximation path.

## Runtime/output artifacts

A Streamlit save writes:

```text
models/{model_name}/model.cbm
models/{model_name}/metadata.json
models/{model_name}/evaluation.json
models/{model_name}/strategy.json
models/{model_name}/oos_predictions.parquet  # when OOS rows are available
```

Treat `models/`, `catboost_info/`, `*.cbm`, cached `*.parquet`/`*.csv`, local imported data, and scratch chart HTML as generated local outputs unless a task explicitly says otherwise.

Repository ignore rules also cover local IFVG stores, job state, saved profiles,
drafts, review ledgers, and generated report/test workspaces. Ignoring them does
not make them disposable: preserve previous studies, approvals and audit evidence
on disk. Do not use `git clean -X` as repository housekeeping. Keep reusable test
fixtures under `tests/`, outside the ignored local stores. Archived documentation
and approved IFVG contracts remain versioned; local agent permissions do not.

### Report delivery and working artifacts (2026-09-22)

The owner wants small audit deliverables, without scripts or diagnostic dumps.
Never commit a `reports/` directory or its contents. Use `reports/` for final,
light peer-review deliverables only: findings, settings, trade/equity tables,
comparisons and relevant charts. No scripts, repositories, dependency trees,
raw event/bar dumps, test fixtures, caches or staging/extraction copies belong
there. Put new research working
stores, per-date replay traces, temporary test fixtures, builder scripts, staging
exports and clean-verification extractions outside this repository, under the
sibling `Claude-Quant-Lab-Research-Artifacts/<task-id>/` directory. Check actual
Git ignore behavior before creating any generated files inside the repository.
Pytest's managed OS temporary directory is also appropriate for synthetic test
fixtures; preserve its test-namespace safeguards when choosing temporary paths.
Never copy a repository, `.git` directory or dependency tree into `reports/`.
Reuse working artifacts instead of multiplying package copies. The owner
authorized the September 22 cleanup: relocate unique historical evidence outside
the repository and remove verified duplicates and disposable working files.
Preserve immutable bytes and identities; record relocations outside reports.
Do not use symlinks/junctions to put working stores back under reports. Keep
historical references labeled as historical, with current retrieval instructions.
Do not claim ignoring files reduces disk usage.

IFVG source-based research also writes immutable generated families under its
selected search store: `research_subjects`, `research_groups`,
`research_approvals`, `research_context_companions`, `research_cohorts`,
`research_labels`, `research_model_inputs`, `research_model_runs`,
`research_regime_executions`, and `research_replay_charts`. Preserve
historical replay/context artifacts and approvals. Each subject binds the exact
saved child and effective configuration sections; profile names and documentation
are not substitutes for those hashes. Targets/costs remain per subject, warmup is
excluded, and model inputs persist before fitting. Inspect these artifacts through
verified readers, including incomplete runs; do not infer success from file
presence. Real launches require the new exact-plan research authorization, which
does not inherit strategy-search scope. Implementation tests/preflight readiness
do not establish a completed real research run.

MBP completeness receipts require an explicit reviewed import; dry-run preview
saves nothing and cannot infer missing completeness. Regime feature eligibility
is a separate, review-ID-bound owner decision over saved assessment gates,
bootstrap evidence and sample floors. It configures a linked B0→B7 study but
grants no fitting until that new research plan is separately authorized.
Linked regime studies preserve the exact precursor numerical protocol. Research
candidate charts use their selected store and separately verified configured
labels; legacy fixed-1R annotations do not define research labels. MBP source
loading stays bounded to one day. Completed real S09a executions may be reused;
incomplete S09a fitting attempts recompute. Real folds purge from the first logical
test-day boundary even if that day has no candidates. Descriptive reports retain
all scoped trades with selection-gate flags; frontier promotion floors stay fixed.

The old `data/models/dashboard_3feature_v1.cbm` exporter is retained compatibility tooling, not automatically a v3 bundle. Canonical bundle location/checksum verification is deferred until the incoming local zip is available.

## Documentation maintenance

IFVG search evaluation corrections (2026-09-08): exclude stamped warmup trades
before metrics/gates and include the initial zero-equity drawdown peak. Current
costed evaluations use `metrics_policy_id=post_warmup_zero_peak_v2`. Legacy
artifacts remain immutable; `scripts/audit_ifvg_search.py` publishes scoped
replacement reports and backs up the operational state before updating its
frontier pointer. Search trade reviews join exact core/v2/input-bar references
and append only to the existing visual review ledger.

When code behavior changes, update docs in the same change. At minimum:

- Strategy/session/feature/label semantics -> `ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/pipeline_state.yaml`, and Strategy-Core docs/matrix.
- UI/workflow changes -> `docs/ML_TRAINING_WORKBENCH.md`.
- Model-bundle/cache output changes -> `ARCHITECTURE.md`, `docs/README.md`, and this file.


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

IFSM gap validity: docs/IFVG_GAP_INVALIDATION_CHOICE.md. The normal IFSM launcher uses the verified daily-close research Core only for the study process and workers; see docs/IFSM_UI_REPLICATION.md for reproducible setup. Original missing-policy defaults remain wick-based. New audit companions include gap_validity_events.parquet with independent source checks; do not interpret physical full traversal as policy invalidity under own_timeframe_close_v1. Incompatible policy/source seeds and resumes must be rejected; rebuild from authorized history. Preserve historical sources and installed/live pins.

### IFSM daily close and Chicago schedules (2026-09-18)

See `docs/IFVG_DAILY_CLOSE_SESSIONS.md`. The isolated
`../Strategy-Core-daily-close` engine adds `scheduled_daily_close_v1`, a proposed
3:55 PM Chicago research deadline (earlier scheduled close minus five minutes),
legal-reopen entry locking and schema-6 clock state. Installed/live pins and
structural calendar/anchors remain unchanged. Legacy holding is reproducible but
excluded from the new 32-profile experiment. Corrected morning is 7:00 AM–10:30 AM
America/Chicago; immutable `ny_0700_1030` remains historical 6:00 AM–9:30 AM Chicago.
New priced trade projection `core_executed_trade_priced_exit_v2` is chosen from
the holding policy before outcomes. `scheduled_close` uses actual partial-R price
and original costs once. Preserve `forced_exit_events`, full source coverage and
independent all-position/all-closure interval audits. Do not infer compliance
from exit time-of-day or conflate synchronous historical fills with broker delivery.
