# IFSM — Task A1: MenthorQ level lookup, entry gates, grouped reports

## Purpose

Make IFSM able to run with and without three level-driven rules — a dealer-gamma regime gate, a "nearest support is GEX 1"
gate, and a Chicago time-slot gate — and report every run by session × regime × time slot. This task implements; it does
not evaluate. Numbers from the smoke run at the end prove the machinery works and nothing more — write nothing about what
they mean.

What is deliberately **not** in this task (it is Task A2, later): the permanent, manifest-bound record family that would
archive the level context on every candidate record for the ML model. In A1 the context is computed at run time from the
lookup wherever it is needed (gates, reports) and written to a plain per-run export for review.

## What is in this package

- `TASK_A1.md` — this file.
- `decisions.md` — binding answers to the discovery's `questions_for_luis.md`. Cite it in the design register.
- `reference/discovery/` — the complete read-only discovery handoff of October 3, 2026 (`findings.md`, `extension_points.md`,
  `open_blockers.md`, `config_schema.md`, `tier2_schema.md`, `data_inventory.md`, `data_coverage.csv`, `harness_run_reference.md`,
  `questions_for_luis.md`, logs, the discovery tool). Read `findings.md` and `extension_points.md` first. Every file path
  and name in this task comes from them; "§n" below means a section of `findings.md`.
- `reference/data/` — the two CSV files the lookup reads, as of Sep 24, 2026 (use as test fixtures; the live copies are at
  `C:\menthorq\data\`).

Repositories (paths as discovery used them): Quant-Lab at `C:/Users/gonza/Documents/Claude-Quant-Lab/` (`QL/`); the
pinned Core snapshot at `C:/Users/gonza/Documents/Claude-Quant-Lab-Research-Artifacts/ifsm-research-core/7c7111e…/src/strategy_core/`
(`SC/`); the Core *repository* is the managed external source described in `QL/research/core/README.md`.

## Step 0 — start from a clean main (authorized)

The Quant-Lab tree was dirty at discovery (the ifvg_lab UI work: `scripts/ifvg_lab_*.py`,
`src/alpha_lab/agents/data_infra/ifvg/presentation/lab/`, `tests/agents/ifvg_lab/`, plus modified tracked files listed at the
end of `reference/discovery/run_log.txt`). Luis has authorized committing and merging it as-is:

1. On the current branch: `git add -A`, commit with a message naming the lab UI work, merge into `main`, push `origin main`.
2. Confirm `git status` is clean on `main`; record HEAD in `CHANGES.md`.
3. `git checkout -b feature/menthorq-level-context`. All Quant-Lab work happens on this branch.
4. Core: locate the Core repository per `QL/research/core/README.md` (the pinned snapshot folder is a checkout, not where
   edits go). Branch it `feature/menthorq-level-context` from commit `7c7111e398c083cf8e966e2e0c5aac8a41cc12c0`. If the
   repository cannot be located, stop and write `questions.md` — do not edit the snapshot in place.

Out of scope and not to be touched: the async-rithmic/protobuf conflict; the five known failing tests (they must not grow);
the sealed-date guards; capacity limits; dependencies.

## 1. The lookup ("provider")

Reads two files and answers one question: for a moment in time, what were that day's levels and regime?

**Inputs** (settings with defaults; schemas in `reference/data/`):
- `C:\menthorq\data\eod_gamma_levels_daily_wide.csv` — key `trading_date`; `source_eod_date`; level columns Call Resistance,
  Put Support, HVL, 1D Min, 1D Max, Call Resistance 0DTE, Put Support 0DTE, HVL 0DTE, Gamma Wall 0DTE, GEX 1 … GEX 10
  (points, NQ front-month scale).
- `C:\menthorq\data\daily_total_dealer_gamma_and_regime.csv` — key `trading_date`; `regime` ∈ {positive, negative};
  `total_net_gex`; `gex_percentile_1y`.

**Loading** (QL side, a new module next to `QL/src/alpha_lab/agents/data_infra/ifvg/day_artifacts.py`, name per repo
convention, e.g. `menthorq_levels.py`): parse once per run; reject duplicate `trading_date`, unparseable numbers, non-ISO
dates; empty cells are null; record `source_file_sha256` per file, `schema_version = 1`, `formula_version = "menthorq_eod_v1"`;
cache keyed by file content hash + policy, never filename alone (§8). Never fill a missing date from a neighbour or a later
file.

**Availability rule** (decision 4): for an aware-UTC timestamp `ts`, convert to `America/Chicago` with DST → `chi`. The
row for `chi.date()` is available iff `06:00 ≤ chi.time() < 17:00` and the level file has that date. Otherwise
`context_available = False`, `unavailable_reason` ∈ {`before_0600`, `after_1700`, `no_level_row`}. Regime is looked up
independently; a missing regime row gives `regime = "unknown"` and does not make the context unavailable.

**The answer ("snapshot")** — a pure, I/O-free value type on the Core side (e.g. `SC/strategies/ifvg_smc/menthorq_levels.py`):

| Field | Type | Rule |
|---|---|---|
| `trading_date` | date | Chicago date of `ts` |
| `source_eod_date` | date | from the level row |
| `source_file_sha256` | str | level file hash |
| `regime` | `positive` / `negative` / `unknown` | regime row, else unknown |
| `total_net_gex`, `gex_percentile_1y` | float / null | regime row |
| `levels` | mapping name → float points | the 19 level columns, null where empty |
| `implied_move_points` | float / null | `(1D Max − 1D Min) / 2`; null if either is null or the result ≤ 0 |
| `selected_instrument_id` | int / null | the source's selected contract for the logical day (`SC/data/databento_parquet.py::_front_month_instrument_id` at the lookup's seam); null if not reachable there — then note it in `questions.md`; never infer rolls from a calendar |
| `roll_flag` | bool / null | `selected_instrument_id` differs from the prior logical day's; null when either is null |

**Derived per-record values** — a pure function of (snapshot, price in points, timestamp). Price = honest entry price in
ticks × tick size, tick size from the instrument contract:

| Field | Type | Rule |
|---|---|---|
| `slot_chicago` | enum | `[08:30,10:00)` → `s1_0830_1000`; `[10:00,12:00)` → `s2_1000_1200`; `[12:00,13:30)` → `s3_1200_1330`; `[13:30,15:10)` → `s4_1330_1510`; else `outside_cash`. Half-open, Chicago local, DST-aware. |
| `hvl_side` | `above` / `below` / `at` / null | price vs HVL; null when HVL null |
| `nearest_above_names` | str / null | every level name whose value is strictly greater than price and equal to the smallest such value, joined with `\|` in file column order; null if none |
| `nearest_above_points`, `nearest_above_distance_points`, `nearest_above_distance_implied` | float / null | that value; value − price; distance ÷ `implied_move_points` (null if implied move null) |
| `nearest_below_names`, `nearest_below_points`, `nearest_below_distance_points`, `nearest_below_distance_implied` | as above | strictly less than price; nearest = largest such value; distance = price − value |
| `nearest_support_is_gex1` | bool / null | `"GEX 1"` ∈ `nearest_below_names`; null when no level below |
| `opening_move_signed` | float / null | `(bar_open_points − prior_cash_close_points) ÷ implied_move_points`; prior cash close = close of the last 1-minute bar with Chicago time ≤ 15:10 on the prior trading day that has bars; null if any input null |
| `opening_move_abs` | float / null | absolute value |

A level exactly equal to price is neither above nor below (decision 5).

## 2. Section fields and registry

Add to `SC/strategies/ifvg_smc/section.py::IfvgSmcSection`; register in
`QL/src/alpha_lab/agents/data_infra/ifvg/search/axis_registry.py` (`SEARCH_AXIS_REGISTRY_V1`, `AXIS_VALUE_REGISTRY_V1`):

| Field | Type | Default | Registered values |
|---|---|---|---|
| `menthorq_context_version` | `str \| None` | `None` | `none`: None; `eod_v1`: `"menthorq_eod_v1"` |
| `regime_gate_policy` | `Literal["off","positive_only","negative_only"]` | `"off"` | one value each |
| `regime_unknown_policy` | `Literal["allow","block"]` | `"allow"` | one value each |
| `nearest_support_gex1_block` | `bool` | `False` | `false`, `true` |

Slot gate: no new field. Register `entry_schedule_policy = "explicit_windows_v1"`, `entry_schedule_timezone = "America/Chicago"`
window values `slot_s1_0830_1000` `(("08:30","10:00"),)`, `slot_s2_1000_1200`, `slot_s3_1200_1330`, `slot_s4_1330_1510`,
`midsession_1000_1330` `(("10:00","13:30"),)` — same shape as the existing `daytime_chicago_0700_1555_v1` value.

Validation: a gate field off its default requires `menthorq_context_version = "menthorq_eod_v1"`; the section validator
rejects the combination with a named error. New registry values are `available; pending`, like the existing ones.

**Identity**: the four fields at their defaults are excluded from `ifvg_profile_hash` (and from `context_config_hash` /
`feature_schema_hash` if they are reachable there) by an explicit named exclusion written the same way the existing neutral
exclusions are (§2, `extension_points.md` last paragraph), so every historical identity is unchanged with the switches off.
Any non-default value is hashed and changes identity.

## 3. Entry gates

Location: the final candidate gate in `SC/strategies/ifvg_smc/reducer.py` (candidate construction near lines 2398–2496;
`execution_block_reasons`). The reducer stays pure: the snapshot for the candidate's final-entry availability timestamp is
passed in through the same door prior-day levels already use (`StrategyLevelState.set_static_levels` / replay `levels_for`;
§7, `extension_points.md` row 1) — extend that hand-off, do not read files inside the reducer. Blocked candidates stay visible
with their reason and emit no eligible decision (existing behavior).

| Gate | Blocks when | Block reason | Unknown handling |
|---|---|---|---|
| Regime | Inside 06:00–17:00 Chicago: `positive_only` and `regime ≠ positive`; `negative_only` and `regime ≠ negative` | `regime_gate` | Inside the window only: missing level row (`no_level_row`) or missing regime row (`regime = "unknown"`) → `regime_unknown_policy`: `allow` passes, `block` blocks with reason `context_unavailable` |
| Nearest support | `nearest_support_gex1_block = True` and `nearest_support_is_gex1 = True` | `nearest_support_gex1` | no level below or context unavailable → passes, recorded unknown |

Longs only; with `enable_shorts = True` the gates do not evaluate and the export says so. The new reasons join the existing
ordered `block_reasons` after the session/schedule reasons.

Decision 12 resolves availability precedence: outside 06:00–17:00 America/Chicago
(`unavailable_reason = before_0600` or `after_1700`), neither gate evaluates and the
candidate passes both, even with `regime_unknown_policy = "block"`. The export
records `regime_gate_blocked = null`, `nearest_support_gate_blocked = null`, and
`gate_status = "not_applicable_outside_hours"`.

## 4. Run-time context for reports, and the per-run export

No new record table in this task. Instead:

- **Report builders ask the lookup.** For every entry candidate, eligible decision and executed trade in a run, the report
  layer calls the lookup with the record's own availability timestamp and entry price and gets the snapshot plus derived
  values. Grouping keys: `entry_session` (existing), `regime`, `slot_chicago`. Unknown stays a bucket, never dropped.
- **Per-run export** `context_export.csv` in the run's report folder: one row per `entry_candidate` (`candidate_id`,
  `decision_id`/`trade_id` when they exist, `availability_ts_utc`, `entry_price_points`, `tick_size`, every snapshot and derived
  field from §1, `context_available`, `unavailable_reason`, `regime_gate_blocked`, `nearest_support_gate_blocked`, `gate_status`), with a
  header block carrying the run identity, `formula_version`, `schema_version` and both `source_file_sha256`. It is written
  only when `menthorq_context_version` is set. It is **not** part of the immutable dataset: not in the manifest, not hashed
  into any identity. Label it as such in the docs; it is review material and the input to Task B's analysis until A2 lands.

Nothing retrospective in the export — no labels, outcomes, MAE/MFE.

## 5. Reports

Declare the pooling policy and add session × regime × slot grouping to the concrete builders the baseline study uses —
executed-trade metrics (`trade_stats.py` consumers; the `search/costed_exports.py` session-rows block is the pattern) and
funded net cash (`QL/src/alpha_lab/propsim/funded/result.py` consumers). Every grouped table reconciles to its ungrouped
total (counts, points, cents). Add a study-level comparison table: one row per configuration (baseline; each switch alone;
all on), columns trades, wins, win rate, expectancy in points, net cash cents, and the same split by regime and by slot.
Roll-flag days are reported in their own rows and excluded from the comparison rows (decision 7).

## 6. Tests — only what changes

Unit tests, pure functions, in the existing layouts (`tests/agents/data_infra/ifvg`, pinned Core `tests/`), using
`reference/data/` as fixtures where real rows help:
- lookup: schema and duplicate keys; availability at 05:59 / 06:00 / 16:59 / 17:00 Chicago on a standard day and on one DST
  change day; slot boundaries at 08:30, 10:00, 12:00, 13:30, 15:10; nearest-level ties (two names at one price, including
  GEX 1), level equal to price, no level below; implied move null and ≤ 0; opening move with and without a prior close.
- gates (reducer level, fixture snapshot): each gate alone, both on, `regime_unknown_policy` both ways, blocked candidates
  keep their evidence and emit no eligible decision.
- gates (decision 12): an Asia-session entry outside the availability window with `regime_unknown_policy = "block"`
  is not blocked; both exported gate flags are null and `gate_status = "not_applicable_outside_hours"`.
- identity: one test — hashes unchanged with all four fields at default, changed with each non-default value.
- export: one test — row count equals the run's entry candidates; a spot check that exported derived values equal the
  lookup's.
- reports: one test — every grouped table reconciles to its ungrouped total.

Not required: resume/streaming identity tests (the lookup is passed through the existing level hand-off those tests already
cover), per-surface report tests, anything outside the files this task touches.

During the work run only the targeted suites: `tests/agents/data_infra/ifvg`, `tests/agents/ifvg_search`, Core `ifvg_smc`
tests. Run the full suite once, at the end, exactly as discovery did (`python -B -m pytest -q -p no:cacheprovider`): the
result must be 3,986 passed / 5 failed / 4 skipped plus the new tests, nothing else failing. Run the existing capacity gate
once with the switches on (the saved maximum transition is 257 bytes under its limit, `open_blockers.md`).

## 7. Parity and smoke run

Parity (before touching code): on unmodified `main` and the unmodified pinned Core, regenerate profile
`ifvg_v2_doc_default_fresh_static_1r` for the first 10 evaluation dates (Jan 13 – Jan 26, 2026) through the normal
preparation path into a scratch store outside the standard dataset directories; record every output table's SHA-256 and
`ifvg_profile_hash` (`e0f318732cb59d844ac14b5e3839862146e7da1f612f9884f767247f66dd39dd` per `config_schema.md`). After
implementation, all four fields at default, Core re-pinned: regenerate the same 10 dates the same way; every table hash and
the profile hash must match byte for byte. The source identity changes with the re-pin by design; output identity must not.

Decision 13 narrows byte-for-byte parity to the seven v2 execution tables plus
`label_source_1m`, and `ifvg_profile_hash`. For v3/context tables, compare content
with all columns carrying Core source identity, pinned commit, source hash or
run identity excluded. List the excluded columns per table, with reasons, in
`parity_report.md`. Every non-excluded column must match.

Smoke run: same profile with `menthorq_context_version = "menthorq_eod_v1"` and `holding_policy = scheduled_daily_close_v1`,
first 30 prepared evaluation dates (Jan 13 – Feb 24, 2026), two configurations — context on, no gates; all gates on
(`regime_gate_policy = positive_only`, `nearest_support_gex1_block = True`, `midsession_1000_1330` windows) — producing
`context_export.csv`, the grouped reports and the comparison table. Record wall-clock per configuration and per stage
(source load, bars, replay, labels, reports); Task B is sized from these. Do not open any date after June 10, 2026.

Both parity and smoke use the normal preparation path (`prepare_ifvg_development_pair.py`-style, into a scratch store)
and call the report builders directly. Neither needs the approval-gated research pipeline (`search/research_executor.py`,
`ifvg_search_job.py` workers). If some report can only be produced through that gated path, do not create a charter,
plan or approval record: produce every report reachable without it, note the rest in `questions.md`, and continue.

## 8. Docs

Per §21: `QL/AGENTS.md` documentation-maintenance items touched by new section fields and report changes; the design
register in `docs/IFVG_PLUGIN_DESIGN_ict_amended_ml_research_revised.md` (record as decisions, cite `decisions.md`); Core
`docs/DECISIONS.md` and `V3_COMPATIBILITY_MATRIX.md` for the new fields; a short page describing the lookup, the availability
rule, the export and its non-archival status; re-pin via `QL/research/core/current.json` and its README procedure so
`python scripts/run_ifsm_research_ui.py --check` passes against the new Core commit. The context formula/record contracts
are not extended in A1 (that is A2).

## Handoff folder — required files

`handoff/task_a1/`:

| File | Contents |
|---|---|
| `CHANGES.md` | Step 0 commits and HEADs (both repos); every file added/changed in both repos with what changed; final names of every new module, field, value, reason and column |
| `decisions_applied.md` | the 11 decisions from `decisions.md`, each with how it was implemented and the file that enforces it; deviations flagged |
| `core_commit.txt` | new Core commit hash, the `current.json` diff, output of `run_ifsm_research_ui.py --check` |
| `parity_report.md` | the before/after hash table for the 10-date run, scratch store paths, exact commands |
| `smoke_run/` | per configuration (two): resolved config, run identity, `context_export.csv`, grouped reports, the comparison table, stage timings |
| `capacity_gate.log` | the capacity gate run with the switches on |
| `docs/` | copies of every updated documentation file |
| `tests/` | copies of every new or changed test file |
| `targeted_tests.log` | targeted suites plus the new tests, verbose, final cycle |
| `test_run.log` | the single full-suite run at the end |
| `run_log.txt` | every command executed, with timestamps, ending with `git status` and `git log -1` for both repos |
| `questions.md` | anything unresolved, or any place two existing patterns conflicted and a choice was made |

## Rules

- Follow the repo's own patterns (`reference/discovery/extension_points.md`); where none fits, pick the closest and record it.
- Do not interpret smoke-run numbers anywhere.
- Do not change sealed-date guards, known test expectations, capacity limits or dependencies.
- Do not invent fields, values or names beyond this document without recording them in `CHANGES.md`.
- Commit on the feature branches only and push them to `origin`; do not merge to `main` after Step 0. No step in this task requires asking Luis for approval; the only stops are the two named in this document.

## Done when

Parity matches byte for byte on the 10 dates; both smoke configurations run end to end on Jan 13 – Feb 24, 2026 and produce
the export, grouped reports and comparison table; the new tests pass and the one full-suite run is otherwise unchanged; the
capacity gate passes with the switches on; `--check` passes against the new Core pin; every handoff file above exists.

## Deferred to Task A2 (do not start)

The manifest-bound `menthorq_context_v1` record family (one table keyed by `candidate_id`, threaded through Core
`context_records.py`/observer/config registry and QL `context_schemas.py`, `context_contracts.py`, `capture_driver.py`,
`context_feature_view.py`, feature blocks, readers/writers, `artifact_io.py` validation, `build_context_validity_report`),
the ML feature allowlisting, and the contract-document extensions. It replaces the per-run export once the rules have been
evaluated in Task B.
