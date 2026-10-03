# IFSM MenthorQ level context — Task A1

Implemented October 3, 2026 from `C:\tasks\ifsm_task_a1_level_context\TASK_A1.md`
and its binding `decisions.md` (decisions 1–13 and implementation rulings).

## Lookup and entry behavior

`menthorq_levels.py` beside `day_artifacts.py` reads the two named EOD CSVs once
per run. Defaults are `C:\menthorq\data\eod_gamma_levels_daily_wide.csv` and
`C:\menthorq\data\daily_total_dealer_gamma_and_regime.csv`; callers may supply
paths to `load_menthorq_levels`. It validates exact column order, unique ISO
trading dates, earlier source EOD dates and finite numeric values. Empty numeric
cells remain null; empty regime cells are unknown. Parsing is cached by both
source byte hashes and the fixed schema/formula/availability policy, never by
filename. Each run has its own completed-bar cash-close history.

An aware timestamp is converted to America/Chicago with DST. Only that Chicago
date's level row is available from 06:00 inclusive to 17:00 exclusive. There is
no adjacent-date fill. Missing regime rows are independently unknown. Outside
hours both gates pass without evaluation, both exported block flags are null,
and `gate_status` is `not_applicable_outside_hours` (decision 12). Inside hours,
missing levels or unknown regime follow `regime_unknown_policy`: allow passes,
block emits `context_unavailable` when the regime gate is enabled.

Core's pure snapshot/derived/gate module is handed through the existing level
callback as `IfvgLevelInputs`. The reducer uses the final candidate availability
and honest entry ticks multiplied by the instrument tick size. Blocked
candidates remain visible and emit no eligible decision. New reasons follow
the existing session/schedule reasons: `regime_gate`, `context_unavailable`,
`nearest_support_gex1`. If shorts are enabled, both MenthorQ gates are bypassed
and the export states `not_applicable_shorts_enabled`.

The five section fields are `menthorq_context_version` (default null),
`regime_gate_policy` (off), `regime_unknown_policy` (allow), and
`nearest_support_gex1_block` (false), and `nearest_support_universe` (`all_19`).
Nondefault gate/universe fields require
`menthorq_eod_v1`; the named validator error is `menthorq_gate_requires_eod_context`.
`MENTHORQ_NEUTRAL_PROFILE_FIELDS` excludes only default values from historical
profile hashes; every nondefault field is hashed. Quant-Lab's effective config
projection and name-free generated-profile hash use the same neutral map, keeping
historical settings and saved profile recognition stable. Typed sections and the
axis registry still expose all five fields. Rule descriptions use the existing
optional legacy defaults and add enabled context/gates to the entry-rule group.
Context configuration/schema
hashes do not accept section fields and are unchanged. Registry capabilities
are available with pending ratification; no launch authority follows.

Slots reuse atomic `enabled_entry_sessions` presets with explicit half-open
Chicago windows: `slot_s1_0830_1000`, `slot_s2_1000_1200`, `slot_s3_1200_1330`,
`slot_s4_1330_1510`, and `midsession_1000_1330`. Structural sessions and holding
deadlines retain their existing definitions.

Nearest levels default to all 19 source columns, strictly above/below the entry price.
Task B's `studied_8` choice limits both nearest sides and the GEX 1 gate to
Call Resistance, Put Support, HVL, 1D Min, 1D Max and GEX 1–3. `all_19` remains
hash-neutral and preserves A1 behavior; `studied_8` enters the section identity.
Ties retain names in file order; equality is neither side. The support gate
blocks only if GEX 1 is one of the tied nearest names. Null/nonpositive implied
move leaves normalized distances/opening moves null. Prior cash close is the
last completed one-minute bar close at or before 15:10 Chicago on the prior
logical trading day with bars. No expected minute is fabricated.

Task B records `selected_instrument_id` (integer or null) and source `raw_symbol`
in the selected preparation job's `catalog.json`, keyed by logical day. The
canonical reader's instrument-only prescan selects the contract in that day's
primary source partition and the same logical window row groups. Existing 2026
catalog entries are backfilled without bar construction. `register_day_artifacts`
reads the catalog; roll flags compare the preceding registered day with bars:
different IDs true, equal IDs false, either missing null. Empty days do not
become predecessors. Receipts never enter immutable day metadata or datasets.

## Review export and reports

`menthorq_reporting.py::write_context_export` writes `context_export.csv` only
with context enabled: one row per run candidate, including warmup candidates,
with decision/trade links and entry-time context. A leading `#` JSON comment
holds run identity, formula/schema versions, both source hashes, and shorts
status. CSV readers use `comment="#"`. The `nearest_support_universe` column
follows `nearest_support_is_gex1`. The export contains no labels, outcomes,
MAE or MFE. It is review material: not manifest-bound, not hashed into dataset
identity, not an archived candidate family, and not ML-allowlisted. Task A2 is
deferred in full.

Ordinary `prepare_ifvg_development_pair` writes enabled exports under
`data/ifvg_preparation_jobs/<resolved_profile>/reports/<v2_artifact_id>/`.
Persisted preparation uses its selected job root and `job_label`; both helpers
accept `report_root` for an explicit review destination and return
`PreparedIfvgPair.context_export_path`. Default-off runs create no review folder.
The destination is checked before writing and must remain outside the selected
immutable v2/v3 roots. Its header uses the existing artifact IDs, verified
manifest hashes, section/evaluation hashes and Core source receipts.

`run_ifvg_v2_evaluation` can consume the same run provider retained by capture.
It adds candidate/decision counts and execution metrics by session × regime ×
slot. The direct builder uses the existing post-warmup candidate-entry cohort,
execution validation and priced scheduled-close economics; costs are applied
once. Unknown buckets are retained, and counts/points reconcile to totals.
Comparisons support any set of actually executed configurations, with regime
and slot splits; they never synthesize measurements for unexecuted arms.

The historical A1 input seam exposed no selected instrument ID. Its authorized
fallback recorded null instrument/roll fields, excluded no comparison rows and
sets the `roll_flag_unavailable` note column. Known roll-flagged rows, when
provided, remain in grouped totals and have separate rows, but not comparisons.
No calendar roll inference or back-adjustment is implemented. Task B uses the
catalog plumbing described above. Decision 21 supersedes the roll exclusion
for its study reports; all points and funded inputs retain roll-day trades.

New funded cash for a scratch replay requires the existing approved plan and
verified price-source path. A1 creates neither plan nor approval. Its smoke
comparison net-cash columns therefore read `not_produced_in_a1`. A separate
consumer, `build_funded_cash_groups`, reads an already saved verified funded
result and reconciles exact receipt-minus-purchase cents to `summaries_cents`.
It follows existing monthly cash-ledger aggregation: cash events group by their
own timestamps, with unknown entry session and explicit cash-event pooling.
It does not allocate account costs or payouts to entry trades.

## Reproduction and scope limits

Task B's separate preparation store uses `prepared_store.py`. Raw input roots
and artifact cache roots are distinct. Its explicit registration binds the
original source allowlist, per-day bars/levels hashes and entering preparation
seeds for each segment. The 2025 logical inventory is the supplied weekdays;
canonical Monday input may read the preceding Sunday physical partition within
the authorized window. The first ten logical weekdays, June 2–13, are warmup.
The three supplied missing weekdays remain missing. The 2026 segment preserves
its original preparation chain and cache bytes; joining stores does not claim
that those bars/levels were rebuilt from 2025 history. Runtime strategy/context
state continues across the selected replay dates.

Saved section verification accepts the original exact mapping hash or the
established Core neutral behavior projection of present saved keys. It never
fills current defaults, rewrites historical records, or ignores active values.
The saved configuration reader and spec-bound strategy-result loader use the
same verifier; returned section dictionaries retain their original fields.

`search/task_b.py` binds the exact thirteen owner-specified configurations,
registered input receipts, existing cost/funded profiles, and decisions 20/21.
The approved scope also binds the two original EOD/regime source hashes and the
completed preparation catalogs. Original source hashes enter the existing
required-input provenance seam; review-export bytes never enter replay/model
inputs or immutable output manifests. Funded evidence binds the exact print
files actually read and verifies those named hashes before reuse.
The standard approval review/record path and `ifvg_search_job.py` worker launch
`search/research_executor.py`'s Task B wiring. Its explicit stage plan validates
inputs, resolves profiles, captures sequential strategy and FSM audit evidence,
runs the current independent account-driven funded ledger, writes factual
tables and verifies saved evidence. Feature construction, model labels, folds,
fitting, predictions, bootstrap and interpretation are not in that stage plan.
After immutable child publication, costed evaluation and lineage persistence,
Task B releases completed in-memory captures before replaying the next child.
Its later audit, funded and reporting stages use verified saved readers.

The funded adapter assigns an existing closure and its mandatory deadline to
the canonical logical interval from the preceding 18:00 ET boundary to the
current boundary. A late holiday closure belongs to the next logical day; it
does not replace the normal deadline on the prior civil date. The adapter
requires exactly one close/deadline per evaluated logical day. Timed account
day-end/release events follow the existing selected evaluation-calendar
convention; processing-completion timers continue across calendar gaps.
Funded output metadata follows the existing comparison adapter: pinned sections
without `exit_policy` use the established `fixed_target_v1` fallback.
The existing immutable store vocabulary includes `task_b_artifacts` for its
funded-account and reconciled-table output envelopes.

`menthorq_study_reporting.py` keeps funded cash at configuration/firm scope.
By-session/regime/slot and cell tables carry points only; the NY comparison
block is points only. Monthly points use Chicago entry month; monthly cash uses
Chicago cash-event month and explicit column labels. All roll-day entries remain
in both replay paths; separate roll rows report point metrics and cash events.
The report requires all thirteen configurations, every existing funded firm,
exact cash-cent reconciliation and baseline/context-on execution parity. The
report-only provider also supplies the baseline/NY-only review exports, without
enabling their context gates. These exports do not enter manifests or ML inputs.

The external A1 harness calls the same bounded v2/v3 capture and report builders
used by normal preparation, with explicit scratch input/output paths. The
default CLI/store paths are unchanged. Trusted prepared bars/levels are copied
byte-for-byte and validated against their original allowlist/seed receipts;
only the requested prefix is replayed. Timings declare prepared-cache reuse.
There is no approval-gated worker, model fitting or extra benchmark sweep.

Decision 13 requires byte hashes for all seven v2 execution tables plus
`label_source_1m`, and the original profile hash. For v3, only proven source/run
identity columns are excluded from content comparisons; per-table exclusions
and reasons are in the handoff parity report. All other columns must agree.

The existing calendar counts both January 19 and February 16 partial sessions.
Its first 30 evaluation dates starting January 13 end February 23, 2026;
February 24 would be a 31st date. A1 uses the literal first-30 selection and
records this discrepancy in the handoff, without dropping a trading session.
Smoke numbers demonstrate machinery only and carry no evaluation conclusion.
