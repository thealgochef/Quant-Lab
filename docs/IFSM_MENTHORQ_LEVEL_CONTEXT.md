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

The four new section fields are `menthorq_context_version` (default null),
`regime_gate_policy` (off), `regime_unknown_policy` (allow), and
`nearest_support_gex1_block` (false). Nondefault gate fields require
`menthorq_eod_v1`; the named validator error is `menthorq_gate_requires_eod_context`.
`MENTHORQ_NEUTRAL_PROFILE_FIELDS` excludes only default values from historical
profile hashes; every nondefault field is hashed. Quant-Lab's effective config
projection and name-free generated-profile hash use the same neutral map, keeping
historical settings and saved profile recognition stable. Typed sections and the
axis registry still expose all four fields. Rule descriptions use the existing
optional legacy defaults and add enabled context/gates to the entry-rule group.
Context configuration/schema
hashes do not accept section fields and are unchanged. Registry capabilities
are available with pending ratification; no launch authority follows.

Slots reuse atomic `enabled_entry_sessions` presets with explicit half-open
Chicago windows: `slot_s1_0830_1000`, `slot_s2_1000_1200`, `slot_s3_1200_1330`,
`slot_s4_1330_1510`, and `midsession_1000_1330`. Structural sessions and holding
deadlines retain their existing definitions.

Nearest levels use all 19 source columns, strictly above/below the entry price.
Ties retain names in file order; equality is neither side. The support gate
blocks only if GEX 1 is one of the tied nearest names. Null/nonpositive implied
move leaves normalized distances/opening moves null. Prior cash close is the
last completed one-minute bar close at or before 15:10 Chicago on the prior
logical trading day with bars. No expected minute is fabricated.

## Review export and reports

`menthorq_reporting.py::write_context_export` writes `context_export.csv` only
with context enabled: one row per run candidate, including warmup candidates,
with decision/trade links and entry-time context. A leading `#` JSON comment
holds run identity, formula/schema versions, both source hashes, and shorts
status. CSV readers use `comment="#"`. The export contains no labels, outcomes,
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

The current input seam exposes no selected instrument ID. The authorized
fallback records null instrument/roll fields, excludes no comparison rows and
sets the `roll_flag_unavailable` note column. Known roll-flagged rows, when
provided, remain in grouped totals and have separate rows, but not comparisons.
No calendar roll inference or back-adjustment is implemented.

New funded cash for a scratch replay requires the existing approved plan and
verified price-source path. A1 creates neither plan nor approval. Its smoke
comparison net-cash columns therefore read `not_produced_in_a1`. A separate
consumer, `build_funded_cash_groups`, reads an already saved verified funded
result and reconciles exact receipt-minus-purchase cents to `summaries_cents`.
It follows existing monthly cash-ledger aggregation: cash events group by their
own timestamps, with unknown entry session and explicit cash-event pooling.
It does not allocate account costs or payouts to entry trades.

## Reproduction and scope limits

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
