# IFVG FSM Evidence Contract — Traceability Matrix

Companion to `IFVG_FSM_AUDITABILITY_REPAIR_PLAN.md`. This document PROVES, column
by column, which required evidence fields are already carried by in-band emission
kinds (reused via QL partition, zero SC change) and which require the new SC audit
channel. Every row: prompt requirement → source symbol (file:line at SC
`f16d27d0` pre-change) → audit table → column → native/derived →
required/nullability. Verified against `records.py` / `reducer.py` /
`fvg.py` / `replay.py` as read on 2026-08-04.

Legend: **native** = value exists on the emitted record today; **derived** =
computed by QL from native values without new SC emission; **new** = requires the
new audit channel. `NOT NULL` unless marked nullable.

## 0. Stamps required on EVERY audit row (§8)

| Requirement | Source | Table column | Kind |
|---|---|---|---|
| entering_seed_hash | `seed_hash` state.py:85-87; stamped QL-side (capture_driver flatten) | `entering_seed_hash` | native (QL stamp) |
| is_warmup / days_of_htf_history | chain loop dataset.py:496-497 | `is_warmup`, `days_of_htf_history` | native (QL stamp) |
| audit_schema_version = 1 | new `IFVG_AUDIT_RECORD_SCHEMA_VERSION` | `audit_schema_version` | new |
| profile hash / section hash / policies | `RecordEnvelope` records.py:160-180 (all reused kinds); audit records carry the same envelope | `envelope_profile_hash`, `envelope_section_config_hash`, … | native |
| cursors | `bar_cursor` records.py:86-91 | per-table cursor columns | native/new (see per-table) |
| nulls carry explicit missing reasons | contract rule | `<col>_missing_reason` companions where a nullable column is semantic | new (QL) |

## 1. Reused in-band kinds (QL partitions them out of the SAME trace)

### 1.1 `htf_tap` → `ifvg_audit_htf_tap`

Emitted at reducer.py:1051-1078 (free slot) and :1118-1146 (slot occupied).

| Prompt requirement | Source symbol | Column | Kind | Nullable |
|---|---|---|---|---|
| tapped gap full geometry | `HtfTapRecord.fvg` records.py:193 (Fvg, flattened `fvg_*`: fvg_id, timeframe_seconds, direction, gap_low/high_ticks, size_ticks, a/c_bar_id, a_open_ts_utc, confirmed_ts_utc, trading_day) | `fvg_*` | native | no |
| trade direction | `.direction` :194 | `direction` | native | no |
| penetration on the tap bar | `.penetration_ticks` :195 (`penetration_ticks` fvg.py:259-266) | `penetration_ticks` | native | no |
| CE reached | `.ce_reached` :196 | `ce_reached` | native | no |
| age at tap | `.htf_age_seconds` :197 | `htf_age_seconds` | native | no |
| remaining fraction | `.remaining_fraction` :198 (`FvgState.remaining_fraction` fvg.py:314-316) | `remaining_fraction` | native | no |
| registry live count | `.registry_live_count` :199 | `registry_live_count` | native | no |
| rank | `.rank` :200 (rank = sorted-tap index; `-1` in occupied scan reducer.py:1135) | `rank` | native | no |
| conflict flag | `.conflicted` :201 | `conflicted` | native | no |
| nearest level context | `.nearest_level_kind/_distance_ticks` :202-203 | same | native | yes (no available level → both null, reason `no_level_available`) |
| sessions | `.session_engine/.session_doc` :204-205 | same | native | no |
| selected / drop | `.selected`, `.drop_reason` :206-207 | same | native | drop_reason null iff selected |
| tap bar identity | `.tap_cursor` :208 + envelope ts | `tap_cursor` | native | no |
| interleaving vs v2 rows | QL `trace_ordinal` dataset.py:513 | `trace_ordinal` | native (QL) | no |

**Drop reasons proven at source** (reducer.py:1033-1042, 1142):
`retention_not_selected` (:1034), `conflicted` (:1036), `outranked` (:1038),
`direction_disabled` (:1040), `slot_occupied` (:1142), selected → `drop_reason = NULL`.

### 1.2 `parent_candidate` → `ifvg_audit_parent_candidate`

Emitted at reducer.py:1843-1861.

| Prompt requirement | Source symbol | Column | Kind | Nullable |
|---|---|---|---|---|
| candidate gap geometry | `.fvg` records.py:214 | `fvg_*` | native | no |
| parent TF | `.parent_tf_seconds` :213 | `parent_tf_seconds` | native | no |
| distance to HTF | `.distance_to_htf_ticks` :215 (`interval_distance_ticks`) | same | native | no |
| window clocks at intake | `.elapsed_parent_bars_since_tap` :216, `.elapsed_1m_bars_since_tap` :217 | same | native | no |
| causality triple | `.confirmed_after/.fully_formed_after/.causality_satisfied` :218-220 | same | native | no |
| rank | `.rank` :221 | `rank` | native | no |
| selected / drop | `.selected/.drop_reason` :222-223 | same | native | drop_reason null iff selected |
| bar identity | envelope `ts_utc`+`trading_day` (records.py:165-166; = 1m availability, unique per day) | `envelope_ts_utc` | derived | no |

**Drop reasons proven at source** (reducer.py:1819-1841): `causality_failed`
(:1820), `reaction_window_expired` (:1822), `distance_gt_profile` (:1824),
`outranked` (:1841), selected → NULL. **Replacement**: a selected row while a
parent already existed increments `parents_replaced` (:1838) but the DEMOTED
parent has NO record today → **evidence gap → closed by
`ParentWindowEventRecord` (`parent_selected` with `prior_parent_fvg_id`)**.

### 1.3 `opposing` → `ifvg_audit_opposing`

Emitted at reducer.py:1893-1954 via `_opposing_emission` (:1956-1986).

| Prompt requirement | Source symbol | Column | Kind | Nullable |
|---|---|---|---|---|
| opposing gap geometry | `.fvg` records.py:239 | `fvg_*` | native | no |
| distance to parent | `.distance_to_parent_ticks` :240 | same | native | no |
| clock | `.elapsed_1m_bars_since_lock` :241 | same | native | no |
| causality triple | :242-244 | same | native | no |
| selected / drop | :245-246 | same | native | drop_reason null iff selected |

**Drop reasons proven at source**: `causality_failed` (reducer.py:1905),
`distance_gt_profile` (:1920), selected → NULL. Replacement visible via
`opposing_replaced` counter (:1926) + a later selected row; the demoted opposing
gap needs no separate death record (it has no window semantics) — recorded as a
documented derivation, not a gap.

### 1.4 `parent_lock` → `ifvg_audit_parent_lock`

Emitted at reducer.py:1186-1200. Native columns: `parent_fvg_id`,
`penetration_ticks`, `ce_reached`, `elapsed_1m_bars_since_selection`,
`lock_cursor` (records.py:227-234). All required; none nullable. Complete — no gap.

### 1.5 `inversion` → `ifvg_audit_inversion`

Emitted at reducer.py:1244-1264. Native columns: `opposing_fvg_id`,
`close_through_margin_ticks`, `bars_armed_to_inversion`, `opposing_size_ticks`,
`sweep_*` (SweepResult flattened), `semantic`, `inversion_cursor`
(records.py:250-258). Complete — no gap.

### 1.6 `setup_resolution` → `ifvg_audit_setup_resolution`

Emitted at `_terminate_pretrade` reducer.py:930-935, `_resolve_trade`
:1753-1762. Native columns records.py:397-419: resolution, direction,
entry_family (nullable pre-entry), entry/stop/tp ticks (nullable), mfe/mae
(nullable), bars_in_trade (nullable), tap/parent_confirmed/lock/armed/
inversion/entry ts (nullable by stage), htf/parent/opposing fvg ids (nullable
by stage). Terminal reason taxonomy = `_terminate_pretrade` reasons
(invalidated_htf_filled, invalidated_parent_filled,
invalidated_parent_structural, expired_parent_retest, expired_parent_search,
expired_opposing_wait, expired_inversion_wait, expired_entry_wait,
missed_out_of_session, dataset_exhaustion_pre_entry) + resolved_tp/resolved_sl.
Complete for terminal identity; fill-depth/clock evidence at death is NOT here →
covered by `ParentSlotDeathRecord`.

### 1.7 `setup_lifecycle_event`, `entry_candidate`, `quarantine`, `geometry_dossier`, `eligible_decision`, `executed_trade`, `candidate_label`

Already persisted in the accepted v2 tables (contracts.py `RecordTable`); the
audit artifact does NOT duplicate them — FK references only
(setup_id/candidate_id/decision_id/trade_id/lifecycle_event_id).

## 2. Required evidence NOT in any in-band record → new SC audit channel

| Prompt requirement | Today's fate | New record (kind) | Key columns |
|---|---|---|---|
| fill/touch/eviction per gap per bar (`first_touch`, `filled`, `evicted_age`, `evicted_cap`) with depth evidence | events consumed as triggers only; `add()` returns discarded replay.py:553-555; registry events replay.py:507-509 unrecorded | `FvgFillEventRecord` (`fvg_fill_event`) | fvg_*, event_kind, bar evidence (BarEvidence of bar_1m), prior/new reached_ticks (prior readable pre-mutation fvg.py:418-424), prior/new penetration, far_boundary_ticks, fill_depth_ticks, remaining_fraction_after, wick_crossed_far_boundary, body_closed_through_far_boundary, age_seconds/age_trading_days, registry_live_count_after, setup_id (nullable, reason `registry_only`), fvg_role, selected_for_setup, setup_phase_before/after, linked_slot_death_event_id (nullable), linked_setup_resolution_event_id (nullable) |
| `filled_before_activation` drop reason | invisible (gap dies in registry with no tap record) | derived: `fvg_fill_event` rows with `event_kind=filled ∧ fvg_role=registry_only` | — |
| `age_evicted` / `cap_evicted` drop reasons | `FvgFillEvent` kinds exist (fvg.py:324) but never persisted | `fvg_fill_event` with event_kind `evicted_age`/`evicted_cap` | — |
| entry-joint causality counterfactual | `del confirmed, fully` reducer.py:1287 | `EntryCausalityRecord` (`entry_joint_causality`) | candidate_id (exact join — emitted after `make_candidate_id` reducer.py:1396), entry_fvg_id (nullable for retest family, reason `no_entry_gap`), confirmed_after, fully_formed_after, policy, satisfied, trigger_ts_utc (inversion ts), evidence cursors |
| parent slot deaths (S1 provisional + S2-S4 terminal + structural variants) with fill-depth and clock evidence | lifecycle rows only (reducer.py:785-793, :821-829, :921-929) | `ParentSlotDeathRecord` (`parent_slot_death`) | event_id, setup_id, parent_fvg_id (nullable for parentless expiry, reason `no_parent_at_death`), phase, death_reason, death_ts_utc, death bar id/cursor/OHLC, physical_fill, structural_close, far_boundary_ticks, prior/new reached_ticks, fill_depth_ticks, wick/body flags, parent_clocks snapshot, remaining_window_bars_by_tf, open_window_timeframes, parentless_interval_started, event_cursor |
| structural invalidation evidence separate from physical fill | boolean branch only reducer.py:799-836 | `FvgInvalidationEventRecord` (`fvg_invalidation_event`) | invalidation_kind, source_timeframe, source_bar_id/OHLC (parent-TF bar), boundary_ticks, close_through_margin_ticks (`close_through_margin_ticks` fvg.py:252-256), strict_comparison_result, setup_id, parent_fvg_id, phase |
| parent window open/select/clear timeline | implicit in state | `ParentWindowEventRecord` (`parent_window_event`) | event_kind (opened reducer.py:1080-1101 / parent_selected :1862-1864 / parent_cleared :794-795, :830-831), setup_id, parent_fvg_id (nullable for opened, reason `window_open_no_parent`), prior_parent_fvg_id (nullable), parent_clocks snapshot, open_window_timeframes |
| parentless interval structure + reconciliation with `parentless_window_live` (reducer.py:899-911) | scalar counter only | `ParentlessIntervalRecord` (derived per interval, emitted at interval END or at day/dataset boundary snapshot-carry) | setup_id, first/last_counted_bar_id, first/last_counted_cursor, bars_count, start/end_ts_utc, end_reason, open TFs at start/end, successor_parent_fvg_id (nullable), eventual_lock, eventual_terminal_reason |
| day funnel long table | `CaptureDayResult.funnel` (QL) | QL-only `ifvg_audit_day_funnel` | source_date, counter, value; no new `_funnel` keys |

All new records additionally carry the (a2) ordering contract fields:
`source_step_ordinal`, `source_bar_id`, `source_bar_cursor`, `reducer_substep`,
`reducer_substep_ordinal`, `core_trace_ordinal_before`,
`core_trace_ordinal_after`, `audit_seq` — NOT NULL, no exceptions.

## 3. Complete drop-reason enumeration (contract-fixed)

`retention_not_selected, conflicted, outranked, direction_disabled,
slot_occupied, filled_before_activation, age_evicted, cap_evicted,
causality_failed, reaction_window_expired, distance_gt_profile, selected,
replaced` — each mapped above to a native column value, a derived predicate, or
a new-record event kind. The audit lane's coverage report must show a nonzero
or provably-zero count for each (provably-zero: `conflicted` under cap=1 [D-1],
structural terminal deaths [current artifact], `replaced` if no replacement
occurred in range).

## 4. Verification obligations carried by this contract

1. Column-by-column presence asserted by Arrow schemas in
   `audit_contracts.py` — a missing required column fails the build.
2. Funnel ⇔ event-row reconciliation exact for every mapped counter
   (`htf_taps`, `taps_conflicted`, `taps_slot_occupied`, `parent_candidates`,
   `parents_replaced`, `parents_locked`, `opposing_candidates`,
   `opposing_replaced`, `opposing_armed`, `inversions`, `inverted_*`,
   `candidate_died_filled`, `candidate_died_structural`, `parentless_window_live`
   (interval checksum), `entry_candidates_*`, `candidate_blocked_*`,
   `eligible_decisions`, `executions_opened`, `resolved_*`, terminal reasons).
3. The cross-channel total ordering is a hard contract (same-minute ambiguity =
   test failure).
4. Nullable columns are enumerated here exhaustively; any other null in a
   required column fails validation with the offending PK.
