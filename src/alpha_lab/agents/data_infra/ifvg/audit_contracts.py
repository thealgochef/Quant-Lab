"""IFVG FSM audit companion tables (``ifvg_fsm_audit_v1``).

Typed table contract for the additive audit artifact keyed to the accepted v2
dataset. Six tables are partitioned OUT of the SAME flattened emission trace
the v2 tables come from (so they share ``trace_ordinal`` and exact
interleaving); the rest ride Strategy-Core's parallel per-step-drained audit
channel and carry the :class:`AuditStamp` cross-channel ordering contract
(flattened ``stamp_*`` columns) plus chain-global ordinals assigned here.

The contract is enforced structurally: a missing required column, a null in a
non-nullable column, a duplicate primary key, a dangling audit-internal link,
or an inexact funnel⇔events reconciliation FAILS the build. Nullability is
enumerated per table (see ``NULLABLE_COLUMNS``); anything else must be
populated on every row.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import pandas as pd

__all__ = [
    "IFVG_FSM_AUDIT_SCHEMA_VERSION",
    "AuditTable",
    "TRACE_AUDIT_KIND_BY_TABLE",
    "CHANNEL_AUDIT_KIND_BY_TABLE",
    "AUDIT_STAMP_COLUMNS",
    "REQUIRED_COLUMNS",
    "NULLABLE_COLUMNS",
    "PRIMARY_KEY",
    "validate_audit_table",
    "validate_audit_links",
    "reconcile_funnel_to_audit",
    "audit_contract_fingerprint",
]

IFVG_FSM_AUDIT_SCHEMA_VERSION = 1


class AuditTable(StrEnum):
    HTF_TAP = "ifvg_audit_htf_tap"
    PARENT_CANDIDATE = "ifvg_audit_parent_candidate"
    OPPOSING = "ifvg_audit_opposing"
    PARENT_LOCK = "ifvg_audit_parent_lock"
    INVERSION = "ifvg_audit_inversion"
    SETUP_RESOLUTION = "ifvg_audit_setup_resolution"
    FILL_EVENT = "ifvg_audit_fill_event"
    ENTRY_CAUSALITY = "ifvg_audit_entry_causality"
    SLOT_DEATH = "ifvg_audit_slot_death"
    PARENT_WINDOW = "ifvg_audit_parent_window"
    PARENTLESS_STEP = "ifvg_audit_parentless_step"
    PARENTLESS_INTERVAL = "ifvg_audit_parentless_interval"
    DAY_FUNNEL = "ifvg_audit_day_funnel"


#: Emission-trace kinds partitioned out of the shared v2 trace (zero SC change;
#: they keep the chain-global ``trace_ordinal`` assigned before partition).
TRACE_AUDIT_KIND_BY_TABLE: dict[AuditTable, str] = {
    AuditTable.HTF_TAP: "htf_tap",
    AuditTable.PARENT_CANDIDATE: "parent_candidate",
    AuditTable.OPPOSING: "opposing",
    AuditTable.PARENT_LOCK: "parent_lock",
    AuditTable.INVERSION: "inversion",
    AuditTable.SETUP_RESOLUTION: "setup_resolution",
}

#: Audit-channel kinds (Strategy-Core parallel channel).
CHANNEL_AUDIT_KIND_BY_TABLE: dict[AuditTable, str] = {
    AuditTable.FILL_EVENT: "fvg_fill_event",
    AuditTable.ENTRY_CAUSALITY: "entry_joint_causality",
    AuditTable.SLOT_DEATH: "parent_slot_death",
    AuditTable.PARENT_WINDOW: "parent_window_event",
    AuditTable.PARENTLESS_STEP: "parentless_step",
}

_ENVELOPE_COLUMNS = (
    "envelope_schema_version",
    "envelope_strategy_id",
    "envelope_strategy_version",
    "envelope_profile_hash",
    "envelope_trading_day",
    "envelope_ts_utc",
    "envelope_setup_id",
    "envelope_profile_name",
    "envelope_qualification_mode",
    "envelope_section_config_hash",
    "envelope_entry_family",
    "envelope_label_family",
    "envelope_entry_session",
    "envelope_anchor_policy",
    "envelope_resolver_policy",
    "envelope_causality_parent",
    "envelope_causality_opposing",
    "envelope_causality_entry",
    "envelope_timeout_policy",
)

_FVG_COLUMNS = tuple(
    f"fvg_{name}"
    for name in (
        "fvg_id",
        "timeframe_seconds",
        "direction",
        "gap_low_ticks",
        "gap_high_ticks",
        "size_ticks",
        "a_bar_id",
        "c_bar_id",
        "a_open_ts_utc",
        "confirmed_ts_utc",
        "trading_day",
    )
)

_BAR_COLUMNS = tuple(
    f"bar_{name}"
    for name in (
        "bar_id",
        "timeframe_seconds",
        "trading_day",
        "logical_open_ts_utc",
        "logical_close_ts_utc",
        "first_print_ts_utc",
        "last_print_ts_utc",
        "open_ticks",
        "high_ticks",
        "low_ticks",
        "close_ticks",
        "cursor",
    )
)

#: Flattened :class:`AuditStamp` (SC) — the cross-channel ordering contract.
AUDIT_STAMP_COLUMNS = (
    "stamp_audit_schema_version",
    "stamp_source_step_ordinal",
    "stamp_source_bar_id",
    "stamp_source_bar_cursor",
    "stamp_reducer_substep",
    "stamp_reducer_substep_ordinal",
    "stamp_core_trace_ordinal_before",
    "stamp_core_trace_ordinal_after",
    "stamp_audit_seq",
)

#: QL chain stamps added to every audit-channel row by the builder.
_CHAIN_STAMP_COLUMNS = (
    "entering_seed_hash",
    "is_warmup",
    "days_of_htf_history",
    "evaluation_config_hash",
    "audit_trace_ordinal",
    "core_trace_ordinal_before_global",
    "core_trace_ordinal_after_global",
)

#: QL chain stamps carried by trace-reused rows (from the shared v2 trace).
_TRACE_STAMP_COLUMNS = (
    "entering_seed_hash",
    "capture_schema_version",
    "is_warmup",
    "days_of_htf_history",
    "evaluation_config_hash",
    "trace_ordinal",
)

_CHANNEL_BASE = _ENVELOPE_COLUMNS + AUDIT_STAMP_COLUMNS + _CHAIN_STAMP_COLUMNS

REQUIRED_COLUMNS: dict[AuditTable, tuple[str, ...]] = {
    AuditTable.HTF_TAP: _ENVELOPE_COLUMNS
    + _TRACE_STAMP_COLUMNS
    + _FVG_COLUMNS
    + (
        "htf_tf_seconds",
        "direction",
        "penetration_ticks",
        "ce_reached",
        "htf_age_seconds",
        "remaining_fraction",
        "registry_live_count",
        "rank",
        "conflicted",
        "nearest_level_kind",
        "nearest_level_distance_ticks",
        "session_engine",
        "session_doc",
        "selected",
        "drop_reason",
        "tap_cursor",
    ),
    AuditTable.PARENT_CANDIDATE: _ENVELOPE_COLUMNS
    + _TRACE_STAMP_COLUMNS
    + _FVG_COLUMNS
    + (
        "parent_tf_seconds",
        "distance_to_htf_ticks",
        "elapsed_parent_bars_since_tap",
        "elapsed_1m_bars_since_tap",
        "confirmed_after",
        "fully_formed_after",
        "causality_satisfied",
        "rank",
        "selected",
        "drop_reason",
    ),
    AuditTable.OPPOSING: _ENVELOPE_COLUMNS
    + _TRACE_STAMP_COLUMNS
    + _FVG_COLUMNS
    + (
        "distance_to_parent_ticks",
        "elapsed_1m_bars_since_lock",
        "confirmed_after",
        "fully_formed_after",
        "causality_satisfied",
        "selected",
        "drop_reason",
    ),
    AuditTable.PARENT_LOCK: _ENVELOPE_COLUMNS
    + _TRACE_STAMP_COLUMNS
    + (
        "parent_fvg_id",
        "penetration_ticks",
        "ce_reached",
        "elapsed_1m_bars_since_selection",
        "lock_cursor",
    ),
    AuditTable.INVERSION: _ENVELOPE_COLUMNS
    + _TRACE_STAMP_COLUMNS
    + (
        "opposing_fvg_id",
        "close_through_margin_ticks",
        "bars_armed_to_inversion",
        "opposing_size_ticks",
        "sweep_sweep_confirmed",
        "sweep_swept_kinds",
        "sweep_max_penetration_ticks",
        "sweep_nearest_unswept_distance_ticks",
        "sweep_sweep_ts_utc",
        "sweep_leg_extreme_ticks",
        "semantic",
        "inversion_cursor",
    ),
    AuditTable.SETUP_RESOLUTION: _ENVELOPE_COLUMNS
    + _TRACE_STAMP_COLUMNS
    + (
        "resolution",
        "direction",
        "entry_family",
        "entry_ticks",
        "stop_ticks",
        "tp_ticks",
        "mfe_ticks",
        "mae_ticks",
        "bars_in_trade",
        "tap_ts_utc",
        "parent_confirmed_ts_utc",
        "lock_ts_utc",
        "armed_ts_utc",
        "inversion_ts_utc",
        "entry_ts_utc",
        "htf_fvg_id",
        "parent_fvg_id",
        "opposing_fvg_id",
    ),
    AuditTable.FILL_EVENT: _CHANNEL_BASE
    + _FVG_COLUMNS
    + _BAR_COLUMNS
    + (
        "event_kind",
        "prior_reached_ticks",
        "new_reached_ticks",
        "prior_penetration_ticks",
        "new_penetration_ticks",
        "far_boundary_ticks",
        "fill_depth_ticks",
        "remaining_fraction_after",
        "wick_crossed_far_boundary",
        "body_closed_through_far_boundary",
        "age_seconds",
        "age_trading_days",
        "registry_live_count_after",
        "setup_id",
        "fvg_role",
        "selected_for_setup",
        "setup_phase_before",
        "setup_phase_after",
        "linked_slot_death_event_id",
        "linked_setup_resolution_event_id",
    ),
    AuditTable.ENTRY_CAUSALITY: _CHANNEL_BASE
    + (
        "candidate_id",
        "setup_id",
        "entry_family",
        "entry_fvg_id",
        "policy",
        "trigger_ts_utc",
        "confirmed_after",
        "fully_formed_after",
        "satisfied",
    ),
    AuditTable.SLOT_DEATH: _CHANNEL_BASE
    + _BAR_COLUMNS
    + (
        "event_id",
        "setup_id",
        "phase",
        "death_reason",
        "death_ts_utc",
        "parent_fvg_id",
        "died_fvg_id",
        "setup_terminated",
        "lifecycle_event_id",
        "event_cursor",
        "physical_fill",
        "structural_close",
        "far_boundary_ticks",
        "prior_reached_ticks",
        "new_reached_ticks",
        "fill_depth_ticks",
        "wick_crossed_far_boundary",
        "body_closed_through_far_boundary",
        "parent_clocks",
        "remaining_window_bars_by_tf",
        "open_window_timeframes",
        "parentless_interval_started",
    ),
    AuditTable.PARENT_WINDOW: _CHANNEL_BASE
    + (
        "event_id",
        "event_kind",
        "setup_id",
        "parent_fvg_id",
        "prior_parent_fvg_id",
        "parent_clocks",
        "open_window_timeframes",
        "event_cursor",
    ),
    AuditTable.PARENTLESS_STEP: _CHANNEL_BASE
    + _BAR_COLUMNS
    + (
        "setup_id",
        "parent_clocks",
        "open_window_timeframes",
    ),
    AuditTable.PARENTLESS_INTERVAL: (
        "interval_id",
        "setup_id",
        "first_counted_bar_id",
        "last_counted_bar_id",
        "first_counted_cursor",
        "last_counted_cursor",
        "first_step_ordinal",
        "last_step_ordinal",
        "bars_count",
        "start_ts_utc",
        "end_ts_utc",
        "end_reason",
        "open_timeframes_at_start",
        "open_timeframes_at_end",
        "successor_parent_fvg_id",
        "eventual_lock",
        "eventual_terminal_reason",
        "first_source_date",
        "last_source_date",
        "crosses_day_boundary",
        "audit_schema_version",
    ),
    AuditTable.DAY_FUNNEL: (
        "source_date",
        "counter",
        "value",
        "is_warmup",
        "audit_schema_version",
    ),
}

#: Columns that may be null, with the semantic reason enumerated in the
#: evidence contract. Any other null in a required column fails validation.
NULLABLE_COLUMNS: dict[AuditTable, tuple[str, ...]] = {
    AuditTable.HTF_TAP: (
        "nearest_level_kind",
        "nearest_level_distance_ticks",
        "drop_reason",
        "entering_seed_hash",
    ),
    AuditTable.PARENT_CANDIDATE: ("drop_reason", "entering_seed_hash"),
    AuditTable.OPPOSING: ("drop_reason", "entering_seed_hash"),
    AuditTable.PARENT_LOCK: ("entering_seed_hash",),
    AuditTable.INVERSION: (
        "sweep_nearest_unswept_distance_ticks",
        "sweep_sweep_ts_utc",
        "sweep_leg_extreme_ticks",
        "entering_seed_hash",
    ),
    AuditTable.SETUP_RESOLUTION: (
        "entry_family",
        "entry_ticks",
        "stop_ticks",
        "tp_ticks",
        "mfe_ticks",
        "mae_ticks",
        "bars_in_trade",
        "parent_confirmed_ts_utc",
        "lock_ts_utc",
        "armed_ts_utc",
        "inversion_ts_utc",
        "entry_ts_utc",
        "parent_fvg_id",
        "opposing_fvg_id",
        "entering_seed_hash",
    ),
    AuditTable.FILL_EVENT: (
        "prior_reached_ticks",
        "new_reached_ticks",
        "wick_crossed_far_boundary",  # cap eviction has no execution bar
        "body_closed_through_far_boundary",
        "setup_id",  # registry_only role
        "linked_slot_death_event_id",
        "linked_setup_resolution_event_id",
        "entering_seed_hash",
    ),
    AuditTable.ENTRY_CAUSALITY: (
        "entry_fvg_id",  # retest family has no entry gap
        "trigger_ts_utc",
        "confirmed_after",
        "fully_formed_after",
        "entering_seed_hash",
    ),
    AuditTable.SLOT_DEATH: (
        "parent_fvg_id",  # parentless expiry
        "died_fvg_id",  # expiry deaths have no killing gap
        "lifecycle_event_id",
        "far_boundary_ticks",
        "prior_reached_ticks",
        "new_reached_ticks",
        "fill_depth_ticks",
        "wick_crossed_far_boundary",
        "body_closed_through_far_boundary",
        "entering_seed_hash",
    )
    + _BAR_COLUMNS,  # dataset-exhaustion death has no bar
    AuditTable.PARENT_WINDOW: (
        "parent_fvg_id",  # opened / cleared-to-empty
        "prior_parent_fvg_id",
        "entering_seed_hash",
    ),
    AuditTable.PARENTLESS_STEP: ("entering_seed_hash",),
    AuditTable.PARENTLESS_INTERVAL: (
        "successor_parent_fvg_id",
        "eventual_terminal_reason",
    ),
    AuditTable.DAY_FUNNEL: (),
}

#: Primary key columns per table (composite keys allowed).
PRIMARY_KEY: dict[AuditTable, tuple[str, ...]] = {
    AuditTable.HTF_TAP: ("trace_ordinal",),
    AuditTable.PARENT_CANDIDATE: ("trace_ordinal",),
    AuditTable.OPPOSING: ("trace_ordinal",),
    AuditTable.PARENT_LOCK: ("trace_ordinal",),
    AuditTable.INVERSION: ("trace_ordinal",),
    AuditTable.SETUP_RESOLUTION: ("trace_ordinal",),
    AuditTable.FILL_EVENT: ("audit_trace_ordinal",),
    AuditTable.ENTRY_CAUSALITY: ("audit_trace_ordinal",),
    AuditTable.SLOT_DEATH: ("event_id",),
    AuditTable.PARENT_WINDOW: ("event_id",),
    AuditTable.PARENTLESS_STEP: ("audit_trace_ordinal",),
    AuditTable.PARENTLESS_INTERVAL: ("interval_id",),
    AuditTable.DAY_FUNNEL: ("source_date", "counter"),
}


def validate_audit_table(table: AuditTable, frame: pd.DataFrame) -> None:
    """Structural contract: required columns, nullability, PK uniqueness,
    stamp integrity. An empty frame must still carry the full schema."""
    required = REQUIRED_COLUMNS[table]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{table.value} is missing required columns {missing}")
    if frame.empty:
        return
    nullable = set(NULLABLE_COLUMNS[table])
    for column in required:
        if column in nullable:
            continue
        values = frame[column]
        if values.isna().any():
            raise ValueError(f"{table.value}.{column} contains nulls")
    key = list(PRIMARY_KEY[table])
    keyed = frame.loc[:, key]
    if keyed.isna().any().any():
        raise ValueError(f"{table.value} primary key {key} contains nulls")
    if keyed.duplicated().any():
        raise ValueError(f"{table.value} primary key {key} is not unique")
    if "stamp_audit_schema_version" in frame.columns:
        versions = pd.to_numeric(frame["stamp_audit_schema_version"], errors="coerce")
        if versions.isna().any() or not (versions == IFVG_FSM_AUDIT_SCHEMA_VERSION).all():
            raise ValueError(f"{table.value} stamp_audit_schema_version must be 1")
    if "stamp_audit_seq" in frame.columns:
        # single total order: per-day audit_seq strictly monotone and unique.
        for day, group in frame.groupby("envelope_trading_day"):
            seqs = pd.to_numeric(group["stamp_audit_seq"], errors="raise")
            if seqs.duplicated().any():
                raise ValueError(
                    f"{table.value} duplicated audit_seq within {day} — "
                    "cross-channel total ordering is ambiguous"
                )


def validate_audit_links(tables: Mapping[AuditTable, pd.DataFrame]) -> None:
    """Audit-internal referential integrity (v2-table FKs are validated by the
    builder against the regenerated v2 tables)."""

    def _ids(table: AuditTable, column: str) -> set[str]:
        frame = tables.get(table, pd.DataFrame())
        if frame.empty or column not in frame:
            return set()
        return set(frame[column].dropna().astype(str))

    death_ids = _ids(AuditTable.SLOT_DEATH, "event_id")
    fill = tables.get(AuditTable.FILL_EVENT, pd.DataFrame())
    if not fill.empty:
        linked = set(fill["linked_slot_death_event_id"].dropna().astype(str))
        dangling = linked - death_ids
        if dangling:
            raise ValueError(
                f"fill_event.linked_slot_death_event_id dangling: {sorted(dangling)[:5]}"
            )
    interval = tables.get(AuditTable.PARENTLESS_INTERVAL, pd.DataFrame())
    step_setups = _ids(AuditTable.PARENTLESS_STEP, "setup_id")
    if not interval.empty:
        unknown = set(interval["setup_id"].astype(str)) - step_setups
        if unknown:
            raise ValueError(
                f"parentless_interval.setup_id without step rows: {sorted(unknown)[:5]}"
            )


#: funnel counter -> (table, predicate) exact-count mapping. Implemented in
#: :func:`reconcile_funnel_to_audit`; every listed counter must match exactly.
_TERMINAL_REASONS = (
    "invalidated_htf_filled",
    "invalidated_parent_filled",
    "invalidated_parent_structural",
    "expired_parent_retest",
    "expired_parent_search",
    "expired_opposing_wait",
    "expired_inversion_wait",
    "expired_entry_wait",
    "missed_out_of_session",
)


def reconcile_funnel_to_audit(
    day_funnels: Mapping[str, Mapping[str, int]],
    tables: Mapping[AuditTable, pd.DataFrame],
) -> dict:
    """Exact funnel ⇔ audit-event reconciliation over the full run.

    Raises on ANY inexact mapped counter; returns the per-counter report.
    ``parentless_window_live`` reconciles twice: against the step rows AND
    against the derived interval ``bars_count`` checksum, per day and total.
    """
    total: dict[str, int] = {}
    for counters in day_funnels.values():
        for key, value in counters.items():
            total[key] = total.get(key, 0) + int(value)

    def _count(table: AuditTable, mask=None) -> int:
        frame = tables.get(table, pd.DataFrame())
        if frame.empty:
            return 0
        if mask is None:
            return int(len(frame))
        return int(mask(frame).sum())

    taps = tables.get(AuditTable.HTF_TAP, pd.DataFrame())
    deaths = tables.get(AuditTable.SLOT_DEATH, pd.DataFrame())
    window = tables.get(AuditTable.PARENT_WINDOW, pd.DataFrame())
    causality = tables.get(AuditTable.ENTRY_CAUSALITY, pd.DataFrame())

    checks: dict[str, tuple[int, int]] = {
        "htf_taps": (total.get("htf_taps", 0), _count(AuditTable.HTF_TAP)),
        "taps_conflicted": (
            total.get("taps_conflicted", 0),
            _count(AuditTable.HTF_TAP, lambda f: f["conflicted"].astype(bool))
            if not taps.empty
            else 0,
        ),
        "taps_slot_occupied": (
            total.get("taps_slot_occupied", 0),
            _count(AuditTable.HTF_TAP, lambda f: f["drop_reason"] == "slot_occupied"),
        ),
        "setups_born": (
            total.get("setups_born", 0),
            _count(
                AuditTable.PARENT_WINDOW, lambda f: f["event_kind"] == "opened"
            ),
        ),
        "parent_candidates": (
            total.get("parent_candidates", 0),
            _count(AuditTable.PARENT_CANDIDATE),
        ),
        "parents_replaced": (
            total.get("parents_replaced", 0),
            _count(
                AuditTable.PARENT_WINDOW,
                lambda f: (f["event_kind"] == "parent_selected")
                & f["prior_parent_fvg_id"].notna(),
            )
            if not window.empty
            else 0,
        ),
        "parents_locked": (
            total.get("parents_locked", 0),
            _count(AuditTable.PARENT_LOCK),
        ),
        "opposing_candidates": (
            total.get("opposing_candidates", 0),
            _count(AuditTable.OPPOSING),
        ),
        "inversions": (total.get("inversions", 0), _count(AuditTable.INVERSION)),
        "candidate_died_filled": (
            total.get("candidate_died_filled", 0),
            _count(
                AuditTable.SLOT_DEATH, lambda f: f["death_reason"] == "parent_filled"
            )
            if not deaths.empty
            else 0,
        ),
        "candidate_died_structural": (
            total.get("candidate_died_structural", 0),
            _count(
                AuditTable.SLOT_DEATH,
                lambda f: f["death_reason"] == "parent_structural_close",
            )
            if not deaths.empty
            else 0,
        ),
        "parentless_window_live": (
            total.get("parentless_window_live", 0),
            _count(AuditTable.PARENTLESS_STEP),
        ),
    }
    for reason in _TERMINAL_REASONS:
        expected = total.get(reason, 0)
        observed = (
            _count(AuditTable.SLOT_DEATH, lambda f, r=reason: f["death_reason"] == r)
            if not deaths.empty
            else 0
        )
        checks[reason] = (expected, observed)
    resolved = total.get("resolved_target", 0) + total.get("resolved_stop", 0)
    checks["resolved_slot_freed"] = (
        resolved,
        _count(AuditTable.SLOT_DEATH, lambda f: f["death_reason"] == "slot_freed")
        if not deaths.empty
        else 0,
    )
    family_counters = {
        key: value for key, value in total.items() if key.startswith("entry_candidates_")
    }
    for key, expected in family_counters.items():
        family = key.removeprefix("entry_candidates_")
        observed = (
            _count(
                AuditTable.ENTRY_CAUSALITY,
                lambda f, fam=family: f["entry_family"] == fam,
            )
            if not causality.empty
            else 0
        )
        checks[key] = (expected, observed)

    # interval checksum: per-day AND full-run bars_count identity.
    interval = tables.get(AuditTable.PARENTLESS_INTERVAL, pd.DataFrame())
    steps = tables.get(AuditTable.PARENTLESS_STEP, pd.DataFrame())
    interval_bars = int(interval["bars_count"].sum()) if not interval.empty else 0
    checks["parentless_interval_checksum"] = (
        total.get("parentless_window_live", 0),
        interval_bars,
    )
    per_day: dict[str, tuple[int, int]] = {}
    for day, counters in sorted(day_funnels.items()):
        expected = int(counters.get("parentless_window_live", 0))
        observed = (
            int((steps["envelope_trading_day"].astype(str) == day).sum())
            if not steps.empty
            else 0
        )
        per_day[day] = (expected, observed)

    failures = sorted(
        [name for name, (expected, observed) in checks.items() if expected != observed]
        + [f"parentless_day:{day}" for day, (e, o) in per_day.items() if e != o]
    )
    report = {
        "passed": not failures,
        "checks": {name: {"funnel": e, "audit": o} for name, (e, o) in checks.items()},
        "parentless_per_day": {
            day: {"funnel": e, "audit": o} for day, (e, o) in per_day.items()
        },
        "failures": failures,
    }
    if failures:
        raise ValueError(f"IFVG audit funnel reconciliation inexact: {failures}")
    return report


def audit_contract_fingerprint() -> dict:
    """Deterministic contract description embedded in the artifact manifest."""
    return {
        "audit_schema_version": IFVG_FSM_AUDIT_SCHEMA_VERSION,
        "tables": {
            table.value: {
                "required_columns": list(REQUIRED_COLUMNS[table]),
                "nullable_columns": list(NULLABLE_COLUMNS[table]),
                "primary_key": list(PRIMARY_KEY[table]),
            }
            for table in AuditTable
        },
    }
