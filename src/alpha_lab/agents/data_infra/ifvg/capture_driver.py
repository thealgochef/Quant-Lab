"""Phase C — one day's capture: SC ``run_day`` over cached artifacts + row flattening.

QL computes NOTHING here: bars/levels come from the Phase-A artifact, the FSM
is SC's ``run_day``, and this module only groups bars by timeframe, hands the
level timeline in as ``levels_for``, and flattens the typed SC emissions into
one wide DataFrame (``kind`` column + unioned per-kind fields; nested
dataclasses flattened with ``<field>_`` prefixes; enums to values; tuples to
JSON strings).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

import pandas as pd
from strategy_core.candles.exchange_calendar import (
    DEFAULT_CONTEXT_SOURCE_COVERAGE,
    ContextSourceCoverage,
)
from strategy_core.strategies.ifvg_smc.context_config import ContextFeatureConfig
from strategy_core.strategies.ifvg_smc.context_features import (
    IfvgContextObserverSeed,
    context_observer_seed_hash,
)
from strategy_core.strategies.ifvg_smc.context_records import (
    CaptureKind,
    IfvgContextEvent,
)
from strategy_core.strategies.ifvg_smc.replay import (
    ContextPerformanceTrace,
    IfvgContextDayResult,
    run_day,
)
from strategy_core.strategies.ifvg_smc.state import IfvgDaySeed, seed_hash
from strategy_core.structures.context import ContextRecord, canonical_sha256

from .config import IfvgCaptureConfig
from .context_contracts import ContextRecordTable, stamp_context_table
from .contracts import IFVG_CAPTURE_SCHEMA_VERSION
from .day_artifacts import DayArtifacts, levels_for_from_frame

__all__ = [
    "CaptureDayResult",
    "ContextCaptureDayResult",
    "capture_single_date",
    "capture_single_date_with_context",
    "flatten_audit_emissions",
    "flatten_emissions",
    "normalize_context_days",
]


@dataclass(frozen=True)
class CaptureDayResult:
    date_str: str
    rows: pd.DataFrame
    funnel: dict[str, int]
    entering_seed_hash: str | None
    end_seed: IfvgDaySeed
    #: FSM audit channel rows (``audit_capture_mode="fsm_audit_v1"`` only);
    #: None when the channel is disabled — the v2 lane never sees them.
    audit_rows: pd.DataFrame | None = None


@dataclass(frozen=True)
class ContextCaptureDayResult:
    """One replay result with unchanged v2 rows and separate typed context."""

    date_str: str
    core_rows: pd.DataFrame
    funnel: dict[str, int]
    entering_seed_hash: str | None
    entering_context_seed_hash: str | None
    end_seed: IfvgDaySeed
    end_context_seed: IfvgContextObserverSeed
    context_events: tuple[IfvgContextEvent, ...]
    confirmed_swings: tuple
    pool_lifecycle_events: tuple
    sweep_link_events: tuple
    performance_trace: ContextPerformanceTrace


def _flat(value: object, prefix: str, out: dict) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        for key, sub in asdict(value).items():  # asdict recurses; re-flatten dicts
            _flat(sub, f"{prefix}{key}_", out)
        return
    if isinstance(value, dict):
        for key, sub in value.items():
            _flat(sub, f"{prefix}{key}_", out)
        return
    name = prefix[:-1]  # trim trailing underscore
    if isinstance(value, Enum):
        out[name] = value.value
    elif isinstance(value, (tuple, list)):
        out[name] = json.dumps([v.value if isinstance(v, Enum) else v for v in value])
    else:
        out[name] = value


def flatten_emissions(emissions, *, entering_seed_hash: str | None) -> pd.DataFrame:
    rows = []
    for emission in emissions:
        if emission.kind == "funnel":
            continue  # tier-1 counters ride CaptureDayResult.funnel, not rows
        row: dict = {
            "kind": emission.kind,
            "entering_seed_hash": entering_seed_hash,
            "capture_schema_version": IFVG_CAPTURE_SCHEMA_VERSION,
        }
        _flat(emission.record, "", row)
        rows.append(row)
    return pd.DataFrame(rows)


def flatten_audit_emissions(
    audit_emissions, *, entering_seed_hash: str | None
) -> pd.DataFrame:
    """Flatten FSM audit-channel emissions on a SEPARATE frame — same ``_flat``
    treatment as the core trace, but the audit frame never joins the v2 union
    (no ``capture_schema_version`` claim; the stamp carries its own version)."""
    rows = []
    for emission in audit_emissions:
        row: dict = {
            "kind": emission.kind,
            "entering_seed_hash": entering_seed_hash,
        }
        _flat(emission.record, "", row)
        rows.append(row)
    return pd.DataFrame(rows)


def capture_single_date(
    date_str: str,
    cfg: IfvgCaptureConfig,
    *,
    artifacts: DayArtifacts,
    seed: IfvgDaySeed | None,
    dataset_exhausted: bool = False,
    audit_capture_mode: str = "disabled",
) -> CaptureDayResult:
    bars_by_tf: dict[int, list] = {}
    for bar in artifacts.bars:
        bars_by_tf.setdefault(bar.timeframe_ticks, []).append(bar)
    entering = seed_hash(seed) if seed is not None else None
    result = run_day(
        bars_by_tf,
        section=cfg.section,
        seed=seed,
        trading_day=date.fromisoformat(date_str),
        tick_size=cfg.tick_size,
        levels_for=levels_for_from_frame(artifacts.level_timeline),
        dataset_exhausted=dataset_exhausted,
        audit_capture_mode=audit_capture_mode,
    )
    rows = flatten_emissions(result.emissions, entering_seed_hash=entering)
    audit_rows = (
        flatten_audit_emissions(
            result.audit_emissions, entering_seed_hash=entering
        )
        if audit_capture_mode != "disabled"
        else None
    )
    return CaptureDayResult(
        date_str=date_str,
        rows=rows,
        funnel=dict(result.funnel.counters),
        entering_seed_hash=entering,
        end_seed=result.end_seed,
        audit_rows=audit_rows,
    )


def capture_single_date_with_context(
    date_str: str,
    cfg: IfvgCaptureConfig,
    *,
    artifacts: DayArtifacts,
    seed: IfvgDaySeed | None,
    context_config: ContextFeatureConfig,
    context_seed: IfvgContextObserverSeed | None,
    strategy_core_commit: str,
    strategy_core_source_tree_hash: str,
    context_source_coverage: ContextSourceCoverage = DEFAULT_CONTEXT_SOURCE_COVERAGE,
    dataset_exhausted: bool = False,
) -> ContextCaptureDayResult:
    """Run Strategy-Core once, retaining v2 emissions only for parity."""

    bars_by_tf: dict[int, list] = {}
    for bar in artifacts.bars:
        bars_by_tf.setdefault(bar.timeframe_ticks, []).append(bar)
    entering = seed_hash(seed) if seed is not None else None
    entering_context = (
        context_observer_seed_hash(context_seed) if context_seed is not None else None
    )
    result = run_day(
        bars_by_tf,
        section=cfg.section,
        seed=seed,
        trading_day=date.fromisoformat(date_str),
        tick_size=cfg.tick_size,
        levels_for=levels_for_from_frame(artifacts.level_timeline),
        dataset_exhausted=dataset_exhausted,
        context_config=context_config,
        context_seed=context_seed,
        context_source_coverage=context_source_coverage,
        context_symbol=cfg.symbol,
        strategy_core_commit=strategy_core_commit,
        strategy_core_source_tree_hash=strategy_core_source_tree_hash,
    )
    if not isinstance(result, IfvgContextDayResult):
        raise TypeError("Strategy-Core did not return an IFVG context result")
    return ContextCaptureDayResult(
        date_str=date_str,
        core_rows=flatten_emissions(result.emissions, entering_seed_hash=entering),
        funnel=dict(result.funnel.counters),
        entering_seed_hash=entering,
        entering_context_seed_hash=entering_context,
        end_seed=result.end_seed,
        end_context_seed=result.end_context_seed,
        context_events=result.context_events,
        confirmed_swings=result.confirmed_swings,
        pool_lifecycle_events=result.pool_lifecycle_events,
        sweep_link_events=result.sweep_link_events,
        performance_trace=result.performance_trace,
    )


def _flat_context(value: object, prefix: str, out: dict[str, Any]) -> None:
    """Flatten typed Strategy-Core context without deriving any formula."""

    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _flat_context(getattr(value, field.name), f"{prefix}{field.name}_", out)
        return
    if isinstance(value, Mapping):
        for key, sub in sorted(value.items(), key=lambda item: str(item[0])):
            _flat_context(sub, f"{prefix}{key}_", out)
        return
    name = prefix[:-1]
    if isinstance(value, Enum):
        out[name] = value.value
    elif isinstance(value, UUID):
        out[name] = str(value)
    elif isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("context record contains a naive timestamp")
        out[name] = value.astimezone(UTC)
    elif isinstance(value, date):
        out[name] = value.isoformat()
    elif isinstance(value, (tuple, list)):
        out[name] = [
            item.value
            if isinstance(item, Enum)
            else (str(item) if isinstance(item, UUID) else item)
            for item in value
        ]
    else:
        out[name] = value


def _record_row(record: object) -> dict[str, Any]:
    row: dict[str, Any] = {}
    _flat_context(record, "", row)
    return row


def _record_id(record: ContextRecord) -> str:
    for field_name in (
        "context_state_id",
        "context_capture_id",
        "structure_state_id",
        "structure_delta_id",
        "displacement_window_id",
        "lifecycle_event_id",
        "sweep_link_id",
        "swing_id",
        "mtf_snapshot_id",
    ):
        value = getattr(record, field_name, None)
        if value is not None:
            return str(value)
    raise ValueError(f"context record {type(record).__name__} has no stable object ID")


def _metadata(day: ContextCaptureDayResult, chain_index: int, warmup_days: int) -> dict:
    return {
        "source_date": day.date_str,
        "is_warmup": chain_index < warmup_days,
        "days_of_htf_history": chain_index,
        "entering_context_seed_hash": day.entering_context_seed_hash,
    }


def _insert_unique(
    rows: dict[str, dict[str, Any]],
    key: str,
    row: dict[str, Any],
    *,
    label: str,
) -> None:
    existing = rows.get(key)
    if existing is not None:
        # Immutable context objects may be referenced again by a later capture
        # (including across a source-day boundary). The day/seed columns annotate
        # the first observation in this replay; they are not part of object
        # identity. Preserve that first annotation, but still fail closed if any
        # actual record field changes under a stable object ID.
        normalization_metadata = {
            "source_date",
            "is_warmup",
            "days_of_htf_history",
            "entering_context_seed_hash",
        }
        existing_object = {
            name: value
            for name, value in existing.items()
            if name not in normalization_metadata
        }
        incoming_object = {
            name: value
            for name, value in row.items()
            if name not in normalization_metadata
        }
        if canonical_sha256(existing_object) != canonical_sha256(incoming_object):
            raise ValueError(f"{label} ID {key} produced conflicting normalized rows")
        return
    rows[key] = row


def normalize_context_days(
    days: Sequence[ContextCaptureDayResult],
    *,
    warmup_days: int = 10,
) -> dict[ContextRecordTable, pd.DataFrame]:
    """Normalize all nine typed objects plus exact candidate/decision/trade links."""

    state_rows: dict[str, dict[str, Any]] = {}
    capture_rows: dict[str, dict[str, Any]] = {}
    structure_rows: dict[str, dict[str, Any]] = {}
    delta_rows: dict[str, dict[str, Any]] = {}
    window_rows: dict[str, dict[str, Any]] = {}
    lifecycle_rows: dict[str, dict[str, Any]] = {}
    member_rows: dict[str, dict[str, Any]] = {}
    sweep_candidates: dict[str, list[tuple[object, dict[str, Any]]]] = {}
    provenance_rows: dict[str, dict[str, Any]] = {}
    candidate_events: dict[str, IfvgContextEvent] = {}
    decision_events: dict[str, IfvgContextEvent] = {}
    trade_events: dict[str, IfvgContextEvent] = {}
    swing_by_id: dict[str, tuple[ContextRecord, dict[str, Any]]] = {}
    core_candidates: dict[str, dict[str, Any]] = {}

    # Geometry linkage is sourced from the unchanged v2 stream, never inferred
    # from context row order or from a nearby capture.  Context capture evidence
    # identifies the candidate record itself; the v2 entry record separately
    # identifies the exact trigger geometry that the experiment contract binds.
    for day in days:
        if day.core_rows.empty:
            continue
        if "kind" not in day.core_rows:
            raise ValueError("context normalization core rows lack record kind")
        candidates = day.core_rows.loc[
            day.core_rows["kind"] == "entry_candidate"
        ]
        for row in candidates.to_dict("records"):
            candidate_id = row.get("candidate_id")
            if candidate_id is None or pd.isna(candidate_id) or not str(candidate_id):
                raise ValueError("v2 entry candidate lacks exact candidate_id")
            key = str(candidate_id)
            if key in core_candidates:
                raise ValueError(f"v2 candidate_id {key} is duplicated in replay stream")
            core_candidates[key] = row

    def add_provenance(record: ContextRecord, meta: dict[str, Any]) -> None:
        object_id = _record_id(record)
        object_type = type(record).__name__
        provenance_id = canonical_sha256(
            {"object_type": object_type, "object_id": object_id}
        )
        full = _record_row(record)
        row = {
            "provenance_id": provenance_id,
            "object_type": object_type,
            "object_id": object_id,
            **{
                key: full.get(key)
                for key in (
                    "schema_version",
                    "feature_set_version",
                    "feature_formula_version",
                    "feature_schema_hash",
                    "context_config_hash",
                    "strategy_core_commit",
                    "strategy_core_source_tree_hash",
                    "symbol",
                    "tick_size",
                    "as_of_ts",
                    "as_of_cursor",
                    "source_close_ts",
                    "source_confirmed_ts",
                    "valid",
                    "warmup_complete",
                    "source_available",
                    "missing_reason",
                )
            },
            **meta,
        }
        _insert_unique(provenance_rows, provenance_id, row, label="provenance")

    for chain_index, day in enumerate(days):
        meta = _metadata(day, chain_index, warmup_days)
        for swing in day.confirmed_swings:
            swing_id = str(swing.swing_id)
            swing_by_id[swing_id] = (swing, meta)
            add_provenance(swing, meta)

        for event in day.context_events:
            state = event.state
            state_row = _record_row(state)
            state_row = {
                key: value
                for key, value in state_row.items()
                if not key.startswith("mtf_snapshot_")
            }
            for selector in (
                "nearest_eqh",
                "nearest_eql",
                "nearest_thesis_supporting",
                "nearest_thesis_opposing",
            ):
                state_row.pop(f"nearest_context_{selector}", None)
            mtf_row = _record_row(state.mtf_snapshot)
            for key, value in mtf_row.items():
                if key != "states" and not key.startswith("local_state_"):
                    state_row[f"mtf_{key}"] = value
            state_row.update(meta)
            _insert_unique(
                state_rows,
                str(state.context_state_id),
                state_row,
                label="context state",
            )
            add_provenance(state, meta)
            add_provenance(state.mtf_snapshot, meta)

            capture_row = {**_record_row(event.capture), **meta}
            _insert_unique(
                capture_rows,
                str(event.capture.context_capture_id),
                capture_row,
                label="context capture",
            )
            add_provenance(event.capture, meta)

            for record, target, key_name, label in (
                *(
                    (item, structure_rows, "structure_state_id", "structure state")
                    for item in event.structure_states
                ),
                *(
                    (item, delta_rows, "structure_delta_id", "structure delta")
                    for item in event.structure_deltas
                ),
                *(
                    (item, window_rows, "displacement_window_id", "displacement window")
                    for item in event.displacement_windows
                ),
            ):
                row = {**_record_row(record), **meta}
                _insert_unique(target, str(getattr(record, key_name)), row, label=label)
                add_provenance(record, meta)
            for link in event.sweep_links:
                sweep_candidates.setdefault(str(link.sweep_link_id), []).append((link, meta))

            capture = event.capture
            if capture.capture_kind is CaptureKind.ENTRY_CANDIDATE:
                if not capture.candidate_id:
                    raise ValueError("entry candidate context lacks exact candidate_id")
                if capture.candidate_id in candidate_events:
                    raise ValueError("candidate_id has more than one context capture")
                candidate_events[capture.candidate_id] = event
            elif capture.capture_kind is CaptureKind.ELIGIBLE_DECISION:
                if not capture.decision_id or not capture.candidate_id:
                    raise ValueError("eligible decision context lacks exact IDs")
                if capture.decision_id in decision_events:
                    raise ValueError("decision_id has more than one context capture")
                decision_events[capture.decision_id] = event
            elif capture.capture_kind is CaptureKind.EXECUTED_TRADE_LINK:
                if not capture.trade_id or not capture.decision_id:
                    raise ValueError("executed trade context lacks exact IDs")
                if capture.trade_id in trade_events:
                    raise ValueError("trade_id has more than one context link")
                trade_events[capture.trade_id] = event

        for lifecycle in day.pool_lifecycle_events:
            row = {**_record_row(lifecycle), **meta}
            _insert_unique(
                lifecycle_rows,
                str(lifecycle.lifecycle_event_id),
                row,
                label="pool lifecycle",
            )
            add_provenance(lifecycle, meta)
            pool = lifecycle.pool
            for swing_id_value in pool.member_swing_ids:
                swing_id = str(swing_id_value)
                member_id = canonical_sha256(
                    {"pool_id": str(pool.pool_id), "swing_id": swing_id}
                )
                swing_entry = swing_by_id.get(swing_id)
                member = {
                    "pool_member_id": member_id,
                    "pool_id": str(pool.pool_id),
                    "member_swing_id": swing_id,
                    "pool_type": pool.pool_type.value,
                    "source_timeframe": pool.source_timeframe,
                    "source_timeframe_seconds": pool.source_timeframe_seconds,
                    "pool_confirmation_ts": pool.confirmation_ts.isoformat(),
                    "feature_set_version": pool.feature_set_version,
                    "feature_formula_version": pool.feature_formula_version,
                    "feature_schema_hash": pool.feature_schema_hash,
                    "context_config_hash": pool.context_config_hash,
                    **meta,
                }
                if swing_entry is not None:
                    swing, _swing_meta = swing_entry
                    member.update(
                        {
                            f"swing_{key}": value
                            for key, value in _record_row(swing).items()
                        }
                    )
                existing_member = member_rows.get(member_id)
                if existing_member is not None:
                    for invariant in (
                        "pool_id",
                        "member_swing_id",
                        "pool_type",
                        "source_timeframe_seconds",
                        "feature_schema_hash",
                        "context_config_hash",
                    ):
                        if existing_member[invariant] != member[invariant]:
                            raise ValueError(
                                f"pool member {member_id} changed {invariant}"
                            )
                else:
                    member_rows[member_id] = member
        for link in day.sweep_link_events:
            sweep_candidates.setdefault(str(link.sweep_link_id), []).append((link, meta))

    sweep_rows: dict[str, dict[str, Any]] = {}
    for link_id, candidates in sweep_candidates.items():
        geometry = {
            (
                str(item.pool_id),
                item.sweep_bar_id,
                item.sweep_cursor,
                item.sweep_depth_ticks,
            )
            for item, _meta in candidates
        }
        if len(geometry) != 1:
            raise ValueError(f"sweep link {link_id} changed immutable geometry")
        link, meta = max(
            candidates,
            key=lambda pair: (
                bool(pair[0].qualifies_opposing_leg),
                bool(pair[0].reclaimed_after_sweep),
                pair[0].as_of_ts,
                pair[0].as_of_cursor,
            ),
        )
        sweep_rows[link_id] = {**_record_row(link), **meta}
        add_provenance(link, meta)

    candidate_links: list[dict[str, Any]] = []
    for candidate_id, event in sorted(candidate_events.items()):
        capture = event.capture
        core_candidate = core_candidates.get(candidate_id)
        if core_candidate is None:
            raise ValueError(
                f"candidate {candidate_id} has no exact v2 entry record for geometry linkage"
            )
        geometry_id = core_candidate.get("trigger_evidence_id")
        geometry_cursor = core_candidate.get("trigger_cursor")
        if (
            geometry_id is None
            or pd.isna(geometry_id)
            or not str(geometry_id)
            or geometry_cursor is None
            or pd.isna(geometry_cursor)
            or not str(geometry_cursor)
        ):
            raise ValueError(
                f"candidate {candidate_id} lacks exact v2 trigger geometry evidence"
            )
        geometry_feature_cursor = core_candidate.get(
            "geometry_feature_as_of_cursor"
        )
        if (
            geometry_feature_cursor is not None
            and not pd.isna(geometry_feature_cursor)
            and str(geometry_feature_cursor) != str(geometry_cursor)
        ):
            raise ValueError(
                f"candidate {candidate_id} has contradictory geometry cursors"
            )
        core_setup_id = core_candidate.get("envelope_setup_id")
        if (
            core_setup_id is not None
            and not pd.isna(core_setup_id)
            and str(core_setup_id) != str(capture.setup_id)
        ):
            raise ValueError(
                f"candidate {candidate_id} context setup differs from exact v2 record"
            )
        candidate_links.append(
            {
                "candidate_id": candidate_id,
                "setup_id": capture.setup_id,
                "stage": CaptureKind.ENTRY_CANDIDATE.value,
                "context_capture_id": str(capture.context_capture_id),
                "context_state_id": str(capture.context_state_id),
                "capture_kind": capture.capture_kind.value,
                "geometry_evidence_id": str(geometry_id),
                "geometry_evidence_cursor": str(geometry_cursor),
                "context_as_of_ts": capture.as_of_ts,
                "feature_as_of_ts": capture.as_of_ts,
                "feature_set_version": capture.feature_set_version,
                "feature_formula_version": capture.feature_formula_version,
                "feature_schema_hash": capture.feature_schema_hash,
                "context_config_hash": capture.context_config_hash,
            }
        )

    decision_links: list[dict[str, Any]] = []
    for decision_id, event in sorted(decision_events.items()):
        capture = event.capture
        candidate = candidate_events.get(capture.candidate_id or "")
        if candidate is None:
            raise ValueError("decision has no exact candidate context capture")
        decision_links.append(
            {
                "decision_id": decision_id,
                "candidate_id": capture.candidate_id,
                "setup_id": capture.setup_id,
                "stage": CaptureKind.ELIGIBLE_DECISION.value,
                "context_capture_id": str(capture.context_capture_id),
                "candidate_context_capture_id": str(
                    candidate.capture.context_capture_id
                ),
                "context_state_id": str(capture.context_state_id),
                "capture_kind": capture.capture_kind.value,
                "geometry_evidence_id": capture.evidence_id,
                "geometry_evidence_cursor": capture.evidence_cursor,
                "context_as_of_ts": capture.as_of_ts,
                "feature_as_of_ts": capture.as_of_ts,
                "feature_set_version": capture.feature_set_version,
                "feature_formula_version": capture.feature_formula_version,
                "feature_schema_hash": capture.feature_schema_hash,
                "context_config_hash": capture.context_config_hash,
            }
        )

    trade_links: list[dict[str, Any]] = []
    for trade_id, event in sorted(trade_events.items()):
        capture = event.capture
        decision = decision_events.get(capture.decision_id or "")
        if decision is None:
            raise ValueError("trade has no exact decision context capture")
        trade_links.append(
            {
                "trade_id": trade_id,
                "decision_id": capture.decision_id,
                "candidate_id": capture.candidate_id,
                "setup_id": capture.setup_id,
                "stage": CaptureKind.EXECUTED_TRADE_LINK.value,
                "context_capture_id": str(capture.context_capture_id),
                "decision_context_capture_id": str(
                    decision.capture.context_capture_id
                ),
                "frozen_from_capture_id": str(capture.frozen_from_capture_id),
                "capture_kind": capture.capture_kind.value,
                "geometry_evidence_id": capture.evidence_id,
                "geometry_evidence_cursor": capture.evidence_cursor,
                "context_as_of_ts": capture.as_of_ts,
                "feature_as_of_ts": capture.as_of_ts,
                "feature_set_version": capture.feature_set_version,
                "feature_formula_version": capture.feature_formula_version,
                "feature_schema_hash": capture.feature_schema_hash,
                "context_config_hash": capture.context_config_hash,
            }
        )

    raw = {
        ContextRecordTable.CONTEXT_STATE: list(state_rows.values()),
        ContextRecordTable.CONTEXT_CAPTURE: list(capture_rows.values()),
        ContextRecordTable.CONTEXT_STRUCTURE_STATE: list(structure_rows.values()),
        ContextRecordTable.CONTEXT_STRUCTURE_DELTA: list(delta_rows.values()),
        ContextRecordTable.CONTEXT_DISPLACEMENT_WINDOW: list(window_rows.values()),
        ContextRecordTable.EQUAL_LEVEL_POOL_LIFECYCLE: list(lifecycle_rows.values()),
        ContextRecordTable.EQUAL_LEVEL_POOL_MEMBER: list(member_rows.values()),
        ContextRecordTable.EQUAL_LEVEL_SWEEP_LINK: list(sweep_rows.values()),
        ContextRecordTable.CONTEXT_VALIDITY_PROVENANCE: list(provenance_rows.values()),
        ContextRecordTable.CANDIDATE_CONTEXT_LINK: candidate_links,
        ContextRecordTable.DECISION_CONTEXT_LINK: decision_links,
        ContextRecordTable.TRADE_CONTEXT_LINK: trade_links,
    }
    return {
        table: stamp_context_table(table, pd.DataFrame(raw[table]))
        for table in ContextRecordTable
    }
