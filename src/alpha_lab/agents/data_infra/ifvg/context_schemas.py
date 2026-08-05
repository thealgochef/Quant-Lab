"""Canonical Arrow registry for all generation-4 IFVG context tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import pandas as pd
import pyarrow as pa

__all__ = [
    "IFVG_CONTEXT_ARROW_REGISTRY",
    "IFVG_CONTEXT_ARROW_REGISTRY_HASH",
    "IFVG_CONTEXT_ARROW_REGISTRY_ID",
    "context_arrow_schema",
    "context_arrow_schema_hash",
    "context_frame_from_rows",
    "context_frame_from_table",
    "context_table_from_frame",
    "context_table_from_rows",
    "validate_context_arrow_table",
]

IFVG_CONTEXT_ARROW_REGISTRY_ID = "ifvg_context_arrow_v1"


class _Table(StrEnum):
    CONTEXT_STATE = "context_state"
    CONTEXT_CAPTURE = "context_capture"
    CONTEXT_STRUCTURE_STATE = "context_structure_state"
    CONTEXT_STRUCTURE_DELTA = "context_structure_delta"
    CONTEXT_DISPLACEMENT_WINDOW = "context_displacement_window"
    EQUAL_LEVEL_POOL_LIFECYCLE = "equal_level_pool_lifecycle"
    EQUAL_LEVEL_POOL_MEMBER = "equal_level_pool_member"
    EQUAL_LEVEL_SWEEP_LINK = "equal_level_sweep_link"
    CONTEXT_VALIDITY_PROVENANCE = "context_validity_provenance"
    CANDIDATE_CONTEXT_LINK = "candidate_context_link"
    DECISION_CONTEXT_LINK = "decision_context_link"
    TRADE_CONTEXT_LINK = "trade_context_link"


UTF8 = pa.string()
INT64 = pa.int64()
FLOAT64 = pa.float64()
BOOL = pa.bool_()
UTC_TS = pa.timestamp("us", tz="UTC")
# Parquet canonicalizes the child name to ``element``.  Name it explicitly so
# a write/read cycle preserves schema equality including nested field names.
UTF8_LIST = pa.list_(pa.field("element", pa.string(), nullable=True))


def _field(name: str, kind: pa.DataType) -> pa.Field:
    # Optional values retain a fixed physical type. Required-value validation is
    # semantic and runs separately from Arrow construction.
    return pa.field(name, kind, nullable=True)


_COMMON = (
    _field("schema_version", INT64),
    _field("feature_set_version", UTF8),
    _field("feature_formula_version", UTF8),
    _field("feature_schema_hash", UTF8),
    _field("context_config_hash", UTF8),
    _field("strategy_core_commit", UTF8),
    _field("strategy_core_source_tree_hash", UTF8),
    _field("symbol", UTF8),
    _field("tick_size", UTF8),
    _field("as_of_ts", UTC_TS),
    _field("as_of_cursor", UTF8),
    _field("source_close_ts", UTC_TS),
    _field("source_confirmed_ts", UTC_TS),
    _field("valid", BOOL),
    _field("warmup_complete", BOOL),
    _field("source_available", BOOL),
    _field("missing_reason", UTF8),
)

_NORMALIZATION = (
    _field("source_date", UTF8),
    _field("is_warmup", BOOL),
    _field("days_of_htf_history", INT64),
    _field("entering_context_seed_hash", UTF8),
)

_STAMP = (
    _field("record_table", UTF8),
    _field("context_container_schema_version", INT64),
    _field("dataset_schema_version", INT64),
)

_POOL_REFERENCE = (
    _field("pool_id", UTF8),
    _field("pool_type", UTF8),
    _field("source_timeframe", UTF8),
    _field("source_timeframe_seconds", INT64),
    _field("lower_bound_ticks", INT64),
    _field("upper_bound_ticks", INT64),
    _field("representative_price_ticks", INT64),
    _field("distance_ticks", INT64),
    _field("distance_normalized_by_atr", FLOAT64),
    _field("swing_count", INT64),
    _field("age_minutes", FLOAT64),
    _field("span_minutes", FLOAT64),
    _field("width_ticks", INT64),
    _field("width_normalized_by_atr", FLOAT64),
    _field("creation_separation_ticks", INT64),
    _field("creation_separation_normalized_by_atr", FLOAT64),
    _field("creation_separation_normalized_by_local_range", FLOAT64),
)

_MTF_COMMON = tuple(
    _field(
        f"mtf_{item.name}",
        item.type,
    )
    for item in _COMMON
)

_MTF_FIELDS = (
    _field("mtf_mtf_snapshot_id", UTF8),
    _field("mtf_setup_id", UTF8),
    _field("mtf_setup_direction", UTF8),
    _field("mtf_active_mtf_timeframes", UTF8_LIST),
    _field("mtf_state_ids", UTF8_LIST),
    _field("mtf_local_structure_state_id", UTF8),
    _field("mtf_aligned_tf_count", INT64),
    _field("mtf_conflicting_tf_count", INT64),
    _field("mtf_neutral_tf_count", INT64),
    _field("mtf_valid_tf_count", INT64),
    _field("mtf_break_aligned_tf_count", INT64),
    _field("mtf_break_conflicting_tf_count", INT64),
    _field("mtf_highest_aligned_tf_seconds", INT64),
    _field("mtf_highest_conflicting_tf_seconds", INT64),
    _field("mtf_lowest_conflicting_tf_seconds", INT64),
    _field("mtf_contiguous_alignment_span", INT64),
    _field("mtf_contiguous_conflict_span", INT64),
    _field("mtf_adjacent_tf_transition_count", INT64),
    _field("mtf_lower_support_higher_conflict", BOOL),
    _field("mtf_lower_conflict_higher_support", BOOL),
    _field("mtf_execution_pullback_inside_htf_trend", BOOL),
    _field("mtf_execution_expansion_against_htf_trend", BOOL),
)

_STRUCTURE_FIELDS = (
    _field("structure_state_id", UTF8),
    _field("source_timeframe", UTF8),
    _field("source_timeframe_seconds", INT64),
    _field("anchor_status", UTF8),
    _field("latest_high_swing_id", UTF8),
    _field("latest_low_swing_id", UTF8),
    _field("prior_high_swing_id", UTF8),
    _field("prior_low_swing_id", UTF8),
    _field("high_relationship", UTF8),
    _field("low_relationship", UTF8),
    _field("swing_sequence_state", UTF8),
    _field("structure_direction", UTF8),
    _field("last_break_id", UTF8),
    _field("last_break_type", UTF8),
    _field("last_break_direction", UTF8),
    _field("broken_swing_id", UTF8),
    _field("break_bar_id", UTF8),
    _field("break_ts", UTC_TS),
    _field("bars_since_break", INT64),
    _field("minutes_since_break", FLOAT64),
    _field("state_age_bars", INT64),
    _field("structure_alignment", INT64),
    _field("break_alignment", INT64),
    _field("last_confirmed_swing_high_ticks", INT64),
    _field("last_confirmed_swing_low_ticks", INT64),
)

_DISPLACEMENT_METRICS: tuple[tuple[str, pa.DataType], ...] = (
    ("observed_bar_count", INT64),
    ("expected_eligible_bar_count", INT64),
    ("missing_bar_count", INT64),
    ("wall_elapsed_minutes", FLOAT64),
    ("eligible_elapsed_minutes", FLOAT64),
    ("range_ticks_mean", FLOAT64),
    ("range_ticks_min", INT64),
    ("range_ticks_max", INT64),
    ("body_ticks_sum", INT64),
    ("upper_wick_ticks_mean", FLOAT64),
    ("lower_wick_ticks_mean", FLOAT64),
    ("true_range_ticks_mean", FLOAT64),
    ("body_fraction_mean", FLOAT64),
    ("body_fraction_min", FLOAT64),
    ("body_fraction_max", FLOAT64),
    ("upper_wick_fraction_mean", FLOAT64),
    ("lower_wick_fraction_mean", FLOAT64),
    ("directional_wick_fraction_mean", FLOAT64),
    ("opposing_wick_fraction_mean", FLOAT64),
    ("raw_close_location_mean", FLOAT64),
    ("expected_close_location_mean", FLOAT64),
    ("setup_close_location_mean", FLOAT64),
    ("overlap_fraction_mean", FLOAT64),
    ("bullish_bar_count", INT64),
    ("bearish_bar_count", INT64),
    ("doji_bar_count", INT64),
    ("directional_bar_count", INT64),
    ("opposing_bar_count", INT64),
    ("directional_bar_fraction", FLOAT64),
    ("opposing_bar_fraction", FLOAT64),
    ("max_consecutive_directional_bars", INT64),
    ("directional_body_ticks_sum", INT64),
    ("opposing_body_ticks_sum", INT64),
    ("raw_close_progress_ticks_mean", FLOAT64),
    ("expected_close_progress_ticks_mean", FLOAT64),
    ("setup_close_progress_ticks_mean", FLOAT64),
    ("raw_net_move_ticks", INT64),
    ("expected_net_move_ticks", INT64),
    ("setup_net_move_ticks", INT64),
    ("raw_net_move_normalized", FLOAT64),
    ("expected_net_move_normalized", FLOAT64),
    ("setup_net_move_normalized", FLOAT64),
    ("raw_velocity_normalized", FLOAT64),
    ("expected_velocity_normalized", FLOAT64),
    ("setup_velocity_normalized", FLOAT64),
    ("path_efficiency_abs", FLOAT64),
    ("raw_path_efficiency_signed", FLOAT64),
    ("expected_path_efficiency_signed", FLOAT64),
    ("setup_path_efficiency_signed", FLOAT64),
    ("max_pullback_ticks", INT64),
    ("max_pullback_fraction", FLOAT64),
    ("bullish_fvg_count", INT64),
    ("bearish_fvg_count", INT64),
    ("directional_fvg_count", INT64),
    ("opposing_fvg_count", INT64),
    ("bullish_fvg_width_sum_ticks", INT64),
    ("bearish_fvg_width_sum_ticks", INT64),
    ("directional_fvg_width_sum_ticks", INT64),
    ("opposing_fvg_width_sum_ticks", INT64),
    ("directional_fvg_width_sum_normalized", FLOAT64),
    ("directional_gap_density", FLOAT64),
)

_POOL_FIELDS = (
    _field("pool_id", UTF8),
    _field("pool_type", UTF8),
    _field("source_timeframe", UTF8),
    _field("source_timeframe_seconds", INT64),
    _field("lower_bound_ticks", INT64),
    _field("upper_bound_ticks", INT64),
    _field("representative_price_ticks", INT64),
    _field("member_swing_ids", UTF8_LIST),
    _field("swing_count", INT64),
    _field("first_pivot_ts", UTC_TS),
    _field("latest_pivot_ts", UTC_TS),
    _field("confirmation_ts", UTC_TS),
    _field("last_update_ts", UTC_TS),
    _field("last_update_cursor", UTF8),
    _field("absolute_separation_ticks", INT64),
    _field("atr14_ticks", FLOAT64),
    _field("local_range20_ticks", INT64),
    _field("separation_normalized_by_atr", FLOAT64),
    _field("separation_normalized_by_local_range", FLOAT64),
    _field("active", BOOL),
    _field("swept", BOOL),
    _field("sweep_link_id", UTF8),
    _field("sweep_ts", UTC_TS),
    _field("reclaimed", BOOL),
    _field("reclaim_ts", UTC_TS),
    _field("invalidation_reason", UTF8),
    _field("expiration_reason", UTF8),
    _field("tolerance_policy", UTF8),
    _field("lifecycle_version", INT64),
)

_SWING_FIELDS = (
    _field("swing_id", UTF8),
    _field("source_timeframe", UTF8),
    _field("source_timeframe_seconds", INT64),
    _field("side", UTF8),
    _field("price_ticks", INT64),
    _field("pivot_bar_id", UTF8),
    _field("pivot_ts", UTC_TS),
    _field("confirmation_bar_id", UTF8),
    _field("confirmation_ts", UTC_TS),
    _field("availability_cursor", UTF8),
    _field("confirmation_strength", INT64),
)


def _prefix(fields: Sequence[pa.Field], prefix: str) -> tuple[pa.Field, ...]:
    return tuple(_field(f"{prefix}{item.name}", item.type) for item in fields)


def _schema(table: _Table, fields: Sequence[pa.Field]) -> pa.Schema:
    names = [field.name for field in fields]
    if len(names) != len(set(names)):
        raise RuntimeError(f"{table.value} Arrow schema contains duplicate columns")
    return pa.schema(
        fields,
        metadata={
            b"ifvg.registry": IFVG_CONTEXT_ARROW_REGISTRY_ID.encode("ascii"),
            b"ifvg.table": table.value.encode("ascii"),
            b"ifvg.schema_version": b"1",
        },
    )


_SCHEMAS: dict[_Table, pa.Schema] = {
    _Table.CONTEXT_STATE: _schema(
        _Table.CONTEXT_STATE,
        (
            *_COMMON,
            _field("context_state_id", UTF8),
            _field("state_payload_hash", UTF8),
            _field("setup_id", UTF8),
            _field("setup_direction", UTF8),
            _field("local_structure_state_id", UTF8),
            _field("active_pool_state_hash", UTF8),
            *(
                _field(f"nearest_context_{selector}_{item.name}", item.type)
                for selector in (
                    "nearest_eqh",
                    "nearest_eql",
                    "nearest_thesis_supporting",
                    "nearest_thesis_opposing",
                )
                for item in _POOL_REFERENCE
            ),
            _field("containing_pool_count", INT64),
            *_MTF_COMMON,
            *_MTF_FIELDS,
            *_NORMALIZATION,
            *_STAMP,
        ),
    ),
    _Table.CONTEXT_CAPTURE: _schema(
        _Table.CONTEXT_CAPTURE,
        (
            *_COMMON,
            _field("context_capture_id", UTF8),
            _field("capture_kind", UTF8),
            _field("context_state_id", UTF8),
            _field("setup_id", UTF8),
            _field("candidate_id", UTF8),
            _field("decision_id", UTF8),
            _field("trade_id", UTF8),
            _field("evidence_id", UTF8),
            _field("evidence_cursor", UTF8),
            _field("displacement_window_ids", UTF8_LIST),
            _field("structure_delta_ids", UTF8_LIST),
            _field("opposing_leg_sweep_link_ids", UTF8_LIST),
            _field("selected_opposing_leg_sweep_link_id", UTF8),
            _field("frozen_from_capture_id", UTF8),
            *_NORMALIZATION,
            *_STAMP,
        ),
    ),
    _Table.CONTEXT_STRUCTURE_STATE: _schema(
        _Table.CONTEXT_STRUCTURE_STATE,
        (*_COMMON, *_STRUCTURE_FIELDS, *_NORMALIZATION, *_STAMP),
    ),
    _Table.CONTEXT_STRUCTURE_DELTA: _schema(
        _Table.CONTEXT_STRUCTURE_DELTA,
        (
            *_COMMON,
            _field("structure_delta_id", UTF8),
            _field("delta_kind", UTF8),
            _field("from_snapshot_id", UTF8),
            _field("to_snapshot_id", UTF8),
            _field("changed_timeframe_count", INT64),
            _field("support_to_conflict_count", INT64),
            _field("conflict_to_support_count", INT64),
            _field("direction_flip_count", INT64),
            _field("newly_confirmed_break_count", INT64),
            *_NORMALIZATION,
            *_STAMP,
        ),
    ),
    _Table.CONTEXT_DISPLACEMENT_WINDOW: _schema(
        _Table.CONTEXT_DISPLACEMENT_WINDOW,
        (
            *_COMMON,
            _field("displacement_window_id", UTF8),
            _field("setup_id", UTF8),
            _field("window_kind", UTF8),
            _field("start_evidence_id", UTF8),
            _field("end_evidence_id", UTF8),
            _field("b0_bar_id", UTF8),
            _field("start_ts", UTC_TS),
            _field("last_included_bar_id", UTF8),
            _field("end_ts", UTC_TS),
            _field("expected_orientation", UTF8),
            _field("observed_bar_ids_hash", UTF8),
            _field("calendar_policy_id", UTF8),
            _field("calendar_schedule_hash", UTF8),
            _field("source_gap_policy_id", UTF8),
            _field("source_coverage_hash", UTF8),
            _field("source_partition_dates", UTF8_LIST),
            _field("unavailable_partition_dates", UTF8_LIST),
            *(
                _field(f"metric_missing_reasons_{name}", UTF8)
                for name, _kind in _DISPLACEMENT_METRICS
            ),
            *(
                _field(f"metrics_{name}", kind)
                for name, kind in _DISPLACEMENT_METRICS
            ),
            *_NORMALIZATION,
            *_STAMP,
        ),
    ),
    _Table.EQUAL_LEVEL_POOL_LIFECYCLE: _schema(
        _Table.EQUAL_LEVEL_POOL_LIFECYCLE,
        (
            *_COMMON,
            _field("lifecycle_event_id", UTF8),
            _field("event_type", UTF8),
            *_prefix(_COMMON, "pool_"),
            *_prefix(_POOL_FIELDS, "pool_"),
            _field("member_swing_id", UTF8),
            *_NORMALIZATION,
            *_STAMP,
        ),
    ),
    _Table.EQUAL_LEVEL_POOL_MEMBER: _schema(
        _Table.EQUAL_LEVEL_POOL_MEMBER,
        (
            _field("pool_member_id", UTF8),
            _field("pool_id", UTF8),
            _field("member_swing_id", UTF8),
            _field("pool_type", UTF8),
            _field("source_timeframe", UTF8),
            _field("source_timeframe_seconds", INT64),
            _field("pool_confirmation_ts", UTC_TS),
            _field("feature_set_version", UTF8),
            _field("feature_formula_version", UTF8),
            _field("feature_schema_hash", UTF8),
            _field("context_config_hash", UTF8),
            *_prefix(_COMMON, "swing_"),
            *_prefix(_SWING_FIELDS, "swing_"),
            *_NORMALIZATION,
            *_STAMP,
        ),
    ),
    _Table.EQUAL_LEVEL_SWEEP_LINK: _schema(
        _Table.EQUAL_LEVEL_SWEEP_LINK,
        (
            *_COMMON,
            _field("sweep_link_id", UTF8),
            _field("pool_id", UTF8),
            _field("pool_type", UTF8),
            _field("source_timeframe", UTF8),
            _field("source_timeframe_seconds", INT64),
            _field("sweep_bar_id", UTF8),
            _field("sweep_cursor", UTF8),
            _field("sweep_ts", UTC_TS),
            _field("sweep_depth_ticks", INT64),
            _field("sweep_depth_normalized", FLOAT64),
            _field("reclaimed_after_sweep", BOOL),
            _field("reclaim_bar_id", UTF8),
            _field("reclaim_ts", UTC_TS),
            _field("reclaim_latency_bars", INT64),
            _field("reclaim_close_distance_ticks", INT64),
            _field("reclaim_close_distance_normalized", FLOAT64),
            _field("setup_id", UTF8),
            _field("opposing_fvg_id", UTF8),
            _field("parent_lock_cursor", UTF8),
            _field("inversion_cursor", UTF8),
            _field("qualifies_opposing_leg", BOOL),
            _field("distance_at_lock_ticks", INT64),
            _field("distance_at_lock_normalized", FLOAT64),
            *_NORMALIZATION,
            *_STAMP,
        ),
    ),
    _Table.CONTEXT_VALIDITY_PROVENANCE: _schema(
        _Table.CONTEXT_VALIDITY_PROVENANCE,
        (
            _field("provenance_id", UTF8),
            _field("object_type", UTF8),
            _field("object_id", UTF8),
            *_COMMON,
            *_NORMALIZATION,
            *_STAMP,
        ),
    ),
    _Table.CANDIDATE_CONTEXT_LINK: _schema(
        _Table.CANDIDATE_CONTEXT_LINK,
        (
            _field("candidate_id", UTF8),
            _field("setup_id", UTF8),
            _field("stage", UTF8),
            _field("context_capture_id", UTF8),
            _field("context_state_id", UTF8),
            _field("capture_kind", UTF8),
            _field("geometry_evidence_id", UTF8),
            _field("geometry_evidence_cursor", UTF8),
            _field("context_as_of_ts", UTC_TS),
            _field("feature_as_of_ts", UTC_TS),
            _field("feature_set_version", UTF8),
            _field("feature_formula_version", UTF8),
            _field("feature_schema_hash", UTF8),
            _field("context_config_hash", UTF8),
            *_STAMP,
        ),
    ),
    _Table.DECISION_CONTEXT_LINK: _schema(
        _Table.DECISION_CONTEXT_LINK,
        (
            _field("decision_id", UTF8),
            _field("candidate_id", UTF8),
            _field("setup_id", UTF8),
            _field("stage", UTF8),
            _field("context_capture_id", UTF8),
            _field("candidate_context_capture_id", UTF8),
            _field("context_state_id", UTF8),
            _field("capture_kind", UTF8),
            _field("geometry_evidence_id", UTF8),
            _field("geometry_evidence_cursor", UTF8),
            _field("context_as_of_ts", UTC_TS),
            _field("feature_as_of_ts", UTC_TS),
            _field("feature_set_version", UTF8),
            _field("feature_formula_version", UTF8),
            _field("feature_schema_hash", UTF8),
            _field("context_config_hash", UTF8),
            *_STAMP,
        ),
    ),
    _Table.TRADE_CONTEXT_LINK: _schema(
        _Table.TRADE_CONTEXT_LINK,
        (
            _field("trade_id", UTF8),
            _field("decision_id", UTF8),
            _field("candidate_id", UTF8),
            _field("setup_id", UTF8),
            _field("stage", UTF8),
            _field("context_capture_id", UTF8),
            _field("decision_context_capture_id", UTF8),
            _field("frozen_from_capture_id", UTF8),
            _field("capture_kind", UTF8),
            _field("geometry_evidence_id", UTF8),
            _field("geometry_evidence_cursor", UTF8),
            _field("context_as_of_ts", UTC_TS),
            _field("feature_as_of_ts", UTC_TS),
            _field("feature_set_version", UTF8),
            _field("feature_formula_version", UTF8),
            _field("feature_schema_hash", UTF8),
            _field("context_config_hash", UTF8),
            *_STAMP,
        ),
    ),
}

IFVG_CONTEXT_ARROW_REGISTRY: Mapping[str, pa.Schema] = MappingProxyType(
    {table.value: schema for table, schema in _SCHEMAS.items()}
)


def _table_name(table: object) -> str:
    value = getattr(table, "value", table)
    return _Table(str(value)).value


def context_arrow_schema(table: object) -> pa.Schema:
    return IFVG_CONTEXT_ARROW_REGISTRY[_table_name(table)]


def context_arrow_schema_hash(table: object) -> str:
    return hashlib.sha256(context_arrow_schema(table).serialize().to_pybytes()).hexdigest()


IFVG_CONTEXT_ARROW_REGISTRY_HASH = hashlib.sha256(
    json.dumps(
        [(table.value, context_arrow_schema_hash(table)) for table in _Table],
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _is_null(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    if getattr(result, "ndim", None) == 0:
        return bool(result)
    return False


def _coerce_value(value: Any, kind: pa.DataType) -> Any:
    if _is_null(value):
        return None
    if pa.types.is_timestamp(kind):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            raise ValueError("context timestamp is naive")
        return stamp.tz_convert("UTC").to_pydatetime()
    if pa.types.is_list(kind):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ValueError("context ID collection is not valid JSON/list data") from error
        if hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, Sequence) or isinstance(value, bytes | bytearray | str):
            raise ValueError("context ID collection must be a sequence")
        return [str(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, str | bytes):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def context_table_from_rows(
    table: object,
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_missing: bool = False,
) -> pa.Table:
    schema = context_arrow_schema(table)
    expected = set(schema.names)
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        actual = set(row)
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        if extra:
            raise ValueError(f"{_table_name(table)} row {index} has undeclared columns {extra}")
        if missing and not allow_missing:
            raise ValueError(f"{_table_name(table)} row {index} is missing columns {missing}")
        normalized.append(
            {
                field.name: _coerce_value(row.get(field.name), field.type)
                for field in schema
            }
        )
    result = pa.Table.from_pylist(normalized, schema=schema)
    validate_context_arrow_table(table, result)
    return result


def context_table_from_frame(
    table: object,
    frame: pd.DataFrame,
    *,
    allow_missing: bool = False,
) -> pa.Table:
    schema = context_arrow_schema(table)
    actual = set(frame.columns)
    expected = set(schema.names)
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra:
        raise ValueError(f"{_table_name(table)} has undeclared columns {extra}")
    if missing and not allow_missing:
        raise ValueError(f"{_table_name(table)} is missing declared columns {missing}")
    if frame.empty:
        return pa.Table.from_pylist([], schema=schema)
    return context_table_from_rows(
        table,
        frame.to_dict("records"),
        allow_missing=allow_missing,
    )


def context_frame_from_rows(
    table: object,
    rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    arrow = context_table_from_rows(table, rows, allow_missing=True)
    return context_frame_from_table(arrow)


def context_frame_from_table(value: pa.Table) -> pd.DataFrame:
    # Keep the in-memory normalization lane JSON-canonical: pandas otherwise
    # coerces nullable integer/float columns to NaN, which is deliberately
    # forbidden by Strategy-Core's canonical serializer.  Physical dtypes are
    # enforced by the Arrow schema at the immutable Parquet boundary.
    rows = value.to_pylist()
    return pd.DataFrame(
        {
            field.name: pd.Series(
                [row[field.name] for row in rows],
                dtype="object",
            )
            for field in value.schema
        }
    )


def validate_context_arrow_table(table: object, value: pa.Table) -> None:
    expected = context_arrow_schema(table)
    if not value.schema.equals(expected, check_metadata=True):
        raise ValueError(
            f"{_table_name(table)} Arrow schema mismatch: "
            f"expected={expected}, observed={value.schema}"
        )
