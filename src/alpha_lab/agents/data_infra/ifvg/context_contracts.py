"""Separate IFVG generation-3 normalized context contracts.

Nothing in this module extends the frozen v2 ``RecordTable`` or its generic
emission flattener.  Context records remain an additive, exact-ID evidence lane.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import pandas as pd

from .context_schemas import context_frame_from_table, context_table_from_frame
from .contracts import RecordTable

__all__ = [
    "IFVG_CONTEXT_CONTAINER_SCHEMA_VERSION",
    "IFVG_CONTEXT_DATASET_SCHEMA_VERSION",
    "ContextRecordTable",
    "stamp_context_table",
    "validate_context_primary_keys",
    "validate_context_table_identity",
    "validate_context_foreign_keys",
    "context_count_reconciliation",
]

IFVG_CONTEXT_CONTAINER_SCHEMA_VERSION = 4
IFVG_CONTEXT_DATASET_SCHEMA_VERSION = 4


class ContextRecordTable(StrEnum):
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


_PRIMARY_KEYS: dict[ContextRecordTable, str] = {
    ContextRecordTable.CONTEXT_STATE: "context_state_id",
    ContextRecordTable.CONTEXT_CAPTURE: "context_capture_id",
    ContextRecordTable.CONTEXT_STRUCTURE_STATE: "structure_state_id",
    ContextRecordTable.CONTEXT_STRUCTURE_DELTA: "structure_delta_id",
    ContextRecordTable.CONTEXT_DISPLACEMENT_WINDOW: "displacement_window_id",
    ContextRecordTable.EQUAL_LEVEL_POOL_LIFECYCLE: "lifecycle_event_id",
    ContextRecordTable.EQUAL_LEVEL_POOL_MEMBER: "pool_member_id",
    ContextRecordTable.EQUAL_LEVEL_SWEEP_LINK: "sweep_link_id",
    ContextRecordTable.CONTEXT_VALIDITY_PROVENANCE: "provenance_id",
    ContextRecordTable.CANDIDATE_CONTEXT_LINK: "candidate_id",
    ContextRecordTable.DECISION_CONTEXT_LINK: "decision_id",
    ContextRecordTable.TRADE_CONTEXT_LINK: "trade_id",
}

_PROVENANCE_COLUMNS = (
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
    "valid",
    "warmup_complete",
    "source_available",
    "missing_reason",
)


def _coerce(table: ContextRecordTable | str) -> ContextRecordTable:
    return table if isinstance(table, ContextRecordTable) else ContextRecordTable(table)


def stamp_context_table(
    table: ContextRecordTable | str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    table = _coerce(table)
    out = frame.copy()
    out["record_table"] = table.value
    out["context_container_schema_version"] = IFVG_CONTEXT_CONTAINER_SCHEMA_VERSION
    out["dataset_schema_version"] = IFVG_CONTEXT_DATASET_SCHEMA_VERSION
    return context_frame_from_table(
        context_table_from_frame(table, out, allow_missing=True)
    )


def validate_context_primary_keys(
    table: ContextRecordTable | str,
    frame: pd.DataFrame,
) -> None:
    table = _coerce(table)
    context_table_from_frame(table, frame)
    if frame.empty:
        return
    key = _PRIMARY_KEYS[table]
    if key not in frame:
        raise ValueError(f"{table.value} is missing primary key {key!r}")
    values = frame[key]
    if values.isna().any() or (values.astype(str) == "").any():
        raise ValueError(f"{table.value}.{key} contains null/empty values")
    duplicates = values.duplicated(keep=False)
    if duplicates.any():
        raise ValueError(
            f"{table.value}.{key} is not unique: "
            f"{sorted(values.loc[duplicates].astype(str).unique())[:5]}"
        )


def validate_context_table_identity(
    table: ContextRecordTable | str,
    frame: pd.DataFrame,
) -> None:
    table = _coerce(table)
    context_table_from_frame(table, frame)
    if frame.empty:
        return
    required = {
        "record_table",
        "context_container_schema_version",
        "dataset_schema_version",
    }
    if table not in {
        ContextRecordTable.EQUAL_LEVEL_POOL_MEMBER,
        ContextRecordTable.CANDIDATE_CONTEXT_LINK,
        ContextRecordTable.DECISION_CONTEXT_LINK,
        ContextRecordTable.TRADE_CONTEXT_LINK,
    }:
        required.update(_PROVENANCE_COLUMNS)
    else:
        required.update(
            {
                "feature_set_version",
                "feature_formula_version",
                "feature_schema_hash",
                "context_config_hash",
            }
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{table.value} is missing identity columns {missing}")
    if not (frame["record_table"].astype(str) == table.value).all():
        raise ValueError(f"{table.value} contains another record_table")
    for column, expected in (
        ("context_container_schema_version", IFVG_CONTEXT_CONTAINER_SCHEMA_VERSION),
        ("dataset_schema_version", IFVG_CONTEXT_DATASET_SCHEMA_VERSION),
    ):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not (values == expected).all():
            raise ValueError(f"{table.value}.{column} must be exactly {expected}")
    for column in (
        "feature_set_version",
        "feature_formula_version",
        "feature_schema_hash",
        "context_config_hash",
    ):
        if frame[column].isna().any() or (frame[column].astype(str) == "").any():
            raise ValueError(f"{table.value}.{column} contains null/empty values")


def _ids(frame: pd.DataFrame, column: str) -> set[str]:
    if frame.empty or column not in frame:
        return set()
    return set(frame[column].dropna().astype(str))


def _require_subset(
    child: pd.DataFrame,
    column: str,
    parents: set[str],
    label: str,
) -> None:
    unknown = _ids(child, column) - parents
    if unknown:
        raise ValueError(f"{label} has missing exact parents: {sorted(unknown)[:5]}")


def validate_context_foreign_keys(
    tables: Mapping[ContextRecordTable | str, pd.DataFrame],
    *,
    core_tables: Mapping[RecordTable | str, pd.DataFrame] | None = None,
) -> None:
    """Validate only exact object IDs; no temporal or setup fallback is accepted."""

    normalized = {_coerce(key): value for key, value in tables.items()}
    states = normalized.get(ContextRecordTable.CONTEXT_STATE, pd.DataFrame())
    captures = normalized.get(ContextRecordTable.CONTEXT_CAPTURE, pd.DataFrame())
    candidates = normalized.get(ContextRecordTable.CANDIDATE_CONTEXT_LINK, pd.DataFrame())
    decisions = normalized.get(ContextRecordTable.DECISION_CONTEXT_LINK, pd.DataFrame())
    trades = normalized.get(ContextRecordTable.TRADE_CONTEXT_LINK, pd.DataFrame())

    required_link_columns = {
        ContextRecordTable.CANDIDATE_CONTEXT_LINK: {
            "candidate_id",
            "context_capture_id",
        },
        ContextRecordTable.DECISION_CONTEXT_LINK: {
            "decision_id",
            "candidate_id",
            "context_capture_id",
            "candidate_context_capture_id",
        },
        ContextRecordTable.TRADE_CONTEXT_LINK: {
            "trade_id",
            "decision_id",
            "candidate_id",
            "context_capture_id",
            "decision_context_capture_id",
            "frozen_from_capture_id",
        },
    }
    for table, frame in (
        (ContextRecordTable.CANDIDATE_CONTEXT_LINK, candidates),
        (ContextRecordTable.DECISION_CONTEXT_LINK, decisions),
        (ContextRecordTable.TRADE_CONTEXT_LINK, trades),
    ):
        if frame.empty:
            continue
        missing = sorted(required_link_columns[table] - set(frame.columns))
        if missing:
            raise ValueError(f"{table.value} is missing exact-ID columns {missing}")

    state_ids = _ids(states, "context_state_id")
    capture_ids = _ids(captures, "context_capture_id")
    candidate_ids = _ids(candidates, "candidate_id")
    decision_ids = _ids(decisions, "decision_id")

    _require_subset(captures, "context_state_id", state_ids, "context_capture.context_state_id")
    _require_subset(
        candidates,
        "context_capture_id",
        capture_ids,
        "candidate_context_link.context_capture_id",
    )
    _require_subset(
        decisions,
        "context_capture_id",
        capture_ids,
        "decision_context_link.context_capture_id",
    )
    _require_subset(
        decisions,
        "candidate_context_capture_id",
        capture_ids,
        "decision_context_link.candidate_context_capture_id",
    )
    _require_subset(
        decisions,
        "candidate_id",
        candidate_ids,
        "decision_context_link.candidate_id",
    )
    _require_subset(
        trades,
        "context_capture_id",
        capture_ids,
        "trade_context_link.context_capture_id",
    )
    _require_subset(
        trades,
        "decision_context_capture_id",
        capture_ids,
        "trade_context_link.decision_context_capture_id",
    )
    _require_subset(
        trades,
        "decision_id",
        decision_ids,
        "trade_context_link.decision_id",
    )
    _require_subset(
        trades,
        "candidate_id",
        candidate_ids,
        "trade_context_link.candidate_id",
    )

    capture_by_id = (
        captures.set_index(captures["context_capture_id"].astype(str), drop=False)
        if not captures.empty
        else pd.DataFrame()
    )
    for link_table, frame, expected_kind in (
        (ContextRecordTable.CANDIDATE_CONTEXT_LINK, candidates, "entry_candidate"),
        (ContextRecordTable.DECISION_CONTEXT_LINK, decisions, "eligible_decision"),
        (ContextRecordTable.TRADE_CONTEXT_LINK, trades, "executed_trade_link"),
    ):
        if frame.empty:
            continue
        for row in frame.to_dict("records"):
            capture = capture_by_id.loc[str(row["context_capture_id"])]
            if str(capture["capture_kind"]) != expected_kind:
                raise ValueError(
                    f"{link_table.value} points to {capture['capture_kind']!r}, "
                    f"expected {expected_kind!r}"
                )
            for exact_id in ("candidate_id", "decision_id", "trade_id"):
                if exact_id in row and pd.notna(row[exact_id]):
                    captured = capture.get(exact_id)
                    if pd.isna(captured) or str(captured) != str(row[exact_id]):
                        raise ValueError(
                            f"{link_table.value}.{exact_id} does not match exact capture"
                        )

    if not decisions.empty:
        candidate_capture_for = dict(
            zip(
                candidates["candidate_id"].astype(str),
                candidates["context_capture_id"].astype(str),
                strict=True,
            )
        )
        for row in decisions.to_dict("records"):
            expected = candidate_capture_for.get(str(row["candidate_id"]))
            if expected != str(row["candidate_context_capture_id"]):
                raise ValueError("decision context used a non-exact candidate join")
    if not trades.empty:
        decision_capture_for = dict(
            zip(
                decisions["decision_id"].astype(str),
                decisions["context_capture_id"].astype(str),
                strict=True,
            )
        )
        for row in trades.to_dict("records"):
            expected = decision_capture_for.get(str(row["decision_id"]))
            if expected != str(row["decision_context_capture_id"]):
                raise ValueError("trade context used a non-exact decision join")
            if str(row.get("frozen_from_capture_id")) != expected:
                raise ValueError("trade context is not frozen from its decision capture")

    if core_tables is not None:
        core = {
            key if isinstance(key, RecordTable) else RecordTable(key): value
            for key, value in core_tables.items()
        }
        exact_pairs = (
            (RecordTable.ENTRY_CANDIDATE, "candidate_id", candidate_ids),
            (RecordTable.ELIGIBLE_DECISION, "decision_id", decision_ids),
            (RecordTable.EXECUTED_TRADE, "trade_id", _ids(trades, "trade_id")),
        )
        for core_table, key, linked in exact_pairs:
            expected = _ids(core.get(core_table, pd.DataFrame()), key)
            if linked != expected:
                raise ValueError(
                    f"{core_table.value} exact context coverage mismatch: "
                    f"missing={sorted(expected - linked)[:5]}, "
                    f"extra={sorted(linked - expected)[:5]}"
                )


def context_count_reconciliation(
    tables: Mapping[ContextRecordTable | str, pd.DataFrame],
) -> dict:
    normalized = {_coerce(key): value for key, value in tables.items()}
    return {
        "schema_version": IFVG_CONTEXT_DATASET_SCHEMA_VERSION,
        "table_rows": {
            table.value: int(len(normalized.get(table, pd.DataFrame())))
            for table in ContextRecordTable
        },
        "unique_context_states": len(
            _ids(
                normalized.get(ContextRecordTable.CONTEXT_STATE, pd.DataFrame()),
                "context_state_id",
            )
        ),
        "unique_context_captures": len(
            _ids(
                normalized.get(ContextRecordTable.CONTEXT_CAPTURE, pd.DataFrame()),
                "context_capture_id",
            )
        ),
        "candidate_links": len(
            _ids(
                normalized.get(ContextRecordTable.CANDIDATE_CONTEXT_LINK, pd.DataFrame()),
                "candidate_id",
            )
        ),
        "decision_links": len(
            _ids(
                normalized.get(ContextRecordTable.DECISION_CONTEXT_LINK, pd.DataFrame()),
                "decision_id",
            )
        ),
        "trade_links": len(
            _ids(normalized.get(ContextRecordTable.TRADE_CONTEXT_LINK, pd.DataFrame()), "trade_id")
        ),
    }
