"""IFVG v2 dataset/report contracts and fail-closed validation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import pandas as pd

__all__ = [
    "IFVG_CAPTURE_SCHEMA_VERSION",
    "IFVG_DATASET_SCHEMA_VERSION",
    "IFVG_REPORT_SCHEMA_VERSION",
    "RecordTable",
    "KIND_TO_TABLE",
    "partition_capture_tables",
    "stamp_table_contract",
    "validate_primary_keys",
    "validate_table_identity",
    "validate_foreign_keys",
    "count_reconciliation",
]

IFVG_CAPTURE_SCHEMA_VERSION = 2
IFVG_DATASET_SCHEMA_VERSION = 2
IFVG_REPORT_SCHEMA_VERSION = 2


class RecordTable(StrEnum):
    SETUP_LIFECYCLE = "setup_lifecycle_event"
    ENTRY_CANDIDATE = "entry_candidate"
    CANDIDATE_LABEL = "candidate_label"
    ELIGIBLE_DECISION = "eligible_decision"
    EXECUTED_TRADE = "executed_trade"
    GEOMETRY_DOSSIER = "geometry_dossier"
    QUARANTINE = "quarantine"


KIND_TO_TABLE: dict[str, RecordTable] = {table.value: table for table in RecordTable}

_PRIMARY_KEYS: dict[RecordTable, str] = {
    RecordTable.SETUP_LIFECYCLE: "lifecycle_event_id",
    RecordTable.ENTRY_CANDIDATE: "candidate_id",
    RecordTable.CANDIDATE_LABEL: "candidate_label_id",
    RecordTable.ELIGIBLE_DECISION: "decision_id",
    RecordTable.EXECUTED_TRADE: "trade_id",
    RecordTable.GEOMETRY_DOSSIER: "candidate_id",
    RecordTable.QUARANTINE: "quarantine_id",
}

_VERSION_COLUMNS: dict[RecordTable, str] = {
    RecordTable.SETUP_LIFECYCLE: "lifecycle_schema_version",
    RecordTable.ENTRY_CANDIDATE: "candidate_schema_version",
    RecordTable.CANDIDATE_LABEL: "candidate_label_schema_version",
    RecordTable.ELIGIBLE_DECISION: "decision_schema_version",
    RecordTable.EXECUTED_TRADE: "trade_schema_version",
    RecordTable.GEOMETRY_DOSSIER: "geometry_schema_version",
    RecordTable.QUARANTINE: "quarantine_schema_version",
}

_ENVELOPE_IDENTITY_COLUMNS = (
    "envelope_strategy_id",
    "envelope_strategy_version",
    "envelope_profile_hash",
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

_EMPTY_TABLE_COLUMNS: dict[RecordTable, tuple[str, ...]] = {
    RecordTable.SETUP_LIFECYCLE: (
        "lifecycle_event_id",
        "from_phase",
        "to_phase",
        "transition",
        "reason",
        "event_cursor",
    ),
    RecordTable.ENTRY_CANDIDATE: (
        "candidate_id",
        "direction",
        "entry_family",
        "trigger_cursor",
        "entry_ticks",
        "proposed_stop_ticks",
        "risk_ticks",
        "proposed_target_ticks",
        "block_reasons",
    ),
    RecordTable.CANDIDATE_LABEL: (
        "candidate_label_id",
        "candidate_id",
        "label_family",
        "r_multiple",
        "label",
        "bars_after_entry_to_resolution",
        "mfe_r",
        "mae_r",
        "censored",
        "section_config_hash",
        "evaluation_config_hash",
        "qualification_mode",
        "entry_family",
        "anchor_policy",
        "resolver_policy",
    ),
    RecordTable.ELIGIBLE_DECISION: (
        "decision_id",
        "candidate_id",
        "direction",
        "entry_cursor",
        "entry_ticks",
        "stop_ticks",
        "risk_ticks",
        "target_ticks",
        "passed_guards",
    ),
    RecordTable.EXECUTED_TRADE: (
        "trade_id",
        "decision_id",
        "candidate_id",
        "direction",
        "status",
        "resolution",
        "entry_cursor",
        "resolution_cursor",
        "entry_ts_utc",
        "resolution_ts_utc",
        "entry_ticks",
        "stop_ticks",
        "target_ticks",
        "risk_ticks",
        "bars_after_entry_to_resolution",
        "mfe_ticks",
        "mae_ticks",
        "realized_ticks",
    ),
    RecordTable.GEOMETRY_DOSSIER: (
        "candidate_id",
        "decision_id",
        "trade_id",
    ),
    RecordTable.QUARANTINE: (
        "quarantine_id",
        "candidate_id",
        "reasons",
        "evidence_cursor",
    ),
}


def _coerce_table(table: RecordTable | str) -> RecordTable:
    return table if isinstance(table, RecordTable) else RecordTable(table)


def stamp_table_contract(
    table: RecordTable | str,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Stamp the common v2 table/schema columns without changing evidence."""
    table = _coerce_table(table)
    out = frame.copy()
    out["record_table"] = table.value
    if "record_schema_version" not in out:
        if "envelope_schema_version" in out:
            out["record_schema_version"] = out["envelope_schema_version"]
        else:
            out["record_schema_version"] = 2
    out["capture_schema_version"] = IFVG_CAPTURE_SCHEMA_VERSION
    out["dataset_schema_version"] = IFVG_DATASET_SCHEMA_VERSION
    version_column = _VERSION_COLUMNS[table]
    if version_column not in out:
        out[version_column] = out["record_schema_version"]
    aliases = {
        "setup_id": "envelope_setup_id",
        "trading_day": "envelope_trading_day",
        "strategy_id": "envelope_strategy_id",
        "strategy_version": "envelope_strategy_version",
        "profile_hash": "envelope_profile_hash",
        "profile_name": "envelope_profile_name",
        "qualification_mode": "envelope_qualification_mode",
        "section_config_hash": "envelope_section_config_hash",
        "entry_family": "envelope_entry_family",
        "label_family": "envelope_label_family",
        "entry_session": "envelope_entry_session",
        "anchor_policy": "envelope_anchor_policy",
        "resolver_policy": "envelope_resolver_policy",
        "causality_parent": "envelope_causality_parent",
        "causality_opposing": "envelope_causality_opposing",
        "causality_entry": "envelope_causality_entry",
        "timeout_policy": "envelope_timeout_policy",
    }
    for target, source in aliases.items():
        if target not in out and source in out:
            out[target] = out[source]
    if out.empty:
        empty_columns = {
            *_EMPTY_TABLE_COLUMNS[table],
            *_ENVELOPE_IDENTITY_COLUMNS,
            "envelope_setup_id",
            *aliases,
        }
        for column in sorted(empty_columns):
            if column not in out:
                out[column] = pd.Series(dtype="object")
    return out


def validate_primary_keys(table: RecordTable | str, frame: pd.DataFrame) -> None:
    """Require a non-null, unique primary key for a materialized v2 table."""
    table = _coerce_table(table)
    if frame.empty:
        return
    key = _PRIMARY_KEYS[table]
    if key not in frame.columns:
        raise ValueError(f"{table.value} table is missing primary key {key!r}")
    if frame[key].isna().any() or (frame[key].astype(str) == "").any():
        raise ValueError(f"{table.value}.{key} contains null/empty values")
    duplicates = frame[key].duplicated(keep=False)
    if duplicates.any():
        values = sorted(frame.loc[duplicates, key].astype(str).unique())
        raise ValueError(f"{table.value}.{key} is not unique: {values[:5]}")


def validate_table_identity(table: RecordTable | str, frame: pd.DataFrame) -> None:
    """Require every non-empty row to pin its resolved v2 behavior identity."""
    table = _coerce_table(table)
    if frame.empty:
        return
    required = {
        "record_table",
        "record_schema_version",
        "capture_schema_version",
        "dataset_schema_version",
        _VERSION_COLUMNS[table],
    }
    if table is not RecordTable.CANDIDATE_LABEL:
        required.update(_ENVELOPE_IDENTITY_COLUMNS)
        required.add("envelope_setup_id")
    else:
        required.update(
            {
                "candidate_id",
                "label_family",
                "section_config_hash",
                "evaluation_config_hash",
                "qualification_mode",
                "entry_family",
                "anchor_policy",
                "resolver_policy",
            }
        )
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{table.value} table is missing identity columns {missing}")
    if not (frame["record_table"].astype(str) == table.value).all():
        raise ValueError(f"{table.value} table contains another record_table")
    for version_column in (
        "record_schema_version",
        "capture_schema_version",
        "dataset_schema_version",
        _VERSION_COLUMNS[table],
    ):
        values = pd.to_numeric(frame[version_column], errors="coerce")
        if values.isna().any() or not (values == 2).all():
            raise ValueError(f"{table.value}.{version_column} must be exactly 2")
    identity_columns = (
        _ENVELOPE_IDENTITY_COLUMNS
        if table is not RecordTable.CANDIDATE_LABEL
        else (
            "section_config_hash",
            "evaluation_config_hash",
            "qualification_mode",
            "entry_family",
            "label_family",
            "anchor_policy",
            "resolver_policy",
        )
    )
    for column in identity_columns:
        values = frame[column]
        if values.isna().any() or (values.astype(str) == "").any():
            raise ValueError(f"{table.value}.{column} contains null/empty values")


def partition_capture_tables(
    capture: pd.DataFrame,
) -> dict[RecordTable, pd.DataFrame]:
    """Split a flattened emission trace into typed v2 tables.

    Context emissions and the read-only ``setup_resolution`` compatibility
    stream remain in replay diagnostics; they cannot enter an executable table.
    """
    result: dict[RecordTable, pd.DataFrame] = {}
    for table in RecordTable:
        frame = (
            capture.loc[capture["kind"] == table.value].copy()
            if not capture.empty and "kind" in capture.columns
            else pd.DataFrame()
        )
        if not frame.empty:
            frame = frame.drop(columns=["kind"], errors="ignore")
        frame = stamp_table_contract(table, frame)
        validate_primary_keys(table, frame)
        if table is not RecordTable.CANDIDATE_LABEL:
            validate_table_identity(table, frame)
        result[table] = frame.reset_index(drop=True)
    return result


def _ids(frame: pd.DataFrame, column: str) -> set[str]:
    if frame.empty or column not in frame:
        return set()
    return set(frame[column].dropna().astype(str))


def validate_foreign_keys(
    tables: Mapping[RecordTable | str, pd.DataFrame],
) -> None:
    """Validate candidate → decision → execution evidence relationships."""
    normalized = {_coerce_table(key): value for key, value in tables.items()}
    candidates = normalized.get(RecordTable.ENTRY_CANDIDATE, pd.DataFrame())
    labels = normalized.get(RecordTable.CANDIDATE_LABEL, pd.DataFrame())
    decisions = normalized.get(RecordTable.ELIGIBLE_DECISION, pd.DataFrame())
    trades = normalized.get(RecordTable.EXECUTED_TRADE, pd.DataFrame())
    geometry = normalized.get(RecordTable.GEOMETRY_DOSSIER, pd.DataFrame())
    quarantine = normalized.get(RecordTable.QUARANTINE, pd.DataFrame())

    candidate_ids = _ids(candidates, "candidate_id")
    decision_ids = _ids(decisions, "decision_id")

    for table, frame, fk, parents in (
        (RecordTable.CANDIDATE_LABEL, labels, "candidate_id", candidate_ids),
        (RecordTable.ELIGIBLE_DECISION, decisions, "candidate_id", candidate_ids),
        (RecordTable.EXECUTED_TRADE, trades, "candidate_id", candidate_ids),
        (RecordTable.GEOMETRY_DOSSIER, geometry, "candidate_id", candidate_ids),
        (RecordTable.QUARANTINE, quarantine, "candidate_id", candidate_ids),
    ):
        unknown = _ids(frame, fk) - parents
        if unknown:
            raise ValueError(
                f"{table.value}.{fk} has missing parents: {sorted(unknown)[:5]}"
            )
    unknown_decisions = _ids(trades, "decision_id") - decision_ids
    if unknown_decisions:
        raise ValueError(
            "executed_trade.decision_id has missing parents: "
            f"{sorted(unknown_decisions)[:5]}"
        )

    if not trades.empty:
        trade_geometry = (
            geometry.dropna(subset=["trade_id"])
            if "trade_id" in geometry
            else geometry
        )
        geometry_trade_ids = _ids(trade_geometry, "trade_id")
        missing_geometry = _ids(trades, "trade_id") - geometry_trade_ids
        if missing_geometry:
            raise ValueError(
                "executed trades are missing immutable geometry: "
                f"{sorted(missing_geometry)[:5]}"
            )

    quarantined = _ids(quarantine, "candidate_id")
    if _ids(decisions, "candidate_id") & quarantined:
        raise ValueError("quarantined candidates cannot produce eligible decisions")
    if _ids(trades, "candidate_id") & quarantined:
        raise ValueError("quarantined candidates cannot produce executed trades")


def count_reconciliation(
    tables: Mapping[RecordTable | str, pd.DataFrame],
) -> dict:
    """JSON-safe table counts and foreign-key reconciliation totals."""
    normalized = {_coerce_table(key): value for key, value in tables.items()}
    counts = {
        table.value: int(len(normalized.get(table, pd.DataFrame())))
        for table in RecordTable
    }
    candidates = normalized.get(RecordTable.ENTRY_CANDIDATE, pd.DataFrame())
    decisions = normalized.get(RecordTable.ELIGIBLE_DECISION, pd.DataFrame())
    trades = normalized.get(RecordTable.EXECUTED_TRADE, pd.DataFrame())
    geometry = normalized.get(RecordTable.GEOMETRY_DOSSIER, pd.DataFrame())
    return {
        "schema_version": IFVG_DATASET_SCHEMA_VERSION,
        "table_rows": counts,
        "unique_candidate_ids": len(_ids(candidates, "candidate_id")),
        "decision_candidate_fk_matches": int(
            decisions["candidate_id"].astype(str).isin(
                _ids(candidates, "candidate_id")
            ).sum()
        )
        if not decisions.empty
        else 0,
        "trade_decision_fk_matches": int(
            trades["decision_id"].astype(str).isin(
                _ids(decisions, "decision_id")
            ).sum()
        )
        if not trades.empty
        else 0,
        "trade_geometry_fk_matches": int(
            trades["trade_id"].astype(str).isin(_ids(geometry, "trade_id")).sum()
        )
        if not trades.empty and "trade_id" in geometry
        else 0,
    }
