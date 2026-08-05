"""Exact parity gate: regenerated v2 stream vs the accepted final-review v2.

The FSM audit artifact may only exist if the audit-enabled replay reproduces
the accepted v2 dataset EXACTLY — row counts, ordered canonical row hashes,
primary/foreign key sets, per-table content hashes. Any difference (including
a serialization-only difference — treated as failure unless separately proven
non-semantic and ratified) fails the task.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .contracts import RecordTable
from .dataset import _canonical_cell, _table_content_hash

__all__ = [
    "EXPECTED_FUNNEL",
    "compare_v2_exact",
    "write_parity_report",
]

#: Accepted-artifact expectations (final review, IFVG_LAB_FINAL_VERIFICATION):
#: 215 setup activations / 132 entry candidates / 33 executed trades /
#: 30 post-warmup resolved trades.
EXPECTED_FUNNEL = {
    "setup_activations": 215,
    "entry_candidates": 132,
    "executed_trades": 33,
    "post_warmup_resolved_trades": 30,
}

_PK_BY_TABLE = {
    RecordTable.SETUP_LIFECYCLE: "lifecycle_event_id",
    RecordTable.ENTRY_CANDIDATE: "candidate_id",
    RecordTable.CANDIDATE_LABEL: "candidate_label_id",
    RecordTable.ELIGIBLE_DECISION: "decision_id",
    RecordTable.EXECUTED_TRADE: "trade_id",
    RecordTable.GEOMETRY_DOSSIER: "candidate_id",
    RecordTable.QUARANTINE: "quarantine_id",
}


def _row_hashes(table: RecordTable, frame: pd.DataFrame) -> list[str]:
    from .manifest import canonical_sha256

    columns = sorted(frame.columns)
    ordered = frame.loc[:, columns]
    key = _PK_BY_TABLE[table]
    if key in ordered and not ordered.empty:
        ordered = ordered.sort_values(key, kind="mergesort")
    return [
        canonical_sha256({column: _canonical_cell(value) for column, value in row.items()})
        for row in ordered.to_dict("records")
    ]


def _ids(frame: pd.DataFrame, column: str) -> set[str]:
    if frame.empty or column not in frame:
        return set()
    return set(frame[column].dropna().astype(str))


def _observed_funnel(tables: dict[RecordTable, pd.DataFrame]) -> dict[str, int]:
    lifecycle = tables.get(RecordTable.SETUP_LIFECYCLE, pd.DataFrame())
    candidates = tables.get(RecordTable.ENTRY_CANDIDATE, pd.DataFrame())
    trades = tables.get(RecordTable.EXECUTED_TRADE, pd.DataFrame())
    activations = (
        int((lifecycle["transition"] == "setup_activated").sum())
        if not lifecycle.empty
        else 0
    )
    resolved_post_warmup = 0
    if not trades.empty:
        resolved = trades["status"].astype(str) == "resolved"
        warm = trades["is_warmup"].astype(bool) if "is_warmup" in trades else False
        resolved_post_warmup = int((resolved & ~warm).sum())
    return {
        "setup_activations": activations,
        "entry_candidates": int(len(candidates)),
        "executed_trades": int(len(trades)),
        "post_warmup_resolved_trades": resolved_post_warmup,
    }


def compare_v2_exact(
    derived: dict[RecordTable, pd.DataFrame],
    accepted: dict[RecordTable, pd.DataFrame],
) -> dict:
    """Full exact-identity comparison; ``passed`` is True only if EVERY table
    matches on count, content hash, ordered row hashes, and key sets, AND the
    expected funnel numbers hold."""
    tables_report: dict[str, dict] = {}
    for table in RecordTable:
        derived_frame = derived.get(table, pd.DataFrame())
        accepted_frame = accepted.get(table, pd.DataFrame())
        derived_hash = _table_content_hash(table, derived_frame)
        accepted_hash = _table_content_hash(table, accepted_frame)
        entry: dict = {
            "derived_rows": int(len(derived_frame)),
            "accepted_rows": int(len(accepted_frame)),
            "derived_content_sha256": derived_hash,
            "accepted_content_sha256": accepted_hash,
            "matched": derived_hash == accepted_hash,
        }
        key = _PK_BY_TABLE[table]
        derived_keys = _ids(derived_frame, key)
        accepted_keys = _ids(accepted_frame, key)
        entry["pk_only_in_derived"] = len(derived_keys - accepted_keys)
        entry["pk_only_in_accepted"] = len(accepted_keys - derived_keys)
        if not entry["matched"]:
            derived_rows = _row_hashes(table, derived_frame)
            accepted_rows = _row_hashes(table, accepted_frame)
            first_mismatch = None
            for index, (a, b) in enumerate(
                zip(derived_rows, accepted_rows, strict=False)
            ):
                if a != b:
                    first_mismatch = index
                    break
            if first_mismatch is None and len(derived_rows) != len(accepted_rows):
                first_mismatch = min(len(derived_rows), len(accepted_rows))
            entry["first_mismatch_row_index"] = first_mismatch
            entry["row_hash_diff_count"] = sum(
                1
                for a, b in zip(derived_rows, accepted_rows, strict=False)
                if a != b
            ) + abs(len(derived_rows) - len(accepted_rows))
        tables_report[table.value] = entry

    observed = _observed_funnel(derived)
    funnel_report = {
        name: {
            "expected": EXPECTED_FUNNEL[name],
            "observed": observed[name],
            "matched": EXPECTED_FUNNEL[name] == observed[name],
        }
        for name in EXPECTED_FUNNEL
    }
    passed = all(item["matched"] for item in tables_report.values()) and all(
        item["matched"] for item in funnel_report.values()
    )
    return {
        "passed": passed,
        "tables": tables_report,
        "expected_funnel": funnel_report,
    }


def write_parity_report(report: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path
