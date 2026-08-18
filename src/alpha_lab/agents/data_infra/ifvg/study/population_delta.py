"""Exact population set relationships with a mandatory match basis (DT §4.3).

Same-profile comparisons use ``native_id_exact`` (v2 primary keys).
Cross-profile comparisons use ``profile_independent_lineage_exact`` ONLY.
Where an exact key cannot be constructed (lineage incomplete, collisions, or
the changed axes alter gap detection / entry-trigger meaning), the population
is DISABLED as ``not_comparable`` with a reason — commonality is never
inferred. No nearest-time, row-order, fuzzy-geometry, or keep-last matching
exists anywhere in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pandas as pd

from ..contracts import RecordTable
from ..search.identities import FrozenContract
from ..search.lineage import NativeLineageMap

__all__ = [
    "FirstDivergence",
    "PopulationDeltaReport",
    "build_native_population_delta",
    "build_lineage_population_delta",
    "DIVERGENCE_REASONS",
]

#: Registered divergence-reason vocabulary (DT §4.3); anything unmappable is
#: ``unattributed`` — never guessed.
DIVERGENCE_REASONS: tuple[str, ...] = (
    "earlier_stale_parent_expiry",
    "different_htf_winner",
    "different_parent_selection",
    "different_fill_invalidation",
    "different_session_gate",
    "slot_became_free",
    "slot_remained_occupied",
    "unattributed",
)

_NATIVE_KEY_BY_KIND: Mapping[str, tuple[RecordTable, str]] = {
    "setup": (RecordTable.SETUP_LIFECYCLE, "envelope_setup_id"),
    "candidate": (RecordTable.ENTRY_CANDIDATE, "candidate_id"),
    "decision": (RecordTable.ELIGIBLE_DECISION, "decision_id"),
    "trade": (RecordTable.EXECUTED_TRADE, "trade_id"),
}

_TERMINAL_REASON_MAP: Mapping[str, str] = {
    "expired_parent_search": "earlier_stale_parent_expiry",
    "invalidated_parent_structural": "different_parent_selection",
    "invalidated_parent_filled": "different_fill_invalidation",
    "invalidated_htf_filled": "different_fill_invalidation",
    "session_gate_reset": "different_session_gate",
}


class FirstDivergence(FrozenContract):
    trading_day: str
    entity_key: str
    side: Literal["added", "removed", "reordered"]
    baseline_cursor: str | None
    challenger_cursor: str | None
    divergence_reason: str | None


class PopulationDeltaReport(FrozenContract):
    entity_kind: Literal["setup", "candidate", "decision", "trade"]
    match_basis: Literal[
        "native_id_exact",
        "profile_independent_lineage_exact",
        "not_comparable",
    ]
    match_basis_reason: str | None
    common_keys: tuple[str, ...]
    added_keys: tuple[str, ...]
    removed_keys: tuple[str, ...]
    jaccard: float | None
    first_divergence: FirstDivergence | None


def _native_keys(tables: Mapping[RecordTable, pd.DataFrame], kind: str) -> frozenset[str]:
    table, column = _NATIVE_KEY_BY_KIND[kind]
    frame = tables.get(table, pd.DataFrame())
    if frame.empty or column not in frame:
        return frozenset()
    return frozenset(frame[column].dropna().astype(str))


def _sets_report(
    kind: str,
    basis: str,
    baseline_keys: frozenset[str],
    challenger_keys: frozenset[str],
    divergence: FirstDivergence | None,
) -> PopulationDeltaReport:
    common = tuple(sorted(baseline_keys & challenger_keys))
    added = tuple(sorted(challenger_keys - baseline_keys))
    removed = tuple(sorted(baseline_keys - challenger_keys))
    union = len(baseline_keys | challenger_keys)
    return PopulationDeltaReport(
        entity_kind=kind,  # type: ignore[arg-type]
        match_basis=basis,  # type: ignore[arg-type]
        match_basis_reason=None,
        common_keys=common,
        added_keys=added,
        removed_keys=removed,
        jaccard=(len(common) / union) if union else 1.0,
        first_divergence=divergence,
    )


def _not_comparable(kind: str, reason: str) -> PopulationDeltaReport:
    return PopulationDeltaReport(
        entity_kind=kind,  # type: ignore[arg-type]
        match_basis="not_comparable",
        match_basis_reason=reason,
        common_keys=(),
        added_keys=(),
        removed_keys=(),
        jaccard=None,
        first_divergence=None,
    )


def _first_divergence_native(
    kind: str,
    baseline_tables: Mapping[RecordTable, pd.DataFrame],
    challenger_tables: Mapping[RecordTable, pd.DataFrame],
    added: frozenset[str],
    removed: frozenset[str],
) -> FirstDivergence | None:
    """Earliest (trading_day, cursor) divergent entity, reason only when a
    lifecycle terminal reason maps 1:1 — otherwise ``unattributed``."""

    table, column = _NATIVE_KEY_BY_KIND[kind]

    def _rows(tables, keys, side):
        frame = tables.get(table, pd.DataFrame())
        if frame.empty or column not in frame:
            return []
        subset = frame.loc[frame[column].astype(str).isin(keys)]
        out = []
        for row in subset.to_dict("records"):
            day = str(row.get("envelope_trading_day"))
            cursor = row.get("event_cursor") or row.get("envelope_ts_utc")
            cursor_str = None if cursor is None or pd.isna(cursor) else str(cursor)
            out.append((day, cursor_str or "", str(row[column]), side, row))
        return out

    entries = _rows(challenger_tables, added, "added") + _rows(
        baseline_tables, removed, "removed"
    )
    if not entries:
        return None
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    day, cursor, key, side, row = entries[0]
    reason = None
    terminal = row.get("reason")
    if terminal is not None and not pd.isna(terminal):
        reason = _TERMINAL_REASON_MAP.get(str(terminal), "unattributed")
    else:
        reason = "unattributed"
    return FirstDivergence(
        trading_day=day,
        entity_key=key,
        side=side,  # type: ignore[arg-type]
        baseline_cursor=cursor if side == "removed" else None,
        challenger_cursor=cursor if side == "added" else None,
        divergence_reason=reason,
    )


def _distinct_section_hashes(
    tables: Mapping[RecordTable, pd.DataFrame],
) -> frozenset[str]:
    hashes: set[str] = set()
    for frame in tables.values():
        if frame is None or frame.empty:
            continue
        if "envelope_section_config_hash" in frame.columns:
            hashes |= {
                value
                for value in frame["envelope_section_config_hash"]
                .dropna()
                .astype(str)
                if value
            }
    return frozenset(hashes)


def build_native_population_delta(
    kind: str,
    baseline_tables: Mapping[RecordTable, pd.DataFrame],
    challenger_tables: Mapping[RecordTable, pd.DataFrame],
) -> PopulationDeltaReport:
    """Same-profile / same-config population delta on native v2 primary keys.

    Native ids embed the profile hash, so this basis is SAME-PROFILE ONLY:
    when both sides carry section hashes and they differ, the comparison is
    disabled (`not_comparable`) instead of fabricating a lawful-looking
    everything-added/removed report — cross-profile deltas use the lineage
    layer exclusively (DT §4.3).
    """

    baseline_hashes = _distinct_section_hashes(baseline_tables)
    challenger_hashes = _distinct_section_hashes(challenger_tables)
    if baseline_hashes and challenger_hashes and baseline_hashes != challenger_hashes:
        return _not_comparable(
            kind,
            "different profile section hashes on the two sides — the native "
            "basis is same-profile only; use the profile-independent lineage "
            "layer",
        )

    baseline_keys = _native_keys(baseline_tables, kind)
    challenger_keys = _native_keys(challenger_tables, kind)
    divergence = _first_divergence_native(
        kind,
        baseline_tables,
        challenger_tables,
        challenger_keys - baseline_keys,
        baseline_keys - challenger_keys,
    )
    return _sets_report(kind, "native_id_exact", baseline_keys, challenger_keys, divergence)


def build_lineage_population_delta(
    kind: str,
    baseline_map: NativeLineageMap,
    challenger_map: NativeLineageMap,
    *,
    lineage_valid: bool,
    lineage_invalid_reason: str | None = None,
) -> PopulationDeltaReport:
    """Cross-profile population delta on the lineage layer ONLY.

    ``lineage_valid`` is the caller's verdict from the changed-axis set (an
    axis that alters gap detection or entry-trigger meaning invalidates the
    lineage keys); False disables the comparison rather than fuzzing it.
    """

    if not lineage_valid:
        return _not_comparable(
            kind,
            lineage_invalid_reason
            or "changed axes alter the lineage key derivation (gap detection / "
            "entry-trigger meaning)",
        )
    for side, lineage_map in (("baseline", baseline_map), ("challenger", challenger_map)):
        if kind in lineage_map.incomplete_kinds:
            return _not_comparable(
                kind, f"{side} lineage incomplete: {lineage_map.incomplete_kinds[kind]}"
            )
        if lineage_map.uniqueness_report.per_entity_kind.get(kind) != "one_to_one":
            return _not_comparable(
                kind,
                f"{side} native→lineage mapping has collisions "
                "(recorded in its LineageUniquenessReport; persist it with "
                "persist_lineage_uniqueness before surfacing this comparison)",
            )
    return _sets_report(
        kind,
        "profile_independent_lineage_exact",
        baseline_map.lineage_keys(kind),
        challenger_map.lineage_keys(kind),
        None,
    )
