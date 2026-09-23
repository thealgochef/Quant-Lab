"""B0 from exact selected Core stage emissions, never timestamp approximations.

The audit companion is an immutable partition of the original emission trace.
Its selection and replacement records retain values the entry row does not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from strategy_core.structures.fvg import close_through_margin_ticks, interval_distance_ticks
from strategy_core.types import Side

from .artifact_io import ArtifactVerificationError
from .audit_contracts import AuditTable, audit_contract_fingerprint, validate_audit_table
from .context_experiment_contracts import canonical_contract_sha256

B0_PROJECTION_VERSION = "ifvg_b0_selected_stage_projection_v2"
LEGACY_B0_PROJECTION_VERSION = "ifvg_b0_partial_entry_projection_v1"
STAGE_FEATURES = {
    AuditTable.PARENT_CANDIDATE: (
        "parent_tf_seconds", "distance_to_htf_ticks", "elapsed_parent_bars_since_tap",
        "elapsed_1m_bars_since_tap",
    ),
    AuditTable.PARENT_LOCK: ("elapsed_1m_bars_since_selection",),
    AuditTable.OPPOSING: ("distance_to_parent_ticks", "elapsed_1m_bars_since_lock"),
    AuditTable.INVERSION: (
        "close_through_margin_ticks", "bars_armed_to_inversion", "opposing_size_ticks",
    ),
}
STRUCTURAL_ENTRY_FEATURES = (
    "entry_fvg_size_ticks", "entry_fvg_gap_low_ticks", "entry_fvg_gap_high_ticks",
)
_SOURCE_TABLES = (*STAGE_FEATURES, AuditTable.PARENT_WINDOW)


@dataclass(frozen=True)
class B0ProjectionSource:
    reference: dict[str, Any]
    tables: dict[AuditTable, pd.DataFrame]


def load_b0_projection_source(store_root: Path, core_replay_id: str) -> B0ProjectionSource:
    """Resolve the current supported audit contract by exact identity, verify bytes."""
    from .fsm_audit_preparation import ChildFsmAuditEnvelope
    from .search.identities import FsmAuditArtifactIdentity
    from .search.store import load_sidecar_bytes, load_verified_envelope

    expected = ChildFsmAuditEnvelope.from_payload(FsmAuditArtifactIdentity(
        core_replay_id=core_replay_id,
        audit_schema_version=1,
        audit_contract_fingerprint=canonical_contract_sha256(audit_contract_fingerprint()),
        neutrality_mechanism_id="dual_drive_ab_v1",
    ))
    audit_id = expected.child_fsm_audit_id
    envelope = load_verified_envelope(
        store_root, "fsm_audit_companions", audit_id, ChildFsmAuditEnvelope
    )
    if envelope != expected:
        raise ArtifactVerificationError("B0 audit companion belongs to another Core source")
    core_ref = json.loads(load_sidecar_bytes(
        store_root, "core_replays", core_replay_id, "artifact_reference.json"
    ))
    if core_ref["core_replay_id"] != core_replay_id:
        raise ArtifactVerificationError("B0 Core reference belongs to another source")
    reconciliation = json.loads(load_sidecar_bytes(
        store_root, "fsm_audit_companions", audit_id, "reconciliation_report.json"
    ))
    if reconciliation.get("passed") is not True:
        raise ArtifactVerificationError("B0 source audit reconciliation did not pass")
    tables, hashes = {}, {}
    for table in _SOURCE_TABLES:
        raw = load_sidecar_bytes(
            store_root, "fsm_audit_companions", audit_id, f"{table.value}.parquet"
        )
        frame = pd.read_parquet(BytesIO(raw))
        validate_audit_table(table, frame)
        tables[table] = frame
        hashes[table.value] = hashlib.sha256(raw).hexdigest()
    return B0ProjectionSource({
        "projection_version": B0_PROJECTION_VERSION,
        "core_replay_id": core_replay_id,
        "v2_dataset_id": core_ref["v2_dataset_artifact_id"],
        "v2_manifest_payload_sha256": core_ref["manifest_payload_sha256"],
        "child_fsm_audit_id": audit_id,
        "audit_contract_fingerprint": envelope.payload.audit_contract_fingerprint,
        "source_table_sha256": hashes,
    }, tables)


def _required(row: pd.Series, key: str) -> Any:
    if key not in row or pd.isna(row[key]):
        raise ArtifactVerificationError(f"unmapped advertised B0/source field: {key}")
    return row[key]


def _exact(frame: pd.DataFrame, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ArtifactVerificationError(
            f"B0 requires one exact {description}; found {len(frame)}"
        )
    return frame.iloc[0]


def _equal(actual: Any, expected: Any, description: str) -> None:
    if pd.isna(actual) or pd.isna(expected) or actual != expected:
        raise ArtifactVerificationError(f"B0 {description} differs from exact Core evidence")


def _geometry_parity(candidate: pd.Series, row: pd.Series, geometry: str) -> None:
    for name in (
        "fvg_id", "timeframe_seconds", "direction", "gap_low_ticks", "gap_high_ticks",
        "size_ticks", "a_bar_id", "c_bar_id", "a_open_ts_utc", "confirmed_ts_utc",
    ):
        _equal(_required(row, f"fvg_{name}"),
               _required(candidate, f"geometry_{geometry}_{name}"),
               f"selected {geometry} {name}")


def validate_b0_feature_mapping(
    frame: pd.DataFrame, advertised_features: tuple[str, ...]
) -> dict[str, dict[str, int | str]]:
    """Structural absence is explicit; fold-local emptiness is a later ML concern."""
    if "entry_family" not in frame:
        raise ArtifactVerificationError("unmapped advertised B0 field: entry_family")
    result = {}
    for name in advertised_features:
        if name not in frame:
            raise ArtifactVerificationError(f"unmapped advertised B0 field: {name}")
        structural = (
            frame["entry_family"].eq("ifvg_retest")
            if name in STRUCTURAL_ENTRY_FEATURES
            else pd.Series(False, index=frame.index)
        )
        null = frame[name].isna()
        if (null & ~structural).any():
            raise ArtifactVerificationError(
                f"unmapped advertised B0 feature {name}: "
                f"{int((null & ~structural).sum())} non-structural nulls"
            )
        if (structural & ~null).any():
            raise ArtifactVerificationError(f"B0 {name} fabricates an entry FVG for ifvg_retest")
        result[name] = {
            "mapping_status": "mapped",
            "non_null_count": int((~null).sum()),
            "structural_null_count": int((null & structural).sum()),
            "unmapped_count": 0,
        }
    return result


def project_b0_candidates(
    candidates: pd.DataFrame,
    source: B0ProjectionSource,
    *,
    decision_bars: pd.DataFrame,
    advertised_features: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join by setup, selected FVG IDs, exact cursors and original trace ordering.

    Bars are the verified replay's 60s input sequence. Their row ranks measure
    reducer steps (including sparse sessions), not wall-clock minutes. Parent
    clocks come directly from the selected Core record and independent audit
    parent-window state, which count completed own-timeframe bars only in S1.
    """
    if decision_bars.empty and not candidates.empty:
        raise ArtifactVerificationError("B0 ordinal validation requires verified decision bars")
    bars = decision_bars.sort_values(["availability_ts_utc", "bar_id"], kind="stable")
    if bars["bar_id"].duplicated().any():
        raise ArtifactVerificationError("B0 decision bars contain duplicate bar IDs")
    if not bars["timeframe_ticks"].eq(60).all():
        raise ArtifactVerificationError("B0 ordinal source includes non-60s bars")
    ordinal_by_id = {str(value): index for index, value in enumerate(bars["bar_id"])}
    ordinal_by_ts = bars.reset_index(drop=True).groupby("availability_ts_utc").indices

    def ordinal(ts: Any, *, bar_id: Any = None) -> int:
        indices = ordinal_by_ts.get(pd.Timestamp(ts), [])
        if len(indices) != 1:
            raise ArtifactVerificationError("B0 stage does not resolve to one decision-bar ordinal")
        index = int(indices[0])
        if bar_id is not None and ordinal_by_id.get(str(bar_id)) != index:
            raise ArtifactVerificationError("B0 cursor and decision-bar timestamp disagree")
        return index

    result = candidates.copy()
    for names in STAGE_FEATURES.values():
        for name in names:
            result[name] = pd.Series(index=result.index, dtype="float64")
    evidence_rows = []
    for index, candidate in candidates.iterrows():
        setup_id = str(_required(candidate, "envelope_setup_id"))
        candidate_id = str(_required(candidate, "candidate_id"))
        entry_ts = pd.Timestamp(_required(candidate, "envelope_ts_utc"))
        entry_trace = int(_required(candidate, "trace_ordinal"))
        scoped = {
            table: frame.loc[frame["envelope_setup_id"].astype(str).eq(setup_id)]
            for table, frame in source.tables.items()
        }
        locks = scoped[AuditTable.PARENT_LOCK]
        lock = _exact(locks.loc[
            locks["lock_cursor"].eq(_required(candidate, "geometry_lock_bar_cursor"))
            & locks["parent_fvg_id"].eq(_required(candidate, "geometry_parent_fvg_id"))
        ], f"parent lock for {candidate_id}")
        parents = scoped[AuditTable.PARENT_CANDIDATE]
        selected_parents = parents.loc[
            parents["selected"].eq(True) & parents["trace_ordinal"].lt(lock["trace_ordinal"])
        ].sort_values("trace_ordinal")
        parent = _exact(selected_parents.tail(1), f"last selected parent for {candidate_id}")
        _geometry_parity(candidate, parent, "parent")
        inversions = scoped[AuditTable.INVERSION]
        inversion = _exact(inversions.loc[
            inversions["inversion_cursor"].eq(_required(candidate, "geometry_inversion_bar_cursor"))
            & inversions["opposing_fvg_id"].eq(_required(candidate, "geometry_opposing_fvg_id"))
        ], f"inversion for {candidate_id}")
        opposing_rows = scoped[AuditTable.OPPOSING]
        opposing = _exact(opposing_rows.loc[
            opposing_rows["selected"].eq(True)
            & opposing_rows["trace_ordinal"].gt(lock["trace_ordinal"])
            & opposing_rows["trace_ordinal"].lt(inversion["trace_ordinal"])
        ].sort_values("trace_ordinal").tail(1), f"last selected opposing for {candidate_id}")
        _geometry_parity(candidate, opposing, "opposing")
        for selected in (parent, opposing):
            if not bool(_required(selected, "causality_satisfied")):
                raise ArtifactVerificationError("B0 selected FVG failed Core causality")
            if pd.Timestamp(selected["fvg_confirmed_ts_utc"]) > selected["envelope_ts_utc"]:
                raise ArtifactVerificationError(
                    "B0 selected FVG unavailable at stage decision time"
                )
        stages = {
            AuditTable.PARENT_CANDIDATE: parent, AuditTable.PARENT_LOCK: lock,
            AuditTable.OPPOSING: opposing, AuditTable.INVERSION: inversion,
        }
        previous_trace = -1
        for table, row in stages.items():
            ts = pd.Timestamp(_required(row, "envelope_ts_utc"))
            trace = int(_required(row, "trace_ordinal"))
            if ts.tzinfo is None or ts > entry_ts or not previous_trace < trace < entry_trace:
                raise ArtifactVerificationError(
                    "B0 selected-stage evidence unavailable at decision time"
                )
            previous_trace = trace
            _equal(row["envelope_section_config_hash"],
                   candidate["envelope_section_config_hash"], "source configuration")
            for name in STAGE_FEATURES[table]:
                value = _required(row, name)
                if value < 0 or int(value) != value:
                    raise ArtifactVerificationError(f"B0 stage {name} is not a nonnegative integer")
                result.at[index, name] = value

        def geometry(name: str, selected_candidate: pd.Series = candidate) -> Any:
            return _required(selected_candidate, f"geometry_{name}")

        checks = {
            "parent_tf_seconds": geometry("parent_timeframe_seconds"),
            "distance_to_htf_ticks": interval_distance_ticks(
                geometry("parent_gap_low_ticks"), geometry("parent_gap_high_ticks"),
                geometry("htf_gap_low_ticks"), geometry("htf_gap_high_ticks")),
            "distance_to_parent_ticks": interval_distance_ticks(
                geometry("opposing_gap_low_ticks"), geometry("opposing_gap_high_ticks"),
                geometry("parent_gap_low_ticks"), geometry("parent_gap_high_ticks")),
            "opposing_size_ticks": geometry("opposing_size_ticks"),
        }
        direction = str(_required(candidate, "direction"))
        if direction not in ("LONG", "SHORT"):
            raise ArtifactVerificationError("B0 direction is unknown")
        _equal(geometry("opposing_direction"),
               "bearish" if direction == "LONG" else "bullish", "opposing direction")
        checks["close_through_margin_ticks"] = close_through_margin_ticks(
            SimpleNamespace(close_ticks=geometry("inversion_bar_close_ticks")),
            boundary_ticks=geometry("opposing_gap_high_ticks" if direction == "LONG"
                                    else "opposing_gap_low_ticks"),
            beyond=Side.HIGH if direction == "LONG" else Side.LOW,
        )
        windows = scoped[AuditTable.PARENT_WINDOW]
        selected_window = _exact(windows.loc[
            windows["event_kind"].eq("parent_selected")
            & windows["parent_fvg_id"].eq(geometry("parent_fvg_id"))
            & windows["core_trace_ordinal_before_global"].eq(parent["trace_ordinal"])
        ], f"selected parent clock event for {candidate_id}")
        opened = _exact(windows.loc[windows["event_kind"].eq("opened")],
                        f"tap/opened clock event for {candidate_id}")
        _equal(selected_window["envelope_ts_utc"], parent["envelope_ts_utc"],
               "selection clock time")
        _equal(opened["event_cursor"], geometry("tap_bar_cursor"), "tap clock cursor")
        _equal(opened["envelope_ts_utc"], geometry("tap_bar_logical_close_ts_utc"),
               "tap clock time")
        if geometry("htf_confirmed_ts_utc") > geometry("tap_bar_logical_close_ts_utc"):
            raise ArtifactVerificationError("B0 HTF geometry unavailable at tap time")
        if (str(candidate["entry_family"]) == "fresh_fvg_continuation"
                and _required(candidate, "entry_fvg_confirmed_ts_utc") > entry_ts):
            raise ArtifactVerificationError("B0 continuation FVG unavailable at decision time")
        checks["elapsed_parent_bars_since_tap"] = dict(json.loads(
            _required(selected_window, "parent_clocks")
        ))[int(parent["parent_tf_seconds"])]
        tap_ordinal = ordinal(geometry("tap_bar_logical_close_ts_utc"),
                              bar_id=geometry("tap_bar_bar_id"))
        parent_ordinal = ordinal(parent["envelope_ts_utc"])
        lock_ordinal = ordinal(lock["envelope_ts_utc"], bar_id=geometry("lock_bar_bar_id"))
        armed_ordinal = ordinal(opposing["envelope_ts_utc"])
        inversion_ordinal = ordinal(inversion["envelope_ts_utc"],
                                    bar_id=geometry("inversion_bar_bar_id"))
        for stage, bar_index in (("tap", tap_ordinal), ("lock", lock_ordinal),
                                 ("inversion", inversion_ordinal)):
            source_bar = bars.iloc[bar_index]
            for component in ("open", "high", "low", "close"):
                _equal(source_bar[f"{component}_ticks"], geometry(f"{stage}_bar_{component}_ticks"),
                       f"{stage} decision bar {component}")
        checks.update({
            "elapsed_1m_bars_since_tap": parent_ordinal - tap_ordinal,
            "elapsed_1m_bars_since_selection": lock_ordinal - parent_ordinal,
            "elapsed_1m_bars_since_lock": armed_ordinal - lock_ordinal,
            "bars_armed_to_inversion": inversion_ordinal - armed_ordinal,
        })
        _equal(checks["elapsed_1m_bars_since_tap"],
               selected_window["stamp_source_step_ordinal"] - opened["stamp_source_step_ordinal"],
               "tap-to-selection reducer ordinal")
        for name, expected in checks.items():
            _equal(result.at[index, name], expected, name)
        evidence_rows.append({
            "candidate_id": candidate_id, "setup_id": setup_id,
            "entry_ts_utc": entry_ts.isoformat(), "entry_trace_ordinal": entry_trace,
            "parent_window_event_id": selected_window["event_id"],
            "parent_clock": checks["elapsed_parent_bars_since_tap"],
            "selected_parent_fvg_id": geometry("parent_fvg_id"),
            "selected_opposing_fvg_id": geometry("opposing_fvg_id"),
            "stage_trace_ordinals": {table.value: int(row["trace_ordinal"])
                                     for table, row in stages.items()},
            "stage_available_ts_utc": {table.value: pd.Timestamp(row["envelope_ts_utc"]).isoformat()
                                       for table, row in stages.items()},
            "bar_ordinals": dict(tap=tap_ordinal, parent_selection=parent_ordinal,
                                 lock=lock_ordinal, armed=armed_ordinal,
                                 inversion=inversion_ordinal),
            "all_formula_clock_ordinal_checks_passed": True,
            "projected_values": {name: int(value) for name, value in checks.items()},
        })
    coverage = validate_b0_feature_mapping(result, advertised_features)
    evidence = {
        "projection_version": B0_PROJECTION_VERSION, "source": source.reference,
        "candidate_count": len(result), "feature_mapping": coverage,
        "decision_bar_sequence_hash": hashlib.sha256(pd.util.hash_pandas_object(
            bars[["bar_id", "availability_ts_utc"]], index=False,
        ).values.tobytes()).hexdigest(),
        "candidate_evidence": evidence_rows,
    }
    evidence["projection_evidence_hash"] = canonical_contract_sha256(evidence)
    return result, evidence
