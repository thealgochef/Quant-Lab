"""Setup-level verifier provider: open by ``VerifierBundleRef``, list all
setups, and serve stage-gated audit evidence for any setup — including the
candidate-less ones the candidate verifier cannot display.

Evidence ordering is the audit channel's hard cross-channel contract:
``source_step_ordinal → reducer_substep → reducer_substep_ordinal →
audit_seq``; trace-reused rows merge in by their global ``trace_ordinal``
against the channel rows' global brackets. Timestamps are a display fallback
only; a same-minute ordering ambiguity is a contract failure and is asserted
here at load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .audit_contracts import AuditTable
from .config import FSM_AUDIT_DATASET_DIR
from .fsm_audit_io import VerifiedFsmAuditArtifact, load_verified_fsm_audit_artifact
from .replay_chart_provider import (
    MissingEvidenceError,
    ReplayContext,
    open_replay_context,
)
from .replay_chart_store import (
    REPLAY_CHART_STORE,
    VerifierBundleRef,
    load_setup_ranges_v2,
    load_verified_replay_chart_artifact_v2,
)

__all__ = [
    "SETUP_STAGE_ORDER",
    "SetupEvidence",
    "SetupReplayContext",
    "list_setups",
    "open_setup_replay_context",
    "resolve_setup_selection",
    "setup_evidence",
]

#: Extended stage ladder for setup-mode review (§10 of the verifier plan).
SETUP_STAGE_ORDER: tuple[str, ...] = (
    "htf_tap",
    "activation",
    "parent_candidate",
    "parent_lock",
    "opposing_selected",
    "inversion",
    "entry_candidate",
    "trade_opened",
    "terminal",
)

_STAGE_BY_EVENT_KIND = {
    "htf_tap": "htf_tap",
    "parent_window_opened": "activation",
    "parent_candidate": "parent_candidate",
    "parent_window_parent_selected": "parent_candidate",
    "parent_window_parent_cleared": "parent_candidate",
    "parent_slot_death_provisional": "parent_candidate",
    "parentless_step": "parent_candidate",
    "parent_lock": "parent_lock",
    "opposing": "opposing_selected",
    "inversion": "inversion",
    "entry_joint_causality": "entry_candidate",
    "fvg_invalidation_event": "parent_candidate",
    "fvg_fill_event": "htf_tap",  # fills gate by ordinal, not stage bucket
    "setup_resolution": "terminal",
    "parent_slot_death_terminal": "terminal",
}


@dataclass(frozen=True)
class SetupReplayContext:
    base: ReplayContext
    bundle: VerifierBundleRef
    fsm_audit: VerifiedFsmAuditArtifact
    setup_ranges: pd.DataFrame


@dataclass(frozen=True)
class SetupEvidence:
    setup_id: str
    stage: str | None
    range_row: dict[str, Any]
    events: pd.DataFrame  # unified, contract-ordered, stage-gated event log
    tap_candidates: pd.DataFrame
    parent_candidates: pd.DataFrame
    opposing: pd.DataFrame
    lock: pd.DataFrame
    inversion: pd.DataFrame
    fill_events: pd.DataFrame
    slot_deaths: pd.DataFrame
    window_events: pd.DataFrame
    parentless_intervals: pd.DataFrame
    entry_causality: pd.DataFrame
    terminal: dict[str, Any] | None
    gating_report: dict[str, Any]


def open_setup_replay_context(
    repo_root: Path,
    bundle: VerifierBundleRef,
) -> SetupReplayContext:
    """Open every verified artifact in the bundle; all identities are pinned
    (a mismatched fsm-audit or replay-chart id refuses to open)."""
    repo_root = Path(repo_root).resolve()
    fsm_audit = load_verified_fsm_audit_artifact(
        repo_root / FSM_AUDIT_DATASET_DIR,
        bundle.fsm_audit_artifact_id,
    )
    if fsm_audit.manifest_payload_sha256 != bundle.fsm_audit_manifest_hash:
        raise MissingEvidenceError(
            "fsm_audit", "fsm-audit manifest hash differs from the bundle reference"
        )
    pair_ref = bundle.pair_ref()
    replay = load_verified_replay_chart_artifact_v2(
        repo_root / REPLAY_CHART_STORE,
        bundle.replay_chart_artifact_id,
        expected_pair=pair_ref,
        expected_fsm_audit_artifact_id=bundle.fsm_audit_artifact_id,
    )
    if (
        replay.manifest.get("manifest_payload_sha256")
        != bundle.replay_chart_manifest_hash
    ):
        raise MissingEvidenceError(
            "replay_chart", "replay-chart manifest hash differs from the bundle"
        )
    base = open_replay_context(repo_root, pair_ref, verified_replay=replay)
    setup_ranges = load_setup_ranges_v2(replay)
    _assert_total_order(fsm_audit)
    return SetupReplayContext(
        base=base,
        bundle=bundle,
        fsm_audit=fsm_audit,
        setup_ranges=setup_ranges,
    )


def _assert_total_order(fsm_audit: VerifiedFsmAuditArtifact) -> None:
    for table in (
        AuditTable.FILL_EVENT,
        AuditTable.ENTRY_CAUSALITY,
        AuditTable.SLOT_DEATH,
        AuditTable.PARENT_WINDOW,
        AuditTable.PARENTLESS_STEP,
    ):
        frame = fsm_audit.tables.get(table, pd.DataFrame())
        if frame.empty:
            continue
        keyed = frame.loc[
            :,
            [
                "stamp_source_step_ordinal",
                "stamp_reducer_substep",
                "stamp_reducer_substep_ordinal",
            ],
        ]
        if keyed.duplicated().any():
            raise MissingEvidenceError(
                "audit_ordering",
                f"{table.value} has an ambiguous cross-channel ordering key",
            )


def list_setups(ctx: SetupReplayContext) -> pd.DataFrame:
    """All setups (expected 215) with the §10.2 filter columns. The
    conflict/suppression flags are PRESENT-BUT-EMPTY when the audit shows zero
    such events — surfaced honestly, never hidden."""
    ranges = ctx.setup_ranges.copy()
    taps = ctx.fsm_audit.tables.get(AuditTable.HTF_TAP, pd.DataFrame())
    conflicted_setups: set[str] = set()
    if not taps.empty:
        conflicted = taps.loc[taps["conflicted"].astype(bool)]
        conflicted_setups = set(
            conflicted["envelope_setup_id"].dropna().astype(str)
        ) - {""}
    ranges["conflict_flag"] = ranges["setup_id"].astype(str).isin(conflicted_setups)
    deaths = ctx.fsm_audit.tables.get(AuditTable.SLOT_DEATH, pd.DataFrame())
    structural_setups: set[str] = set()
    if not deaths.empty:
        structural_setups = set(
            deaths.loc[deaths["structural_close"].astype(bool), "setup_id"].astype(str)
        )
    ranges["structural_suppression_flag"] = (
        ranges["setup_id"].astype(str).isin(structural_setups)
    )
    ranges["parentless"] = ranges["parentless_interval_count"].astype(int) > 0
    return ranges.reset_index(drop=True)


def resolve_setup_selection(ctx: SetupReplayContext, query: str) -> dict[str, Any]:
    """Resolve a setup id (exact or unique prefix) to its range row."""
    text = str(query).strip().lower()
    if not text:
        raise MissingEvidenceError("selection", "empty setup selection")
    ids = ctx.setup_ranges["setup_id"].astype(str)
    exact = ctx.setup_ranges.loc[ids == text]
    if len(exact) == 1:
        return exact.iloc[0].to_dict()
    prefixed = ctx.setup_ranges.loc[ids.str.startswith(text)]
    if len(prefixed) == 1:
        return prefixed.iloc[0].to_dict()
    if len(prefixed) > 1:
        raise MissingEvidenceError("selection", f"setup prefix {text!r} is ambiguous")
    raise MissingEvidenceError("selection", f"no setup matches {text!r}")


def _setup_channel_rows(
    ctx: SetupReplayContext, table: AuditTable, setup_id: str, column: str = "setup_id"
) -> pd.DataFrame:
    frame = ctx.fsm_audit.tables.get(table, pd.DataFrame())
    if frame.empty or column not in frame:
        return pd.DataFrame()
    return frame.loc[frame[column].astype(str) == setup_id].reset_index(drop=True)


def _setup_trace_rows(
    ctx: SetupReplayContext, table: AuditTable, setup_id: str
) -> pd.DataFrame:
    frame = ctx.fsm_audit.tables.get(table, pd.DataFrame())
    if frame.empty or "envelope_setup_id" not in frame:
        return pd.DataFrame()
    return frame.loc[
        frame["envelope_setup_id"].astype(str) == setup_id
    ].reset_index(drop=True)


def _event_row(
    kind: str,
    stage: str,
    ts: Any,
    *,
    order_key: tuple,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_kind": kind,
        "stage": stage,
        "ts_utc": ts,
        "order_bracket": order_key[0],
        "order_channel": order_key[1],
        "order_step": order_key[2],
        "order_substep": order_key[3],
        "order_substep_ordinal": order_key[4],
        "order_seq": order_key[5],
        **payload,
    }


def _channel_key(row: dict[str, Any]) -> tuple:
    # channel rows sort strictly BEFORE the core row whose global ordinal
    # equals their `after` bracket; among themselves the a2 contract orders.
    return (
        int(row["core_trace_ordinal_after_global"]),
        0,
        int(row["stamp_source_step_ordinal"]),
        str(row["stamp_reducer_substep"]),
        int(row["stamp_reducer_substep_ordinal"]),
        int(row["stamp_audit_seq"]),
    )


def _trace_key(row: dict[str, Any]) -> tuple:
    return (int(row["trace_ordinal"]), 1, 0, "", 0, 0)


def _stage_gate_key(
    stage: str,
    *,
    log: pd.DataFrame,
    tap_candidates: pd.DataFrame,
    windows: pd.DataFrame,
    parent_candidates: pd.DataFrame,
    lock: pd.DataFrame,
    opposing: pd.DataFrame,
    inversion: pd.DataFrame,
    causality: pd.DataFrame,
) -> tuple | None:
    """The order point of the canonical event for ``stage``; None when the
    setup never reached it (degrades to the terminal view)."""

    def _max_trace(frame: pd.DataFrame) -> tuple | None:
        if frame.empty:
            return None
        row = frame.loc[frame["trace_ordinal"].astype(int).idxmax()].to_dict()
        return _trace_key(row)

    def _first_channel(frame: pd.DataFrame, mask=None) -> tuple | None:
        if frame.empty:
            return None
        scoped = frame if mask is None else frame.loc[mask(frame)]
        if scoped.empty:
            return None
        keys = sorted(_channel_key(row) for row in scoped.to_dict("records"))
        return keys[0]

    if stage == "htf_tap":
        return _max_trace(tap_candidates)
    if stage == "activation":
        return _first_channel(windows, lambda f: f["event_kind"] == "opened")
    if stage == "parent_candidate":
        lock_key = _max_trace(lock)
        if lock_key is not None:
            return lock_key
        return _max_trace(parent_candidates) or _first_channel(
            windows, lambda f: f["event_kind"] == "opened"
        )
    if stage == "parent_lock":
        return _max_trace(lock)
    if stage == "opposing_selected":
        selected_rows = (
            opposing.loc[opposing["selected"].astype(bool)]
            if not opposing.empty
            else pd.DataFrame()
        )
        return _max_trace(selected_rows)
    if stage == "inversion":
        return _max_trace(inversion)
    if stage in ("entry_candidate", "trade_opened"):
        if causality.empty:
            return None
        keys = sorted(_channel_key(row) for row in causality.to_dict("records"))
        return keys[-1]
    return None


def setup_evidence(
    ctx: SetupReplayContext,
    setup_id: str,
    *,
    stage: str | None = None,
) -> SetupEvidence:
    """All persisted evidence for one setup, contract-ordered; ``stage`` gates
    the unified event log point-in-time (events after the stage gate are
    hidden and counted in the gating report)."""
    if stage is not None and stage not in SETUP_STAGE_ORDER:
        raise MissingEvidenceError("stage", f"unknown setup stage {stage!r}")
    matches = ctx.setup_ranges.loc[
        ctx.setup_ranges["setup_id"].astype(str) == str(setup_id)
    ]
    if len(matches) != 1:
        raise MissingEvidenceError("setup", f"setup {setup_id} is not in this bundle")
    range_row = matches.iloc[0].to_dict()

    taps_all = ctx.fsm_audit.tables.get(AuditTable.HTF_TAP, pd.DataFrame())
    selected_tap = _setup_trace_rows(ctx, AuditTable.HTF_TAP, str(setup_id))
    tap_candidates = pd.DataFrame()
    if not selected_tap.empty and not taps_all.empty:
        tap_cursor = str(selected_tap.iloc[0]["tap_cursor"])
        tap_candidates = taps_all.loc[
            taps_all["tap_cursor"].astype(str) == tap_cursor
        ].reset_index(drop=True)

    parent_candidates = _setup_trace_rows(ctx, AuditTable.PARENT_CANDIDATE, str(setup_id))
    opposing = _setup_trace_rows(ctx, AuditTable.OPPOSING, str(setup_id))
    lock = _setup_trace_rows(ctx, AuditTable.PARENT_LOCK, str(setup_id))
    inversion = _setup_trace_rows(ctx, AuditTable.INVERSION, str(setup_id))
    resolution = _setup_trace_rows(ctx, AuditTable.SETUP_RESOLUTION, str(setup_id))
    fills = _setup_channel_rows(ctx, AuditTable.FILL_EVENT, str(setup_id))
    deaths = _setup_channel_rows(ctx, AuditTable.SLOT_DEATH, str(setup_id))
    windows = _setup_channel_rows(ctx, AuditTable.PARENT_WINDOW, str(setup_id))
    steps = _setup_channel_rows(ctx, AuditTable.PARENTLESS_STEP, str(setup_id))
    intervals = _setup_channel_rows(ctx, AuditTable.PARENTLESS_INTERVAL, str(setup_id))
    causality = _setup_channel_rows(ctx, AuditTable.ENTRY_CAUSALITY, str(setup_id))

    events: list[dict[str, Any]] = []
    for _, row in tap_candidates.iterrows():
        events.append(
            _event_row(
                "htf_tap",
                "htf_tap",
                row["envelope_ts_utc"],
                order_key=_trace_key(row),
                payload={
                    "fvg_id": row["fvg_fvg_id"],
                    "selected": bool(row["selected"]),
                    "drop_reason": row["drop_reason"],
                },
            )
        )
    for _, row in parent_candidates.iterrows():
        events.append(
            _event_row(
                "parent_candidate",
                "parent_candidate",
                row["envelope_ts_utc"],
                order_key=_trace_key(row),
                payload={
                    "fvg_id": row["fvg_fvg_id"],
                    "selected": bool(row["selected"]),
                    "drop_reason": row["drop_reason"],
                },
            )
        )
    for _, row in opposing.iterrows():
        events.append(
            _event_row(
                "opposing",
                "opposing_selected",
                row["envelope_ts_utc"],
                order_key=_trace_key(row),
                payload={
                    "fvg_id": row["fvg_fvg_id"],
                    "selected": bool(row["selected"]),
                    "drop_reason": row["drop_reason"],
                },
            )
        )
    for _, row in lock.iterrows():
        events.append(
            _event_row(
                "parent_lock",
                "parent_lock",
                row["envelope_ts_utc"],
                order_key=_trace_key(row),
                payload={"fvg_id": row["parent_fvg_id"], "selected": True,
                         "drop_reason": None},
            )
        )
    for _, row in inversion.iterrows():
        events.append(
            _event_row(
                "inversion",
                "inversion",
                row["envelope_ts_utc"],
                order_key=_trace_key(row),
                payload={"fvg_id": row["opposing_fvg_id"], "selected": True,
                         "drop_reason": None, "semantic": row["semantic"]},
            )
        )
    for _, row in resolution.iterrows():
        events.append(
            _event_row(
                "setup_resolution",
                "terminal",
                row["envelope_ts_utc"],
                order_key=_trace_key(row),
                payload={"fvg_id": row["htf_fvg_id"], "selected": False,
                         "drop_reason": None, "resolution": row["resolution"]},
            )
        )
    for _, row in windows.iterrows():
        kind = f"parent_window_{row['event_kind']}"
        stage_name = "activation" if row["event_kind"] == "opened" else "parent_candidate"
        events.append(
            _event_row(
                kind,
                stage_name,
                row["envelope_ts_utc"],
                order_key=_channel_key(row),
                payload={"fvg_id": row["parent_fvg_id"], "selected": None,
                         "drop_reason": None},
            )
        )
    for _, row in fills.iterrows():
        events.append(
            _event_row(
                "fvg_fill_event",
                _STAGE_BY_EVENT_KIND["fvg_fill_event"],
                row["envelope_ts_utc"],
                order_key=_channel_key(row),
                payload={
                    "fvg_id": row["fvg_fvg_id"],
                    "selected": bool(row["selected_for_setup"]),
                    "drop_reason": None,
                    "fill_kind": row["event_kind"],
                    "fill_depth_ticks": row["fill_depth_ticks"],
                },
            )
        )
    for _, row in deaths.iterrows():
        terminal_death = bool(row["setup_terminated"])
        events.append(
            _event_row(
                "parent_slot_death",
                "terminal" if terminal_death else "parent_candidate",
                row["envelope_ts_utc"],
                order_key=_channel_key(row),
                payload={
                    "fvg_id": row["parent_fvg_id"],
                    "selected": None,
                    "drop_reason": row["death_reason"],
                    "terminated": terminal_death,
                },
            )
        )
    for _, row in causality.iterrows():
        events.append(
            _event_row(
                "entry_joint_causality",
                "entry_candidate",
                row["envelope_ts_utc"],
                order_key=_channel_key(row),
                payload={
                    "fvg_id": row["entry_fvg_id"],
                    "selected": bool(row["satisfied"]),
                    "drop_reason": None,
                    "candidate_id": row["candidate_id"],
                },
            )
        )
    for _, row in steps.iterrows():
        events.append(
            _event_row(
                "parentless_step",
                "parent_candidate",
                row["envelope_ts_utc"],
                order_key=_channel_key(row),
                payload={"fvg_id": None, "selected": None, "drop_reason": None},
            )
        )

    log = pd.DataFrame(events)
    hidden = 0
    if not log.empty:
        order_columns = [
            "order_bracket",
            "order_channel",
            "order_step",
            "order_substep",
            "order_substep_ordinal",
            "order_seq",
        ]
        if log.loc[:, order_columns].duplicated().any():
            raise MissingEvidenceError(
                "audit_ordering", f"setup {setup_id} evidence ordering is ambiguous"
            )
        log = log.sort_values(order_columns, kind="mergesort").reset_index(drop=True)
        if stage is not None and stage != "terminal":
            # TRUE point-in-time gate: everything at or before the canonical
            # stage event's ORDER POINT is visible; everything after — fills
            # included — is hidden and counted. A stage the setup never
            # reached degrades to the terminal view (its whole life).
            gate_key = _stage_gate_key(
                stage,
                log=log,
                tap_candidates=tap_candidates,
                windows=windows,
                parent_candidates=parent_candidates,
                lock=lock,
                opposing=opposing,
                inversion=inversion,
                causality=causality,
            )
            if gate_key is not None:
                keys = list(
                    zip(
                        *(log[column] for column in order_columns),
                        strict=True,
                    )
                )
                mask = [key <= gate_key for key in keys]
                visible = log.loc[mask].reset_index(drop=True)
                hidden = int(len(log) - len(visible))
                log = visible

    terminal_row = None
    terminal_deaths = (
        deaths.loc[deaths["setup_terminated"].astype(bool)]
        if not deaths.empty
        else pd.DataFrame()
    )
    if len(terminal_deaths):
        terminal_row = terminal_deaths.iloc[0].to_dict()
    return SetupEvidence(
        setup_id=str(setup_id),
        stage=stage,
        range_row=range_row,
        events=log,
        tap_candidates=tap_candidates,
        parent_candidates=parent_candidates,
        opposing=opposing,
        lock=lock,
        inversion=inversion,
        fill_events=fills,
        slot_deaths=deaths,
        window_events=windows,
        parentless_intervals=intervals,
        entry_causality=causality,
        terminal=terminal_row,
        gating_report={
            "stage": stage,
            "hidden_events": hidden,
            "total_events": hidden + int(len(log)),
        },
    )
