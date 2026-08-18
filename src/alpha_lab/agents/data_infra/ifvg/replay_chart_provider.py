"""Pure evidence provider for the IFVG visual trade verifier (context_v1 lane).

Resolves a candidate/decision/trade by exact ID into everything the chart
needs — zones, lifecycle, risk, execution, structure, displacement, pools,
sessions, and per-tier model statuses — from the verified v2/v3 pair and the
``ifvg_replay_chart_v1`` companion artifact.

Hard rules:

* exact IDs only — no setup-only, nearest-time, row-order, or keep-last match;
* model runs are optional enrichment, never a prerequisite for the chart;
* authorization lives here, not in widgets: nothing at or past 2026-06-11 is
  readable and there is no ``allow_sealed`` parameter anywhere;
* absent persisted evidence raises :class:`MissingEvidenceError` or lands in
  the gating report — it is never approximated.

Point-in-time gating follows the ``stage_gate_ordinal_cursor_ts_v1`` policy:
compare by trace ordinal when both sides have one, else by event cursor, else
by timestamp — and a pure timestamp TIE is indeterminate, never ordered.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .artifact_io import (
    VerifiedIfvgPair,
    load_verified_ifvg_pair,
    load_verified_label_source_bars,
)
from .config import V2_DATASET_DIR, V3_DATASET_DIR
from .context_contracts import ContextRecordTable
from .context_run_store import (
    CONTEXT_RUN_CATALOG,
    CONTEXT_RUN_STORE,
    StoredContextRun,
    list_context_run_catalog,
    load_context_experiment_run,
)
from .contracts import RecordTable
from .replay_chart_store import (
    AUTHORIZED_CUTOFF_UTC,
    PRIMARY_LABEL_FAMILY,
    REPLAY_CHART_CATALOG,
    REPLAY_CHART_STORE,
    REPLAY_TIMEFRAMES_SECONDS,
    ArtifactPairRef,
    VerifiedReplayChartArtifact,
    find_replay_artifact,
    load_verified_replay_chart_artifact,
    read_replay_chart_catalog,
)

__all__ = [
    "MODEL_TIERS",
    "MAX_1M_BARS_PER_RENDER",
    "FsmStage",
    "StageGate",
    "MissingEvidenceError",
    "ReplayAuthorizationError",
    "RangeTooLargeError",
    "ZoneEvidence",
    "CandidateEvidence",
    "ReplayContext",
    "open_replay_context",
    "list_candidates",
    "resolve_selection",
    "candidate_evidence",
    "bars_for_pane",
    "chart_range",
    "session_scheme_windows",
]

MODEL_TIERS = ("M0", "M1_PRIMARY", "M2", "M3")
MAX_1M_BARS_PER_RENDER = 3000
_CUTOFF = pd.Timestamp(AUTHORIZED_CUTOFF_UTC)
_ZONE_ROLES = ("htf", "parent", "opposing", "entry_fvg")
_TRANSITION_BARS = ("tap", "lock", "inversion", "entry")


class MissingEvidenceError(LookupError):
    """Requested evidence is not persisted; it is never approximated."""

    def __init__(self, evidence_kind: str, detail: str, *, candidate_id: str | None = None):
        self.evidence_kind = evidence_kind
        self.candidate_id = candidate_id
        super().__init__(f"missing {evidence_kind} evidence: {detail}")


class ReplayAuthorizationError(PermissionError):
    """The request touches protected (2026-06-11) or sealed (>= 2026-06-12) data."""


class RangeTooLargeError(ValueError):
    """The requested bar range exceeds the render safety limit; narrow it."""


class FsmStage(StrEnum):
    TAP = "tap"
    PARENT = "parent"
    LOCK = "lock"
    OPPOSING = "opposing"
    INVERSION = "inversion"
    ENTRY = "entry"
    RESOLUTION = "resolution"


STAGE_ORDER: tuple[FsmStage, ...] = (
    FsmStage.TAP,
    FsmStage.PARENT,
    FsmStage.LOCK,
    FsmStage.OPPOSING,
    FsmStage.INVERSION,
    FsmStage.ENTRY,
    FsmStage.RESOLUTION,
)


@dataclass(frozen=True)
class StageGate:
    stage: FsmStage
    ts_utc: pd.Timestamp
    event_cursor: str | None
    trace_ordinal: int | None
    source_event_id: str
    source_kind: str


@dataclass(frozen=True)
class ZoneEvidence:
    role: str
    fvg_id: str
    timeframe_seconds: int
    direction: str
    gap_low_ticks: int
    gap_high_ticks: int
    size_ticks: int
    a_bar_id: str
    c_bar_id: str
    a_open_ts_utc: pd.Timestamp
    confirmed_ts_utc: pd.Timestamp
    trading_day: str


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    mode: str
    stage: str | None
    identity: dict[str, Any]
    lineage: dict[str, Any]
    range_row: dict[str, Any]
    zones: tuple[ZoneEvidence, ...]
    transition_bars: dict[str, dict[str, Any]]
    lifecycle: pd.DataFrame
    risk: dict[str, Any]
    execution: dict[str, Any] | None
    counterfactual_labels: tuple[dict[str, Any], ...]
    structure: pd.DataFrame
    structure_stage_summary: pd.DataFrame
    displacement: pd.DataFrame
    pools: pd.DataFrame
    pool_members: pd.DataFrame
    sweep_links: pd.DataFrame
    sessions: dict[str, Any]
    model: dict[str, dict[str, Any]]
    stage_gates: dict[str, StageGate]
    gating_report: dict[str, Any]
    anchor_240m_status: str


@dataclass(frozen=True)
class ReplayContext:
    repo_root: Path
    pair_ref: ArtifactPairRef
    pair: VerifiedIfvgPair
    replay: VerifiedReplayChartArtifact
    bars_1m: pd.DataFrame
    tier_runs: dict[str, StoredContextRun | None]
    tier_notes: dict[str, str]
    m3_qualifying_ids: frozenset[str] = field(default_factory=frozenset)


def _table(pair: VerifiedIfvgPair, table: RecordTable | ContextRecordTable) -> pd.DataFrame:
    source = pair.v2 if isinstance(table, RecordTable) else pair.v3
    return source.tables[table]


def _run_matches_pair(run: StoredContextRun, pair_ref: ArtifactPairRef) -> bool:
    artifact_pair = run.config.dataset.artifact_pair
    return (
        artifact_pair.v2.artifact_id == pair_ref.v2_dataset_id
        and artifact_pair.v2.manifest_payload_sha256 == pair_ref.v2_manifest_hash
        and artifact_pair.v3.artifact_id == pair_ref.v3_dataset_id
        and artifact_pair.v3.manifest_payload_sha256 == pair_ref.v3_manifest_hash
    )


def open_replay_context(
    repo_root: Path,
    pair: ArtifactPairRef,
    *,
    replay_artifact_id: str | None = None,
    verified_pair: VerifiedIfvgPair | None = None,
    verified_replay: VerifiedReplayChartArtifact | None = None,
) -> ReplayContext:
    """Open all verified evidence sources for one exact v2/v3 pair.

    Model runs are discovered from the run catalog and attached as optional
    enrichment; a pair with zero runs still opens.  A caller that already
    holds the verified pair may pass it to skip re-verification; its identity
    is still checked against the pair reference below.
    """
    repo_root = Path(repo_root).resolve()
    if verified_pair is None:
        verified_pair = load_verified_ifvg_pair(
            v2_root=repo_root / V2_DATASET_DIR,
            v2_artifact_id=pair.v2_dataset_id,
            v3_root=repo_root / V3_DATASET_DIR,
            v3_artifact_id=pair.v3_dataset_id,
        )
    if verified_pair.v2.reference.artifact_id != pair.v2_dataset_id:
        raise MissingEvidenceError("pair", "v2 artifact ID differs from the pair reference")
    if verified_pair.v3.reference.artifact_id != pair.v3_dataset_id:
        raise MissingEvidenceError("pair", "v3 artifact ID differs from the pair reference")
    if verified_pair.v2.reference.manifest_payload_sha256 != pair.v2_manifest_hash:
        raise MissingEvidenceError("pair", "v2 manifest hash differs from the pair reference")
    if verified_pair.v3.reference.manifest_payload_sha256 != pair.v3_manifest_hash:
        raise MissingEvidenceError("pair", "v3 manifest hash differs from the pair reference")

    if verified_replay is not None:
        # A setup-aware (v2-kind) artifact already verified against this pair
        # by its own loader; identity is re-checked here.
        if verified_replay.source_pair.as_dict() != pair.as_dict():
            raise MissingEvidenceError(
                "replay_chart_artifact",
                "verified replay artifact was built for a different pair",
            )
        replay = verified_replay
    else:
        if replay_artifact_id is None:
            catalog = read_replay_chart_catalog(repo_root / REPLAY_CHART_CATALOG)
            replay_artifact_id = find_replay_artifact(catalog, pair)
            if replay_artifact_id is None:
                raise MissingEvidenceError(
                    "replay_chart_artifact",
                    "no replay-chart artifact for this exact pair; "
                    "run scripts/ifvg_build_replay_chart.py build",
                )
        replay = load_verified_replay_chart_artifact(
            repo_root / REPLAY_CHART_STORE, replay_artifact_id, expected_pair=pair
        )
    bars_1m = load_verified_label_source_bars(verified_pair.v2).copy()
    bars_1m["close_ts_utc"] = pd.to_datetime(bars_1m["close_ts_utc"], utc=True, errors="raise")
    bars_1m = bars_1m.sort_values("close_ts_utc", kind="mergesort").reset_index(drop=True)

    tier_runs: dict[str, StoredContextRun | None] = {tier: None for tier in MODEL_TIERS}
    tier_notes: dict[str, str] = {}
    try:
        catalog_rows = list_context_run_catalog(catalog_path=repo_root / CONTEXT_RUN_CATALOG)
    except Exception:
        catalog_rows = []
    matching: dict[str, list[StoredContextRun]] = {}
    for row in catalog_rows:
        try:
            run = load_context_experiment_run(
                row["run_id"], base_dir=repo_root / CONTEXT_RUN_STORE
            )
        except Exception:
            tier_notes.setdefault("load_failures", "")
            tier_notes["load_failures"] = (
                tier_notes["load_failures"] + f"{row['run_id'][:12]} "
            ).strip()
            continue
        if not _run_matches_pair(run, pair):
            continue
        matching.setdefault(run.config.feature_tier.value, []).append(run)
    for tier, runs in matching.items():
        runs = sorted(runs, key=lambda item: item.result.run_id)
        if tier in tier_runs:
            tier_runs[tier] = runs[0]
            if len(runs) > 1:
                tier_notes[tier] = (
                    f"{len(runs)} runs share this tier and pair; showing "
                    f"{runs[0].result.run_id[:12]}"
                )

    sweeps = _table(verified_pair, ContextRecordTable.EQUAL_LEVEL_SWEEP_LINK)
    qualifying = frozenset(
        sweeps.loc[sweeps["qualifies_opposing_leg"].fillna(False), "sweep_link_id"].astype(str)
    )
    captures = _table(verified_pair, ContextRecordTable.CONTEXT_CAPTURE)
    entry_captures = captures[captures["capture_kind"] == "entry_candidate"]
    m3_ids = frozenset(
        entry_captures.loc[
            entry_captures["selected_opposing_leg_sweep_link_id"].astype(str).isin(qualifying),
            "candidate_id",
        ].astype(str)
    )

    return ReplayContext(
        repo_root=repo_root,
        pair_ref=pair,
        pair=verified_pair,
        replay=replay,
        bars_1m=bars_1m,
        tier_runs=tier_runs,
        tier_notes=tier_notes,
        m3_qualifying_ids=m3_ids,
    )


def list_candidates(ctx: ReplayContext) -> pd.DataFrame:
    """All 132 candidates with the flags the navigation filters need."""
    candidates = _table(ctx.pair, RecordTable.ENTRY_CANDIDATE)
    executed = _table(ctx.pair, RecordTable.EXECUTED_TRADE).set_index("candidate_id")
    labels = _table(ctx.pair, RecordTable.CANDIDATE_LABEL)
    primary = labels[labels["label_family"] == PRIMARY_LABEL_FAMILY].set_index("candidate_id")
    ranges = ctx.replay.candidate_ranges.set_index("candidate_id")

    rows = []
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate["candidate_id"])
        range_row = ranges.loc[candidate_id]
        label_row = primary.loc[candidate_id] if candidate_id in primary.index else None
        trade_row = executed.loc[candidate_id] if candidate_id in executed.index else None
        block_reasons = _parse_block_reasons(candidate.get("block_reasons"))
        rows.append(
            {
                "candidate_id": candidate_id,
                "setup_id": str(candidate["setup_id"]),
                "decision_id": range_row["decision_id"],
                "trade_id": range_row["trade_id"],
                "trading_day": str(candidate["trading_day"]),
                "entry_ts_utc": pd.Timestamp(candidate["envelope_ts_utc"]),
                "entry_family": str(candidate["entry_family"]),
                "entry_session": str(candidate["entry_session"]),
                "is_warmup": bool(candidate["is_warmup"]),
                "executed": trade_row is not None,
                "blocked": len(block_reasons) > 0,
                "block_reasons": ", ".join(block_reasons),
                "label": str(label_row["label"]) if label_row is not None else None,
                "censored": bool(label_row["censored"]) if label_row is not None else False,
                "resolution": (
                    str(trade_row["resolution"]) if trade_row is not None else None
                ),
                "realized_r": (
                    float(trade_row["realized_r"]) if trade_row is not None else None
                ),
                "display_end_source": range_row["display_end_source"],
                "m3_qualifying": candidate_id in ctx.m3_qualifying_ids,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["trading_day", "entry_ts_utc", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)


def _parse_block_reasons(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [str(item) for item in raw]
    text = str(raw).strip()
    if not text or text == "[]":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    return [str(item) for item in parsed] if isinstance(parsed, list) else [text]


def resolve_selection(
    ctx: ReplayContext,
    *,
    candidate_id: str | None = None,
    decision_id: str | None = None,
    trade_id: str | None = None,
    setup_id: str | None = None,
) -> str:
    """Exact-ID selection; exactly one identifier kind, no fallback matching.

    ``setup_id`` is the fourth exact kind (``ifvg_prop_robust_config_search_v1``
    R2): it resolves into the candidate verifier only when that exact setup has
    exactly ONE entry candidate. A candidate-less or multi-candidate setup
    refuses with the exact count — the setup-mode verifier is the drill surface
    for those; nothing is ever picked by policy or fuzzy matching.
    """
    given = [
        value for value in (candidate_id, decision_id, trade_id, setup_id) if value
    ]
    if len(given) != 1:
        raise MissingEvidenceError("selection", "exactly one exact ID is required")
    candidates = _table(ctx.pair, RecordTable.ENTRY_CANDIDATE)
    if candidate_id is not None:
        if candidate_id not in set(candidates["candidate_id"].astype(str)):
            raise MissingEvidenceError("selection", f"unknown candidate: {candidate_id}")
        return candidate_id
    if setup_id is not None:
        matches = candidates.loc[
            candidates["setup_id"].astype(str) == str(setup_id), "candidate_id"
        ]
        if len(matches) == 1:
            return str(matches.iloc[0])
        if len(matches) == 0:
            raise MissingEvidenceError(
                "selection",
                f"setup {setup_id} has no entry candidate; open it in setup mode",
            )
        raise MissingEvidenceError(
            "selection",
            f"setup {setup_id} has {len(matches)} entry candidates; open it in "
            "setup mode or select one exact candidate_id",
        )
    dossiers = _table(ctx.pair, RecordTable.GEOMETRY_DOSSIER)
    column = "decision_id" if decision_id is not None else "trade_id"
    value = decision_id if decision_id is not None else trade_id
    matches = dossiers.loc[dossiers[column].astype(str) == str(value), "candidate_id"]
    if len(matches) != 1:
        raise MissingEvidenceError("selection", f"unknown {column}: {value}")
    return str(matches.iloc[0])


# ---------------------------------------------------------------------------
# Stage gates and point-in-time comparison


def _cursor_ts(cursor: str | None) -> pd.Timestamp | None:
    if not cursor:
        return None
    try:
        return pd.Timestamp(str(cursor).split("|", 1)[0]).tz_convert("UTC")
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class _GateKeys:
    ts: pd.Timestamp | None
    cursor: str | None
    ordinal: int | None


def _gate_visible(gate: StageGate, keys: _GateKeys) -> bool | None:
    """True/False when the ordering is established; None when indeterminate.

    stage_gate_ordinal_cursor_ts_v1: trace ordinal, then cursor, then
    timestamp; a pure timestamp tie is never ordered.
    """
    if keys.ordinal is not None and gate.trace_ordinal is not None:
        return int(keys.ordinal) <= int(gate.trace_ordinal)
    if keys.cursor and gate.event_cursor:
        if str(keys.cursor) == str(gate.event_cursor):
            return True
        return str(keys.cursor) < str(gate.event_cursor)
    if keys.ts is None or pd.isna(keys.ts):
        return None
    if keys.ts < gate.ts_utc:
        return True
    if keys.ts > gate.ts_utc:
        return False
    return None


def _build_stage_gates(
    ctx: ReplayContext,
    candidate: dict[str, Any],
    dossier: pd.Series,
    lifecycle: pd.DataFrame,
    trade: pd.Series | None,
    range_row: pd.Series,
) -> tuple[dict[str, StageGate], list[dict[str, str]]]:
    gates: dict[str, StageGate] = {}
    ungateable: list[dict[str, str]] = []
    candidate_id = str(candidate["candidate_id"])
    setup_id = str(candidate["setup_id"])

    def _add(stage: FsmStage, gate: StageGate | None, missing_detail: str) -> None:
        if gate is None:
            ungateable.append(
                {"kind": "stage_gate", "object_id": stage.value, "missing": missing_detail}
            )
        else:
            gates[stage.value] = gate

    def _lifecycle_gate(stage: FsmStage, transition: str, source: str) -> StageGate | None:
        rows = lifecycle[lifecycle["transition"] == transition]
        if stage in (FsmStage.ENTRY, FsmStage.RESOLUTION):
            rows = rows[rows["candidate_id"].astype(str) == candidate_id]
        if rows.empty:
            return None
        row = rows.sort_values("trace_ordinal", kind="mergesort").iloc[0]
        return StageGate(
            stage=stage,
            ts_utc=pd.Timestamp(row["envelope_ts_utc"]),
            event_cursor=str(row["event_cursor"]),
            trace_ordinal=int(row["trace_ordinal"]),
            source_event_id=str(row["lifecycle_event_id"]),
            source_kind=source,
        )

    captures = _table(ctx.pair, ContextRecordTable.CONTEXT_CAPTURE)
    setup_captures = captures[captures["setup_id"].astype(str) == setup_id]

    tap_rows = setup_captures[setup_captures["capture_kind"] == "htf_tap"]
    if not tap_rows.empty:
        row = tap_rows.iloc[0]
        _add(
            FsmStage.TAP,
            StageGate(
                stage=FsmStage.TAP,
                ts_utc=pd.Timestamp(row["as_of_ts"]),
                event_cursor=str(row["as_of_cursor"]),
                trace_ordinal=None,
                source_event_id=str(row["context_capture_id"]),
                source_kind="context_capture:htf_tap",
            ),
            "",
        )
    else:
        _add(
            FsmStage.TAP,
            _lifecycle_gate(FsmStage.TAP, "setup_activated", "lifecycle:setup_activated"),
            "no htf_tap capture and no setup_activated event",
        )

    parent_fvg_id = dossier.get("geometry_parent_fvg_id")
    parent_gate: StageGate | None = None
    if pd.notna(parent_fvg_id):
        parent_rows = setup_captures[
            (setup_captures["capture_kind"] == "parent_candidate")
            & (setup_captures["evidence_id"].astype(str) == str(parent_fvg_id))
        ]
        if not parent_rows.empty:
            row = parent_rows.sort_values("as_of_cursor", kind="mergesort").iloc[0]
            parent_gate = StageGate(
                stage=FsmStage.PARENT,
                ts_utc=pd.Timestamp(row["as_of_ts"]),
                event_cursor=str(row["as_of_cursor"]),
                trace_ordinal=None,
                source_event_id=str(row["context_capture_id"]),
                source_kind="context_capture:parent_candidate",
            )
    if parent_gate is None and pd.notna(dossier.get("geometry_parent_confirmed_ts_utc")):
        parent_gate = StageGate(
            stage=FsmStage.PARENT,
            ts_utc=pd.Timestamp(dossier["geometry_parent_confirmed_ts_utc"]),
            event_cursor=None,
            trace_ordinal=None,
            source_event_id=str(parent_fvg_id),
            source_kind="geometry_dossier.geometry_parent_confirmed_ts_utc[fallback]",
        )
    _add(FsmStage.PARENT, parent_gate, "no selected-parent capture or dossier confirmation")

    _add(
        FsmStage.LOCK,
        _lifecycle_gate(FsmStage.LOCK, "parent_locked", "lifecycle:parent_locked"),
        "no parent_locked event",
    )
    _add(
        FsmStage.OPPOSING,
        _lifecycle_gate(FsmStage.OPPOSING, "opposing_armed", "lifecycle:opposing_armed"),
        "no opposing_armed event",
    )
    _add(
        FsmStage.INVERSION,
        _lifecycle_gate(FsmStage.INVERSION, "opposing_inverted", "lifecycle:opposing_inverted"),
        "no opposing_inverted event",
    )

    if trade is not None:
        entry_gate = _lifecycle_gate(FsmStage.ENTRY, "trade_opened", "lifecycle:trade_opened")
        if entry_gate is None:
            entry_gate = StageGate(
                stage=FsmStage.ENTRY,
                ts_utc=pd.Timestamp(trade["entry_ts_utc"]),
                event_cursor=str(trade["entry_cursor"]),
                trace_ordinal=None,
                source_event_id=str(trade["trade_id"]),
                source_kind="executed_trade.entry_cursor[fallback]",
            )
    else:
        entry_gate = StageGate(
            stage=FsmStage.ENTRY,
            ts_utc=pd.Timestamp(candidate["envelope_ts_utc"]),
            event_cursor=str(candidate["trigger_cursor"]),
            trace_ordinal=int(candidate["trace_ordinal"]),
            source_event_id=candidate_id,
            source_kind="entry_candidate.envelope_ts_utc",
        )
    _add(FsmStage.ENTRY, entry_gate, "no entry evidence")

    if trade is not None:
        resolution_gate = _lifecycle_gate(
            FsmStage.RESOLUTION, "trade_resolved", "lifecycle:trade_resolved"
        )
        if resolution_gate is None:
            resolution_gate = StageGate(
                stage=FsmStage.RESOLUTION,
                ts_utc=pd.Timestamp(trade["resolution_ts_utc"]),
                event_cursor=str(trade["resolution_cursor"]),
                trace_ordinal=None,
                source_event_id=str(trade["trade_id"]),
                source_kind="executed_trade.resolution_cursor[fallback]",
            )
    else:
        display_source = str(range_row["display_end_source"])
        display_end = range_row["display_end_ts"]
        if pd.notna(display_end):
            resolution_gate = StageGate(
                stage=FsmStage.RESOLUTION,
                ts_utc=pd.Timestamp(display_end),
                event_cursor=None,
                trace_ordinal=None,
                source_event_id=candidate_id,
                source_kind=f"candidate_bar_range.display_end_ts[{display_source}]",
            )
        else:
            resolution_gate = None
    _add(FsmStage.RESOLUTION, resolution_gate, "no resolution or setup-end evidence")

    return gates, ungateable


def _stage_index(stage: str) -> int:
    return [item.value for item in STAGE_ORDER].index(stage)


# ---------------------------------------------------------------------------
# Evidence assembly


def _zone_evidence(dossier: pd.Series) -> tuple[ZoneEvidence, ...]:
    zones = []
    for role in _ZONE_ROLES:
        prefix = f"geometry_{role}_"
        if pd.isna(dossier.get(f"{prefix}fvg_id")):
            continue
        zones.append(
            ZoneEvidence(
                role=role,
                fvg_id=str(dossier[f"{prefix}fvg_id"]),
                timeframe_seconds=int(dossier[f"{prefix}timeframe_seconds"]),
                direction=str(dossier[f"{prefix}direction"]),
                gap_low_ticks=int(dossier[f"{prefix}gap_low_ticks"]),
                gap_high_ticks=int(dossier[f"{prefix}gap_high_ticks"]),
                size_ticks=int(dossier[f"{prefix}size_ticks"]),
                a_bar_id=str(dossier[f"{prefix}a_bar_id"]),
                c_bar_id=str(dossier[f"{prefix}c_bar_id"]),
                a_open_ts_utc=pd.Timestamp(dossier[f"{prefix}a_open_ts_utc"]),
                confirmed_ts_utc=pd.Timestamp(dossier[f"{prefix}confirmed_ts_utc"]),
                trading_day=str(dossier[f"{prefix}trading_day"]),
            )
        )
    return tuple(zones)


def _transition_bar_snapshots(dossier: pd.Series) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for name in _TRANSITION_BARS:
        prefix = f"geometry_{name}_bar_"
        if pd.isna(dossier.get(f"{prefix}bar_id")):
            continue
        snapshots[name] = {
            "bar_id": str(dossier[f"{prefix}bar_id"]),
            "timeframe_seconds": int(dossier[f"{prefix}timeframe_seconds"]),
            "trading_day": str(dossier[f"{prefix}trading_day"]),
            "logical_open_ts_utc": pd.Timestamp(dossier[f"{prefix}logical_open_ts_utc"]),
            "logical_close_ts_utc": pd.Timestamp(dossier[f"{prefix}logical_close_ts_utc"]),
            "open_ticks": int(dossier[f"{prefix}open_ticks"]),
            "high_ticks": int(dossier[f"{prefix}high_ticks"]),
            "low_ticks": int(dossier[f"{prefix}low_ticks"]),
            "close_ticks": int(dossier[f"{prefix}close_ticks"]),
            "cursor": str(dossier[f"{prefix}cursor"]),
        }
    return snapshots


_STAGE_CAPTURE_KINDS: tuple[tuple[str, str], ...] = (
    ("tap", "htf_tap"),
    ("lock", "parent_lock"),
    ("inversion", "inversion"),
    ("entry", "entry_candidate"),
    ("decision", "eligible_decision"),
    ("trade", "executed_trade_link"),
)


def _stage_captures(ctx: ReplayContext, candidate: dict[str, Any]) -> pd.DataFrame:
    """Max-1 capture per stage: setup-scoped for tap/lock/inversion, exact-ID after."""
    captures = _table(ctx.pair, ContextRecordTable.CONTEXT_CAPTURE)
    setup_id = str(candidate["setup_id"])
    candidate_id = str(candidate["candidate_id"])
    selected = []
    for stage, kind in _STAGE_CAPTURE_KINDS:
        rows = captures[captures["capture_kind"] == kind]
        if stage in ("tap", "lock", "inversion"):
            rows = rows[rows["setup_id"].astype(str) == setup_id]
        else:
            rows = rows[rows["candidate_id"].astype(str) == candidate_id]
        if rows.empty:
            continue
        row = rows.iloc[0].copy()
        row["verifier_stage"] = stage
        selected.append(row)
    if not selected:
        return pd.DataFrame()
    return pd.DataFrame(selected).reset_index(drop=True)


def _structure_evidence(
    ctx: ReplayContext, stage_captures: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = _table(ctx.pair, ContextRecordTable.CONTEXT_STATE)
    structure = _table(ctx.pair, ContextRecordTable.CONTEXT_STRUCTURE_STATE)
    structure_index = structure.set_index("structure_state_id")
    state_index = states.set_index("context_state_id")

    summary_rows = []
    structure_rows = []
    for capture in stage_captures.to_dict("records"):
        state_id = capture.get("context_state_id")
        if state_id is None or pd.isna(state_id) or str(state_id) not in state_index.index:
            continue
        state = state_index.loc[str(state_id)]
        summary_rows.append(
            {
                "stage": capture["verifier_stage"],
                "as_of_ts": pd.Timestamp(capture["as_of_ts"]),
                "as_of_cursor": str(capture["as_of_cursor"]),
                "mtf_aligned_tf_count": state.get("mtf_aligned_tf_count"),
                "mtf_conflicting_tf_count": state.get("mtf_conflicting_tf_count"),
                "mtf_neutral_tf_count": state.get("mtf_neutral_tf_count"),
                "mtf_valid_tf_count": state.get("mtf_valid_tf_count"),
                "mtf_highest_aligned_tf_seconds": state.get("mtf_highest_aligned_tf_seconds"),
                "mtf_highest_conflicting_tf_seconds": state.get(
                    "mtf_highest_conflicting_tf_seconds"
                ),
            }
        )
        mtf_ids = state.get("mtf_state_ids")
        if mtf_ids is None or (isinstance(mtf_ids, float) and pd.isna(mtf_ids)):
            continue
        for structure_id in list(mtf_ids):
            if str(structure_id) not in structure_index.index:
                continue
            row = structure_index.loc[str(structure_id)]
            structure_rows.append(
                {
                    "stage": capture["verifier_stage"],
                    "structure_state_id": str(structure_id),
                    "source_timeframe": row["source_timeframe"],
                    "source_timeframe_seconds": int(row["source_timeframe_seconds"]),
                    "anchor_status": row.get("anchor_status"),
                    "structure_direction": row["structure_direction"],
                    "high_relationship": row["high_relationship"],
                    "low_relationship": row["low_relationship"],
                    "swing_sequence_state": row["swing_sequence_state"],
                    "last_break_type": row.get("last_break_type"),
                    "last_break_direction": row.get("last_break_direction"),
                    "break_bar_id": row.get("break_bar_id"),
                    "break_ts": row.get("break_ts"),
                    "broken_swing_id": row.get("broken_swing_id"),
                    "last_confirmed_swing_high_ticks": row.get(
                        "last_confirmed_swing_high_ticks"
                    ),
                    "last_confirmed_swing_low_ticks": row.get("last_confirmed_swing_low_ticks"),
                    "valid": row.get("valid"),
                    "source_confirmed_ts": row.get("source_confirmed_ts"),
                    "as_of_ts": row.get("as_of_ts"),
                    "as_of_cursor": row.get("as_of_cursor"),
                }
            )
    return (
        pd.DataFrame(structure_rows),
        pd.DataFrame(summary_rows),
    )


def _displacement_evidence(
    ctx: ReplayContext, stage_captures: pd.DataFrame
) -> pd.DataFrame:
    windows = _table(ctx.pair, ContextRecordTable.CONTEXT_DISPLACEMENT_WINDOW)
    window_index = windows.set_index("displacement_window_id")
    rows = []
    seen: set[str] = set()
    metric_columns = [
        "metrics_path_efficiency_abs",
        "metrics_body_fraction_mean",
        "metrics_opposing_wick_fraction_mean",
        "metrics_overlap_fraction_mean",
        "metrics_max_consecutive_directional_bars",
        "metrics_setup_net_move_normalized",
        "metrics_observed_bar_count",
        "metrics_missing_bar_count",
    ]
    for capture in stage_captures.to_dict("records"):
        window_ids = capture.get("displacement_window_ids")
        if window_ids is None or (isinstance(window_ids, float) and pd.isna(window_ids)):
            continue
        for window_id in list(window_ids):
            key = str(window_id)
            if key in seen or key not in window_index.index:
                continue
            seen.add(key)
            window = window_index.loc[key]
            rows.append(
                {
                    "displacement_window_id": key,
                    "stage": capture["verifier_stage"],
                    "capture_as_of_ts": pd.Timestamp(capture["as_of_ts"]),
                    "capture_as_of_cursor": str(capture["as_of_cursor"]),
                    "window_as_of_cursor": (
                        str(window["as_of_cursor"])
                        if pd.notna(window.get("as_of_cursor"))
                        else None
                    ),
                    "window_kind": window["window_kind"],
                    "b0_bar_id": window["b0_bar_id"],
                    "start_ts": window["start_ts"],
                    "last_included_bar_id": window["last_included_bar_id"],
                    "end_ts": window["end_ts"],
                    "expected_orientation": window.get("expected_orientation"),
                    "valid": window.get("valid"),
                    "missing_reason": window.get("missing_reason"),
                    **{column: window.get(column) for column in metric_columns},
                }
            )
    return pd.DataFrame(rows)


def _pool_evidence(
    ctx: ReplayContext,
    candidate: dict[str, Any],
    stage_captures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    states = _table(ctx.pair, ContextRecordTable.CONTEXT_STATE).set_index("context_state_id")
    sweeps = _table(ctx.pair, ContextRecordTable.EQUAL_LEVEL_SWEEP_LINK)
    lifecycle = _table(ctx.pair, ContextRecordTable.EQUAL_LEVEL_POOL_LIFECYCLE)
    members = _table(ctx.pair, ContextRecordTable.EQUAL_LEVEL_POOL_MEMBER)

    setup_id = str(candidate["setup_id"])
    pool_ids: set[str] = set()
    selected_link: str | None = None
    link_ids: set[str] = set()
    for capture in stage_captures.to_dict("records"):
        state_id = capture.get("context_state_id")
        if state_id is not None and not pd.isna(state_id) and str(state_id) in states.index:
            state = states.loc[str(state_id)]
            for prefix in (
                "nearest_context_nearest_eqh",
                "nearest_context_nearest_eql",
                "nearest_context_nearest_thesis_supporting",
                "nearest_context_nearest_thesis_opposing",
            ):
                pool_id = state.get(f"{prefix}_pool_id")
                if pool_id is not None and not pd.isna(pool_id):
                    pool_ids.add(str(pool_id))
        if capture["verifier_stage"] == "entry":
            selected = capture.get("selected_opposing_leg_sweep_link_id")
            if selected is not None and not pd.isna(selected):
                selected_link = str(selected)
            raw_links = capture.get("opposing_leg_sweep_link_ids")
            if raw_links is not None and not (
                isinstance(raw_links, float) and pd.isna(raw_links)
            ):
                link_ids.update(str(item) for item in list(raw_links))

    setup_sweeps = sweeps[
        (sweeps["setup_id"].astype(str) == setup_id)
        | (sweeps["sweep_link_id"].astype(str).isin(link_ids))
    ].copy()
    pool_ids.update(setup_sweeps["pool_id"].astype(str))
    setup_sweeps["selected"] = setup_sweeps["sweep_link_id"].astype(str) == (
        selected_link or ""
    )

    pool_events = lifecycle[lifecycle["pool_pool_id"].astype(str).isin(pool_ids)].copy()
    pool_events = pool_events.sort_values("as_of_cursor", kind="mergesort")
    pool_members = members[members["pool_id"].astype(str).isin(pool_ids)].copy()
    return pool_events, pool_members, setup_sweeps, pool_ids


def _model_evidence(ctx: ReplayContext, candidate_id: str) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for tier in MODEL_TIERS:
        run = ctx.tier_runs.get(tier)
        if run is None:
            payload[tier] = {"status": "run_not_available"}
            continue
        predictions = run.predictions
        rows = (
            predictions[predictions["candidate_id"].astype(str) == candidate_id]
            if not predictions.empty
            else predictions
        )
        if rows is None or len(rows) == 0:
            payload[tier] = {
                "status": "candidate_not_in_oos_folds",
                "run_id": run.result.run_id,
            }
            continue
        row = rows.iloc[0]
        fold_index = int(row["fold_index"])
        training_window_end = None
        try:
            fold = run.result.folds[fold_index]
            train_days = list(fold.train_days)
            training_window_end = max(train_days) if train_days else None
        except (IndexError, AttributeError, TypeError):
            training_window_end = None
        payload[tier] = {
            "status": "prediction_available",
            "run_id": run.result.run_id,
            "fold_index": fold_index,
            "probability": float(row["probability"]),
            "target": str(row["target"]) if "target" in row else None,
            "training_prevalence": (
                float(row["training_prevalence"]) if "training_prevalence" in row else None
            ),
            "training_window_end": training_window_end,
            "note": "counterfactual development evidence — not a live signal",
        }
        if tier in ctx.tier_notes:
            payload[tier]["tier_note"] = ctx.tier_notes[tier]
    return payload


def session_scheme_windows() -> dict[str, dict[str, Any]]:
    """Session window geometry from the exact strategy_core schemes."""
    from strategy_core.constants import IFVG_DOC_SESSION_SCHEME, RESEARCH_SESSION_SCHEME

    result: dict[str, dict[str, Any]] = {}
    for name, scheme in (("doc", IFVG_DOC_SESSION_SCHEME), ("engine", RESEARCH_SESSION_SCHEME)):
        result[name] = {
            "timezone": scheme.timezone,
            "trading_day_boundary": scheme.trading_day_boundary.isoformat(),
            "sessions": {
                session: {
                    "start": window.start.isoformat(),
                    "end": window.end.isoformat(),
                    "crosses_midnight": bool(window.crosses_midnight),
                }
                for session, window in sorted(scheme.sessions.items())
            },
        }
    return result


def candidate_evidence(
    ctx: ReplayContext,
    candidate_id: str,
    *,
    mode: Literal["full_audit", "point_in_time"] = "full_audit",
    stage: FsmStage | str | None = None,
) -> CandidateEvidence:
    """Assemble (and, in point-in-time mode, gate) all evidence for a candidate."""
    if mode not in ("full_audit", "point_in_time"):
        raise ValueError("mode must be full_audit or point_in_time")
    if mode == "point_in_time":
        if stage is None:
            raise MissingEvidenceError("stage_gate", "point-in-time mode requires a stage")
        stage = FsmStage(stage)
    elif stage is not None:
        raise ValueError("stage is only meaningful in point_in_time mode")

    candidates = _table(ctx.pair, RecordTable.ENTRY_CANDIDATE)
    matches = candidates[candidates["candidate_id"].astype(str) == candidate_id]
    if len(matches) != 1:
        raise MissingEvidenceError("candidate", candidate_id, candidate_id=candidate_id)
    candidate = matches.iloc[0].to_dict()

    dossiers = _table(ctx.pair, RecordTable.GEOMETRY_DOSSIER)
    dossier_rows = dossiers[dossiers["candidate_id"].astype(str) == candidate_id]
    if len(dossier_rows) != 1:
        raise MissingEvidenceError("geometry_dossier", candidate_id, candidate_id=candidate_id)
    dossier = dossier_rows.iloc[0]

    ranges = ctx.replay.candidate_ranges
    range_rows = ranges[ranges["candidate_id"].astype(str) == candidate_id]
    if len(range_rows) != 1:
        raise MissingEvidenceError("candidate_bar_range", candidate_id, candidate_id=candidate_id)
    range_row = range_rows.iloc[0]

    lifecycle_all = _table(ctx.pair, RecordTable.SETUP_LIFECYCLE)
    lifecycle = (
        lifecycle_all[lifecycle_all["setup_id"].astype(str) == str(candidate["setup_id"])]
        .sort_values("trace_ordinal", kind="mergesort")
        .reset_index(drop=True)
    )
    executed = _table(ctx.pair, RecordTable.EXECUTED_TRADE)
    trade_rows = executed[executed["candidate_id"].astype(str) == candidate_id]
    trade = trade_rows.iloc[0] if len(trade_rows) == 1 else None

    labels = _table(ctx.pair, RecordTable.CANDIDATE_LABEL)
    label_rows = labels[labels["candidate_id"].astype(str) == candidate_id]

    gates, ungateable = _build_stage_gates(ctx, candidate, dossier, lifecycle, trade, range_row)
    gating_report: dict[str, Any] = {
        "policy": "stage_gate_ordinal_cursor_ts_v1",
        "ungateable": list(ungateable),
        "hidden": [],
    }

    gate: StageGate | None = None
    if mode == "point_in_time":
        gate = gates.get(stage.value)
        if gate is None:
            raise MissingEvidenceError(
                "stage_gate",
                f"stage {stage.value} cannot be gated for this candidate",
                candidate_id=candidate_id,
            )

    def _record_hidden(kind: str, object_id: str, reason: str) -> None:
        gating_report["hidden"].append(
            {"kind": kind, "object_id": object_id, "reason": reason}
        )

    def _visible(kind: str, object_id: str, keys: _GateKeys) -> bool:
        if gate is None:
            return True
        verdict = _gate_visible(gate, keys)
        if verdict is None:
            gating_report["ungateable"].append(
                {"kind": kind, "object_id": object_id, "missing": "no orderable gate key"}
            )
            return False
        if not verdict:
            _record_hidden(kind, object_id, f"after stage {gate.stage.value}")
        return verdict

    stage_position = _stage_index(stage.value) if gate is not None else len(STAGE_ORDER)

    # The candidate's trigger cursor is the persisted confirmation cursor of
    # its trigger evidence (the entry FVG) — an exact ordering key, so the
    # entry zone does not fall to a timestamp tie at the entry gate.
    trigger_evidence_id = (
        str(candidate["trigger_evidence_id"])
        if pd.notna(candidate.get("trigger_evidence_id"))
        else None
    )
    trigger_cursor = (
        str(candidate["trigger_cursor"]) if pd.notna(candidate.get("trigger_cursor")) else None
    )

    def _zone_cursor(zone: ZoneEvidence) -> str | None:
        if trigger_evidence_id is not None and zone.fvg_id == trigger_evidence_id:
            return trigger_cursor
        return None

    zones = tuple(
        zone
        for zone in _zone_evidence(dossier)
        if _visible(
            "zone",
            f"{zone.role}:{zone.fvg_id}",
            _GateKeys(ts=zone.confirmed_ts_utc, cursor=_zone_cursor(zone), ordinal=None),
        )
    )
    transition_bars = {
        name: snapshot
        for name, snapshot in _transition_bar_snapshots(dossier).items()
        if _visible(
            "transition_bar",
            f"{name}:{snapshot['bar_id']}",
            _GateKeys(
                ts=snapshot["logical_close_ts_utc"],
                cursor=snapshot["cursor"],
                ordinal=None,
            ),
        )
    }

    if gate is not None:
        keep = lifecycle["trace_ordinal"].astype(int) <= (
            gate.trace_ordinal
            if gate.trace_ordinal is not None
            else lifecycle["trace_ordinal"].astype(int).max()
        )
        if gate.trace_ordinal is None:
            keep = pd.Series(
                [
                    _visible(
                        "lifecycle_event",
                        str(row["lifecycle_event_id"]),
                        _GateKeys(
                            ts=pd.Timestamp(row["envelope_ts_utc"]),
                            cursor=str(row["event_cursor"]),
                            ordinal=None,
                        ),
                    )
                    for row in lifecycle.to_dict("records")
                ],
                index=lifecycle.index,
            )
        lifecycle_visible = lifecycle[keep].reset_index(drop=True)
    else:
        lifecycle_visible = lifecycle

    risk = {
        "direction": str(candidate["direction"]),
        "entry_ticks": int(dossier["geometry_entry_ticks"]),
        "stop_ticks": int(dossier["geometry_stop_ticks"]),
        "target_ticks": int(dossier["geometry_target_ticks"]),
        "manipulation_swing_ticks": (
            int(dossier["geometry_manipulation_swing_ticks"])
            if pd.notna(dossier.get("geometry_manipulation_swing_ticks"))
            else None
        ),
        "sl_buffer_ticks": (
            int(dossier["geometry_sl_buffer_ticks"])
            if pd.notna(dossier.get("geometry_sl_buffer_ticks"))
            else None
        ),
        "risk_ticks": (
            int(candidate["risk_ticks"]) if pd.notna(candidate.get("risk_ticks")) else None
        ),
    }
    show_entry_layer = stage_position >= _stage_index(FsmStage.ENTRY.value)
    show_resolution_layer = stage_position >= _stage_index(FsmStage.RESOLUTION.value)
    if not show_entry_layer:
        _record_hidden("risk", candidate_id, "hidden before entry stage")
        risk = {"direction": risk["direction"]}

    execution = None
    if trade is not None:
        execution = {
            "trade_id": str(trade["trade_id"]),
            "decision_id": str(trade["decision_id"]),
            "entry_ts_utc": pd.Timestamp(trade["entry_ts_utc"]),
            "entry_cursor": str(trade["entry_cursor"]),
            "entry_ticks": int(trade["entry_ticks"]),
            "stop_ticks": int(trade["stop_ticks"]),
            "target_ticks": int(trade["target_ticks"]),
            "entry_session": str(trade["entry_session"]),
            "is_warmup": bool(trade["is_warmup"]),
        }
        if show_resolution_layer:
            execution.update(
                {
                    "resolution": str(trade["resolution"]),
                    "resolution_ts_utc": pd.Timestamp(trade["resolution_ts_utc"]),
                    "resolution_cursor": str(trade["resolution_cursor"]),
                    "status": str(trade["status"]),
                    "bars_after_entry_to_resolution": int(
                        trade["bars_after_entry_to_resolution"]
                    ),
                    "mfe_ticks": int(trade["mfe_ticks"]),
                    "mae_ticks": int(trade["mae_ticks"]),
                    "realized_ticks": int(trade["realized_ticks"]),
                    "realized_r": float(trade["realized_r"]),
                }
            )
        else:
            _record_hidden("execution_outcome", str(trade["trade_id"]), "hidden before resolution")
        if not show_entry_layer:
            _record_hidden("execution", str(trade["trade_id"]), "hidden before entry stage")
            execution = None

    if show_resolution_layer:
        counterfactual = tuple(
            {
                "label_family": str(row["label_family"]),
                "r_multiple": float(row["r_multiple"]) if pd.notna(row.get("r_multiple")) else None,
                "label": str(row["label"]),
                "censored": bool(row["censored"]),
                "censor_reason": (
                    str(row["censor_reason"]) if pd.notna(row.get("censor_reason")) else None
                ),
                "mfe_r": float(row["mfe_r"]) if pd.notna(row.get("mfe_r")) else None,
                "mae_r": float(row["mae_r"]) if pd.notna(row.get("mae_r")) else None,
                "resolution_bar_id": (
                    str(row["resolution_bar_id"])
                    if pd.notna(row.get("resolution_bar_id"))
                    else None
                ),
                "target_ticks": (
                    int(row["target_ticks"]) if pd.notna(row.get("target_ticks")) else None
                ),
                "entry_bar_id": (
                    str(row["entry_bar_id"]) if pd.notna(row.get("entry_bar_id")) else None
                ),
            }
            for row in label_rows.to_dict("records")
        )
    else:
        counterfactual = ()
        if len(label_rows):
            _record_hidden("counterfactual_labels", candidate_id, "hidden before resolution")

    stage_captures = _stage_captures(ctx, candidate)
    if gate is not None and not stage_captures.empty:
        capture_stage_limit = {
            "tap": FsmStage.TAP,
            "lock": FsmStage.LOCK,
            "inversion": FsmStage.INVERSION,
            "entry": FsmStage.ENTRY,
            "decision": FsmStage.ENTRY,
            "trade": FsmStage.ENTRY,
        }
        keep_stages = [
            row["verifier_stage"]
            for row in stage_captures.to_dict("records")
            if _stage_index(capture_stage_limit[row["verifier_stage"]].value) <= stage_position
        ]
        dropped = set(stage_captures["verifier_stage"]) - set(keep_stages)
        for name in sorted(dropped):
            _record_hidden("stage_capture", name, f"after stage {gate.stage.value}")
        stage_captures = stage_captures[
            stage_captures["verifier_stage"].isin(keep_stages)
        ].reset_index(drop=True)

    structure, structure_summary = _structure_evidence(ctx, stage_captures)
    if gate is not None and not structure.empty:
        keep_rows = []
        for _, row in structure.iterrows():
            break_ts = row.get("break_ts")
            if (
                break_ts is not None
                and not pd.isna(break_ts)
                and not _visible(
                    "structure_break",
                    str(row["structure_state_id"]),
                    _GateKeys(ts=pd.Timestamp(break_ts), cursor=None, ordinal=None),
                )
            ):
                row = row.copy()
                for column in (
                    "last_break_type",
                    "last_break_direction",
                    "break_bar_id",
                    "break_ts",
                    "broken_swing_id",
                ):
                    row[column] = None
            keep_rows.append(row)
        structure = pd.DataFrame(keep_rows).reset_index(drop=True)

    displacement = _displacement_evidence(ctx, stage_captures)
    if gate is not None and not displacement.empty:
        keep = [
            _visible(
                "displacement_window",
                str(row["displacement_window_id"]),
                _GateKeys(
                    ts=(
                        pd.Timestamp(row["end_ts"])
                        if row.get("end_ts") is not None and not pd.isna(row["end_ts"])
                        else None
                    ),
                    cursor=row.get("window_as_of_cursor"),
                    ordinal=None,
                ),
            )
            and _visible(
                "displacement_window_capture",
                str(row["displacement_window_id"]),
                _GateKeys(
                    ts=pd.Timestamp(row["capture_as_of_ts"]),
                    cursor=row.get("capture_as_of_cursor"),
                    ordinal=None,
                ),
            )
            for row in displacement.to_dict("records")
        ]
        displacement = displacement[pd.Series(keep, index=displacement.index)].reset_index(
            drop=True
        )

    pool_events, pool_members, sweep_links, _pool_ids = _pool_evidence(
        ctx, candidate, stage_captures
    )
    if gate is not None:
        if not pool_events.empty:
            keep = [
                _visible(
                    "pool_event",
                    str(row["lifecycle_event_id"]),
                    _GateKeys(
                        ts=pd.Timestamp(row["as_of_ts"]),
                        cursor=str(row["as_of_cursor"]),
                        ordinal=None,
                    ),
                )
                for row in pool_events.to_dict("records")
            ]
            pool_events = pool_events[pd.Series(keep, index=pool_events.index)]
        if not pool_events.empty:
            confirmed = [
                _visible(
                    "pool",
                    str(row["pool_pool_id"]),
                    _GateKeys(
                        ts=(
                            pd.Timestamp(row["pool_confirmation_ts"])
                            if pd.notna(row.get("pool_confirmation_ts"))
                            else None
                        ),
                        cursor=None,
                        ordinal=None,
                    ),
                )
                for row in pool_events.to_dict("records")
            ]
            pool_events = pool_events[pd.Series(confirmed, index=pool_events.index)]
        if not pool_members.empty:
            keep = [
                _visible(
                    "pool_member",
                    str(row["pool_member_id"]),
                    _GateKeys(
                        ts=(
                            pd.Timestamp(row["swing_confirmation_ts"])
                            if pd.notna(row.get("swing_confirmation_ts"))
                            else None
                        ),
                        cursor=(
                            str(row["swing_availability_cursor"])
                            if pd.notna(row.get("swing_availability_cursor"))
                            else None
                        ),
                        ordinal=None,
                    ),
                )
                for row in pool_members.to_dict("records")
            ]
            pool_members = pool_members[pd.Series(keep, index=pool_members.index)]
        if not sweep_links.empty:
            keep = [
                _visible(
                    "sweep_link",
                    str(row["sweep_link_id"]),
                    _GateKeys(
                        ts=(
                            pd.Timestamp(row["sweep_ts"])
                            if pd.notna(row.get("sweep_ts"))
                            else None
                        ),
                        cursor=(
                            str(row["sweep_cursor"])
                            if pd.notna(row.get("sweep_cursor"))
                            else None
                        ),
                        ordinal=None,
                    ),
                )
                for row in sweep_links.to_dict("records")
            ]
            sweep_links = sweep_links[pd.Series(keep, index=sweep_links.index)].copy()
            for column in ("reclaim_bar_id", "reclaim_ts"):
                if column in sweep_links.columns:
                    hide = [
                        pd.notna(row.get("reclaim_ts"))
                        and not _visible(
                            "sweep_reclaim",
                            str(row["sweep_link_id"]),
                            _GateKeys(
                                ts=pd.Timestamp(row["reclaim_ts"]), cursor=None, ordinal=None
                            ),
                        )
                        for row in sweep_links.to_dict("records")
                    ]
                    sweep_links.loc[pd.Series(hide, index=sweep_links.index), column] = None

    # Collapse the pool event stream to per-pool state as of the gate (or latest).
    if not pool_events.empty:
        pools = (
            pool_events.sort_values("as_of_cursor", kind="mergesort")
            .groupby("pool_pool_id", sort=False)
            .tail(1)
            .reset_index(drop=True)
        )
        if gate is not None:
            state_columns = (
                ("pool_swept", "pool_sweep_ts"),
                ("pool_reclaimed", "pool_reclaim_ts"),
            )
            for column, ts_column in state_columns:
                adjusted = []
                for row in pools.to_dict("records"):
                    value = row.get(column)
                    ts_value = row.get(ts_column)
                    if (
                        value
                        and pd.notna(ts_value)
                        and not _visible(
                            column,
                            str(row["pool_pool_id"]),
                            _GateKeys(ts=pd.Timestamp(ts_value), cursor=None, ordinal=None),
                        )
                    ):
                        value = False
                    adjusted.append(value)
                pools[column] = adjusted
    else:
        pools = pool_events

    model = _model_evidence(ctx, candidate_id)
    if gate is not None and not show_entry_layer:
        for tier in list(model):
            if model[tier].get("status") == "prediction_available":
                model[tier] = {
                    "status": "hidden_until_entry",
                    "run_id": model[tier].get("run_id"),
                }
                _record_hidden("model_prediction", tier, "hidden before entry stage")

    identity = {
        "strategy_id": str(candidate["strategy_id"]),
        "strategy_version": str(candidate["strategy_version"]),
        "profile_name": str(candidate["profile_name"]),
        "profile_hash": str(candidate["profile_hash"]),
        "section_config_hash": str(candidate["section_config_hash"]),
        "label_family": str(candidate["label_family"]),
        "entry_family": str(candidate["entry_family"]),
        "qualification_mode": str(candidate["qualification_mode"]),
    }
    lineage = {
        "setup_id": str(candidate["setup_id"]),
        "candidate_id": candidate_id,
        "decision_id": range_row["decision_id"],
        "trade_id": range_row["trade_id"],
        "geometry_evidence_id": range_row["geometry_evidence_id"],
        "geometry_evidence_cursor": range_row["geometry_evidence_cursor"],
        "context_capture_id": range_row["context_capture_id"],
        "v2_artifact_id": ctx.pair_ref.v2_dataset_id,
        "v3_artifact_id": ctx.pair_ref.v3_dataset_id,
        "replay_chart_artifact_id": ctx.replay.artifact_id,
    }
    sessions = {
        "entry_session": str(candidate["entry_session"]),
        "in_engine_session": (
            bool(candidate["in_engine_session"])
            if pd.notna(candidate.get("in_engine_session"))
            else None
        ),
        "in_doc_session": (
            bool(candidate["in_doc_session"])
            if pd.notna(candidate.get("in_doc_session"))
            else None
        ),
        "schemes": session_scheme_windows(),
    }

    return CandidateEvidence(
        candidate_id=candidate_id,
        mode=mode,
        stage=stage.value if isinstance(stage, FsmStage) else None,
        identity=identity,
        lineage=lineage,
        range_row=range_row.to_dict(),
        zones=zones,
        transition_bars=transition_bars,
        lifecycle=lifecycle_visible,
        risk=risk,
        execution=execution,
        counterfactual_labels=counterfactual,
        structure=structure,
        structure_stage_summary=structure_summary,
        displacement=displacement,
        pools=pools,
        pool_members=pool_members,
        sweep_links=sweep_links,
        sessions=sessions,
        model=model,
        stage_gates=gates,
        gating_report=gating_report,
        anchor_240m_status=ctx.replay.anchor_240m_status,
    )


# ---------------------------------------------------------------------------
# Bars and ranges


def _authorize_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:
    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts > end_ts:
        raise ReplayAuthorizationError("bar range bounds are invalid")
    if end_ts > _CUTOFF:
        raise ReplayAuthorizationError(
            "bar range extends past the authorized development cutoff "
            f"({AUTHORIZED_CUTOFF_UTC}); protected and sealed data are refused"
        )


def bars_for_pane(
    ctx: ReplayContext,
    *,
    timeframe_seconds: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Authorized, bounded bar slice for one chart pane."""
    start_ts = pd.Timestamp(start_ts)
    end_ts = pd.Timestamp(end_ts)
    _authorize_range(start_ts, end_ts)
    if timeframe_seconds == 60:
        bars = ctx.bars_1m
        close = bars["close_ts_utc"]
        mask = (close >= start_ts) & ((close - pd.Timedelta(seconds=60)) <= end_ts)
        selected = bars[mask]
        if len(selected) > MAX_1M_BARS_PER_RENDER:
            raise RangeTooLargeError(
                f"requested 1m range holds {len(selected)} bars; the render limit is "
                f"{MAX_1M_BARS_PER_RENDER} — narrow the range"
            )
        return selected.reset_index(drop=True).copy()
    if timeframe_seconds not in REPLAY_TIMEFRAMES_SECONDS:
        raise MissingEvidenceError("bars", f"unsupported pane timeframe {timeframe_seconds}")
    bars = ctx.replay.bars_tf
    frame = bars[bars["timeframe_seconds"] == timeframe_seconds]
    opens = pd.to_datetime(frame["logical_open_ts_utc"], utc=True)
    closes = pd.to_datetime(frame["logical_close_ts_utc"], utc=True)
    mask = (closes >= start_ts) & (opens <= end_ts)
    return frame[mask].reset_index(drop=True).copy()


def chart_range(
    ctx: ReplayContext,
    evidence: CandidateEvidence,
    range_kind: Literal["setup", "trade", "formation", "custom"],
    *,
    custom_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Deterministic chart range bounds; PIT gates truncate the end."""
    row = evidence.range_row
    bar_close = dict(
        zip(ctx.bars_1m["bar_id"].astype(str), ctx.bars_1m["close_ts_utc"], strict=True)
    )
    display_end = pd.Timestamp(row["display_end_ts"])
    if range_kind == "setup":
        first = row.get("first_bar_id_1m")
        last = row.get("last_bar_id_1m")
        if first is None or last is None:
            raise MissingEvidenceError("chart_range", "setup range has no bar span")
        start_ts, end_ts = bar_close[str(first)], bar_close[str(last)]
    elif range_kind == "trade":
        anchor = pd.Timestamp(row["entry_anchor_ts"])
        pad = pd.Timedelta(minutes=30)
        start_ts, end_ts = anchor - pad, display_end + pd.Timedelta(minutes=15)
    elif range_kind == "formation":
        formation = row.get("formation_start_ts")
        if formation is None or pd.isna(formation):
            raise MissingEvidenceError("chart_range", "formation start is not persisted")
        start_ts, end_ts = pd.Timestamp(formation), display_end + pd.Timedelta(minutes=15)
    elif range_kind == "custom":
        if custom_bounds is None:
            raise MissingEvidenceError("chart_range", "custom range requires explicit bounds")
        start_ts, end_ts = pd.Timestamp(custom_bounds[0]), pd.Timestamp(custom_bounds[1])
    else:
        raise ValueError(f"unknown range kind: {range_kind}")

    if evidence.stage is not None:
        gate = evidence.stage_gates.get(evidence.stage)
        if gate is not None:
            end_ts = min(end_ts, gate.ts_utc)
    end_ts = min(end_ts, _CUTOFF)
    _authorize_range(start_ts, end_ts)
    return start_ts, end_ts
