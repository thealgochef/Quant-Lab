"""Immutable ``ifvg_replay_chart_v1`` companion artifact: build, verify, catalog.

The replay-chart artifact carries everything a visual trade verifier needs
beyond the verified v2/v3 pair itself:

* ``bars_tf.parquet`` — higher-timeframe bars resampled deterministically from
  the *verified* v2 1-minute label source (``trading_day_18et_elapsed_v1``
  rule); the identity chain therefore stays rooted at the v2 manifest.
* ``candidate_bar_range.parquet`` — one row per candidate with deterministic
  chart-range anchors, separated end-time semantics, and exact evidence refs.

Per-day engine tbars files are used only as a build-time corroboration oracle;
their hashes live in ``corroboration_report.json`` which is deliberately kept
OUTSIDE the canonical manifest payload so external oracle bytes can never
alter the artifact identity.  A corroboration mismatch still refuses to
publish.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .artifact_io import VerifiedIfvgPair, load_verified_label_source_bars
from .config import DEFAULT_DATA_DIR, IfvgCaptureConfig
from .context_contracts import ContextRecordTable
from .contracts import RecordTable
from .development_access import DevelopmentAccessAudit, DevelopmentDataAccess
from .manifest import canonical_sha256, file_sha256

__all__ = [
    "REPLAY_CHART_STORE",
    "REPLAY_CHART_CATALOG",
    "REPLAY_TIMEFRAMES_SECONDS",
    "RESAMPLE_RULE_ID",
    "ANCHOR_240M_STATUS",
    "AUTHORIZED_CUTOFF_UTC",
    "PRIMARY_LABEL_FAMILY",
    "ReplayChartStoreError",
    "ArtifactPairRef",
    "VerifiedReplayChartArtifact",
    "trading_day_open_utc",
    "resample_label_bars",
    "replay_chart_effective_config",
    "replay_chart_identity",
    "build_candidate_bar_ranges",
    "build_replay_chart_artifact",
    "load_verified_replay_chart_artifact",
    "read_replay_chart_catalog",
    "find_replay_artifact",
    "update_replay_chart_catalog",
    "REPLAY_CHART_SCHEMA_VERSION_V2",
    "SETUP_RANGE_POLICY_V2",
    "STAGE_GATING_POLICY_V2",
    "VerifierBundleRef",
    "replay_chart_effective_config_v2",
    "build_setup_bar_ranges",
    "build_replay_chart_artifact_v2",
    "load_verified_replay_chart_artifact_v2",
    "load_setup_ranges_v2",
    "find_replay_artifact_v2",
    "update_replay_chart_catalog_v2",
]

REPLAY_CHART_STORE = Path("data/ifvg_datasets/replay_chart/v1")
REPLAY_CHART_CATALOG = Path("data/ifvg_datasets/replay_chart_catalog_v1.json")

REPLAY_CHART_SCHEMA_VERSION = 1
#: Setup-aware generation. The v1 artifact is retained and still loads; new
#: builds pin the fsm-audit artifact and add ``setup_bar_range.parquet``.
REPLAY_CHART_SCHEMA_VERSION_V2 = 2
SETUP_RANGE_POLICY_V2 = "setup_range_v2"
STAGE_GATING_POLICY_V2 = "stage_gate_ordinal_cursor_ts_v2"
BAR_SCHEMA_VERSION = 1
RESAMPLE_RULE_ID = "trading_day_18et_elapsed_v1"
REPLAY_TIMEFRAMES_SECONDS = (180, 300, 600, 900, 1800, 3600, 14400)
CANDIDATE_RANGE_POLICY = "candidate_range_v1"
STAGE_GATING_POLICY = "stage_gate_ordinal_cursor_ts_v1"
PARTIAL_BAR_POLICY = "final_window_clipped_at_last_observed_1m_close_v1"
RANGE_PAD_1M_BARS = 15
ANCHOR_240M_STATUS = "experimental_q40_open"
AUTHORIZED_CUTOFF_UTC = "2026-06-10T21:00:00Z"
# End-time semantics use the static 1R label family as the counterfactual path.
PRIMARY_LABEL_FAMILY = "static_r_1_next_bar_stop_first_v1"

_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ET = ZoneInfo("America/New_York")

DISPLAY_END_SOURCES = (
    "trade_resolution",
    "candidate_label_resolution",
    "setup_invalidation",
    "setup_end",
    "authorized_cutoff_censor",
)


class ReplayChartStoreError(ValueError):
    """The replay-chart artifact failed identity, policy, or content checks."""


@dataclass(frozen=True)
class ArtifactPairRef:
    """Exact v2/v3 pair identity.  A profile name alone is never a data identity."""

    profile_name: str
    v2_dataset_id: str
    v2_manifest_hash: str
    v3_dataset_id: str
    v3_manifest_hash: str

    def __post_init__(self) -> None:
        for name in ("v2_dataset_id", "v2_manifest_hash", "v3_dataset_id", "v3_manifest_hash"):
            if not _FULL_SHA256.fullmatch(getattr(self, name)):
                raise ReplayChartStoreError(f"pair reference {name} must be a full SHA-256")
        if not self.profile_name:
            raise ReplayChartStoreError("pair reference requires a profile name")

    def as_dict(self) -> dict[str, str]:
        return dict(sorted(asdict(self).items()))

    @classmethod
    def from_verified_pair(cls, pair: VerifiedIfvgPair) -> ArtifactPairRef:
        candidates = pair.v2.tables[RecordTable.ENTRY_CANDIDATE]
        profiles = sorted(candidates["profile_name"].dropna().astype(str).unique())
        if len(profiles) != 1:
            raise ReplayChartStoreError("verified pair does not carry exactly one profile name")
        return cls(
            profile_name=profiles[0],
            v2_dataset_id=pair.reference.v2.artifact_id,
            v2_manifest_hash=pair.reference.v2.manifest_payload_sha256,
            v3_dataset_id=pair.reference.v3.artifact_id,
            v3_manifest_hash=pair.reference.v3.manifest_payload_sha256,
        )


@dataclass(frozen=True, slots=True)
class VerifiedReplayChartArtifact:
    artifact_id: str
    directory: Path
    manifest: dict[str, Any]
    source_pair: ArtifactPairRef
    bars_tf: pd.DataFrame
    candidate_ranges: pd.DataFrame
    anchor_240m_status: str


def trading_day_open_utc(trading_day: str) -> pd.Timestamp:
    """18:00 ET on the calendar day before ``trading_day``, DST-aware."""
    day = date.fromisoformat(trading_day)
    opened = datetime.combine(day - timedelta(days=1), time(18, 0), tzinfo=_ET)
    return pd.Timestamp(opened).tz_convert("UTC")


def _parse_bar_days(bar_ids: pd.Series, *, expected_tf: int) -> pd.Series:
    parts = bar_ids.astype(str).str.split(":", expand=True)
    if parts.shape[1] != 3:
        raise ReplayChartStoreError("bar IDs must have the <tf>s:<day>:<index> shape")
    tf = parts[0].str.removesuffix("s")
    if not (tf == str(expected_tf)).all():
        raise ReplayChartStoreError(f"bar IDs are not uniformly {expected_tf}s bars")
    return parts[1]


def resample_label_bars(bars_1m: pd.DataFrame, timeframe_seconds: int) -> pd.DataFrame:
    """Deterministic ``trading_day_18et_elapsed_v1`` aggregation of verified 1m bars.

    Windows are elapsed-time buckets anchored at the 18:00 ET trading-day open,
    NOT bar-count buckets — naive ``index // k`` grouping diverges from the
    engine on early-close days.
    """
    if timeframe_seconds not in REPLAY_TIMEFRAMES_SECONDS:
        raise ReplayChartStoreError(f"unsupported replay timeframe: {timeframe_seconds}")
    required = {"bar_id", "close_ts_utc", "open_ticks", "high_ticks", "low_ticks", "close_ticks"}
    missing = sorted(required - set(bars_1m.columns))
    if missing:
        raise ReplayChartStoreError(f"1m bars are missing columns {missing}")

    work = bars_1m.copy()
    work["trading_day"] = _parse_bar_days(work["bar_id"], expected_tf=60)
    close_ts = pd.to_datetime(work["close_ts_utc"], utc=True, errors="raise")
    open_ts = close_ts - pd.Timedelta(seconds=60)
    day_open = work["trading_day"].map(
        {day: trading_day_open_utc(day) for day in work["trading_day"].unique()}
    )
    elapsed = (open_ts - day_open).dt.total_seconds()
    if (elapsed < 0).any() or (elapsed % 60 != 0).any():
        raise ReplayChartStoreError("1m bars are not aligned to their trading-day open")
    work["_window"] = (elapsed // timeframe_seconds).astype(int)
    work["_open_ts"] = open_ts
    work["_close_ts"] = close_ts
    work = work.sort_values(["trading_day", "_close_ts"], kind="mergesort")

    grouped = work.groupby(["trading_day", "_window"], sort=True)
    frame = grouped.agg(
        open_ticks=("open_ticks", "first"),
        high_ticks=("high_ticks", "max"),
        low_ticks=("low_ticks", "min"),
        close_ticks=("close_ticks", "last"),
        volume=("volume", "sum") if "volume" in work.columns else ("open_ticks", "size"),
        trade_count=(
            ("trade_count", "sum") if "trade_count" in work.columns else ("open_ticks", "size")
        ),
        observed_1m_count=("bar_id", "size"),
        _last_close=("_close_ts", "max"),
    ).reset_index()

    frame = frame.rename(columns={"_window": "bar_index"})
    frame["timeframe_seconds"] = int(timeframe_seconds)
    frame["expected_1m_count"] = int(timeframe_seconds // 60)
    frame["bar_id"] = (
        str(timeframe_seconds)
        + "s:"
        + frame["trading_day"]
        + ":"
        + frame["bar_index"].astype(str)
    )
    day_open_out = frame["trading_day"].map(
        {day: trading_day_open_utc(day) for day in frame["trading_day"].unique()}
    )
    frame["logical_open_ts_utc"] = day_open_out + pd.to_timedelta(
        frame["bar_index"] * timeframe_seconds, unit="s"
    )
    logical_full_close = frame["logical_open_ts_utc"] + pd.Timedelta(seconds=timeframe_seconds)
    last_window = frame["bar_index"] == frame.groupby("trading_day")["bar_index"].transform("max")
    frame["is_final_partial"] = last_window & (frame["_last_close"] < logical_full_close)
    # A finalized-at-day-end bar closes when its last 1m source bar closed.
    frame["logical_close_ts_utc"] = logical_full_close.where(
        ~frame["is_final_partial"], frame["_last_close"]
    )

    columns = [
        "timeframe_seconds",
        "trading_day",
        "bar_index",
        "bar_id",
        "logical_open_ts_utc",
        "logical_close_ts_utc",
        "open_ticks",
        "high_ticks",
        "low_ticks",
        "close_ticks",
        "volume",
        "trade_count",
        "observed_1m_count",
        "expected_1m_count",
        "is_final_partial",
    ]
    return frame[columns].reset_index(drop=True)


def replay_chart_effective_config(
    pair_ref: ArtifactPairRef,
    *,
    label_source_sha256: str,
) -> dict[str, Any]:
    """The complete resolved policy set; every load-bearing policy enters identity."""
    return {
        "artifact_kind": "ifvg_replay_chart_v1",
        "replay_chart_schema_version": REPLAY_CHART_SCHEMA_VERSION,
        "bar_schema_version": BAR_SCHEMA_VERSION,
        "source_pair": pair_ref.as_dict(),
        "source_label_table_sha256": label_source_sha256,
        "resample_rule": RESAMPLE_RULE_ID,
        "timeframes_seconds": list(REPLAY_TIMEFRAMES_SECONDS),
        "development_cutoff_utc": AUTHORIZED_CUTOFF_UTC,
        "candidate_range_policy": CANDIDATE_RANGE_POLICY,
        "primary_label_family": PRIMARY_LABEL_FAMILY,
        "stage_gating_policy": STAGE_GATING_POLICY,
        "partial_bar_policy": PARTIAL_BAR_POLICY,
        "range_pad_1m_bars": RANGE_PAD_1M_BARS,
        "anchor_240m_status": ANCHOR_240M_STATUS,
    }


def replay_chart_identity(effective_config: dict[str, Any]) -> str:
    return canonical_sha256(effective_config)


def _label_source_sha256(pair: VerifiedIfvgPair) -> str:
    matching = [
        entry
        for entry in pair.v2.manifest.get("artifacts", ())
        if str(entry.get("path", "")).endswith("/label_source_1m.parquet")
    ]
    if len(matching) != 1:
        raise ReplayChartStoreError("v2 artifact has no unique label-source manifest entry")
    return str(matching[0]["sha256"])


def build_candidate_bar_ranges(
    pair: VerifiedIfvgPair,
    bars_1m: pd.DataFrame,
) -> pd.DataFrame:
    """One row per candidate: range anchors, separated end times, exact refs."""
    candidates = pair.v2.tables[RecordTable.ENTRY_CANDIDATE]
    dossiers = pair.v2.tables[RecordTable.GEOMETRY_DOSSIER]
    lifecycle = pair.v2.tables[RecordTable.SETUP_LIFECYCLE]
    executed = pair.v2.tables[RecordTable.EXECUTED_TRADE]
    labels = pair.v2.tables[RecordTable.CANDIDATE_LABEL]
    links = pair.v3.tables[ContextRecordTable.CANDIDATE_CONTEXT_LINK]

    bar_close = pd.to_datetime(bars_1m["close_ts_utc"], utc=True, errors="raise")
    bar_order = bar_close.sort_values(kind="mergesort")
    ordered_close_ns = bar_order.astype("int64").to_numpy()
    ordered_ids = bars_1m["bar_id"].astype(str).to_numpy()[bar_order.index.to_numpy()]

    life = lifecycle.copy()
    life["_ts"] = pd.to_datetime(life["envelope_ts_utc"], utc=True, errors="raise")
    by_setup = life.groupby("setup_id", sort=False)
    activated = (
        life[life["transition"] == "setup_activated"]
        .set_index("setup_id", verify_integrity=True)["_ts"]
    )
    ended = life[life["transition"] == "setup_ended"].groupby("setup_id")["_ts"].max()
    last_event = by_setup["_ts"].max()
    ordinal_min = by_setup["trace_ordinal"].min()
    ordinal_max = by_setup["trace_ordinal"].max()

    dossier_index = dossiers.set_index("candidate_id", verify_integrity=True)
    executed_index = executed.set_index("candidate_id", verify_integrity=True)
    link_index = links.set_index("candidate_id", verify_integrity=True)
    primary = labels[labels["label_family"] == PRIMARY_LABEL_FAMILY]
    if primary["candidate_id"].duplicated().any():
        raise ReplayChartStoreError("primary label family has duplicate candidate labels")
    label_index = primary.set_index("candidate_id", verify_integrity=True)
    label_bar_close = dict(zip(bars_1m["bar_id"].astype(str), bar_close, strict=True))
    cutoff = pd.Timestamp(AUTHORIZED_CUTOFF_UTC)

    def _bar_span(start: pd.Timestamp, end: pd.Timestamp) -> tuple[str | None, str | None]:
        first = int(np.searchsorted(ordered_close_ns, start.value, "left"))
        last = int(np.searchsorted(ordered_close_ns, end.value, "right")) - 1
        first = max(first - RANGE_PAD_1M_BARS, 0)
        last = min(last + RANGE_PAD_1M_BARS, len(ordered_ids) - 1)
        if last < first:
            return None, None
        return str(ordered_ids[first]), str(ordered_ids[last])

    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate["candidate_id"])
        setup_id = str(candidate["setup_id"])
        dossier = dossier_index.loc[candidate_id]
        trade = executed_index.loc[candidate_id] if candidate_id in executed_index.index else None
        label = label_index.loc[candidate_id] if candidate_id in label_index.index else None
        link = link_index.loc[candidate_id] if candidate_id in link_index.index else None

        setup_start = activated.get(setup_id)
        formation_candidates = [
            pd.Timestamp(dossier[f"geometry_{zone}_a_open_ts_utc"])
            for zone in ("htf", "parent", "opposing", "entry_fvg")
            if pd.notna(dossier.get(f"geometry_{zone}_a_open_ts_utc"))
        ]
        formation_start = min(formation_candidates) if formation_candidates else None
        candidate_ts = pd.Timestamp(candidate["envelope_ts_utc"])

        if trade is not None:
            entry_anchor = pd.Timestamp(trade["entry_ts_utc"])
            entry_anchor_source = "executed_trade.entry_ts_utc"
        else:
            entry_anchor = candidate_ts
            entry_anchor_source = "entry_candidate.envelope_ts_utc"

        trade_resolution = (
            pd.Timestamp(trade["resolution_ts_utc"]) if trade is not None else None
        )
        label_resolution = None
        label_censored = bool(label["censored"]) if label is not None else False
        if label is not None and pd.notna(label.get("resolution_bar_id")):
            label_resolution = label_bar_close.get(str(label["resolution_bar_id"]))
            if label_resolution is None:
                raise ReplayChartStoreError(
                    f"candidate label resolution bar is not in the verified 1m stream: "
                    f"{label['resolution_bar_id']}"
                )
        setup_end = ended.get(setup_id)
        if setup_end is None or pd.isna(setup_end):
            setup_end = last_event.get(setup_id)

        if trade_resolution is not None:
            display_end, display_source = trade_resolution, "trade_resolution"
        elif label_resolution is not None and not label_censored:
            display_end, display_source = label_resolution, "candidate_label_resolution"
        elif label_censored:
            display_end, display_source = cutoff, "authorized_cutoff_censor"
        elif setup_end is not None and pd.notna(setup_end):
            display_end, display_source = pd.Timestamp(setup_end), "setup_end"
        else:
            display_end, display_source = candidate_ts, "setup_end"

        has_setup_start = setup_start is not None and pd.notna(setup_start)
        span_start = setup_start if has_setup_start else candidate_ts
        first_bar, last_bar = _bar_span(pd.Timestamp(span_start), pd.Timestamp(display_end))

        rows.append(
            {
                "candidate_id": candidate_id,
                "setup_id": setup_id,
                "decision_id": (
                    str(dossier["decision_id"]) if pd.notna(dossier.get("decision_id")) else None
                ),
                "trade_id": (
                    str(dossier["trade_id"]) if pd.notna(dossier.get("trade_id")) else None
                ),
                "trading_day": str(candidate["trading_day"]),
                "is_warmup": bool(candidate["is_warmup"]),
                "setup_start_ts": setup_start,
                "setup_start_ts_source": "setup_lifecycle_event.envelope_ts_utc[setup_activated]",
                "formation_start_ts": formation_start,
                "formation_start_ts_source": "geometry_dossier.geometry_*_a_open_ts_utc[min]",
                "entry_anchor_ts": entry_anchor,
                "entry_anchor_ts_source": entry_anchor_source,
                "setup_end_ts": setup_end,
                "candidate_label_resolution_ts": label_resolution,
                "candidate_label_censored": label_censored,
                "trade_resolution_ts": trade_resolution,
                "display_end_ts": display_end,
                "display_end_source": display_source,
                "lifecycle_trace_ordinal_min": int(ordinal_min.get(setup_id)),
                "lifecycle_trace_ordinal_max": int(ordinal_max.get(setup_id)),
                "candidate_trace_ordinal": int(candidate["trace_ordinal"]),
                "context_capture_id": (
                    str(link["context_capture_id"]) if link is not None else None
                ),
                "geometry_evidence_id": (
                    str(link["geometry_evidence_id"]) if link is not None else None
                ),
                "geometry_evidence_cursor": (
                    str(link["geometry_evidence_cursor"]) if link is not None else None
                ),
                "htf_timeframe_seconds": int(dossier["geometry_htf_timeframe_seconds"]),
                "parent_timeframe_seconds": int(dossier["geometry_parent_timeframe_seconds"]),
                "first_bar_id_1m": first_bar,
                "last_bar_id_1m": last_bar,
            }
        )

    frame = pd.DataFrame(rows)
    if frame["candidate_id"].duplicated().any():
        raise ReplayChartStoreError("candidate bar ranges must be unique per candidate")
    unknown = set(frame["display_end_source"]) - set(DISPLAY_END_SOURCES)
    if unknown:
        raise ReplayChartStoreError(f"unknown display end sources: {sorted(unknown)}")
    return frame


def _corroborate_against_tbars(
    bars_tf: pd.DataFrame,
    *,
    repo_root: Path,
    data_dir: Path,
    audit: DevelopmentAccessAudit,
) -> dict[str, Any]:
    """Compare resampled bars to the engine's per-day tbars oracle.

    Any mismatch refuses publication.  Missing day files are recorded, not
    fatal — the replay bars derive solely from the verified 1m source.
    """
    config = IfvgCaptureConfig(data_dir=repo_root / data_dir)
    access = DevelopmentDataAccess(audit=audit)
    days = sorted(bars_tf["trading_day"].unique())
    corroborated: dict[str, str] = {}
    uncorroborated: list[str] = []
    compared_columns = ["open_ticks", "high_ticks", "low_ticks", "close_ticks"]
    for day in days:
        if not access.exists(day, config.bars_path):
            uncorroborated.append(day)
            continue
        oracle = access.read_parquet(day, config.bars_path)
        ours = bars_tf[bars_tf["trading_day"] == day]
        for timeframe in REPLAY_TIMEFRAMES_SECONDS:
            oracle_tf = oracle[oracle["timeframe_ticks"] == timeframe]
            ours_tf = ours[ours["timeframe_seconds"] == timeframe]
            oracle_map = oracle_tf.set_index("bar_id")[compared_columns]
            ours_map = ours_tf.set_index("bar_id")[compared_columns]
            if sorted(oracle_map.index) != sorted(ours_map.index):
                raise ReplayChartStoreError(
                    f"tbars corroboration bar-set mismatch: {day} @ {timeframe}s"
                )
            aligned = oracle_map.loc[ours_map.index]
            if not (aligned.to_numpy() == ours_map.to_numpy()).all():
                raise ReplayChartStoreError(
                    f"tbars corroboration OHLC mismatch: {day} @ {timeframe}s"
                )
        corroborated[day] = file_sha256(access.construct_path(day, config.bars_path))
    audit.assert_zero_protected()
    return {
        "oracle": "ifvg_tbars_per_day",
        "compared_columns": compared_columns,
        "timeframes_seconds": list(REPLAY_TIMEFRAMES_SECONDS),
        "corroborated_days": sorted(corroborated),
        "per_day_tbars_sha256": dict(sorted(corroborated.items())),
        "uncorroborated_days": uncorroborated,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _artifact_entry(path: Path, root: Path, *, rows: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        entry["rows"] = int(rows)
    return entry


def build_replay_chart_artifact(
    pair: VerifiedIfvgPair,
    *,
    repo_root: Path,
    base_dir: Path | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    corroborate: bool = True,
) -> Path:
    """Build (or idempotently re-verify) the replay-chart artifact for a pair."""
    repo_root = Path(repo_root).resolve()
    base = Path(base_dir) if base_dir is not None else repo_root / REPLAY_CHART_STORE
    pair_ref = ArtifactPairRef.from_verified_pair(pair)
    effective_config = replay_chart_effective_config(
        pair_ref, label_source_sha256=_label_source_sha256(pair)
    )
    artifact_id = replay_chart_identity(effective_config)
    destination = Path(base).resolve() / artifact_id
    if destination.exists():
        load_verified_replay_chart_artifact(base, artifact_id, expected_pair=pair_ref)
        return destination

    bars_1m = load_verified_label_source_bars(pair.v2)
    bars_tf = pd.concat(
        [resample_label_bars(bars_1m, timeframe) for timeframe in REPLAY_TIMEFRAMES_SECONDS],
        ignore_index=True,
    )
    audit = DevelopmentAccessAudit()
    if corroborate:
        corroboration = _corroborate_against_tbars(
            bars_tf, repo_root=repo_root, data_dir=data_dir, audit=audit
        )
    else:
        corroboration = {"oracle": "skipped", "corroborated_days": [], "uncorroborated_days": []}
    candidate_ranges = build_candidate_bar_ranges(pair, bars_1m)

    base.mkdir(parents=True, exist_ok=True)
    temporary = Path(base).resolve() / f".{artifact_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        artifacts: list[dict[str, Any]] = []
        config_path = temporary / "effective_config.json"
        _write_json(config_path, effective_config)
        artifacts.append(_artifact_entry(config_path, temporary))

        bars_path = temporary / "bars_tf.parquet"
        bars_tf.to_parquet(bars_path, index=False)
        artifacts.append(_artifact_entry(bars_path, temporary, rows=len(bars_tf)))

        ranges_path = temporary / "candidate_bar_range.parquet"
        candidate_ranges.to_parquet(ranges_path, index=False)
        artifacts.append(_artifact_entry(ranges_path, temporary, rows=len(candidate_ranges)))

        access_path = temporary / "data_access_audit.json"
        _write_json(access_path, audit.as_dict())
        artifacts.append(_artifact_entry(access_path, temporary))

        # Corroboration is verification evidence about an EXTERNAL oracle; it
        # stays outside the canonical manifest payload so oracle bytes can
        # never change the artifact identity.
        _write_json(temporary / "corroboration_report.json", corroboration)

        manifest_core = {
            "manifest_schema_version": 1,
            "artifact_kind": "ifvg_replay_chart_v1",
            "replay_chart_artifact_id": artifact_id,
            "immutable": True,
            "source_pair": pair_ref.as_dict(),
            "effective_config": effective_config,
            "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        }
        _write_json(
            temporary / "manifest.json",
            {**manifest_core, "manifest_payload_sha256": canonical_sha256(manifest_core)},
        )
        if destination.exists():
            raise FileExistsError("replay-chart artifact appeared concurrently")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    update_replay_chart_catalog(
        artifact_id,
        pair_ref,
        manifest_payload_sha256=manifest["manifest_payload_sha256"],
        catalog_path=repo_root / REPLAY_CHART_CATALOG,
    )
    return destination


def load_verified_replay_chart_artifact(
    base_dir: Path,
    artifact_id: str,
    *,
    expected_pair: ArtifactPairRef,
) -> VerifiedReplayChartArtifact:
    """Verified exact-ID load; refuses pair mismatch and policy/content drift."""
    if not _FULL_SHA256.fullmatch(artifact_id):
        raise ReplayChartStoreError("replay-chart artifact ID must be a full SHA-256")
    root = Path(base_dir).resolve()
    directory = (root / artifact_id).resolve()
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise ReplayChartStoreError("replay-chart artifact escaped its store") from error
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayChartStoreError("replay-chart manifest is unreadable") from error
    if manifest.get("replay_chart_artifact_id") != artifact_id:
        raise ReplayChartStoreError("replay-chart manifest identity mismatch")
    if manifest.get("artifact_kind") != "ifvg_replay_chart_v1":
        raise ReplayChartStoreError("replay-chart manifest kind mismatch")
    core = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    if canonical_sha256(core) != manifest.get("manifest_payload_sha256"):
        raise ReplayChartStoreError("replay-chart manifest payload hash mismatch")

    effective_config = manifest.get("effective_config") or {}
    if replay_chart_identity(effective_config) != artifact_id:
        raise ReplayChartStoreError("replay-chart identity does not match its policy set")
    if effective_config.get("anchor_240m_status") != ANCHOR_240M_STATUS:
        raise ReplayChartStoreError("replay-chart 240m anchor status is not the frozen Q-40 flag")
    if effective_config.get("resample_rule") != RESAMPLE_RULE_ID:
        raise ReplayChartStoreError("replay-chart resample rule is not the ratified policy")

    stored_pair = manifest.get("source_pair") or {}
    if stored_pair != expected_pair.as_dict():
        raise ReplayChartStoreError(
            "replay-chart artifact was built for a different v2/v3 pair"
        )

    frames: dict[str, pd.DataFrame] = {}
    for entry in manifest.get("artifacts", ()):
        relative = str(entry.get("path"))
        path = (directory / relative).resolve()
        try:
            path.relative_to(directory)
        except ValueError as error:
            raise ReplayChartStoreError("replay-chart artifact path escapes its dataset") from error
        if not path.is_file():
            raise ReplayChartStoreError(f"replay-chart artifact is missing: {relative}")
        if path.stat().st_size != entry.get("bytes"):
            raise ReplayChartStoreError(f"replay-chart artifact byte mismatch: {relative}")
        if file_sha256(path) != entry.get("sha256"):
            raise ReplayChartStoreError(f"replay-chart artifact SHA-256 mismatch: {relative}")
        if relative.endswith(".parquet"):
            frame = pd.read_parquet(path)
            if "rows" in entry and len(frame) != entry["rows"]:
                raise ReplayChartStoreError(f"replay-chart row-count mismatch: {relative}")
            frames[relative] = frame

    if {"bars_tf.parquet", "candidate_bar_range.parquet"} - set(frames):
        raise ReplayChartStoreError("replay-chart artifact is missing its core tables")
    return VerifiedReplayChartArtifact(
        artifact_id=artifact_id,
        directory=directory,
        manifest=manifest,
        source_pair=ArtifactPairRef(**stored_pair),
        bars_tf=frames["bars_tf.parquet"],
        candidate_ranges=frames["candidate_bar_range.parquet"],
        anchor_240m_status=str(effective_config["anchor_240m_status"]),
    )


def read_replay_chart_catalog(catalog_path: Path) -> dict[str, dict[str, str]]:
    path = Path(catalog_path)
    if not path.exists():
        return {}
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayChartStoreError("replay-chart catalog is unreadable") from error
    if not isinstance(catalog, dict):
        raise ReplayChartStoreError("replay-chart catalog must be an object")
    return catalog


def find_replay_artifact(
    catalog: dict[str, dict[str, str]],
    pair: ArtifactPairRef,
) -> str | None:
    """Resolve a replay artifact by exact pair identity (all five fields)."""
    wanted = pair.as_dict()
    matches = [
        artifact_id
        for artifact_id, entry in catalog.items()
        if isinstance(entry, dict)
        and all(entry.get(key) == value for key, value in wanted.items())
    ]
    if len(matches) > 1:
        raise ReplayChartStoreError("replay-chart catalog has duplicate pair entries")
    return matches[0] if matches else None


def update_replay_chart_catalog(
    artifact_id: str,
    pair: ArtifactPairRef,
    *,
    manifest_payload_sha256: str,
    catalog_path: Path,
) -> None:
    if not _FULL_SHA256.fullmatch(artifact_id):
        raise ReplayChartStoreError("catalog artifact ID must be a full SHA-256")
    path = Path(catalog_path)
    catalog = read_replay_chart_catalog(path)
    catalog[artifact_id] = {
        **pair.as_dict(),
        "replay_chart_manifest_payload_sha256": manifest_payload_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    _write_json(temporary, dict(sorted(catalog.items())))
    os.replace(temporary, path)


# ── setup-aware replay-chart artifact v2 (VerifierBundleRef) ─────────────────


@dataclass(frozen=True)
class VerifierBundleRef:
    """Frozen five-artifact bundle identity for the setup-level verifier.

    ``ArtifactPairRef`` stays exactly as-is (pair identity only, never
    overloaded); this ref ADDS the fsm-audit and replay-chart identities the
    setup verifier needs. Provider and tab open by this ref.
    """

    profile_name: str
    v2_dataset_id: str
    v2_manifest_hash: str
    v3_dataset_id: str
    v3_manifest_hash: str
    fsm_audit_artifact_id: str
    fsm_audit_manifest_hash: str
    replay_chart_artifact_id: str
    replay_chart_manifest_hash: str

    def __post_init__(self) -> None:
        for name in (
            "v2_dataset_id",
            "v2_manifest_hash",
            "v3_dataset_id",
            "v3_manifest_hash",
            "fsm_audit_artifact_id",
            "fsm_audit_manifest_hash",
            "replay_chart_artifact_id",
            "replay_chart_manifest_hash",
        ):
            if not _FULL_SHA256.fullmatch(getattr(self, name)):
                raise ReplayChartStoreError(
                    f"bundle reference {name} must be a full SHA-256"
                )
        if not self.profile_name:
            raise ReplayChartStoreError("bundle reference requires a profile name")

    def pair_ref(self) -> ArtifactPairRef:
        return ArtifactPairRef(
            profile_name=self.profile_name,
            v2_dataset_id=self.v2_dataset_id,
            v2_manifest_hash=self.v2_manifest_hash,
            v3_dataset_id=self.v3_dataset_id,
            v3_manifest_hash=self.v3_manifest_hash,
        )

    def as_dict(self) -> dict[str, str]:
        return dict(sorted(asdict(self).items()))


def replay_chart_effective_config_v2(
    pair_ref: ArtifactPairRef,
    *,
    label_source_sha256: str,
    fsm_audit_artifact_id: str,
    fsm_audit_manifest_hash: str,
) -> dict[str, Any]:
    """The v2 policy set: everything from v1 PLUS the fsm-audit pins and the
    bumped setup-aware range/gating policies. New content identity; the v1
    artifact is untouched and remains catalogued."""
    if not _FULL_SHA256.fullmatch(fsm_audit_artifact_id):
        raise ReplayChartStoreError("fsm audit artifact id must be a full SHA-256")
    if not _FULL_SHA256.fullmatch(fsm_audit_manifest_hash):
        raise ReplayChartStoreError("fsm audit manifest hash must be a full SHA-256")
    return {
        "artifact_kind": "ifvg_replay_chart_v2",
        "replay_chart_schema_version": REPLAY_CHART_SCHEMA_VERSION_V2,
        "bar_schema_version": BAR_SCHEMA_VERSION,
        "source_pair": pair_ref.as_dict(),
        "source_label_table_sha256": label_source_sha256,
        "fsm_audit_artifact_id": fsm_audit_artifact_id,
        "fsm_audit_manifest_hash": fsm_audit_manifest_hash,
        "resample_rule": RESAMPLE_RULE_ID,
        "timeframes_seconds": list(REPLAY_TIMEFRAMES_SECONDS),
        "development_cutoff_utc": AUTHORIZED_CUTOFF_UTC,
        "candidate_range_policy": CANDIDATE_RANGE_POLICY,
        "setup_range_policy": SETUP_RANGE_POLICY_V2,
        "primary_label_family": PRIMARY_LABEL_FAMILY,
        "stage_gating_policy": STAGE_GATING_POLICY_V2,
        "partial_bar_policy": PARTIAL_BAR_POLICY,
        "range_pad_1m_bars": RANGE_PAD_1M_BARS,
        "anchor_240m_status": ANCHOR_240M_STATUS,
    }


def _fvg_tf_seconds(fvg_id: object) -> int | None:
    if fvg_id is None or (isinstance(fvg_id, float) and pd.isna(fvg_id)):
        return None
    text = str(fvg_id)
    head = text.split(":", 1)[0]
    if head.endswith("s") and head[:-1].isdigit():
        return int(head[:-1])
    return None


_TERMINAL_END_SOURCE = {
    "slot_freed": "trade_resolution",
    "dataset_exhaustion": "setup_end",
    "dataset_exhaustion_pre_entry": "setup_end",
    "invalidated_htf_filled": "setup_invalidation",
    "invalidated_parent_filled": "setup_invalidation",
    "invalidated_parent_structural": "setup_invalidation",
}


def build_setup_bar_ranges(
    fsm_audit_tables: dict,
    *,
    candidate_ranges: pd.DataFrame,
) -> pd.DataFrame:
    """One row per setup (activation → terminal), sourced from persisted audit
    evidence only — never inferred by timestamp proximity."""
    from .audit_contracts import AuditTable

    windows = fsm_audit_tables.get(AuditTable.PARENT_WINDOW, pd.DataFrame())
    deaths = fsm_audit_tables.get(AuditTable.SLOT_DEATH, pd.DataFrame())
    taps = fsm_audit_tables.get(AuditTable.HTF_TAP, pd.DataFrame())
    locks = fsm_audit_tables.get(AuditTable.PARENT_LOCK, pd.DataFrame())
    resolutions = fsm_audit_tables.get(AuditTable.SETUP_RESOLUTION, pd.DataFrame())
    causality = fsm_audit_tables.get(AuditTable.ENTRY_CAUSALITY, pd.DataFrame())
    intervals = fsm_audit_tables.get(AuditTable.PARENTLESS_INTERVAL, pd.DataFrame())

    if windows.empty:
        raise ReplayChartStoreError("fsm audit artifact has no parent-window events")
    opened = windows.loc[windows["event_kind"] == "opened"]
    terminal = (
        deaths.loc[deaths["setup_terminated"].astype(bool)]
        if not deaths.empty
        else pd.DataFrame()
    )
    if terminal.empty:
        raise ReplayChartStoreError("fsm audit artifact has no terminal deaths")
    selected_taps = (
        taps.loc[taps["selected"].astype(bool)] if not taps.empty else pd.DataFrame()
    )
    cutoff = pd.Timestamp(AUTHORIZED_CUTOFF_UTC)

    rows: list[dict[str, Any]] = []
    for opened_row in opened.to_dict("records"):
        setup_id = str(opened_row["setup_id"])
        mine_terminal = terminal.loc[terminal["setup_id"].astype(str) == setup_id]
        if len(mine_terminal) != 1:
            raise ReplayChartStoreError(
                f"setup {setup_id} has {len(mine_terminal)} terminal deaths"
            )
        death = mine_terminal.iloc[0].to_dict()
        tap_row: dict[str, Any] = {}
        if not selected_taps.empty:
            mine_tap = selected_taps.loc[
                selected_taps["envelope_setup_id"].astype(str) == setup_id
            ]
            if len(mine_tap):
                tap_row = mine_tap.iloc[0].to_dict()
        lock_row: dict[str, Any] = {}
        if not locks.empty:
            mine_lock = locks.loc[locks["envelope_setup_id"].astype(str) == setup_id]
            if len(mine_lock):
                lock_row = mine_lock.iloc[0].to_dict()
        resolution_row: dict[str, Any] = {}
        if not resolutions.empty:
            mine_res = resolutions.loc[
                resolutions["envelope_setup_id"].astype(str) == setup_id
            ]
            if len(mine_res):
                resolution_row = mine_res.iloc[-1].to_dict()
        candidate_ids: list[str] = []
        if not causality.empty:
            candidate_ids = sorted(
                causality.loc[
                    causality["setup_id"].astype(str) == setup_id, "candidate_id"
                ]
                .astype(str)
                .unique()
                .tolist()
            )
        interval_ids: list[str] = []
        if not intervals.empty:
            interval_ids = sorted(
                intervals.loc[
                    intervals["setup_id"].astype(str) == setup_id, "interval_id"
                ]
                .astype(str)
                .tolist()
            )
        parent_tf = None
        if not windows.empty:
            mine_selected = windows.loc[
                (windows["event_kind"] == "parent_selected")
                & (windows["setup_id"].astype(str) == setup_id)
            ]
            if len(mine_selected):
                parent_tf = _fvg_tf_seconds(
                    mine_selected.iloc[-1]["parent_fvg_id"]
                )
        death_reason = str(death["death_reason"])
        terminal_ts = pd.Timestamp(death["death_ts_utc"])
        end_source = _TERMINAL_END_SOURCE.get(death_reason, "setup_end")
        display_end_ts = terminal_ts
        if display_end_ts > cutoff:
            display_end_ts = cutoff
            end_source = "authorized_cutoff_censor"
        htf_id = tap_row.get("fvg_fvg_id") or resolution_row.get("htf_fvg_id")
        rows.append(
            {
                "setup_id": setup_id,
                "activation_ts_utc": opened_row["envelope_ts_utc"],
                "activation_cursor": opened_row["event_cursor"],
                "activation_source": "parent_window_opened",
                "tap_ts_utc": tap_row.get("envelope_ts_utc"),
                "tap_cursor": tap_row.get("tap_cursor"),
                "lock_ts_utc": lock_row.get("envelope_ts_utc"),
                "lock_cursor": lock_row.get("lock_cursor"),
                "armed_ts_utc": resolution_row.get("armed_ts_utc"),
                "inversion_ts_utc": resolution_row.get("inversion_ts_utc"),
                "first_candidate_ts_utc": resolution_row.get("entry_ts_utc"),
                "terminal_ts_utc": terminal_ts,
                "terminal_cursor": death["event_cursor"],
                "terminal_reason": death_reason,
                "phase_at_death": str(death["phase"]),
                "terminal_source": "parent_slot_death",
                "display_end_ts_utc": display_end_ts,
                "display_end_source": end_source,
                "htf_fvg_id": htf_id,
                "htf_tf_seconds": tap_row.get("htf_tf_seconds"),
                "parent_tf_seconds": parent_tf,
                "session_engine_at_activation": tap_row.get("session_engine"),
                "session_doc_at_activation": tap_row.get("session_doc"),
                "candidate_ids": json.dumps(candidate_ids),
                "candidate_count": len(candidate_ids),
                "candidate_less": not candidate_ids,
                "parentless_interval_ids": json.dumps(interval_ids),
                "parentless_interval_count": len(interval_ids),
                "q40_exposed": (
                    int(tap_row.get("htf_tf_seconds") or 0) == 14400
                ),
                "setup_range_policy": SETUP_RANGE_POLICY_V2,
            }
        )
    frame = pd.DataFrame(rows)
    if frame["setup_id"].duplicated().any():
        raise ReplayChartStoreError("setup ranges produced duplicate setup ids")
    known_candidates = (
        set(candidate_ranges["candidate_id"].astype(str))
        if not candidate_ranges.empty and "candidate_id" in candidate_ranges
        else set()
    )
    for candidate_json in frame["candidate_ids"]:
        for candidate_id in json.loads(candidate_json):
            if known_candidates and candidate_id not in known_candidates:
                raise ReplayChartStoreError(
                    f"setup range references unknown candidate {candidate_id}"
                )
    return frame.sort_values("setup_id", kind="mergesort").reset_index(drop=True)


def build_replay_chart_artifact_v2(
    pair: VerifiedIfvgPair,
    fsm_audit,
    *,
    repo_root: Path,
    base_dir: Path | None = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    corroborate: bool = True,
) -> Path:
    """Build (or idempotently re-verify) the setup-aware replay-chart artifact.

    ``fsm_audit`` is a VerifiedFsmAuditArtifact; its identity enters the
    canonical policy set, so a different audit artifact yields a different
    replay-chart identity. The v1 artifact is never touched."""
    repo_root = Path(repo_root).resolve()
    base = Path(base_dir) if base_dir is not None else repo_root / REPLAY_CHART_STORE
    pair_ref = ArtifactPairRef.from_verified_pair(pair)
    effective_config = replay_chart_effective_config_v2(
        pair_ref,
        label_source_sha256=_label_source_sha256(pair),
        fsm_audit_artifact_id=fsm_audit.artifact_id,
        fsm_audit_manifest_hash=fsm_audit.manifest_payload_sha256,
    )
    artifact_id = replay_chart_identity(effective_config)
    destination = Path(base).resolve() / artifact_id
    if destination.exists():
        load_verified_replay_chart_artifact_v2(
            base, artifact_id, expected_pair=pair_ref
        )
        return destination

    bars_1m = load_verified_label_source_bars(pair.v2)
    bars_tf = pd.concat(
        [resample_label_bars(bars_1m, timeframe) for timeframe in REPLAY_TIMEFRAMES_SECONDS],
        ignore_index=True,
    )
    audit = DevelopmentAccessAudit()
    if corroborate:
        corroboration = _corroborate_against_tbars(
            bars_tf, repo_root=repo_root, data_dir=data_dir, audit=audit
        )
    else:
        corroboration = {"oracle": "skipped", "corroborated_days": [], "uncorroborated_days": []}
    candidate_ranges = build_candidate_bar_ranges(pair, bars_1m)
    setup_ranges = build_setup_bar_ranges(
        fsm_audit.tables, candidate_ranges=candidate_ranges
    )

    base.mkdir(parents=True, exist_ok=True)
    temporary = Path(base).resolve() / f".{artifact_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        artifacts: list[dict[str, Any]] = []
        config_path = temporary / "effective_config.json"
        _write_json(config_path, effective_config)
        artifacts.append(_artifact_entry(config_path, temporary))
        bars_path = temporary / "bars_tf.parquet"
        bars_tf.to_parquet(bars_path, index=False)
        artifacts.append(_artifact_entry(bars_path, temporary, rows=len(bars_tf)))
        ranges_path = temporary / "candidate_bar_range.parquet"
        candidate_ranges.to_parquet(ranges_path, index=False)
        artifacts.append(_artifact_entry(ranges_path, temporary, rows=len(candidate_ranges)))
        setup_path = temporary / "setup_bar_range.parquet"
        setup_ranges.to_parquet(setup_path, index=False)
        artifacts.append(_artifact_entry(setup_path, temporary, rows=len(setup_ranges)))
        access_path = temporary / "data_access_audit.json"
        _write_json(access_path, audit.as_dict())
        artifacts.append(_artifact_entry(access_path, temporary))
        # Corroboration stays OUTSIDE canonical identity (existing decision).
        _write_json(temporary / "corroboration_report.json", corroboration)

        manifest_core = {
            "manifest_schema_version": 1,
            "artifact_kind": "ifvg_replay_chart_v2",
            "replay_chart_artifact_id": artifact_id,
            "immutable": True,
            "source_pair": pair_ref.as_dict(),
            "effective_config": effective_config,
            "artifacts": sorted(artifacts, key=lambda item: item["path"]),
        }
        _write_json(
            temporary / "manifest.json",
            {**manifest_core, "manifest_payload_sha256": canonical_sha256(manifest_core)},
        )
        if destination.exists():
            raise FileExistsError("replay-chart v2 artifact appeared concurrently")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    # NOTE: no catalog write here — per the publication gate, no catalog
    # points at the new bundle until the final QL commit hash is recorded.
    return destination


def load_verified_replay_chart_artifact_v2(
    base_dir: Path,
    artifact_id: str,
    *,
    expected_pair: ArtifactPairRef,
    expected_fsm_audit_artifact_id: str | None = None,
) -> VerifiedReplayChartArtifact:
    """Verified exact-ID load of a v2 (setup-aware) artifact."""
    if not _FULL_SHA256.fullmatch(artifact_id):
        raise ReplayChartStoreError("replay-chart artifact ID must be a full SHA-256")
    root = Path(base_dir).resolve()
    directory = (root / artifact_id).resolve()
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise ReplayChartStoreError("replay-chart artifact escaped its store") from error
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayChartStoreError("replay-chart manifest is unreadable") from error
    if manifest.get("replay_chart_artifact_id") != artifact_id:
        raise ReplayChartStoreError("replay-chart manifest identity mismatch")
    if manifest.get("artifact_kind") != "ifvg_replay_chart_v2":
        raise ReplayChartStoreError("replay-chart manifest kind is not v2")
    core = {key: value for key, value in manifest.items() if key != "manifest_payload_sha256"}
    if canonical_sha256(core) != manifest.get("manifest_payload_sha256"):
        raise ReplayChartStoreError("replay-chart manifest payload hash mismatch")
    effective_config = manifest.get("effective_config") or {}
    if replay_chart_identity(effective_config) != artifact_id:
        raise ReplayChartStoreError("replay-chart identity does not match its policy set")
    if effective_config.get("anchor_240m_status") != ANCHOR_240M_STATUS:
        raise ReplayChartStoreError("replay-chart 240m anchor status is not the frozen Q-40 flag")
    if effective_config.get("resample_rule") != RESAMPLE_RULE_ID:
        raise ReplayChartStoreError("replay-chart resample rule is not the ratified policy")
    if effective_config.get("setup_range_policy") != SETUP_RANGE_POLICY_V2:
        raise ReplayChartStoreError("replay-chart setup range policy mismatch")
    if (
        expected_fsm_audit_artifact_id is not None
        and effective_config.get("fsm_audit_artifact_id")
        != expected_fsm_audit_artifact_id
    ):
        raise ReplayChartStoreError("replay-chart pins a different fsm-audit artifact")
    stored_pair = manifest.get("source_pair") or {}
    if stored_pair != expected_pair.as_dict():
        raise ReplayChartStoreError(
            "replay-chart artifact was built for a different v2/v3 pair"
        )
    frames: dict[str, pd.DataFrame] = {}
    for entry in manifest.get("artifacts", ()):
        relative = str(entry.get("path"))
        path = (directory / relative).resolve()
        try:
            path.relative_to(directory)
        except ValueError as error:
            raise ReplayChartStoreError("replay-chart artifact path escapes its dataset") from error
        if not path.is_file():
            raise ReplayChartStoreError(f"replay-chart artifact is missing: {relative}")
        if path.stat().st_size != entry.get("bytes"):
            raise ReplayChartStoreError(f"replay-chart artifact byte mismatch: {relative}")
        if file_sha256(path) != entry.get("sha256"):
            raise ReplayChartStoreError(f"replay-chart artifact SHA-256 mismatch: {relative}")
        if relative.endswith(".parquet"):
            frame = pd.read_parquet(path)
            if "rows" in entry and len(frame) != entry["rows"]:
                raise ReplayChartStoreError(f"replay-chart row-count mismatch: {relative}")
            frames[relative] = frame
    required = {"bars_tf.parquet", "candidate_bar_range.parquet", "setup_bar_range.parquet"}
    if required - set(frames):
        raise ReplayChartStoreError("replay-chart v2 artifact is missing its core tables")
    verified = VerifiedReplayChartArtifact(
        artifact_id=artifact_id,
        directory=directory,
        manifest=manifest,
        source_pair=ArtifactPairRef(**stored_pair),
        bars_tf=frames["bars_tf.parquet"],
        candidate_ranges=frames["candidate_bar_range.parquet"],
        anchor_240m_status=str(effective_config["anchor_240m_status"]),
    )
    return verified


def load_setup_ranges_v2(artifact: VerifiedReplayChartArtifact) -> pd.DataFrame:
    """The setup range table of a verified v2 artifact (hash already checked
    at load; this is a plain read of the verified file)."""
    path = artifact.directory / "setup_bar_range.parquet"
    if not path.is_file():
        raise ReplayChartStoreError("verified artifact lacks setup_bar_range.parquet")
    return pd.read_parquet(path)


def find_replay_artifact_v2(
    catalog: dict[str, dict[str, str]],
    bundle_pair: ArtifactPairRef,
    *,
    fsm_audit_artifact_id: str,
) -> str | None:
    """Resolve a SETUP-AWARE artifact by pair + fsm-audit identity."""
    wanted = bundle_pair.as_dict()
    matches = [
        artifact_id
        for artifact_id, entry in catalog.items()
        if isinstance(entry, dict)
        and entry.get("artifact_kind") == "ifvg_replay_chart_v2"
        and entry.get("fsm_audit_artifact_id") == fsm_audit_artifact_id
        and all(entry.get(key) == value for key, value in wanted.items())
    ]
    if len(matches) > 1:
        raise ReplayChartStoreError("replay-chart catalog has duplicate v2 bundle entries")
    return matches[0] if matches else None


def update_replay_chart_catalog_v2(
    artifact_id: str,
    pair: ArtifactPairRef,
    *,
    fsm_audit_artifact_id: str,
    fsm_audit_manifest_hash: str,
    manifest_payload_sha256: str,
    catalog_path: Path,
) -> None:
    """Additive v2 catalog entry — ONLY called at the publication gate, after
    the final QL commit hash is recorded in the artifact manifests."""
    if not _FULL_SHA256.fullmatch(artifact_id):
        raise ReplayChartStoreError("catalog artifact ID must be a full SHA-256")
    path = Path(catalog_path)
    catalog = read_replay_chart_catalog(path)
    catalog[artifact_id] = {
        **pair.as_dict(),
        "artifact_kind": "ifvg_replay_chart_v2",
        "fsm_audit_artifact_id": fsm_audit_artifact_id,
        "fsm_audit_manifest_hash": fsm_audit_manifest_hash,
        "replay_chart_manifest_payload_sha256": manifest_payload_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    _write_json(temporary, dict(sorted(catalog.items())))
    os.replace(temporary, path)
