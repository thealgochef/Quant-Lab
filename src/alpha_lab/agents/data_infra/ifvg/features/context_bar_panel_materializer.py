"""Context-bar panel materializer (R6.1 workstream A; owner decision Q3).

Builds the immutable ``IFVG_CONTEXT_BAR_PANEL_V1`` artifact for ONE
owner-registered interval (5m or 15m — separate artifacts, protocols, folds,
fits) from a VERIFIED replay-chart artifact and nothing else: the
``bars_tf.parquet`` bytes are re-read from the artifact directory and hashed
against the manifest entry — the in-memory frame is never trusted. The
recipe (source identities, interval, block, formula/materializer versions,
every panel policy stamp, both schema hashes) is the payload; the produced
tables are envelope facts (R5B artifact pattern).

Rules (``features/context_bar_panel_contract.py``): completed bars only
(``is_final_partial`` rows are excluded and counted); every window needs the
13 complete same-day source bars ``t−12..t`` (``observed_1m_count ==
expected_1m_count`` on every one of them — otherwise ALL seven features are
null, ``cbp_valid=False``, ``cbp_missing_reason="source_bar_incomplete"``
and the offending bars are written to the validity sidecar); windows never
cross the 18:00 ET trading-day boundary (named-session boundaries do NOT
reset the lookback); a positional lookback shorter than 12 completed
same-day bars is ``insufficient_trading_day_lookback``; a non-contiguous
window (``bar_index_t − bar_index_{t−12} != 12``) is ``lookback_window_gap``;
a legitimate zero-denominator formula NaN keeps the row VALID. Prohibited
inputs by construction: labels, outcomes, MFE/MAE, resolution,
future-session statistics, post-entry evidence, prop outcomes, MBP-1 features.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, model_validator
from strategy_core.constants import IFVG_DOC_SESSION_SCHEME
from strategy_core.decisions.sessions import classify_session

from ..manifest import file_sha256
from ..replay_chart_store import (
    REPLAY_TIMEFRAMES_SECONDS,
    VerifiedReplayChartArtifact,
)
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    register_identity_pair,
)
from ..search.store import load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope
from .arrow_tables import (
    arrow_schema_hash,
    bytes_sha256,
    frame_from_arrow_bytes,
    frame_to_arrow_bytes,
)
from .context_bar_panel_contract import (
    CONTEXT_BAR_PANEL_BLOCK_KEY,
    CONTEXT_BAR_PANEL_FEATURES,
    CONTEXT_BAR_PANEL_FORMULA_VERSION,
    CONTEXT_BAR_PANEL_MATERIALIZER_VERSION,
    INTENSITY_SOURCE_FIELD,
    LOOKBACK_BARS,
    LOOKBACK_POLICY_ID,
    MINIMUM_SOURCE_BARS,
    PANEL_AS_OF_POLICY_ID_V1,
    PANEL_MISSING_REASONS,
    PANEL_RESAMPLE_RULE_ID,
    PANEL_SESSION_STATES,
    PARTIAL_BAR_POLICY_ID,
    SESSION_RESET_POLICY_ID,
    SESSION_SCHEME_ID,
    STD_DDOF,
    WARMUP_POLICY_ID,
    assert_panel_interval_registered,
)

__all__ = [
    "CONTEXT_BAR_PANEL_STORE",
    "PANEL_SIDECAR",
    "VALIDITY_SIDECAR",
    "CONTEXT_BAR_PANEL_SCHEMA",
    "CONTEXT_BAR_PANEL_SCHEMA_HASH",
    "CONTEXT_BAR_PANEL_VALIDITY_SCHEMA",
    "CONTEXT_BAR_PANEL_VALIDITY_SCHEMA_HASH",
    "PanelSourcePairRef",
    "ContextBarPanelArtifactPayload",
    "ContextBarPanelArtifactEnvelope",
    "compute_context_bar_panel_features",
    "materialize_context_bar_panel",
    "panel_table_bytes",
    "validity_table_bytes",
    "verify_context_bar_panel_frame",
    "save_context_bar_panel_artifact",
    "load_context_bar_panel_artifact",
    "load_context_bar_panel_frame",
    "load_context_bar_panel_validity",
    "load_verified_context_bar_panel",
]

CONTEXT_BAR_PANEL_STORE = "context_bar_panels"
PANEL_SIDECAR = "context_bar_panel.arrow"
VALIDITY_SIDECAR = "context_bar_panel_validity.parquet"

_ET = ZoneInfo(IFVG_DOC_SESSION_SCHEME.timezone)
_N = LOOKBACK_BARS

CONTEXT_BAR_PANEL_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("row_id", pa.large_string()),
        pa.field("trading_day", pa.large_string()),
        pa.field("bar_index", pa.int64()),
        pa.field("bar_open_ts_utc", pa.large_string()),
        pa.field("bar_close_ts_utc", pa.large_string()),
        pa.field("interval_seconds", pa.int64()),
        pa.field("open_ticks", pa.int64()),
        pa.field("high_ticks", pa.int64()),
        pa.field("low_ticks", pa.int64()),
        pa.field("close_ticks", pa.int64()),
        pa.field("volume", pa.int64()),
        pa.field("trade_count", pa.int64()),
        pa.field("observed_1m_count", pa.int64()),
        pa.field("expected_1m_count", pa.int64()),
        pa.field("cbp_realized_range_12", pa.float64()),
        pa.field("cbp_realized_volatility_12", pa.float64()),
        pa.field("cbp_range_compression_ratio_12", pa.float64()),
        pa.field("cbp_path_efficiency_12", pa.float64()),
        pa.field("cbp_session_state", pa.large_string()),
        pa.field("cbp_volume_intensity_zscore_12", pa.float64()),
        pa.field("cbp_bar_position_in_session", pa.float64()),
        pa.field("cbp_valid", pa.bool_()),
        pa.field("cbp_missing_reason", pa.large_string()),
    ]
)
CONTEXT_BAR_PANEL_SCHEMA_HASH = arrow_schema_hash(CONTEXT_BAR_PANEL_SCHEMA)

CONTEXT_BAR_PANEL_VALIDITY_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("row_id", pa.large_string()),
        pa.field("offending_bar_id", pa.large_string()),
        pa.field("observed_1m_count", pa.int64()),
        pa.field("expected_1m_count", pa.int64()),
        pa.field("reason", pa.large_string()),
    ]
)
CONTEXT_BAR_PANEL_VALIDITY_SCHEMA_HASH = arrow_schema_hash(CONTEXT_BAR_PANEL_VALIDITY_SCHEMA)

_REQUIRED_BAR_COLUMNS = (
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
)


class PanelSourcePairRef(FrozenContract):
    """SHA-patterned mirror of the replay chart's ``ArtifactPairRef``."""

    profile_name: str = Field(min_length=1)
    v2_dataset_id: str = Field(pattern=SHA256_PATTERN)
    v2_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    v3_dataset_id: str = Field(pattern=SHA256_PATTERN)
    v3_manifest_hash: str = Field(pattern=SHA256_PATTERN)


class ContextBarPanelArtifactPayload(FrozenContract):
    replay_chart_artifact_id: str = Field(pattern=SHA256_PATTERN)
    replay_chart_manifest_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    replay_chart_artifact_kind: str
    source_pair: PanelSourcePairRef
    bars_tf_sha256: str = Field(pattern=SHA256_PATTERN)
    resample_rule_id: str
    panel_interval_seconds: int = Field(ge=60)
    resolved_feature_block_id: str = Field(pattern=SHA256_PATTERN)
    formula_version: str
    materializer_version: str
    panel_as_of_policy_id: str
    partial_bar_policy_id: str
    lookback_bars: int = Field(ge=1)
    minimum_source_bars: int = Field(ge=2)
    std_ddof: int = Field(ge=0)
    lookback_policy_id: str
    session_reset_policy_id: str
    warmup_policy_id: str
    session_scheme_id: str
    intensity_source_field: str
    panel_table_schema_hash: str = Field(pattern=SHA256_PATTERN)
    panel_validity_schema_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _registered(self):
        assert_panel_interval_registered(self.panel_interval_seconds)
        if self.minimum_source_bars != self.lookback_bars + 1:
            raise ValueError("minimum_source_bars is the current bar + the lookback bars")
        if self.resample_rule_id != PANEL_RESAMPLE_RULE_ID:
            raise ValueError("panel bars must come from the ratified resample rule")
        return self


class ContextBarPanelArtifactEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "context_bar_panel_artifact_id"

    context_bar_panel_artifact_id: str = Field(pattern=SHA256_PATTERN)
    payload: ContextBarPanelArtifactPayload
    #: post-materialization facts (produced-table binding, never pre-run identity)
    panel_table_sha256: str = Field(pattern=SHA256_PATTERN)
    validity_table_sha256: str = Field(pattern=SHA256_PATTERN)
    row_count: int = Field(ge=0)
    validity_row_count: int = Field(ge=0)
    excluded_partial_bar_count: int = Field(ge=0)
    first_bar_close_ts_utc: str | None
    last_bar_close_ts_utc: str | None


# ── session helpers ──────────────────────────────────────────────────────────


def _session_window_utc(local: datetime, session: str) -> tuple[datetime, datetime] | None:
    """The named session's [open, close) as UTC instants around ``local``
    (DST-aware through the scheme's zone; Asia crosses midnight)."""

    window = IFVG_DOC_SESSION_SCHEME.sessions.get(session)
    if window is None:
        return None
    day: date = local.date()
    if window.crosses_midnight:
        if local.time() >= window.start:
            open_day, close_day = day, day + timedelta(days=1)
        else:
            open_day, close_day = day - timedelta(days=1), day
    else:
        open_day = close_day = day
    opened = datetime.combine(open_day, window.start, tzinfo=_ET).astimezone(UTC)
    closed = datetime.combine(close_day, window.end, tzinfo=_ET).astimezone(UTC)
    return opened, closed


def _session_facts(close_ts: pd.Timestamp) -> tuple[str, float]:
    """``(session_state, bar_position_in_session)`` of one COMPLETED bar.

    The session OF THE BAR is the session in force at the bar's FINAL
    instant (``SESSION_CLASSIFICATION_INSTANT_POLICY``): a completed bar
    that closes exactly at a session open contains no trading of that
    session and belongs to the preceding state; the position is measured
    from the close instant itself against that session's [open, close).
    """

    instant = close_ts.to_pydatetime().astimezone(UTC)
    final_instant = instant - timedelta(microseconds=1)
    info = classify_session(final_instant, IFVG_DOC_SESSION_SCHEME)
    session = str(info.session)
    if session not in PANEL_SESSION_STATES:  # pragma: no cover - scheme invariant
        raise AssertionError(f"unregistered session state {session!r}")
    window = _session_window_utc(info.local_ts, session)
    if window is None:
        return session, float("nan")
    opened, closed = window
    span = (closed - opened).total_seconds()
    if span <= 0:  # pragma: no cover - scheme invariant
        return session, float("nan")
    position = (instant - opened).total_seconds() / span
    return session, float(min(1.0, max(0.0, position)))


# ── the formula contract ─────────────────────────────────────────────────────


def compute_context_bar_panel_features(
    bars: pd.DataFrame, *, interval_seconds: int, lookback_bars: int = LOOKBACK_BARS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The seven features over ONE interval's completed bars.

    Returns ``(panel_frame, validity_frame)``. Deterministic and row-order
    invariant (rows are re-sorted by ``(trading_day, bar_index)``).
    """

    if lookback_bars != _N:
        raise ValueError(f"the V1 panel lookback is fixed at {_N} bars")
    interval = assert_panel_interval_registered(interval_seconds)
    missing = sorted(set(_REQUIRED_BAR_COLUMNS) - set(bars.columns))
    if missing:
        raise ValueError(f"replay-chart bars lack required columns: {missing}")
    frame = bars[bars["timeframe_seconds"].astype(int) == interval].copy()
    frame = frame[~frame["is_final_partial"].astype(bool)].copy()
    frame["trading_day"] = frame["trading_day"].astype(str)
    frame["bar_index"] = frame["bar_index"].astype(int)
    frame = frame.sort_values(["trading_day", "bar_index"], kind="mergesort").reset_index(
        drop=True
    )
    if frame["bar_id"].astype(str).duplicated().any():
        raise ValueError("replay-chart bars repeat a bar id")
    if frame.duplicated(["trading_day", "bar_index"]).any():
        raise ValueError("replay-chart bars repeat a (trading_day, bar_index)")
    opens = pd.to_datetime(frame["logical_open_ts_utc"], utc=True, errors="raise")
    closes = pd.to_datetime(frame["logical_close_ts_utc"], utc=True, errors="raise")

    high = frame["high_ticks"].to_numpy(dtype=float)
    low = frame["low_ticks"].to_numpy(dtype=float)
    close = frame["close_ticks"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float)
    observed = frame["observed_1m_count"].to_numpy(dtype=np.int64)
    expected = frame["expected_1m_count"].to_numpy(dtype=np.int64)
    bar_index = frame["bar_index"].to_numpy(dtype=np.int64)
    days = frame["trading_day"].to_numpy(dtype=object)
    bar_ids = frame["bar_id"].astype(str).to_numpy(dtype=object)
    ranges = high - low

    n = len(frame)
    feature_values = {name: np.full(n, np.nan) for name in CONTEXT_BAR_PANEL_FEATURES}
    session_state: list[str | None] = [None] * n
    valid = np.zeros(n, dtype=bool)
    reasons: list[str | None] = [None] * n
    validity_rows: list[dict[str, Any]] = []

    # positional index within each trading day
    positions = np.zeros(n, dtype=np.int64)
    day_start = 0
    for index in range(n):
        if index > 0 and days[index] != days[index - 1]:
            day_start = index
        positions[index] = index - day_start

    for t in range(n):
        session, bar_position = _session_facts(closes.iloc[t])
        session_state[t] = session
        if positions[t] < _N:
            reasons[t] = "insufficient_trading_day_lookback"
            continue
        first = t - _N
        if int(bar_index[t]) - int(bar_index[first]) != _N:
            reasons[t] = "lookback_window_gap"
            continue
        window = slice(first, t + 1)  # the 13 source bars t−N..t
        incomplete = np.flatnonzero(observed[window] != expected[window])
        if len(incomplete):
            reasons[t] = "source_bar_incomplete"
            for offset in incomplete:
                source = first + int(offset)
                validity_rows.append(
                    {
                        "row_id": str(bar_ids[t]),
                        "offending_bar_id": str(bar_ids[source]),
                        "observed_1m_count": int(observed[source]),
                        "expected_1m_count": int(expected[source]),
                        "reason": "source_bar_incomplete",
                    }
                )
            continue
        prior = slice(first, t)  # bars t−N..t−1
        recent = slice(first + 1, t + 1)  # bars t−N+1..t
        deltas = np.diff(close[window])  # Δc_i, i in t−N+1..t
        feature_values["cbp_realized_range_12"][t] = float(np.mean(ranges[recent]))
        feature_values["cbp_realized_volatility_12"][t] = float(np.std(deltas, ddof=STD_DDOF))
        prior_range_mean = float(np.mean(ranges[prior]))
        feature_values["cbp_range_compression_ratio_12"][t] = (
            float(ranges[t] / prior_range_mean) if prior_range_mean != 0.0 else np.nan
        )
        path = float(np.sum(np.abs(deltas)))
        feature_values["cbp_path_efficiency_12"][t] = (
            float(abs(close[t] - close[first]) / path) if path != 0.0 else np.nan
        )
        prior_volume = volume[prior]
        volume_std = float(np.std(prior_volume, ddof=STD_DDOF))
        feature_values["cbp_volume_intensity_zscore_12"][t] = (
            float((volume[t] - float(np.mean(prior_volume))) / volume_std)
            if volume_std != 0.0
            else np.nan
        )
        feature_values["cbp_bar_position_in_session"][t] = bar_position
        valid[t] = True

    panel = pd.DataFrame(
        {
            "row_id": bar_ids,
            "trading_day": days,
            "bar_index": bar_index,
            "bar_open_ts_utc": [ts.isoformat() for ts in opens],
            "bar_close_ts_utc": [ts.isoformat() for ts in closes],
            "interval_seconds": np.full(n, interval, dtype=np.int64),
            "open_ticks": frame["open_ticks"].to_numpy(dtype=np.int64),
            "high_ticks": frame["high_ticks"].to_numpy(dtype=np.int64),
            "low_ticks": frame["low_ticks"].to_numpy(dtype=np.int64),
            "close_ticks": frame["close_ticks"].to_numpy(dtype=np.int64),
            "volume": frame["volume"].to_numpy(dtype=np.int64),
            "trade_count": frame["trade_count"].to_numpy(dtype=np.int64),
            "observed_1m_count": observed,
            "expected_1m_count": expected,
        }
    )
    for name in CONTEXT_BAR_PANEL_FEATURES:
        if name == "cbp_session_state":
            panel[name] = [
                state if valid[index] else None for index, state in enumerate(session_state)
            ]
        else:
            panel[name] = feature_values[name]
    panel["cbp_valid"] = valid
    panel["cbp_missing_reason"] = reasons
    unknown = set(r for r in reasons if r is not None) - set(PANEL_MISSING_REASONS)
    if unknown:  # pragma: no cover - closed vocabulary
        raise AssertionError(f"unregistered panel reasons {sorted(unknown)}")
    validity = pd.DataFrame(
        validity_rows,
        columns=["row_id", "offending_bar_id", "observed_1m_count", "expected_1m_count", "reason"],
    )
    return panel, validity


# ── artifact bytes ───────────────────────────────────────────────────────────


def _panel_for_schema(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.loc[:, list(CONTEXT_BAR_PANEL_SCHEMA.names)].copy()
    out["cbp_session_state"] = out["cbp_session_state"].astype(object).where(
        out["cbp_session_state"].notna(), None
    )
    out["cbp_missing_reason"] = out["cbp_missing_reason"].astype(object).where(
        out["cbp_missing_reason"].notna(), None
    )
    return out


def panel_table_bytes(frame: pd.DataFrame) -> bytes:
    return frame_to_arrow_bytes(_panel_for_schema(frame), CONTEXT_BAR_PANEL_SCHEMA)


def validity_table_bytes(frame: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(
        frame.loc[:, list(CONTEXT_BAR_PANEL_VALIDITY_SCHEMA.names)],
        schema=CONTEXT_BAR_PANEL_VALIDITY_SCHEMA,
        preserve_index=False,
    ).replace_schema_metadata(None)
    sink = BytesIO()
    pq.write_table(table, sink, compression="zstd")
    return sink.getvalue()


def _validity_from_bytes(data: bytes) -> pd.DataFrame:
    return pq.read_table(BytesIO(data)).to_pandas()


def _resolved_panel_block():
    from .feature_blocks import resolve_available_block  # noqa: PLC0415

    return resolve_available_block(CONTEXT_BAR_PANEL_BLOCK_KEY)


def _manifest_entry(replay: VerifiedReplayChartArtifact, name: str) -> dict[str, Any]:
    entries = [
        entry
        for entry in replay.manifest.get("artifacts", ())
        if str(entry.get("path")) == name
    ]
    if len(entries) != 1:
        raise ValueError(f"replay-chart manifest has no unique {name} entry")
    return entries[0]


def materialize_context_bar_panel(
    replay: VerifiedReplayChartArtifact,
    *,
    panel_interval_seconds: int,
    lookback_bars: int = LOOKBACK_BARS,
    resolved_block=None,
) -> tuple[ContextBarPanelArtifactEnvelope, pd.DataFrame, pd.DataFrame]:
    """The ONLY input is a verified replay-chart artifact; the bars are
    re-read from its directory and rehashed against the manifest."""

    if not isinstance(replay, VerifiedReplayChartArtifact):
        raise TypeError("materialize_context_bar_panel requires a VerifiedReplayChartArtifact")
    interval = assert_panel_interval_registered(panel_interval_seconds)
    if interval not in REPLAY_TIMEFRAMES_SECONDS:  # pragma: no cover - leaf invariant
        raise ValueError("panel interval is not a ratified replay timeframe")
    entry = _manifest_entry(replay, "bars_tf.parquet")
    bars_path = (Path(replay.directory) / "bars_tf.parquet").resolve()
    try:
        bars_path.relative_to(Path(replay.directory).resolve())
    except ValueError as error:  # pragma: no cover - defensive
        raise ValueError("bars_tf.parquet escapes the replay-chart artifact") from error
    if not bars_path.is_file():
        raise ValueError("replay-chart artifact lacks bars_tf.parquet on disk")
    if bars_path.stat().st_size != int(entry.get("bytes", -1)):
        raise ValueError("bars_tf.parquet byte size disagrees with the manifest")
    bars_sha256 = file_sha256(bars_path)
    if bars_sha256 != entry.get("sha256"):
        raise ValueError("bars_tf.parquet bytes do not hash to the manifest entry")
    bars = pd.read_parquet(bars_path)
    if "rows" in entry and len(bars) != int(entry["rows"]):
        raise ValueError("bars_tf.parquet row count disagrees with the manifest")
    block = resolved_block if resolved_block is not None else _resolved_panel_block()
    if tuple(block.payload.feature_names) != CONTEXT_BAR_PANEL_FEATURES:
        raise ValueError("resolved block does not name the seven panel features in order")
    if block.payload.feature_block_key != CONTEXT_BAR_PANEL_BLOCK_KEY:
        raise ValueError("resolved block is not the context-bar panel block")
    effective = replay.manifest.get("effective_config") or {}
    if effective.get("resample_rule") != PANEL_RESAMPLE_RULE_ID:
        raise ValueError("replay-chart resample rule is not the ratified policy")

    panel, validity = compute_context_bar_panel_features(
        bars, interval_seconds=interval, lookback_bars=lookback_bars
    )
    all_interval = bars[bars["timeframe_seconds"].astype(int) == interval]
    excluded_partial = int(all_interval["is_final_partial"].astype(bool).sum())
    pair = replay.source_pair
    payload = ContextBarPanelArtifactPayload(
        replay_chart_artifact_id=replay.artifact_id,
        replay_chart_manifest_payload_sha256=str(replay.manifest["manifest_payload_sha256"]),
        replay_chart_artifact_kind=str(replay.manifest.get("artifact_kind")),
        source_pair=PanelSourcePairRef(
            profile_name=pair.profile_name,
            v2_dataset_id=pair.v2_dataset_id,
            v2_manifest_hash=pair.v2_manifest_hash,
            v3_dataset_id=pair.v3_dataset_id,
            v3_manifest_hash=pair.v3_manifest_hash,
        ),
        bars_tf_sha256=bars_sha256,
        resample_rule_id=PANEL_RESAMPLE_RULE_ID,
        panel_interval_seconds=interval,
        resolved_feature_block_id=block.resolved_feature_block_id,
        formula_version=CONTEXT_BAR_PANEL_FORMULA_VERSION,
        materializer_version=CONTEXT_BAR_PANEL_MATERIALIZER_VERSION,
        panel_as_of_policy_id=PANEL_AS_OF_POLICY_ID_V1,
        partial_bar_policy_id=PARTIAL_BAR_POLICY_ID,
        lookback_bars=_N,
        minimum_source_bars=MINIMUM_SOURCE_BARS,
        std_ddof=STD_DDOF,
        lookback_policy_id=LOOKBACK_POLICY_ID,
        session_reset_policy_id=SESSION_RESET_POLICY_ID,
        warmup_policy_id=WARMUP_POLICY_ID,
        session_scheme_id=SESSION_SCHEME_ID,
        intensity_source_field=INTENSITY_SOURCE_FIELD,
        panel_table_schema_hash=CONTEXT_BAR_PANEL_SCHEMA_HASH,
        panel_validity_schema_hash=CONTEXT_BAR_PANEL_VALIDITY_SCHEMA_HASH,
    )
    panel_bytes = panel_table_bytes(panel)
    validity_bytes = validity_table_bytes(validity)
    envelope = ContextBarPanelArtifactEnvelope.from_payload(
        payload,
        panel_table_sha256=bytes_sha256(panel_bytes),
        validity_table_sha256=bytes_sha256(validity_bytes),
        row_count=int(len(panel)),
        validity_row_count=int(len(validity)),
        excluded_partial_bar_count=excluded_partial,
        first_bar_close_ts_utc=str(panel["bar_close_ts_utc"].iloc[0]) if len(panel) else None,
        last_bar_close_ts_utc=str(panel["bar_close_ts_utc"].iloc[-1]) if len(panel) else None,
    )
    return envelope, panel, validity


def verify_context_bar_panel_frame(
    envelope: ContextBarPanelArtifactEnvelope, frame: pd.DataFrame
) -> None:
    """The frame IS the artifact's table, or refuse (rehash, never assert)."""

    if bytes_sha256(panel_table_bytes(frame)) != envelope.panel_table_sha256:
        raise ValueError(
            "the supplied panel frame does not hash to the artifact's panel_table_sha256"
        )


def save_context_bar_panel_artifact(
    root: Path,
    envelope: ContextBarPanelArtifactEnvelope,
    panel: pd.DataFrame,
    validity: pd.DataFrame,
):
    panel_bytes = panel_table_bytes(panel)
    validity_bytes = validity_table_bytes(validity)
    if bytes_sha256(panel_bytes) != envelope.panel_table_sha256:
        raise ValueError("panel frame does not hash to the envelope's panel_table_sha256")
    if bytes_sha256(validity_bytes) != envelope.validity_table_sha256:
        raise ValueError("validity frame does not hash to the envelope's validity_table_sha256")
    return save_or_reuse_envelope(
        Path(root),
        CONTEXT_BAR_PANEL_STORE,
        envelope,
        extra_files={PANEL_SIDECAR: panel_bytes, VALIDITY_SIDECAR: validity_bytes},
    )


def load_context_bar_panel_artifact(
    root: Path, artifact_id: str
) -> ContextBarPanelArtifactEnvelope:
    return load_verified_envelope(
        Path(root), CONTEXT_BAR_PANEL_STORE, artifact_id, ContextBarPanelArtifactEnvelope
    )


def load_context_bar_panel_frame(
    root: Path, envelope: ContextBarPanelArtifactEnvelope
) -> pd.DataFrame:
    data = load_sidecar_bytes(
        Path(root), CONTEXT_BAR_PANEL_STORE, envelope.context_bar_panel_artifact_id, PANEL_SIDECAR
    )
    if bytes_sha256(data) != envelope.panel_table_sha256:
        raise ValueError("stored panel table fails the envelope hash check")
    frame = frame_from_arrow_bytes(data)
    if len(frame) != envelope.row_count:
        raise ValueError("stored panel row count disagrees with the envelope")
    return frame


def load_context_bar_panel_validity(
    root: Path, envelope: ContextBarPanelArtifactEnvelope
) -> pd.DataFrame:
    data = load_sidecar_bytes(
        Path(root),
        CONTEXT_BAR_PANEL_STORE,
        envelope.context_bar_panel_artifact_id,
        VALIDITY_SIDECAR,
    )
    if bytes_sha256(data) != envelope.validity_table_sha256:
        raise ValueError("stored validity table fails the envelope hash check")
    frame = _validity_from_bytes(data)
    if len(frame) != envelope.validity_row_count:
        raise ValueError("stored validity row count disagrees with the envelope")
    return frame


def load_verified_context_bar_panel(
    root: Path, artifact_id: str
) -> tuple[ContextBarPanelArtifactEnvelope, pd.DataFrame]:
    """Envelope + manifest-verified panel frame (the seam every regime
    consumer uses — a caller-provided frame is never trusted)."""

    envelope = load_context_bar_panel_artifact(root, artifact_id)
    return envelope, load_context_bar_panel_frame(root, envelope)


def _example_payload() -> ContextBarPanelArtifactPayload:
    return ContextBarPanelArtifactPayload(
        replay_chart_artifact_id="a" * 64,
        replay_chart_manifest_payload_sha256="b" * 64,
        replay_chart_artifact_kind="ifvg_replay_chart_v1",
        source_pair=PanelSourcePairRef(
            profile_name="ifvg_v2_doc_default_fresh_static_1r",
            v2_dataset_id="1" * 64,
            v2_manifest_hash="2" * 64,
            v3_dataset_id="3" * 64,
            v3_manifest_hash="4" * 64,
        ),
        bars_tf_sha256="c" * 64,
        resample_rule_id=PANEL_RESAMPLE_RULE_ID,
        panel_interval_seconds=300,
        resolved_feature_block_id="d" * 64,
        formula_version=CONTEXT_BAR_PANEL_FORMULA_VERSION,
        materializer_version=CONTEXT_BAR_PANEL_MATERIALIZER_VERSION,
        panel_as_of_policy_id=PANEL_AS_OF_POLICY_ID_V1,
        partial_bar_policy_id=PARTIAL_BAR_POLICY_ID,
        lookback_bars=LOOKBACK_BARS,
        minimum_source_bars=MINIMUM_SOURCE_BARS,
        std_ddof=STD_DDOF,
        lookback_policy_id=LOOKBACK_POLICY_ID,
        session_reset_policy_id=SESSION_RESET_POLICY_ID,
        warmup_policy_id=WARMUP_POLICY_ID,
        session_scheme_id=SESSION_SCHEME_ID,
        intensity_source_field=INTENSITY_SOURCE_FIELD,
        panel_table_schema_hash=CONTEXT_BAR_PANEL_SCHEMA_HASH,
        panel_validity_schema_hash=CONTEXT_BAR_PANEL_VALIDITY_SCHEMA_HASH,
    )


register_identity_pair(
    name="ContextBarPanelArtifact",
    envelope_cls=ContextBarPanelArtifactEnvelope,
    payload_cls=ContextBarPanelArtifactPayload,
    id_field="context_bar_panel_artifact_id",
    example_factory=_example_payload,
    extra_envelope_fields=(
        "panel_table_sha256",
        "validity_table_sha256",
        "row_count",
        "validity_row_count",
        "excluded_partial_bar_count",
        "first_bar_close_ts_utc",
        "last_bar_close_ts_utc",
    ),
)

# static guard: the schema names the seven features + validity fields exactly
_schema_features = tuple(name for name in CONTEXT_BAR_PANEL_SCHEMA.names if name.startswith("cbp_"))
if _schema_features != (*CONTEXT_BAR_PANEL_FEATURES, "cbp_valid", "cbp_missing_reason"):
    raise AssertionError("the panel schema must carry the seven features + validity fields")
