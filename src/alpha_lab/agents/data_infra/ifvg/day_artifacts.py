"""Phase A — per-day artifacts: one canonical reader drain per store day.

Produces two parquets per day (profile-independent, parallel-warmable):

* ``ifvg_tbars_<atag>``   — all configured TIME bars (SC ``build_time_bars_from_frame``),
  serialized field-for-field so :class:`strategy_core.types.Bar` reconstructs
  bit-exactly.
* ``ifvg_levels_<atag>``  — the level timeline: ``StrategyLevelState`` (with NY
  ranges + ``prev_ny_*`` opt-ins) folded over the SAME drained trade stream,
  snapshotted at every 1m bar close.

Trust stamps (parquet metadata): the ENTERING ``prev_full_hl`` and
``prev_ny_hl`` seeds; a file whose stamps don't match the expected seeds is
rebuilt (the prev_full_hl lesson). The file also stamps the day's OWN outputs
(day H/L, NY-session H/L) so the NEXT day's seeds normally come from the prior
artifact without a second store walk; the SC store walks
(``prior_full_day_extremes`` / ``prior_day_session_extremes``) are the
fallback for parallel warmers racing ahead of their neighbors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from strategy_core.candles.time_batch import build_time_bars_from_frame
from strategy_core.constants import RESEARCH_SESSION_SCHEME
from strategy_core.data.databento_parquet import DatabentoParquetSource
from strategy_core.data.events import DataQualityWarning as ScDataQualityWarning
from strategy_core.data.prior_day import (
    prior_day_session_extremes,
    prior_full_day_extremes,
)
from strategy_core.runtime.levels import StrategyLevelState
from strategy_core.types import Bar, BarKind, CloseReason, Level, Side, Trade

from .config import IfvgCaptureConfig

__all__ = [
    "DayArtifacts",
    "DaySeeds",
    "seeds_for_day",
    "build_day_artifacts",
    "write_day_artifacts",
    "load_day_artifacts",
    "levels_for_from_frame",
]

_META_KEY = b"ifvg_artifacts_meta"


@dataclass(frozen=True)
class DaySeeds:
    """The ENTERING seeds for one day (both None = cold start / first store day)."""

    prev_day: date | None
    prev_full_hl: tuple[int, int] | None  # ticks
    prev_ny_day: date | None
    prev_ny_hl: tuple[int, int] | None  # ticks

    def meta(self) -> dict:
        return {
            "prev_day": self.prev_day.isoformat() if self.prev_day else None,
            "prev_full_hl": list(self.prev_full_hl) if self.prev_full_hl else None,
            "prev_ny_day": self.prev_ny_day.isoformat() if self.prev_ny_day else None,
            "prev_ny_hl": list(self.prev_ny_hl) if self.prev_ny_hl else None,
        }


@dataclass(frozen=True)
class DayArtifacts:
    date_str: str
    bars: list[Bar]
    #: 1m-close-keyed level snapshots: {close_ts_utc: (Level, ...)}.
    level_timeline: dict[datetime, tuple[Level, ...]]
    seeds: DaySeeds
    day_hl: tuple[int, int] | None  # this day's own extremes (ticks)
    ny_hl: tuple[int, int] | None  # this day's NY-session extremes (ticks)
    reader_warnings: tuple[str, ...]


def seeds_for_day(date_str: str, cfg: IfvgCaptureConfig) -> DaySeeds:
    """Entering seeds — prior artifact's stamped outputs when present, else the
    SC store walks (parallel-warmer fallback)."""
    td = date.fromisoformat(date_str)
    symbol_dir = Path(cfg.data_dir) / cfg.symbol
    prior = _prior_artifact_meta(td, cfg)
    if prior is not None:
        prev_day_s, meta = prior
        day_hl = meta.get("day_hl")
        ny_hl = meta.get("ny_hl")
        return DaySeeds(
            prev_day=date.fromisoformat(prev_day_s) if day_hl else None,
            prev_full_hl=tuple(day_hl) if day_hl else None,
            prev_ny_day=date.fromisoformat(prev_day_s) if ny_hl else None,
            prev_ny_hl=tuple(ny_hl) if ny_hl else None,
        )
    full = prior_full_day_extremes(symbol_dir, td, requested_symbol=cfg.symbol)
    sessions = prior_day_session_extremes(
        symbol_dir, td, sessions=("ny",), requested_symbol=cfg.symbol
    )
    ny = sessions.get("ny")
    return DaySeeds(
        prev_day=full.source_day if full else None,
        prev_full_hl=(full.high_ticks, full.low_ticks) if full else None,
        prev_ny_day=ny.source_day if ny else None,
        prev_ny_hl=(ny.high_ticks, ny.low_ticks) if ny else None,
    )


def _prior_artifact_meta(td: date, cfg: IfvgCaptureConfig) -> tuple[str, dict] | None:
    """Most recent prior day (within 10 calendar days) that has a bars artifact."""
    import pyarrow.parquet as pq

    for back in range(1, 11):
        prev = td - timedelta(days=back)
        path = cfg.bars_path(prev.isoformat())
        if not path.exists():
            continue
        try:
            md = pq.read_metadata(path).metadata or {}
            meta = json.loads(md.get(_META_KEY, b"{}"))
        except Exception:
            return None
        if meta.get("day_hl"):
            return prev.isoformat(), meta
        continue  # empty day (no trades) — keep walking back, like the SC walk
    return None


def build_day_artifacts(date_str: str, cfg: IfvgCaptureConfig, seeds: DaySeeds) -> DayArtifacts:
    """ONE drain of the canonical reader -> bars + level timeline + own extremes."""
    td = date.fromisoformat(date_str)
    source = DatabentoParquetSource.for_trading_day(
        Path(cfg.data_dir) / cfg.symbol, td, requested_symbol=cfg.symbol
    )
    warnings = tuple(w.message for w in source.pending_warnings)

    level_state = StrategyLevelState(
        tick_size=cfg.tick_size,
        session_range_names=("asia", "london", "ny"),
        emit_prior_session_levels=("ny",),
    )
    if seeds.prev_day is not None and seeds.prev_full_hl is not None:
        level_state.load_prior_day_summary(
            seeds.prev_day, high_ticks=seeds.prev_full_hl[0], low_ticks=seeds.prev_full_hl[1]
        )
    if seeds.prev_ny_day is not None and seeds.prev_ny_hl is not None:
        level_state.load_prior_session_range(
            seeds.prev_ny_day, "ny", high_ticks=seeds.prev_ny_hl[0], low_ticks=seeds.prev_ny_hl[1]
        )

    trades: list[Trade] = []
    for event in source.events():
        if isinstance(event, ScDataQualityWarning) or not isinstance(event, Trade):
            continue
        trades.append(event)

    if not trades:
        return DayArtifacts(date_str, [], {}, seeds, None, None, warnings)

    frame = pd.DataFrame(
        {
            "ts_event": [t.event_ts_utc for t in trades],
            "price": [t.price_ticks * cfg.tick_size for t in trades],
            "size": [t.size for t in trades],
        }
    )
    bars = build_time_bars_from_frame(
        frame, cfg.timeframes_seconds(), scheme=RESEARCH_SESSION_SCHEME, tick_size=cfg.tick_size
    )

    # Level timeline: fold the SAME trade stream, snapshot at every 1m close.
    bars_1m = [b for b in bars if b.timeframe_ticks == 60]
    timeline: dict[datetime, tuple[Level, ...]] = {}
    idx = 0
    n = len(trades)
    for bar in bars_1m:
        while idx < n and trades[idx].event_ts_utc <= bar.close_ts_utc:
            level_state.process_trade(trades[idx])
            idx += 1
        timeline[bar.close_ts_utc] = level_state.levels()

    day_hl = (
        max(b.high_ticks for b in bars_1m),
        min(b.low_ticks for b in bars_1m),
    )
    ny_bars = [
        b
        for b in bars_1m
        if _session_of_bucket_start(b.open_ts_utc) == "ny" and b.trading_day == td
    ]
    ny_hl = (
        (max(b.high_ticks for b in ny_bars), min(b.low_ticks for b in ny_bars))
        if ny_bars
        else None
    )
    return DayArtifacts(date_str, bars, timeline, seeds, day_hl, ny_hl, warnings)


def _session_of_bucket_start(ts_utc: datetime) -> str:
    from strategy_core.decisions.sessions import classify_session

    return classify_session(ts_utc, RESEARCH_SESSION_SCHEME).session


# ── serialization ─────────────────────────────────────────────────────────────

_BAR_COLUMNS = (
    "timeframe_ticks",
    "trading_day",
    "bar_index",
    "bar_id",
    "open_ts_utc",
    "close_ts_utc",
    "open_ticks",
    "high_ticks",
    "low_ticks",
    "close_ticks",
    "volume",
    "trade_count",
    "is_complete",
    "is_partial",
    "close_reason",
    "kind",
)


def write_day_artifacts(artifacts: DayArtifacts, cfg: IfvgCaptureConfig) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta = {
        "seeds": artifacts.seeds.meta(),
        "day_hl": list(artifacts.day_hl) if artifacts.day_hl else None,
        "ny_hl": list(artifacts.ny_hl) if artifacts.ny_hl else None,
        "warnings": list(artifacts.reader_warnings),
    }
    meta_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")

    bars_frame = pd.DataFrame(
        {
            "timeframe_ticks": [b.timeframe_ticks for b in artifacts.bars],
            "trading_day": [b.trading_day.isoformat() for b in artifacts.bars],
            "bar_index": [b.bar_index for b in artifacts.bars],
            "bar_id": [b.bar_id for b in artifacts.bars],
            "open_ts_utc": [b.open_ts_utc for b in artifacts.bars],
            "close_ts_utc": [b.close_ts_utc for b in artifacts.bars],
            "open_ticks": [b.open_ticks for b in artifacts.bars],
            "high_ticks": [b.high_ticks for b in artifacts.bars],
            "low_ticks": [b.low_ticks for b in artifacts.bars],
            "close_ticks": [b.close_ticks for b in artifacts.bars],
            "volume": [b.volume for b in artifacts.bars],
            "trade_count": [b.trade_count for b in artifacts.bars],
            "is_complete": [b.is_complete for b in artifacts.bars],
            "is_partial": [b.is_partial for b in artifacts.bars],
            "close_reason": [
                b.close_reason.value if b.close_reason else None for b in artifacts.bars
            ],
            "kind": [b.kind.value for b in artifacts.bars],
        }
    )
    table = pa.Table.from_pandas(bars_frame, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), _META_KEY: meta_bytes})
    path = cfg.bars_path(artifacts.date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)

    rows = []
    for ts, levels in artifacts.level_timeline.items():
        for level in levels:
            rows.append(
                {
                    "close_ts_utc": ts,
                    "name": level.name,
                    "price": level.price,
                    "side": level.side.value,
                    "available_from": level.available_from,
                }
            )
    levels_frame = pd.DataFrame(
        rows, columns=["close_ts_utc", "name", "price", "side", "available_from"]
    )
    ltable = pa.Table.from_pandas(levels_frame, preserve_index=False)
    ltable = ltable.replace_schema_metadata(
        {**(ltable.schema.metadata or {}), _META_KEY: meta_bytes}
    )
    pq.write_table(ltable, cfg.levels_path(artifacts.date_str))


def load_day_artifacts(
    date_str: str, cfg: IfvgCaptureConfig, *, expected_seeds: DaySeeds | None
) -> DayArtifacts | None:
    """Load a trusted artifact pair, or ``None`` (missing / seed mismatch)."""
    import pyarrow.parquet as pq

    bars_path = cfg.bars_path(date_str)
    levels_path = cfg.levels_path(date_str)
    if not bars_path.exists() or not levels_path.exists():
        return None
    try:
        md = pq.read_metadata(bars_path).metadata or {}
        meta = json.loads(md.get(_META_KEY, b"{}"))
    except Exception:
        return None
    if expected_seeds is not None and meta.get("seeds") != expected_seeds.meta():
        return None

    bars_frame = pd.read_parquet(bars_path)
    bars = [
        Bar(
            timeframe_ticks=int(r.timeframe_ticks),
            trading_day=date.fromisoformat(r.trading_day),
            bar_index=int(r.bar_index),
            bar_id=r.bar_id,
            open_ts_utc=r.open_ts_utc.to_pydatetime(),
            close_ts_utc=r.close_ts_utc.to_pydatetime(),
            open_ticks=int(r.open_ticks),
            high_ticks=int(r.high_ticks),
            low_ticks=int(r.low_ticks),
            close_ticks=int(r.close_ticks),
            volume=int(r.volume),
            trade_count=int(r.trade_count),
            is_complete=bool(r.is_complete),
            is_partial=bool(r.is_partial),
            close_reason=CloseReason(r.close_reason) if pd.notna(r.close_reason) else None,
            kind=BarKind(r.kind),
        )
        for r in bars_frame.itertuples(index=False)
    ]
    levels_frame = pd.read_parquet(levels_path)
    timeline: dict[datetime, tuple[Level, ...]] = {}
    if len(levels_frame):
        for ts, group in levels_frame.groupby("close_ts_utc", sort=True):
            timeline[ts.to_pydatetime()] = tuple(
                Level(
                    name=r.name,
                    price=float(r.price),
                    side=Side(r.side),
                    available_from=(
                        r.available_from.to_pydatetime() if pd.notna(r.available_from) else None
                    ),
                )
                for r in group.itertuples(index=False)
            )
    seeds_meta = meta.get("seeds", {})
    seeds = DaySeeds(
        prev_day=(
            date.fromisoformat(seeds_meta["prev_day"]) if seeds_meta.get("prev_day") else None
        ),
        prev_full_hl=(
            tuple(seeds_meta["prev_full_hl"]) if seeds_meta.get("prev_full_hl") else None
        ),
        prev_ny_day=(
            date.fromisoformat(seeds_meta["prev_ny_day"]) if seeds_meta.get("prev_ny_day") else None
        ),
        prev_ny_hl=(tuple(seeds_meta["prev_ny_hl"]) if seeds_meta.get("prev_ny_hl") else None),
    )
    return DayArtifacts(
        date_str=date_str,
        bars=bars,
        level_timeline=timeline,
        seeds=seeds,
        day_hl=tuple(meta["day_hl"]) if meta.get("day_hl") else None,
        ny_hl=tuple(meta["ny_hl"]) if meta.get("ny_hl") else None,
        reader_warnings=tuple(meta.get("warnings", ())),
    )


def levels_for_from_frame(timeline: dict[datetime, tuple[Level, ...]]):
    """The ``run_day`` ``levels_for`` closure over a loaded timeline (exact-match
    on the 1m close instants; both sides use the same batch bars, so the keys
    align by construction — a miss returns the empty tuple)."""

    def _levels_for(ts_utc: datetime) -> tuple[Level, ...]:
        return timeline.get(ts_utc, ())

    return _levels_for
