"""Ordered trade-print paths from local MBP-1 files, with bar reconciliation.

For each strategy execution this module reads the exchange trade prints
(``action == 'T'``) of the dominant non-spread contract in each UTC day file —
the same front-month rule the Core's Databento reader uses — keeps them in
source order ``(ts_event, sequence, file row)``, and builds the path the
account engine checks.

The prints are ACCEPTED as the execution's price evidence only when every
one-minute candle rebuilt from them over the whole position (entry minute to
resolution minute) equals the study's own saved candle (open, high, low,
close and print count). A mismatch or missing file is recorded; depending on
the plan's policy the execution then fails the run or falls back to a labeled
minute-candle approximation. No data is fetched and nothing is written.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from alpha_lab.propsim.funded.clock import NS, to_ns
from alpha_lab.propsim.funded.paths import (
    OBS_EXIT_TOUCH,
    OBS_PRINT,
    ExecutionPath,
    StrategyExecution,
    minute_scenario_path,
)

__all__ = [
    "DATA_ROOT",
    "PrintDay",
    "load_print_day",
    "build_paths",
    "PathEvidence",
]

DATA_ROOT = Path(__file__).resolve().parents[4] / "data" / "databento" / "NQ"
MINUTE = 60 * NS
TICK = 0.25


@dataclass(frozen=True)
class PrintDay:
    utc_date: str
    file: str
    file_sha256: str | None
    instrument_id: int
    symbol: str
    ts_ns: np.ndarray
    ticks: np.ndarray


@dataclass(frozen=True)
class PathEvidence:
    trade_id: str
    fidelity: str
    reason: str
    prints: int
    minutes_checked: int
    minutes_matched: int
    symbol: str | None
    ordering_note: str | None


def _day_file(root: Path, utc_day: date) -> Path | None:
    folder = root / utc_day.isoformat()
    for name in ("mbp10.parquet", "mbp1.parquet", "trades.parquet"):
        if (folder / name).is_file():
            return folder / name
    return None


def load_print_day(root: Path, utc_day: date) -> PrintDay | None:
    path = _day_file(root, utc_day)
    if path is None:
        return None
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    columns = [c for c in ("ts_event", "action", "price", "instrument_id", "symbol",
                           "sequence") if c in names]
    table = parquet.read(columns=columns)
    if "action" in names:
        table = table.filter(pc.equal(pc.cast(table["action"], pa.string()), "T"))
    symbols = pc.cast(table["symbol"], pa.string())
    table = table.filter(pc.invert(pc.fill_null(pc.match_substring(symbols, "-"), False)))
    counts = pc.value_counts(table["instrument_id"])
    ranked = [(c.as_py(), v.as_py()) for v, c in zip(counts.field("values"),
                                                    counts.field("counts"), strict=True)]
    if not ranked:
        return None
    instrument = max(ranked)[1]  # dominant by trade count, ties -> larger id
    table = table.filter(pc.equal(table["instrument_id"], instrument))
    ts = table["ts_event"].cast(pa.timestamp("ns", "UTC")).cast(pa.int64()).to_numpy()
    seq = table["sequence"].to_numpy() if "sequence" in columns else np.zeros(len(ts))
    price = table["price"].to_numpy()
    ticks = np.rint(price / TICK).astype(np.int64)
    if not np.allclose(ticks * TICK, price):
        raise ValueError(f"{path}: trade prices off the 0.25 grid")
    order = np.lexsort((np.arange(len(ts)), seq, ts))  # stable source order
    symbol = str(table["symbol"][0].as_py()) if table.num_rows else ""
    return PrintDay(utc_date=utc_day.isoformat(), file=f"{path.parent.name}/{path.name}",
                    file_sha256=None, instrument_id=int(instrument), symbol=symbol,
                    ts_ns=ts[order], ticks=ticks[order])


def _utc_day(ts_ns: int) -> date:
    return datetime.fromtimestamp(ts_ns // NS, tz=UTC).date()


def _window_prints(days: dict[date, PrintDay | None], start_ns: int, end_ns: int
                   ) -> tuple[np.ndarray, np.ndarray, str | None] | None:
    ts_parts, px_parts, symbols = [], [], set()
    cursor = _utc_day(start_ns)
    last = _utc_day(end_ns)
    while cursor <= last:
        day = days.get(cursor)
        if day is None:
            return None
        lo = np.searchsorted(day.ts_ns, start_ns, side="left")
        hi = np.searchsorted(day.ts_ns, end_ns, side="left")
        ts_parts.append(day.ts_ns[lo:hi])
        px_parts.append(day.ticks[lo:hi])
        symbols.add(day.symbol)
        cursor = date.fromordinal(cursor.toordinal() + 1)
    if len(symbols) != 1:
        return None  # a contract roll inside one position is not reconstructed
    return np.concatenate(ts_parts), np.concatenate(px_parts), symbols.pop()


def _rebuild_minutes(ts: np.ndarray, px: np.ndarray) -> dict[int, tuple[int, int, int, int, int]]:
    out: dict[int, tuple[int, int, int, int, int]] = {}
    if not len(ts):
        return out
    minutes = ts - ts % MINUTE
    boundaries = np.flatnonzero(np.diff(minutes)) + 1
    for chunk_ts, chunk_px in zip(np.split(minutes, boundaries), np.split(px, boundaries),
                                  strict=True):
        out[int(chunk_ts[0])] = (int(chunk_px[0]), int(chunk_px.max()), int(chunk_px.min()),
                                 int(chunk_px[-1]), int(len(chunk_px)))
    return out


def build_paths(
    executions: tuple[StrategyExecution, ...],
    bars: pd.DataFrame,
    *,
    policy: str,
    data_root: Path = DATA_ROOT,
) -> tuple[dict[str, ExecutionPath], list[PathEvidence], list[dict]]:
    """Paths for every non-warmup execution, plus per-trade evidence rows."""

    needed: set[date] = set()
    for execution in executions:
        if execution.is_warmup:
            continue
        needed.add(_utc_day(to_ns(execution.entry_ts_utc) - MINUTE))
        needed.add(_utc_day(to_ns(execution.exit_ts_utc)))
    days: dict[date, PrintDay | None] = {}
    for day in sorted(needed):
        days[day] = load_print_day(data_root, day)
    # fill gaps between needed days (multi-day windows are rare)
    bar_open = bars["open_ns"].to_numpy()
    bar_rows = bars[["open_ticks", "high_ticks", "low_ticks", "close_ticks",
                     "trade_count"]].to_numpy(dtype=np.int64)
    paths: dict[str, ExecutionPath] = {}
    evidence: list[PathEvidence] = []
    source_files: dict[str, dict] = defaultdict(dict)
    for loaded in days.values():
        if loaded is not None:
            source_files[loaded.file] = {"utc_date": loaded.utc_date, "symbol": loaded.symbol,
                                         "instrument_id": loaded.instrument_id,
                                         "trade_prints": int(len(loaded.ts_ns))}
    for execution in executions:
        if execution.is_warmup:
            continue
        entry_ns = to_ns(execution.entry_ts_utc)
        resolution_close = to_ns(execution.exit_ts_utc)  # close of the resolution minute
        window = _window_prints(days, entry_ns, resolution_close)
        lo = np.searchsorted(bar_open, entry_ns, side="left")
        hi = np.searchsorted(bar_open, resolution_close, side="left")
        study_minutes = {int(bar_open[i]): tuple(int(v) for v in bar_rows[i])
                         for i in range(lo, hi)}
        reason, path, note = "", None, None
        matched = 0
        if window is None:
            reason = "trade prints unavailable for the whole position (missing file or roll)"
        else:
            ts, px, symbol = window
            rebuilt = _rebuild_minutes(ts, px)
            matched = sum(1 for k, v in study_minutes.items() if rebuilt.get(k) == v)
            extra = set(rebuilt) - set(study_minutes)
            if matched != len(study_minutes) or extra:
                reason = (f"rebuilt candles differ from the study's candles "
                          f"({matched}/{len(study_minutes)} minutes match)")
            else:
                path, note = _print_path(execution, ts, px, resolution_close)
                reason = "every candle of the position rebuilt exactly from the prints"
        if path is None:
            if policy == "ordered_trade_prints_required":
                evidence.append(PathEvidence(execution.trade_id, "missing", reason, 0,
                                             len(study_minutes), matched, None, None))
                continue
            order = ("favorable_first" if policy == "minute_bars_favorable_first"
                     else "adverse_first")
            minute_rows = [(k, k + MINUTE, v[0], v[1], v[2], v[3])
                           for k, v in sorted(study_minutes.items())]
            path = minute_scenario_path(execution, minute_rows, order=order)
        evidence.append(PathEvidence(
            trade_id=execution.trade_id, fidelity=path.fidelity, reason=reason,
            prints=int(len(path.ts_ns)) if path.fidelity == "ordered_trade_prints" else 0,
            minutes_checked=len(study_minutes), minutes_matched=matched,
            symbol=window[2] if window else None, ordering_note=note,
        ))
        paths[execution.trade_id] = path
    return paths, evidence, [{"file": k, **v} for k, v in sorted(source_files.items())]


def _print_path(execution: StrategyExecution, ts: np.ndarray, px: np.ndarray,
                resolution_close: int) -> tuple[ExecutionPath | None, str | None]:
    sign = execution.sign
    offsets = (px - execution.entry_ticks) * sign
    stop_off = (execution.stop_ticks - execution.entry_ticks) * sign  # negative
    target_off = (execution.target_ticks - execution.entry_ticks) * sign
    note = None
    if execution.exit_reason == "scheduled_close":
        if not len(px) or int(px[-1]) != execution.exit_ticks:
            return None, "scheduled exit price differs from the last print"
        cut = len(px)
        kinds = np.full(cut, OBS_PRINT, dtype=np.int8)
    else:
        resolution_open = resolution_close - MINUTE
        stop_hits = np.flatnonzero(offsets <= stop_off)
        target_hits = np.flatnonzero(offsets >= target_off)
        first_stop = int(stop_hits[0]) if len(stop_hits) else None
        first_target = int(target_hits[0]) if len(target_hits) else None
        chosen = first_stop if execution.exit_reason == "stop" else first_target
        if chosen is None or not (resolution_open <= int(ts[chosen]) < resolution_close):
            return None, "the recorded exit level is not first reached in the resolution minute"
        other = first_target if execution.exit_reason == "stop" else first_stop
        if other is not None and other < chosen:
            # the study's one-minute stop-first rule resolved a minute where the
            # prints show the other level first; the strategy result is kept
            # (the account path keeps every print while the recorded position existed)
            note = ("prints reach the other exit level first inside the resolution "
                    "minute; the study's one-minute stop-first result is kept")
        cut = chosen + 1
        kinds = np.full(cut, OBS_PRINT, dtype=np.int8)
        kinds[-1] = OBS_EXIT_TOUCH
    return ExecutionPath(
        trade_id=execution.trade_id, fidelity="ordered_trade_prints",
        ts_ns=ts[:cut].astype(np.int64), price_ticks=px[:cut].astype(np.int64),
        continuous=np.zeros(cut, dtype=bool), kind=kinds,
        strategy_exit_ns=int(ts[cut - 1]) if execution.exit_reason != "scheduled_close"
        else resolution_close,
        notes=(note,) if note else (),
    ), note
