"""Ordered trade prints for the candles a position is open in.

Immutable market input shared by every configuration-and-firm comparison in a
worker process (one day is loaded once and read by all of them; nothing here is
mutable account or strategy state).

For each one-minute candle Strategy-Core steps, the prints of the front-month
contract inside the candle's scheduled window ``[open, close)`` are accepted as
the position's price observations only when they rebuild that candle exactly
(open, high, low, close and print count). Otherwise the minute uses the
declared one-minute approximation and is counted as approximated. Nothing is
fetched and nothing is written.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from alpha_lab.propsim.funded.clock import to_ns
from alpha_lab.propsim.funded.position_walk import MinuteObservations, minute_approximation
from alpha_lab.propsim.funded.price_evidence import DATA_ROOT, PrintDay, load_print_day

__all__ = ["DayPrints", "NoPrints", "bar_window_ns"]


def bar_window_ns(bar) -> tuple[int, int]:
    opened = bar.logical_open_ts_utc or bar.open_ts_utc
    closed = bar.logical_close_ts_utc or bar.close_ts_utc
    return to_ns(opened), to_ns(closed)


class DayPrints:
    """Prints of the UTC day files spanning one strategy trading day."""

    def __init__(self, first_ns: int, last_ns: int, *, data_root: Path = DATA_ROOT) -> None:
        self.days: list[PrintDay] = []
        self.missing: list[str] = []
        cursor = datetime.fromtimestamp(first_ns // 1_000_000_000, tz=UTC).date()
        last = datetime.fromtimestamp(last_ns // 1_000_000_000, tz=UTC).date()
        while cursor <= last:
            loaded = load_print_day(data_root, cursor)
            if loaded is None:
                self.missing.append(cursor.isoformat())
            else:
                self.days.append(loaded)
            cursor = date.fromordinal(cursor.toordinal() + 1)
        self._cache: dict[int, MinuteObservations | None] = {}
        self.minutes_checked = 0
        self.minutes_matched = 0

    @property
    def files(self) -> list[dict]:
        return [{"file": d.file, "utc_date": d.utc_date, "symbol": d.symbol,
                 "trade_prints": int(len(d.ts_ns))} for d in self.days]

    def _prints(self, start_ns: int, end_ns: int) -> tuple[np.ndarray, np.ndarray] | None:
        parts_ts, parts_px, symbols = [], [], set()
        for day in self.days:
            if not len(day.ts_ns) or day.ts_ns[-1] < start_ns or day.ts_ns[0] >= end_ns:
                continue
            lo = np.searchsorted(day.ts_ns, start_ns, side="left")
            hi = np.searchsorted(day.ts_ns, end_ns, side="left")
            if hi > lo:
                parts_ts.append(day.ts_ns[lo:hi])
                parts_px.append(day.ticks[lo:hi])
                symbols.add(day.symbol)
        if len(symbols) > 1:
            return None  # a contract roll inside one minute is not reconstructed
        if not parts_ts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        return np.concatenate(parts_ts), np.concatenate(parts_px)

    def exact_minute(self, bar) -> MinuteObservations | None:
        """Prints of this candle if they rebuild it exactly, else None (cached)."""

        open_ns, close_ns = bar_window_ns(bar)
        if open_ns in self._cache:
            return self._cache[open_ns]
        self.minutes_checked += 1
        found = self._prints(open_ns, close_ns)
        obs = None
        if found is not None:
            ts, px = found
            rebuilt = ((int(px[0]), int(px.max()), int(px.min()), int(px[-1]), int(len(px)))
                       if len(px) else None)
            wanted = (int(bar.open_ticks), int(bar.high_ticks), int(bar.low_ticks),
                      int(bar.close_ticks), int(bar.trade_count))
            if rebuilt == wanted:
                self.minutes_matched += 1
                obs = MinuteObservations(
                    open_ns=open_ns, close_ns=close_ns, close_ticks=int(bar.close_ticks),
                    ts_ns=ts.astype(np.int64), price_ticks=px.astype(np.int64),
                    continuous=np.zeros(len(px), dtype=bool),
                    fidelity="ordered_trade_prints",
                )
        self._cache[open_ns] = obs
        return obs

    def minute(self, bar, sign: int) -> MinuteObservations:
        exact = self.exact_minute(bar)
        if exact is not None:
            return exact
        open_ns, close_ns = bar_window_ns(bar)
        return minute_approximation(open_ns, close_ns, int(bar.open_ticks),
                                    int(bar.high_ticks), int(bar.low_ticks),
                                    int(bar.close_ticks), sign)


class NoPrints:
    """Declared minute-candle approximation for every minute (engineering only)."""

    files: list[dict] = []
    missing: list[str] = []
    minutes_checked = 0
    minutes_matched = 0

    def exact_minute(self, bar):  # noqa: D401 - same interface
        return None

    def minute(self, bar, sign: int) -> MinuteObservations:
        open_ns, close_ns = bar_window_ns(bar)
        return minute_approximation(open_ns, close_ns, int(bar.open_ticks),
                                    int(bar.high_ticks), int(bar.low_ticks),
                                    int(bar.close_ticks), sign)

