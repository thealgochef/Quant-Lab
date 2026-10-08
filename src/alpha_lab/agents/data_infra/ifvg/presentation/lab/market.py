"""Index measures from the study's stored one-minute E-mini bars (read only).

The bars come from the funded plan's bound, verified strategy package
(``shared/market_bars.parquet``; the package is opened through
:func:`plan_strategy_package`, which checks its run id and manifest hash).
Nothing is downloaded and no day after the study's cutoff is read.

Definitions (``docs/ifvg-redesign-fixes/followup-1/CALCULATION_DEFINITIONS.md``, which
corrects ``CALCULATIONS.md``): daily close = the last one-minute close of each trading
day; beta = least-squares slope of daily results on the daily E-mini change × $20;
buy and hold one E-mini from ONE entry instant, the first study trading day's open, to
the last study trading day's close, with its first $2,000 fall from a running high on
the same one-minute closes (:data:`BENCHMARK_VERSION`); two separately versioned
market-condition label sets — retrospective (each day's own close, whole-study
threshold) and known at entry (only closes completed before each day's open).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import time as dt_time
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "CONDITIONS",
    "ENTRY_KNOWN",
    "ENTRY_KNOWN_VERSION",
    "ENTRY_MEASURES",
    "E_MINI_POINT_VALUE",
    "MIN_HISTORY",
    "NOT_ENOUGH",
    "RETROSPECTIVE",
    "RETROSPECTIVE_VERSION",
    "BuyAndHold",
    "ConditionCard",
    "ConditionLabels",
    "EntryKnownDay",
    "EntryMeasure",
    "IndexTie",
    "Stretch",
    "buy_and_hold",
    "condition_cards",
    "condition_labels",
    "daily_close_times",
    "daily_closes",
    "entry_known_days",
    "entry_known_labels",
    "entry_measure",
    "index_tie",
    "load_index_minutes",
    "load_study_index_minutes",
    "longest_stretch",
    "stretches",
    "study_package_root",
    "trade_day",
    "trading_day_of",
    "trading_day_open_utc",
    "transition_table",
]

E_MINI_POINT_VALUE = 20.0
TICK = 0.25
NOT_ENOUGH = "Not enough history"
CONDITIONS = ("Rising · quiet", "Rising · volatile", "Falling · quiet", "Falling · volatile")
LOOKBACK = 10


def study_package_root(plan: Any) -> Path | None:
    """The plan's bound verified strategy package (run id and manifest hash must match)."""

    if (plan is None or getattr(plan, "source", None) is None
            or not getattr(plan.source, "package_root_name", None)):
        # Registered-input batches have no bound strategy package. Do not borrow
        # historical market bars for optional index/benchmark diagnostics.
        return None
    from alpha_lab.agents.data_infra.ifvg.presentation.funded_trade_review import (
        plan_strategy_package,
    )
    from alpha_lab.propsim.funded.sources import ARCHIVE_ROOT

    return plan_strategy_package(plan, ARCHIVE_ROOT)


def load_index_minutes(package_root: Path, *, cutoff_utc: Any = None) -> pd.DataFrame:
    """One-minute E-mini bars of the package (prices in index points), in time order."""

    bars = pd.read_parquet(
        Path(package_root) / "shared/market_bars.parquet",
        columns=["timeframe_seconds", "trading_day", "logical_open_ts_utc",
                 "logical_close_ts_utc", "open_ticks", "high_ticks", "low_ticks",
                 "close_ticks", "volume"])
    bars = bars[bars["timeframe_seconds"] == 60].copy()
    if cutoff_utc is not None:
        bars = bars[bars["logical_close_ts_utc"] <= pd.Timestamp(cutoff_utc)]
    for column in ("open", "high", "low", "close"):
        bars[column] = bars[f"{column}_ticks"].astype(float) * TICK
    bars["trading_day"] = bars["trading_day"].astype(str)
    return bars.sort_values("logical_close_ts_utc").reset_index(drop=True)


def load_study_index_minutes(plan: Any, *, cutoff_utc: Any = None) -> pd.DataFrame | None:
    """Canonical minutes from the exact bound package or registered-input source.

    Registered views retain the original full-size NQ price proxy, contract
    roll receipts and completion status. They never select only traded minutes.
    """
    if getattr(getattr(plan, "source", None), "kind", None) == "verified_task_b_registered_inputs":
        from .registered_market import load_companion

        return load_companion(plan)
    root = study_package_root(plan)
    return None if root is None else load_index_minutes(root, cutoff_utc=cutoff_utc)


def daily_closes(minutes: pd.DataFrame) -> pd.Series:
    """Last one-minute close of each stored trading day (points), by trading day."""

    return minutes.groupby("trading_day", sort=True)["close"].last()


# ── tie to the index ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class IndexTie:
    beta: float | None
    r_squared: float | None
    days: int


def index_tie(daily: Sequence[tuple[str, float]], closes: pd.Series) -> IndexTie:
    """Slope of daily results on the daily E-mini change × $20 over the same days."""

    days = list(closes.index)
    change: dict[str, float] = {}
    for previous, day in zip(days, days[1:], strict=False):
        change[day] = (float(closes[day]) - float(closes[previous])) * E_MINI_POINT_VALUE
    pairs = [(change[d], v) for d, v in daily if d in change]
    if len(pairs) < 3:
        return IndexTie(None, None, len(pairs))
    x = np.asarray([p[0] for p in pairs])
    y = np.asarray([p[1] for p in pairs])
    if x.var() == 0:
        return IndexTie(None, None, len(pairs))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - fitted) ** 2).sum()) / total if total else None
    return IndexTie(float(slope), r2, len(pairs))


#: the benchmark's definition version (analytical correction A10, September 25, 2026).
#: v1 took its profit from the first trading day's CLOSE but looked for the $2,000 fall
#: from that day's 5:00 PM OPEN, so a morning fall described a position entered only
#: that afternoon. v2 uses one entry instant for both.
BENCHMARK_VERSION = "buy_and_hold_one_emini_first_open_v2"


@dataclass(frozen=True)
class BuyAndHold:
    """One E-mini bought at the first study trading day's open and held to the last close.

    Profit and the first $2,000 fall share one entry instant, one price path (the
    stored one-minute closes), one quantity (one E-mini, $20 a point) and no costs.
    """

    result: float | None
    #: open of the first stored one-minute bar of the first study trading day
    entry_price: float | None
    #: last one-minute close of the last study trading day
    exit_price: float | None
    #: close instant of the first one-minute bar whose close is ``limit`` below the
    #: running high (the entry price counts as the first high)
    breach_utc: pd.Timestamp | None
    limit: float
    entry_utc: pd.Timestamp | None = None
    exit_utc: pd.Timestamp | None = None
    version: str = BENCHMARK_VERSION


def buy_and_hold(minutes: pd.DataFrame, calendar: Sequence[str], *,
                 limit: float = 2000.0) -> BuyAndHold:
    """Hold one E-mini over the study's trading days, from one entry instant.

    Entry: the open price of the first stored one-minute bar of the first study
    trading day (its 5:00 PM Chicago open). Exit: the last one-minute close of
    the last study trading day. Result = (exit − entry) × $20, no costs. The fall
    from a running high is followed on the one-minute closes from that same
    entry (the entry price is the first high), held through nights and weekends;
    a fall inside a minute that recovers by its close is not seen.
    """

    empty = BuyAndHold(None, None, None, None, limit)
    if not calendar:
        return empty
    closes = daily_closes(minutes)
    first, last = calendar[0], calendar[-1]
    if first not in closes.index or last not in closes.index:
        return empty
    window = minutes[minutes["trading_day"].isin(set(calendar))]
    if window.empty or window["trading_day"].iloc[0] != first:
        return empty
    entry_price = float(window["open"].iloc[0])
    path = window["close"].to_numpy(dtype=float)
    high = np.maximum.accumulate(np.maximum(path, entry_price))
    fall = (high - path) * E_MINI_POINT_VALUE
    hit = np.flatnonzero(fall >= limit)
    breach = window["logical_close_ts_utc"].iloc[int(hit[0])] if hit.size else None
    exit_price = float(closes[last])
    return BuyAndHold(
        result=(exit_price - entry_price) * E_MINI_POINT_VALUE,
        entry_price=entry_price, exit_price=exit_price, breach_utc=breach, limit=limit,
        entry_utc=pd.Timestamp(window["logical_open_ts_utc"].iloc[0]),
        exit_utc=pd.Timestamp(window["logical_close_ts_utc"].iloc[-1]))


# ── market conditions ─────────────────────────────────────────────────────
#
# Two separately versioned label sets (analytical correction A1, September 25, 2026):
#
# - RETROSPECTIVE (``condition_labels``): day D from its OWN last close and a volatility
#   window that includes D's own change, against the median of the WHOLE study. A valid
#   description with hindsight; it uses prices from after any entry on D, so it was not
#   known when a trade was entered. Its numbers are unchanged from the first release.
# - ENTRY-KNOWN (``entry_known_labels``): day D only from stored trading days whose close
#   instant is before D's 5:00 PM Chicago open, and a threshold from the measures of days
#   up to D (each measure itself built only from closes completed before that day's open).
#   Too little history is its own state (NOT_ENOUGH), never a zero-valued condition.

#: the retrospective labels' definition (numbers unchanged; named for what they are)
RETROSPECTIVE_VERSION = "retrospective_daily_close_v1"
#: the entry-known labels' definition, versioned separately from frozen economic results
ENTRY_KNOWN_VERSION = "entry_known_prior_closes_v1"
#: label modes (``ConditionLabels.mode``)
RETROSPECTIVE = "retrospective"
ENTRY_KNOWN = "entry_known"
#: an entry-known threshold needs at least this many volatility measures (days up to D)
MIN_HISTORY = 10


@dataclass(frozen=True)
class ConditionLabels:
    #: study trading day → label (one of CONDITIONS or NOT_ENOUGH)
    labels: dict[str, str]
    first_labeled: str | None
    #: retrospective: the whole study's median; entry-known: the last study day's
    #: threshold (None when that day has none)
    volatility_median: float | None
    #: ``RETROSPECTIVE`` (described with hindsight) or ``ENTRY_KNOWN`` (known at entry)
    mode: str = RETROSPECTIVE
    version: str = RETROSPECTIVE_VERSION


def _condition(trend: str, volatile: bool) -> str:
    """``Rising · volatile`` and so on (one of ``CONDITIONS``)."""

    return f"{trend} · {'volatile' if volatile else 'quiet'}"


def condition_labels(closes: pd.Series, calendar: Sequence[str]) -> ConditionLabels:
    """RETROSPECTIVE labels (``retrospective_daily_close_v1``), described with hindsight.

    Rising/Falling: day D's own last close against the close 10 stored trading days
    earlier. Volatility = sample standard deviation of the last 10 daily percentage
    changes, D's own change included; Volatile when above the median of that measure
    over the WHOLE study's labeled trading days. So a label uses prices from after any
    entry on D and from later days: it was not known when a trade was entered (see
    :func:`entry_known_labels` for labels that were).
    """

    days = list(closes.index)
    position = {d: i for i, d in enumerate(days)}
    pct = closes.pct_change()
    trend: dict[str, str] = {}
    vol: dict[str, float] = {}
    for day in calendar:
        i = position.get(day)
        if i is None or i < LOOKBACK:
            continue
        trend[day] = "Rising" if closes.iloc[i] > closes.iloc[i - LOOKBACK] else "Falling"
        vol[day] = float(pct.iloc[i - LOOKBACK + 1: i + 1].std(ddof=1))
    median = float(np.median(list(vol.values()))) if vol else None
    labels = {}
    for day in calendar:
        if day in trend and median is not None:
            labels[day] = _condition(trend[day], vol[day] > median)
        else:
            labels[day] = NOT_ENOUGH
    first = next((d for d in calendar if labels[d] != NOT_ENOUGH), None)
    return ConditionLabels(labels=labels, first_labeled=first, volatility_median=median,
                           mode=RETROSPECTIVE, version=RETROSPECTIVE_VERSION)


def daily_close_times(minutes: pd.DataFrame) -> pd.Series:
    """Close instant of each stored trading day: its last one-minute bar's logical close.

    The same bar :func:`daily_closes` takes the close from (``minutes`` is in close
    order), by trading day.
    """

    return minutes.groupby("trading_day", sort=True)["logical_close_ts_utc"].last()


def trading_day_open_utc(day: str) -> pd.Timestamp:
    """Trading day ``day``'s open: 5:00 PM Chicago on the calendar day before its name.

    A trading day runs from that 5:00 PM open to its 4:00 PM close and is named by its
    closing date (a Monday opens on Sunday at 5:00 PM). Computed on the Chicago wall
    clock, so daylight-saving changes keep 5:00 PM.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import CHICAGO

    previous = (pd.Timestamp(str(day)[:10]) - pd.Timedelta(days=1)).date()
    return pd.Timestamp.combine(previous, dt_time(17)).tz_localize(CHICAGO).tz_convert("UTC")


def trading_day_of(value: Any) -> str | None:
    """The trading day an instant belongs to: from 5:00 PM Chicago on, the next day's.

    An evening entry (5:00 PM or later) belongs to the next calendar day's trading day.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import CHICAGO, utc_instant

    stamp = utc_instant(value)
    if stamp is None:
        return None
    local = stamp.tz_convert(CHICAGO)
    day = local.date() + timedelta(days=1) if local.hour >= 17 else local.date()
    return day.isoformat()


@dataclass(frozen=True)
class EntryKnownDay:
    """How one trading day's entry-known label was reached, from inputs before its open."""

    day: str
    #: the day's 5:00 PM Chicago open (UTC); every input closed before it
    open_utc: pd.Timestamp
    #: completed observations: stored trading days before ``day`` whose close instant is
    #: before ``open_utc``
    inputs: int
    last_input_day: str | None
    last_input_close_utc: pd.Timestamp | None
    #: "Rising"/"Falling", or None with fewer than lookback + 1 completed closes
    trend: str | None
    #: sample standard deviation of the last ``lookback`` daily changes (None: too few)
    measure: float | None
    #: how many stored trading days up to ``day`` (and ``day`` itself) have a measure
    measures: int
    #: median of those measures; None with fewer than ``min_history``
    threshold: float | None
    label: str


def _trend_and_measure(values: Sequence[float], lookback: int) -> tuple[str, float] | None:
    """Trend and volatility measure from completed closes (None: too few closes)."""

    if len(values) < lookback + 1:
        return None
    window = np.asarray(values[-(lookback + 1):], dtype=float)
    changes = window[1:] / window[:-1] - 1.0
    trend = "Rising" if window[-1] > window[0] else "Falling"
    return trend, float(np.std(changes, ddof=1))


def entry_known_days(closes: pd.Series, close_times: pd.Series, calendar: Sequence[str], *,
                     lookback: int = LOOKBACK,
                     min_history: int = MIN_HISTORY) -> dict[str, EntryKnownDay]:
    """Entry-known label of each ``calendar`` day, with the inputs that produced it.

    Completed observations for day D: stored trading days strictly before D whose
    recorded close instant (``close_times``) is before D's 5:00 PM Chicago open. A
    stored day without a close instant is never treated as completed. Nothing from
    D itself or later is read.

    - Trend(D): "Rising" when the last completed close is above the completed close
      ``lookback`` trading days before it, else "Falling".
    - Measure v(D): sample standard deviation (ddof=1) of the last ``lookback`` daily
      percentage changes among completed closes. Both need lookback + 1 closes.
    - Threshold(D): median of v(d) over every stored trading day d before D that has a
      measure, and D's own; each v(d) uses only closes completed before d's open, so
      all are known at D's open. Needs at least ``min_history`` measures.
    - Label: "{trend} · volatile" when v(D) > threshold(D), "{trend} · quiet" otherwise;
      ``NOT_ENOUGH`` when a measure or the threshold is missing.
    """

    if lookback < 2 or min_history < 1:
        raise ValueError("lookback must be at least 2 and min_history at least 1")
    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import utc_instant

    close_utc: dict[str, pd.Timestamp] = {}
    for day, instant in close_times.items():
        stamp = utc_instant(instant, naive="utc")  # the column is named ..._ts_utc
        if stamp is not None:
            close_utc[str(day)] = stamp
    value = {str(d): float(v) for d, v in closes.items() if pd.notna(v)}
    stored = sorted(d for d in value if d in close_utc)
    facts: dict[str, tuple[pd.Timestamp, list[str], tuple[str, float] | None]] = {}

    def known(day: str) -> tuple[pd.Timestamp, list[str], tuple[str, float] | None]:
        if day not in facts:
            opens = trading_day_open_utc(day)
            completed = [d for d in stored if d < day and close_utc[d] < opens]
            facts[day] = (opens, completed,
                          _trend_and_measure([value[d] for d in completed], lookback))
        return facts[day]

    out: dict[str, EntryKnownDay] = {}
    for raw in calendar:
        day = str(raw)
        opens, completed, own = known(day)
        pool = [known(d)[2][1] for d in stored if d < day and known(d)[2] is not None]
        if own is not None:
            pool.append(own[1])
        threshold = float(np.median(pool)) if len(pool) >= min_history else None
        label = (NOT_ENOUGH if own is None or threshold is None
                 else _condition(own[0], own[1] > threshold))
        last = completed[-1] if completed else None
        out[day] = EntryKnownDay(
            day=day, open_utc=opens, inputs=len(completed), last_input_day=last,
            last_input_close_utc=close_utc[last] if last is not None else None,
            trend=own[0] if own else None, measure=own[1] if own else None,
            measures=len(pool), threshold=threshold, label=label)
    return out


def entry_known_labels(closes: pd.Series, close_times: pd.Series, calendar: Sequence[str], *,
                       lookback: int = LOOKBACK, min_history: int = MIN_HISTORY
                       ) -> ConditionLabels:
    """ENTRY-KNOWN labels (``entry_known_prior_closes_v1``): known at each day's open.

    See :func:`entry_known_days` for the definition. A trade is labeled by its entry's
    trading day (the saved ``trading_day``; an evening entry belongs to the next day's).
    ``volatility_median`` is the last study day's threshold (None when it has none).
    """

    days = entry_known_days(closes, close_times, calendar, lookback=lookback,
                            min_history=min_history)
    labels = {day: item.label for day, item in days.items()}
    first = next((d for d, lab in labels.items() if lab != NOT_ENOUGH), None)
    last = next(reversed(days.values()), None)
    return ConditionLabels(labels=labels, first_labeled=first,
                           volatility_median=last.threshold if last is not None else None,
                           mode=ENTRY_KNOWN, version=ENTRY_KNOWN_VERSION)


@dataclass(frozen=True)
class ConditionCard:
    label: str
    days: int
    trades: int
    share_of_trades: float | None
    net: float
    win_rate: float | None
    average: float | None
    first_day: str | None = None
    last_day: str | None = None


def trade_day(trade: dict[str, Any], by: str) -> str:
    """The trading day a trade is labeled by: its entry's, or its exit's.

    A trading day runs from the 5:00 PM open to the 4:00 PM close and is named
    by its closing date; the saved ``trading_day`` is the entry's trading day (an
    evening entry already carries the next calendar day's). Entry-known labels are
    only ever looked up by the entry's day: the exit's day uses later information.
    """

    if by == "entry":
        return str(trade.get("trading_day"))
    # 5:00 PM onward belongs to the next calendar day's trading day
    day = trading_day_of(trade.get("exit_utc"))
    return str(trade.get("trading_day")) if day is None else day


def condition_cards(labels: ConditionLabels, trades: Sequence[dict[str, Any]], *,
                    by: str = "entry") -> list[ConditionCard]:
    counts = Counter(labels.labels.values())
    total = len(trades)
    grouped: dict[str, list[float]] = {name: [] for name in (*CONDITIONS, NOT_ENOUGH)}
    for trade in trades:
        label = labels.labels.get(trade_day(trade, by), NOT_ENOUGH)
        grouped.setdefault(label, []).append(float(trade.get("net_pnl_usd") or 0.0))
    cards = []
    for name in (*CONDITIONS, NOT_ENOUGH):
        values = np.asarray(grouped.get(name, []), dtype=float)
        days = [d for d, lab in labels.labels.items() if lab == name]
        cards.append(ConditionCard(
            label=name, days=counts.get(name, 0), trades=int(values.size),
            share_of_trades=(values.size / total) if total else None,
            net=float(values.sum()),
            win_rate=float((values > 0).mean()) if values.size else None,
            average=float(values.mean()) if values.size else None,
            first_day=days[0] if days else None, last_day=days[-1] if days else None))
    return cards


def transition_table(labels: ConditionLabels) -> dict[str, dict[str, int]]:
    """Counts of consecutive labeled study days: from today's label to tomorrow's."""

    sequence = [lab for lab in labels.labels.values()]
    table = {a: dict.fromkeys(CONDITIONS, 0) for a in CONDITIONS}
    for today, tomorrow in zip(sequence, sequence[1:], strict=False):
        if today in table and tomorrow in table[today]:
            table[today][tomorrow] += 1
    return table


@dataclass(frozen=True)
class Stretch:
    label: str
    first_day: str
    last_day: str
    days: int
    net: float = 0.0
    trading_days: tuple[str, ...] = field(default=(), repr=False)


def stretches(labels: ConditionLabels, daily: Sequence[tuple[str, float]]) -> list[Stretch]:
    """Runs of consecutive study days with the same label, with their net result."""

    result = dict(daily)
    out: list[Stretch] = []
    run: list[str] = []
    current = None
    for day, label in labels.labels.items():
        if label != current and run:
            out.append(Stretch(current, run[0], run[-1], len(run),
                               round(sum(result.get(d, 0.0) for d in run), 2), tuple(run)))
            run = []
        current = label
        run.append(day)
    if run:
        out.append(Stretch(current, run[0], run[-1], len(run),
                           round(sum(result.get(d, 0.0) for d in run), 2), tuple(run)))
    return out


def longest_stretch(items: Sequence[Stretch]) -> Stretch | None:
    labeled = [s for s in items if s.label != NOT_ENOUGH]
    return max(labeled, key=lambda s: (s.days, s.first_day)) if labeled else None


# ── entry conditions ──────────────────────────────────────────────────────

ENTRY_MEASURES = {
    "volatility": ("Volatility", "Average one-minute range over the last 20 minutes",
                   "Average true range of the 20 one-minute bars that closed by the entry, "
                   "divided by the entry price."),
    "trend": ("Trend", "Price change over the last hour",
              "Change of the one-minute close over the 60 bars before the entry, divided by "
              "the close 60 bars earlier."),
    "volume": ("Volume", "Contracts traded in the last 20 minutes",
               "E-mini contracts traded in the 20 one-minute bars that closed by the entry."),
    "time": ("Time of day", "Hours since the 5:00 PM open",
             "Hours from the trading day's 5:00 PM Chicago open to the entry."),
}


@dataclass(frozen=True)
class EntryMeasure:
    measure: str
    x: tuple[float, ...]
    r_multiple: tuple[float, ...]
    winner: tuple[bool, ...]
    correlation: float | None
    p_value: float | None
    clear_link: bool


def _entry_x(minutes: pd.DataFrame, closes_ns: np.ndarray, trade: dict[str, Any],
             measure: str) -> float | None:
    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import CHICAGO, utc_instant

    entry = utc_instant(trade.get("entry_utc"))
    if entry is None:
        return None
    end = int(np.searchsorted(closes_ns, entry.value, side="right"))
    if measure == "time":
        local = entry.tz_convert(CHICAGO)
        # the 5:00 PM open on the wall clock (not 17 elapsed hours: DST Sundays)
        start = pd.Timestamp.combine(local.date(), dt_time(17)).tz_localize(CHICAGO)
        if local < start:
            start = pd.Timestamp.combine(local.date() - timedelta(days=1),
                                         dt_time(17)).tz_localize(CHICAGO)
        return (local - start).total_seconds() / 3600.0
    if measure == "trend":
        if end < 61:
            return None
        now, before = minutes["close"].iat[end - 1], minutes["close"].iat[end - 61]
        return float((now - before) / before)
    if end < 21:
        return None
    window = minutes.iloc[end - 20: end]
    if measure == "volume":
        return float(window["volume"].sum())
    previous = minutes["close"].iloc[end - 21: end - 1].to_numpy()
    high, low = window["high"].to_numpy(), window["low"].to_numpy()
    true_range = np.maximum(high, previous) - np.minimum(low, previous)
    price = float(trade.get("entry_ticks") or 0) * TICK or float(window["close"].iat[-1])
    return float(true_range.mean() / price)


def entry_measure(minutes: pd.DataFrame, trades: Sequence[dict[str, Any]], *,
                  measure: str = "volatility") -> EntryMeasure:
    """One entry-condition measure per trade against its result in multiples of initial risk."""

    closes_ns = minutes["logical_close_ts_utc"].astype("int64").to_numpy()
    xs, rs, wins = [], [], []
    for trade in trades:
        risk = float(trade.get("initial_risk_usd") or 0.0)
        value = _entry_x(minutes, closes_ns, trade, measure)
        if value is None or risk <= 0:
            continue
        result = float(trade.get("net_pnl_usd") or 0.0)
        xs.append(value)
        rs.append(result / risk)
        wins.append(result > 0)
    correlation = p_value = None
    if len(xs) >= 3 and np.std(xs) > 0 and np.std(rs) > 0:
        from scipy.stats import pearsonr

        test = pearsonr(xs, rs)
        correlation, p_value = float(test.statistic), float(test.pvalue)
    clear = (correlation is not None and abs(correlation) >= 0.1
             and p_value is not None and p_value < 0.05)
    return EntryMeasure(measure=measure, x=tuple(xs), r_multiple=tuple(rs), winner=tuple(wins),
                        correlation=correlation, p_value=p_value, clear_link=clear)


#: the earlier private name, kept for callers written before it was public
_trade_day = trade_day
