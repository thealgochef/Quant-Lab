"""ROOT-STRUCTURE CENSUS — calibration + real-data smoke of the Phase-F time-bar builders.

READ-ONLY against the Databento store; writes ONLY scratch_root_census_results.json at
the QL root. Untracked scratch. No thresholds, no verdicts — distributions only.

WINDOW: the 20 consecutive trading days ending 2026-02-13 that have data in the store.
First 5 are WARMUP (feed registries, excluded from headline aggregates); remaining 15
are COUNTED. The window must not intersect the sealed range 2026-06-12..2026-07-10.

DEFINITIONS (stated verbatim in FVG_CENSUS.md):
* Bars: Phase-F BATCH time bars (build_time_bars_from_frame) at 60/180/300/600/900/
  1800/3600/14400 s, built from the canonical reader's trade stream, ONE drain/day.
* FVG, per timeframe independently, at that timeframe's own bar closes, over the bar
  series CONCATENATED across days (triplets may span the trading-day boundary; both
  COMPLETE and END_OF_DAY bars participate):
    Bullish at bar i (i>=2): bars[i-2].high < bars[i].low; interval [bars[i-2].high, bars[i].low]
    Bearish at bar i:        bars[i-2].low  > bars[i].high; interval [bars[i].high, bars[i-2].low]
    Formation ts = bars[i].close_ts_utc.  Size = interval width in ticks. NO minimum-size filter.
* Fill tracking against the 60s series, wicks counting (closed-interval overlap),
  only 60s bars with close_ts STRICTLY AFTER formation ts participate:
    FIRST TOUCH = first 60s bar whose [low, high] overlaps the gap interval at all.
    FULL FILL   = first 60s bar by which price has cumulatively traversed the entire
                  interval (bullish: running min(low) <= interval lo; bearish:
                  running max(high) >= interval hi).
  Elapsed 60s-bar count = number of emitted 60s bars with close_ts in (formation, event].
* Key-level taps: the day's trades fold through StrategyLevelState (seeded for day 1
  via the QL dataset builder's _get_session_hl_for_date walk); zones from build_zones
  at default proximity; detection against the day's 60s bars.
    Geometry 1 = the existing representative-price straddle (detect_touches as-is,
                 available_from gate enforced).
    Geometry 2 = bar range intersects [rep - 4 ticks, rep + 4 ticks] (same gate, same
                 first-touch scope, auxiliary scan).
  A merged zone's tap counts toward EVERY constituent level source.
* E6 overlap: a key-level (geometry-1 first) tap coincides with an FVG root at
  tolerance +/-N ticks iff the SAME 60s bar also FIRST-touched a live unfilled 1H or
  4H gap AND the zone representative price lies within N ticks of that gap's interval.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict, deque
from datetime import date, datetime, timedelta
from pathlib import Path

_QL_ROOT = Path(__file__).resolve().parent
if str(_QL_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_QL_ROOT / "src"))

import numpy as np
import pandas as pd

import strategy_core as sc
from strategy_core.candles._buckets import trading_day_start_utc
from strategy_core.candles.time_batch import build_time_bars_from_frame
from strategy_core.constants import DEFAULT_TICK_SIZE, RESEARCH_SESSION_SCHEME
from strategy_core.data.databento_parquet import DatabentoParquetSource
from strategy_core.decisions.touch import DEFAULT_DIRECTION_FROM_SIDE, detect_touches, is_touch
from strategy_core.decisions.zones import build_zones
from strategy_core.runtime.levels import StrategyLevelState
from strategy_core.types import Trade

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import _get_session_hl_for_date
from alpha_lab.agents.data_infra.ml.engine_decision import TRADE_TICK, _round_to_ticks

DATA_DIR = Path(r"C:\Users\gonza\Documents\Claude-Quant-Lab\data\databento")
SYMBOL = "NQ"
TICK = DEFAULT_TICK_SIZE
TIMEFRAMES = (60, 180, 300, 600, 900, 1800, 3600, 14400)
TF_LABEL = {60: "1m", 180: "3m", 300: "5m", 600: "10m", 900: "15m", 1800: "30m", 3600: "1H", 14400: "4H"}
ICT_PRIOR_TICKS = {60: 4, 180: 6, 300: 8, 600: 10, 900: 12, 1800: 16, 3600: 24, 14400: 40}
REGISTRY_TFS = (3600, 14400)
END_DAY = date(2026, 2, 13)
N_DAYS = 20
N_WARMUP = 5
SEALED = (date(2026, 6, 12), date(2026, 7, 10))
BAND_TICKS = 4  # geometry 2 half-width
OUT = _QL_ROOT / "scratch_root_census_results.json"

UTIL_CFG = DashboardUtilityConfig(
    tp_points=15.0, sl_points=15.0, trap_mfe_min=5.0, interaction_window_minutes=5,
    level_proximity_pts=0.5, bar_type="147t", include_approach_features=True,
    approach_window_minutes=15,
)


def _enumerate_days() -> list[date]:
    """Walk back from END_DAY over weekdays whose store directory exists, newest
    first, until 20 are found; return chronological."""
    days: list[date] = []
    d = END_DAY
    while len(days) < N_DAYS:
        if d.weekday() < 5 and (DATA_DIR / SYMBOL / d.isoformat()).is_dir():
            days.append(d)
        d -= timedelta(days=1)
        if (END_DAY - d).days > 60:
            raise RuntimeError("could not find 20 store days within 60 calendar days")
    return sorted(days)


def _seed_before(first_day: date) -> tuple[tuple[int, int], str]:
    """The QL dataset builder's seed walk: most recent non-empty prior store day's
    full-day H/L via _get_session_hl_for_date, converted to ticks."""
    d = first_day - timedelta(days=1)
    for _ in range(15):
        if d.weekday() < 5 and (DATA_DIR / SYMBOL / d.isoformat()).is_dir():
            hl = _get_session_hl_for_date(DATA_DIR, SYMBOL, d.isoformat(), UTIL_CFG, None)
            if hl is not None:
                return (
                    (_round_to_ticks(hl[0], TRADE_TICK), _round_to_ticks(hl[1], TRADE_TICK)),
                    d.isoformat(),
                )
        d -= timedelta(days=1)
    raise RuntimeError("no non-empty seed day found before the window")


class Gap:
    __slots__ = (
        "tf", "direction", "lo", "hi", "formed_ts", "formed_day", "counted",
        "prev2_close_ts", "span_wall_s", "first_touch_ts", "first_touch_day",
        "filled_ts", "filled_day", "reached",
    )

    def __init__(self, tf, direction, lo, hi, formed_ts, formed_day, counted, prev2_close_ts):
        self.tf = tf
        self.direction = direction  # 'bullish' | 'bearish'
        self.lo = lo  # interval bounds in ticks, lo < hi
        self.hi = hi
        self.formed_ts = formed_ts
        self.formed_day = formed_day
        self.counted = counted
        self.prev2_close_ts = prev2_close_ts
        self.span_wall_s = (formed_ts - prev2_close_ts).total_seconds()
        self.first_touch_ts = None
        self.first_touch_day = None
        self.filled_ts = None
        self.filled_day = None
        # running extreme INTO the gap since formation (bullish: min low; bearish: max high)
        self.reached = None


def main() -> int:
    t_census0 = time.perf_counter()
    days = _enumerate_days()
    assert days[-1] == END_DAY, f"window must end {END_DAY}: {days[-1]}"
    for d in days:
        assert not (SEALED[0] <= d <= SEALED[1]), f"day {d} in sealed range"
    warmup = set(days[:N_WARMUP])
    print(f"window: {days[0]}..{days[-1]} ({len(days)} days, warmup {sorted(warmup)})")

    t = time.perf_counter()
    seed, seed_day = _seed_before(days[0])
    seed_s = time.perf_counter() - t
    print(f"seed day {seed_day}: H/L ticks {seed} ({seed_s:.1f}s)")

    state = StrategyLevelState(scheme=RESEARCH_SESSION_SCHEME, tick_size=TICK)
    state.load_prior_day_summary(
        date.fromisoformat(seed_day), high_ticks=seed[0], low_ticks=seed[1]
    )

    results: dict = {
        "meta": {
            "strategy_core_file": sc.__file__,
            "window": [d.isoformat() for d in days],
            "warmup": [d.isoformat() for d in sorted(warmup)],
            "counted": [d.isoformat() for d in days if d not in warmup],
            "seed_day": seed_day, "seed_ticks": list(seed), "seed_walk_s": seed_s,
            "tick_size": TICK, "band_ticks": BAND_TICKS,
            "timeframes": list(TIMEFRAMES),
        },
        "days": [], "e4": {}, "e6": {}, "gaps": [],
    }

    rolling: dict[int, deque] = {tf: deque(maxlen=2) for tf in TIMEFRAMES}
    live_gaps: list[Gap] = []       # formed, not yet fully filled
    all_gaps: list[Gap] = []
    all_60s_bars: list = []          # (close_ts, day) for global ordinals — kept light
    day_universe: dict[date, tuple[datetime, int]] = {}  # day -> (start_utc, n_60s_buckets)
    occupied60: dict[date, set[int]] = {}                # day -> occupied 60s wall buckets

    for day in days:
        t_day0 = time.perf_counter()
        counted = day not in warmup

        # ── E9 (registry at day start) ────────────────────────────────────────
        reg_start = {
            TF_LABEL[tf]: sum(
                1 for g in live_gaps if g.tf == tf and g.filled_ts is None
            )
            for tf in REGISTRY_TFS
        }

        # ── ONE canonical-reader drain: trades captured + level-state fold ────
        source = DatabentoParquetSource.for_trading_day(
            DATA_DIR / SYMBOL, day, requested_symbol=SYMBOL
        )
        ts_list: list[int] = []
        px_list: list[int] = []
        sz_list: list[int] = []
        levels_at_start = None
        for event in source.events():
            if not isinstance(event, Trade):
                continue
            ts = event.event_ts_utc
            value = getattr(ts, "value", None)
            if value is None:
                delta = ts - datetime(1970, 1, 1, tzinfo=ts.tzinfo)
                value = ((delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1000
            ts_list.append(int(value))
            px_list.append(event.price_ticks)
            sz_list.append(event.size)
            levels_now = state.process_trade(event)
            if levels_at_start is None:
                levels_at_start = [lv.name for lv in levels_now]
        n_trades = len(ts_list)
        drain_s = time.perf_counter() - t_day0
        if n_trades == 0:
            raise RuntimeError(f"{day}: no trades from the canonical reader")

        # ── E2: batch time bars, all 8 TFs, from the retained stream ──────────
        t = time.perf_counter()
        frame = pd.DataFrame(
            {
                "ts_event": pd.DatetimeIndex(np.asarray(ts_list, dtype=np.int64), tz="UTC"),
                "price": np.asarray(px_list, dtype=np.int64) * TICK,
                "size": np.asarray(sz_list, dtype=np.int64),
            }
        )
        del ts_list, px_list, sz_list
        bars = build_time_bars_from_frame(frame, TIMEFRAMES)
        del frame
        build_s = time.perf_counter() - t
        by_tf = {tf: [b for b in bars if b.timeframe_ticks == tf] for tf in TIMEFRAMES}
        stray = {b.trading_day for b in bars if b.trading_day != day}
        if stray:
            raise RuntimeError(f"{day}: reader emitted foreign trading days {stray}")

        day_start = trading_day_start_utc(day, RESEARCH_SESSION_SCHEME)
        day_end = trading_day_start_utc(day + timedelta(days=1), RESEARCH_SESSION_SCHEME)
        day_len = int((day_end - day_start).total_seconds())
        day_universe[day] = (day_start, day_len // 60)
        occupied60[day] = {
            int((b.open_ts_utc - day_start).total_seconds()) // 60 for b in by_tf[60]
        }
        bar_stats = {}
        for tf in TIMEFRAMES:
            universe = day_len // tf
            occupied = len(by_tf[tf])
            bar_stats[TF_LABEL[tf]] = {"bars": occupied, "empty_buckets": universe - occupied}

        # ── E3: formations this day (rolling triplets, concatenated series) ───
        new_gaps: list[Gap] = []
        for tf in TIMEFRAMES:
            roll = rolling[tf]
            for b in by_tf[tf]:
                if len(roll) == 2:
                    p2, _p1 = roll[0], roll[1]
                    if p2.high_ticks < b.low_ticks:
                        new_gaps.append(Gap(
                            tf, "bullish", p2.high_ticks, b.low_ticks,
                            b.close_ts_utc, day, counted, p2.close_ts_utc,
                        ))
                    elif p2.low_ticks > b.high_ticks:
                        new_gaps.append(Gap(
                            tf, "bearish", b.high_ticks, p2.low_ticks,
                            b.close_ts_utc, day, counted, p2.close_ts_utc,
                        ))
                roll.append(b)
        new_gaps.sort(key=lambda g: g.formed_ts)
        all_gaps.extend(new_gaps)

        # ── E5: interleaved fill scan over the day's 60s bars ─────────────────
        # gap_ft_by_bar: close_ts -> [gap, ...] first-touched on that bar (for E6)
        gap_ft_by_bar: dict[datetime, list[Gap]] = defaultdict(list)
        pending = deque(new_gaps)
        for b in by_tf[60]:
            while pending and pending[0].formed_ts < b.close_ts_utc:
                live_gaps.append(pending.popleft())
            lo_t, hi_t = b.low_ticks, b.high_ticks
            for g in live_gaps:
                if g.filled_ts is not None:
                    continue
                if lo_t <= g.hi and hi_t >= g.lo:  # wicks-counting overlap
                    if g.first_touch_ts is None:
                        g.first_touch_ts = b.close_ts_utc
                        g.first_touch_day = day
                        gap_ft_by_bar[b.close_ts_utc].append(g)
                    if g.direction == "bullish":
                        g.reached = lo_t if g.reached is None else min(g.reached, lo_t)
                        if g.reached <= g.lo:
                            g.filled_ts = b.close_ts_utc
                            g.filled_day = day
                    else:
                        g.reached = hi_t if g.reached is None else max(g.reached, hi_t)
                        if g.reached >= g.hi:
                            g.filled_ts = b.close_ts_utc
                            g.filled_day = day
            all_60s_bars.append((b.close_ts_utc, day))
        live_gaps.extend(pending)  # gaps formed at the day's final bar close
        live_gaps = [g for g in live_gaps if g.filled_ts is None]

        # ── E4: key-level taps against the day's 60s bars ─────────────────────
        levels = list(state.levels())
        zones_g1 = build_zones(list(levels))
        bars60 = by_tf[60]
        touches = detect_touches(bars60, zones_g1, tick_size=TICK, trading_day=day)
        zone_by_rep = {z.representative_price: z for z in zones_g1}

        # auxiliary scans: geometry-2 first touches, re-tap totals (both
        # geometries), and available_from suppression counts — same gate, same
        # zone set, custom straddle tests as specified.
        zones_g2 = build_zones(list(levels))
        g2_first: dict[float, datetime] = {}
        g1_total: dict[float, int] = defaultdict(int)
        g2_total: dict[float, int] = defaultdict(int)
        suppressed: dict[float, int] = defaultdict(int)
        band = BAND_TICKS * TICK
        for b in bars60:
            low_p = b.low_ticks * TICK
            high_p = b.high_ticks * TICK
            for z in zones_g2:
                rep = z.representative_price
                hit_g1 = is_touch(low_p, high_p, rep)
                hit_g2 = (low_p <= rep + band) and (high_p >= rep - band)
                gated = z.available_from is not None and b.close_ts_utc < z.available_from
                if gated:
                    if hit_g1:
                        suppressed[rep] += 1
                    continue
                if hit_g1:
                    g1_total[rep] += 1
                if hit_g2:
                    g2_total[rep] += 1
                    if rep not in g2_first:
                        g2_first[rep] = b.close_ts_utc

        per_source: dict[str, dict] = defaultdict(
            lambda: {"g1_first": 0, "g1_total": 0, "g2_first": 0, "g2_total": 0,
                     "suppressed": 0, "directions": defaultdict(int)}
        )
        for z in zones_g1:
            rep = z.representative_price
            direction = DEFAULT_DIRECTION_FROM_SIDE[z.side].value
            for name in z.names:
                row = per_source[name]
                row["g1_first"] += 1 if z.touched else 0
                row["g1_total"] += g1_total[rep]
                row["g2_first"] += 1 if rep in g2_first else 0
                row["g2_total"] += g2_total[rep]
                row["suppressed"] += suppressed[rep]
                if z.touched:
                    row["directions"][direction] += 1
        results["e4"][day.isoformat()] = {
            "counted": counted,
            "sources": {
                k: {**v, "directions": dict(v["directions"])} for k, v in per_source.items()
            },
            "zones": [
                {"rep": z.representative_price, "names": list(z.names),
                 "side": z.side.value, "touched": z.touched}
                for z in zones_g1
            ],
        }

        # ── E6: same-bar overlap of level taps with 1H/4H FVG first touches ───
        e6 = {"taps": len(touches), "coincide": {"0": 0, "4": 0, "8": 0}, "detail": []}
        for touch in touches:
            gaps_here = [
                g for g in gap_ft_by_bar.get(touch.bar_ts_utc, ()) if g.tf in REGISTRY_TFS
            ]
            zone = zone_by_rep[touch.representative_price]
            for tol in (0, 4, 8):
                tol_p = tol * TICK
                hit = any(
                    (g.lo * TICK - tol_p) <= touch.representative_price <= (g.hi * TICK + tol_p)
                    for g in gaps_here
                )
                if hit:
                    e6["coincide"][str(tol)] += 1
            if gaps_here:
                e6["detail"].append({
                    "ts": touch.bar_ts_utc.isoformat(), "source": zone.names[0],
                    "gap_tfs": [TF_LABEL[g.tf] for g in gaps_here],
                })
        results["e6"][day.isoformat()] = e6

        day_s = time.perf_counter() - t_day0
        results["days"].append({
            "day": day.isoformat(), "warmup": not counted, "n_trades": n_trades,
            "drain_s": round(drain_s, 2), "bar_build_s": round(build_s, 2),
            "wall_s": round(day_s, 2), "bars": bar_stats,
            "levels_at_start": levels_at_start,
            "registry_unfilled_at_start": reg_start,
            "gaps_formed": len(new_gaps),
        })
        print(f"{day} trades={n_trades} drain={drain_s:.1f}s wall={day_s:.1f}s "
              f"60s={bar_stats['1m']['bars']} 4H={bar_stats['4H']['bars']} "
              f"gaps+={len(new_gaps)} taps={len(touches)}")
        _dump(results)  # progressive

    # ── post-pass: global 60s ordinals + per-gap elapsed metrics ──────────────
    close_arr = np.array([pd.Timestamp(ts).value for ts, _ in all_60s_bars], dtype=np.int64)
    day_offset: dict[date, int] = {}
    acc = 0
    for d in days:
        day_offset[d] = acc
        acc += day_universe[d][1]
    occ_global = np.array(sorted(
        day_offset[d] + b for d in days for b in occupied60[d]
    ), dtype=np.int64)

    def _abs_bucket(ts: datetime) -> int:
        for d in reversed(days):
            start, universe = day_universe[d]
            if ts >= start:
                rel = int((ts - start).total_seconds()) // 60
                return day_offset[d] + min(rel, universe - 1)
        raise RuntimeError(f"ts before window: {ts}")

    def _ns(ts: datetime) -> int:
        return int(pd.Timestamp(ts).value)

    gap_rows = []
    for g in all_gaps:
        formed_ord = int(np.searchsorted(close_arr, _ns(g.formed_ts), side="right"))
        b_lo = _abs_bucket(g.prev2_close_ts)
        b_hi = _abs_bucket(g.formed_ts)
        total_between = max(0, b_hi - b_lo - 1)
        occ_between = int(
            np.searchsorted(occ_global, b_hi) - np.searchsorted(occ_global, b_lo + 1)
        )
        row = {
            "tf": TF_LABEL[g.tf], "direction": g.direction,
            "lo_ticks": g.lo, "hi_ticks": g.hi, "size_ticks": g.hi - g.lo,
            "formed_ts": g.formed_ts.isoformat(), "formed_day": g.formed_day.isoformat(),
            "counted": g.counted, "span_wall_s": g.span_wall_s,
            "span_empty_60s": max(0, total_between - occ_between),
            "first_touch_ts": None, "ft_wall_min": None, "ft_bars_60": None,
            "filled_ts": None, "fill_wall_min": None, "fill_bars_60": None,
        }
        if g.first_touch_ts is not None:
            ft_ord = int(np.searchsorted(close_arr, _ns(g.first_touch_ts), side="right"))
            row["first_touch_ts"] = g.first_touch_ts.isoformat()
            row["ft_wall_min"] = round((g.first_touch_ts - g.formed_ts).total_seconds() / 60, 2)
            row["ft_bars_60"] = ft_ord - formed_ord
            row["first_touch_day"] = g.first_touch_day.isoformat()
        if g.filled_ts is not None:
            fl_ord = int(np.searchsorted(close_arr, _ns(g.filled_ts), side="right"))
            row["filled_ts"] = g.filled_ts.isoformat()
            row["fill_wall_min"] = round((g.filled_ts - g.formed_ts).total_seconds() / 60, 2)
            row["fill_bars_60"] = fl_ord - formed_ord
            row["filled_day"] = g.filled_day.isoformat()
        gap_rows.append(row)
    results["gaps"] = gap_rows

    # ── E7 aggregates over COUNTED-day gaps ───────────────────────────────────
    counted_days = [d for d in days if d not in warmup]

    def _pct(values, q):
        return None if not values else float(np.percentile(np.asarray(values, dtype=float), q))

    e7 = {}
    for tf in TIMEFRAMES:
        rows = [r for r in gap_rows if r["tf"] == TF_LABEL[tf] and r["counted"]]
        per_day_bull = defaultdict(int)
        per_day_bear = defaultdict(int)
        for r in rows:
            (per_day_bull if r["direction"] == "bullish" else per_day_bear)[r["formed_day"]] += 1
        bull_series = [per_day_bull[d.isoformat()] for d in counted_days]
        bear_series = [per_day_bear[d.isoformat()] for d in counted_days]
        sizes = [r["size_ticks"] for r in rows]
        touched = [r for r in rows if r["first_touch_ts"] is not None]
        filled = [r for r in rows if r["filled_ts"] is not None]
        e7[TF_LABEL[tf]] = {
            "n_gaps": len(rows),
            "per_day_bullish": {"mean": float(np.mean(bull_series)), "median": float(np.median(bull_series))},
            "per_day_bearish": {"mean": float(np.mean(bear_series)), "median": float(np.median(bear_series))},
            "size_ticks": {
                "min": min(sizes) if sizes else None, "p10": _pct(sizes, 10),
                "p25": _pct(sizes, 25), "median": _pct(sizes, 50), "p75": _pct(sizes, 75),
                "p90": _pct(sizes, 90), "max": max(sizes) if sizes else None,
            },
            "ict_prior_ticks": ICT_PRIOR_TICKS[tf],
            "share_touched": None if not rows else len(touched) / len(rows),
            "share_filled": None if not rows else len(filled) / len(rows),
            "share_untouched": None if not rows else 1 - len(touched) / len(rows),
            "ft_wall_min": {"median": _pct([r["ft_wall_min"] for r in touched], 50),
                            "p90": _pct([r["ft_wall_min"] for r in touched], 90)},
            "ft_bars_60": {"median": _pct([r["ft_bars_60"] for r in touched], 50),
                           "p90": _pct([r["ft_bars_60"] for r in touched], 90)},
            "fill_wall_min": {"median": _pct([r["fill_wall_min"] for r in filled], 50),
                              "p90": _pct([r["fill_wall_min"] for r in filled], 90)},
            "fill_bars_60": {"median": _pct([r["fill_bars_60"] for r in filled], 50),
                             "p90": _pct([r["fill_bars_60"] for r in filled], 90)},
        }
    results["e7"] = e7

    # ── E8: three per-day series over counted days ────────────────────────────
    e8a = {d.isoformat(): defaultdict(int) for d in counted_days}
    for r in gap_rows:
        if r["tf"] in ("1H", "4H") and r.get("first_touch_day") in e8a:
            e8a[r["first_touch_day"]][f"{r['tf']}_{r['direction']}"] += 1
    e8b = {}
    for d in counted_days:
        srcs = results["e4"][d.isoformat()]["sources"]
        e8b[d.isoformat()] = {
            src: {"first": v["g1_first"], "directions": v["directions"]}
            for src, v in srcs.items()
        }
    e8c = {
        d.isoformat(): results["e6"][d.isoformat()]["coincide"]["4"] for d in counted_days
    }

    def _series_stats(values):
        return {"mean": float(np.mean(values)), "median": float(np.median(values))}

    e8a_totals = [sum(v.values()) for v in ({k: dict(v) for k, v in e8a.items()}).values()]
    e8b_totals = [
        sum(v["first"] for v in e8b[d.isoformat()].values()) for d in counted_days
    ]
    results["e8"] = {
        "a_fvg_first_touches": {k: dict(v) for k, v in e8a.items()},
        "a_stats": _series_stats(e8a_totals),
        "b_level_first_touches": e8b,
        "b_stats": _series_stats(e8b_totals),
        "c_overlap_at_4t": e8c,
        "c_stats": _series_stats(list(e8c.values())),
    }

    # ── E5 end-of-run registry state ─────────────────────────────────────────
    results["e5_end_state"] = {
        TF_LABEL[tf]: {
            "untouched": sum(
                1 for r in gap_rows if r["tf"] == TF_LABEL[tf] and r["first_touch_ts"] is None
            ),
            "touched_unfilled": sum(
                1 for r in gap_rows
                if r["tf"] == TF_LABEL[tf] and r["first_touch_ts"] is not None
                and r["filled_ts"] is None
            ),
        }
        for tf in TIMEFRAMES
    }

    results["meta"]["census_wall_s"] = round(time.perf_counter() - t_census0, 1)
    results["meta"]["per_day_mean_s"] = round(
        float(np.mean([d["wall_s"] for d in results["days"]])), 1
    )
    _dump(results)
    print(f"census wall {results['meta']['census_wall_s']}s "
          f"(per-day mean {results['meta']['per_day_mean_s']}s) -> {OUT}")
    return 0


def _dump(results: dict) -> None:
    tmp = str(OUT) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)
    Path(tmp).replace(OUT)


if __name__ == "__main__":
    raise SystemExit(main())
