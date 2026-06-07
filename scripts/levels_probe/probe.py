# ruff: noqa: B023,E741
"""INVESTIGATE (READ-ONLY): why are RTH touches scarce and why does overnight revert more?

Reuses the PRODUCTION engine functions verbatim (_build_bars_for_date / _compute_levels_for_date /
build_zones / is_touch / detect_touches) so diagnostics match the cached dataset. Scans EVERY
bar-zone straddle (not just the deduped first-touch) per Globex day, tags sessions, flags
available-from look-ahead, and computes per-touch travel/vol/volume context for E1-E4.

Per-date outputs: out/zones_<d>.parquet (one row per zone) + out/touches_<d>.parquet (one row per
recorded first-touch with context). NO fixes, NO production changes. PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import date, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from strategy_core import build_zones
from strategy_core.decisions.touch import is_touch

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
    _build_bars_for_date,
    _compute_levels_for_date,
    _ensure_et_index,
    _get_session_hl_for_date,
)
from alpha_lab.agents.data_infra.ml.engine_decision import (
    TRADE_TICK,
    bars_et_to_engine,
    levels_to_engine,
)

DATA_DIR = Path("C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento")
SYM = "NQ"
ET = ZoneInfo("US/Eastern")
OUT = Path(__file__).parent / "out"
util = DashboardUtilityConfig(
    tp_points=15.0,
    sl_points=15.0,
    trap_mfe_min=5.0,
    interaction_window_minutes=5,
    level_proximity_pts=0.5,
    bar_type="147t",
    include_approach_features=True,
    approach_window_minutes=15,
)


def sess(ts_et: pd.Timestamp) -> str:
    h = ts_et.hour + ts_et.minute / 60.0
    if 9.5 <= h < 16.25:
        return "ny_rth"
    if 8 <= h < 9.5:
        return "premarket"
    if 1 <= h < 8:
        return "london"
    if h >= 18 or h < 1:
        return "asia"
    return "post"  # 16:15-18:00


def avail_from(names, td):
    """ET timestamp from which the zone is knowable (latest constituent's session close)."""
    cands = []
    for n in names:
        nl = n.lower()
        if nl in ("pdh", "pdl"):
            cands.append(
                pd.Timestamp.combine(td - timedelta(days=1), time(18, 0)).tz_localize(ET)
            )  # day start
        elif nl.startswith("asia"):
            cands.append(
                pd.Timestamp.combine(td, time(1, 0)).tz_localize(ET)
            )  # asia closes 01:00 ET
        elif nl.startswith("london"):
            cands.append(
                pd.Timestamp.combine(td, time(8, 0)).tz_localize(ET)
            )  # london closes 08:00 ET
        else:
            cands.append(pd.Timestamp.combine(td - timedelta(days=1), time(18, 0)).tz_localize(ET))
    return max(cands)


def probe_date(date_str, prev_state):
    bars = _build_bars_for_date(DATA_DIR, SYM, date_str, util)
    if bars.empty:
        return None, None, _get_session_hl_for_date(DATA_DIR, SYM, date_str, util, *prev_state)
    bars_et = _ensure_et_index(bars)
    levels = _compute_levels_for_date(bars_et, date_str, *prev_state)
    new_state = _get_session_hl_for_date(DATA_DIR, SYM, date_str, util, *prev_state)
    if not levels:
        return None, None, new_state
    td = date.fromisoformat(date_str)
    tick = TRADE_TICK
    eng_bars = bars_et_to_engine(bars_et, td, tick)
    zones = build_zones(levels_to_engine(levels))  # PRODUCTION zone build

    # closes (ET) aligned to eng_bars order
    closes_et = [pd.Timestamp(b.close_ts_utc).tz_convert(ET) for b in eng_bars]
    lows = np.array([b.low_ticks * tick for b in eng_bars])
    highs = np.array([b.high_ticks * tick for b in eng_bars])
    # raw session extremes (to flag self-touch / degenerate levels)
    raw_extremes = {round(l["price"], 4): l["name"] for l in levels}

    # context arrays from bars_et (OHLCV, ET index)
    o = bars_et["open"].values
    c = bars_et["close"].values
    hi = bars_et["high"].values
    lo = bars_et["low"].values
    vol = bars_et["volume"].values if "volume" in bars_et else np.zeros(len(bars_et))
    sess_open_px = float(o[0])
    rth_mask = np.array([sess(t) == "ny_rth" for t in closes_et])
    rth_open_px = float(o[np.argmax(rth_mask)]) if rth_mask.any() else np.nan

    zrows, trows = [], []
    for zi, z in enumerate(zones):
        rep = z.representative_price
        straddle_idx = [i for i in range(len(eng_bars)) if is_touch(lows[i], highs[i], rep)]
        af = avail_from(z.names, td)
        [sess(closes_et[i]) for i in straddle_idx]
        rth_straddles = [i for i in straddle_idx if sess(closes_et[i]) == "ny_rth"]
        first = straddle_idx[0] if straddle_idx else None
        first_et = closes_et[first] if first is not None else None
        first_sess = sess(first_et) if first is not None else None
        is_raw = (len(z.names) == 1) and (round(rep, 4) in raw_extremes)
        zrows.append(
            {
                "date": date_str,
                "zone_idx": zi,
                "rep_price": rep,
                "side": z.side.value,
                "names": "|".join(z.names),
                "n_levels": len(z.names),
                "available_from_et": str(af.time()),
                "n_straddles": len(straddle_idx),
                "first_straddle_et": str(first_et) if first_et is not None else None,
                "first_straddle_session": first_sess,
                "has_rth_straddle": len(rth_straddles) > 0,
                "rth_straddle_first_et": str(closes_et[rth_straddles[0]])
                if rth_straddles
                else None,
                "lookahead_first_before_avail": bool(first_et is not None and first_et < af),
                "rep_is_raw_session_extreme_standalone": bool(is_raw),
                "n_rth_straddles": len(rth_straddles),
                "first_straddle_overnight_but_rth_swept": bool(
                    first_sess not in (None, "ny_rth") and len(rth_straddles) > 0
                ),
            }
        )
        # recorded touch context (first straddle = the detect_touches record)
        if first is not None:
            loc = first

            def win(arr, w):
                a = arr[max(0, loc - w) : loc]
                return a

            atr14 = (
                float(np.mean(hi[max(0, loc - 14) : loc] - lo[max(0, loc - 14) : loc]))
                if loc >= 1
                else np.nan
            )
            rets = np.diff(c[max(0, loc - 31) : loc + 1]) if loc >= 1 else np.array([])
            ret_std30 = float(np.std(rets)) if len(rets) > 1 else np.nan
            net_disp30 = (
                float(abs(c[loc] - c[loc - 30]))
                if loc >= 30
                else (float(abs(c[loc] - c[0])) if loc > 0 else 0.0)
            )
            apr = win(hi, 30)
            aprl = win(lo, 30)
            approach_range30 = float(apr.max() - aprl.min()) if len(apr) else np.nan
            trows.append(
                {
                    "date": date_str,
                    "zone_idx": zi,
                    "event_ts": str(first_et),
                    "session": first_sess,
                    "level_type": z.names[0],
                    "direction": ("LONG" if z.side.value == "LOW" else "SHORT"),
                    "rep_price": rep,
                    "n_levels": len(z.names),
                    "lookahead": bool(first_et < af),
                    "rep_is_raw_extreme": bool(is_raw),
                    "bar_index": int(loc),
                    "vol_atr14_pts": atr14,
                    "ret_std30_pts": ret_std30,
                    "net_disp30_pts": net_disp30,
                    "approach_range30_pts": approach_range30,
                    "dist_session_open_pts": float(abs(rep - sess_open_px)),
                    "dist_rth_open_pts": float(abs(rep - rth_open_px))
                    if not np.isnan(rth_open_px)
                    else np.nan,
                    "vol_touch": float(vol[loc]),
                    "vol_mean30": float(np.mean(win(vol, 30))) if loc else np.nan,
                }
            )
    return pd.DataFrame(zrows), pd.DataFrame(trows), new_state


def available(start, end):
    out = []
    d = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    while d <= d1:
        if (DATA_DIR / SYM / d.isoformat()).is_dir():
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--warm", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    warm_start = (date.fromisoformat(a.start) - timedelta(days=a.warm)).isoformat()
    target = set(available(a.start, a.end))
    prev = (None, None, None)
    for ds in available(warm_start, a.end):
        zf = OUT / f"zones_{ds}.parquet"
        if ds in target and zf.exists():
            prev = _get_session_hl_for_date(DATA_DIR, SYM, ds, util, *prev)
            continue
        try:
            z, t, prev = probe_date(ds, prev)
        except Exception as e:
            sys.stderr.write(f"FAIL {ds}: {e}\n")
            with contextlib.suppress(Exception):
                prev = _get_session_hl_for_date(DATA_DIR, SYM, ds, util, *prev)
            continue
        if ds in target and z is not None:
            z.to_parquet(zf, index=False)
            (t if t is not None else pd.DataFrame()).to_parquet(
                OUT / f"touches_{ds}.parquet", index=False
            )
            print(f"{ds}: {len(z)} zones, {0 if t is None else len(t)} touches", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
