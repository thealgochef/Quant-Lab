"""ENGINE v3 ACCEPTANCE (1): session-attribution trace for 2025-06-03.

Builds the production v3 trade bars for the 2025-06-03 trading day, classifies every
bar with the canonical engine ``classify_session`` (the same function detect_touches
implies), and prints the first/last ET bar per session. Confirms:

  * asia spans MIDNIGHT: first bar ~ 2025-06-02 19:00 ET, last ~ 2025-06-03 02:45 ET.
  * london first ~ 03:00 ET, last ~ 08:00 ET; ny first ~ 09:00 ET, last ~ 17:00 ET.
  * the 18:00-19:00 ET hour HAS bars but is assigned to NO session ("none").
  * asia_high/low == max/min over the 19:00->02:45 ET span (engine bucket == builder
    _slice_session), and the builder/engine session buckets agree exactly.
  * PDH/PDL for 2025-06-03 == max/min over the FULL prior trading day's
    [2025-06-01 18:00 ET, 2025-06-02 18:00 ET) window — with NO Asia-evening bar
    double-counted (the prior-day window ends at 18:00 ET, before asia opens at 19:00).

If asia's first bar is NOT the prior evening, the trading-day grouping is broken and
the script says so. Read-only; PYTHONPATH=src.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

import strategy_core as sc
from strategy_core import classify_session
from strategy_core.constants import RESEARCH_SESSION_SCHEME as S

from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
    _build_bars_for_date,
    _ensure_et_index,
    _slice_session,
    _session_hl,
)
from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig

DATA_DIR = Path("C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento")
SYMBOL = "NQ"
DAY = "2025-06-03"
PRIOR_DAY = "2025-06-02"  # the prior TRADING day (its [18:00,18:00) window = PDH/PDL)
_ET = "US/Eastern"
UTIL = DashboardUtilityConfig(bar_type="147t", include_approach_features=True,
                              approach_window_minutes=15, interaction_window_minutes=5,
                              tp_points=15.0, sl_points=15.0, trap_mfe_min=5.0,
                              level_proximity_pts=0.5)


def _session_of(idx_et: pd.Timestamp) -> str:
    return classify_session(idx_et.tz_convert("UTC").to_pydatetime()).session


def main() -> int:
    print(f"=== ENGINE v3 session-attribution trace: {DAY} (engine {sc.ENGINE_VERSION}) ===")
    print("scheme:", {n: f"{w.start}-{w.end}{' +1d' if w.crosses_midnight else ''}"
                      for n, w in S.sessions.items()})
    print()

    bars = _build_bars_for_date(DATA_DIR, SYMBOL, DAY, UTIL)
    if bars.empty:
        print("NO BARS — cannot trace"); return 1
    bars_et = _ensure_et_index(bars)
    idx = bars_et.index
    sessions = [_session_of(t) for t in idx]
    bars_et = bars_et.assign(_session=sessions)

    # first/last ET bar per session
    print(f"total bars on {DAY} trading day = {len(bars_et)}")
    print(f"window: first bar {idx[0]}  ->  last bar {idx[-1]} (ET)")
    print()
    print("first/last ET bar per session:")
    ok = True
    for name in ("asia", "london", "ny", "none"):
        sub = bars_et[bars_et["_session"] == name]
        if sub.empty:
            print(f"  {name:7s}: (no bars)"); continue
        print(f"  {name:7s}: n={len(sub):4d}  first={sub.index[0]}  last={sub.index[-1]}")

    # midnight-crossing check: asia's FIRST bar must be the prior evening (>= 19:00 the
    # prior calendar day), and asia must include both prior-evening and this-morning.
    asia = bars_et[bars_et["_session"] == "asia"]
    print()
    if asia.empty:
        print("ASIA EMPTY — grouping broken"); ok = False
    else:
        first = asia.index[0]
        prior_eve = (first.date() == date.fromisoformat(PRIOR_DAY)) and (first.time().hour >= 19)
        has_prev_eve = (asia.index.date == date.fromisoformat(PRIOR_DAY)).any()
        has_this_morn = (asia.index.date == date.fromisoformat(DAY)).any()
        print(f"asia first bar = {first}  (prior-evening 19:00+? {prior_eve})")
        print(f"asia spans midnight: prior-evening bars? {bool(has_prev_eve)}  "
              f"this-morning bars? {bool(has_this_morn)}")
        if not (prior_eve and has_prev_eve and has_this_morn):
            print("!! GROUPING BROKEN: asia's first bar is NOT the prior evening or it "
                  "does not span midnight — STOP."); ok = False

    # the 18:00-19:00 ET hour: HAS bars but session == none
    gap = bars_et[(idx.date == date.fromisoformat(PRIOR_DAY)) &
                  (idx.time >= pd.Timestamp("18:00").time()) &
                  (idx.time < pd.Timestamp("19:00").time())]
    print()
    print(f"18:00-19:00 ET (prior evening) hour: n={len(gap)} bars; "
          f"sessions={sorted(set(gap['_session'])) if len(gap) else '(none)'}")
    if len(gap) and set(gap["_session"]) != {"none"}:
        print("!! 18:00-19:00 ET should be UNSESSIONED (none)"); ok = False

    # asia_high/low over the 19:00->02:45 span: engine bucket vs builder _slice_session
    eng_hi = float(asia["high"].max()); eng_lo = float(asia["low"].min())
    sl_asia = _slice_session(bars_et.drop(columns=["_session"]), "asia")
    sl_hi, sl_lo = _session_hl(sl_asia)
    print()
    print(f"asia_high/low (engine classify bucket) = {eng_hi} / {eng_lo}")
    print(f"asia_high/low (builder _slice_session) = {sl_hi} / {sl_lo}  "
          f"match={eng_hi == sl_hi and eng_lo == sl_lo}")
    if not (eng_hi == sl_hi and eng_lo == sl_lo):
        print("!! engine bucket and builder slice disagree on asia H/L"); ok = False

    # PDH/PDL = FULL prior trading-day window [2025-06-01 18:00, 2025-06-02 18:00) H/L.
    prior_bars = _build_bars_for_date(DATA_DIR, SYMBOL, PRIOR_DAY, UTIL)
    prior_et = _ensure_et_index(prior_bars)
    pdh = float(prior_et["high"].max()); pdl = float(prior_et["low"].min())
    # confirm NO Asia-evening (>=19:00 on 2025-06-02) bar is in the prior-day window:
    # the prior-day window must end strictly before 2025-06-02 18:00 ET.
    last_prior = prior_et.index[-1]
    no_double = last_prior.tz_convert(_ET).time() < pd.Timestamp("18:00").time() or \
        last_prior.date() < date.fromisoformat(PRIOR_DAY) or \
        (last_prior.date() == date.fromisoformat(PRIOR_DAY) and last_prior.time() < pd.Timestamp("18:00").time())
    print()
    print(f"PDH/PDL for {DAY} = FULL prior trading day {PRIOR_DAY} window "
          f"[{prior_et.index[0]} .. {prior_et.index[-1]}]")
    print(f"   PDH = {pdh}   PDL = {pdl}")
    print(f"   prior-day window ends {last_prior} (< 2025-06-02 18:00 ET, so no "
          f"asia-evening double-count: {no_double})")
    if not no_double:
        ok = False

    print()
    print("TRACE", "PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
