"""Phase 8.1 golden capture / equivalence — honest-entry orchestration relocation.

Captures the FULL per-touch honest-entry orchestration record (decision_ts, drop
flag+reason, entry_price, forward-window bounds, label/encoded/mfe/mae) for every
DETECTED touch on the 3 core days, driving the SAME production bars/levels/touches
the dashboard-utility builder produces.

Two modes:
  --mode golden   : run the INLINE Phase-8 orchestration (copied verbatim from the
                    pre-refactor engine_decision.process_single_date_engine honest
                    path) and write golden.json.
  --mode engine   : run the SAME loop but route the orchestration through
                    strategy_core.resolve_honest_outcome (the relocated engine
                    function), write engine.json.
  --mode compare  : byte-compare golden.json vs engine.json, field-by-field.

The capture builds bars/levels/touches with the production helpers (so the touch set
is the real one), then for each touch records the orchestration outcome. resolve_outcome
is the SAME pure function in both modes; the ONLY thing under test is the orchestration
(decision_ts / flatten / cutoff / entry / forward selection) relocation.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

CQL_SRC = r"C:/Users/gonza/Documents/Claude-Quant-Lab/src"
SC_SRC = r"C:/Users/gonza/Documents/Strategy-core/src"
for p in (SC_SRC, CQL_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

import strategy_core as sc  # noqa: E402
from strategy_core import build_zones, detect_touches, resolve_outcome  # noqa: E402
from strategy_core.constants import (  # noqa: E402
    DECISION_OFFSET_MINUTES,
    FLATTEN_TIME,
)

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig  # noqa: E402
from alpha_lab.agents.data_infra.ml import dashboard_utility_builder as B  # noqa: E402
from alpha_lab.agents.data_infra.ml import engine_decision as ED  # noqa: E402
from alpha_lab.agents.data_infra.tick_store import TickStore  # noqa: E402

DATA_DIR = Path(r"C:/Users/gonza/Documents/Trade-Dashboard/data/databento")
SYMBOL = "NQ"
CORE_DAYS = ["2025-07-15", "2025-07-07", "2025-07-11"]
OUT_DIR = Path(r"C:/Users/gonza/Documents/Claude-Quant-Lab/data/experiment/phase8_1")
_ET = "US/Eastern"


def _prev_ny_hl(day: str, cfg) -> tuple[float, float] | None:
    """Prior trading day's NY-RTH H/L, walking back to the most recent available day."""
    d = date.fromisoformat(day)
    prev = None
    for back in range(1, 8):
        cand = (d - timedelta(days=back)).isoformat()
        if (DATA_DIR / SYMBOL / cand / "mbp10.parquet").exists():
            prev = cand
            break
    if prev is None:
        return None
    ny, _asia, _london = B._get_session_hl_for_date(
        DATA_DIR, SYMBOL, prev, cfg, None, None, None
    )
    return ny


def _build_day(day: str, cfg):
    """Production bars_et + levels + engine touches for one day."""
    bars = B._build_bars_for_date(DATA_DIR, SYMBOL, day, cfg)
    if bars.empty:
        return None, None, None
    bars_et = B._ensure_et_index(bars)
    prev_ny = _prev_ny_hl(day, cfg)
    levels = B._compute_levels_for_date(bars_et, day, prev_ny, None, None)
    if not levels:
        return bars_et, levels, []
    td = date.fromisoformat(day)
    eng_bars = ED.bars_et_to_engine(bars_et, td, ED.TRADE_TICK)
    eng_levels = ED.levels_to_engine(levels)
    zones = build_zones(eng_levels)
    touches = detect_touches(eng_bars, zones, tick_size=ED.TRADE_TICK, trading_day=td)
    return bars_et, levels, touches


def _fwd_bounds(forward_bars):
    if not forward_bars:
        return None, None, 0
    first = forward_bars[0].close_ts_utc.isoformat()
    last = forward_bars[-1].close_ts_utc.isoformat()
    return first, last, len(forward_bars)


def _record(touch, *, drop, reason, decision_ts_utc, entry_price, forward_bars, outcome):
    first, last, n = _fwd_bounds(forward_bars)
    return {
        "touch_bar_ts": touch.bar_ts_utc.isoformat(),
        "rep_price": round(float(touch.representative_price), 6),
        "direction": touch.direction.value,
        "level_type": touch.level_type,
        "decision_ts": decision_ts_utc.isoformat() if decision_ts_utc else None,
        "drop": drop,
        "reason": reason,
        "entry_price": (round(float(entry_price), 6) if entry_price is not None else None),
        "fwd_first": first,
        "fwd_last": last,
        "fwd_count": n,
        "label": (outcome.label if outcome else None),
        "label_encoded": (outcome.label_encoded if outcome else None),
        "max_mfe": (outcome.max_mfe if outcome else None),
        "max_mae": (outcome.max_mae if outcome else None),
    }


def _capture_day_golden(day, bars_et, touches, cfg, entry_store):
    """INLINE Phase-8 orchestration, verbatim, capturing every touch."""
    td = date.fromisoformat(day)
    rth_cutoff = pd.Timestamp(f"{day} 16:15:00", tz=_ET)
    decision_offset = cfg.interaction_window_minutes
    recs = []
    for touch in touches:
        bar_ts_et = pd.Timestamp(touch.bar_ts_utc).tz_convert(_ET)
        decision_ts_et = bar_ts_et + pd.Timedelta(minutes=decision_offset)
        decision_ts_utc = ED._to_utc_dt(decision_ts_et)

        if ED._at_or_after_flatten(decision_ts_et):
            recs.append(_record(touch, drop=True, reason="flatten",
                                decision_ts_utc=decision_ts_utc, entry_price=None,
                                forward_bars=[], outcome=None))
            continue
        if decision_ts_et >= rth_cutoff:
            recs.append(_record(touch, drop=True, reason="cutoff",
                                decision_ts_utc=decision_ts_utc, entry_price=None,
                                forward_bars=[], outcome=None))
            continue

        entry_price = ED._trade_price_at(entry_store, SYMBOL, decision_ts_utc)
        if entry_price is None:
            recs.append(_record(touch, drop=True, reason="no_fill",
                                decision_ts_utc=decision_ts_utc, entry_price=None,
                                forward_bars=[], outcome=None))
            continue

        forward = bars_et[
            (bars_et.index > decision_ts_et) & (bars_et.index < rth_cutoff)
        ]
        forward_bars = ED.bars_et_to_engine(forward, td, ED.TRADE_TICK)
        if not forward_bars:
            recs.append(_record(touch, drop=True, reason="no_forward",
                                decision_ts_utc=decision_ts_utc, entry_price=entry_price,
                                forward_bars=[], outcome=None))
            continue

        outcome = resolve_outcome(
            entry_points=float(entry_price),
            direction=touch.direction,
            forward_bars=forward_bars,
            tick_size=ED.TRADE_TICK,
            tp_points=cfg.tp_points,
            sl_points=cfg.sl_points,
            trap_mfe_min=cfg.trap_mfe_min,
        )
        recs.append(_record(touch, drop=False, reason=None,
                            decision_ts_utc=decision_ts_utc, entry_price=entry_price,
                            forward_bars=forward_bars, outcome=outcome))
    return recs


def _capture_day_engine(day, bars_et, touches, cfg, entry_store):
    """SAME loop but route orchestration through resolve_honest_outcome."""
    td = date.fromisoformat(day)
    # full-day engine bars for the forward window (the engine slices internally)
    day_bars = ED.bars_et_to_engine(bars_et, td, ED.TRADE_TICK)

    def trade_price_at(ts_utc: datetime):
        return ED._trade_price_at(entry_store, SYMBOL, ts_utc)

    recs = []
    for touch in touches:
        res = sc.resolve_honest_outcome(
            touch,
            day_bars,
            trade_price_at,
            tick_size=ED.TRADE_TICK,
            tp_points=cfg.tp_points,
            sl_points=cfg.sl_points,
            trap_mfe_min=cfg.trap_mfe_min,
            decision_offset_minutes=cfg.interaction_window_minutes,
        )
        if isinstance(res, sc.HonestEntryDrop):
            recs.append(_record(touch, drop=True, reason=res.reason,
                                decision_ts_utc=res.decision_ts_utc,
                                entry_price=res.entry_price,
                                forward_bars=[], outcome=None))
        else:
            # recompute forward bounds the same way the engine selected them
            decision_ts_utc = touch.bar_ts_utc + timedelta(
                minutes=cfg.interaction_window_minutes
            )
            rth_cutoff = pd.Timestamp(f"{day} 16:15:00", tz=_ET)
            decision_ts_et = pd.Timestamp(decision_ts_utc).tz_convert(_ET)
            forward = bars_et[
                (bars_et.index > decision_ts_et) & (bars_et.index < rth_cutoff)
            ]
            forward_bars = ED.bars_et_to_engine(forward, td, ED.TRADE_TICK)
            entry_price = ED._trade_price_at(entry_store, SYMBOL, decision_ts_utc)
            recs.append(_record(touch, drop=False, reason=None,
                                decision_ts_utc=decision_ts_utc,
                                entry_price=entry_price,
                                forward_bars=forward_bars, outcome=res))
    return recs


def run(mode: str):
    cfg = DashboardUtilityConfig()
    all_recs: dict[str, list] = {}
    n_touch = 0
    n_drop = 0
    for day in CORE_DAYS:
        bars_et, levels, touches = _build_day(day, cfg)
        if touches is None:
            all_recs[day] = []
            continue
        entry_store = TickStore(DATA_DIR)
        entry_store.register_symbol_date(SYMBOL, day)
        try:
            if mode == "golden":
                recs = _capture_day_golden(day, bars_et, touches, cfg, entry_store)
            else:
                recs = _capture_day_engine(day, bars_et, touches, cfg, entry_store)
        finally:
            entry_store.close()
        all_recs[day] = recs
        n_touch += len(recs)
        n_drop += sum(1 for r in recs if r["drop"])
    out = OUT_DIR / (f"{mode}.json")
    out.write_text(json.dumps(all_recs, indent=2, sort_keys=True))
    print(f"[{mode}] wrote {out}: n_touches={n_touch} n_dropped={n_drop} "
          f"n_traded={n_touch - n_drop}")
    return all_recs


def compare():
    g = json.loads((OUT_DIR / "golden.json").read_text())
    e = json.loads((OUT_DIR / "engine.json").read_text())
    n_compared = 0
    n_drop = 0
    diffs = []
    days = sorted(set(g) | set(e))
    for day in days:
        gr = g.get(day, [])
        er = e.get(day, [])
        if len(gr) != len(er):
            diffs.append(f"{day}: touch count {len(gr)} vs {len(er)}")
            continue
        for i, (a, b) in enumerate(zip(gr, er)):
            n_compared += 1
            if a.get("drop"):
                n_drop += 1
            keys = sorted(set(a) | set(b))
            for k in keys:
                if a.get(k) != b.get(k):
                    diffs.append(
                        f"{day}[{i}] touch={a.get('touch_bar_ts')} field={k}: "
                        f"golden={a.get(k)!r} engine={b.get(k)!r}"
                    )
    if diffs:
        print("BYTE_IDENTICAL=NO")
        for d in diffs[:50]:
            print("  DIFF:", d)
        print(f"total diffs: {len(diffs)}")
    else:
        print("BYTE_IDENTICAL=YES")
    print(f"n_touches_compared={n_compared} n_dropped={n_drop} "
          f"n_traded={n_compared - n_drop}")
    return not diffs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["golden", "engine", "compare"], required=True)
    args = ap.parse_args()
    if args.mode == "compare":
        ok = compare()
        sys.exit(0 if ok else 1)
    else:
        run(args.mode)
