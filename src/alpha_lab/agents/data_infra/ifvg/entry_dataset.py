"""Entry-candidate dataset: label families + gate features over capture rows.

One row per ``entry_candidate`` capture row (BOTH families, selected or not —
explicit family pooling), composed offline from the capture frame + the cached
Phase-A bars. Labels go through the SHARED SC kernel
(``resolve_ifvg_outcome`` — per-call r-relative pairs); QL adds only
composition: realized/net columns (cost model: NQ $2.64/side commission +
0.5-tick slippage per side), next-open slippage measurement, and the setup-
stage feature joins (tap/parent/opposing/inversion measurements pivoted onto
the entry row by ``setup_id``).

Direction is derived, not stored: ``stop < entry`` == LONG (risk >= 1 tick
guarantees strict inequality).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from strategy_core.strategies.ifvg_smc.labels import resolve_ifvg_outcome
from strategy_core.types import Bar, Direction

from .config import IfvgCaptureConfig
from .day_artifacts import load_day_artifacts

__all__ = ["build_entry_dataset", "NQ_COST_POINTS_ROUND_TURN"]

#: NQ round-turn friction in POINTS: commission $2.64/side x2 = $5.28 -> /$20
#: per point = 0.264 pts, plus 0.5 tick slippage per side = 0.25 pts.
NQ_COST_POINTS_ROUND_TURN = 0.514

_R_FAMILIES = (1.0, 1.5, 2.0)

_STAGE_FEATURES = {
    "htf_tap": (
        "htf_tf_seconds",
        "fvg_size_ticks",
        "penetration_ticks",
        "ce_reached",
        "htf_age_seconds",
        "remaining_fraction",
        "registry_live_count",
        "nearest_level_kind",
        "nearest_level_distance_ticks",
    ),
    "parent_candidate": (
        "parent_tf_seconds",
        "fvg_size_ticks",
        "distance_to_htf_ticks",
        "elapsed_1m_bars_since_tap",
        "confirmed_after",
        "fully_formed_after",
    ),
    "parent_lock": ("penetration_ticks", "ce_reached", "elapsed_1m_bars_since_selection"),
    "opposing": (
        "fvg_size_ticks",
        "distance_to_parent_ticks",
        "elapsed_1m_bars_since_lock",
        "confirmed_after",
        "fully_formed_after",
    ),
    "inversion": (
        "close_through_margin_ticks",
        "bars_armed_to_inversion",
        "sweep_sweep_confirmed",
        "sweep_swept_kinds",
        "sweep_max_penetration_ticks",
        "sweep_nearest_unswept_distance_ticks",
    ),
}

_STAGE_PREFIX = {
    "htf_tap": "tap",
    "parent_candidate": "parent",
    "parent_lock": "lock",
    "opposing": "opp",
    "inversion": "inv",
}


def _atr14_ticks(bars_1m: list[Bar], before: datetime) -> float | None:
    """Mean true range (ticks) over the last 14 COMPLETE 1m bars closing
    strictly before ``before``."""
    window: list[int] = []
    prev_close: int | None = None
    for bar in bars_1m:
        if bar.close_ts_utc >= before:
            break
        tr = bar.high_ticks - bar.low_ticks
        if prev_close is not None:
            tr = max(tr, abs(bar.high_ticks - prev_close), abs(bar.low_ticks - prev_close))
        window.append(tr)
        prev_close = bar.close_ticks
    if len(window) < 14:
        return None
    return sum(window[-14:]) / 14.0


def build_entry_dataset(capture: pd.DataFrame, cfg: IfvgCaptureConfig) -> pd.DataFrame:
    """Compose the per-entry-candidate research dataset from a capture frame."""
    if capture.empty:
        return pd.DataFrame()
    entries = capture[capture["kind"] == "entry_candidate"].copy()
    if entries.empty:
        return pd.DataFrame()

    # ── stage-feature pivots (selected rows only, keyed by setup_id) ─────────
    for kind, columns in _STAGE_FEATURES.items():
        stage = capture[capture["kind"] == kind]
        if "selected" in stage.columns and kind in ("htf_tap", "parent_candidate", "opposing"):
            stage = stage[stage["selected"] == True]  # noqa: E712 — pandas mask
        stage = stage.drop_duplicates(subset=["envelope_setup_id"], keep="last")
        prefix = _STAGE_PREFIX[kind]
        available = [c for c in columns if c in stage.columns]
        renamed = stage[["envelope_setup_id", *available]].rename(
            columns={c: f"{prefix}_{c}" for c in available}
        )
        entries = entries.merge(renamed, on="envelope_setup_id", how="left")

    rows = []
    bars_cache: dict[str, list[Bar]] = {}
    for row in entries.itertuples(index=False):
        day = row.envelope_trading_day
        day_str = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]
        if day_str not in bars_cache:
            artifacts = load_day_artifacts(day_str, cfg, expected_seeds=None)
            bars_cache[day_str] = (
                [b for b in artifacts.bars if b.timeframe_ticks == 60] if artifacts else []
            )
        bars_1m = bars_cache[day_str]
        entry_ts = row.envelope_ts_utc
        entry_ticks = int(row.entry_ticks)
        stop_ticks = int(row.stop_ticks)
        direction = Direction.LONG if stop_ticks < entry_ticks else Direction.SHORT
        sign = 1 if direction is Direction.LONG else -1
        forward = [b for b in bars_1m if b.close_ts_utc > entry_ts]
        risk_pts = abs(entry_ticks - stop_ticks) * cfg.tick_size

        out = {
            "setup_id": row.envelope_setup_id,
            "trading_day": day_str,
            "entry_ts_utc": entry_ts,
            "entry_family": row.entry_family,
            "selected": bool(row.selected),
            "drop_reason": row.drop_reason,
            "direction": direction.value,
            "entry_ticks": entry_ticks,
            "stop_ticks": stop_ticks,
            "risk_ticks": int(row.risk_ticks),
            "risk_points": risk_pts,
            "session_engine": row.in_engine_session,
            "session_doc": row.in_doc_session,
            "bars_since_inversion": row.bars_since_inversion,
            "entry_to_parent_ticks": row.entry_to_parent_ticks,
            "is_warmup": bool(row.is_warmup),
            "days_of_htf_history": int(row.days_of_htf_history),
            "profile_hash": row.envelope_profile_hash,
            "strategy_version": row.envelope_strategy_version,
        }
        for col in entries.columns:
            if col.split("_", 1)[0] in _STAGE_PREFIX.values() and "_" in col:
                out[col] = getattr(row, col, None)

        atr = _atr14_ticks(bars_1m, entry_ts)
        out["atr14_1m_ticks"] = atr
        out["risk_atr"] = (abs(entry_ticks - stop_ticks) / atr) if atr else None
        highs = [b.high_ticks for b in bars_1m if b.close_ts_utc <= entry_ts]
        lows = [b.low_ticks for b in bars_1m if b.close_ts_utc <= entry_ts]
        if highs and lows and max(highs) > min(lows):
            day_hi, day_lo = max(highs), min(lows)
            out["day_range_ticks"] = day_hi - day_lo
            out["position_in_day_range"] = (entry_ticks - day_lo) / (day_hi - day_lo)
            out["dist_day_high_ticks"] = day_hi - entry_ticks
            out["dist_day_low_ticks"] = entry_ticks - day_lo
        else:
            out["day_range_ticks"] = out["position_in_day_range"] = None
            out["dist_day_high_ticks"] = out["dist_day_low_ticks"] = None

        # ── labels through the shared kernel, per R family ───────────────────
        if forward:
            for r in _R_FAMILIES:
                outcome = resolve_ifvg_outcome(
                    entry_ticks=entry_ticks,
                    stop_ticks=stop_ticks,
                    direction=direction,
                    forward_bars_1m=forward,
                    tick_size=cfg.tick_size,
                    r_multiple=r,
                )
                tag = f"r{str(r).replace('.', '')}"
                out[f"label_{tag}"] = outcome.label
                out[f"bars_to_res_{tag}"] = outcome.bars_to_resolution
                if r == 1.0:
                    out["mfe_r"] = outcome.mfe_r
                    out["mae_r"] = outcome.mae_r
                # Realized points: barrier for win/loss, day-end close for timeout.
                if outcome.label == "win":
                    realized = r * risk_pts
                elif outcome.label == "loss":
                    realized = -risk_pts
                else:
                    realized = sign * (forward[-1].close_ticks - entry_ticks) * cfg.tick_size
                out[f"realized_pts_{tag}"] = realized
                out[f"realized_r_net_{tag}"] = (
                    (realized - NQ_COST_POINTS_ROUND_TURN) / risk_pts if risk_pts else None
                )
            out["entry_slippage_next_open_pts"] = (
                sign * (forward[0].open_ticks - entry_ticks) * cfg.tick_size
            )
            out["label_window_end"] = forward[-1].close_ts_utc
        else:
            for r in _R_FAMILIES:
                tag = f"r{str(r).replace('.', '')}"
                out[f"label_{tag}"] = "no_forward"
            out["label_window_end"] = entry_ts
        rows.append(out)
    return pd.DataFrame(rows)
