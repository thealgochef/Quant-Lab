"""Pure data/chart-payload builders for the IFVG Lab tab (plan Part B).

NO streamlit import here — everything is a pure function over frames/paths so
``tests/agents/test_ifvg_lab_tab.py`` can exercise the logic headlessly:

* per-day payload loading (sealed-day guard lives IN the loader),
* FVG zone dedup by ``fvg_id`` with role/style mapping,
* level step-series honoring ``available_from``,
* stage-marker extraction per ``setup_id``,
* trade overlay math (entry/SL/TP 1R-1.5R-2R/MFE-MAE/manipulation swing),
* run-scoped trade matching (``setup_id + entry_family + entry_ts``),
* session background bands (engine + doc ET schemes),
* the "all gaps (recomputed)" SC-detector audit pass,
* the Plotly figures (single figure per render; zones are deduped FIRST so the
  shape count stays at ~tens, never one shape per tap row).

The streamlit layer (``ifvg_lab_tab.py``) is a thin wrapper over these.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from strategy_core.constants import IFVG_DOC_SESSIONS, RESEARCH_SESSION_SCHEME

# Single sources of truth: sealed boundary + both ET session schemes.
from alpha_lab.agents.data_infra.ifvg.config import SEALED_HOLDOUT_START

__all__ = [
    "TICK_SIZE",
    "TIMEFRAME_OPTIONS",
    "ENGINE_SESSION_WINDOWS",
    "DOC_SESSION_WINDOWS",
    "SEALED_HOLDOUT_START",
    "list_replay_days",
    "load_day_payload",
    "dedup_zones",
    "zone_style",
    "level_series",
    "stage_markers",
    "manipulation_swing_price",
    "trade_overlays",
    "trade_key",
    "run_trade_keys",
    "scope_overlays",
    "sealed_replay_available",
    "scheme_windows",
    "session_bands",
    "recompute_day_gaps",
    "build_replay_figure",
    "build_equity_figure",
    "build_r_histogram_figure",
    "build_calibration_figure",
    "build_coverage_figure",
    "flatten_config",
    "config_diff_frame",
]

TICK_SIZE = 0.25

#: The 8 recorded timeframes (the exact bars the reducer consumed).
TIMEFRAME_OPTIONS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "10m": 600,
    "15m": 900,
    "30m": 1800,
    "1H": 3600,
    "4H": 14400,
}

_ET = ZoneInfo("America/New_York")

#: Engine scheme (strategy_core RESEARCH_SESSION_SCHEME) as HH:MM ET windows.
ENGINE_SESSION_WINDOWS: dict[str, tuple[str, str]] = {
    name: (w.start.strftime("%H:%M"), w.end.strftime("%H:%M"))
    for name, w in RESEARCH_SESSION_SCHEME.sessions.items()
}
#: Doc scheme (ifvg-strat.md §6.4, strategy_core IFVG_DOC_SESSIONS).
DOC_SESSION_WINDOWS: dict[str, tuple[str, str]] = dict(IFVG_DOC_SESSIONS)


def scheme_windows(scheme) -> dict[str, tuple[str, str]]:
    """A runtime SessionScheme's windows as HH:MM ET pairs (band input) — so a
    custom capture profile's bands show ITS windows, not the canonical ones."""
    return {
        name: (w.start.strftime("%H:%M"), w.end.strftime("%H:%M"))
        for name, w in scheme.sessions.items()
    }


# ── per-day payload loading (SEALED guard lives HERE, not in the widget) ──────


def _day_paths(symbol_dir: Path | str, artifacts_tag: str, capture_tag: str, day: str):
    day_dir = Path(symbol_dir) / day
    return (
        day_dir / f"ifvg_tbars_{artifacts_tag}.parquet",
        day_dir / f"ifvg_levels_{artifacts_tag}.parquet",
        day_dir / f"ifvg_capture_{capture_tag}.parquet",
    )


def list_replay_days(
    symbol_dir: Path | str,
    artifacts_tag: str,
    capture_tag: str,
    *,
    sealed: bool = False,
) -> list[str]:
    """Store days that have all three replay parquets.

    THE SEALED FILTER IS ENFORCED IN THIS LOADER, not in any widget: the
    default listing contains only days strictly before ``SEALED_HOLDOUT_START``.
    ``sealed=True`` returns ONLY sealed days and exists solely for the
    ledger-gated sealed-replay unlock path.
    """
    root = Path(symbol_dir)
    if not root.exists():
        return []
    days: list[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        day = entry.name
        if (day >= SEALED_HOLDOUT_START) != sealed:
            continue
        if all(p.exists() for p in _day_paths(root, artifacts_tag, capture_tag, day)):
            days.append(day)
    return days


def load_day_payload(
    symbol_dir: Path | str,
    artifacts_tag: str,
    capture_tag: str,
    day: str,
    *,
    allow_sealed: bool = False,
) -> dict:
    """Read the three per-day parquets (read-only). REFUSES sealed days unless
    the caller is the explicit ledger-gated sealed-replay path."""
    if day >= SEALED_HOLDOUT_START and not allow_sealed:
        raise ValueError(
            f"day {day} is in the sealed holdout (>= {SEALED_HOLDOUT_START}); "
            "sealed replay unlocks only for a saved run with ledgered sealed validations"
        )
    bars_path, levels_path, capture_path = _day_paths(
        symbol_dir, artifacts_tag, capture_tag, day
    )
    return {
        "day": day,
        "bars": pd.read_parquet(bars_path),
        "levels": pd.read_parquet(levels_path),
        "capture": pd.read_parquet(capture_path),
    }


# ── FVG zones: dedup by fvg_id + role/style mapping ───────────────────────────

_KIND_TO_ROLE = {"htf_tap": "htf", "parent_candidate": "parent", "opposing": "opposing"}
#: When one fvg_id appears in several roles, the highest-priority role wins.
_ROLE_PRIORITY = {"htf": 0, "parent": 1, "opposing": 2, "entry_fvg": 3}

ZONE_COLORS = {
    "htf": "#4C78A8",  # tapped HTF gaps (1H/4H)
    "parent": "#9467BD",  # parent-TF gaps
    "opposing": "#E45756",  # opposing 1m gaps
    "entry_fvg": "#2CA02C",  # the fresh 1m entry FVG
    "recomputed": "#8C8C8C",  # detector audit pass — visually distinct
}


def _rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def dedup_zones(capture: pd.DataFrame) -> pd.DataFrame:
    """One row per ``fvg_id`` from the logged capture rows.

    Role coding: htf (htf_tap rows), parent (parent_candidate), opposing,
    entry_fvg (entry_candidate rows' ``entry_fvg_*`` block). The LAST logged
    state wins for ``selected``/``drop_reason`` (a parent/opposing gap can be
    selected then replaced); role conflicts resolve to the highest-priority
    role. ``selected`` is True where the capture logged no explicit False
    (htf taps carry no selection flag).
    """
    columns_out = [
        "fvg_id",
        "role",
        "gap_low_ticks",
        "gap_high_ticks",
        "direction",
        "timeframe_seconds",
        "confirmed_ts_utc",
        "selected",
        "drop_reason",
        "setup_id",
        "ts",
    ]
    # Empty store days serialize as zero-column parquets — nothing to draw.
    if capture.empty or "kind" not in capture.columns:
        return pd.DataFrame(columns=columns_out)
    parts: list[pd.DataFrame] = []
    base = capture[capture["kind"].isin(_KIND_TO_ROLE)]
    if len(base):
        parts.append(
            pd.DataFrame(
                {
                    "fvg_id": base["fvg_fvg_id"],
                    "role": base["kind"].map(_KIND_TO_ROLE),
                    "gap_low_ticks": pd.to_numeric(base["fvg_gap_low_ticks"], errors="coerce"),
                    "gap_high_ticks": pd.to_numeric(base["fvg_gap_high_ticks"], errors="coerce"),
                    "direction": base["fvg_direction"],
                    "timeframe_seconds": pd.to_numeric(
                        base["fvg_timeframe_seconds"], errors="coerce"
                    ),
                    "confirmed_ts_utc": base["fvg_confirmed_ts_utc"],
                    "selected": base["selected"],
                    "drop_reason": base["drop_reason"],
                    "setup_id": base["envelope_setup_id"],
                    "ts": base["envelope_ts_utc"],
                }
            )
        )
    entries = capture[capture["kind"] == "entry_candidate"]
    if len(entries):
        parts.append(
            pd.DataFrame(
                {
                    "fvg_id": entries["entry_fvg_fvg_id"],
                    "role": "entry_fvg",
                    "gap_low_ticks": pd.to_numeric(
                        entries["entry_fvg_gap_low_ticks"], errors="coerce"
                    ),
                    "gap_high_ticks": pd.to_numeric(
                        entries["entry_fvg_gap_high_ticks"], errors="coerce"
                    ),
                    "direction": entries["entry_fvg_direction"],
                    "timeframe_seconds": pd.to_numeric(
                        entries["entry_fvg_timeframe_seconds"], errors="coerce"
                    ),
                    "confirmed_ts_utc": entries["entry_fvg_confirmed_ts_utc"],
                    "selected": entries["selected"],
                    "drop_reason": entries["drop_reason"],
                    "setup_id": entries["envelope_setup_id"],
                    "ts": entries["envelope_ts_utc"],
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=columns_out)
    zones = pd.concat(parts, ignore_index=True)
    zones = zones[zones["fvg_id"].notna() & (zones["fvg_id"].astype(str) != "")]
    if zones.empty:
        return pd.DataFrame(columns=columns_out)
    zones["_prio"] = zones["role"].map(_ROLE_PRIORITY)
    # Sort so the LAST row per fvg_id = highest-priority role, latest state.
    zones = zones.sort_values(["_prio", "ts"], kind="mergesort")
    out = zones.groupby("fvg_id", as_index=False).last()
    out["selected"] = out["selected"].map(lambda v: True if pd.isna(v) else bool(v))
    out["drop_reason"] = out["drop_reason"].where(out["drop_reason"].notna(), None)
    return out.drop(columns=["_prio"]).sort_values("confirmed_ts_utc").reset_index(drop=True)


def zone_style(role: str, selected: bool, *, recomputed: bool = False) -> dict:
    """Plotly rect style: role color; selected solid vs non-selected/replaced
    dimmed+dashed; recomputed gaps dotted grey (visually distinct)."""
    if recomputed:
        color = ZONE_COLORS["recomputed"]
        return {
            "fillcolor": _rgba(color, 0.05),
            "line": {"color": color, "width": 1, "dash": "dot"},
        }
    color = ZONE_COLORS.get(role, ZONE_COLORS["recomputed"])
    if selected:
        return {
            "fillcolor": _rgba(color, 0.20),
            "line": {"color": color, "width": 1.5, "dash": "solid"},
        }
    return {
        "fillcolor": _rgba(color, 0.06),
        "line": {"color": color, "width": 1, "dash": "dash"},
    }


# ── levels: step-series honoring available_from ───────────────────────────────

LEVEL_STYLE = {
    "pdh": {"color": "#54585E", "dash": "solid"},
    "pdl": {"color": "#54585E", "dash": "solid"},
    "asia_high": {"color": "#2E9990", "dash": "dash"},
    "asia_low": {"color": "#2E9990", "dash": "dash"},
    "london_high": {"color": "#B8860B", "dash": "dash"},
    "london_low": {"color": "#B8860B", "dash": "dash"},
    "ny_high": {"color": "#4C78A8", "dash": "dash"},
    "ny_low": {"color": "#4C78A8", "dash": "dash"},
    "prev_ny_high": {"color": "#8C8C8C", "dash": "dot"},
    "prev_ny_low": {"color": "#8C8C8C", "dash": "dot"},
}


def level_series(levels: pd.DataFrame) -> pd.DataFrame:
    """Tidy per-name step series over the 1m snapshot timeline.

    ``price`` is masked to NaN at every snapshot BEFORE the level's
    ``available_from`` instant, so a level line only ever draws from its
    availability time (the v3 zero-lookahead discipline). A NaT
    ``available_from`` is treated as always-available (defensive)."""
    if levels.empty:
        return pd.DataFrame(columns=["name", "close_ts_utc", "price"])
    out = levels[["name", "close_ts_utc", "price", "available_from"]].copy()
    close = pd.to_datetime(out["close_ts_utc"], utc=True)
    avail = pd.to_datetime(out["available_from"], utc=True, errors="coerce")
    hidden = avail.notna() & (close < avail)
    out["price"] = pd.to_numeric(out["price"], errors="coerce").where(~hidden, np.nan)
    return out[["name", "close_ts_utc", "price"]].reset_index(drop=True)


# ── stage markers per setup ───────────────────────────────────────────────────

STAGE_ORDER = ("tap", "parent", "lock", "armed", "inversion", "entry", "resolution")

_RESOLUTION_STAGE_COLS = {
    "tap": "tap_ts_utc",
    "parent": "parent_confirmed_ts_utc",
    "lock": "lock_ts_utc",
    "armed": "armed_ts_utc",
    "inversion": "inversion_ts_utc",
    "entry": "entry_ts_utc",
    "resolution": "envelope_ts_utc",
}


def _fmt(value) -> str:
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


def stage_markers(capture: pd.DataFrame, setup_id: str) -> list[dict]:
    """Ordered stage markers for one setup: tap -> parent -> lock -> armed ->
    inversion (sweep annotation) -> entry -> resolution.

    Timestamps come from the setup's resolution row (the full logged chain);
    setups without a resolution row fall back to the per-kind emission rows.
    Tooltips carry the tap penetration/CE/conflicted fields and the inversion
    sweep annotation (swept kinds, max penetration, confirmed flag)."""
    if capture.empty or "envelope_setup_id" not in capture.columns:
        return []
    sub = capture[capture["envelope_setup_id"] == setup_id]
    if sub.empty:
        return []
    res_rows = sub[sub["kind"] == "resolution"]
    stages: dict[str, pd.Timestamp] = {}
    if len(res_rows):
        res = res_rows.iloc[-1]
        for stage, col in _RESOLUTION_STAGE_COLS.items():
            ts = res.get(col)
            if pd.notna(ts):
                stages[stage] = pd.Timestamp(ts)
    else:
        fallback = {
            "tap": ("htf_tap", "first"),
            "parent": ("parent_candidate", "first"),
            "lock": ("parent_lock", "first"),
            "armed": ("opposing", "last"),
            "inversion": ("inversion", "first"),
            "entry": ("entry_candidate", "first"),
        }
        for stage, (kind, which) in fallback.items():
            rows = sub[sub["kind"] == kind]
            if len(rows):
                ts = rows.iloc[0 if which == "first" else -1]["envelope_ts_utc"]
                if pd.notna(ts):
                    stages[stage] = pd.Timestamp(ts)

    texts: dict[str, str] = {}
    taps = sub[sub["kind"] == "htf_tap"]
    if len(taps):
        tap = taps.iloc[0]
        bits = [f"penetration {_fmt(tap.get('penetration_ticks'))}t"]
        bits.append(f"CE {'reached' if bool(tap.get('ce_reached')) else 'not reached'}")
        if bool(tap.get("conflicted")):
            bits.append("CONFLICTED")
        if pd.notna(tap.get("nearest_level_kind")):
            bits.append(f"near {tap.get('nearest_level_kind')}")
        texts["tap"] = "; ".join(bits)
    invs = sub[sub["kind"] == "inversion"]
    if len(invs):
        inv = invs.iloc[-1]
        swept = inv.get("sweep_swept_kinds")
        try:
            kinds = json.loads(swept) if isinstance(swept, str) else list(swept or [])
        except (TypeError, ValueError):
            kinds = [str(swept)]
        uniq = sorted(set(map(str, kinds)))
        texts["inversion"] = (
            f"sweep {'CONFIRMED' if bool(inv.get('sweep_sweep_confirmed')) else 'unconfirmed'}; "
            f"swept {', '.join(uniq) if uniq else 'nothing'}; "
            f"max penetration {_fmt(inv.get('sweep_max_penetration_ticks'))}t; "
            f"close-through {_fmt(inv.get('close_through_margin_ticks'))}t"
        )
    if len(res_rows):
        texts["resolution"] = str(res_rows.iloc[-1].get("resolution"))

    return [
        {"stage": stage, "ts": stages[stage], "text": texts.get(stage, stage)}
        for stage in STAGE_ORDER
        if stage in stages
    ]


# ── trade overlays ────────────────────────────────────────────────────────────


def manipulation_swing_price(
    entry_ticks: float, stop_ticks: float, *, tick_size: float = TICK_SIZE
) -> float:
    """The manipulation swing recovered from the logged stop -/+ the 1-tick
    buffer: LONG swing low = stop + 1; SHORT swing high = stop - 1."""
    is_long = stop_ticks < entry_ticks
    swing_ticks = stop_ticks + 1 if is_long else stop_ticks - 1
    return swing_ticks * tick_size


def trade_overlays(capture: pd.DataFrame, *, tick_size: float = TICK_SIZE) -> list[dict]:
    """One overlay dict per logged entry candidate (selected AND dropped).

    Prices are ticks * tick_size. TP levels at entry +/- r * risk for
    r in {1, 1.5, 2}; MFE/MAE excursion levels from the resolution row's
    ``mfe_ticks``/``mae_ticks``; outcome/resolution ts attach only to the
    executed (selected) family. Direction derives from stop-vs-entry."""
    if capture.empty or "kind" not in capture.columns:
        return []
    entries = capture[capture["kind"] == "entry_candidate"]
    if entries.empty:
        return []
    res_by_sid: dict[str, pd.Series] = {}
    for _, row in capture[capture["kind"] == "resolution"].iterrows():
        res_by_sid[str(row["envelope_setup_id"])] = row

    out: list[dict] = []
    for _, row in entries.iterrows():
        if pd.isna(row.get("entry_ticks")) or pd.isna(row.get("stop_ticks")):
            continue
        sid = str(row["envelope_setup_id"])
        family = str(row["entry_family"])
        selected = bool(row["selected"]) if pd.notna(row["selected"]) else False
        entry_ticks = float(row["entry_ticks"])
        stop_ticks = float(row["stop_ticks"])
        is_long = stop_ticks < entry_ticks
        sign = 1.0 if is_long else -1.0
        risk_ticks = abs(entry_ticks - stop_ticks)
        entry_price = entry_ticks * tick_size
        overlay = {
            "setup_id": sid,
            "entry_family": family,
            "selected": selected,
            "drop_reason": (
                str(row["drop_reason"]) if pd.notna(row.get("drop_reason")) else None
            ),
            "direction": "LONG" if is_long else "SHORT",
            "entry_ts": pd.Timestamp(row["envelope_ts_utc"]),
            "entry_price": entry_price,
            "stop_price": stop_ticks * tick_size,
            "risk_ticks": risk_ticks,
            "risk_points": risk_ticks * tick_size,
            "tp_prices": {
                r: entry_price + sign * r * risk_ticks * tick_size for r in (1.0, 1.5, 2.0)
            },
            "swing_price": manipulation_swing_price(
                entry_ticks, stop_ticks, tick_size=tick_size
            ),
            "resolution": None,
            "resolution_ts": None,
            "bars_in_trade": None,
            "mfe_price": None,
            "mae_price": None,
        }
        res = res_by_sid.get(sid)
        executed = (
            selected
            and res is not None
            and str(res.get("entry_family")) == family
            and pd.notna(res.get("entry_ts_utc"))
        )
        if executed:
            overlay["resolution"] = str(res["resolution"])
            overlay["resolution_ts"] = pd.Timestamp(res["envelope_ts_utc"])
            if pd.notna(res.get("bars_in_trade")):
                overlay["bars_in_trade"] = int(res["bars_in_trade"])
            if pd.notna(res.get("mfe_ticks")):
                overlay["mfe_price"] = entry_price + sign * float(res["mfe_ticks"]) * tick_size
            if pd.notna(res.get("mae_ticks")):
                overlay["mae_price"] = entry_price - sign * float(res["mae_ticks"]) * tick_size
        out.append(overlay)
    return out


# ── run-scoped trade matching ─────────────────────────────────────────────────


def trade_key(setup_id: str, entry_family: str, entry_ts) -> tuple[str, str, str]:
    """Canonical (setup_id, entry_family, entry_ts UTC iso) match key."""
    ts = pd.Timestamp(entry_ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return (str(setup_id), str(entry_family), ts.isoformat())


def sealed_replay_available(run: dict | None) -> bool:
    """The ONLY condition that unlocks sealed-day replay: a selected saved run
    with sealed validations on the append-only ledger. Pure so the gate is
    regression-testable outside Streamlit."""
    return bool(run and (run.get("sealed_validations") or []))


def run_trade_keys(trades: list[dict]) -> set[tuple[str, str, str]]:
    """Match keys for a saved run's trade list (``trade_stats.trades``)."""
    keys = set()
    for trade in trades or []:
        if trade.get("entry_ts_utc") is None:
            continue
        keys.add(trade_key(trade["setup_id"], trade["entry_family"], trade["entry_ts_utc"]))
    return keys


def scope_overlays(
    overlays: list[dict], keys: set[tuple[str, str, str]] | None
) -> list[dict]:
    """Restrict overlays to a run's admitted trades (None = no scoping)."""
    if keys is None:
        return overlays
    return [
        o for o in overlays if trade_key(o["setup_id"], o["entry_family"], o["entry_ts"]) in keys
    ]


# ── session background bands ──────────────────────────────────────────────────


def session_bands(
    trading_day: str,
    windows: dict[str, tuple[str, str]],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict]:
    """ET session windows placed onto trading day ``trading_day`` (which spans
    prev-day 18:00 ET -> 17:00/18:00 ET), clipped to [start_ts, end_ts].

    Windows crossing midnight (start > end, e.g. engine asia 19:00-02:45, doc
    asia 16:00-01:45) start on the previous calendar day; windows starting at
    or after 18:00 ET sit wholly on the previous calendar day."""
    day = date.fromisoformat(trading_day)
    prev = day - timedelta(days=1)
    out: list[dict] = []
    for name, (start_s, end_s) in windows.items():
        sh, sm = (int(v) for v in start_s.split(":"))
        eh, em = (int(v) for v in end_s.split(":"))
        w_start, w_end = time(sh, sm), time(eh, em)
        if w_start > w_end:  # crosses midnight
            d0, d1 = prev, day
        elif w_start >= time(18, 0):
            d0, d1 = prev, prev
        else:
            d0, d1 = day, day
        t0 = pd.Timestamp(datetime.combine(d0, w_start, tzinfo=_ET)).tz_convert("UTC")
        t1 = pd.Timestamp(datetime.combine(d1, w_end, tzinfo=_ET)).tz_convert("UTC")
        t0 = max(t0, pd.Timestamp(start_ts))
        t1 = min(t1, pd.Timestamp(end_ts))
        if t1 > t0:
            out.append({"name": name, "start": t0, "end": t1})
    return out


# ── "all gaps (recomputed)" audit pass ────────────────────────────────────────


def recompute_day_gaps(bars: pd.DataFrame, timeframe_seconds: int) -> pd.DataFrame:
    """Re-run SC's batch FVG detector over the day's recorded bars at one
    timeframe (detection-completeness audit; drawn visually distinct).

    No prior-day detector tail is available here, so cross-boundary triplets
    at the very start of the day are not re-detected — the logged zones remain
    the ground truth; this is an audit overlay only."""
    from strategy_core.structures.fvg import detect_fvgs_over_bars
    from strategy_core.types import Bar, BarKind, CloseReason

    sub = bars[bars["timeframe_ticks"] == timeframe_seconds].sort_values("open_ts_utc")
    objs = [
        Bar(
            timeframe_ticks=int(r.timeframe_ticks),
            trading_day=date.fromisoformat(str(r.trading_day)),
            bar_index=int(r.bar_index),
            bar_id=str(r.bar_id),
            open_ts_utc=pd.Timestamp(r.open_ts_utc).to_pydatetime(),
            close_ts_utc=pd.Timestamp(r.close_ts_utc).to_pydatetime(),
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
        for r in sub.itertuples(index=False)
    ]
    gaps = detect_fvgs_over_bars(objs, timeframe_seconds=timeframe_seconds)
    return pd.DataFrame(
        {
            "fvg_id": [g.fvg_id for g in gaps],
            "gap_low_ticks": [g.gap_low_ticks for g in gaps],
            "gap_high_ticks": [g.gap_high_ticks for g in gaps],
            "direction": [str(g.direction) for g in gaps],
            "confirmed_ts_utc": [pd.Timestamp(g.confirmed_ts_utc) for g in gaps],
        }
    )


# ── the replay figure ─────────────────────────────────────────────────────────

_BAND_COLORS = {"engine": "rgba(76,120,168,0.07)", "doc": "rgba(230,159,0,0.07)"}
_STAGE_SYMBOLS = {
    "tap": "circle",
    "parent": "diamond",
    "lock": "square",
    "armed": "triangle-right",
    "inversion": "x",
    "entry": "star",
    "resolution": "circle-open",
}


def _bar_high_at(tf_bars: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    """High of the last bar whose close_ts <= ts (marker y anchor)."""
    if not len(tf_bars):
        return None
    closes = tf_bars["close_ts_utc"].values  # datetime64[ns] UTC wall-clock
    target = pd.Timestamp(ts)
    if target.tzinfo is not None:
        target = target.tz_convert("UTC").tz_localize(None)
    idx = int(np.searchsorted(closes, np.datetime64(target), side="right")) - 1
    idx = min(max(idx, 0), len(tf_bars) - 1)
    return float(tf_bars["high_ticks"].iloc[idx]) * TICK_SIZE


# Possible later upgrade: streamlit-lightweight-charts for TradingView-feel
# panning/zooming — the payload builders above are renderer-agnostic on purpose.
def build_replay_figure(
    *,
    day: str,
    bars: pd.DataFrame,
    levels: pd.DataFrame,
    zones: pd.DataFrame,
    overlays: list[dict],
    markers: list[dict],
    timeframe_seconds: int = 60,
    max_bar_index: int | None = None,
    show_engine_bands: bool = True,
    show_doc_bands: bool = False,
    show_levels: bool = True,
    show_zones: bool = True,
    show_stages: bool = True,
    show_trades: bool = True,
    show_dropped: bool = True,
    show_extra_tps: bool = True,
    recomputed_zones: pd.DataFrame | None = None,
    selected_setup_id: str | None = None,
    tick_size: float = TICK_SIZE,
    engine_windows: dict[str, tuple[str, str]] | None = None,
) -> go.Figure:
    """The single replay candlestick figure (all 8 visual-contract elements).

    ``max_bar_index`` truncates to ``bars[0..i]`` of the selected timeframe (the
    scrub slider): every layer is clipped to the truncation cutoff so stages
    appear in order as the slider advances."""
    tf_bars = (
        bars[bars["timeframe_ticks"] == timeframe_seconds]
        .sort_values("open_ts_utc")
        .reset_index(drop=True)
    )
    fig = go.Figure()
    if tf_bars.empty:
        fig.update_layout(title=f"{day} — no bars at {timeframe_seconds}s")
        return fig
    if max_bar_index is not None:
        tf_bars = tf_bars.iloc[: max(1, max_bar_index + 1)]
    start_ts = pd.Timestamp(tf_bars["open_ts_utc"].iloc[0])
    cutoff = pd.Timestamp(tf_bars["close_ts_utc"].iloc[-1])

    # 2) session background bands (engine + doc ET schemes).
    for scheme_key, windows, on in (
        ("engine", engine_windows or ENGINE_SESSION_WINDOWS, show_engine_bands),
        ("doc", DOC_SESSION_WINDOWS, show_doc_bands),
    ):
        if not on:
            continue
        for band in session_bands(day, windows, start_ts, cutoff):
            fig.add_vrect(
                x0=band["start"],
                x1=band["end"],
                fillcolor=_BAND_COLORS[scheme_key],
                line_width=0,
                layer="below",
                annotation_text=f"{band['name']} ({scheme_key})",
                annotation_position="top left" if scheme_key == "engine" else "bottom left",
                annotation_font_size=9,
                annotation_font_color="#8C8C8C",
            )

    # 3) levels, availability-windowed, labeled.
    if show_levels and len(levels):
        series = level_series(levels)
        series = series[pd.to_datetime(series["close_ts_utc"], utc=True) <= cutoff]
        for name, group in series.groupby("name"):
            if group["price"].isna().all():
                continue
            style = LEVEL_STYLE.get(str(name), {"color": "#8C8C8C", "dash": "dot"})
            fig.add_trace(
                go.Scatter(
                    x=group["close_ts_utc"],
                    y=group["price"],
                    mode="lines",
                    line={"color": style["color"], "dash": style["dash"], "width": 1},
                    name=str(name),
                    legendgroup="levels",
                    legendgrouptitle_text="levels",
                    hovertemplate=f"{name}: %{{y:.2f}}<br>%{{x}}<extra></extra>",
                )
            )
            last = group.dropna(subset=["price"]).iloc[-1]
            fig.add_annotation(
                x=last["close_ts_utc"],
                y=last["price"],
                text=str(name),
                showarrow=False,
                xanchor="left",
                font={"size": 9, "color": style["color"]},
            )

    # 4) FVG zones — deduped by fvg_id FIRST (tens of shapes, never per tap).
    if show_zones and len(zones):
        visible = zones[pd.to_datetime(zones["confirmed_ts_utc"], utc=True) <= cutoff]
        hover_x, hover_y, hover_t = [], [], []
        for _, z in visible.iterrows():
            style = zone_style(str(z["role"]), bool(z["selected"]))
            y0 = float(z["gap_low_ticks"]) * tick_size
            y1 = float(z["gap_high_ticks"]) * tick_size
            fig.add_shape(
                type="rect",
                x0=z["confirmed_ts_utc"],
                x1=cutoff,
                y0=y0,
                y1=y1,
                fillcolor=style["fillcolor"],
                line=style["line"],
                layer="below",
            )
            hover_x.append(z["confirmed_ts_utc"])
            hover_y.append((y0 + y1) / 2)
            state = "selected" if bool(z["selected"]) else f"dropped ({z['drop_reason']})"
            hover_t.append(
                f"{z['role']} FVG {z['fvg_id']}<br>{z['direction']} "
                f"[{y0:.2f}, {y1:.2f}] · {state}"
            )
        if hover_x:
            fig.add_trace(
                go.Scatter(
                    x=hover_x,
                    y=hover_y,
                    mode="markers",
                    marker={"size": 5, "color": "#8C8C8C", "opacity": 0.4},
                    hovertext=hover_t,
                    hoverinfo="text",
                    name="FVG zones",
                    legendgroup="zones",
                )
            )
    if recomputed_zones is not None and len(recomputed_zones):
        known = set(zones["fvg_id"]) if len(zones) else set()
        extra = recomputed_zones[
            (pd.to_datetime(recomputed_zones["confirmed_ts_utc"], utc=True) <= cutoff)
            & ~recomputed_zones["fvg_id"].isin(known)
        ]
        style = zone_style("", False, recomputed=True)
        for _, z in extra.iterrows():
            fig.add_shape(
                type="rect",
                x0=z["confirmed_ts_utc"],
                x1=cutoff,
                y0=float(z["gap_low_ticks"]) * tick_size,
                y1=float(z["gap_high_ticks"]) * tick_size,
                fillcolor=style["fillcolor"],
                line=style["line"],
                layer="below",
            )

    # 5) stage markers (+ the manipulation swing under item 5's contract).
    sel_overlay = next(
        (
            o
            for o in overlays
            if o["selected"]
            and o["entry_ts"] <= cutoff
            and (selected_setup_id is None or o["setup_id"] == selected_setup_id)
        ),
        None,
    )
    if show_stages and markers:
        vis = [m for m in markers if pd.Timestamp(m["ts"]) <= cutoff]
        if vis:
            ys = []
            for m in vis:
                high = _bar_high_at(tf_bars, m["ts"])
                ys.append(high * 1.0006 if high is not None else None)
            fig.add_trace(
                go.Scatter(
                    x=[m["ts"] for m in vis],
                    y=ys,
                    mode="markers+text",
                    text=[m["stage"] for m in vis],
                    textposition="top center",
                    textfont={"size": 9},
                    marker={
                        "size": 10,
                        "color": "#333333",
                        "symbol": [
                            _STAGE_SYMBOLS.get(m["stage"], "circle") for m in vis
                        ],
                    },
                    hovertext=[f"{m['stage']}: {m['text']}" for m in vis],
                    hoverinfo="text",
                    name="setup stages",
                    legendgroup="stages",
                )
            )
        if sel_overlay is not None:
            armed_ts = next(
                (pd.Timestamp(m["ts"]) for m in markers if m["stage"] == "armed"), None
            )
            x0 = armed_ts if armed_ts is not None else sel_overlay["entry_ts"]
            fig.add_trace(
                go.Scatter(
                    x=[x0, min(sel_overlay["entry_ts"], cutoff)],
                    y=[sel_overlay["swing_price"]] * 2,
                    mode="lines",
                    line={"color": "#333333", "dash": "dashdot", "width": 1},
                    name="manipulation swing (stop -/+ 1t)",
                    legendgroup="stages",
                    hoverinfo="name+y",
                )
            )

    # 6) trades: entry arrow / SL / TP 1R+1.5R/2R / outcome / MFE-MAE.
    if show_trades:
        for o in overlays:
            if o["entry_ts"] > cutoff:
                continue
            if not o["selected"]:
                if show_dropped:
                    fig.add_trace(
                        go.Scatter(
                            x=[o["entry_ts"]],
                            y=[o["entry_price"]],
                            mode="markers",
                            marker={
                                "symbol": "x-thin",
                                "size": 9,
                                "color": "#8C8C8C",
                                "line": {"width": 1.5, "color": "#8C8C8C"},
                            },
                            opacity=0.55,
                            hovertext=(
                                f"DROPPED candidate {o['setup_id']} · {o['entry_family']}"
                                f"<br>reason: {o['drop_reason']}"
                            ),
                            hoverinfo="text",
                            name="dropped candidates",
                            legendgroup="trades",
                            showlegend=False,
                        )
                    )
                continue
            is_long = o["direction"] == "LONG"
            end_x = min(o["resolution_ts"] or cutoff, cutoff)
            fig.add_trace(
                go.Scatter(
                    x=[o["entry_ts"]],
                    y=[o["entry_price"]],
                    mode="markers",
                    marker={
                        "symbol": "triangle-up" if is_long else "triangle-down",
                        "size": 13,
                        "color": "#2CA02C" if is_long else "#D62728",
                    },
                    hovertext=(
                        f"{o['direction']} entry {o['setup_id']} · {o['entry_family']}"
                        f"<br>entry {o['entry_price']:.2f} · risk {o['risk_ticks']:.0f}t"
                    ),
                    hoverinfo="text",
                    name="entries",
                    legendgroup="trades",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[o["entry_ts"], end_x],
                    y=[o["stop_price"]] * 2,
                    mode="lines",
                    line={"color": "#D62728", "dash": "solid", "width": 1.5},
                    name="SL",
                    legendgroup="trades",
                    showlegend=False,
                    hoverinfo="name+y",
                )
            )
            for r_mult, price in o["tp_prices"].items():
                if r_mult != 1.0 and not show_extra_tps:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[o["entry_ts"], end_x],
                        y=[price] * 2,
                        mode="lines",
                        line={
                            "color": "#2CA02C",
                            "dash": "solid" if r_mult == 1.0 else "dash",
                            "width": 1.5 if r_mult == 1.0 else 1,
                        },
                        name=f"TP {r_mult:g}R",
                        legendgroup="trades",
                        showlegend=False,
                        hoverinfo="name+y",
                    )
                )
            for label, price in (("MFE", o["mfe_price"]), ("MAE", o["mae_price"])):
                if price is None:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[o["entry_ts"], end_x],
                        y=[price] * 2,
                        mode="lines",
                        line={"color": "#8C8C8C", "dash": "dot", "width": 1},
                        name=label,
                        legendgroup="trades",
                        showlegend=False,
                        hoverinfo="name+y",
                    )
                )
            if o["resolution"] is not None and o["resolution_ts"] <= cutoff:
                fig.add_annotation(
                    x=o["resolution_ts"],
                    y=o["entry_price"],
                    text=(
                        f"{o['resolution']}"
                        + (f" ({o['bars_in_trade']} bars)" if o["bars_in_trade"] else "")
                    ),
                    showarrow=True,
                    arrowhead=2,
                    font={"size": 10},
                    bgcolor="rgba(255,255,255,0.75)",
                )

    # 1) the candlesticks themselves (top layer).
    fig.add_trace(
        go.Candlestick(
            x=tf_bars["open_ts_utc"],
            open=tf_bars["open_ticks"] * tick_size,
            high=tf_bars["high_ticks"] * tick_size,
            low=tf_bars["low_ticks"] * tick_size,
            close=tf_bars["close_ticks"] * tick_size,
            name="price",
            increasing_line_color="#2E9990",
            decreasing_line_color="#D62728",
            showlegend=False,
        )
    )
    tf_label = next(
        (k for k, v in TIMEFRAME_OPTIONS.items() if v == timeframe_seconds),
        f"{timeframe_seconds}s",
    )
    fig.update_layout(
        title=f"{day} · {tf_label} · bars 0..{len(tf_bars) - 1}",
        height=640,
        xaxis_rangeslider_visible=False,
        yaxis_title="price",
        margin={"l": 40, "r": 90, "t": 48, "b": 24},
        legend={"orientation": "h", "y": -0.06},
        hovermode="closest",
    )
    return fig


# ── experiment result figures ─────────────────────────────────────────────────


def build_equity_figure(trade_stats: dict, unit: str = "usd") -> go.Figure:
    """Equity curve + running close-to-close drawdown, one unit ($ or R)."""
    curve = (trade_stats.get("equity") or {}).get(unit) or {}
    equity = curve.get("equity") or []
    ts = curve.get("timestamps") or list(range(len(equity)))
    eq = np.asarray([v if v is not None else np.nan for v in equity], dtype=float)
    peak = np.maximum.accumulate(np.nan_to_num(eq, nan=0.0)) if len(eq) else eq
    dd = peak - np.nan_to_num(eq, nan=0.0)
    label = "$" if unit == "usd" else "R"
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04
    )
    fig.add_trace(
        go.Scatter(x=ts, y=eq, mode="lines", name=f"equity ({label})",
                   line={"color": "#4C78A8", "width": 2}),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=ts, y=-dd, mode="lines", name=f"drawdown ({label})",
                   line={"color": "#D62728", "width": 1.5}, fill="tozeroy",
                   fillcolor="rgba(214,39,40,0.15)"),
        row=2, col=1,
    )
    fig.update_layout(height=440, margin={"l": 40, "r": 20, "t": 30, "b": 24},
                      legend={"orientation": "h", "y": 1.08})
    return fig


def build_r_histogram_figure(r_histogram: dict) -> go.Figure:
    bins = r_histogram.get("bins") or []
    counts = r_histogram.get("counts") or []
    fig = go.Figure(
        go.Bar(x=bins, y=counts, marker_color="#4C78A8", name="trades")
    )
    fig.update_layout(
        height=320, xaxis_title="net R bin", yaxis_title="trades",
        margin={"l": 40, "r": 20, "t": 30, "b": 60}, bargap=0.15,
    )
    return fig


def build_calibration_figure(model_section: dict) -> go.Figure:
    rows = model_section.get("calibration") or []
    mean_p = [r["mean_p"] for r in rows]
    actual = [r["actual"] for r in rows]
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    lo = min([0.0, *mean_p, *actual])
    hi = max([1.0, *mean_p, *actual])
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect",
                   line={"color": "#8C8C8C", "dash": "dot", "width": 1})
    )
    fig.add_trace(
        go.Scatter(
            x=mean_p, y=actual, mode="lines+markers", name="model",
            marker={"size": 9, "color": "#4C78A8"},
            line={"color": "#4C78A8", "width": 2},
            hovertext=[f"n={n}" for n in ns], hoverinfo="text+x+y",
        )
    )
    fig.update_layout(
        height=340, xaxis_title="mean predicted p(win)", yaxis_title="actual win rate",
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    return fig


def build_coverage_figure(model_section: dict) -> go.Figure:
    rows = model_section.get("coverage") or []
    thr = [f"{r['thr']:g}" for r in rows]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("coverage", "mean net R"))
    fig.add_trace(
        go.Bar(x=thr, y=[r["coverage"] for r in rows], marker_color="#4C78A8",
               hovertext=[f"n={r['n']}" for r in rows], name="coverage"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=thr,
            y=[r["mean_net_r"] for r in rows],
            marker_color="#2E9990",
            hovertext=[
                f"n={r['n']} · win {r['win_rate']:.2f}" if r["win_rate"] is not None
                else f"n={r['n']}"
                for r in rows
            ],
            name="mean net R",
        ),
        row=2, col=1,
    )
    fig.update_layout(height=420, showlegend=False,
                      margin={"l": 40, "r": 20, "t": 40, "b": 30},
                      xaxis2_title="p(win) threshold")
    return fig


# ── config flatten/diff (compare + run-configuration panels) ──────────────────


def flatten_config(config: dict, prefix: str = "") -> dict[str, str]:
    """Dotted-path flattening of a resolved config dict; scalars stringified,
    lists/tuples JSON-dumped, None rendered as ``(none)``."""
    out: dict[str, str] = {}
    for key, value in (config or {}).items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten_config(value, prefix=f"{path}."))
        elif value is None:
            out[path] = "(none)"
        elif isinstance(value, list | tuple):
            out[path] = json.dumps(list(value))
        else:
            out[path] = str(value)
    return out


def config_diff_frame(
    config_a: dict, config_b: dict, label_a: str = "run A", label_b: str = "run B"
) -> pd.DataFrame:
    """Field-aligned comparison of two resolved configs with a differs flag."""
    flat_a, flat_b = flatten_config(config_a), flatten_config(config_b)
    fields = sorted(set(flat_a) | set(flat_b))
    return pd.DataFrame(
        {
            "field": fields,
            label_a: [flat_a.get(f, "(absent)") for f in fields],
            label_b: [flat_b.get(f, "(absent)") for f in fields],
            "differs": [flat_a.get(f, "(absent)") != flat_b.get(f, "(absent)") for f in fields],
        }
    )
