"""Performance statistics over validated IFVG v2 executions only.

Candidate labels and eligible decisions are counterfactual/reconciliation
streams and are rejected.  Resolution time is exact, the entry candle is
excluded by contract, and dataset-exhausted open trades carry no realized P&L
and therefore cannot enter this function.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contracts import (
    IFVG_DATASET_SCHEMA_VERSION,
    RecordTable,
    validate_primary_keys,
    validate_table_identity,
)

__all__ = ["wilson_ci", "mean_ci", "compute_trade_stats", "RESOLUTION_TIME_NOTE"]

_Z95 = 1.959963984540054
DOLLARS_PER_POINT = 20.0
#: NQ trading-day span in minutes (18:00 -> 17:00 ET) for the exposure ratio.
SESSION_MINUTES_PER_DAY = 1380

RESOLUTION_TIME_NOTE = (
    "Trade ordering uses the exact Strategy-Core resolution cursor/timestamp; "
    "the confirmation/entry candle is excluded and the first forward 1m bar is 1."
)

_ASSUMPTIONS = (
    "1-contract sizing, $20/NQ point, round-turn cost netted into every PnL figure.",
    "Rows are validated IFVG v2 resolved executed trades; candidates and decisions "
    "are rejected.",
    "Resolution timestamps and one-based bars-after-entry come from Strategy-Core.",
    "Same-bar stop/target ambiguity is stop-first.",
    "Confidence intervals use a 10,000-sample trading-day cluster bootstrap.",
    "Daily Sharpe/Sortino annualize over TRADED days only (no-trade days contribute no "
    "0-PnL observation; sparse configs read higher in magnitude).",
    "One-setup/one-trade replay forbids overlapping executed trades.",
    "Intrabar drawdown folds the GROSS (cost-free) MAE excursion onto the cost-netted "
    "curve; a drawdown starting at the initial zero peak counts depth but no duration.",
)

_R_HIST_EDGES = (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0)


def wilson_ci(k: int, n: int) -> tuple[float, float] | None:
    """Wilson 95% score interval for a binomial rate; ``None`` when n == 0."""
    if n <= 0:
        return None
    p = k / n
    z2 = _Z95 * _Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = _Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (center - half, center + half)


def mean_ci(values: pd.Series) -> tuple[float, float] | None:
    """Normal-approximation 95% CI on the mean; ``None`` when n < 2."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    m = float(clean.mean())
    half = _Z95 * float(clean.std(ddof=1)) / math.sqrt(len(clean))
    return (m - half, m + half)


def _f(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _sided(ordered: pd.DataFrame, fn) -> dict:
    return {
        "all": fn(ordered),
        "long": fn(ordered[ordered["direction"] == "LONG"]),
        "short": fn(ordered[ordered["direction"] == "SHORT"]),
    }


def _counts(sub: pd.DataFrame) -> dict:
    return {
        "trades": int(len(sub)),
        "winners": int((sub["_label"] == "win").sum()),
        "losers": int((sub["_label"] == "loss").sum()),
    }


def _win_rate(sub: pd.DataFrame) -> dict:
    n = len(sub)
    k = int((sub["_label"] == "win").sum())
    return {
        "n": n,
        "win_rate": _f(k / n) if n else None,
    }


def _pnl_block(sub: pd.DataFrame, col: str) -> dict:
    """One unit's PnL block ($ via ``_pnl_usd`` or R via ``_net_r``)."""
    series = pd.to_numeric(sub[col], errors="coerce").dropna()
    if series.empty:
        return {"n": 0}
    pos, neg = series[series > 0], series[series < 0]
    gross_profit = float(pos.sum())
    gross_loss = float(-neg.sum())
    avg_win = float(pos.mean()) if len(pos) else None
    avg_loss = float(-neg.mean()) if len(neg) else None
    win_rate = len(pos) / len(series)
    payoff = (avg_win / avg_loss) if avg_win is not None and avg_loss else None
    return {
        "n": int(len(series)),
        "net": float(series.sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_per_trade": float(series.mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": float(pos.max()) if len(pos) else None,
        "largest_loss": float(neg.min()) if len(neg) else None,
        "profit_factor": _f(gross_profit / gross_loss) if gross_loss else None,
        "expectancy_per_trade": float(series.mean()),
        "avg_win_avg_loss_ratio": _f(payoff) if payoff is not None else None,
        "payoff_adjusted_expectancy": (
            _f(win_rate * avg_win - (1 - win_rate) * avg_loss)
            if avg_win is not None and avg_loss is not None
            else float(series.mean())
        ),
    }


def _time_block(sub: pd.DataFrame) -> dict:
    minutes = pd.to_numeric(sub["_minutes_in_trade"], errors="coerce").dropna()
    win_min = sub.loc[sub["_label"] == "win", "_minutes_in_trade"].dropna()
    loss_min = sub.loc[sub["_label"] == "loss", "_minutes_in_trade"].dropna()
    return {
        "avg_minutes_in_trade": _f(minutes.mean()) if len(minutes) else None,
        "avg_minutes_winners": _f(win_min.mean()) if len(win_min) else None,
        "avg_minutes_losers": _f(loss_min.mean()) if len(loss_min) else None,
    }


def _sharpe(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    sd = float(clean.std(ddof=1))
    return _f(float(clean.mean()) / sd) if sd > 0 else None


def _sortino(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if len(clean) < 2:
        return None
    downside = float(np.sqrt(np.mean(np.square(np.minimum(clean.to_numpy(), 0.0)))))
    return _f(float(clean.mean()) / downside) if downside > 0 else None


def _drawdown(equity: np.ndarray, mae: np.ndarray, res_ts: pd.Series) -> dict:
    """Close-to-close DD episodes + intrabar (MAE-folded) max DD over one curve.

    Episode duration = peak resolution ts -> recovery resolution ts; the last
    episode is flagged ongoing when equity never recovers its peak.
    """
    peak, peak_ts = 0.0, None
    episodes: list[dict] = []
    depth, started = 0.0, None
    max_intrabar = 0.0
    ts_list = list(res_ts)
    for i, eq in enumerate(equity):
        prev_eq = equity[i - 1] if i else 0.0
        low = prev_eq - max(float(mae[i]), 0.0)
        max_intrabar = max(max_intrabar, peak - low, peak - eq)
        if eq >= peak:
            if started is not None:
                dur = None
                if peak_ts is not None and ts_list[i] is not None:
                    dur = (ts_list[i] - peak_ts).total_seconds() / 60.0
                episodes.append({"depth": depth, "duration_minutes": dur, "recovered": True})
                started, depth = None, 0.0
            peak, peak_ts = eq, ts_list[i]
        else:
            if started is None:
                started = i
            depth = max(depth, peak - eq)
    if started is not None:
        dur = None
        if peak_ts is not None and ts_list[-1] is not None:
            dur = (ts_list[-1] - peak_ts).total_seconds() / 60.0
        episodes.append({"depth": depth, "duration_minutes": dur, "recovered": False})
    depths = [e["depth"] for e in episodes]
    durations = [e["duration_minutes"] for e in episodes if e["duration_minutes"] is not None]
    return {
        "max_drawdown_close": max(depths) if depths else 0.0,
        "max_drawdown_intrabar": max_intrabar,
        "avg_drawdown_depth": _f(float(np.mean(depths))) if depths else None,
        "avg_drawdown_duration_minutes": _f(float(np.mean(durations))) if durations else None,
        "max_drawdown_duration_minutes": _f(float(np.max(durations))) if durations else None,
        "n_episodes": len(episodes),
        "ongoing_episode": bool(episodes and not episodes[-1]["recovered"]),
        "episodes": episodes,
    }


def _streaks(labels: pd.Series) -> dict:
    """Maximum consecutive resolved wins/losses."""
    best = {"win": 0, "loss": 0}
    current_label, run = None, 0
    for label in labels:
        if label not in ("win", "loss"):
            continue
        run = run + 1 if label == current_label else 1
        current_label = label
        best[label] = max(best[label], run)
    return {"max_consecutive_wins": best["win"], "max_consecutive_losses": best["loss"]}


def _r_histogram(net_r: pd.Series) -> dict:
    clean = pd.to_numeric(net_r, errors="coerce").dropna().to_numpy()
    edges = (-np.inf, *_R_HIST_EDGES, np.inf)
    counts, _ = np.histogram(clean, bins=np.array(edges))
    labels = []
    for i in range(len(edges) - 1):
        lo = "-inf" if np.isinf(edges[i]) else f"{edges[i]:g}"
        hi = "+inf" if np.isinf(edges[i + 1]) else f"{edges[i + 1]:g}"
        labels.append(f"[{lo}, {hi})")
    return {"bins": labels, "counts": [int(c) for c in counts]}


def _mfe_mae(ordered: pd.DataFrame) -> dict:
    mfe = pd.to_numeric(ordered["_mfe_r"], errors="coerce")
    mae = pd.to_numeric(ordered["_mae_r"], errors="coerce")
    losers = ordered["_label"] == "loss"
    winners = ordered["_label"] == "win"

    def _dist(series: pd.Series) -> dict:
        clean = series.dropna()
        if clean.empty:
            return {"n": 0}
        return {
            "n": int(len(clean)),
            "mean": _f(clean.mean()),
            "median": _f(clean.median()),
            "p90": _f(clean.quantile(0.9)),
            "max": _f(clean.max()),
        }

    return {
        "mfe_r": _dist(mfe),
        "mae_r": _dist(mae),
        "avg_mfe_r_of_losers": _f(mfe[losers].dropna().mean()) if losers.any() else None,
        "avg_mae_r_of_winners": _f(mae[winners].dropna().mean()) if winners.any() else None,
    }


_TRADE_LIST_COLUMNS = (
    "setup_id",
    "candidate_id",
    "decision_id",
    "trade_id",
    "trading_day",
    "entry_family",
    "direction",
    "session_engine",
    "session_doc",
    "session_custom",
    "entry_ticks",
    "stop_ticks",
    "risk_ticks",
)


def _validate_and_normalize_executed_trades(
    work: pd.DataFrame,
    *,
    tick_size: float,
) -> pd.DataFrame:
    """Fail closed before any performance arithmetic."""
    # Identity uniqueness is the first diagnostic even when a caller also
    # omitted other required execution fields.
    validate_primary_keys(RecordTable.EXECUTED_TRADE, work)
    required_identity = {
        "record_table",
        "record_schema_version",
        "capture_schema_version",
        "dataset_schema_version",
        "trade_schema_version",
        "setup_id",
        "candidate_id",
        "trade_id",
        "decision_id",
        "status",
        "resolution",
        "trading_day",
        "direction",
        "entry_ts_utc",
        "resolution_ts_utc",
        "entry_cursor",
        "resolution_cursor",
        "entry_ticks",
        "stop_ticks",
        "target_ticks",
        "risk_ticks",
        "bars_after_entry_to_resolution",
        "realized_ticks",
        "mfe_ticks",
        "mae_ticks",
    }
    missing = sorted(required_identity - set(work.columns))
    if missing:
        raise ValueError(
            "performance input must be a validated executed_trade table; "
            f"missing {missing}"
        )
    validate_table_identity(RecordTable.EXECUTED_TRADE, work)
    if work["setup_id"].isna().any() or work["candidate_id"].isna().any():
        raise ValueError("executed_trade setup_id/candidate_id must be non-null")
    if work["decision_id"].isna().any() or (
        work["decision_id"].astype(str) == ""
    ).any():
        raise ValueError("executed_trade decision_id must be non-null")
    if not (work["status"] == "resolved").all():
        raise ValueError(
            "only resolved executed_trade rows may enter performance statistics"
        )
    if not work["resolution"].isin(("target", "stop")).all():
        raise ValueError("resolved executed_trade resolution must be target or stop")

    normalized = work.copy()
    normalized["direction"] = normalized["direction"].astype(str).str.upper()
    if not normalized["direction"].isin(("LONG", "SHORT")).all():
        raise ValueError("executed_trade direction must be LONG or SHORT")
    normalized["_label"] = normalized["resolution"].map(
        {"target": "win", "stop": "loss"}
    )
    normalized["_bars_to_res"] = pd.to_numeric(
        normalized["bars_after_entry_to_resolution"], errors="raise"
    )
    if (normalized["_bars_to_res"] < 1).any():
        raise ValueError("executed_trade cannot resolve on the entry bar")

    risk_ticks = pd.to_numeric(normalized["risk_ticks"], errors="raise")
    normalized["_risk_points"] = risk_ticks * tick_size
    if (normalized["_risk_points"] <= 0).any():
        raise ValueError("executed_trade risk must be positive")

    entry_ticks = pd.to_numeric(normalized["entry_ticks"], errors="raise")
    stop_ticks = pd.to_numeric(normalized["stop_ticks"], errors="raise")
    target_ticks = pd.to_numeric(normalized["target_ticks"], errors="raise")
    expected_risk = (entry_ticks - stop_ticks).abs()
    if not (risk_ticks == expected_risk).all():
        raise ValueError("executed_trade risk_ticks disagrees with entry/stop")
    is_long = normalized["direction"] == "LONG"
    wrong_side = (
        is_long & ~((stop_ticks < entry_ticks) & (target_ticks > entry_ticks))
    ) | (
        ~is_long & ~((stop_ticks > entry_ticks) & (target_ticks < entry_ticks))
    )
    if wrong_side.any():
        raise ValueError("executed_trade stop/target is on the wrong side")

    realized_ticks = pd.to_numeric(normalized["realized_ticks"], errors="raise")
    expected_realized = pd.Series(
        np.where(
            normalized["resolution"] == "stop",
            -risk_ticks,
            (target_ticks - entry_ticks).abs(),
        ),
        index=normalized.index,
    )
    if not (realized_ticks == expected_realized).all():
        raise ValueError("executed_trade realized_ticks disagrees with its barrier")
    normalized["_realized_pts"] = realized_ticks * tick_size
    normalized["_gross_r"] = normalized["_realized_pts"] / normalized["_risk_points"]
    normalized["_mfe_r"] = (
        pd.to_numeric(normalized["mfe_ticks"], errors="raise")
        * tick_size
        / normalized["_risk_points"]
    )
    normalized["_mae_r"] = (
        pd.to_numeric(normalized["mae_ticks"], errors="raise")
        * tick_size
        / normalized["_risk_points"]
    )
    normalized["_tp_r"] = (
        (target_ticks - entry_ticks).abs() * tick_size
        / normalized["_risk_points"]
    )

    entry_ts = pd.to_datetime(normalized["entry_ts_utc"], utc=True, errors="raise")
    resolution_ts = pd.to_datetime(
        normalized["resolution_ts_utc"], utc=True, errors="raise"
    )
    if resolution_ts.isna().any() or not (resolution_ts > entry_ts).all():
        raise ValueError("executed_trade resolution must be strictly after entry")
    if normalized["resolution_cursor"].isna().any() or (
        normalized["resolution_cursor"].astype(str) == ""
    ).any():
        raise ValueError("executed_trade resolution_cursor is required")
    if (
        normalized["resolution_cursor"].astype(str)
        == normalized["entry_cursor"].astype(str)
    ).any():
        raise ValueError("executed_trade cannot resolve at its entry cursor")

    # The reducer has one execution slot.  A later trade may start only after
    # the prior trade's exact resolution event.
    chronological = pd.DataFrame(
        {"entry": entry_ts, "resolution": resolution_ts}
    ).sort_values("entry", kind="mergesort")
    if len(chronological) > 1:
        previous_resolution = chronological["resolution"].shift(1)
        overlaps = chronological["entry"] <= previous_resolution
        if overlaps.fillna(False).any():
            raise ValueError("executed_trade intervals overlap")
    return normalized


def _cluster_bootstrap(
    ordered: pd.DataFrame,
    *,
    evaluation_config_hash: str,
    samples: int = 10_000,
) -> dict:
    days = sorted(ordered["trading_day"].astype(str).unique())
    if len(days) < 2:
        return {
            "available": False,
            "reason": "fewer_than_two_trading_days",
            "samples": samples,
            "confidence_level": 0.95,
        }
    seed = int(evaluation_config_hash[:16], 16)
    rng = np.random.default_rng(seed)
    groups = [
        ordered[ordered["trading_day"].astype(str) == day]
        for day in days
    ]
    cluster_n = np.array([len(group) for group in groups], dtype=float)
    cluster_wins = np.array(
        [int((group["_label"] == "win").sum()) for group in groups],
        dtype=float,
    )
    cluster_net_r = np.array(
        [float(group["_net_r"].sum()) for group in groups],
        dtype=float,
    )
    cluster_pnl = np.array(
        [float(group["_pnl_usd"].sum()) for group in groups],
        dtype=float,
    )
    sampled_indices = rng.integers(
        0, len(days), size=(samples, len(days))
    )
    sampled_n = cluster_n[sampled_indices].sum(axis=1)
    win_rates = cluster_wins[sampled_indices].sum(axis=1) / sampled_n
    mean_net_r = cluster_net_r[sampled_indices].sum(axis=1) / sampled_n
    mean_pnl_usd = cluster_pnl[sampled_indices].sum(axis=1) / sampled_n

    def interval(values: np.ndarray) -> list[float]:
        return [
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        ]

    return {
        "available": True,
        "samples": samples,
        "confidence_level": 0.95,
        "seed_source": "evaluation_config_hash",
        "n_trading_days": len(days),
        "win_rate": interval(win_rates),
        "mean_net_r": interval(mean_net_r),
        "mean_pnl_usd": interval(mean_pnl_usd),
    }


def _trade_list(ordered: pd.DataFrame) -> list[dict]:
    # NOTE: iterrows (not itertuples) — the working columns start with "_" and
    # itertuples silently renames those to positional fields.
    records = []
    for _, row in ordered.iterrows():
        rec = {c: row[c] for c in _TRADE_LIST_COLUMNS if c in ordered.columns}
        rec.update(
            {
                "entry_ts_utc": _iso(row.get("entry_ts_utc")),
                "resolution_ts_utc": _iso(row["_res_ts"]),
                "label": row["_label"],
                "tp_r_multiple": _f(row["_tp_r"]),
                "risk_points": _f(row["_risk_points"]),
                "realized_pts": _f(row["_realized_pts"]),
                "net_r": _f(row["_net_r"]),
                "pnl_usd": _f(row["_pnl_usd"]),
                "bars_to_res": _bars(row["_bars_to_res"]),
                "minutes_in_trade": _f(row["_minutes_in_trade"]),
                "mfe_r": _f(row["_mfe_r"]),
                "mae_r": _f(row["_mae_r"]),
            }
        )
        records.append(_scrub(rec))
    return records


def _iso(ts) -> str | None:
    if ts is None or (isinstance(ts, float) and math.isnan(ts)) or pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


def _bars(value) -> int | None:
    try:
        return int(value) if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def _scrub(rec: dict) -> dict:
    out = {}
    for key, value in rec.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        out[key] = None if (value is not None and pd.api.types.is_scalar(value)
                            and pd.isna(value)) else value
    return out


def compute_trade_stats(
    work: pd.DataFrame,
    *,
    cost_points: float,
    evaluation_config_hash: str,
    tick_size: float = 0.25,
) -> dict:
    """Performance metrics over validated resolved IFVG v2 executions only."""
    if len(evaluation_config_hash) != 64:
        raise ValueError("evaluation_config_hash must be a SHA-256 hex string")
    try:
        int(evaluation_config_hash, 16)
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation_config_hash must be a SHA-256 hex string") from exc
    ordered = _validate_and_normalize_executed_trades(
        work, tick_size=tick_size
    )
    if ordered.empty:
        return {
            "schema_version": IFVG_DATASET_SCHEMA_VERSION,
            "record_table": RecordTable.EXECUTED_TRADE.value,
            "n": 0,
            "cluster_bootstrap_ci95": {
                "available": False,
                "reason": "fewer_than_two_trading_days",
                "samples": 10_000,
                "confidence_level": 0.95,
            },
            "assumptions": list(_ASSUMPTIONS),
        }
    ordered["_pnl_usd"] = (
        pd.to_numeric(ordered["_realized_pts"], errors="coerce") - cost_points
    ) * DOLLARS_PER_POINT
    ordered["_net_r"] = (
        pd.to_numeric(ordered["_realized_pts"], errors="coerce") - cost_points
    ) / pd.to_numeric(ordered["_risk_points"], errors="raise")
    entry = pd.to_datetime(ordered["entry_ts_utc"], utc=True)
    res = pd.to_datetime(ordered["resolution_ts_utc"], utc=True, errors="raise")
    ordered["_res_ts"] = res
    ordered["_minutes_in_trade"] = (res - entry).dt.total_seconds() / 60.0
    ordered = ordered.sort_values("_res_ts", kind="mergesort").reset_index(drop=True)

    daily = ordered.groupby("trading_day")["_pnl_usd"].sum()
    n_days = int(ordered["trading_day"].nunique())
    total_minutes = float(ordered["_minutes_in_trade"].sum())

    curves = {}
    for unit, col, mae_scale in (
        ("usd", "_pnl_usd", DOLLARS_PER_POINT),
        ("r", "_net_r", None),
    ):
        values = pd.to_numeric(ordered[col], errors="coerce").fillna(0.0).to_numpy()
        equity = np.cumsum(values)
        mae_r = pd.to_numeric(ordered["_mae_r"], errors="coerce").fillna(0.0).to_numpy()
        if mae_scale is None:
            mae = mae_r
        else:
            risk = pd.to_numeric(ordered["_risk_points"], errors="coerce").fillna(0.0).to_numpy()
            mae = mae_r * risk * mae_scale
        dd = _drawdown(equity, mae, ordered["_res_ts"])
        net = float(equity[-1]) if len(equity) else 0.0
        curves[unit] = {
            "equity": [_f(v) for v in equity],
            "timestamps": [_iso(t) for t in ordered["_res_ts"]],
            "net": net,
            "drawdown": dd,
            "romad": _f(net / dd["max_drawdown_close"]) if dd["max_drawdown_close"] else None,
        }

    return {
        "schema_version": IFVG_DATASET_SCHEMA_VERSION,
        "record_table": RecordTable.EXECUTED_TRADE.value,
        "evaluation_config_hash": evaluation_config_hash,
        "n": int(len(ordered)),
        "counts": _sided(ordered, _counts),
        "win_rate": _sided(ordered, _win_rate),
        "pnl_usd": _sided(ordered, lambda sub: _pnl_block(sub, "_pnl_usd")),
        "pnl_r": _sided(ordered, lambda sub: _pnl_block(sub, "_net_r")),
        "cluster_bootstrap_ci95": _cluster_bootstrap(
            ordered,
            evaluation_config_hash=evaluation_config_hash,
        ),
        "time": {
            **_sided(ordered, _time_block),
            "trades_per_day": _f(len(ordered) / n_days) if n_days else None,
            "n_traded_days": n_days,
            "exposure_fraction": (
                _f(total_minutes / (n_days * SESSION_MINUTES_PER_DAY)) if n_days else None
            ),
        },
        "risk_adjusted": {
            "sharpe_per_trade_r": _sharpe(ordered["_net_r"]),
            "sortino_per_trade_r": _sortino(ordered["_net_r"]),
            "sharpe_daily_annualized": (
                _f(s * math.sqrt(252)) if (s := _sharpe(daily)) is not None else None
            ),
            "sortino_daily_annualized": (
                _f(s * math.sqrt(252)) if (s := _sortino(daily)) is not None else None
            ),
            "n_days": n_days,
        },
        "equity": curves,
        "streaks": _streaks(ordered["_label"]),
        "r_histogram": _r_histogram(ordered["_net_r"]),
        "mfe_mae": _mfe_mae(ordered),
        "trades": _trade_list(ordered),
        "assumptions": list(_ASSUMPTIONS),
    }
