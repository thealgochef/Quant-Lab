"""Strategy gate inputs over executed trades (§3.4; brief §9.7).

Reuses the verified ``compute_trade_stats`` core (executions-only validation,
bootstrap CI, drawdown) and derives the gate-specific inputs the plan adds:
time-under-water, chronological-thirds sign consistency, session stability,
and day/setup concentration shares — all from the SAME normalized executed
frame (``_validate_and_normalize_executed_trades``), so realized R is the
honest labeler value, never a resolution-map approximation.
"""

from __future__ import annotations

import pandas as pd

from ..contracts import RecordTable
from ..trade_stats import _validate_and_normalize_executed_trades, compute_trade_stats
from .charter import PlannedVsRealizedEdge
from .identities import FrozenContract, ImmutableMap

__all__ = ["StrategyMetrics", "compute_strategy_metrics", "per_trade_net_r"]


def research_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Keep capture history intact, but exclude explicitly marked warmup rows.

    Projected execution tables without this column already define their cohort.
    An ambiguous flag must not silently admit a warmup observation.
    """
    if "is_warmup" not in trades:
        return trades
    flags = trades["is_warmup"]
    if flags.isna().any() or not flags.isin([True, False]).all():
        raise ValueError("executed-trade warmup flags must be non-null booleans")
    return trades.loc[~flags.astype(bool)].copy()


def per_trade_net_r(
    trades: pd.DataFrame, *, cost_points: float, tick_size: float = 0.25
) -> pd.Series:
    """The per-trade NET R vector (``(realized − cost) / risk``) of the
    validated, normalized executed-trade table, indexed by ``trade_id`` —
    exactly the vector ``compute_strategy_metrics`` averages into
    ``net_expectancy_r`` (R6.1-FIX §3.5: the raw accounting basis of the
    regime concentration facts)."""

    ordered = _validate_and_normalize_executed_trades(research_trades(trades), tick_size=tick_size)
    if ordered.empty:
        return pd.Series(dtype=float, name="net_r")
    realized = pd.to_numeric(ordered["_realized_pts"], errors="coerce")
    risk = pd.to_numeric(ordered["_risk_points"], errors="raise")
    net = ((realized - float(cost_points)) / risk).astype(float)
    net.index = ordered["trade_id"].astype(str).to_numpy()
    net.name = "net_r"
    return net


class StrategyMetrics(FrozenContract):
    executed_trades: int
    independent_days: int
    gross_expectancy_r: float | None
    net_expectancy_r: float | None
    profit_factor: float | None
    max_drawdown_r: float | None
    time_under_water_days: int | None
    time_block_sign_consistency: float | None
    session_stability_score: float | None
    top_day_pnl_share: float | None
    top_setup_pnl_share: float | None
    net_expectancy_bootstrap_ci95: tuple[float, float] | None
    planned_vs_realized: PlannedVsRealizedEdge | None
    trade_stats: ImmutableMap[str, object]


def _bootstrap_ci(stats: dict) -> tuple[float, float] | None:
    """The trading-day cluster-bootstrap 95% CI on mean net R, when available."""

    block = stats.get("cluster_bootstrap_ci95") or {}
    if not block.get("available"):
        return None
    interval = block.get("mean_net_r")
    if not interval or len(interval) != 2:
        return None
    return (float(interval[0]), float(interval[1]))


def _empty(stats: dict) -> StrategyMetrics:
    return StrategyMetrics(
        executed_trades=0,
        independent_days=0,
        gross_expectancy_r=None,
        net_expectancy_r=None,
        profit_factor=None,
        max_drawdown_r=None,
        time_under_water_days=None,
        time_block_sign_consistency=None,
        session_stability_score=None,
        top_day_pnl_share=None,
        top_setup_pnl_share=None,
        net_expectancy_bootstrap_ci95=None,
        planned_vs_realized=None,
        trade_stats=dict(stats),
    )


def compute_strategy_metrics(
    tables,
    *,
    cost_points: float,
    evaluation_config_hash: str,
    tick_size: float = 0.25,
    tp_r_multiple: float = 1.0,
) -> StrategyMetrics:
    trades = research_trades(tables.get(RecordTable.EXECUTED_TRADE, pd.DataFrame()))
    stats = compute_trade_stats(
        trades,
        cost_points=cost_points,
        evaluation_config_hash=evaluation_config_hash,
        tick_size=tick_size,
    )
    ordered = _validate_and_normalize_executed_trades(trades, tick_size=tick_size)
    if ordered.empty:
        return _empty(stats)

    realized = pd.to_numeric(ordered["_realized_pts"], errors="coerce")
    risk = pd.to_numeric(ordered["_risk_points"], errors="raise")
    ordered = ordered.assign(
        _gross_r=realized / risk,
        _net_r=(realized - cost_points) / risk,
        _cost_r=cost_points / risk,
        _day=ordered["trading_day"].astype(str),
    )
    ordered = ordered.sort_values(
        ["resolution_ts_utc", "trade_id"], kind="mergesort"
    ).reset_index(drop=True)

    net = ordered["_net_r"].astype(float)
    gross = ordered["_gross_r"].astype(float)
    equity = net.cumsum()
    # Initial losses draw down from zero, before the first closed trade.
    drawdown = equity.cummax().clip(lower=0.0) - equity
    max_drawdown = float(drawdown.max()) if len(drawdown) else 0.0

    # time under water: the longest run of distinct trading days spent below
    # the running equity peak
    longest = 0
    current_days: set[str] = set()
    for index in range(len(ordered)):
        if drawdown.iloc[index] > 1e-12:
            current_days.add(ordered["_day"].iloc[index])
            longest = max(longest, len(current_days))
        else:
            current_days = set()

    days = sorted(ordered["_day"].unique())
    thirds = max(1, len(days) // 3)
    blocks = [days[:thirds], days[thirds : 2 * thirds], days[2 * thirds :]]
    signs = []
    for block in blocks:
        if not block:
            continue
        block_net = float(ordered.loc[ordered["_day"].isin(block), "_net_r"].sum())
        signs.append(1 if block_net > 0 else (-1 if block_net < 0 else 0))
    consistency = (
        max(signs.count(1), signs.count(-1)) / len(signs) if signs else None
    )

    session_column = next(
        (c for c in ("entry_session", "session_doc") if c in ordered), None
    )
    stability = None
    if session_column is not None:
        overall_sign = 1 if net.sum() > 0 else (-1 if net.sum() < 0 else 0)
        per_session = ordered.groupby(ordered[session_column].astype(str))["_net_r"].sum()
        if len(per_session) and overall_sign != 0:
            agree = sum(
                1
                for value in per_session
                if (1 if value > 0 else (-1 if value < 0 else 0)) == overall_sign
            )
            stability = agree / len(per_session)

    positive = float(net[net > 0].sum())
    negative = float(-net[net < 0].sum())
    profit_factor = (positive / negative) if negative > 0 else None

    day_pnl = ordered.groupby("_day")["_net_r"].sum()
    day_abs = float(day_pnl.abs().sum())
    top_day_share = float(day_pnl.abs().max() / day_abs) if day_abs > 0 else None
    setup_column = next(
        (c for c in ("envelope_setup_id", "setup_id") if c in ordered), None
    )
    top_setup_share = None
    if setup_column is not None:
        setup_pnl = ordered.groupby(ordered[setup_column].astype(str))["_net_r"].sum()
        setup_abs = float(setup_pnl.abs().sum())
        top_setup_share = (
            float(setup_pnl.abs().max() / setup_abs) if setup_abs > 0 else None
        )

    winners = gross[gross > 0]
    losers = gross[gross < 0]
    avg_winner = float(winners.mean()) if len(winners) else 0.0
    avg_loser = float(losers.mean()) if len(losers) else 0.0
    cost_r_mean = float(ordered["_cost_r"].astype(float).mean())
    planned = PlannedVsRealizedEdge(
        planned_rrr=float(tp_r_multiple),
        realized_avg_winner_r=avg_winner,
        realized_avg_loser_r=avg_loser,
        realized_payoff_ratio=(avg_winner / abs(avg_loser)) if avg_loser else 0.0,
        gross_expectancy_r=float(gross.mean()),
        cost_r=cost_r_mean,
        net_expectancy_r=float(net.mean()),
        slippage_commission_drag_r=cost_r_mean,
    )
    return StrategyMetrics(
        executed_trades=int(len(ordered)),
        independent_days=int(len(days)),
        gross_expectancy_r=float(gross.mean()),
        net_expectancy_r=float(net.mean()),
        profit_factor=profit_factor,
        max_drawdown_r=max_drawdown,
        time_under_water_days=longest,
        time_block_sign_consistency=consistency,
        session_stability_score=stability,
        top_day_pnl_share=top_day_share,
        top_setup_pnl_share=top_setup_share,
        net_expectancy_bootstrap_ci95=_bootstrap_ci(stats),
        planned_vs_realized=planned,
        trade_stats=dict(stats),
    )
