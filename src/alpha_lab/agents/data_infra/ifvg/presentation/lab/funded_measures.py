"""Funded results measures (``CALCULATIONS.md``: table, tiles, checks, verdict, findings).

Every function reads one saved funded comparison (:class:`FundedStudy`) and
returns plain values; the screens format them. Money figures that the result
stores (net cash, received, costs, payouts) are READ from the result, never
recomputed. Measures the result does not store (Sharpe, Sortino, drawdown of
the trade path, resampled ranges) are computed from the saved trade rows, one
configuration at one firm at a time. Firms are never added together.

Definitions follow CALCULATIONS.md exactly; where the saved data cannot supply
an input the value is ``None`` and the screen shows a placeholder.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
    COMPLETED,
    FundedStudy,
    daily_results,
    firm_name,
    ordered_trades,
    trade_path,
)

__all__ = [
    "DEFAULT_SEED",
    "EULER_GAMMA",
    "RANGE_LEVELS",
    "BootstrapRange",
    "Concentration",
    "DropLargestCheck",
    "Finding",
    "FundedRow",
    "GateRow",
    "HeldLeg",
    "HeldLegs",
    "SharpeConfidence",
    "VerdictPart",
    "bootstrap_mean_ranges",
    "concentration",
    "deflated_sharpe",
    "drop_largest_payout",
    "findings",
    "funded_row",
    "held_legs_from_rows",
    "held_to_close",
    "held_to_deadline_legs",
    "split_entry_cost",
    "max_drawdown",
    "quality_gates",
    "ranking",
    "sharpe_annualized",
    "sharpe_confidence",
    "sortino_annualized",
    "verdict",
]

#: fixed default seed for every resampling on the redesigned screens (rule 9)
DEFAULT_SEED = 20260923
EULER_GAMMA = 0.5772156649015329
#: 68% / 90% / 95% ranges as (low percentile, high percentile)
RANGE_LEVELS: dict[int, tuple[float, float]] = {68: (16.0, 84.0), 90: (5.0, 95.0),
                                                 95: (2.5, 97.5)}
TRADING_DAYS_PER_YEAR = 252


# ── series measures ───────────────────────────────────────────────────────


def max_drawdown(path: Sequence[float]) -> float:
    """Largest fall of a cumulative path from its running high (the path starts at $0)."""

    values = np.asarray(path, dtype=float)
    if values.size == 0:
        return 0.0
    return float((np.maximum.accumulate(values) - values).max())


def sharpe_annualized(daily: Sequence[float]) -> float | None:
    """mean(daily) ÷ sample standard deviation(daily) × √252, over every study day."""

    values = np.asarray(daily, dtype=float)
    if values.size < 2:
        return None
    sd = values.std(ddof=1)
    if sd == 0:
        return None
    return float(values.mean() / sd * math.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_annualized(daily: Sequence[float]) -> float | None:
    """mean(daily) ÷ √(mean(min(daily, 0)²)) × √252, over every study day."""

    values = np.asarray(daily, dtype=float)
    if values.size == 0:
        return None
    downside = math.sqrt(float(np.mean(np.minimum(values, 0.0) ** 2)))
    if downside == 0:
        return None
    return float(values.mean() / downside * math.sqrt(TRADING_DAYS_PER_YEAR))


def _moments(daily: np.ndarray) -> tuple[float, float, float] | None:
    """Daily (not annualized) Sharpe, sample skew g3 and kurtosis g4 (not excess).

    ``None`` when the days don't vary (for example no trade at all): the moments
    are undefined then, and nothing may stand in for them (rule 6).
    """

    if daily.size < 2:
        return None
    mean = daily.mean()
    sd = daily.std(ddof=1)
    centered = daily - mean
    m2 = float(np.mean(centered ** 2))
    if not sd or not m2:
        return None
    g3 = float(np.mean(centered ** 3) / m2 ** 1.5)
    g4 = float(np.mean(centered ** 4) / m2 ** 2)
    return float(mean / sd), g3, g4


def _psr_z(sr: float, benchmark: float, g3: float, g4: float, days: int) -> float | None:
    denominator = 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2
    if denominator <= 0 or days < 2:
        return None
    return (sr - benchmark) * math.sqrt(days - 1) / math.sqrt(denominator)


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(p))


@dataclass(frozen=True)
class SharpeConfidence:
    """Probabilistic Sharpe ratio against 0, and the deflated Sharpe ratio for N compared.

    Both use the configuration's daily funded results over every study day (days
    without a trade count as $0), treat the days as independent draws with the
    observed skew and kurtosis, and report Φ of a z-score — a normal-approximation
    confidence score, like one minus a one-sided p-value — comparing the observed
    daily Sharpe ratio with a benchmark: 0 (``above_zero``), or the expected best of
    the N configurations compared at this firm (``deflated``). Neither is
    a probability that the edge is genuine or that payouts will continue, and the
    N-configuration adjustment does not account for research done before the study
    (correction A5). Every measure is ``None`` when the daily results don't vary.
    """

    daily_sharpe: float | None
    skew: float | None
    kurtosis: float | None
    days: int
    above_zero: float | None
    #: configurations compared at this firm (N) and the deflation benchmark
    tested: int | None = None
    benchmark_annualized: float | None = None
    deflated: float | None = None


def sharpe_confidence(daily: Sequence[float]) -> SharpeConfidence:
    values = np.asarray(daily, dtype=float)
    moments = _moments(values)
    if moments is None:
        return SharpeConfidence(daily_sharpe=None, skew=None, kurtosis=None,
                                days=int(values.size), above_zero=None)
    sr, g3, g4 = moments
    z = _psr_z(sr, 0.0, g3, g4, values.size)
    return SharpeConfidence(daily_sharpe=sr, skew=g3, kurtosis=g4, days=int(values.size),
                            above_zero=None if z is None else _norm_cdf(z))


def deflated_sharpe(study: FundedStudy, configuration: str, firm_key: str) -> SharpeConfidence:
    """Probabilistic Sharpe of one configuration, then deflated for the N compared at the firm.

    SR0 = √V × ((1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))), V = sample variance of
    the daily Sharpe ratios of the N completed configurations at this firm. N
    counts every completed configuration; one whose days never vary (no trades)
    enters V with a daily Sharpe of 0 (decision log).
    """

    base = sharpe_confidence([v for _, v in daily_results(study, configuration, firm_key)])
    ratios = []
    for summary in study.completed_at(firm_key):
        values = np.asarray([v for _, v in daily_results(
            study, str(summary["configuration"]), firm_key)], dtype=float)
        sd = values.std(ddof=1) if values.size > 1 else 0.0
        ratios.append(float(values.mean() / sd) if sd else 0.0)
    n = len(ratios)
    if n < 2 or base.daily_sharpe is None:
        return SharpeConfidence(**{**base.__dict__, "tested": n or None})
    variance = float(np.var(ratios, ddof=1))
    benchmark = math.sqrt(variance) * ((1 - EULER_GAMMA) * _norm_ppf(1 - 1 / n)
                                       + EULER_GAMMA * _norm_ppf(1 - 1 / (n * math.e)))
    z = _psr_z(base.daily_sharpe, benchmark, base.skew, base.kurtosis, base.days)
    return SharpeConfidence(
        daily_sharpe=base.daily_sharpe, skew=base.skew, kurtosis=base.kurtosis, days=base.days,
        above_zero=base.above_zero, tested=n,
        benchmark_annualized=benchmark * math.sqrt(TRADING_DAYS_PER_YEAR),
        deflated=None if z is None else _norm_cdf(z))


# ── funded results table ──────────────────────────────────────────────────


@dataclass(frozen=True)
class FundedRow:
    """One ranking-table row: one configuration at one firm."""

    configuration: str
    firm_key: str
    firm: str
    completed: bool
    rank: int | None
    reason: str = ""
    net_cash_cents: int | None = None
    received_cents: int | None = None
    payouts: int | None = None
    costs_cents: int | None = None
    accounts: int | None = None
    lost_before_payout: int | None = None
    lost_after_payout: int | None = None
    largest_payout_cents: int | None = None
    median_payout_cents: int | None = None
    trades: int | None = None
    #: net cash ÷ account costs
    cash_per_dollar: float | None = None
    worst_drawdown: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    #: strategy measures (no accounts), never recomputed from funded trades
    net_r: float | None = None
    win_rate_pct: float | None = None


def _payouts_received_cents(study: FundedStudy, configuration: str, firm_key: str) -> list[int]:
    return sorted(int(round(float(p.get("trader_usd") or 0) * 100))
                  for p in study.rows("payout_events", configuration, firm_key)
                  if p.get("event") == "received")


def _median_cents(values: list[int]) -> int | None:
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) // 2


def funded_row(study: FundedStudy, configuration: str, firm_key: str) -> FundedRow:
    summary = study.summary(configuration, firm_key) or {}
    name = firm_name(study, firm_key)
    if summary.get("status") != COMPLETED:
        failure = next((row for row in (study.result.get("full_range_batch") or {}).get(
            "failed_configurations", [])
            if (row.get("configuration") or row.get("batch_id")) == configuration), {})
        disposition = next((row for row in (study.result.get("mffu_batch") or {}).get(
            "dispositions", []) if row.get("variant_id") == configuration), {})
        return FundedRow(configuration=configuration, firm_key=firm_key, firm=name,
                         completed=False, rank=None,
                         reason=str(disposition.get("reason") or summary.get("reason")
                                    or failure.get("reason")
                                    or "No result was saved."))
    costs = int(summary["account_costs_cents"])
    net = int(summary["net_cash_earned_cents"])
    measures = study.strategy_measures(configuration) or {}
    daily = ([v for _, v in daily_results(study, configuration, firm_key)]
             if study.calendar else None)
    paid = _payouts_received_cents(study, configuration, firm_key)
    rank = summary.get("rank_within_firm")
    return FundedRow(
        configuration=configuration, firm_key=firm_key, firm=name, completed=True,
        rank=None if rank is None else int(rank),
        net_cash_cents=net, received_cents=int(summary["payouts_received_cents"]),
        payouts=int(summary["payouts_received_count"]), costs_cents=costs,
        accounts=int(summary["accounts_purchased"]),
        lost_before_payout=int(summary["accounts_lost_before_first_payout"]),
        lost_after_payout=int(summary["accounts_lost_after_a_payout"]),
        largest_payout_cents=(int(summary["largest_single_payout_cents"])
                              if int(summary["payouts_received_count"]) else None),
        median_payout_cents=_median_cents(paid),
        trades=int(summary.get("trades_taken") or 0),
        cash_per_dollar=(net / costs) if costs else None,
        worst_drawdown=max_drawdown(trade_path(study, configuration, firm_key)),
        sharpe=sharpe_annualized(daily) if daily is not None else None,
        sortino=sortino_annualized(daily) if daily is not None else None,
        net_r=measures.get("net_r_after_costs"),
        win_rate_pct=measures.get("win_rate_pct"),
    )


def ranking(study: FundedStudy, firm_key: str) -> list[FundedRow]:
    """Every configuration at one firm: completed by saved rank, then not completed."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_matrix import is_mffu_plan

    summaries = (study.result.get("summaries_cents") or {}).values()
    keys = ([*study.configurations] if is_mffu_plan(study.plan) else
            [str(s.get("configuration")) for s in summaries if s.get("firm_key") == firm_key])
    rows = [funded_row(study, key, firm_key) for key in dict.fromkeys(keys)]
    rows.sort(key=lambda r: (not r.completed, r.rank if r.rank is not None else 10 ** 9,
                             r.configuration))
    return rows


# ── checks on the leader ──────────────────────────────────────────────────


@dataclass(frozen=True)
class DropLargestCheck:
    firm_key: str
    leader: str | None
    leader_after_cents: int | None
    runner_up: str | None
    runner_up_after_cents: int | None
    passed: bool | None


def drop_largest_payout(study: FundedStudy, firm_key: str) -> DropLargestCheck:
    """Subtract each configuration's largest payout from its net cash and re-rank."""

    rows = study.completed_at(firm_key)
    if not rows:
        return DropLargestCheck(firm_key, None, None, None, None, None)
    leader = str(rows[0]["configuration"])
    after = sorted(((int(s["net_cash_earned_cents"])
                     - (int(s["largest_single_payout_cents"])
                        if int(s["payouts_received_count"]) else 0),
                     str(s["configuration"])) for s in rows), reverse=True)
    top_value = after[0][0]
    leader_value = next(v for v, k in after if k == leader)
    runner = next(((v, k) for v, k in after if k != leader), (None, None))
    # a tie keeps the leader at rank 1, as the saved ranking shares tied ranks
    return DropLargestCheck(firm_key=firm_key, leader=leader, leader_after_cents=leader_value,
                            runner_up=runner[1], runner_up_after_cents=runner[0],
                            passed=leader_value >= top_value)


@dataclass(frozen=True)
class BootstrapRange:
    mean: float
    count: int
    paths: int
    seed: int
    #: level (68, 90, 95) → (low, high)
    ranges: dict[int, tuple[float, float]] = field(default_factory=dict)


def bootstrap_mean_ranges(values: Sequence[float], *, paths: int = 20_000,
                          seed: int = DEFAULT_SEED) -> BootstrapRange | None:
    """Resample the trades' net results with replacement (same count), mean of each."""

    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = np.empty(paths)
    chunk = max(1, 2_000_000 // max(1, data.size))
    for start in range(0, paths, chunk):
        stop = min(paths, start + chunk)
        means[start:stop] = data[rng.integers(0, data.size, (stop - start, data.size))].mean(axis=1)
    ranges = {level: (float(np.percentile(means, lo)), float(np.percentile(means, hi)))
              for level, (lo, hi) in RANGE_LEVELS.items()}
    return BootstrapRange(mean=float(data.mean()), count=int(data.size), paths=paths, seed=seed,
                          ranges=ranges)


# ── quality gates (strategy replay, no accounts) ──────────────────────────


@dataclass(frozen=True)
class GateRow:
    gate: str
    required: str
    #: the strategy replay's value as text, or None when the replay did not store it
    value: str | None
    passed: bool | None
    #: True when the saved threshold itself is an open owner decision
    needs_decision: bool = False
    note: str = ""


#: gates whose saved threshold is still the owner's open decision (TASKS.md)
OPEN_DECISION_GATES = frozenset({"max_time_under_water_days"})


def quality_gates(study: FundedStudy, configuration: str,
                  thresholds: dict[str, Any] | None, *, funded_days_with_trade: int | None = None,
                  funded_best_day_share: float | None = None) -> list[GateRow]:
    """The saved gate thresholds evaluated on the stored strategy measures.

    A gate whose input the strategy replay did not store gets ``value=None``
    ("Not in export"); a funded-trade figure, when given, is shown only as a
    labeled note, never as the gate's input.
    """

    m = study.strategy_measures(configuration) or {}
    t = thresholds or {}
    rows: list[GateRow] = []

    def add(key: str, gate: str, required: str, value: str | None, passed: bool | None,
            note: str = "") -> None:
        if key not in t:
            required, passed = "Not saved", None
        rows.append(GateRow(gate=gate, required=required, value=value, passed=passed,
                            needs_decision=key in OPEN_DECISION_GATES, note=note))

    trades = m.get("trades")
    add("min_executed_trades", "Trades", f"{t.get('min_executed_trades', 0):,.0f}+",
        None if trades is None else f"{int(trades):,}",
        None if trades is None or "min_executed_trades" not in t
        else int(trades) >= float(t["min_executed_trades"]))
    add("min_independent_days", "Days with a trade", f"{t.get('min_independent_days', 0):,.0f}+",
        None, None,
        "" if funded_days_with_trade is None
        else f"{funded_days_with_trade} on funded trades")
    expectancy = m.get("expectancy_r_per_trade")
    add("min_net_expectancy_r", "Result per trade", f"{t.get('min_net_expectancy_r', 0):g}R+",
        None if expectancy is None else f"{float(expectancy):.2f}R",
        None if expectancy is None or "min_net_expectancy_r" not in t
        else float(expectancy) >= float(t["min_net_expectancy_r"]))
    pf = m.get("profit_factor")
    add("min_profit_factor", "Profit factor", f"{t.get('min_profit_factor', 0):g}+",
        None if pf is None else f"{float(pf):.2f}",
        None if pf is None or "min_profit_factor" not in t
        else float(pf) >= float(t["min_profit_factor"]))
    dd = m.get("max_drawdown_r")
    add("max_drawdown_r", "Drawdown", f"{t.get('max_drawdown_r', 0):g}R max",
        None if dd is None else f"{float(dd):.2f}R",
        None if dd is None or "max_drawdown_r" not in t
        else float(dd) <= float(t["max_drawdown_r"]))
    add("max_top_day_pnl_share", "Best day's share of profit",
        f"{float(t.get('max_top_day_pnl_share', 0)) * 100:.0f}% max", None, None,
        "" if funded_best_day_share is None
        else f"{funded_best_day_share * 100:.0f}% on funded days")
    underwater = m.get("longest_trading_days_under_water")
    add("max_time_under_water_days", "Trading days under water",
        f"{t.get('max_time_under_water_days', 0):,.0f} max",
        None if underwater is None else f"{int(underwater):,}",
        None if underwater is None or "max_time_under_water_days" not in t
        else int(underwater) <= float(t["max_time_under_water_days"]))
    rows.append(GateRow(
        gate="Session stability · time-block consistency · best setup's share",
        required=(f"{t.get('min_session_stability_score', '—')} · "
                  f"{t.get('min_time_block_sign_consistency', '—')} · "
                  f"{float(t.get('max_top_setup_pnl_share', 0)) * 100:.0f}%"
                  if t else "Not saved"),
        value=None, passed=None))
    return rows


# ── concentration ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Concentration:
    largest_account_share: float | None
    five_largest_trades_share: float | None
    best_month: str | None
    best_month_share: float | None
    largest_payout_share: float | None
    best_day_share: float | None


def concentration(study: FundedStudy, configuration: str, firm_key: str) -> Concentration:
    summary = study.summary(configuration, firm_key) or {}
    received = int(summary.get("payouts_received_cents") or 0) / 100
    net_cash = int(summary.get("net_cash_earned_cents") or 0) / 100
    journeys = study.rows("account_journeys", configuration, firm_key)
    account_received = [float(j.get("received_usd") or 0) for j in journeys]
    results = np.asarray([float(t.get("net_pnl_usd") or 0)
                          for t in ordered_trades(study, configuration, firm_key)])
    total_profit = float(results.sum()) if results.size else 0.0
    months = [(float(m.get("net_usd") or 0), str(m.get("month")))
              for m in study.rows("monthly_results", configuration, firm_key)]
    best_month = max(months) if months else None
    largest = int(summary.get("largest_single_payout_cents") or 0) / 100
    daily = (np.asarray([v for _, v in daily_results(study, configuration, firm_key)])
             if study.calendar else np.asarray([]))
    winning = float(daily[daily > 0].sum()) if daily.size else 0.0
    return Concentration(
        largest_account_share=(max(account_received) / received) if received > 0 else None,
        five_largest_trades_share=(float(np.sort(results)[-5:].sum()) / total_profit
                                   if total_profit > 0 else None),
        best_month=best_month[1] if best_month else None,
        best_month_share=(best_month[0] / net_cash) if best_month and net_cash > 0 else None,
        largest_payout_share=(largest / received) if received > 0 else None,
        best_day_share=(float(daily.max()) / winning) if winning > 0 else None,
    )


def held_to_close(study: FundedStudy, configuration: str, firm_key: str) -> tuple[int, float]:
    """(count, net result) of trades whose last part closed at the scheduled daily close.

    Whole trades, first halves included; kept for compatibility. The Summary uses
    :func:`held_to_deadline_legs`, which separates the held halves (correction A9).
    """

    held = [t for t in ordered_trades(study, configuration, firm_key)
            if t.get("exit_kind") == "scheduled_close"]
    return len(held), round(sum(float(t.get("net_pnl_usd") or 0) for t in held), 2)


@dataclass(frozen=True)
class HeldLeg:
    """One trade held to the daily deadline, split into its two legs (exact cents)."""

    seq: Any
    first_cents: int  # the half closed at the target (0 without a half exit)
    rest_cents: int  # the contracts held to the deadline
    recorded_net_cents: int
    had_partial: bool = False


@dataclass(frozen=True)
class HeldLegs:
    """Trades held to the 3:55 PM daily deadline, with first halves and remainders apart.

    Correction A9. Reconstructed from each trade's recorded ticks, quantities and
    the saved cost per contract per fill; a stored figure is never changed.
    ``reconciled`` only when, for every trade, first + rest equals the recorded
    net and the three fill costs equal the recorded costs. ``whole_cents`` is the
    sum of the recorded nets.
    """

    count: int
    first_halves_cents: int
    remainders_cents: int
    whole_cents: int
    reconciled: bool
    half_exit: bool
    per_trade: tuple[HeldLeg, ...] = ()

    @property
    def partial_count(self) -> int:
        return sum(leg.had_partial for leg in self.per_trade)

    @property
    def partial_remainders_cents(self) -> int:
        return sum(leg.rest_cents for leg in self.per_trade if leg.had_partial)

    @property
    def whole_deadline_count(self) -> int:
        return sum(not leg.had_partial for leg in self.per_trade)

    @property
    def whole_deadline_cents(self) -> int:
        return sum(leg.recorded_net_cents for leg in self.per_trade if not leg.had_partial)


def split_entry_cost(quantity: int, half: int, mills: int) -> tuple[int, int]:
    """The entry fill's cost (cents) allocated to (the half, the rest) by quantity.

    Allocate the posted per-fill rounded total proportionally in integer cents;
    the final leg receives the residual. No extra fee is charged.
    """

    from alpha_lab.propsim.funded.position_walk import fill_cost_cents
    from alpha_lab.propsim.funded.reporting_legs import allocate_entry_fee

    return allocate_entry_fee(fill_cost_cents(quantity, mills), quantity, half)


def held_legs_from_rows(rows: Sequence[dict[str, Any]], *, tick_value_cents: int | None,
                        mills: int | None, half_exit: bool) -> HeldLegs:
    """Split recorded deadline trades into legs (see :class:`HeldLegs`).

    For a trade with a half fill: first = (scale-out − entry) ticks × sign × tick
    value × half quantity − the entry cost allocated to the half − the half's own
    fill cost; rest = (exit − entry) ticks × sign × tick value × final quantity −
    the entry cost allocated to the rest − the final fill cost. Without a half
    fill: first = 0 and rest = the whole trade. Missing sizing, a fill that is not
    a whole number of cents or any difference from the recorded net or costs
    leaves the legs unreconciled. Fractional-cent unit products are allowed by
    the bound per-fill half-up convention. Whole deadline positions remain a
    separate population through ``whole_deadline_*`` / ``partial_*``.
    """

    from alpha_lab.propsim.funded.reporting_legs import money_cents, trade_legs

    legs: list[HeldLeg] = []
    reconciled = tick_value_cents is not None and mills is not None
    for row in rows:
        net = money_cents(row, "net_pnl")
        scaled = int(row.get("scale_out_quantity") or 0) > 0
        if tick_value_cents is None or mills is None:
            legs.append(HeldLeg(row.get("seq"), 0, net, net, scaled))
            continue
        try:
            value = trade_legs(row, tick_value_cents=tick_value_cents, mills=mills)
            first, rest = value.first_cents, value.remaining_cents
        except (KeyError, TypeError, ValueError):
            reconciled = False
            first, rest = 0, net
        legs.append(HeldLeg(row.get("seq"), first, rest, net, scaled))
    return HeldLegs(
        count=len(legs), first_halves_cents=sum(leg.first_cents for leg in legs),
        remainders_cents=sum(leg.rest_cents for leg in legs),
        whole_cents=sum(leg.recorded_net_cents for leg in legs),
        reconciled=bool(reconciled and legs), half_exit=half_exit, per_trade=tuple(legs))


def held_to_deadline_legs(study: FundedStudy, configuration: str, firm_key: str) -> HeldLegs:
    """The trades with ``exit_kind == "scheduled_close"``, split into their legs.

    Tick value and cost per contract per fill come from the saved
    ``settings.sizing_by_configuration[configuration]``.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.firm_race import _sizing

    rows = [t for t in ordered_trades(study, configuration, firm_key)
            if t.get("exit_kind") == "scheduled_close"]
    try:
        _quantity, tick_value, mills, half_exit = _sizing(study, configuration)
    except (KeyError, TypeError, ValueError):
        tick_value = mills = None
        half_exit = any(t.get("scale_out_ns") is not None for t in rows)
    return held_legs_from_rows(rows, tick_value_cents=tick_value, mills=mills,
                               half_exit=half_exit)


# ── verdict and findings ──────────────────────────────────────────────────


@dataclass(frozen=True)
class VerdictPart:
    part: str
    status: str | None  # None: "Not shown"
    tone: str  # blue / orange / neutral
    text: str


def verdict(*, integrity_ok: bool, integrity_text: str, range95_low: float | None,
            deflated: float | None, edge_text: str, worst_drawdown: float | None,
            loss_limit: float | None, lost_before_payout: int, account_text: str,
            trading_days: int, has_unseen_result: bool, sample_text: str) -> list[VerdictPart]:
    """The four-part verdict with the default rules of CALCULATIONS.md."""

    parts = [VerdictPart("Data integrity", "Pass" if integrity_ok else "Fail",
                         "blue" if integrity_ok else "orange", integrity_text)]
    # each check passes, fails or is unavailable; unavailable is never counted as a
    # failure (correction A11) and the weakest state is always shown (correction A5)
    known = [passed for passed in (None if range95_low is None else range95_low > 0,
                                   None if deflated is None else deflated >= 0.5)
             if passed is not None]
    if not known:
        parts.append(VerdictPart("Edge", "Not available", "neutral", edge_text))
    elif len(known) == 1:  # one check could not be made
        parts.append(VerdictPart("Edge", "Partly checked", "neutral", edge_text) if known[0]
                     else VerdictPart("Edge", "Weak", "orange", edge_text))
    elif all(known):
        parts.append(VerdictPart("Edge", "Holds", "blue", edge_text))
    elif any(known):
        parts.append(VerdictPart("Edge", "Weak", "orange", edge_text))
    else:
        parts.append(VerdictPart("Edge", "Not supported", "orange", edge_text))
    watch = ((worst_drawdown is not None and loss_limit is not None
              and worst_drawdown > loss_limit) or lost_before_payout > 0)
    parts.append(VerdictPart("Account risk", "Watch" if watch else "OK",
                             "orange" if watch else "blue", account_text))
    limited = trading_days < 250 or not has_unseen_result
    parts.append(VerdictPart("Sample", "Limited" if limited else "Adequate",
                             "orange" if limited else "blue", sample_text))
    return parts


@dataclass(frozen=True)
class Finding:
    severity: str  # High / Medium / Info
    title: str
    text: str
    next_step: str


def _possessive(name: str) -> str:
    """``TakeProfitTrader's``, ``MyFundedFutures'``."""

    return f"{name}'" if name.endswith("s") else f"{name}'s"


def _whole(usd: float, *, signed: bool = False) -> str:
    """Whole dollars with the screens' minus sign (``−$2,000``, ``+$2,600``)."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.format import money_whole

    return money_whole(usd, signed=signed)


#: the firm finding's next step and the flat one's (correction A3)
FIRM_RACE_NEXT = "see its limits under Risk and simulation before relying on it."
FLAT_RACE_NEXT = ("run the ledger version under Risk and simulation before drawing account "
                  "conclusions.")


def _early_losses(share: float, *, basis: str | None, detail: str | None,
                  boundaries: tuple[float, float] | None, firm: str | None) -> Finding:
    owner = _possessive(firm) if firm else "the firm's"
    if basis == "firm":
        where = f" ({detail})" if detail else ""
        return Finding(
            "Medium", "Early account failures (conditional model)",
            f"In conditional resampling of the recorded trades with {owner} ledger rules"
            f"{where}, {share * 100:.0f}% of first accounts failed before any payout was "
            "received. It reuses the recorded trades (already shaped by the historical "
            "accounts' entry selection and skipped opportunities) in fixed slots, with "
            "shortened results and compressed intratrade paths, so it is not an exact "
            "fresh-account probability.",
            FIRM_RACE_NEXT)
    lower, upper = ((_whole(boundaries[0]), _whole(boundaries[1], signed=True))
                    if boundaries else ("the lower boundary", "the upper boundary"))
    inside = ", ".join(part for part in (f"{lower} / {upper}" if boundaries else "", detail or "")
                       if part)
    return Finding(
        "Medium", "Early losses in the fixed-boundary diagnostic",
        f"In the fixed closed-profit diagnostic{f' ({inside})' if inside else ''}, "
        f"{share * 100:.0f}% of paths fell to {lower} before reaching {upper}. It uses closed "
        f"trade results only, not {owner} account rules. Run conditional resampling with "
        f"{owner} rules under Risk and simulation for the ledger version.",
        FLAT_RACE_NEXT)


def _held_finding(*, held: HeldLegs | None, held_count: int, held_net: float,
                  total_profit: float, half_exit: bool) -> Finding | None:
    if total_profit <= 0:
        return None
    total = _whole(total_profit)
    if held is not None and held.half_exit and held.reconciled:
        rest = held.partial_remainders_cents / 100
        if rest < total_profit:
            return None  # the held halves alone don't carry the profit
        return Finding(
            "Medium", "Held halves carry the profit",
            f"The held halves of {held.partial_count} trades kept to the daily deadline made "
            f"{_whole(rest)}, more than the configuration's total trading profit ({total}). "
            f"Their first halves, closed at the target, made "
            f"{_whole(held.first_halves_cents / 100)} (whole trades "
            f"{_whole((held.whole_cents - held.whole_deadline_cents) / 100)})."
            + (f" Another {held.whole_deadline_count} whole positions closed at the deadline "
               f"without a partial, making {_whole(held.whole_deadline_cents / 100)}."
               if held.whole_deadline_count else ""),
            "test a capped hold on the second half.")
    if held is not None:
        held_count, held_net, half_exit = held.count, held.whole_cents / 100, held.half_exit
    if held_net < total_profit:
        return None
    return Finding(
        "Medium", "Trades held to the daily deadline carry the profit",
        f"Total profit from {held_count} trades held to the 3:55 PM deadline: "
        f"{_whole(held_net)}, more than the configuration's total trading profit ({total}).",
        "test a capped hold on the second half." if half_exit
        else "test an earlier exit for trades still open late in the day.")


def findings(*, largest_account_share: float | None, five_largest_share: float | None,
             died_first_share: float | None, total_profit: float, beta_r2: float | None,
             held: HeldLegs | None = None, held_count: int = 0, held_net: float = 0.0,
             half_exit: bool = False, race_basis: str | None = None,
             race_detail: str | None = None, boundaries: tuple[float, float] | None = None,
             firm: str | None = None, beta_days: int | None = None) -> list[Finding]:
    """The four findings and when each fires (CALCULATIONS.md defaults, corrections A3–A9).

    ``died_first_share`` is the figure of whichever model the screen quotes:
    ``race_basis`` ``"firm"`` (conditional resampling with the firm's ledger rules;
    ``race_detail`` its paths and horizon, e.g. "1,000 paths, to June 10, 2026") or
    ``"flat"`` (the fixed closed-profit boundary diagnostic; ``boundaries`` its
    (lower, upper) and ``race_detail`` its path count). Either fires at 15% or more
    of the figure shown. ``held`` gives the deadline trades' legs (correction A9);
    ``held_count``/``held_net`` are the whole-trade fallback when it is absent.
    ``beta_days`` is the number of days in the straight-line fit.
    """

    out: list[Finding] = []
    if ((largest_account_share is not None and largest_account_share >= 0.75)
            or (five_largest_share is not None and five_largest_share >= 0.60)):
        pieces = []
        if largest_account_share is not None:
            pieces.append(f"One account produced {largest_account_share * 100:.0f}% of cash "
                          "received")
        if five_largest_share is not None:
            pieces.append(f"five trades carry {five_largest_share * 100:.0f}% of trading profit")
        out.append(Finding("High", "Concentrated result", ", and ".join(pieces) + ".",
                           "confirm on the unseen window and the older years before sizing up."))
    if died_first_share is not None and died_first_share >= 0.15:
        out.append(_early_losses(died_first_share, basis=race_basis, detail=race_detail,
                                 boundaries=boundaries, firm=firm))
    held_finding = _held_finding(held=held, held_count=held_count, held_net=held_net,
                                 total_profit=total_profit, half_exit=half_exit)
    if held_finding is not None:
        out.append(held_finding)
    if beta_r2 is not None and beta_r2 < 0.2:
        fit = f"R² {beta_r2:.2f}" + (f", {beta_days:,} days" if beta_days else "")
        out.append(Finding(
            "Info", "Low daily linear association with the index",
            "A straight-line fit of daily results on the daily E-mini change explains about "
            f"{beta_r2 * 100:.0f}% of their day-to-day variation ({fit}). This is an "
            "observation about daily co-movement; it doesn't show that the result was "
            "independent of market direction.",
            "compare rising and falling periods on Market conditions (labels known at entry) "
            "before relying on it."))
    return out
