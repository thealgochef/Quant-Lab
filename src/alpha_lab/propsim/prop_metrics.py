"""Immutable prop result metric blocks (CS §5.6) from walked simulations.

Every block is computed from a set of :class:`AccountWalkResult`s (one per
bootstrap draw, or a single historical path) over a horizon expressed in
played trading days. Lower-tail metrics (P10, drought P90) are first-class.
Pass/breach/payout detection is EVENT-based: the strictly ordered account
event stream is the record, never a verdict-string heuristic — a path that
passes evaluation and later breaches funded still counts as an evaluation
pass, and its breach is scoped to the phase it occurred in.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alpha_lab.agents.data_infra.ifvg.search.identities import FrozenContract
from alpha_lab.propsim.account import AccountWalkResult
from alpha_lab.propsim.firm_contracts import AccountPhase

__all__ = [
    "EvaluationFitness",
    "FundedFitness",
    "CashExtraction",
    "PortfolioFitness",
    "PayoutReliabilityVector",
    "build_evaluation_fitness",
    "build_payout_reliability_vector",
]


class EvaluationFitness(FrozenContract):
    pass_probability: float
    median_days_to_pass: float | None
    breach_probability: float
    expiration_probability: float
    expected_fees_paid: float


class FundedFitness(FrozenContract):
    first_payout_probability_30d: float
    first_payout_probability_60d: float
    three_payout_probability: float
    breach_probability_90d: float
    median_account_lifetime_days: float | None


class CashExtraction(FrozenContract):
    expected_net_payout_90d: float
    p10_net_payout_90d: float
    median_days_between_payouts: float | None
    p90_payout_drought_days: float | None
    expected_replacement_cost: float


class PortfolioFitness(FrozenContract):
    probability_at_least_one_payout_90d: float
    probability_all_accounts_breach_90d: float
    expected_portfolio_net_payout_90d: float
    p10_portfolio_net_payout_90d: float
    payout_concentration_by_firm: tuple[tuple[str, float], ...]


class PayoutReliabilityVector(FrozenContract):
    first_payout_probability_30d: float
    first_payout_probability_60d: float
    three_payout_probability: float
    payout_probability_per_rolling_30d: float
    median_days_between_payouts: float | None
    p90_payout_drought_days: float | None
    expected_net_payout_90d: float
    p10_net_payout_90d: float
    breach_probability_90d: float
    expected_replacement_cost: float


def _fraction(count: int, total: int) -> float:
    return count / total if total else 0.0


def _percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def _event_day_ordinals(result: AccountWalkResult) -> list[int]:
    """The 1-based played-day ordinal each event landed on, stream-aligned.

    Day boundaries come from the first equity_update seen on each new
    calendar-day key (the walk emits at least one per traded day). Events
    before the first traded day (a deferred opening fee) land on day 1.
    """

    ordinals: list[int] = []
    ordinal = 0
    seen_days: set[str] = set()
    for event in result.events:
        day_key = event.event_ts_utc[:10]
        if event.event_type == "equity_update" and day_key not in seen_days:
            seen_days.add(day_key)
            ordinal += 1
        ordinals.append(max(1, ordinal))
    return ordinals


_PASS_REASONS = ("evaluation_passed", "evaluation_passed_no_funded_phase")


def build_evaluation_fitness(results: Sequence[AccountWalkResult]) -> EvaluationFitness:
    total = len(results)
    passes = 0
    days_to_pass: list[float] = []
    evaluation_breaches = 0
    expirations = 0
    for result in results:
        ordinals = _event_day_ordinals(result)
        passed_at: int | None = None
        breached_in_evaluation = False
        expired = False
        for event, day_ordinal in zip(result.events, ordinals, strict=True):
            if (
                passed_at is None
                and event.event_type == "phase_transition"
                and event.payload.reason in _PASS_REASONS
            ):
                passed_at = day_ordinal
            if (
                event.event_type == "breach"
                and event.account_phase is AccountPhase.EVALUATION
            ):
                breached_in_evaluation = True
            if (
                event.event_type == "phase_transition"
                and event.payload.to_phase is AccountPhase.EXPIRED
            ):
                expired = True
        if passed_at is not None:
            passes += 1
            days_to_pass.append(float(passed_at))
        if breached_in_evaluation:
            evaluation_breaches += 1
        if expired:
            expirations += 1
    return EvaluationFitness(
        pass_probability=_fraction(passes, total),
        median_days_to_pass=_percentile(days_to_pass, 50),
        breach_probability=_fraction(evaluation_breaches, total),
        expiration_probability=_fraction(expirations, total),
        expected_fees_paid=(
            float(np.mean([result.total_fees for result in results])) if results else 0.0
        ),
    )


def build_payout_reliability_vector(
    results: Sequence[AccountWalkResult],
    *,
    horizon_days_90: int = 90,
) -> PayoutReliabilityVector:
    """The payout-first reliability vector over one simulation's paths."""

    total = len(results)
    first_30 = 0
    first_60 = 0
    three_payouts = 0
    breach_90 = 0
    payout_gaps: list[float] = []
    droughts: list[float] = []
    net_payout_90: list[float] = []
    replacement_costs: list[float] = []
    rolling_30_hits = 0
    rolling_30_windows = 0
    for result in results:
        ordinals = _event_day_ordinals(result)
        payout_days: list[int] = []
        breach_within_horizon = False
        trader_paid = 0.0
        replacement_cost = 0.0
        for event, day_ordinal in zip(result.events, ordinals, strict=True):
            if event.event_type == "payout":
                payout_days.append(day_ordinal)
                if day_ordinal <= horizon_days_90:
                    trader_paid += event.payload.trader_amount
            elif event.event_type == "breach" and day_ordinal <= horizon_days_90:
                breach_within_horizon = True
            elif event.event_type == "replacement":
                replacement_cost += event.payload.reset_fee
        payout_count = len(payout_days)
        if payout_days and payout_days[0] <= 30:
            first_30 += 1
        if payout_days and payout_days[0] <= 60:
            first_60 += 1
        if payout_count >= 3:
            three_payouts += 1
        if breach_within_horizon:
            breach_90 += 1
        if payout_count >= 2:
            payout_gaps.extend(
                float(b - a) for a, b in zip(payout_days, payout_days[1:], strict=False)
            )
        walked = result.final_state.account_age_days
        last = 0
        for day in (*payout_days, walked or 0):
            droughts.append(float(day - last))
            last = day
        net_payout_90.append(trader_paid - result.total_fees)
        replacement_costs.append(replacement_cost)
        windows = max(1, (walked or 0) // 30)
        rolling_30_windows += windows
        rolling_30_hits += min(windows, payout_count)
    return PayoutReliabilityVector(
        first_payout_probability_30d=_fraction(first_30, total),
        first_payout_probability_60d=_fraction(first_60, total),
        three_payout_probability=_fraction(three_payouts, total),
        payout_probability_per_rolling_30d=_fraction(rolling_30_hits, rolling_30_windows),
        median_days_between_payouts=_percentile(payout_gaps, 50),
        p90_payout_drought_days=_percentile(droughts, 90),
        expected_net_payout_90d=(
            float(np.mean(net_payout_90)) if net_payout_90 else 0.0
        ),
        p10_net_payout_90d=_percentile(net_payout_90, 10) or 0.0,
        breach_probability_90d=_fraction(breach_90, total),
        expected_replacement_cost=(
            float(np.mean(replacement_costs)) if replacement_costs else 0.0
        ),
    )
