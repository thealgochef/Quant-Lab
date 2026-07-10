"""Day-level block bootstrap Monte Carlo over the evaluation walk.

Whole trading days are resampled with replacement (a day's trades stay
together, in their intra-day order) and each run walks until PASS or BUST —
an evaluation has no calendar limit — bounded by ``max_days`` as a runaway
guard (runs that hit it count as ``incomplete``). Seeded and deterministic:
one ``numpy`` generator drawn sequentially.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from alpha_lab.propsim.engine import EvaluationWalk
from alpha_lab.propsim.models import Ruleset, TradePath

_WILSON_Z = 1.959963984540054  # 95%


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return (0.0, 1.0)
    z2 = _WILSON_Z * _WILSON_Z
    phat = successes / trials
    denom = 1.0 + z2 / trials
    center = (phat + z2 / (2.0 * trials)) / denom
    half = (
        _WILSON_Z
        * math.sqrt(phat * (1.0 - phat) / trials + z2 / (4.0 * trials * trials))
        / denom
    )
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass(frozen=True)
class BootstrapSummary:
    """Monte Carlo summary for one (fill column × breach mode) cell."""

    n_runs: int
    seed: int
    max_days: int
    p_pass: float
    p_bust: float
    p_incomplete: float
    pass_ci95_low: float
    pass_ci95_high: float
    days_to_pass_median: float | None
    days_to_pass_p10: float | None
    days_to_pass_p90: float | None
    days_to_bust_median: float | None
    bust_reasons: dict[str, int]


def run_bootstrap(
    day_blocks: Sequence[tuple[date, Sequence[TradePath]]],
    ruleset: Ruleset,
    *,
    column: str,
    breach_mode: str,
    n_runs: int = 10_000,
    seed: int = 42,
    max_days: int = 1_000,
) -> BootstrapSummary:
    """Bootstrap P(pass)/P(bust) and days-to-outcome quantiles."""
    if not day_blocks:
        msg = "day-level bootstrap requires at least one trading day of trades"
        raise ValueError(msg)
    if n_runs <= 0:
        msg = f"n_runs must be positive, got {n_runs}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    n_days = len(day_blocks)
    passes = 0
    busts = 0
    days_to_pass: list[int] = []
    days_to_bust: list[int] = []
    bust_reasons: Counter[str] = Counter()
    for _ in range(n_runs):
        walk = EvaluationWalk(ruleset, column=column, breach_mode=breach_mode)
        verdict: str | None = None
        for _day in range(max_days):
            _, day_trades = day_blocks[int(rng.integers(0, n_days))]
            verdict = walk.play_day(day_trades)
            if verdict is not None:
                break
        result = walk.result()
        if verdict == "pass":
            passes += 1
            days_to_pass.append(result.days_to_outcome or 0)
        elif verdict == "bust":
            busts += 1
            days_to_bust.append(result.days_to_outcome or 0)
            bust_reasons[result.bust_reason or "unknown"] += 1
    incomplete = n_runs - passes - busts
    ci_low, ci_high = wilson_interval(passes, n_runs)

    def _pct(values: list[int], q: float) -> float | None:
        return float(np.percentile(np.asarray(values), q)) if values else None

    return BootstrapSummary(
        n_runs=n_runs,
        seed=seed,
        max_days=max_days,
        p_pass=passes / n_runs,
        p_bust=busts / n_runs,
        p_incomplete=incomplete / n_runs,
        pass_ci95_low=ci_low,
        pass_ci95_high=ci_high,
        days_to_pass_median=_pct(days_to_pass, 50),
        days_to_pass_p10=_pct(days_to_pass, 10),
        days_to_pass_p90=_pct(days_to_pass, 90),
        days_to_bust_median=_pct(days_to_bust, 50),
        bust_reasons=dict(sorted(bust_reasons.items())),
    )
