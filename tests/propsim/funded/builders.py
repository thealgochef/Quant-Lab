"""Synthetic builders for the funded-payout engine tests.

Every fixture here is SYNTHETIC: hand-made prices on hand-made dates. Passing
these tests says nothing about historical price coverage or real payouts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alpha_lab.propsim.funded.campaign import CampaignInputs, TradingDay
from alpha_lab.propsim.funded.clock import (
    ELAPSED_48_HOURS,
    TWO_BUSINESS_DAYS_FED_1600,
    to_ns,
)
from alpha_lab.propsim.funded.paths import (
    OBS_EXIT_TOUCH,
    OBS_PRINT,
    ExecutionPath,
    StrategyExecution,
)
from alpha_lab.propsim.funded.profiles import (
    MYFUNDEDFUTURES_PROFILE,
    TAKEPROFITTRADER_PROFILE,
)

BASE = 100_000  # ticks; the absolute level is irrelevant, offsets matter
TICK_USD = 5  # one mini = $5 per tick


def ticks_for(dollars: float) -> int:
    value = round(dollars / TICK_USD)
    assert value * TICK_USD == dollars, "use whole-tick dollar moves in fixtures"
    return value


def execution(trade_id: str, *, entry: str, exit: str, move_usd: float, day: str,
              reason: str = "target", stop_usd: float = 1_000, direction: str = "long",
              entry_ticks: int = BASE) -> StrategyExecution:
    sign = 1 if direction == "long" else -1
    exit_ticks = entry_ticks + sign * ticks_for(move_usd)
    return StrategyExecution(
        trade_id=trade_id, trading_day=day, direction=direction,
        entry_ts_utc=entry, entry_ticks=entry_ticks,
        stop_ticks=entry_ticks - sign * ticks_for(stop_usd),
        target_ticks=exit_ticks if reason == "target" else None,
        exit_ts_utc=exit, exit_ticks=exit_ticks, exit_reason=reason,
        is_warmup=False, entry_chart="5-minute chart",
    )


def path(ex: StrategyExecution, points: Sequence[tuple[str, float]], *,
         touch: bool = True) -> ExecutionPath:
    """Ordered PRINT path: (utc time, open profit in USD per mini) points.

    With ``touch`` the strategy exit is appended as the exit-touch print at
    the execution's exit time and exit price.
    """

    sign = ex.sign
    ts = [to_ns(t) for t, _ in points]
    px = [ex.entry_ticks + sign * ticks_for(usd) for _, usd in points]
    kinds = [OBS_PRINT] * len(points)
    if touch:
        ts.append(to_ns(ex.exit_ts_utc))
        px.append(ex.exit_ticks)
        kinds.append(OBS_EXIT_TOUCH)
    return ExecutionPath(
        trade_id=ex.trade_id, fidelity="ordered_trade_prints",
        ts_ns=np.asarray(ts, dtype=np.int64), price_ticks=np.asarray(px, dtype=np.int64),
        continuous=np.zeros(len(ts), dtype=bool), kind=np.asarray(kinds, dtype=np.int8),
        strategy_exit_ns=to_ns(ex.exit_ts_utc),
    )


def weekday_days(dates: Sequence[str]) -> tuple[TradingDay, ...]:
    """Trading days closing 4:00 PM CST (22:00 UTC) and reopening 5:00 PM."""

    out = []
    for day in dates:
        out.append(TradingDay(
            trading_day=day, day_end_ns=to_ns(f"{day}T22:00:00Z"),
            reopen_ns=to_ns(f"{day}T23:00:00Z"), deadline_ns=to_ns(f"{day}T21:55:00Z"),
        ))
    return tuple(out)


def make_inputs(executions, paths, *, days, start: str, cutoff: str, quantity: int = 1,
                cost_per_side_cents: int = 0, processing=ELAPSED_48_HOURS,
                profiles=(TAKEPROFITTRADER_PROFILE, MYFUNDEDFUTURES_PROFILE),
                instrument: str = "mini") -> CampaignInputs:
    return CampaignInputs(
        executions=tuple(executions), paths={p.trade_id: p for p in paths},
        trading_days=weekday_days(days) if isinstance(days[0], str) else tuple(days),
        start_ns=to_ns(start), cutoff_ns=to_ns(cutoff), instrument=instrument,
        quantity=quantity, cost_per_side_cents=cost_per_side_cents,
        processing=processing, profiles=tuple(profiles),
    )



def synthetic_sample_result(*, validated: bool = True) -> dict:
    """A clearly labeled SYNTHETIC result exercising every screen/export state.

    Covers: payouts received, a large payout then a failure, accounts lost
    before any payout, vacancies waiting for credits, a growth purchase, a
    request still processing at the cutoff and zero-payout months.
    """

    from alpha_lab.propsim.funded.campaign import resumed_equivalence, run_campaign
    from alpha_lab.propsim.funded.plan import (
        FUNDED_QUESTION,
        MATERIAL_LIMITATIONS,
        PILOT_OWNER_DECISIONS,
    )
    from alpha_lab.propsim.funded.result import build_result, validate_result

    days = ["2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16", "2026-01-20",
            "2026-01-21", "2026-02-10", "2026-02-11", "2026-03-10", "2026-03-11"]
    specs = [
        ("s1", "2026-01-13", 5_500, "target", [(1_000,), (-300,), (4_000,)]),
        ("s2", "2026-01-15", -700, "stop", [(-200,)]),
        ("s3", "2026-01-20", 1_200, "target", [(-900,), (600,)]),
        ("s4", "2026-01-21", -2_500, "stop", [(-1_500,), (-2_050,)]),
        ("s5", "2026-02-10", 2_800, "target", [(1_000,), (2_000,)]),
        ("s6", "2026-02-11", -400, "stop", [(-100,)]),
        ("s7", "2026-03-11", 2_700, "target", [(900,)]),
    ]
    executions, paths = [], []
    for trade_id, day, move, reason, pts in specs:
        ex = execution(trade_id, entry=f"{day}T15:00:00Z", exit=f"{day}T16:00:00Z",
                       move_usd=move, day=day, reason=reason,
                       stop_usd=max(abs(move), 500) if reason == "stop" else 1_500)
        points = [(f"{day}T15:{10 + 5 * i:02d}:00Z", usd) for i, (usd,) in enumerate(pts)]
        executions.append(ex)
        paths.append(path(ex, points))
    inputs = make_inputs(executions, paths, days=days, start="2026-01-12T23:00:00Z",
                         cutoff="2026-03-11T22:00:00Z", cost_per_side_cents=514,
                         processing=TWO_BUSINESS_DAYS_FED_1600)
    instances = run_campaign(inputs)
    result = build_result(
        inputs, instances,
        run_identity={"label": "SYNTHETIC ENGINEERING SAMPLE — not historical"},
        price_evidence={"policy": "synthetic_fixture", "trades_with_ordered_prints": 7,
                        "trades_with_minute_approximation": 0},
        context={
            "funded_plan_id": "0" * 64, "purpose": "engineering_sample",
            "question": FUNDED_QUESTION,
            "source": {"description": "Synthetic fixture executions (not a study)",
                       "profile_id": "SYNTHETIC"},
            "owner_decisions": [d.model_dump(mode="json") for d in PILOT_OWNER_DECISIONS],
            "limitations": list(MATERIAL_LIMITATIONS),
        },
    )
    if validated:
        result["validation"] = validate_result(result, instances,
                                               resumed_equivalence(inputs))
    return result
