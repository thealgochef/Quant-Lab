"""Deterministic stress scenarios over the day-block stream (CS §5.4).

Each registered scenario is a pure, seeded transform of the ordered day
blocks; scenario identity (id + params + seed_offset) rides inside the
simulation identity. Nine scenarios are registered for V1.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    FrozenContract,
    ImmutableMap,
)
from alpha_lab.propsim.account import AccountTrade

__all__ = ["StressScenarioSpec", "STRESS_SCENARIOS_V1", "apply_stress_scenario"]


class StressScenarioSpec(FrozenContract):
    scenario_id: str
    name: str
    params: ImmutableMap[str, float]
    seed_offset: int


STRESS_SCENARIOS_V1: tuple[StressScenarioSpec, ...] = (
    StressScenarioSpec(
        scenario_id="stress_cost_x1_5_v1",
        name="Costs ×1.5",
        params={"extra_cost_points": 0.257},
        seed_offset=1,
    ),
    StressScenarioSpec(
        scenario_id="stress_cost_x2_v1",
        name="Costs ×2",
        params={"extra_cost_points": 0.514},
        seed_offset=2,
    ),
    StressScenarioSpec(
        scenario_id="stress_slippage_1t_v1",
        name="One extra tick of slippage per side",
        params={"extra_cost_points": 0.5},
        seed_offset=3,
    ),
    StressScenarioSpec(
        scenario_id="stress_drop_best_day_v1",
        name="Best day removed",
        params={"drop_best_days": 1},
        seed_offset=4,
    ),
    StressScenarioSpec(
        scenario_id="stress_drop_best_3_days_v1",
        name="Best three days removed",
        params={"drop_best_days": 3},
        seed_offset=5,
    ),
    StressScenarioSpec(
        scenario_id="stress_worst_day_first_v1",
        name="Worst day first",
        params={"worst_first": 1},
        seed_offset=6,
    ),
    StressScenarioSpec(
        scenario_id="stress_worst_3_days_first_v1",
        name="Worst three days first",
        params={"worst_first": 3},
        seed_offset=7,
    ),
    StressScenarioSpec(
        scenario_id="stress_shuffle_days_a_v1",
        name="Deterministic day shuffle A",
        params={"shuffle": 1},
        seed_offset=8,
    ),
    StressScenarioSpec(
        scenario_id="stress_shuffle_days_b_v1",
        name="Deterministic day shuffle B",
        params={"shuffle": 1},
        seed_offset=9,
    ),
)

_SCENARIOS_BY_ID = {spec.scenario_id: spec for spec in STRESS_SCENARIOS_V1}


def _day_pnl(block: Sequence[AccountTrade]) -> float:
    return sum(trade.points for trade in block)


def apply_stress_scenario(
    day_blocks: Sequence[tuple[date, Sequence[AccountTrade]]],
    scenario_id: str,
    *,
    base_seed: int,
) -> tuple[tuple[date, tuple[AccountTrade, ...]], ...]:
    """The scenario-transformed stream (pure; identical inputs → identical output)."""

    spec = _SCENARIOS_BY_ID.get(scenario_id)
    if spec is None:
        raise ValueError(f"unregistered stress scenario {scenario_id!r}")
    blocks = [(day, tuple(trades)) for day, trades in day_blocks]
    extra_cost = float(spec.params.get("extra_cost_points", 0.0))
    if extra_cost:
        blocks = [
            (
                day,
                tuple(
                    AccountTrade(
                        **{
                            **trade.__dict__,
                            "points": trade.points - extra_cost,
                        }
                    )
                    for trade in trades
                ),
            )
            for day, trades in blocks
        ]
    drop_best = int(spec.params.get("drop_best_days", 0))
    if drop_best:
        ranked = sorted(blocks, key=lambda item: _day_pnl(item[1]), reverse=True)
        dropped = {id(item) for item in ranked[:drop_best]}
        blocks = [item for item in blocks if id(item) not in dropped]
    worst_first = int(spec.params.get("worst_first", 0))
    if worst_first:
        ranked = sorted(blocks, key=lambda item: _day_pnl(item[1]))
        moved = ranked[:worst_first]
        moved_ids = {id(item) for item in moved}
        blocks = moved + [item for item in blocks if id(item) not in moved_ids]
    if int(spec.params.get("shuffle", 0)):
        rng = np.random.default_rng(base_seed + spec.seed_offset)
        order = rng.permutation(len(blocks))
        blocks = [blocks[int(index)] for index in order]
    return tuple(blocks)
