"""Phase C — one day's capture: SC ``run_day`` over cached artifacts + row flattening.

QL computes NOTHING here: bars/levels come from the Phase-A artifact, the FSM
is SC's ``run_day``, and this module only groups bars by timeframe, hands the
level timeline in as ``levels_for``, and flattens the typed SC emissions into
one wide DataFrame (``kind`` column + unioned per-kind fields; nested
dataclasses flattened with ``<field>_`` prefixes; enums to values; tuples to
JSON strings).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime
from enum import Enum

import pandas as pd

from strategy_core.strategies.ifvg_smc.replay import run_day
from strategy_core.strategies.ifvg_smc.state import IfvgDaySeed, seed_hash

from .config import IfvgCaptureConfig
from .day_artifacts import DayArtifacts, levels_for_from_frame

__all__ = ["CaptureDayResult", "capture_single_date", "flatten_emissions"]


@dataclass(frozen=True)
class CaptureDayResult:
    date_str: str
    rows: pd.DataFrame
    funnel: dict[str, int]
    entering_seed_hash: str | None
    end_seed: IfvgDaySeed


def _flat(value: object, prefix: str, out: dict) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        for key, sub in asdict(value).items():  # asdict recurses; re-flatten dicts
            _flat(sub, f"{prefix}{key}_", out)
        return
    if isinstance(value, dict):
        for key, sub in value.items():
            _flat(sub, f"{prefix}{key}_", out)
        return
    name = prefix[:-1]  # trim trailing underscore
    if isinstance(value, Enum):
        out[name] = value.value
    elif isinstance(value, (tuple, list)):
        out[name] = json.dumps([v.value if isinstance(v, Enum) else v for v in value])
    else:
        out[name] = value


def flatten_emissions(emissions, *, entering_seed_hash: str | None) -> pd.DataFrame:
    rows = []
    for emission in emissions:
        if emission.kind == "funnel":
            continue  # tier-1 counters ride CaptureDayResult.funnel, not rows
        row: dict = {"kind": emission.kind, "entering_seed_hash": entering_seed_hash}
        _flat(emission.record, "", row)
        rows.append(row)
    return pd.DataFrame(rows)


def capture_single_date(
    date_str: str,
    cfg: IfvgCaptureConfig,
    *,
    artifacts: DayArtifacts,
    seed: IfvgDaySeed | None,
) -> CaptureDayResult:
    bars_by_tf: dict[int, list] = {}
    for bar in artifacts.bars:
        bars_by_tf.setdefault(bar.timeframe_ticks, []).append(bar)
    entering = seed_hash(seed) if seed is not None else None
    result = run_day(
        bars_by_tf,
        section=cfg.section,
        seed=seed,
        trading_day=date.fromisoformat(date_str),
        tick_size=cfg.tick_size,
        levels_for=levels_for_from_frame(artifacts.level_timeline),
    )
    rows = flatten_emissions(result.emissions, entering_seed_hash=entering)
    return CaptureDayResult(
        date_str=date_str,
        rows=rows,
        funnel=dict(result.funnel.counters),
        entering_seed_hash=entering,
        end_seed=result.end_seed,
    )
