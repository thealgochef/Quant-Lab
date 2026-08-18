"""Funnel deltas over ``DayFunnelRecord`` counters (DT §4.4).

Built from the UNION of both sides' counter vocabularies — a counter present
on one side only is reported (zero-filled), never silently dropped.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..search.identities import FrozenContract, ImmutableMap

__all__ = ["FunnelCounterDelta", "FunnelDeltaReport", "build_funnel_delta"]

#: Conditional stage-conversion ratios (numerator, denominator) over the
#: counter vocabulary; reported only when both counters exist on both sides.
_CONVERSIONS: tuple[tuple[str, str, str], ...] = (
    ("tap_to_setup", "setups_born", "htf_taps"),
    ("setup_to_lock", "parents_locked", "setups_born"),
    ("lock_to_armed", "opposing_armed", "parents_locked"),
    ("armed_to_execution", "executions_opened", "opposing_armed"),
)


class FunnelCounterDelta(FrozenContract):
    counter: str
    baseline: int
    challenger: int
    delta: int
    missing_from: str | None  # "baseline" | "challenger" | None


class FunnelDeltaReport(FrozenContract):
    totals: tuple[FunnelCounterDelta, ...]
    per_day: ImmutableMap[str, tuple[FunnelCounterDelta, ...]]
    conversions: ImmutableMap[str, tuple[float | None, float | None]]
    terminal_reason_deltas: tuple[FunnelCounterDelta, ...]
    vocabulary_mismatches: tuple[str, ...]


def _totals(day_funnels: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for counters in day_funnels.values():
        for key, value in counters.items():
            totals[key] = totals.get(key, 0) + int(value)
    return totals


def _delta_rows(
    baseline: Mapping[str, int], challenger: Mapping[str, int]
) -> tuple[FunnelCounterDelta, ...]:
    rows = []
    for counter in sorted(set(baseline) | set(challenger)):
        missing = None
        if counter not in baseline:
            missing = "baseline"
        elif counter not in challenger:
            missing = "challenger"
        left = int(baseline.get(counter, 0))
        right = int(challenger.get(counter, 0))
        rows.append(
            FunnelCounterDelta(
                counter=counter,
                baseline=left,
                challenger=right,
                delta=right - left,
                missing_from=missing,
            )
        )
    return tuple(rows)


def _conversion(totals: Mapping[str, int], numerator: str, denominator: str) -> float | None:
    if numerator not in totals or denominator not in totals or not totals[denominator]:
        return None
    return totals[numerator] / totals[denominator]


def build_funnel_delta(
    baseline_day_funnels: Mapping[str, Mapping[str, int]],
    challenger_day_funnels: Mapping[str, Mapping[str, int]],
    *,
    baseline_terminal_reasons: Mapping[str, int] | None = None,
    challenger_terminal_reasons: Mapping[str, int] | None = None,
) -> FunnelDeltaReport:
    baseline_totals = _totals(baseline_day_funnels)
    challenger_totals = _totals(challenger_day_funnels)
    per_day = {
        day: _delta_rows(
            baseline_day_funnels.get(day, {}), challenger_day_funnels.get(day, {})
        )
        for day in sorted(set(baseline_day_funnels) | set(challenger_day_funnels))
    }
    mismatches = tuple(
        sorted(set(baseline_totals).symmetric_difference(challenger_totals))
    )
    conversions = {
        name: (
            _conversion(baseline_totals, numerator, denominator),
            _conversion(challenger_totals, numerator, denominator),
        )
        for name, numerator, denominator in _CONVERSIONS
    }
    return FunnelDeltaReport(
        totals=_delta_rows(baseline_totals, challenger_totals),
        per_day=per_day,
        conversions=conversions,
        terminal_reason_deltas=_delta_rows(
            baseline_terminal_reasons or {}, challenger_terminal_reasons or {}
        ),
        vocabulary_mismatches=mismatches,
    )
