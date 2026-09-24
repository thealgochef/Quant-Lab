"""Mandatory integrity check: a study's minimum independent trading days must be
reachable within the dates it evaluates (repair R5).

The strategy metric ``independent_days`` counts the distinct trading days among
a configuration's executed trades after warmup rows are excluded
(``strategy_metrics.research_trades`` / ``compute_strategy_metrics``), and the
gate passes when that count is at least the threshold (``gates.py``). Its
largest possible value is therefore the number of distinct evaluated dates: the
replay dates minus the warmup dates. A threshold above that count fails every
configuration by construction.

This module only reports that contradiction. It never clamps, rounds or
substitutes a saved threshold, never changes gate evaluation or historical gate
outcomes, and deliberately lives outside the pydantic contracts: stored charters
and approvals are re-validated when they are loaded, so a historic charter that
carries such a threshold stays loadable and inspectable while every new launch
path (satisfiability, approval request, charter validation, worker entry and
child boundary) refuses it.

The check applies to the development access policy
(``development_explicit_dates_before_path_v2``). The five-day verification
policy is exempt: it runs verification control-flow gates, not research
thresholds. Messages are plain English for the study screens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from ..development_access import FROZEN_WARMUP_DATES
from .charter import ResolvedStrategyGateThresholds

__all__ = [
    "CHECKED_ACCESS_POLICY_IDS",
    "DEFAULT_MIN_INDEPENDENT_DAYS",
    "DEVELOPMENT_ACCESS_POLICY_ID",
    "VERIFICATION_ACCESS_POLICY_ID",
    "DayThresholdCheck",
    "charter_day_threshold_check",
    "check_day_threshold",
    "draft_day_threshold_check",
    "evaluated_date_count",
]

DEVELOPMENT_ACCESS_POLICY_ID = "development_explicit_dates_before_path_v2"
VERIFICATION_ACCESS_POLICY_ID = "verification_fixed_allowlist_max5_v1"
#: Access policies whose research gates are checked against the evaluated dates.
CHECKED_ACCESS_POLICY_IDS: frozenset[str] = frozenset({DEVELOPMENT_ACCESS_POLICY_ID})
#: The charter default applied when a draft leaves the threshold unset.
DEFAULT_MIN_INDEPENDENT_DAYS = ResolvedStrategyGateThresholds.model_fields[
    "min_independent_days"
].default


@dataclass(frozen=True)
class DayThresholdCheck:
    """The outcome of comparing the threshold with the evaluated dates."""

    applies: bool
    threshold: Any
    evaluated_dates: int
    warmup_dates: int
    problem: str | None

    @property
    def passed(self) -> bool:
        return self.problem is None

    @property
    def detail(self) -> str:
        """One plain-English sentence for a review table."""

        if self.problem is not None:
            return self.problem
        if not self.applies:
            return "The five-day verification check does not use this research threshold."
        if not _is_number(self.threshold):
            return "The minimum independent trading days is checked once it is a number."
        return (
            f"Minimum independent trading days is {_format_number(self.threshold)}; this "
            f"study evaluates {_days(self.evaluated_dates, 'trading day')}"
            f"{_warmup_note(self.warmup_dates)}."
        )


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _format_number(value: Any) -> str:
    number = float(value)
    return f"{int(number):,}" if number.is_integer() else f"{number:,}"


def _days(count: int, noun: str) -> str:
    return f"{count:,} {noun}" + ("" if count == 1 else "s")


def _warmup_note(warmup: int) -> str:
    return f" (after {_days(warmup, 'warmup day')})" if warmup else ""


def evaluated_date_count(replay_dates: Sequence[str], warmup_dates: Sequence[str]) -> int:
    """Distinct replay dates that are not warmup dates."""

    warmup = {str(day) for day in warmup_dates}
    return len({str(day) for day in replay_dates} - warmup)


def check_day_threshold(
    *,
    min_independent_days: Any,
    replay_dates: Sequence[str],
    warmup_dates: Sequence[str],
    access_policy_id: str,
) -> DayThresholdCheck:
    """Compare one threshold with the evaluated-date count of one date policy.

    ``replay_dates`` may include the warmup dates (a charter) or list only the
    evidence dates (a draft): warmup dates are removed either way. A value that
    is not a number is left to the thresholds step, which refuses it.
    """

    warmup = len({str(day) for day in warmup_dates})
    evaluated = evaluated_date_count(replay_dates, warmup_dates)
    if str(access_policy_id) not in CHECKED_ACCESS_POLICY_IDS:
        return DayThresholdCheck(False, min_independent_days, evaluated, warmup, None)
    if not _is_number(min_independent_days) or not float(min_independent_days) > evaluated:
        return DayThresholdCheck(True, min_independent_days, evaluated, warmup, None)
    shown = _format_number(min_independent_days)
    if evaluated == 0:
        problem = (
            f"Minimum independent trading days is {shown}, but no evaluation dates are "
            "selected yet (warmup days are not counted), so no configuration could pass. "
            "Choose the study's trading days first; this threshold must then be at most "
            "the number of evaluated days. The saved value has not been changed."
        )
    else:
        problem = (
            f"Minimum independent trading days is {shown}, but this study evaluates only "
            f"{_days(evaluated, 'trading day')}{_warmup_note(warmup)}. No configuration "
            f"can pass. Change this threshold to at most {evaluated:,} before running; "
            "the saved value has not been changed."
        )
    return DayThresholdCheck(True, min_independent_days, evaluated, warmup, problem)


def _get(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        return source[name]
    return getattr(source, name)


def charter_day_threshold_check(charter: Any) -> DayThresholdCheck:
    """The check for a charter payload or its JSON intent mapping."""

    date_policy = _get(charter, "date_policy")
    gates = _get(_get(charter, "objective_policy"), "feasibility_gates")
    return check_day_threshold(
        min_independent_days=_get(gates, "min_independent_days"),
        replay_dates=tuple(_get(date_policy, "replay_dates")),
        warmup_dates=tuple(_get(date_policy, "warmup_dates")),
        access_policy_id=str(_get(date_policy, "access_policy_id")),
    )


def draft_day_threshold_check(
    steps: Mapping[str, Any],
    *,
    run_scope: str | None = None,
    benchmarks: Mapping[str, Any] | None = None,
) -> DayThresholdCheck:
    """The check for a saved draft's steps, mirroring charter assembly: the
    verification scope uses the exempt five-day policy; every other scope
    replays the warmup prefix (the draft's saved warmup dates, else the frozen
    ten-date prefix) followed by the selected evidence dates, and the warmup
    dates are never counted.

    ``benchmarks`` overrides the draft's benchmarks step (a flow that skips the
    step contributes nothing, so the charter default applies). Reading only:
    the draft is never modified.
    """

    validation = steps.get("validation") or {}
    scope = run_scope if run_scope is not None else validation.get("run_scope")
    gates = (
        (benchmarks if benchmarks is not None else steps.get("benchmarks") or {}).get(
            "strategy_gates"
        )
        or {}
    )
    threshold = gates.get("min_independent_days", DEFAULT_MIN_INDEPENDENT_DAYS)
    evidence = tuple(str(day) for day in validation.get("real_dates") or ())
    if scope == "verification_5d":
        return check_day_threshold(
            min_independent_days=threshold,
            replay_dates=evidence,
            warmup_dates=(),
            access_policy_id=VERIFICATION_ACCESS_POLICY_ID,
        )
    warmup = tuple(str(day) for day in validation.get("warmup_dates") or FROZEN_WARMUP_DATES)
    return check_day_threshold(
        min_independent_days=threshold,
        replay_dates=(*warmup, *evidence),
        warmup_dates=warmup,
        access_policy_id=DEVELOPMENT_ACCESS_POLICY_ID,
    )
