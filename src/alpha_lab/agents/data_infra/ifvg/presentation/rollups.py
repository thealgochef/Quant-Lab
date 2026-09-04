"""Deterministic section roll-ups (UI-3; plan §6.3) — pure.

A ``SectionRollup`` summarises the ``MetricReading``s of one section with
ONE status, one sentence, the main reason and the next thing to inspect.
The rule is fixed and order-sensitive:

1. FAIL — any blocking child fails (or is corrupt);
2. BLOCKED — required evidence is missing / unavailable, or an authorization
   is missing;
3. INCONCLUSIVE — no failure, but a reading is inconclusive or evidence is
   unavailable;
4. WARNING — usable with cautions: a warning reading, or a pass under a
   proposed (unratified) threshold;
5. PASS — every gated reading passes (descriptive readings do not count);
6. INFORMATIONAL — no gate applies (every reading is descriptive);
7. UNAVAILABLE — no reading at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType

from ..search.identities import FrozenContract
from .metric_registry import MetricReading
from .status_vocabulary import STATUS_SPECS, UiStatus, status_chip

__all__ = ["SECTION_LABELS", "RollupSection", "SectionRollup", "rollup_section"]


class RollupSection(StrEnum):
    DATA_INTEGRITY = "data_integrity"
    STRATEGY_QUALITY = "strategy_quality"
    PROBABILITY_SKILL = "probability_skill"
    CALIBRATION = "calibration"
    STABILITY = "stability"
    PROP_FEASIBILITY = "prop_feasibility"
    ROBUSTNESS = "robustness"
    AUTHORIZATION_READINESS = "authorization_readiness"
    CAPACITY_AND_PERFORMANCE = "capacity_and_performance"


SECTION_LABELS: Mapping[RollupSection, str] = MappingProxyType(
    {
        RollupSection.DATA_INTEGRITY: "Data integrity",
        RollupSection.STRATEGY_QUALITY: "Strategy quality",
        RollupSection.PROBABILITY_SKILL: "Probability skill",
        RollupSection.CALIBRATION: "Calibration",
        RollupSection.STABILITY: "Stability",
        RollupSection.PROP_FEASIBILITY: "Prop feasibility",
        RollupSection.ROBUSTNESS: "Robustness",
        RollupSection.AUTHORIZATION_READINESS: "Authorization readiness",
        RollupSection.CAPACITY_AND_PERFORMANCE: "Capacity and performance",
    }
)


class SectionRollup(FrozenContract):
    section: RollupSection
    status: UiStatus
    chip: str
    sentence: str
    main_reason: str
    inspect_next: str
    reading_count: int
    passed_count: int
    failed_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]


_DEFAULT_INSPECT: Mapping[UiStatus, str] = MappingProxyType(
    {
        UiStatus.FAIL: "the failing metric's row and the gate it is read against",
        UiStatus.BLOCKED: "the missing evidence or authorization named in the reason",
        UiStatus.INCONCLUSIVE: "the sample size and the interval behind the inconclusive reading",
        UiStatus.WARNING: "the proposed thresholds and their ratification status",
        UiStatus.PASS: "the metric rows under Research details",
        UiStatus.INFORMATIONAL: "the descriptive rows under Research details",
        UiStatus.UNAVAILABLE: "the persisted report under Technical identity & audit",
    }
)

_FAILING = frozenset({UiStatus.FAIL, UiStatus.CORRUPT})
_MISSING = frozenset({UiStatus.UNAVAILABLE, UiStatus.BLOCKED})
_UNDECIDED = frozenset({UiStatus.INCONCLUSIVE, UiStatus.UNAVAILABLE})


def rollup_section(
    section: RollupSection | str,
    readings: Sequence[MetricReading],
    *,
    required_keys: Sequence[str] = (),
    missing_authorization: str | None = None,
    inspect_next: str | None = None,
) -> SectionRollup:
    """The deterministic roll-up of ``readings`` for ``section``."""

    section = RollupSection(section)
    label = SECTION_LABELS[section]
    ordered = tuple(readings)
    by_key = {reading.technical_key: reading for reading in ordered}
    failing = tuple(
        reading
        for reading in ordered
        if reading.status in _FAILING and (reading.blocking or reading.status is UiStatus.CORRUPT)
    )
    missing = tuple(
        key
        for key in required_keys
        if key not in by_key or by_key[key].status in _MISSING
    )
    passed = tuple(reading for reading in ordered if reading.status is UiStatus.PASS)

    if failing:
        first = failing[0]
        status = UiStatus.FAIL
        reason = f"{first.human_name} — {first.interpretation}"
    elif missing_authorization or missing:
        status = UiStatus.BLOCKED
        if missing_authorization:
            reason = missing_authorization
        else:
            names = ", ".join(
                by_key[key].human_name if key in by_key else key for key in missing
            )
            reason = f"required evidence missing or unavailable: {names}"
    else:
        undecided = tuple(reading for reading in ordered if reading.status in _UNDECIDED)
        cautions = tuple(
            reading
            for reading in ordered
            if reading.status is UiStatus.WARNING
            or (reading.status is UiStatus.PASS and reading.proposed)
        )
        if undecided:
            first = undecided[0]
            status = UiStatus.INCONCLUSIVE
            reason = f"{first.human_name} — {first.interpretation}"
        elif cautions:
            first = cautions[0]
            status = UiStatus.WARNING
            reason = (
                f"{first.human_name} passes under a proposed (unratified) threshold"
                if first.status is UiStatus.PASS
                else f"{first.human_name} — {first.interpretation}"
            )
        elif passed:
            status = UiStatus.PASS
            reason = f"every gated metric passes ({len(passed)} of {len(ordered)} readings gated)"
        elif ordered:
            status = UiStatus.INFORMATIONAL
            reason = "no gate applies to this section; the values are descriptive"
        else:
            status = UiStatus.UNAVAILABLE
            reason = "no evidence was persisted for this section"
    sentence = f"{label}: {STATUS_SPECS[status].label} — {reason}"
    return SectionRollup(
        section=section,
        status=status,
        chip=status_chip(status),
        sentence=sentence,
        main_reason=reason,
        inspect_next=inspect_next or _DEFAULT_INSPECT.get(status, "the persisted evidence"),
        reading_count=len(ordered),
        passed_count=len(passed),
        failed_keys=tuple(reading.technical_key for reading in failing),
        missing_keys=missing,
    )
