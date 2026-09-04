"""The reviewer verdict vocabulary (UI-2; owner Q3; plan F-06).

The ledger keys of ``ifvg_visual_review_v1`` are PRESERVED; the owner-approved
labels map one-to-one onto them; ``not_applicable`` is the one additive key
(admitted by the existing validator — no v2 schema). ``Unreviewed`` is an
UNSAVED UI state only: opening a case creates no ledger row and preselects
no verdict; nothing persists without the explicit Save Review action.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from ..visual_review_store import REVIEW_VERDICTS

__all__ = [
    "REVIEW_SAVE_STATES",
    "UNREVIEWED",
    "UNREVIEWED_LABEL",
    "VERDICT_DEFINITIONS",
    "VERDICT_LABELS",
    "label_for_verdict",
    "verdict_for_label",
    "verdict_options",
]

#: The UI-only sentinel — never a ledger value.
UNREVIEWED = "unreviewed"
UNREVIEWED_LABEL = "Unreviewed"

#: Owner Q3 labels, keyed by the persisted ledger value, in ledger order.
VERDICT_LABELS: Mapping[str, str] = MappingProxyType(
    {
        "correct": "Correct",
        "incorrect": "Incorrect",
        "questionable": "Needs investigation",
        "insufficient_evidence": "Unclear",
        "not_applicable": "Not applicable",
    }
)

VERDICT_DEFINITIONS: Mapping[str, str] = MappingProxyType(
    {
        "correct": "the persisted evidence matches the strategy specification for this case",
        "incorrect": "the persisted evidence contradicts the strategy specification",
        "questionable": (
            "a possible problem or inconsistency that warrants follow-up, not yet asserting "
            "incorrectness"
        ),
        "insufficient_evidence": (
            "evidence incomplete, ambiguous or insufficient to judge correctness"
        ),
        "not_applicable": "the review question does not apply to this case",
    }
)

#: The visible persistence states of the review form.
REVIEW_SAVE_STATES: tuple[str, ...] = ("unsaved", "saving", "saved")

if tuple(VERDICT_LABELS) != tuple(REVIEW_VERDICTS):  # pragma: no cover - wiring guard
    raise RuntimeError("the review vocabulary must mirror the ledger verdicts exactly")

_KEY_BY_LABEL = {label: key for key, label in VERDICT_LABELS.items()}


def verdict_options() -> tuple[str, ...]:
    """The selectbox options: Unreviewed FIRST (nothing preselected), then the
    owner-approved labels in ledger order."""

    return (UNREVIEWED_LABEL, *VERDICT_LABELS.values())


def verdict_for_label(label: str) -> str | None:
    """The ledger key of a label; ``None`` for Unreviewed (nothing to persist)."""

    if label == UNREVIEWED_LABEL:
        return None
    try:
        return _KEY_BY_LABEL[label]
    except KeyError as error:
        raise ValueError(f"unknown verdict label {label!r}") from error


def label_for_verdict(key: str | None) -> str:
    """The label of a ledger key; an unknown (legacy) value renders as-is and
    ``None`` renders as Unreviewed."""

    if key is None:
        return UNREVIEWED_LABEL
    return VERDICT_LABELS.get(str(key), str(key))
