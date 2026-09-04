"""The one UI status vocabulary (UI-1; plan §6.1).

Thirteen typed statuses, each a glyph + word + color token + meaning; color
is never the sole carrier. The existing persisted vocabularies
(``StudyStatusKey``, the pipeline stage statuses, the heatmap glyph classes,
the empty-state keys) MAP onto ``UiStatus`` through additive adapters — the
old enums stay the persisted / contract vocabulary.

Interpretation rules (plan §6.2 excerpt, the parts every screen needs now):

* a gate value is PASS only when it was EVALUATED and true; ``None`` /
  unevaluated / default-only evidence is UNAVAILABLE, never green;
* a reference comparison follows the registered direction; without a
  reference the value is INFORMATIONAL; a ``target`` direction reports the
  distance only (no good / bad); ``descriptive`` is always INFORMATIONAL;
* evidence kinds (missing, not evaluated, corrupt, not applicable, not
  selected, proposed, superseded, …) map to their non-green states.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from ..search.identities import FrozenContract
from ..study_status import EmptyStateKey, StudyStatusKey

__all__ = [
    "COLOR_TOKENS",
    "STATUS_SPECS",
    "StatusSpec",
    "UiStatus",
    "status_chip",
    "status_for_evidence",
    "status_from_gate",
    "status_from_reference",
    "ui_status_for_empty_state",
    "ui_status_for_heatmap_class",
    "ui_status_for_pipeline_stage_status",
    "ui_status_for_study_status",
]


class UiStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    WARNING = "warning"
    INCONCLUSIVE = "inconclusive"
    INFORMATIONAL = "informational"
    NOT_APPLICABLE = "not_applicable"
    NOT_SELECTED = "not_selected"
    UNAVAILABLE = "unavailable"
    CORRUPT = "corrupt"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    SUPERSEDED = "superseded"


#: green PASS · red FAIL · amber WARNING / INCONCLUSIVE / proposed · blue
#: INFORMATIONAL · gray NOT_APPLICABLE / NOT_SELECTED / UNAVAILABLE · purple
#: research-only / experimental (plan §6.1).
COLOR_TOKENS: frozenset[str] = frozenset({"green", "red", "amber", "blue", "gray", "purple"})


class StatusSpec(FrozenContract):
    status: UiStatus
    label: str
    glyph: str
    color_token: str
    meaning: str
    blocks_next_action: bool


def _spec(
    status: UiStatus,
    label: str,
    glyph: str,
    color_token: str,
    meaning: str,
    *,
    blocks: bool = False,
) -> StatusSpec:
    return StatusSpec(
        status=status,
        label=label,
        glyph=glyph,
        color_token=color_token,
        meaning=meaning,
        blocks_next_action=blocks,
    )


STATUS_SPECS: Mapping[UiStatus, StatusSpec] = MappingProxyType(
    {
        UiStatus.PASS: _spec(
            UiStatus.PASS,
            "Pass",
            "✓",
            "green",
            "The evaluated gate or reference was met by the observed evidence.",
        ),
        UiStatus.FAIL: _spec(
            UiStatus.FAIL,
            "Fail",
            "✕",
            "red",
            "The evaluated gate or reference was not met; the next action is blocked.",
            blocks=True,
        ),
        UiStatus.BLOCKED: _spec(
            UiStatus.BLOCKED,
            "Blocked",
            "⛔",
            "red",
            "Required evidence, authorization or capability is absent; nothing here proceeds.",
            blocks=True,
        ),
        UiStatus.WARNING: _spec(
            UiStatus.WARNING,
            "Warning",
            "▲",
            "amber",
            "Usable with cautions — a proposed (unratified) threshold or a degraded input.",
        ),
        UiStatus.INCONCLUSIVE: _spec(
            UiStatus.INCONCLUSIVE,
            "Inconclusive",
            "?",
            "amber",
            "No failure, but the evidence is insufficient or undefined to conclude either way.",
        ),
        UiStatus.INFORMATIONAL: _spec(
            UiStatus.INFORMATIONAL,
            "Informational",
            "ℹ",
            "blue",
            "A descriptive value with no gate applied; it neither passes nor fails.",
        ),
        UiStatus.NOT_APPLICABLE: _spec(
            UiStatus.NOT_APPLICABLE,
            "Not applicable",
            "—",
            "gray",
            "The question does not apply to this case or selection.",
        ),
        UiStatus.NOT_SELECTED: _spec(
            UiStatus.NOT_SELECTED,
            "Not selected",
            "○",
            "gray",
            "Nothing is selected yet; choose an item to evaluate it.",
        ),
        UiStatus.UNAVAILABLE: _spec(
            UiStatus.UNAVAILABLE,
            "Unavailable",
            "∅",
            "gray",
            "The evidence is missing or was never evaluated; it is not shown as passing.",
        ),
        UiStatus.CORRUPT: _spec(
            UiStatus.CORRUPT,
            "Corrupt",
            "⚠",
            "red",
            "The artifact exists but failed verification; nothing derived from it is trusted.",
            blocks=True,
        ),
        UiStatus.IN_PROGRESS: _spec(
            UiStatus.IN_PROGRESS,
            "In progress",
            "▶",
            "blue",
            "Work is running or checkpointed; results are partial until it completes.",
        ),
        UiStatus.COMPLETE: _spec(
            UiStatus.COMPLETE,
            "Complete",
            "■",
            "green",
            "The work finished (or was verified-reused); its evidence is final.",
        ),
        UiStatus.SUPERSEDED: _spec(
            UiStatus.SUPERSEDED,
            "Superseded",
            "↻",
            "purple",
            "A newer artifact or decision replaced this one; it is kept read-only for audit.",
        ),
    }
)


def status_chip(status: UiStatus | str) -> str:
    """Glyph + word — the only chip form (color never carries meaning alone)."""

    spec = STATUS_SPECS[UiStatus(status)]
    return f"{spec.glyph} {spec.label}"


def status_from_gate(value: object, *, evaluated: bool = True) -> UiStatus:
    """PASS only for an EVALUATED ``True``; ``False`` FAIL; anything else UNAVAILABLE."""

    if not evaluated or not isinstance(value, bool):
        return UiStatus.UNAVAILABLE
    return UiStatus.PASS if value else UiStatus.FAIL


_Direction = Literal["higher_better", "lower_better", "target", "descriptive"]
_DIRECTIONS: frozenset[str] = frozenset({"higher_better", "lower_better", "target", "descriptive"})


def status_from_reference(
    value: float | int | None,
    *,
    direction: _Direction | str,
    reference: float | int | None = None,
) -> UiStatus:
    """Direction-aware reference status; ``target`` and ``descriptive`` are
    informational (distance shown by the caller, never a good / bad label)."""

    if direction not in _DIRECTIONS:
        raise ValueError(f"unregistered metric direction {direction!r}")
    if value is None or isinstance(value, bool):
        return UiStatus.UNAVAILABLE
    if direction in ("target", "descriptive") or reference is None:
        return UiStatus.INFORMATIONAL
    observed = float(value)
    bound = float(reference)
    if direction == "higher_better":
        return UiStatus.PASS if observed >= bound else UiStatus.FAIL
    return UiStatus.PASS if observed <= bound else UiStatus.FAIL


_EVIDENCE_KINDS: Mapping[str, UiStatus] = MappingProxyType(
    {
        "missing": UiStatus.UNAVAILABLE,
        "not_evaluated": UiStatus.UNAVAILABLE,
        "unavailable": UiStatus.UNAVAILABLE,
        "corrupt": UiStatus.CORRUPT,
        "not_applicable": UiStatus.NOT_APPLICABLE,
        "not_selected": UiStatus.NOT_SELECTED,
        "in_progress": UiStatus.IN_PROGRESS,
        "complete": UiStatus.COMPLETE,
        "superseded": UiStatus.SUPERSEDED,
        "proposed": UiStatus.WARNING,
        "blocked": UiStatus.BLOCKED,
        "inconclusive": UiStatus.INCONCLUSIVE,
        "informational": UiStatus.INFORMATIONAL,
        "pass": UiStatus.PASS,
        "fail": UiStatus.FAIL,
    }
)


def status_for_evidence(kind: str) -> UiStatus:
    try:
        return _EVIDENCE_KINDS[str(kind)]
    except KeyError:
        raise ValueError(f"unregistered evidence kind {kind!r}") from None


# ── additive adapters over the persisted vocabularies ───────────────────────

_STUDY_STATUS: Mapping[StudyStatusKey, UiStatus] = MappingProxyType(
    {
        StudyStatusKey.DRAFT: UiStatus.INFORMATIONAL,
        StudyStatusKey.FROZEN: UiStatus.COMPLETE,
        StudyStatusKey.QUEUED: UiStatus.IN_PROGRESS,
        StudyStatusKey.RUNNING: UiStatus.IN_PROGRESS,
        StudyStatusKey.REPLAY_FAILED: UiStatus.FAIL,
        StudyStatusKey.STRATEGY_REJECTED: UiStatus.FAIL,
        StudyStatusKey.PROP_REJECTED: UiStatus.FAIL,
        StudyStatusKey.ROBUST_FINALIST: UiStatus.PASS,
        StudyStatusKey.SELECTED_REPRESENTATIVE: UiStatus.INFORMATIONAL,
        StudyStatusKey.SUPERSEDED: UiStatus.SUPERSEDED,
        StudyStatusKey.BLOCKED: UiStatus.BLOCKED,
    }
)


def ui_status_for_study_status(key: StudyStatusKey | str) -> UiStatus:
    return _STUDY_STATUS[StudyStatusKey(key)]


_STAGE_STATUS: Mapping[str, UiStatus] = MappingProxyType(
    {
        "pending": UiStatus.NOT_SELECTED,
        "queued": UiStatus.IN_PROGRESS,
        "running": UiStatus.IN_PROGRESS,
        "checkpointed": UiStatus.IN_PROGRESS,
        "completed": UiStatus.COMPLETE,
        "reused": UiStatus.COMPLETE,
        "failed": UiStatus.FAIL,
        "cancel_requested": UiStatus.WARNING,
        "cancelled_at_safe_boundary": UiStatus.WARNING,
        "blocked": UiStatus.BLOCKED,
    }
)


def ui_status_for_pipeline_stage_status(value: str) -> UiStatus:
    try:
        return _STAGE_STATUS[str(value)]
    except KeyError:
        raise ValueError(f"unregistered pipeline stage status {value!r}") from None


#: Heatmap glyph classes encode DATA ADEQUACY, never goodness (plan F-05):
#: a stable plateau is informational — the metric's registered direction
#: (and the colorscale it selects) carries the goodness.
_HEATMAP_CLASS: Mapping[str, UiStatus] = MappingProxyType(
    {
        "stable_plateau": UiStatus.INFORMATIONAL,
        "knife_edge_point": UiStatus.WARNING,
        "failed_region": UiStatus.FAIL,
        "insufficient_data": UiStatus.INCONCLUSIVE,
        "blocked_cell": UiStatus.BLOCKED,
    }
)


def ui_status_for_heatmap_class(cell_class: str) -> UiStatus:
    try:
        return _HEATMAP_CLASS[str(cell_class)]
    except KeyError:
        raise ValueError(f"unregistered heatmap cell class {cell_class!r}") from None


_EMPTY_STATE: Mapping[EmptyStateKey, UiStatus] = MappingProxyType(
    {
        EmptyStateKey.NO_CONFIGURATIONS_PASS: UiStatus.FAIL,
        EmptyStateKey.NO_VERIFIED_FIRM_CONTRACT: UiStatus.BLOCKED,
        EmptyStateKey.BLOCKED_SEARCH_AXIS: UiStatus.BLOCKED,
        EmptyStateKey.INSUFFICIENT_SAMPLE: UiStatus.INCONCLUSIVE,
        EmptyStateKey.CHILD_REPLAY_FAILED: UiStatus.FAIL,
        EmptyStateKey.PROP_NOT_RUN_STRATEGY_GATE: UiStatus.NOT_APPLICABLE,
        EmptyStateKey.NO_MODEL_RESULT: UiStatus.UNAVAILABLE,
        EmptyStateKey.ARTIFACT_UNAVAILABLE: UiStatus.UNAVAILABLE,
        EmptyStateKey.PROTECTED_RANGE_REFUSAL: UiStatus.BLOCKED,
        EmptyStateKey.VERIFICATION_AUTHORIZATION_MISSING: UiStatus.BLOCKED,
        EmptyStateKey.CAPABILITY_PLANNED: UiStatus.NOT_APPLICABLE,
        EmptyStateKey.LINEAGE_NOT_COMPARABLE: UiStatus.NOT_APPLICABLE,
        EmptyStateKey.BROWSER_QA_UNAVAILABLE: UiStatus.UNAVAILABLE,
        EmptyStateKey.INSUFFICIENT_REGIME_PARTITION: UiStatus.INCONCLUSIVE,
        EmptyStateKey.REGIME_STATUS_BELOW_MINIMUM: UiStatus.BLOCKED,
        EmptyStateKey.NO_RUNS: UiStatus.NOT_SELECTED,
        EmptyStateKey.NOT_SELECTED: UiStatus.NOT_SELECTED,
        EmptyStateKey.NOT_APPLICABLE: UiStatus.NOT_APPLICABLE,
        EmptyStateKey.NOT_CONFIGURED: UiStatus.NOT_SELECTED,
        EmptyStateKey.RESUME_AVAILABLE: UiStatus.WARNING,
        EmptyStateKey.ARTIFACT_MISSING: UiStatus.UNAVAILABLE,
        EmptyStateKey.ARTIFACT_CORRUPT: UiStatus.CORRUPT,
        EmptyStateKey.LEGACY_READ_ONLY: UiStatus.INFORMATIONAL,
        EmptyStateKey.SUPERSEDED: UiStatus.SUPERSEDED,
        EmptyStateKey.PURPOSE_UNRESOLVED: UiStatus.BLOCKED,
        EmptyStateKey.RUNNER_UNAVAILABLE: UiStatus.BLOCKED,
        EmptyStateKey.LAUNCH_NOT_STARTED: UiStatus.WARNING,
        EmptyStateKey.AUTHORIZATION_NOT_READY: UiStatus.BLOCKED,
        EmptyStateKey.STORE_NAMESPACE_UNVERIFIED: UiStatus.BLOCKED,
        EmptyStateKey.SEED_PRODUCTION_NOT_AUTHORIZED: UiStatus.BLOCKED,
        # UI-2 (plan §6.7 / §9 Phase 2): the Verification Center and draft states
        EmptyStateKey.SHORTLIST_UNAVAILABLE: UiStatus.UNAVAILABLE,
        EmptyStateKey.WINDOW_NOT_SELECTED: UiStatus.NOT_SELECTED,
        EmptyStateKey.SEED_MISSING: UiStatus.BLOCKED,
        EmptyStateKey.FINAL_AUTHORIZATION_UNSIGNED: UiStatus.BLOCKED,
        EmptyStateKey.PREFLIGHT_REFUSED: UiStatus.BLOCKED,
        EmptyStateKey.DRAFT_ARCHIVED: UiStatus.INFORMATIONAL,
        EmptyStateKey.DRAFT_SESSION_ONLY: UiStatus.INFORMATIONAL,
    }
)


def ui_status_for_empty_state(key: EmptyStateKey | str) -> UiStatus:
    return _EMPTY_STATE[EmptyStateKey(key)]
