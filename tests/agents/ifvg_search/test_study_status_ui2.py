"""UI-2: the additive §31 states the Verification Center, the draft lifecycle
and the review surface render intentionally (plan §6.7)."""

from __future__ import annotations

from alpha_lab.agents.data_infra.ifvg.presentation.status_vocabulary import (
    UiStatus,
    ui_status_for_empty_state,
)
from alpha_lab.agents.data_infra.ifvg.study_status import (
    EMPTY_STATE_PRESENTATIONS,
    EmptyStateKey,
)

_UI2_STATES = {
    "shortlist_unavailable": EmptyStateKey.SHORTLIST_UNAVAILABLE,
    "window_not_selected": EmptyStateKey.WINDOW_NOT_SELECTED,
    "seed_missing": EmptyStateKey.SEED_MISSING,
    "final_authorization_unsigned": EmptyStateKey.FINAL_AUTHORIZATION_UNSIGNED,
    "preflight_refused": EmptyStateKey.PREFLIGHT_REFUSED,
    "draft_archived": EmptyStateKey.DRAFT_ARCHIVED,
    "draft_session_only": EmptyStateKey.DRAFT_SESSION_ONLY,
}


def test_every_ui2_state_is_registered_with_the_full_presentation() -> None:
    for state_id, key in _UI2_STATES.items():
        presentation = EMPTY_STATE_PRESENTATIONS[state_id]
        assert presentation.key is key
        assert presentation.heading and presentation.explanation
        assert presentation.next_action, state_id  # every UI-2 state names its next action
        lowered = presentation.explanation.lower()
        for forbidden in ("c:\\", "traceback", "/users/"):
            assert forbidden not in lowered


def test_verification_center_states_name_their_owning_gates() -> None:
    assert "SeedProductionAuthorizationRef" in (
        EMPTY_STATE_PRESENTATIONS["seed_production_not_authorized"].owning_gate
    )
    assert "seed" in EMPTY_STATE_PRESENTATIONS["seed_missing"].owning_gate.lower()
    assert "VerificationAuthorizationRef" in (
        EMPTY_STATE_PRESENTATIONS["final_authorization_unsigned"].owning_gate
    )
    assert "preflight" in EMPTY_STATE_PRESENTATIONS["preflight_refused"].owning_gate.lower()
    assert "shortlist" in EMPTY_STATE_PRESENTATIONS["shortlist_unavailable"].owning_gate.lower()
    # a seed job is never a launch of the verification and never research evidence
    assert "preparation" in EMPTY_STATE_PRESENTATIONS["seed_missing"].explanation.lower()
    # the session-only draft state says nothing was written
    assert "no file" in EMPTY_STATE_PRESENTATIONS["draft_session_only"].explanation.lower()
    assert "restore" in EMPTY_STATE_PRESENTATIONS["draft_archived"].next_action.lower()


def test_every_ui2_state_maps_additively_onto_a_ui_status() -> None:
    """The presentation vocabulary maps every new state (plan §6.1): a missing
    shortlist is UNAVAILABLE, an unselected window NOT SELECTED, the seed /
    final-authorization / preflight blocks are BLOCKED, the draft lifecycle
    states are INFORMATIONAL — never a bare KeyError, never PASS."""

    expected = {
        EmptyStateKey.SHORTLIST_UNAVAILABLE: UiStatus.UNAVAILABLE,
        EmptyStateKey.WINDOW_NOT_SELECTED: UiStatus.NOT_SELECTED,
        EmptyStateKey.SEED_MISSING: UiStatus.BLOCKED,
        EmptyStateKey.FINAL_AUTHORIZATION_UNSIGNED: UiStatus.BLOCKED,
        EmptyStateKey.PREFLIGHT_REFUSED: UiStatus.BLOCKED,
        EmptyStateKey.DRAFT_ARCHIVED: UiStatus.INFORMATIONAL,
        EmptyStateKey.DRAFT_SESSION_ONLY: UiStatus.INFORMATIONAL,
    }
    for key, status in expected.items():
        assert ui_status_for_empty_state(key) is status, key
        assert ui_status_for_empty_state(key) is not UiStatus.PASS
    for key in EmptyStateKey:  # every registered key maps (no silent gap)
        assert isinstance(ui_status_for_empty_state(key), UiStatus), key
