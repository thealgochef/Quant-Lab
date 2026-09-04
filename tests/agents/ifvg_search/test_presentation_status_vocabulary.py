"""UI-1 §6.1 — the one status vocabulary every screen maps onto (pure)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.status_vocabulary import (
    COLOR_TOKENS,
    STATUS_SPECS,
    UiStatus,
    status_chip,
    status_for_evidence,
    status_from_gate,
    status_from_reference,
    ui_status_for_empty_state,
    ui_status_for_heatmap_class,
    ui_status_for_pipeline_stage_status,
    ui_status_for_study_status,
)
from alpha_lab.agents.data_infra.ifvg.study_status import (
    EMPTY_STATE_PRESENTATIONS,
    HEATMAP_GLYPHS,
    PIPELINE_STAGE_STATUS_PRESENTATIONS,
    StudyStatusKey,
)


def test_thirteen_statuses_each_carry_glyph_word_and_color_token() -> None:
    assert [status.value for status in UiStatus] == [
        "pass",
        "fail",
        "blocked",
        "warning",
        "inconclusive",
        "informational",
        "not_applicable",
        "not_selected",
        "unavailable",
        "corrupt",
        "in_progress",
        "complete",
        "superseded",
    ]
    assert set(STATUS_SPECS) == set(UiStatus)
    for status, spec in STATUS_SPECS.items():
        assert spec.status is status
        assert spec.glyph.strip() and spec.label.strip()
        assert spec.color_token in COLOR_TOKENS
        assert len(spec.meaning) > 20
        chip = status_chip(status)
        assert spec.glyph in chip and spec.label in chip  # never color alone
    assert STATUS_SPECS[UiStatus.PASS].color_token == "green"
    assert STATUS_SPECS[UiStatus.FAIL].color_token == "red"
    assert STATUS_SPECS[UiStatus.WARNING].color_token == "amber"
    assert STATUS_SPECS[UiStatus.INCONCLUSIVE].color_token == "amber"
    assert STATUS_SPECS[UiStatus.INFORMATIONAL].color_token == "blue"
    for gray in (
        UiStatus.NOT_APPLICABLE,
        UiStatus.NOT_SELECTED,
        UiStatus.UNAVAILABLE,
    ):
        assert STATUS_SPECS[gray].color_token == "gray"
    assert {
        status for status, spec in STATUS_SPECS.items() if spec.blocks_next_action
    } == {UiStatus.FAIL, UiStatus.BLOCKED, UiStatus.CORRUPT}


def test_gate_status_is_green_only_for_an_evaluated_true() -> None:
    assert status_from_gate(True) is UiStatus.PASS
    assert status_from_gate(False) is UiStatus.FAIL
    assert status_from_gate(None) is UiStatus.UNAVAILABLE
    # default-only / unevaluated evidence is NEVER PASS (plan F-07)
    assert status_from_gate(True, evaluated=False) is UiStatus.UNAVAILABLE
    assert status_from_gate("yes") is UiStatus.UNAVAILABLE  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "direction", "reference", "expected"),
    [
        (0.8, "higher_better", 0.5, UiStatus.PASS),
        (0.4, "higher_better", 0.5, UiStatus.FAIL),
        (0.2, "lower_better", 0.35, UiStatus.PASS),
        (0.5, "lower_better", 0.35, UiStatus.FAIL),
        (0.5, "higher_better", None, UiStatus.INFORMATIONAL),
        (None, "higher_better", 0.5, UiStatus.UNAVAILABLE),
        (1.2, "target", 1.0, UiStatus.INFORMATIONAL),  # distance only, no good/bad
        (3.0, "descriptive", None, UiStatus.INFORMATIONAL),
    ],
)
def test_reference_status_rules(value, direction, reference, expected) -> None:
    assert status_from_reference(value, direction=direction, reference=reference) is expected


def test_reference_status_refuses_an_unregistered_direction() -> None:
    with pytest.raises(ValueError, match="direction"):
        status_from_reference(1.0, direction="sideways", reference=0.0)


def test_evidence_kinds_map_to_non_green_states() -> None:
    assert status_for_evidence("missing") is UiStatus.UNAVAILABLE
    assert status_for_evidence("not_evaluated") is UiStatus.UNAVAILABLE
    assert status_for_evidence("corrupt") is UiStatus.CORRUPT
    assert status_for_evidence("not_applicable") is UiStatus.NOT_APPLICABLE
    assert status_for_evidence("not_selected") is UiStatus.NOT_SELECTED
    assert status_for_evidence("proposed") is UiStatus.WARNING
    assert status_for_evidence("superseded") is UiStatus.SUPERSEDED
    with pytest.raises(ValueError, match="evidence kind"):
        status_for_evidence("fine")


def test_existing_vocabularies_map_additively_onto_ui_status() -> None:
    for key in StudyStatusKey:
        assert isinstance(ui_status_for_study_status(key), UiStatus)
    assert ui_status_for_study_status(StudyStatusKey.RUNNING) is UiStatus.IN_PROGRESS
    assert ui_status_for_study_status(StudyStatusKey.BLOCKED) is UiStatus.BLOCKED
    assert ui_status_for_study_status(StudyStatusKey.SUPERSEDED) is UiStatus.SUPERSEDED
    for value in PIPELINE_STAGE_STATUS_PRESENTATIONS:
        assert isinstance(ui_status_for_pipeline_stage_status(value), UiStatus)
    assert ui_status_for_pipeline_stage_status("failed") is UiStatus.FAIL
    assert ui_status_for_pipeline_stage_status("blocked") is UiStatus.BLOCKED
    assert ui_status_for_pipeline_stage_status("reused") is UiStatus.COMPLETE
    for cell_class in HEATMAP_GLYPHS:
        assert isinstance(ui_status_for_heatmap_class(cell_class), UiStatus)
    # glyph classes encode DATA ADEQUACY, never goodness (plan F-05)
    assert ui_status_for_heatmap_class("stable_plateau") is UiStatus.INFORMATIONAL
    assert ui_status_for_heatmap_class("insufficient_data") is UiStatus.INCONCLUSIVE
    for state_id, presentation in EMPTY_STATE_PRESENTATIONS.items():
        assert isinstance(ui_status_for_empty_state(presentation.key), UiStatus), state_id
