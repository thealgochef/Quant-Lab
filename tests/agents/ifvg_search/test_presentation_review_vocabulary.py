"""UI-2 (owner Q3; plan F-06): the reviewer verdict vocabulary.

The ledger keys of ``ifvg_visual_review_v1`` are preserved; ``Unreviewed`` is
an UNSAVED UI state only (never a ledger value); the owner-approved labels
map one-to-one onto the persisted keys; ``not_applicable`` is the one
additive key; every verdict carries a definition.
"""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.review_vocabulary import (
    REVIEW_SAVE_STATES,
    UNREVIEWED,
    UNREVIEWED_LABEL,
    VERDICT_DEFINITIONS,
    VERDICT_LABELS,
    label_for_verdict,
    verdict_for_label,
    verdict_options,
)
from alpha_lab.agents.data_infra.ifvg.visual_review_store import REVIEW_VERDICTS


def test_owner_approved_labels_map_onto_the_preserved_ledger_keys() -> None:
    assert dict(VERDICT_LABELS) == {
        "correct": "Correct",
        "incorrect": "Incorrect",
        "insufficient_evidence": "Unclear",
        "questionable": "Needs investigation",
        "not_applicable": "Not applicable",
    }
    assert tuple(VERDICT_LABELS) == REVIEW_VERDICTS  # the ledger vocabulary, in order
    assert "not_applicable" in REVIEW_VERDICTS  # the one additive key (owner Q3)
    assert REVIEW_VERDICTS[:4] == (
        "correct",
        "incorrect",
        "questionable",
        "insufficient_evidence",
    )  # existing keys untouched, in their original order
    for key in REVIEW_VERDICTS:
        assert VERDICT_DEFINITIONS[key]
    assert "ambiguous" in VERDICT_DEFINITIONS["insufficient_evidence"]
    assert "not yet asserting incorrectness" in VERDICT_DEFINITIONS["questionable"]
    assert "does not apply" in VERDICT_DEFINITIONS["not_applicable"]


def test_unreviewed_is_a_ui_state_never_a_ledger_value() -> None:
    assert UNREVIEWED == "unreviewed"
    assert UNREVIEWED not in REVIEW_VERDICTS
    assert UNREVIEWED_LABEL == "Unreviewed"
    options = verdict_options()
    assert options[0] == UNREVIEWED_LABEL  # opening a case preselects nothing
    assert options[1:] == tuple(VERDICT_LABELS.values())
    assert verdict_for_label(UNREVIEWED_LABEL) is None  # nothing to persist
    assert verdict_for_label("Needs investigation") == "questionable"
    assert verdict_for_label("Unclear") == "insufficient_evidence"
    with pytest.raises(ValueError, match="unknown verdict label"):
        verdict_for_label("Great")
    assert label_for_verdict("questionable") == "Needs investigation"
    assert label_for_verdict(None) == UNREVIEWED_LABEL
    assert label_for_verdict("legacy_value") == "legacy_value"  # v1 rows render as-is
    assert REVIEW_SAVE_STATES == ("unsaved", "saving", "saved")
