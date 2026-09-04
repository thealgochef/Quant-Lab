"""UI-3 §6.4 — helper text, the glossary and the explicit exemptions (pure)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.help_registry import (
    GLOSSARY,
    GLOSSARY_TERMS,
    HELP_EXEMPTIONS,
    HELP_REGISTRY,
    HelpEntry,
    glossary_markdown,
    help_for_metric,
    help_text,
)


def test_every_help_entry_is_complete_and_renders_every_field() -> None:
    assert len(HELP_REGISTRY) >= 100
    for control_id, entry in HELP_REGISTRY.items():
        assert isinstance(entry, HelpEntry)
        assert entry.control_id == control_id
        assert "." in control_id  # <surface>.<control>
        assert entry.what_it_changes.strip() and entry.why.strip(), control_id
        assert entry.default.strip() and entry.availability.strip(), control_id
        assert set(entry.requires) <= {"replay", "refit", "resimulation"}, control_id
        text = help_text(control_id)
        assert entry.what_it_changes in text
        assert entry.why in text
        assert entry.default in text
        assert ("new identity" if entry.changes_identity else "unchanged") in text
        assert ("Owner approval: required" if entry.owner_approval else "not required") in text
    with pytest.raises(ValueError, match="unregistered help"):
        help_text("nowhere.nothing")


def test_glossary_defines_the_listed_acronyms() -> None:
    assert GLOSSARY_TERMS == (
        "FVG",
        "IFVG",
        "FSM",
        "HTF",
        "LTF",
        "OOS",
        "PIT",
        "MBP-1",
        "R multiple",
        "Brier",
        "Brier skill",
        "AUC",
        "EQH/EQL",
        "MFE/MAE",
        "AMI",
        "Q-40",
    )
    for term in GLOSSARY_TERMS:
        entry = GLOSSARY[term]
        assert entry.term == term
        assert entry.expansion.strip() and len(entry.definition) > 30, term
    rendered = glossary_markdown()
    for term in GLOSSARY_TERMS:
        assert f"**{term}**" in rendered


def test_help_exemptions_are_explicit_and_justified() -> None:
    assert HELP_EXEMPTIONS
    for (script, label), rationale in HELP_EXEMPTIONS.items():
        assert script.startswith("ifvg_") and script.endswith(".py"), script
        assert label.strip(), script
        assert len(rationale) > 15, (script, label)
    # navigation-only controls are the intended exemptions; consequential
    # actions never are
    assert ("ifvg_study_wizard.py", "Back") in HELP_EXEMPTIONS
    assert ("ifvg_study_wizard.py", "Next") in HELP_EXEMPTIONS
    for consequential in (
        ("ifvg_results_tab.py", "Delete draft permanently"),
        ("ifvg_active_runs_tab.py", "Request Safe Cancel"),
        ("ifvg_pipeline_tab.py", "Publish and Activate Catalog Entry"),
        ("ifvg_pipeline_tab.py", "Freeze Pipeline Specification and Launch"),
        ("ifvg_study_wizard.py", "Freeze Search Charter and Launch"),
    ):
        assert consequential not in HELP_EXEMPTIONS, consequential


def test_metric_help_is_built_from_the_registry() -> None:
    text = help_for_metric("brier_score")
    assert "Brier" in text and "lower is better" in text
    assert "reference" in text.lower()
    with pytest.raises(ValueError, match="unregistered metric"):
        help_for_metric("no_such_metric")
