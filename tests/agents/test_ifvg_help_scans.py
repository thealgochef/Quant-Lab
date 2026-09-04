"""UI-3 §6.4 source scans: every widget of every UI script carries ``help=``
or is a registered, justified, LIVE exemption; every ``help_text("…")`` id
exists; every collapsed label carries an accessible name and help; the
presentation package stays Streamlit-free."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.help_registry import (
    HELP_EXEMPTIONS,
    HELP_REGISTRY,
)

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / "scripts"

#: every Streamlit UI script of the IFVG Lab (the R4–UI-2 scripts plus the
#: Context Research / Replay / Data & Audit tab and the verifier)
HELP_SCAN_SCRIPTS: tuple[str, ...] = (
    "ifvg_ui_common.py",
    "ifvg_study_tab.py",
    "ifvg_study_wizard.py",
    "ifvg_active_runs_tab.py",
    "ifvg_results_tab.py",
    "ifvg_results_compare.py",
    "ifvg_pipeline_tab.py",
    "ifvg_mbp1_panels.py",
    "ifvg_regime_panels.py",
    "ifvg_verification_center.py",
    "ifvg_lab_tab.py",
    "ifvg_verifier_tab.py",
)

_WIDGET = re.compile(
    r"\.(selectbox|radio|checkbox|text_input|number_input|text_area|button|slider|"
    r"multiselect|date_input|toggle|select_slider|file_uploader|download_button|"
    r"form_submit_button|color_picker|time_input)\("
)
_HELP_ID = re.compile(r"help_text\(\s*[\"']([^\"']+)[\"']\s*\)")
_METRIC_HELP_ID = re.compile(r"help_for_metric\(\s*[\"']([^\"']+)[\"']\s*\)")


def _widget_calls(source: str) -> list[tuple[int, str, str]]:
    """(line, widget kind, full call text) for every widget call."""

    calls: list[tuple[int, str, str]] = []
    for match in _WIDGET.finditer(source):
        index, depth = match.end(), 1
        while depth and index < len(source):
            char = source[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        line = source.count("\n", 0, match.start()) + 1
        calls.append((line, match.group(1), source[match.start() : index]))
    return calls


def _label_of(call: str) -> str:
    """The first positional argument as a label: a string literal (an f-string
    is reduced to its static prefix) or ``<expr:…>`` for a non-literal."""

    body = call[call.index("(") + 1 :].lstrip()
    match = re.match(r"(f?)(\"\"\"|'''|\"|')", body)
    if match is None:
        return "<expr:" + body.split(",", 1)[0].strip()[:40] + ">"
    prefix, quote = match.group(1), match.group(2)
    start = match.end()
    end = body.index(quote, start)
    text = body[start:end]
    if prefix == "f" and "{" in text:
        text = text[: text.index("{")]
    return text.strip()


def _sources() -> dict[str, str]:
    return {name: (_SCRIPTS / name).read_text(encoding="utf-8") for name in HELP_SCAN_SCRIPTS}


def test_every_widget_has_help_or_a_registered_exemption() -> None:
    """Plan §6.4: an unregistered missing helper fails the scan."""

    missing: list[tuple[str, int, str, str]] = []
    for name, source in _sources().items():
        for line, kind, call in _widget_calls(source):
            if "help=" in call:
                continue
            label = _label_of(call)
            if (name, label) in HELP_EXEMPTIONS:
                continue
            missing.append((name, line, kind, label))
    assert not missing, "widgets without help or a registered exemption:\n" + "\n".join(
        f"  {name}:{line} {kind} {label!r}" for name, line, kind, label in missing
    )


def test_help_ids_exist_and_exemptions_are_live() -> None:
    sources = _sources()
    used: set[str] = set()
    for name, source in sources.items():
        for control_id in _HELP_ID.findall(source):
            assert control_id in HELP_REGISTRY, (name, control_id)
            used.add(control_id)
        for key in _METRIC_HELP_ID.findall(source):
            from alpha_lab.agents.data_infra.ifvg.presentation.metric_registry import (
                describe,
            )

            describe(key)  # raises for an unregistered metric
    # every registered control is wired somewhere (no dead help entries)
    unused = sorted(set(HELP_REGISTRY) - used)
    assert not unused, f"help entries no widget uses: {unused}"
    # every exemption names a LIVE widget still lacking help (no stale entries)
    for (script, label), _rationale in HELP_EXEMPTIONS.items():
        assert script in sources, script
        live = [
            call
            for _line, _kind, call in _widget_calls(sources[script])
            if "help=" not in call and _label_of(call) == label
        ]
        assert live, f"stale exemption: {script} {label!r} no longer lacks help"


def test_collapsed_labels_carry_accessible_names_and_help() -> None:
    """Plan §6.4: a collapsed label still needs an accessible name (two or
    more words) and help — never an anonymous control."""

    for name, source in _sources().items():
        for line, _kind, call in _widget_calls(source):
            if 'label_visibility="collapsed"' not in call:
                continue
            label = _label_of(call)
            assert not label.startswith("<expr:"), (name, line, label)
            assert len(label.split()) >= 2, (name, line, label)
            assert "help=" in call, (name, line, label)


@pytest.mark.parametrize(
    "module",
    (
        "metric_registry",
        "rollups",
        "help_registry",
        "labels",
        "status_vocabulary",
        "run_purpose",
        "charter_satisfiability",
        "flows",
        "review_vocabulary",
    ),
)
def test_presentation_modules_are_streamlit_free(module: str) -> None:
    source = (
        _REPO / "src/alpha_lab/agents/data_infra/ifvg/presentation" / f"{module}.py"
    ).read_text(encoding="utf-8")
    assert "import streamlit" not in source and "from streamlit" not in source, module
    assert "st_module" not in source, module
