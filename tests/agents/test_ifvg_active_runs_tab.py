"""FUX-MON-001..004 AppTests: fragment body + manual fallback, phase
checklist, keyboard funnel, exact child table, sanitized detail, safe
cancel, missing-status CLI fallback (TEST_MATRIX §3.11)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

apptest = pytest.importorskip("streamlit.testing.v1")

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_active_runs_tab as monitor  # noqa: E402
import ifvg_study_tab as study_tab  # noqa: E402

from tests.agents.ifvg_search.study_ui_fixture import (  # noqa: E402
    build_completed_search,
)

_MON = monitor._MON


@pytest.fixture(scope="module")
def completed_search(tmp_path_factory) -> dict:
    return build_completed_search(
        tmp_path_factory.mktemp("monitor_ui"), with_contract=False
    )


def _app() -> None:
    import ifvg_active_runs_tab as monitor
    import ifvg_study_tab as study_tab
    import streamlit as st

    monitor.render_active_runs(st, roots=study_tab.workspace_roots(st))


def _patch(monkeypatch, fixture: dict) -> None:
    monkeypatch.setattr(study_tab, "STORE_ROOT_RESEARCH", fixture["store_root"])
    monkeypatch.setattr(study_tab, "STATE_ROOT", fixture["state_root"])
    monkeypatch.setattr(study_tab, "DRAFT_ROOT", fixture["draft_root"])


def _run(monkeypatch, fixture: dict):
    _patch(monkeypatch, fixture)
    at = apptest.AppTest.from_function(_app, default_timeout=120)
    at.run()
    assert not at.exception
    return at


def test_monitor_renders_checklist_funnel_and_manual_refresh(
    monkeypatch, completed_search
) -> None:
    """FUX-MON-001: plain body under the fragment + always-present Refresh +
    phase checklist + the five keyboard funnel buttons with exact labels."""

    at = _run(monkeypatch, completed_search)
    labels = [b.label for b in at.button]
    assert "Refresh" in labels
    # FUX §16.2 (adversarial F13): the exact quoted parent-progress labels
    tables = str([t.value.to_dict() for t in at.table])
    for quoted in (
        "Profiles Generated",
        "Replays Completed",
        "Strategy-Gate Passes",
        "Prop-Feasible Configs",
        "Robust Finalists",
    ):
        assert quoted in tables
    assert "Generated · 4" in labels
    assert "Replay Valid · 4" in labels
    assert "Strategy Pass · 1" in labels
    assert "Prop Feasible · 1" in labels
    assert "Robust · 1" in labels
    checklist = "\n".join(str(c.value) for c in at.code)
    assert "✓ search_complete" in checklist
    assert "charter_frozen" in checklist


def test_funnel_buttons_filter_the_exact_child_table(
    monkeypatch, completed_search
) -> None:
    """FUX-MON-002: exact columns, pagination, skipped-stage copy."""

    at = _run(monkeypatch, completed_search)
    frames = at.dataframe
    assert frames  # the child table rendered
    table = frames[0].value
    assert list(table.columns) == [
        "Config",
        "Replay",
        "Strategy Gate",
        "Prop Simulation",
        "Robustness",
        "Status",
        "Human Explanation",
    ]
    assert len(table) == 4
    assert (table["Prop Simulation"] == "Not run — strategy gate failed").sum() == 3
    next(b for b in at.button if b.label == "Strategy Pass · 1").click().run()
    assert not at.exception
    filtered = at.dataframe[0].value
    assert len(filtered) == 1
    assert "Robust Finalist" in str(
        filtered["Status"].iloc[0]
    ) or "Development Exploratory Representative" in str(filtered["Status"].iloc[0])


def test_child_detail_shows_identities_and_sanitized_text(
    monkeypatch, completed_search
) -> None:
    """FUX-MON-003: identities, sanitized failure text, attempt history."""

    at = _run(monkeypatch, completed_search)
    codes = " ".join(str(c.value) for c in at.code)
    assert completed_search["search_id"] in codes  # charter identity block
    captions = " ".join(str(c.value) for c in at.caption)
    # FUX §16.5 / CS §12 (adversarial F14): membership + costed linkage
    assert "membership_id" in captions
    assert "costed_evaluation_id" in captions
    assert "Attempt history" in captions
    assert "replay invocation" in captions
    # sanitized: no local absolute path anywhere in the rendered surface
    everything = codes + captions + "\n".join(
        str(m.value) for m in at.markdown
    )
    assert "C:\\Users" not in everything.replace(
        str(completed_search["store_root"]), ""
    )


def test_safe_cancel_requires_confirmation_then_writes_the_sentinel(
    monkeypatch, completed_search, tmp_path
) -> None:
    """FUX-MON-004 — on a RUNNING copy of the state (terminal runs render no
    cancel control)."""

    import shutil

    running_root = tmp_path / "running_state"
    source = completed_search["state_root"] / completed_search["search_id"]
    target = running_root / completed_search["search_id"]
    target.mkdir(parents=True)
    state_file = source / "search_state.json"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    state["phase"] = "replays"
    (target / "search_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    fixture = dict(completed_search, state_root=running_root)
    at = _run(monkeypatch, fixture)
    cancel = next(b for b in at.button if b.label == "Request Safe Cancel")
    assert cancel.disabled  # confirmation checkbox gates the control
    confirm = next(c for c in at.checkbox if c.key == f"{_MON}cancel_confirm")
    confirm.check().run()
    next(
        b for b in at.button if b.label == "Request Safe Cancel"
    ).click().run()
    assert not at.exception
    sentinel = target / "cancel.requested"
    assert sentinel.exists()
    successes = " ".join(str(s.value) for s in at.success)
    assert "safe" in successes.lower()
    shutil.rmtree(running_root, ignore_errors=True)


def test_missing_status_file_renders_the_cli_escape_hatch(
    monkeypatch, completed_search, tmp_path
) -> None:
    """FUX §16.7: sanitized state + the exact project-relative command."""

    ghost_root = tmp_path / "ghost_state"
    (ghost_root / ("b" * 64)).mkdir(parents=True)
    fixture = dict(completed_search, state_root=ghost_root)
    _patch(monkeypatch, fixture)
    monkeypatch.setattr(study_tab, "STATE_ROOT", ghost_root)

    def _body() -> None:
        import ifvg_active_runs_tab as monitor
        import ifvg_study_tab as study_tab
        import streamlit as st

        monitor._render_monitor_body(
            st,
            roots=study_tab.workspace_roots(st),
            search_id="b" * 64,
        )

    at = apptest.AppTest.from_function(_body, default_timeout=60)
    at.run()
    assert not at.exception
    codes = " ".join(str(c.value) for c in at.code)
    assert (
        f"python scripts/ifvg_search_job.py status --search-id {'b' * 64}"
        in codes
    )
    headings = " ".join(str(h.value) for h in at.subheader)
    assert "Artifact missing" in headings  # UI-1: missing, never "unavailable"
    everything = codes + " ".join(str(c.value) for c in at.caption)
    assert str(ghost_root) not in everything  # no local path disclosure


def test_empty_state_when_no_jobs_exist(monkeypatch, tmp_path) -> None:
    fixture = {
        "store_root": tmp_path / "search_test" / "v1",
        "state_root": tmp_path / "state",
        "draft_root": tmp_path / "drafts",
    }
    at = _run(monkeypatch, fixture)
    headings = " ".join(str(h.value) for h in at.subheader)
    # UI-1 (plan §6.7): an empty job root is the distinct NO_RUNS state
    assert "No search runs exist yet" in headings
    assert "Artifact unavailable" not in headings
