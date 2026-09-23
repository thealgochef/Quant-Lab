"""Progress reports actual work states and never resume a saved study on render."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.presentation import workspace  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.workspace import (  # noqa: E402
    StudySummary,
    elapsed,
    search_activity,
    search_progress,
    search_status,
)


def _children(**counts):
    return [{"state": state} for state, count in counts.items() for _ in range(count)]


def _app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace._progress(st, ifvg_workspace._RECOVERY_STUDY, ifvg_workspace._RECOVERY_ROOTS)


def _render(monkeypatch, tmp_path, state, *, kind="search", status=None):
    import ifvg_research_pipeline
    import ifvg_workspace
    from streamlit.testing.v1 import AppTest

    def forbidden(*args, **kwargs):
        pytest.fail("Rendering progress must not launch or resume work")

    monkeypatch.setattr(ifvg_research_pipeline, "launch_existing", forbidden)
    roots = {"state_root": tmp_path / "jobs", "pipeline_state_root": tmp_path / "pipelines"}
    study = StudySummary(
        key="a" * 64,
        kind=kind,
        name="Saved study",
        question="Question",
        dates="Dates",
        status=status or search_status(state),
        scope="research",
        state=state,
    )
    monkeypatch.setattr(ifvg_workspace, "_RECOVERY_STUDY", study, raising=False)
    monkeypatch.setattr(ifvg_workspace, "_RECOVERY_ROOTS", roots, raising=False)
    at = AppTest.from_function(_app).run()
    assert not at.exception
    return at, roots, study


def _text(at):
    return " ".join(
        str(element.value)
        for kind in ("caption", "warning", "info", "subheader")
        for element in at.get(kind)
    )


def test_all_attempted_is_not_all_completed():
    state = {"phase": "replays", "children": _children(completed=147, failed=45)}
    counts = search_progress(state)
    assert counts.total == counts.attempted == 192
    assert counts.completed == 147 and counts.failed == 45
    assert counts.running == counts.queued == counts.reused == 0
    assert search_activity(state) == "Finalizing configuration results"
    assert search_status(state) == "Running"


def test_exclusive_counts_include_queued_stopped_blocked_and_unknown():
    counts = search_progress({"children": _children(
        completed=1, reused=2, failed=3, running=1, queued=4,
        cancelled_at_safe_boundary=5, blocked=6, new_unrecognized_state=7,
    )})
    assert counts.total == 29
    assert counts.attempted == 7
    assert counts.completed == 1 and counts.reused == 2
    assert counts.queued == 4 and counts.stopped == 5
    assert counts.blocked == 6 and counts.unresolved == 7


@pytest.mark.parametrize(("phase", "expected"), [
    ("charter_frozen", "Verifying study inputs"),
    ("children_enumerated", "Preparing input artifacts"),
    ("artifacts_prewarmed", "Preparing configuration replays"),
    ("replays", "Evaluating configurations"),
    ("frontier_complete", "Saving study results"),
])
def test_phase_labels_do_not_claim_replays_during_preparation(phase, expected):
    assert search_activity({"phase": phase}) == expected


def test_zero_children_preparation_is_not_false_progress(monkeypatch, tmp_path):
    at, _, _ = _render(monkeypatch, tmp_path, {"phase": "charter_frozen", "children": []})
    assert "Verifying study inputs" in _text(at)
    assert "Configuration counts become available after input verification" in _text(at)
    assert "Elapsed time: Not recorded" in _text(at)
    assert not at.get("progress")
    assert "Evaluating configurations" not in _text(at)


def test_input_verification_counts_are_separate_from_replay_completion(monkeypatch, tmp_path):
    state = {
        "phase": "charter_frozen",
        "children": [],
        "phase_notes": {
            "preparation_completed_configurations": "19",
            "preparation_planned_configurations": "192",
        },
    }
    at, _, _ = _render(monkeypatch, tmp_path, state)
    assert "Input checks: 19 of 192 configuration input sets verified." in _text(at)
    assert not at.get("progress")
    assert "configurations attempted" not in _text(at)


def test_full_attempts_show_failures_and_no_queued_work(monkeypatch, tmp_path):
    state = {"phase": "replays", "children": _children(completed=147, failed=45)}
    at, _, _ = _render(monkeypatch, tmp_path, state)
    text = _text(at)
    assert "192 of 192 configurations attempted" in text
    assert "Failed: 45 · Running: 0 · Queued: 0" in text
    assert "45 configurations failed. These are not completed results." in text
    assert "No configurations are currently queued or marked running" in text
    assert at.get("progress")[0].proto.text == "147 completed · 0 reused · 192 configurations total"
    assert at.get("progress")[0].proto.value == 76


def test_stop_request_is_visible_and_saved_without_resuming(monkeypatch, tmp_path):
    at, roots, study = _render(
        monkeypatch, tmp_path, {"phase": "replays", "children": _children(running=1, queued=2)}
    )
    sentinel = roots["state_root"] / study.key / "cancel.requested"
    assert not sentinel.exists()
    at.button(key="ifvg_workspace_cancel").click().run()
    assert not at.exception and sentinel.exists()
    at.run()
    assert "Stop requested. Current work will finish at a safe boundary" in _text(at)
    assert at.button(key="ifvg_workspace_cancel").disabled
    assert not any(button.label == "Resume study" for button in at.button)


def test_paused_has_manual_resume_and_retains_failures(monkeypatch, tmp_path):
    state = {
        "phase": "cancelled",
        "children": _children(completed=147, failed=45),
        "attempt_started_at_utc": "2026-09-10T01:00:00+00:00",
        "attempt_finished_at_utc": "2026-09-10T04:30:00+00:00",
    }
    at, roots, _ = _render(monkeypatch, tmp_path, state)
    assert "The study is paused. Saved progress has been retained." in _text(at)
    assert "Current attempt elapsed time: 3h 30m" in _text(at)
    assert at.button(key="ifvg_workspace_run").label == "Resume study"
    assert not any(button.label == "Stop after current work" for button in at.button)
    at.button(key="ifvg_workspace_refresh").click().run()
    assert not at.exception
    assert not roots["state_root"].exists()


def test_running_timer_advances_beyond_last_checkpoint(monkeypatch):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 10, 4, 30, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(workspace, "datetime", Clock)
    state = {
        "phase": "replays",
        "started_at_utc": "2026-09-09T01:00:00+00:00",
        "attempt_started_at_utc": "2026-09-10T02:00:00+00:00",
        "updated_at_utc": "2026-09-10T02:05:00+00:00",
    }
    assert elapsed(state) == "2h 30m"


@pytest.mark.parametrize("state", [
    {},
    {"phase": "cancelled"},
    {"phase": "cancelled", "started_at_utc": "2026-09-10T02:00:00+00:00"},
    {"phase": "cancelled", "attempt_started_at_utc": "not a timestamp"},
])
def test_missing_legacy_timing_stays_unavailable(state):
    assert elapsed(state) == "Not recorded"


def test_pipeline_progress_and_attempt_timer_are_unchanged(monkeypatch, tmp_path):
    state = {
        "current_stage": "02_replay",
        "stages": {
            "00_check": {"in_plan": True, "status": "completed"},
            "02_replay": {"in_plan": True, "status": "running"},
            "03_evidence": {"in_plan": True, "status": "pending"},
        },
        "attempts": [{"started_at": "2026-09-10T01:00:00+00:00",
                      "ended_at": "2026-09-10T02:05:00+00:00"}],
    }
    at, _, _ = _render(monkeypatch, tmp_path, state, kind="pipeline", status="Running")
    assert "Evaluate strategies" in _text(at)
    assert "Elapsed time: 1h 5m" in _text(at)
    assert at.get("progress")[0].proto.text == "1 of 3 work items complete"
    assert "configurations attempted" not in _text(at)
