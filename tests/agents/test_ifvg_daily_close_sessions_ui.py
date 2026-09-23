"""Real IFSM authoring and serialized-worker contracts; no economic replays."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, date, datetime, time
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from strategy_core.types import Direction

from alpha_lab.agents.data_infra.ifvg.ifsm_replication import (
    CORRECTED_MORNING,
    LEGACY_MORNING,
    corrected_morning_copy,
    export_fixed_configuration,
    fixed_axis_values,
    import_fixed_configuration,
    uses_legacy_morning,
)
from alpha_lab.agents.data_infra.ifvg.presentation.strategy_rules import describe_strategy
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    AxisAuthorizationError,
    CompositeAxisValue,
    assert_axes_authorized,
    resolve_axis_overrides,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import load_draft, save_draft
from tests.agents.test_ifsm_replication_ui import (
    BASELINE,
    ROOT,
    _draft,
    _fixed_widget,
    _focused_app,
    _focused_roots,
    _open,
    _roots,
    _save,
    apptest,
    wizard,
)
from tests.agents.test_ifvg_gap_invalidation_ui import _worker_children

pytestmark = [
    pytest.mark.usefixtures("developer_presentation"),
    pytest.mark.skipif(
        "holding_policy" not in SEARCH_AXIS_REGISTRY_V1,
        reason="Requires the isolated daily-close Core",
    ),
]

HOLDING = "holding_policy.scheduled_daily_close_v1"
SCHEDULES = (
    "enabled_entry_sessions.asia-london-ny",
    "enabled_entry_sessions.all_open_market_v1",
    "enabled_entry_sessions.daytime_chicago_0700_1555_v1",
    CORRECTED_MORNING,
)
MORNING_LABEL = "Morning - 7:00 AM to 10:30 AM Chicago time"
LEGACY_LABEL = "Legacy morning - 6:00 AM to 9:30 AM Chicago time (historical)"
CHICAGO = ZoneInfo("America/Chicago")


def _selected(schedule=CORRECTED_MORNING, distance=160, width=1, minute_parent=False):
    parents = "1m-3m-5m-10m-15m-30m" if minute_parent else "3m-5m-10m-15m-30m"
    return {
        "enabled_entry_sessions": schedule,
        "holding_policy": HOLDING,
        "opposing_parent_distance_ticks_max": f"opposing_parent_distance_ticks_max.{distance}",
        "opposing_min_gap_ticks": f"opposing_min_gap_ticks.{width}",
        "parent_timeframes": f"parent_timeframes.{parents}",
        "htf_gap_invalidation_policy": "htf_gap_invalidation_policy.own_timeframe_close_v1",
    }


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_actual_fixed_preset_save_reload_import_and_worker_one_profile(
    monkeypatch, tmp_path, schedule
):
    roots = _roots(monkeypatch, tmp_path)
    original = _draft()
    app = _open(original, roots)
    for axis, value in _selected(schedule).items():
        _fixed_widget(app, axis).set_value(value).run()
    assert not app.exception
    sessions = _fixed_widget(app, "enabled_entry_sessions")
    assert MORNING_LABEL in sessions.options and LEGACY_LABEL in sessions.options
    saved = _save(app, original, roots)
    assert fixed_axis_values(saved) == _selected(schedule)
    reopened = _open(saved, roots)
    assert _fixed_widget(reopened, "enabled_entry_sessions").value == schedule
    exported = export_fixed_configuration(BASELINE, fixed_axis_values(saved))
    assert import_fixed_configuration(exported, BASELINE) == _selected(schedule)
    fields, children = _worker_children(saved, roots)
    assert fields["max_child_count"] == len(children) == 1
    resolved = resolve_profile_config({"section_overrides": children[0].section_overrides})
    section = resolved.section
    assert section.holding_policy == "scheduled_daily_close_v1"
    assert (
        section.daily_close_timezone,
        section.daily_close_time,
        section.daily_close_buffer_minutes,
    ) == ("America/Chicago", "15:55", 5)
    assert section.session_scheme == resolve_profile_config().section.session_scheme
    assert section.doc_sessions == resolve_profile_config().section.doc_sessions
    expected = {
        SCHEDULES[0]: ("legacy_doc_sessions_v1", ()),
        SCHEDULES[1]: ("all_open_market_v1", ()),
        SCHEDULES[2]: ("explicit_windows_v1", (("07:00", "15:55"),)),
        SCHEDULES[3]: ("explicit_windows_v1", (("07:00", "10:30"),)),
    }
    assert (section.entry_schedule_policy, section.entry_schedule_windows) == expected[schedule]
    assert section.entry_schedule_timezone == "America/Chicago"
    assert not roots["research"].exists()


def test_actual_ifsm_screen_save_reopen_edit_and_import(monkeypatch, tmp_path):
    import run_ifsm_research_ui as launcher
    import strategy_core

    from alpha_lab.agents.data_infra.ifvg.presentation import workspace_mode

    monkeypatch.setattr(launcher, "CORE", Path(strategy_core.__file__).resolve().parents[2])
    monkeypatch.setattr(launcher, "WORKSPACE", tmp_path)
    monkeypatch.setattr(workspace_mode, "DEVELOPER_MODE", False)
    app = apptest.AppTest.from_file(str(ROOT / "scripts/ifsm_research_ui.py"), default_timeout=120)
    app.run()
    app.button(key="ifvg_workspace_new").click().run()
    next(w for w in app.radio if w.label == "What would you like to research?").set_value(
        "Evaluate"
    ).run()
    next(w for w in app.button if w.label == "Configure study").click().run()
    next(w for w in app.button if w.label == "Next").click().run()
    assert not app.exception
    _fixed_widget(app, "enabled_entry_sessions").set_value(CORRECTED_MORNING).run()
    _fixed_widget(app, "holding_policy").set_value(HOLDING).run()
    next(w for w in app.button if w.label == "Save draft").click().run()
    identifier = app.session_state[f"{wizard.STATE_PREFIX}draft_id"]
    saved = load_draft(tmp_path / "drafts", identifier)
    exported = export_fixed_configuration(BASELINE, fixed_axis_values(saved))
    reopened = apptest.AppTest.from_file(
        str(ROOT / "scripts/ifsm_research_ui.py"), default_timeout=120
    )
    reopened.session_state["ifvg_workspace_screen"] = "new"
    reopened.session_state[f"{wizard.STATE_PREFIX}draft_id"] = identifier
    reopened.run()
    assert not reopened.exception
    assert _fixed_widget(reopened, "enabled_entry_sessions").value == CORRECTED_MORNING
    assert _fixed_widget(reopened, "holding_policy").value == HOLDING
    _fixed_widget(reopened, "enabled_entry_sessions").set_value(SCHEDULES[1]).run()
    next(w for w in reopened.button if w.label == "Save draft").click().run()
    assert (
        fixed_axis_values(load_draft(tmp_path / "drafts", identifier))["enabled_entry_sessions"]
        == SCHEDULES[1]
    )
    next(w for w in reopened.text_area if w.label == "Paste exported configuration JSON").set_value(
        exported
    ).run()
    next(w for w in reopened.button if w.label == "Apply imported configuration").click().run()
    assert not reopened.exception
    assert _fixed_widget(reopened, "enabled_entry_sessions").value == CORRECTED_MORNING
    assert _fixed_widget(reopened, "holding_policy").value == HOLDING


def test_legacy_draft_is_visible_unchanged_and_corrected_copy_is_distinct(monkeypatch, tmp_path):
    roots = _roots(monkeypatch, tmp_path)
    legacy = _draft(fixed={"enabled_entry_sessions": LEGACY_MORNING})
    save_draft(roots["drafts"], legacy)
    before = json.loads(json.dumps(asdict(legacy)))
    app = _open(legacy, roots)
    assert _fixed_widget(app, "enabled_entry_sessions").value == LEGACY_MORNING
    assert LEGACY_LABEL in _fixed_widget(app, "enabled_entry_sessions").options
    assert "This saved legacy preset" in " ".join(str(w.value) for w in app.caption)
    copy = corrected_morning_copy(legacy)
    assert json.loads(json.dumps(asdict(legacy))) == before
    assert asdict(load_draft(roots["drafts"], legacy.draft_id)) == before
    assert copy.draft_id != legacy.draft_id and copy.cloned_from == legacy.draft_id
    assert uses_legacy_morning(legacy) and not uses_legacy_morning(copy)
    assert copy.frozen_search_id is None
    _, children = _worker_children(legacy, roots)
    section = resolve_profile_config({"section_overrides": children[0].section_overrides}).section
    assert section.entry_schedule_policy == "legacy_doc_sessions_v1"
    assert section.doc_sessions == {"ny_0700_1030": ("07:00", "10:30")}
    assert section.holding_policy == "legacy_unrestricted_v1"
    legacy_hash = resolve_profile_config(
        {"section_overrides": children[0].section_overrides}
    ).section_config_hash
    _, corrected = _worker_children(copy, roots)
    corrected_hash = resolve_profile_config(
        {"section_overrides": corrected[0].section_overrides}
    ).section_config_hash
    assert legacy_hash != corrected_hash


def test_actual_workspace_corrected_copy_preserves_legacy_source(monkeypatch, tmp_path):
    roots = _focused_roots(monkeypatch, tmp_path)
    legacy = _draft(fixed={"enabled_entry_sessions": LEGACY_MORNING})
    save_draft(roots["draft_root"], legacy)
    before = asdict(load_draft(roots["draft_root"], legacy.draft_id))
    app = apptest.AppTest.from_function(_focused_app, default_timeout=120)
    app.session_state["ifvg_workspace_screen"] = "detail"
    app.session_state["ifvg_workspace_selected_study"] = legacy.draft_id
    app.run()
    assert not app.exception
    app.button(key="ifvg_workspace_correct_morning").click().run()
    assert not app.exception
    new_id = app.session_state[f"{wizard.STATE_PREFIX}draft_id"]
    assert new_id != legacy.draft_id
    assert _fixed_widget(app, "enabled_entry_sessions").value == CORRECTED_MORNING
    copied = load_draft(roots["draft_root"], new_id)
    assert copied.cloned_from == legacy.draft_id and copied.frozen_search_id is None
    assert asdict(load_draft(roots["draft_root"], legacy.draft_id)) == before


def test_exact_32_fixed_profiles_no_injected_defaults_or_legacy_contender(monkeypatch, tmp_path):
    roots = _roots(monkeypatch, tmp_path)
    hashes, loaded, counts = set(), [], dict.fromkeys(SCHEDULES, 0)
    for schedule, distance, width, minute_parent in product(
        SCHEDULES, (80, 160), (1, 4), (False, True)
    ):
        selected = _selected(schedule, distance, width, minute_parent)
        fields, children = _worker_children(_draft(fixed=selected), roots)
        assert fields["max_child_count"] == len(children) == 1
        assert set(fields["axes"]) == set(selected)
        resolved = resolve_profile_config({"section_overrides": children[0].section_overrides})
        hashes.add(resolved.section_config_hash)
        loaded.append(resolved.section)
        counts[schedule] += 1
    assert len(hashes) == len(loaded) == 32 and set(counts.values()) == {8}
    assert all(s.holding_policy == "scheduled_daily_close_v1" for s in loaded)
    assert all(s.htf_gap_invalidation_policy == "own_timeframe_close_v1" for s in loaded)
    assert not any("ny_0700_1030" in s.enabled_entry_sessions for s in loaded)
    assert sum(s.entry_schedule_windows == (("07:00", "10:30"),) for s in loaded) == 8
    assert not roots["research"].exists()


def test_composites_round_trip_and_require_supported_approval():
    for axis, value_id in (
        ("enabled_entry_sessions", CORRECTED_MORNING),
        ("holding_policy", HOLDING),
    ):
        value = AXIS_VALUE_REGISTRY_V1[value_id]
        assert isinstance(value, CompositeAxisValue)
        loaded = CompositeAxisValue.model_validate_json(value.model_dump_json())
        selected = {axis: value_id}
        assert resolve_axis_overrides(selected, {value_id: loaded}) == resolve_axis_overrides(
            selected
        )
        with pytest.raises(AxisAuthorizationError, match="ratification"):
            assert_axes_authorized(selected)
    for field in ("entry_schedule_windows", "entry_schedule_timezone", "daily_close_time"):
        with pytest.raises(AxisAuthorizationError, match="not searchable"):
            assert_axes_authorized(
                {field: SEARCH_AXIS_REGISTRY_V1[field].baseline_value_id}, require_ratified=False
            )


@pytest.mark.parametrize("day", [date(2026, 3, 6), date(2026, 3, 9)])
@pytest.mark.parametrize(
    ("wall", "allowed"),
    [
        (time(6, 59), False),
        (time(6, 59, 59), False),
        (time(7), True),
        (time(7, 0, 1), True),
        (time(10, 29), True),
        (time(10, 29, 59), True),
        (time(10, 30), False),
        (time(10, 30, 1), False),
    ],
)
def test_corrected_preset_gates_actual_executions_across_march_clock_change(
    monkeypatch, day, wall, allowed
):
    from tests.agents.ifvg_search import test_session_entry_windows as fixture

    monkeypatch.setattr(fixture, "_VALUE_ID", CORRECTED_MORNING)
    entry = datetime.combine(day, wall, tzinfo=CHICAGO).astimezone(UTC)
    orchestrator, _, emissions = fixture._drive_candidate(monkeypatch, entry, Direction.LONG)
    executed = [row for row in emissions if row.kind == "executed_trade"]
    # The actual entry emission may be open-trade state; use Core's authoritative state.
    opened = orchestrator._reducer.active_trade_count == 1
    assert opened == allowed
    assert not executed  # the fixture stops at entry; no invented closed result


def test_holding_and_entry_rules_describe_distinct_clocks():
    section = resolve_profile_config(
        {"section_overrides": resolve_axis_overrides(_selected())}
    ).section
    description = describe_strategy(section)
    assert description.status == "available"
    text = " ".join(description.detail_bullets)
    assert "7:00 AM–10:30 AM" in text and "3:55 PM" in text
    assert "end of a morning entry window does not close" in text
    assert "daily or weekend market closures is prohibited" in text
    assert "Session or trading-day changes do not close" not in text
    exported = json.loads(export_fixed_configuration(BASELINE, _selected()))
    assert exported["effective_section"]["entry_schedule_windows"] == [["07:00", "10:30"]]
