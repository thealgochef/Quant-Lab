"""Entry-session chart bands follow the verified child's saved configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section

from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.replay_chart_provider import (
    MissingEvidenceError,
    _saved_session_scheme_windows,
    session_scheme_windows,
)
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import resolve_axis_overrides

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from ifvg_verifier_charts import session_band_intervals  # noqa: E402


def _context(section):
    return SimpleNamespace(
        pair=SimpleNamespace(
            v2=SimpleNamespace(reports={"effective_config.json": {"section": section}})
        )
    )


def _saved_ny_section():
    profile = resolve_profile_config(
        {
            "section_overrides": resolve_axis_overrides(
                {"enabled_entry_sessions": "enabled_entry_sessions.ny_0700_1030"}
            )
        }
    )
    return json.loads(json.dumps(profile.effective_config))


def test_saved_default_chart_geometry_preserves_existing_session_bands():
    section = default_ifvg_smc_section().model_dump(mode="json")
    assert _saved_session_scheme_windows(_context(section)) == session_scheme_windows()


@pytest.mark.parametrize(
    ("day", "start", "end"),
    [
        ("2026-01-13", "12:00", "15:30"),
        ("2026-06-09", "11:00", "14:30"),
    ],
)
def test_saved_ny_preset_chart_band_tracks_eastern_daylight_saving(day, start, end):
    section = _saved_ny_section()
    before = json.dumps(section, sort_keys=True)
    schemes = _saved_session_scheme_windows(_context(section))
    intervals = session_band_intervals(
        schemes, "doc", pd.Timestamp(f"{day}T00:00Z"), pd.Timestamp(f"{day}T23:59Z")
    )
    assert intervals == [
        {
            "session": "ny_0700_1030",
            "start": pd.Timestamp(f"{day}T{start}Z"),
            "end": pd.Timestamp(f"{day}T{end}Z"),
        }
    ]
    assert schemes["engine"] == session_scheme_windows()["engine"]
    assert json.dumps(section, sort_keys=True) == before


def test_saved_windows_are_not_replaced_with_todays_preset():
    section = _saved_ny_section()
    section["doc_sessions"] = {"saved_custom_window": ["06:15", "10:45"]}
    section["enabled_entry_sessions"] = ["saved_custom_window"]
    schemes = _saved_session_scheme_windows(_context(section))
    assert schemes["doc"]["sessions"] == {
        "saved_custom_window": {
            "start": "06:15:00", "end": "10:45:00", "crosses_midnight": False
        }
    }


@pytest.mark.parametrize("missing", ["doc_sessions", "session_scheme"])
def test_incomplete_saved_configuration_never_falls_back_to_default_bands(missing):
    section = _saved_ny_section()
    section.pop(missing)
    with pytest.raises(MissingEvidenceError, match="saved entry-session windows"):
        _saved_session_scheme_windows(_context(section))


def test_missing_saved_section_is_reported_as_missing_evidence():
    with pytest.raises(MissingEvidenceError, match="saved effective section"):
        _saved_session_scheme_windows(_context(None))
