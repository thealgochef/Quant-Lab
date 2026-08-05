"""Part C tests: session-scheme tag identity + custom-scheme plumbing + job files.

The load-bearing lock: the DEFAULT config's artifacts_tag/capture_tag are hard
literals — the 624 warmed Phase-A files and every canonical capture live under
those names. No test here launches a job or touches ``data/databento``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, time
from pathlib import Path

import pytest
from strategy_core.constants import IFVG_DOC_SESSION_SCHEME

from alpha_lab.agents.data_infra.ifvg.config import (
    IfvgCaptureConfig,
    custom_session_capture_config,
    legacy_ifvg_capture_config,
)
from alpha_lab.agents.data_infra.ifvg.day_artifacts import _session_of_bucket_start

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ifvg_recapture_job as recapture  # noqa: E402

#: Canonical tags — MUST NEVER CHANGE (regression lock).
DEFAULT_ATAG = "a9016426641071c3"
DEFAULT_CTAG = "035d9e14ff3276ed"
LEGACY_ATAG = "466b5fe8e7952ecd"
LEGACY_CTAG = "2a40b18e0b273ee0"

#: The canonical windows as time pairs (asia crosses midnight).
_DEFAULT_WINDOWS = {
    "asia": (time(16, 0), time(1, 45)),
    "london": (time(2, 0), time(7, 0)),
    "ny": (time(8, 0), time(14, 0)),
}
_CUSTOM_WINDOWS = {
    "asia": (time(16, 0), time(1, 45)),
    "london": (time(2, 0), time(7, 0)),
    "ny": (time(8, 30), time(13, 0)),  # ny times moved
}


# ── tag identity ──────────────────────────────────────────────────────────────


def test_default_tags_regression_locked() -> None:
    cfg = IfvgCaptureConfig()
    assert cfg.artifacts_tag() == DEFAULT_ATAG
    assert cfg.capture_tag() == DEFAULT_CTAG


def test_legacy_tags_remain_read_only_and_stable() -> None:
    cfg = legacy_ifvg_capture_config()
    assert cfg.artifacts_tag() == LEGACY_ATAG
    assert cfg.capture_tag() == LEGACY_CTAG
    assert cfg.identity_lane == "legacy_v1"
    assert cfg.section.execution_enabled is False


def test_custom_ny_window_rolls_both_tags() -> None:
    cfg = custom_session_capture_config(_CUSTOM_WINDOWS)
    assert cfg.artifacts_tag() != DEFAULT_ATAG
    assert cfg.capture_tag() != DEFAULT_CTAG


def test_custom_scheme_preserves_fixed_axes() -> None:
    cfg = custom_session_capture_config(_CUSTOM_WINDOWS)
    scheme = cfg.session_scheme
    assert sorted(scheme.sessions) == ["asia", "london", "ny"]
    assert scheme.timezone == IFVG_DOC_SESSION_SCHEME.timezone
    assert scheme.trading_day_boundary == IFVG_DOC_SESSION_SCHEME.trading_day_boundary
    assert scheme.closed_window == IFVG_DOC_SESSION_SCHEME.closed_window
    # crosses_midnight derived per window: start > end only for asia.
    assert scheme.sessions["asia"].crosses_midnight is True
    assert scheme.sessions["london"].crosses_midnight is False
    assert scheme.sessions["ny"].crosses_midnight is False
    # the section's CONTRACT scheme rolled with it (profile_hash -> capture_tag)
    assert cfg.section.session_scheme.sessions["ny"].start == "08:30"
    assert cfg.section.session_scheme.sessions["asia"].crosses_midnight is True


def test_custom_tags_are_deterministic() -> None:
    a = custom_session_capture_config(_CUSTOM_WINDOWS)
    b = custom_session_capture_config(dict(reversed(list(_CUSTOM_WINDOWS.items()))))
    assert a.artifacts_tag() == b.artifacts_tag()
    assert a.capture_tag() == b.capture_tag()
    assert a.artifacts_tag() == a.artifacts_tag()  # stable across calls


def test_default_times_round_trip_to_canonical_tags() -> None:
    """Windows identical to the research scheme produce the CANONICAL tags —
    the conditional signature appends nothing for the default payload."""
    cfg = custom_session_capture_config(_DEFAULT_WINDOWS)
    assert cfg.artifacts_tag() == DEFAULT_ATAG
    assert cfg.capture_tag() == DEFAULT_CTAG


def test_custom_windows_reject_wrong_names() -> None:
    with pytest.raises(ValueError, match="exactly"):
        custom_session_capture_config({"asia": (time(19, 0), time(2, 45))})
    bad = dict(_CUSTOM_WINDOWS)
    bad["overnight"] = bad.pop("asia")
    with pytest.raises(ValueError, match="exactly"):
        custom_session_capture_config(bad)


# ── day_artifacts default-path identity ───────────────────────────────────────


def test_session_of_bucket_start_default_scheme_unchanged() -> None:
    # 2026-06-05 15:00 UTC = 11:00 ET (EDT) -> ny under the engine scheme.
    ny_ts = datetime(2026, 6, 5, 15, 0, tzinfo=UTC)
    assert _session_of_bucket_start(ny_ts, IFVG_DOC_SESSION_SCHEME) == "ny"
    # 00:30 UTC = 20:30 ET previous evening -> asia (crosses midnight).
    asia_ts = datetime(2026, 6, 5, 0, 30, tzinfo=UTC)
    assert _session_of_bucket_start(asia_ts, IFVG_DOC_SESSION_SCHEME) == "asia"
    # default config carries the SAME scheme object (behavioral identity).
    assert IfvgCaptureConfig().session_scheme is IFVG_DOC_SESSION_SCHEME


def test_session_of_bucket_start_honors_custom_scheme() -> None:
    # 12:10 UTC = 08:10 ET (EDT): NY under the default 08:00 open, but outside
    # the custom 08:30 open.
    ts = datetime(2026, 6, 5, 12, 10, tzinfo=UTC)
    assert _session_of_bucket_start(ts, IFVG_DOC_SESSION_SCHEME) == "ny"
    custom = custom_session_capture_config(_CUSTOM_WINDOWS)
    assert _session_of_bucket_start(ts, custom.session_scheme) == "none"


# ── job runner file plumbing (tmp_path only; nothing launched) ────────────────


def test_job_status_round_trip(tmp_path: Path) -> None:
    job_dir = tmp_path / "atag_ctag"
    status = {
        "state": "warming",
        "day": "2026-01-05",
        "done_days": 3,
        "total_days": 156,
        "started_utc": "2026-07-29T00:00:00+00:00",
    }
    recapture.write_job_status(job_dir, status)
    read = recapture.read_job_status(job_dir)
    assert read is not None
    assert {k: read[k] for k in status} == status
    assert read["updated_utc"]  # stamped on every write
    assert recapture.read_job_status(tmp_path / "missing") is None
    (job_dir / "status.json").write_text("{not json", encoding="utf-8")
    assert recapture.read_job_status(job_dir) is None


def test_parse_windows_arg_round_trip() -> None:
    text = "asia=19:00-02:45,london=03:00-08:00,ny=09:30-16:00"
    windows = recapture.parse_windows_arg(text)
    assert windows == {
        "asia": ("19:00", "02:45"),
        "london": ("03:00", "08:00"),
        "ny": ("09:30", "16:00"),
    }
    assert recapture.windows_to_arg(windows) == text
    with pytest.raises(ValueError, match="name=HH:MM-HH:MM"):
        recapture.parse_windows_arg("asia=19:00")


def test_list_profiles_default_first_and_upsert(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    profiles = recapture.list_profiles(profiles_path=path)
    assert len(profiles) == 1
    assert profiles[0]["is_default"] is True
    assert profiles[0]["atag"] == DEFAULT_ATAG
    assert profiles[0]["ctag"] == DEFAULT_CTAG

    entry = {
        "name": "late ny",
        "windows": {"asia": ["19:00", "02:45"], "london": ["03:00", "08:00"],
                    "ny": ["09:30", "16:00"]},
        "atag": "aaaa",
        "ctag": "cccc",
        "dataset_path": "x.parquet",
        "created_utc": "2026-07-29T00:00:00+00:00",
        "status": "ready",
    }
    recapture.upsert_profile(entry, profiles_path=path)
    profiles = recapture.list_profiles(profiles_path=path)
    assert [p.get("is_default", False) for p in profiles] == [True, False]
    assert profiles[1]["name"] == "late ny"

    # upsert with the same ctag REPLACES, never duplicates.
    recapture.upsert_profile({**entry, "name": "renamed"}, profiles_path=path)
    profiles = recapture.list_profiles(profiles_path=path)
    assert len(profiles) == 2
    assert profiles[1]["name"] == "renamed"

    # a second ctag appends.
    recapture.upsert_profile({**entry, "ctag": "dddd", "name": "other"}, profiles_path=path)
    assert len(recapture.list_profiles(profiles_path=path)) == 3
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert {e["ctag"] for e in raw} == {"cccc", "dddd"}
