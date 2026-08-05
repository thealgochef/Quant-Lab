"""Frozen custom-session recapture helpers.

Parsing and profile-registry reads remain available so old dashboard state can
be migrated.  Launch/run operations are refused during the IFVG v2 repair:
only ``run_ifvg_repair_verification.py`` may read market data.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

JOBS_DIR = _REPO_ROOT / "data" / "ifvg_experiments" / "recapture_jobs"
PROFILES_PATH = _REPO_ROOT / "data" / "ifvg_profiles.json"
_SESSION_NAMES = ("asia", "london", "ny")
_REFUSAL = (
    "custom IFVG recapture is disabled during the v2 correctness repair; "
    "only the fixed nonsealed verification replay is authorized"
)

__all__ = [
    "JOBS_DIR",
    "PROFILES_PATH",
    "parse_windows_arg",
    "windows_to_arg",
    "windows_to_times",
    "config_for_windows",
    "job_dir_for",
    "write_job_status",
    "read_job_status",
    "list_profiles",
    "upsert_profile",
    "launch_recapture_job",
]


def parse_windows_arg(text: str) -> dict[str, tuple[str, str]]:
    windows: dict[str, tuple[str, str]] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, equals, span = part.partition("=")
        start, separator, end = span.partition("-")
        if not equals or not separator:
            raise ValueError(f"window {part!r} must be name=HH:MM-HH:MM")
        # Parsing now validates the values instead of deferring malformed times.
        time.fromisoformat(start.strip())
        time.fromisoformat(end.strip())
        windows[name.strip()] = (start.strip(), end.strip())
    return windows


def windows_to_arg(windows: dict[str, tuple[str, str]]) -> str:
    return ",".join(
        f"{name}={start}-{end}"
        for name, (start, end) in sorted(windows.items())
    )


def windows_to_times(
    windows: dict[str, tuple[str, str]],
) -> dict[str, tuple[time, time]]:
    return {
        name: (time.fromisoformat(start), time.fromisoformat(end))
        for name, (start, end) in windows.items()
    }


def config_for_windows(windows: dict[str, tuple[str, str]]):
    from alpha_lab.agents.data_infra.ifvg.config import (
        custom_session_capture_config,
    )

    return custom_session_capture_config(windows_to_times(windows))


def job_dir_for(atag: str, ctag: str, jobs_dir: Path | None = None) -> Path:
    return (jobs_dir or JOBS_DIR) / f"{atag}_{ctag}"


def write_job_status(job_dir: Path, status: dict) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    stamped = {
        **status,
        "updated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    temporary = job_dir / "status.json.tmp"
    temporary.write_text(json.dumps(stamped, indent=2), encoding="utf-8")
    temporary.replace(job_dir / "status.json")


def read_job_status(job_dir: Path | str) -> dict | None:
    path = Path(job_dir) / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _default_profile() -> dict:
    from alpha_lab.agents.data_infra.ifvg.config import (
        V2_DATASET_DIR,
        IfvgCaptureConfig,
    )

    cfg = IfvgCaptureConfig()
    return {
        "name": "IFVG v2 document default",
        "windows": {
            name: (
                window.start.strftime("%H:%M"),
                window.end.strftime("%H:%M"),
            )
            for name, window in cfg.session_scheme.sessions.items()
        },
        "atag": cfg.artifacts_tag(),
        "ctag": cfg.capture_tag(),
        "dataset_path": str(V2_DATASET_DIR),
        "created_utc": None,
        "status": "manifest_discovery",
        "is_default": True,
        "execution_enabled": True,
        "legacy_candidate_only": False,
    }


def list_profiles(profiles_path: Path | None = None) -> list[dict]:
    """V2 default first; saved v1 profiles are marked candidate-only."""
    path = profiles_path or PROFILES_PATH
    entries: list[dict] = []
    if path.exists():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = []
    default = _default_profile()
    migrated = [
        {
            **entry,
            "execution_enabled": False,
            "legacy_candidate_only": True,
            "status": "legacy_candidate_only",
        }
        for entry in entries
        if entry.get("ctag") != default["ctag"]
    ]
    return [default, *migrated]


def upsert_profile(entry: dict, profiles_path: Path | None = None) -> None:
    """Maintain the legacy registry without changing its execution status."""
    path = profiles_path or PROFILES_PATH
    entries: list[dict] = []
    if path.exists():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = []
    entries = [
        item for item in entries if item.get("ctag") != entry.get("ctag")
    ]
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    temporary.replace(path)


def launch_recapture_job(
    windows: dict[str, tuple[str, str]],
    *,
    name: str | None = None,
    workers: int = 1,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> Path:
    del windows, name, workers, start, end, limit
    raise PermissionError(_REFUSAL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows")
    parser.add_argument("--windows-json")
    parser.add_argument("--name")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if bool(args.windows) == bool(args.windows_json):
        parser.error("exactly one of --windows / --windows-json is required")
    print(f"REFUSED: {_REFUSAL}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
