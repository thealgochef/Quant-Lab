"""Session-scheme re-capture job (plan Part C): custom-window Phase A + capture.

Rebuilds the FULL per-day artifact/capture chain under a CUSTOM session-window
scheme (times only; names/timezone/boundary fixed) and writes the entry
dataset for the derived capture_tag. Every output filename carries the new
atag/ctag, so the canonical capture under ``data/databento/NQ/`` is never
touched; a windows spec that matches the canonical scheme is REFUSED (its tags
would collide with the canonical files).

Progress: ``data/ifvg_experiments/recapture_jobs/<atag>_<ctag>/status.json``
is rewritten after EVERY day (plus ``job.log`` next to it); on success the
profile is upserted into ``data/ifvg_profiles.json`` for the dashboard's
profile selector. Sealed discipline: the job CAPTURES sealed days (caches
only) — no statistic over them is computed here or anywhere downstream.

Usage:
    PYTHONPATH=src python scripts/ifvg_recapture_job.py \
        --windows "asia=19:00-02:45,london=03:00-08:00,ny=09:30-16:00" \
        [--windows-json '{"asia": ["19:00", "02:45"], ...}'] [--name label]
        [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--limit N] [--workers 4]

The dashboard imports :func:`launch_recapture_job` (detached ``Popen``, the W3
warmer detach precedent), :func:`read_job_status` and :func:`list_profiles`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time as time_mod
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime, time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

JOBS_DIR = _REPO_ROOT / "data" / "ifvg_experiments" / "recapture_jobs"
PROFILES_PATH = _REPO_ROOT / "data" / "ifvg_profiles.json"

_SESSION_NAMES = ("asia", "london", "ny")

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


# ── windows parsing ───────────────────────────────────────────────────────────


def parse_windows_arg(text: str) -> dict[str, tuple[str, str]]:
    """``"asia=19:00-02:45,london=03:00-08:00,ny=09:00-17:00"`` -> HH:MM pairs."""
    windows: dict[str, tuple[str, str]] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, span = part.partition("=")
        start, sep, end = span.partition("-")
        if not sep:
            raise ValueError(f"window {part!r} must be name=HH:MM-HH:MM")
        windows[name.strip()] = (start.strip(), end.strip())
    return windows


def windows_to_arg(windows: dict[str, tuple[str, str]]) -> str:
    return ",".join(f"{n}={s}-{e}" for n, (s, e) in sorted(windows.items()))


def windows_to_times(windows: dict[str, tuple[str, str]]) -> dict[str, tuple[time, time]]:
    return {
        name: (time.fromisoformat(start), time.fromisoformat(end))
        for name, (start, end) in windows.items()
    }


def config_for_windows(windows: dict[str, tuple[str, str]]):
    """The custom capture config for HH:MM windows (import deferred so the
    launcher helpers stay importable without a warmed environment)."""
    from alpha_lab.agents.data_infra.ifvg.config import custom_session_capture_config

    return custom_session_capture_config(windows_to_times(windows))


# ── status file + profile registry ────────────────────────────────────────────


def job_dir_for(atag: str, ctag: str, jobs_dir: Path | None = None) -> Path:
    return (jobs_dir or JOBS_DIR) / f"{atag}_{ctag}"


def write_job_status(job_dir: Path, status: dict) -> None:
    """Atomic-ish rewrite (tmp + replace) so a polling dashboard never reads a
    half-written JSON."""
    job_dir.mkdir(parents=True, exist_ok=True)
    status = {**status, "updated_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    tmp = job_dir / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    tmp.replace(job_dir / "status.json")


def read_job_status(job_dir: Path | str) -> dict | None:
    path = Path(job_dir) / "status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _default_profile() -> dict:
    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig

    cfg = IfvgCaptureConfig()
    ctag = cfg.capture_tag()
    return {
        "name": "default (engine scheme)",
        "windows": {
            name: (w.start.strftime("%H:%M"), w.end.strftime("%H:%M"))
            for name, w in cfg.session_scheme.sessions.items()
        },
        "atag": cfg.artifacts_tag(),
        "ctag": ctag,
        "dataset_path": str(
            Path(cfg.data_dir) / cfg.symbol / f"ifvg_entry_dataset_{ctag}.parquet"
        ),
        "created_utc": None,
        "status": "ready",
        "is_default": True,
    }


def list_profiles(profiles_path: Path | None = None) -> list[dict]:
    """All capture profiles, the DEFAULT profile always first."""
    path = profiles_path or PROFILES_PATH
    entries: list[dict] = []
    if path.exists():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = []
    default = _default_profile()
    return [default, *[e for e in entries if e.get("ctag") != default["ctag"]]]


def upsert_profile(entry: dict, profiles_path: Path | None = None) -> None:
    """Insert/replace by ``ctag`` (the full capture identity)."""
    path = profiles_path or PROFILES_PATH
    entries: list[dict] = []
    if path.exists():
        try:
            entries = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = []
    entries = [e for e in entries if e.get("ctag") != entry.get("ctag")]
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    tmp.replace(path)


# ── detached launcher (the W3 warmer detach precedent) ────────────────────────


def launch_recapture_job(
    windows: dict[str, tuple[str, str]],
    *,
    name: str | None = None,
    workers: int = 4,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
) -> Path:
    """Launch the job as a DETACHED subprocess; returns its job_dir. The
    initial status.json is written BEFORE spawn so the dashboard's first poll
    always finds one."""
    cfg = config_for_windows(windows)
    atag, ctag = cfg.artifacts_tag(), cfg.capture_tag()
    job_dir = job_dir_for(atag, ctag)
    write_job_status(
        job_dir,
        {
            "state": "warming",
            "day": None,
            "done_days": 0,
            "total_days": None,
            "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "windows": {k: list(v) for k, v in sorted(windows.items())},
            "name": name,
            "atag": atag,
            "ctag": ctag,
        },
    )
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--windows",
        windows_to_arg(windows),
        "--workers",
        str(workers),
    ]
    if name:
        args += ["--name", name]
    if start:
        args += ["--start", start]
    if end:
        args += ["--end", end]
    if limit:
        args += ["--limit", str(limit)]
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    creationflags = 0
    if sys.platform == "win32":
        creationflags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    log = open(job_dir / "job.log", "ab")  # noqa: SIM115 — handle passes to the child
    try:
        subprocess.Popen(
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(_REPO_ROOT),
            env=env,
            creationflags=creationflags,
            close_fds=False,
        )
    finally:
        log.close()
    return job_dir


# ── Phase A warm (spawn-safe worker; windows travel as HH:MM strings) ─────────


def warm_one_custom(date_str: str, windows: dict[str, tuple[str, str]]) -> tuple[str, str, float]:
    """Top-level spawn-safe worker: (date, status, seconds). Mirrors
    ``ifvg_artifact_warmer.warm_one`` but builds the CUSTOM cfg per call."""
    from alpha_lab.agents.data_infra.ifvg.day_artifacts import (
        build_day_artifacts,
        seeds_for_day,
        write_day_artifacts,
    )

    cfg = config_for_windows(windows)
    t0 = time_mod.time()
    if cfg.bars_path(date_str).exists() and cfg.levels_path(date_str).exists():
        return date_str, "cached", 0.0
    try:
        seeds = seeds_for_day(date_str, cfg)
        artifacts = build_day_artifacts(date_str, cfg, seeds)
        write_day_artifacts(artifacts, cfg)
    except Exception as exc:  # noqa: BLE001 — a bad day must not kill the warm
        return date_str, f"ERROR: {exc!r}", time_mod.time() - t0
    status = "ok" if artifacts.bars else "empty"
    return date_str, status, time_mod.time() - t0


def _warm_phase_a(
    days: list[str],
    windows: dict[str, tuple[str, str]],
    job_dir: Path,
    base_status: dict,
    workers: int,
) -> int:
    """Chunked-pool parallel warm (the warmer's RSS-bounding recipe); returns
    the error count. Status is rewritten after every completed day."""
    cfg = config_for_windows(windows)
    pending = [
        d for d in days if not (cfg.bars_path(d).exists() and cfg.levels_path(d).exists())
    ]
    done = len(days) - len(pending)
    errors = 0
    write_job_status(
        job_dir, {**base_status, "state": "warming", "done_days": done, "day": None}
    )
    chunk_size = max(1, workers * 6)
    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(warm_one_custom, d, windows): d for d in chunk}
            for future in as_completed(futures):
                date_str, status, seconds = future.result()
                done += 1
                if status.startswith("ERROR"):
                    errors += 1
                print(f"[warm {done}/{len(days)}] {date_str}: {status} ({seconds:.1f}s)",
                      flush=True)
                write_job_status(
                    job_dir,
                    {**base_status, "state": "warming", "day": date_str, "done_days": done},
                )
    return errors


# ── the job itself ────────────────────────────────────────────────────────────


def run_job(
    windows: dict[str, tuple[str, str]],
    *,
    name: str | None,
    workers: int,
    start: str | None,
    end: str | None,
    limit: int | None,
) -> int:
    from ifvg_artifact_warmer import available_store_days

    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
    from alpha_lab.agents.data_infra.ifvg.dataset import build_ifvg_capture
    from alpha_lab.agents.data_infra.ifvg.entry_dataset import build_entry_dataset

    cfg = config_for_windows(windows)
    atag, ctag = cfg.artifacts_tag(), cfg.capture_tag()
    job_dir = job_dir_for(atag, ctag)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    default_cfg = IfvgCaptureConfig()
    if atag == default_cfg.artifacts_tag() or ctag == default_cfg.capture_tag():
        msg = ("windows match the canonical scheme — tags would collide with the "
               "canonical capture (use the default profile instead)")
        write_job_status(
            job_dir,
            {"state": "refused", "error": msg, "started_utc": started,
             "updated_utc": started, "atag": atag, "ctag": ctag,
             "windows": {k: list(v) for k, v in sorted(windows.items())}},
        )
        print(f"REFUSED: {msg}")
        return 2
    days = available_store_days(cfg.symbol, Path(cfg.data_dir))
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    if limit:
        days = days[:limit]
    base_status = {
        "state": "warming",
        "day": None,
        "done_days": 0,
        "total_days": len(days),
        "started_utc": started,
        "windows": {k: list(v) for k, v in sorted(windows.items())},
        "name": name,
        "atag": atag,
        "ctag": ctag,
    }
    if not days:
        write_job_status(job_dir, {**base_status, "state": "failed", "error": "no store days"})
        print("no store days matched")
        return 1
    print(f"recapture job over {len(days)} days ({days[0]}..{days[-1]}) atag={atag} ctag={ctag}")

    try:
        t0 = time_mod.time()
        warm_errors = _warm_phase_a(days, windows, job_dir, base_status, workers)
        print(f"phase A done in {time_mod.time()-t0:.0f}s ({warm_errors} errors)")

        def progress(i: int, n: int, date_str: str) -> None:
            write_job_status(
                job_dir,
                {**base_status, "state": "capturing", "day": date_str, "done_days": i},
            )
            if i % 10 == 0 or i == n:
                print(f"  [capture {i}/{n}] {date_str} ({time_mod.time()-t0:.0f}s)", flush=True)

        chain = build_ifvg_capture(days, cfg, progress_fn=progress)
        capture = chain.frame()
        print(
            f"capture done: {len(capture)} rows, {len(chain.cached_days)} cached, "
            f"{len(chain.rebuilt_days)} rebuilt inline"
        )

        write_job_status(
            job_dir, {**base_status, "state": "dataset", "day": None, "done_days": len(days)}
        )
        entries = build_entry_dataset(capture, cfg)
        dataset_path = Path(cfg.data_dir) / cfg.symbol / f"ifvg_entry_dataset_{ctag}.parquet"
        if len(entries):
            entries.to_parquet(dataset_path, index=False)
        print(f"entry dataset: {len(entries)} rows -> {dataset_path}")

        # A 0-row capture slice is a completed job but not a usable profile —
        # status "empty" keeps it out of the dashboard's profile dropdown.
        upsert_profile(
            {
                "name": name or windows_to_arg(windows),
                "windows": {k: list(v) for k, v in sorted(windows.items())},
                "atag": atag,
                "ctag": ctag,
                "dataset_path": str(dataset_path) if len(entries) else None,
                "created_utc": started,
                "status": "ready" if len(entries) else "empty",
            }
        )
        write_job_status(
            job_dir, {**base_status, "state": "done", "day": None, "done_days": len(days)}
        )
        print("DONE")
        return 0
    except Exception as exc:  # noqa: BLE001 — the status file must record failure
        traceback.print_exc()
        write_job_status(job_dir, {**base_status, "state": "failed", "error": repr(exc)})
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", default=None,
                        help='e.g. "asia=19:00-02:45,london=03:00-08:00,ny=09:00-17:00"')
    parser.add_argument("--windows-json", default=None,
                        help='e.g. \'{"asia": ["19:00", "02:45"], ...}\'')
    parser.add_argument("--name", default=None, help="profile label for the dashboard")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if bool(args.windows) == bool(args.windows_json):
        parser.error("exactly one of --windows / --windows-json is required")
    if args.windows:
        windows = parse_windows_arg(args.windows)
    else:
        raw = json.loads(args.windows_json)
        windows = {k: (v[0], v[1]) for k, v in raw.items()}
    return run_job(
        windows,
        name=args.name,
        workers=args.workers,
        start=args.start,
        end=args.end,
        limit=args.limit,
    )


if __name__ == "__main__":
    raise SystemExit(main())
