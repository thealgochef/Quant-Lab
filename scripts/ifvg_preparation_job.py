"""Persisted process controls for IFVG pair preparation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_lab.agents.data_infra.ifvg.preparation import (  # noqa: E402
    PREPARATION_JOB_ROOT,
    read_preparation_state,
)


def _job_dir(profile: str) -> Path:
    if not profile.replace("_", "").replace("-", "").isalnum():
        raise ValueError("unsafe profile name")
    return ROOT / PREPARATION_JOB_ROOT / profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("start", "status", "cancel"))
    parser.add_argument(
        "--profile",
        default="ifvg_v2_doc_default_fresh_static_1r",
    )
    parser.add_argument("--allow-rebuild-day-artifacts", action="store_true")
    args = parser.parse_args()
    job_dir = _job_dir(args.profile)
    job_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        state = read_preparation_state(job_dir)
        print(json.dumps(None if state is None else asdict(state), default=str, sort_keys=True))
        return 0
    if args.command == "cancel":
        (job_dir / "cancel.requested").touch(exist_ok=True)
        print(json.dumps({"status": "cancellation_requested"}))
        return 0
    command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_ifvg_development_pair.py"),
        "--profile",
        args.profile,
    ]
    if args.allow_rebuild_day_artifacts:
        command.append("--allow-rebuild-day-artifacts")
    log = (job_dir / "job.log").open("a", encoding="utf-8")
    process = subprocess.Popen(  # noqa: S603 - exact local interpreter/script only
        command,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    log.close()
    (job_dir / "pid").write_text(f"{process.pid}\n", encoding="ascii")
    print(json.dumps({"status": "preparing", "pid": process.pid}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
