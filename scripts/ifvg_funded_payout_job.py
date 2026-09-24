"""Funded payout simulation job: start a detached worker for one frozen plan.

    python scripts/ifvg_funded_payout_job.py start  --plan-id <id> --store-root ... --state-root ...
    python scripts/ifvg_funded_payout_job.py status --plan-id <id> --state-root ...
    python scripts/ifvg_funded_payout_job.py publish --plan-id <id> ...   (retry the review folder)

The worker reads only the verified plan envelope from the store. It never
launches other studies, purchases, withdrawals or live orders.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

_ID = re.compile(r"^[0-9a-f]{64}$")


def _plan_id(value: str) -> str:
    if not _ID.match(value):
        raise SystemExit("plan id must be a 64-hex identifier")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "status", "worker", "publish"))
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--store-root", default=str(ROOT / "data/ifsm_ui_replication/search/v1"))
    parser.add_argument("--state-root",
                        default=str(ROOT / "data/ifsm_ui_replication/funded_jobs"))
    parser.add_argument("--reports-root", default=str(ROOT / "reports"))
    args = parser.parse_args(argv)
    plan_id = _plan_id(args.plan_id)
    from alpha_lab.propsim.funded.runner import (
        publish_review,
        read_state,
        run_plan,
        write_state,
    )

    state_root = Path(args.state_root)
    if args.command == "status":
        print(json.dumps(read_state(state_root, plan_id), sort_keys=True))
        return 0
    if args.command == "publish":
        state = publish_review(plan_id=plan_id, store_root=Path(args.store_root),
                               state_root=state_root, reports_root=Path(args.reports_root))
        print(json.dumps({"review_folder": state.get("review_folder"),
                          "review_error": state.get("review_error")}))
        return 0 if state.get("review_folder") else 1
    if args.command == "worker":
        try:
            state = run_plan(plan_id=plan_id, store_root=Path(args.store_root),
                             state_root=state_root, reports_root=Path(args.reports_root))
        except Exception:
            return 1
        return 0 if state.get("status") == "Completed" else 1
    existing = read_state(state_root, plan_id)
    if existing and existing.get("status") in ("Running", "Completed"):
        print(json.dumps({"refused": f"plan already {existing['status'].lower()}"}))
        return 1
    job_dir = state_root / plan_id
    job_dir.mkdir(parents=True, exist_ok=True)
    write_state(state_root, plan_id, status="Running", phase="queued")
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--plan-id", plan_id,
               "--store-root", args.store_root, "--state-root", str(state_root),
               "--reports-root", args.reports_root]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0)
    with (job_dir / "job.log").open("ab") as log:
        process = subprocess.Popen(  # noqa: S603 - exact interpreter + repo script
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    print(json.dumps({"plan_id": plan_id, "pid": process.pid}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
