"""Funded configuration comparison job: start a detached worker for one frozen plan.

    python scripts/ifvg_funded_comparison_job.py start --plan-id <id> --store-root ...
    python scripts/ifvg_funded_comparison_job.py status --plan-id <id> --state-root ...
    python scripts/ifvg_funded_comparison_job.py publish --plan-id <id> ...  (retry export)
    python scripts/ifvg_funded_comparison_job.py republish --plan-id <id>
        [--review-findings FILE.md] [--approximated-minutes FILE.csv]
        [--reference-reconciliation FILE.csv]  (new export version, same result)

The worker reads only the verified plan envelope and its stored owner approval.
It never launches other studies, purchases, withdrawals or live orders. Each
configuration runs in its own process; ``--workers`` bounds the process count.
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
    parser.add_argument("command",
                        choices=("start", "status", "worker", "publish", "republish"))
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--store-root", default=str(ROOT / "data/ifsm_ui_replication/search/v1"))
    parser.add_argument("--state-root",
                        default=str(ROOT / "data/ifsm_ui_replication/funded_comparison_jobs"))
    parser.add_argument("--reports-root", default=str(ROOT / "reports"))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--review-findings", default=None,
                        help="recorded independent review findings (Markdown)")
    parser.add_argument("--approximated-minutes", default=None,
                        help="compact substituted-minute analysis (CSV)")
    parser.add_argument("--reference-reconciliation", default=None,
                        help="reference-record reconciliation (CSV)")
    args = parser.parse_args(argv)
    plan_id = _plan_id(args.plan_id)
    from alpha_lab.propsim.funded.comparison_runner import (
        publish_comparison_review,
        run_comparison_plan,
    )
    from alpha_lab.propsim.funded.runner import read_state, write_state

    state_root = Path(args.state_root)
    if args.command == "status":
        print(json.dumps(read_state(state_root, plan_id), sort_keys=True))
        return 0
    if args.command == "republish":
        import csv

        from alpha_lab.propsim.funded.comparison_runner import republish_comparison_review

        extra = {}
        for key, value in (("approximated_minutes", args.approximated_minutes),
                           ("reference_record_reconciliation", args.reference_reconciliation)):
            if value:
                with Path(value).open(encoding="utf-8", newline="") as handle:
                    extra[key] = list(csv.DictReader(handle))
        findings = (Path(args.review_findings).read_text(encoding="utf-8")
                    if args.review_findings else None)
        out = republish_comparison_review(
            plan_id=plan_id, store_root=Path(args.store_root), state_root=state_root,
            reports_root=Path(args.reports_root), review_findings=findings,
            extra_supplements=extra)
        print(json.dumps(out, sort_keys=True))
        return 0
    if args.command == "publish":
        state = publish_comparison_review(plan_id=plan_id, store_root=Path(args.store_root),
                                          state_root=state_root,
                                          reports_root=Path(args.reports_root))
        print(json.dumps({"review_folder": state.get("review_folder"),
                          "review_error": state.get("review_error")}))
        return 0 if state.get("review_folder") else 1
    if args.command == "worker":
        try:
            state = run_comparison_plan(plan_id=plan_id, store_root=Path(args.store_root),
                                        state_root=state_root,
                                        reports_root=Path(args.reports_root),
                                        workers=args.workers)
        except Exception:
            return 1
        return 0 if state.get("status") == "Completed" else 1
    existing = read_state(state_root, plan_id)
    if existing and existing.get("status") in ("Running", "Completed"):
        print(json.dumps({"refused": f"plan already {existing['status'].lower()}"}))
        return 1
    env, core_root = worker_environment(Path(args.store_root), plan_id)
    if env is None:
        print(json.dumps({"refused": "no local Strategy-Core checkout has the exact source "
                                     "this plan froze (commit and uncommitted-change hash)"}))
        return 1
    job_dir = state_root / plan_id
    job_dir.mkdir(parents=True, exist_ok=True)
    write_state(state_root, plan_id, status="Running", phase="queued", kind="funded_comparison",
                core_source_root=None if core_root is None else str(core_root))
    command = [sys.executable, str(Path(__file__).resolve()), "worker", "--plan-id", plan_id,
               "--store-root", args.store_root, "--state-root", str(state_root),
               "--reports-root", args.reports_root]
    if args.workers:
        command += ["--workers", str(args.workers)]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0)
    with (job_dir / "job.log").open("ab") as log:
        process = subprocess.Popen(  # noqa: S603 - exact interpreter + repo script
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags,
            env=env)
    print(json.dumps({"plan_id": plan_id, "pid": process.pid,
                      "core_source_root": None if core_root is None else str(core_root)}))
    return 0


def worker_environment(store_root: Path, plan_id: str):
    """(environment, research Core root) for the worker; (None, None) when refused.

    A version-2 plan runs on the one local checkout whose commit and uncommitted
    source hash equal the plan's frozen ``core_source`` (the worker re-checks it).
    A version-1 plan keeps this process's environment (the application's Core).
    """

    import os

    from alpha_lab.propsim.funded.comparison_runner import is_v2, load_plan
    from alpha_lab.propsim.funded.research_core_sources import find_core_checkout

    env = dict(os.environ)
    plan = load_plan(store_root, plan_id)
    if not is_v2(plan):
        return env, None
    core = find_core_checkout(plan.core_source)
    if core is None:
        return None, None
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in (ROOT / "src", ROOT / "scripts",
                                                         core / "src"))
    env["IFSM_RESEARCH_CORE"] = str(core)
    return env, core


if __name__ == "__main__":
    raise SystemExit(main())
