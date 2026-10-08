"""Worker entry for the exact, approved six-configuration full-range batch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from research_workspace import require_external_work_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "resume"))
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    try:
        require_external_work_paths(Path(__file__).resolve().parents[1], {
            "--store-root": args.store_root, "--state-root": args.state_root,
        })
    except ValueError as error:
        parser.error(str(error))
    core_src = str(args.core_root.resolve() / "src")
    lab_src = str(Path(__file__).resolve().parents[1] / "src")
    # Spawned Windows workers receive exactly this same source ordering.
    inherited = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    os.environ["PYTHONPATH"] = os.pathsep.join([core_src, lab_src, *inherited])
    sys.path[:0] = [core_src, lab_src]
    from alpha_lab.propsim.funded.full_range_batch import run_full_range_batch

    run_full_range_batch(
        plan_id=args.plan_id,
        store_root=args.store_root,
        state_root=args.state_root,
        reports_root=args.reports_root,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
