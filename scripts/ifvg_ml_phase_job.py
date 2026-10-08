"""Source-selected worker for the approved development-only IFSM ML phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research_workspace import require_external_work_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--stage", choices=("shadow", "fit", "policies"), required=True)
    parser.add_argument("--reference", choices=("MCB062", "MCB025"))
    parser.add_argument("--stop-after-days", type=int)
    args = parser.parse_args()
    try:
        require_external_work_paths(Path(__file__).resolve().parents[1], {
            "--store": args.store, "--work": args.work,
        })
    except ValueError as error:
        parser.error(str(error))
    sys.path[:0] = [str(args.core.resolve() / "src"), str(args.runtime.resolve() / "src")]
    from alpha_lab.propsim.funded.ml_phase.runner import fit_all, run_policies, run_shadow

    if args.stage == "fit":
        fit_all(store=args.store, plan_id=args.plan, work=args.work)
    else:
        if args.reference is None:
            parser.error("a shadow or policies stage requires --reference")
        function = run_shadow if args.stage == "shadow" else run_policies
        function(
            store=args.store,
            plan_id=args.plan,
            reference=args.reference,
            work=args.work,
            stop_after_days=args.stop_after_days,
        )


if __name__ == "__main__":
    main()
