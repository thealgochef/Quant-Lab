"""Detached sequential R5–R6 research-group worker. Importing launches nothing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    from alpha_lab.agents.data_infra.ifvg.search.research_runs import (
        cancel_research_group,
        launch_research_group,
        read_research_group,
        run_research_group,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "start", "resume", "status", "cancel"))
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--state-root", required=True, type=Path)
    args = parser.parse_args(argv)
    action = {
        "worker": run_research_group,
        "start": launch_research_group,
        "resume": launch_research_group,
        "status": read_research_group,
        "cancel": cancel_research_group,
    }[args.command]
    result = action(args.store_root, args.group_id, args.state_root)
    print(json.dumps(result, sort_keys=True, default=str))
    return 1 if result.get("status") in ("failed", "blocked") else 0


if __name__ == "__main__":
    raise SystemExit(main())
