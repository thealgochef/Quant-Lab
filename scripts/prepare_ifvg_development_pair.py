"""Prepare the fixed, nonsealed IFVG development artifact pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_lab.agents.data_infra.ifvg.preparation import (  # noqa: E402
    prepare_ifvg_development_pair_persisted,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="ifvg_v2_doc_default_fresh_static_1r",
    )
    parser.add_argument(
        "--allow-rebuild-day-artifacts",
        action="store_true",
        help="Allow rebuilding authorized intermediate day artifacts.",
    )
    args = parser.parse_args()

    def progress(completed: int, total: int, day: str) -> None:
        print(json.dumps({"completed": completed, "total": total, "source_date": day}))

    prepared = prepare_ifvg_development_pair_persisted(
        repo_root=ROOT,
        profile_name=args.profile,
        cached_artifacts_only=not args.allow_rebuild_day_artifacts,
        progress_fn=progress,
    )
    print(
        json.dumps(
            {
                "status": prepared.preparation_state.status.value,
                "v2_artifact_id": prepared.pair.v2.reference.artifact_id,
                "v3_artifact_id": prepared.pair.v3.reference.artifact_id,
                "protected_counters": prepared.access_audit["protected_counters"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
