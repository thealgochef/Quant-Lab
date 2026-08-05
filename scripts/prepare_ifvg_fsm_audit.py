"""Prepare the immutable ``ifvg_fsm_audit_v1`` companion artifact.

Runs the audit-enabled replay over the fixed permitted development chain
(cached day artifacts only, by default), gates on EXACT v2 parity against the
accepted final-review dataset, and publishes atomically. Writes the deep
parity report to ``reports/ifvg_fsm_audit/IFVG_FSM_AUDITABILITY_PARITY_REPORT.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_lab.agents.data_infra.ifvg.fsm_audit_preparation import (  # noqa: E402
    prepare_ifvg_fsm_audit_persisted,
    summarize_result,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="ifvg_v2_doc_default_fresh_static_1r")
    parser.add_argument(
        "--allow-rebuild-day-artifacts",
        action="store_true",
        help="Allow rebuilding authorized intermediate day artifacts.",
    )
    parser.add_argument(
        "--parity-report",
        default=str(
            ROOT / "reports" / "ifvg_fsm_audit" / "IFVG_FSM_AUDITABILITY_PARITY_REPORT.json"
        ),
    )
    args = parser.parse_args()

    def progress(completed: int, total: int, day: str) -> None:
        print(
            json.dumps({"completed": completed, "total": total, "source_date": day}),
            flush=True,
        )

    result = prepare_ifvg_fsm_audit_persisted(
        repo_root=ROOT,
        profile_name=args.profile,
        cached_artifacts_only=not args.allow_rebuild_day_artifacts,
        parity_report_path=Path(args.parity_report),
        progress_fn=progress,
    )
    print(summarize_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
