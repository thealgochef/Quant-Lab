"""Evidence script (preserved; FINDING-6 resolution).

Scores every consecutive-5-store-day window in the accepted dataset's
post-warmup range by funnel-path coverage, producing
`WINDOW_COVERAGE_SCAN.json` for the owner's decision 21/R-5. Reads ONLY the
already-authorized accepted artifacts (no raw-source discovery). Run from the
Quant-Lab repo root:

    python QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/window_coverage_scan.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

import pandas as pd  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.config import (  # noqa: E402
    FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
    FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
)
from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.dataset import load_accepted_v2_tables  # noqa: E402

FSM_AUDIT_ARTIFACT_ID = "7e55ee89d9492fa8338cefa3c6389c9e80c4b139f1ec3b3a1a87ab7ac701fe41"


def main() -> None:
    root = Path("data/ifvg_datasets/v2") / FSM_AUDIT_ACCEPTED_V2_DATASET_ID / "exploration"
    tables = load_accepted_v2_tables(
        root,
        expected_dataset_id=FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
        expected_manifest_payload_sha256=FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    days = sorted(
        {entry[0].split("/")[0] for entry in manifest["identity"]["permitted_source_hashes"]}
    )

    def day_counts(table: RecordTable) -> dict[str, int]:
        frame = tables[table]
        if frame.empty or "envelope_trading_day" not in frame:
            return {}
        return frame.groupby(frame["envelope_trading_day"].astype(str)).size().to_dict()

    lifecycle = day_counts(RecordTable.SETUP_LIFECYCLE)
    candidates = day_counts(RecordTable.ENTRY_CANDIDATE)
    decisions = day_counts(RecordTable.ELIGIBLE_DECISION)
    trades = day_counts(RecordTable.EXECUTED_TRADE)

    audit_root = (
        Path("data/ifvg_datasets/fsm_audit/v1") / FSM_AUDIT_ARTIFACT_ID / "exploration"
    )
    funnel = pd.read_parquet(audit_root / "ifvg_audit_day_funnel.parquet")
    audit_days = set(funnel["source_date"].astype(str))

    rows = []
    non_warmup = [day for day in days if day >= "2026-01-13"]
    for start in range(len(non_warmup) - 4):
        window = non_warmup[start : start + 5]
        score = {
            "window_start": window[0],
            "window_end": window[-1],
            "days": window,
            "lifecycle": sum(lifecycle.get(day, 0) for day in window),
            "candidates": sum(candidates.get(day, 0) for day in window),
            "decisions": sum(decisions.get(day, 0) for day in window),
            "trades": sum(trades.get(day, 0) for day in window),
            "audit_day_coverage": sum(1 for day in window if day in audit_days),
        }
        score["paths_covered"] = sum(
            1 for key in ("lifecycle", "candidates", "decisions", "trades") if score[key] > 0
        )
        rows.append(score)
    rows.sort(
        key=lambda row: (
            row["paths_covered"],
            row["trades"],
            row["decisions"],
            row["candidates"],
            row["lifecycle"],
        ),
        reverse=True,
    )
    payload = {
        "scored_windows_desc": rows,
        "audit_day_universe": len(audit_days),
        "evidence_sources": {
            "v2_dataset": (
                f"{FSM_AUDIT_ACCEPTED_V2_DATASET_ID} (accepted final-review, manifest-verified)"
            ),
            "fsm_audit": f"{FSM_AUDIT_ARTIFACT_ID} (day_funnel.source_date)",
            "no_raw_source_reads": True,
        },
    }
    out = Path(__file__).resolve().parent / "WINDOW_COVERAGE_SCAN.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("windows scored:", len(rows), "; top:", rows[0]["window_start"], "..", rows[0]["window_end"])


if __name__ == "__main__":
    main()
