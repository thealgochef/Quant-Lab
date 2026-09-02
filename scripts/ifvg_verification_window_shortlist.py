"""Rebuild the verification-window shortlist on LOGICAL trading days
(HARDENING-BACKEND Phase 3 §5.1; F-22).

Reads ONLY already-authorized immutable artifacts — the FSM-audit accepted
v2 dataset (manifest-verified via ``load_accepted_v2_tables``) and the
FSM-audit day funnel — exactly as ``../R1/window_coverage_scan.py`` did.
No raw source is discovered, listed or read; no allowlist is hashed into
``register_program_allowlist``; the owner's selection is NOT performed.

    python scripts/ifvg_verification_window_shortlist.py --out-dir <evidence folder>

Importing this module launches nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: The FSM-audit artifact whose day funnel provides audit-day coverage (the
#: same artifact R1 scored; its manifest is verified below).
FSM_AUDIT_ARTIFACT_ID = "7e55ee89d9492fa8338cefa3c6389c9e80c4b139f1ec3b3a1a87ab7ac701fe41"


def build(out_dir: Path, *, window_length: int = 5) -> dict:
    import pandas as pd  # noqa: PLC0415

    from alpha_lab.agents.data_infra.ifvg.config import (  # noqa: PLC0415
        FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
        FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
        FSM_AUDIT_DATASET_DIR,
        V2_DATASET_DIR,
    )
    from alpha_lab.agents.data_infra.ifvg.dataset import load_accepted_v2_tables  # noqa: PLC0415
    from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256  # noqa: PLC0415
    from alpha_lab.agents.data_infra.ifvg.search.trading_calendar import (  # noqa: PLC0415
        CME_GLOBEX_18ET_WEEKDAY_V1,
        EVIDENCE_VERIFIED_SESSIONS,
        PERMITTED_WINDOW_LAST_DAY,
        REGISTERED_FULL_CLOSURES,
        inventory_from_permitted_source_hashes,
        logical_trading_days,
    )
    from alpha_lab.agents.data_infra.ifvg.search.verification_window import (  # noqa: PLC0415
        build_logical_day_coverage,
        build_verification_window_shortlist,
        render_shortlist_markdown,
        shortlist_document,
    )

    dataset_root = V2_DATASET_DIR / FSM_AUDIT_ACCEPTED_V2_DATASET_ID / "exploration"
    tables = load_accepted_v2_tables(
        dataset_root,
        expected_dataset_id=FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
        expected_manifest_payload_sha256=FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
    )
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    inventory = inventory_from_permitted_source_hashes(
        manifest["identity"]["permitted_source_hashes"]
    )
    audit_root = FSM_AUDIT_DATASET_DIR / FSM_AUDIT_ARTIFACT_ID / "exploration"
    audit_manifest = json.loads((audit_root / "manifest.json").read_text(encoding="utf-8"))
    core = {k: v for k, v in audit_manifest.items() if k != "manifest_payload_sha256"}
    if audit_manifest.get("manifest_payload_sha256") != canonical_sha256(core):
        raise SystemExit("the FSM-audit artifact manifest does not hash its payload; refused")
    funnel = pd.read_parquet(audit_root / "ifvg_audit_day_funnel.parquet")
    funnel_days = frozenset(funnel["source_date"].astype(str))
    first_logical = logical_trading_days("2026-01-01", "2026-01-07")[0]
    logical_days = logical_trading_days(first_logical, PERMITTED_WINDOW_LAST_DAY)
    coverage = build_logical_day_coverage(
        logical_days=logical_days,
        inventory=inventory,
        tables=tables,
        funnel_days=funnel_days,
        policy=CME_GLOBEX_18ET_WEEKDAY_V1,
    )
    shortlist = build_verification_window_shortlist(
        coverage,
        inventory=inventory,
        evidence_source_dataset_id=FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
        evidence_source_manifest_sha256=FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
        audit_artifact_id=FSM_AUDIT_ARTIFACT_ID,
        window_length=window_length,
    )
    document = shortlist_document(
        shortlist,
        generated_at_utc=datetime.now(UTC).isoformat(),
        owner_selection="NOT PERFORMED",
        register_program_allowlist_called=False,
        no_raw_source_reads=True,
        evidence_sources={
            "v2_dataset": (
                f"{FSM_AUDIT_ACCEPTED_V2_DATASET_ID} (FSM-audit accepted final-review dataset, "
                f"manifest {FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256}, verified by "
                "load_accepted_v2_tables)"
            ),
            "fsm_audit": (
                f"{FSM_AUDIT_ARTIFACT_ID} (ifvg_audit_day_funnel.source_date; manifest hash "
                "verified)"
            ),
            "inventory": "manifest identity.permitted_source_hashes (no raw-source reads)",
        },
        calendar={
            "policy_id": CME_GLOBEX_18ET_WEEKDAY_V1.policy_id,
            "registered_full_closures": [list(item) for item in REGISTERED_FULL_CLOSURES],
            "evidence_verified_sessions": dict(EVIDENCE_VERIFIED_SESSIONS),
            "logical_day_count": len(logical_days),
            "physical_partition_count": len(inventory),
            "audit_funnel_day_count": len(funnel_days),
            "audit_funnel_days_not_logical": sorted(
                day for day in funnel_days if day not in set(logical_days)
            ),
            "logical_days_without_audit_funnel": sorted(set(logical_days) - funnel_days),
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "LOGICAL_WINDOW_COVERAGE_SCAN.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "VERIFICATION_WINDOW_SHORTLIST.md").write_text(
        render_shortlist_markdown(shortlist), encoding="utf-8"
    )
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(
            ROOT / "QL-FSM-PROP-SEARCH-DASHBOARD" / "implementation-progress" / "HARDENING-BACKEND"
        ),
    )
    parser.add_argument("--window-length", type=int, default=5)
    args = parser.parse_args(argv)
    document = build(Path(args.out_dir), window_length=int(args.window_length))
    shortlist = document["shortlist"]
    print(
        json.dumps(
            {
                "shortlist_id": document["shortlist_id"],
                "candidate_windows": shortlist["candidate_window_count"],
                "eligible_windows": shortlist["eligible_window_count"],
                "entries": [
                    {
                        "label": entry["label"],
                        "rank": entry["rank"],
                        "days": entry["window"]["days"],
                        "eligible": entry["window"]["eligible"],
                        "rank_key": entry["window"]["rank_key"],
                        "seed_chain_replay_day_count": entry["window"][
                            "seed_chain_replay_day_count"
                        ],
                    }
                    for entry in shortlist["entries"]
                ],
                "owner_selection": "NOT PERFORMED",
                "register_program_allowlist_called": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
