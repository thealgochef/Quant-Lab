"""Evidence script (preserved verbatim in intent; FINDING-6 resolution).

Builds `COVERAGE_MATRIX_PROPOSED.json` for the candidate verification
allowlist from the already-authorized accepted artifacts ONLY (no raw-source
discovery). Run from the Quant-Lab repo root:

    python QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R1/coverage_matrix_build.py
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
from alpha_lab.agents.data_infra.ifvg.dataset import load_accepted_v2_tables  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.search.verification import (  # noqa: E402
    PROPOSED_VERIFICATION_ALLOWLIST,
    build_verification_coverage_matrix,
)

FSM_AUDIT_ARTIFACT_ID = "7e55ee89d9492fa8338cefa3c6389c9e80c4b139f1ec3b3a1a87ab7ac701fe41"


def main() -> None:
    root = Path("data/ifvg_datasets/v2") / FSM_AUDIT_ACCEPTED_V2_DATASET_ID / "exploration"
    tables = load_accepted_v2_tables(
        root,
        expected_dataset_id=FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
        expected_manifest_payload_sha256=FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    permitted_days = tuple(
        sorted({entry[0].split("/")[0] for entry in manifest["identity"]["permitted_source_hashes"]})
    )
    audit_root = (
        Path("data/ifvg_datasets/fsm_audit/v1") / FSM_AUDIT_ARTIFACT_ID / "exploration"
    )
    funnel = pd.read_parquet(audit_root / "ifvg_audit_day_funnel.parquet")
    audit_counts = funnel.groupby(funnel["source_date"].astype(str)).size().to_dict()

    matrix = build_verification_coverage_matrix(
        candidate_allowlist=PROPOSED_VERIFICATION_ALLOWLIST,
        v2_tables=tables,
        evidence_source_dataset_id=FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
        evidence_source_manifest_sha256=FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
        permitted_source_days=permitted_days,
        audit_day_counts={str(k): int(v) for k, v in audit_counts.items()},
        replay_chart_days=None,
    )
    out = Path(__file__).resolve().parent / "COVERAGE_MATRIX_PROPOSED.json"
    out.write_text(
        json.dumps(matrix.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("coverage_matrix_id:", matrix.coverage_matrix_id)


if __name__ == "__main__":
    main()
