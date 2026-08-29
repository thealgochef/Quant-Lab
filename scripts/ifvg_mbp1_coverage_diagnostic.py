"""Bounded real-data MBP-1 coverage diagnostic CLI (R5B.1; owner Q1 item 6).

Characterizes the ACTUAL partition-quality evidence of the one canonical
authorized ≤5-day fixture under coverage policy v2 and persists an immutable
``Mbp1CoverageDiagnosticReport`` into the verification namespace. It never
infers completeness from raw sequence continuity.

The real run is an owner action gated by the R1 real-slice gate: it refuses
— before any source path is constructed — without a persisted, verified
``VerificationRunEnvelope`` whose ``VerificationAuthorizationRef`` binds the
requested allowlist, the verification policy, a verified coverage-matrix
artifact over exactly those days, and the ONE canonical program allowlist.
Partition-scope evidence (verified gap manifests + typed recovery /
condition records) enters only through ``--evidence-json`` (a JSON file of
STORE IDS and typed records — never a path); without it every partition is
``completeness_unknown`` and no interval fact is computed. Importing this
module launches nothing; ``--synthetic-source-artifact-id`` runs the report
shape over an already-persisted synthetic source artifact (fixture
provenance only) and is the only path that executes without the owner's
authorization.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _verification_namespace(store_root: Path) -> Path:
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_diagnostic import (  # noqa: PLC0415
        assert_verification_namespace,
    )

    try:
        return assert_verification_namespace(Path(store_root))
    except PermissionError as error:
        raise SystemExit(str(error)) from error


def _synthetic(args) -> int:
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_diagnostic import (  # noqa: PLC0415
        build_mbp1_coverage_diagnostic,
        save_mbp1_coverage_diagnostic,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_evidence import (  # noqa: PLC0415
        Mbp1EvidenceProvenance,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (  # noqa: PLC0415
        load_mbp1_source_artifact,
    )

    store_root = _verification_namespace(Path(args.store_root))
    source = load_mbp1_source_artifact(store_root, args.synthetic_source_artifact_id)
    provenances = {
        row.evidence_provenance for row in source.payload.ordered_partitions
    }
    # a SYNTHETIC artifact stores its canonical event bytes (real artifacts
    # never do) and carries no owner-reviewed evidence
    if not source.events_stored or Mbp1EvidenceProvenance.OWNER_REVIEWED in provenances:
        raise SystemExit(
            "the synthetic path accepts synthetic-fixture source artifacts only; "
            "a real source artifact requires the authorized real path"
        )
    days = tuple(sorted({row.trading_day for row in source.payload.ordered_partitions}))
    report = build_mbp1_coverage_diagnostic(
        source,
        run_scope="synthetic_fixture",
        allowlist=days,
        authorization_content_hash=None,
    )
    save_mbp1_coverage_diagnostic(store_root, report)
    print(
        json.dumps(
            {
                "status": "synthetic_shape_persisted",
                "mbp1_coverage_diagnostic_id": report.mbp1_coverage_diagnostic_id,
                "rows": len(report.payload.rows),
            }
        )
    )
    return 0


def _real(args) -> int:
    """The owner-authorized path: fail-before-path without the persisted
    VerificationRunEnvelope + authorization + coverage matrix + canonical
    allowlist; the source reads then run through the VerificationReplayPolicy only."""

    from alpha_lab.agents.data_infra.ifvg.development_access import (  # noqa: PLC0415
        VerificationReplayPolicy,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_diagnostic import (  # noqa: PLC0415
        assert_diagnostic_authorized,
        build_mbp1_coverage_diagnostic,
        load_partition_evidence_manifest,
        save_mbp1_coverage_diagnostic,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        load_verified_envelope,
    )
    from alpha_lab.agents.data_infra.ifvg.search.verification import (  # noqa: PLC0415
        VerificationRunEnvelope,
    )

    store_root = _verification_namespace(Path(args.store_root))
    if not args.verification_run_id or not _HEX64.match(args.verification_run_id):
        raise SystemExit(
            "the real diagnostic requires --verification-run-id (the persisted, "
            "owner-authorized VerificationRunEnvelope); it does not exist yet — "
            "refused before any source path"
        )
    run = load_verified_envelope(
        store_root, "verification_runs", args.verification_run_id, VerificationRunEnvelope
    )
    days = tuple(run.payload.allowlist)
    policy = VerificationReplayPolicy(days)
    try:
        authorization_hash = assert_diagnostic_authorized(
            store_root=store_root,
            run_envelope=run,
            access_policy=policy,
            allowlist=days,
        )
    except PermissionError as error:
        raise SystemExit(str(error)) from error
    # Partition-scope evidence is an owner-reviewed input loaded ONLY from the
    # verified stores (manifest ids + typed records; never a path). Without
    # it the diagnostic characterizes the partitions as completeness_unknown
    # and computes no interval fact — it never invents positive evidence.
    coverage_evidence = None
    manifest_ids: tuple[str, ...] = ()
    if args.evidence_json:
        try:
            manifest = json.loads(Path(args.evidence_json).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"evidence manifest is unreadable: {error}") from error
        coverage_evidence, manifest_ids = load_partition_evidence_manifest(store_root, manifest)
        unknown = sorted(set(coverage_evidence) - set(days))
        if unknown:
            raise SystemExit(f"evidence manifest names days outside the allowlist: {unknown}")
    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig  # noqa: PLC0415
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (  # noqa: PLC0415
        build_mbp1_source_artifact_from_paths,
        save_mbp1_source_artifact,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (  # noqa: PLC0415
        MIN_DAY_COVERAGE_FRACTION,
        R5B_WINDOW_SPECS,
        Mbp1SourceContract,
    )

    config = IfvgCaptureConfig(data_dir=ROOT / "data/databento")
    contract = Mbp1SourceContract(
        instrument="NQ",
        contract_roll_policy_id="front_month_open_interest_roll_v1",
        feature_window_specs=R5B_WINDOW_SPECS,
        coverage_policy={"min_day_coverage_fraction": MIN_DAY_COVERAGE_FRACTION},
    )
    envelope, _bytes = build_mbp1_source_artifact_from_paths(
        days,
        access_policy=policy,
        path_factory=lambda day: config.data_dir / "NQ" / day / "mbp1.parquet",
        contract=contract,
        authorized_date_set_id=run.payload.allowlist_hash,
        coverage_evidence=coverage_evidence,
    )
    policy.assert_zero_forbidden_access()
    save_mbp1_source_artifact(store_root, envelope, {})
    report = build_mbp1_coverage_diagnostic(
        envelope,
        run_scope="verification_5d",
        allowlist=days,
        authorization_content_hash=authorization_hash,
        partition_evidence_manifest_ids=manifest_ids,
    )
    save_mbp1_coverage_diagnostic(store_root, report)
    print(
        json.dumps(
            {
                "status": "verification_diagnostic_persisted",
                "mbp1_coverage_diagnostic_id": report.mbp1_coverage_diagnostic_id,
                "partition_evidence_manifest_ids": list(manifest_ids),
                "access_audit": policy.audit_dict(),
            },
            default=str,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-root", default=str(ROOT / "data/ifvg_datasets/search_test/v1")
    )
    parser.add_argument("--verification-run-id", default=None)
    parser.add_argument("--synthetic-source-artifact-id", default=None)
    parser.add_argument(
        "--evidence-json",
        default=None,
        help=(
            "JSON of store-verified partition evidence per trading day: "
            '{"<day>": [{"manifest_id": "<64-hex>", "channel_map_verified": false, '
            '"recovery_boundaries": [...], "dataset_condition": {...}|null}, ...]}'
        ),
    )
    args = parser.parse_args(argv)
    if args.synthetic_source_artifact_id:
        if not _HEX64.match(args.synthetic_source_artifact_id):
            raise SystemExit("synthetic source artifact id must be 64-hex")
        return _synthetic(args)
    return _real(args)


if __name__ == "__main__":
    raise SystemExit(main())
