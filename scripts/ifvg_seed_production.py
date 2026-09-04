"""Seed-production CLI (HARDENING-BACKEND Phase 3 §5.3–§5.5; F-16 / F-21).

Subcommands:

* ``packet`` — write the UNSIGNED seed-production authorization packet
  (placeholders that fail validation; nothing is persisted to a store);
* ``verify-authorization`` — verify a persisted authorization against the
  current store namespace, head witness, profile, chain and inventory
  BEFORE any source path exists (reports the typed verdict);
* ``run`` — run ONE authorized seed-production chain: requires a persisted,
  signed authorization id; synthetic provenance is refused outside test
  namespaces; the only outputs are the seed snapshot, the access audit and
  the run receipt.
* ``register-authorization`` (UI-2) — persist a COMPLETED packet (the owner
  filled ``approved_by`` / ``approved_at`` / ``effective_from`` /
  ``owner_decision_refs`` outside this workspace) through the backend's
  ``persist_seed_production_authorization`` seam: a packet that still carries
  a placeholder is the typed refusal ``packet_not_signed``; every backend
  refusal (namespace, witness, provenance confinement) keeps its reason.

``--receipt-out PATH`` (UI-2; ``register-authorization`` and ``run``) writes
the printed result JSON to a file as well — the Verification Center's refresh
picks the exact ids up from there (a refusal writes nothing).

Importing this module launches nothing. Exit codes: 0 ok, 2 typed refusal,
1 unexpected error.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_DEFAULT_PROFILE = "ifvg_v2_doc_default_fresh_static_1r"


def _load_inventory(path: Path) -> dict[str, tuple[str, str]]:
    from alpha_lab.agents.data_infra.ifvg.search.trading_calendar import (  # noqa: PLC0415
        inventory_from_permitted_source_hashes,
    )

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(document, dict) and "identity" in document:
        document = document["identity"]["permitted_source_hashes"]
    if isinstance(document, list):
        return inventory_from_permitted_source_hashes(document)
    if isinstance(document, dict):
        return {str(day): (str(entry[0]), str(entry[1])) for day, entry in document.items()}
    raise SystemExit("the inventory JSON must be a manifest, a permitted-hash list, or a day map")


def _refused(reason: str, detail: str) -> int:
    print(json.dumps({"status": "refused", "reason": reason, "detail": detail}, sort_keys=True))
    return 2


def _emit(payload: dict, receipt_out: str | None) -> None:
    """Print the result JSON and, when asked, write it as the receipt file
    (atomically; the Verification Center's refresh reads it)."""

    text = json.dumps(payload, sort_keys=True)
    print(text)
    if receipt_out:
        path = Path(receipt_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text + "\n", encoding="utf-8")
        tmp.replace(path)


def _register(args) -> int:
    from alpha_lab.agents.data_infra.ifvg.search.seed_production import (  # noqa: PLC0415
        OWNER_PLACEHOLDER,
        SeedProductionAuthorizationError,
        SeedProductionAuthorizationPayload,
        persist_seed_production_authorization,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import has_envelope  # noqa: PLC0415

    document = json.loads(Path(args.packet_json).read_text(encoding="utf-8"))
    payload_document = document.get("payload") if isinstance(document, dict) else None
    if not isinstance(payload_document, dict):
        payload_document = document if isinstance(document, dict) else None
    if not isinstance(payload_document, dict):
        return _refused("packet_malformed", "the packet JSON carries no payload object")
    if any(
        isinstance(value, str) and OWNER_PLACEHOLDER in value
        for value in payload_document.values()
    ) or any(
        OWNER_PLACEHOLDER in str(ref) for ref in payload_document.get("owner_decision_refs", ())
    ):
        return _refused(
            "packet_not_signed",
            "the packet still carries owner placeholders (approved_by / approved_at / "
            "effective_from / owner_decision_refs); only the owner completes them",
        )
    try:
        payload = SeedProductionAuthorizationPayload.model_validate(payload_document)
    except ValueError as error:
        return _refused("packet_invalid", str(error))
    root = Path(args.store_root)
    try:
        from alpha_lab.agents.data_infra.ifvg.search.identities import (  # noqa: PLC0415
            canonical_contract_sha256,
        )

        expected_id = canonical_contract_sha256(payload)
        reused = has_envelope(root, "seed_production_authorizations", expected_id)
        envelope = persist_seed_production_authorization(root, payload)
    except SeedProductionAuthorizationError as error:
        return _refused(error.reason, str(error))
    _emit(
        {
            "status": "authorization_registered",
            "seed_production_authorization_id": envelope.seed_production_authorization_id,
            "provenance": envelope.payload.provenance,
            "seed_chain_replay_day_count": len(envelope.payload.ordered_seed_chain_replay_days),
            "first_intended_verification_day": envelope.payload.first_intended_verification_day,
            "reused": bool(reused),
        },
        args.receipt_out,
    )
    return 0


def _packet(args) -> int:
    from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config  # noqa: PLC0415
    from alpha_lab.agents.data_infra.ifvg.search.seed_production import (  # noqa: PLC0415
        SeedProductionAuthorizationError,
        build_seed_production_packet,
        render_seed_production_packet_markdown,
    )

    resolved = resolve_profile_config({"profile_name": args.profile_name})
    try:
        packet = build_seed_production_packet(
            Path(args.store_root),
            baseline_profile_name=args.profile_name,
            resolved_section_config_hash=resolved.section_config_hash,
            first_intended_verification_day=args.first_verification_day,
            inventory=_load_inventory(Path(args.inventory_json)),
            quant_lab_source_identity=args.quant_lab_source_identity,
            strategy_core_commit=args.strategy_core_commit,
            strategy_core_source_identity=args.strategy_core_source_identity,
        )
    except SeedProductionAuthorizationError as error:
        return _refused(error.reason, str(error))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "SEED_PRODUCTION_PACKET.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "SEED_PRODUCTION_PACKET.md").write_text(
        render_seed_production_packet_markdown(
            packet, title="Seed-production authorization packet (UNSIGNED)"
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "unsigned_packet_written",
                "seed_chain_replay_day_count": packet["seed_chain_replay_day_count"],
                "logical_trading_day_count": packet["logical_trading_day_count"],
                "out_dir": str(out_dir),
                "separately_authorized_preparation": True,
            },
            sort_keys=True,
        )
    )
    return 0


def _verify(args) -> int:
    from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config  # noqa: PLC0415
    from alpha_lab.agents.data_infra.ifvg.search.seed_production import (  # noqa: PLC0415
        SeedProductionAuthorizationError,
        _load_authorization,
        seed_chain_source_inventory_hash,
        verify_seed_production_authorization,
    )

    root = Path(args.store_root)
    resolved = resolve_profile_config({"profile_name": args.profile_name})
    try:
        envelope = _load_authorization(root, args.authorization_id)
        chain = envelope.payload.ordered_seed_chain_replay_days
        inventory_hash = seed_chain_source_inventory_hash(
            chain, _load_inventory(Path(args.inventory_json))
        )
        verified = verify_seed_production_authorization(
            root,
            envelope,
            expected_profile_name=args.profile_name,
            expected_section_config_hash=resolved.section_config_hash,
            expected_chain_replay_days=chain,
            expected_source_inventory_hash=inventory_hash,
            expected_quant_lab_source_identity=args.quant_lab_source_identity,
            expected_strategy_core=(
                (args.strategy_core_commit, args.strategy_core_source_identity)
                if args.strategy_core_commit and args.strategy_core_source_identity
                else None
            ),
            now=datetime.now(UTC).isoformat(),
        )
    except SeedProductionAuthorizationError as error:
        return _refused(error.reason, str(error))
    except ValueError as error:
        return _refused("source_inventory_mismatch", str(error))
    print(
        json.dumps(
            {
                "status": "authorization_verified",
                "seed_production_authorization_id": verified.seed_production_authorization_id,
                "provenance": verified.payload.provenance,
                "seed_chain_replay_day_count": len(verified.payload.ordered_seed_chain_replay_days),
                "first_intended_verification_day": verified.payload.first_intended_verification_day,
            },
            sort_keys=True,
        )
    )
    return 0


def _run(args) -> int:
    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig  # noqa: PLC0415
    from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config  # noqa: PLC0415
    from alpha_lab.agents.data_infra.ifvg.search.seed_production import (  # noqa: PLC0415
        SeedProductionAuthorizationError,
        run_seed_production_chain,
    )

    root = Path(args.store_root)
    resolved = resolve_profile_config({"profile_name": args.profile_name})
    cfg = replace(
        IfvgCaptureConfig(),
        section=resolved.section,
        data_dir=Path(args.data_dir) if args.data_dir else IfvgCaptureConfig().data_dir,
    )
    try:
        result = run_seed_production_chain(
            root=root,
            authorization_id=args.authorization_id,
            cfg=cfg,
            resolved_profile=resolved,
            source_inventory=_load_inventory(Path(args.inventory_json)),
            now=datetime.now(UTC).isoformat(),
            repo_root=Path(args.repo_root) if args.repo_root else None,
            artifact_provenance_dates=tuple(args.artifact_provenance_dates or ()),
            cached_artifacts_only=not args.allow_artifact_rebuild,
        )
    except SeedProductionAuthorizationError as error:
        return _refused(error.reason, str(error))
    except PermissionError as error:
        return _refused("access_refused", str(error))
    payload = result.receipt.payload
    _emit(
        {
            "status": "seed_production_completed"
            if not result.reused
            else "seed_production_reused",
            "seed_snapshot_id": result.snapshot.seed_snapshot_id,
            "seed_production_run_id": result.receipt.seed_production_run_id,
            "seed_chain_replay_day_count": payload.chain_replay_day_count,
            "logical_trading_day_count": payload.logical_trading_day_count,
            "first_intended_verification_day": payload.first_intended_verification_day,
            "separately_authorized_preparation": True,
            "verification_evidence_footprint_days": 0,
            "stores_written": list(payload.stores_written),
        },
        args.receipt_out,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    packet = sub.add_parser("packet", help="write the UNSIGNED seed-production packet")
    packet.add_argument("--store-root", required=True)
    packet.add_argument("--profile-name", default=_DEFAULT_PROFILE)
    packet.add_argument("--first-verification-day", required=True)
    packet.add_argument("--inventory-json", required=True)
    packet.add_argument("--quant-lab-source-identity", required=True)
    packet.add_argument("--strategy-core-commit", required=True)
    packet.add_argument("--strategy-core-source-identity", required=True)
    packet.add_argument("--out-dir", required=True)

    verify = sub.add_parser("verify-authorization", help="verify a persisted authorization")
    verify.add_argument("--store-root", required=True)
    verify.add_argument("--authorization-id", required=True)
    verify.add_argument("--profile-name", default=_DEFAULT_PROFILE)
    verify.add_argument("--inventory-json", required=True)
    verify.add_argument("--quant-lab-source-identity", default=None)
    verify.add_argument("--strategy-core-commit", default=None)
    verify.add_argument("--strategy-core-source-identity", default=None)

    run = sub.add_parser("run", help="run ONE authorized seed-production chain")
    run.add_argument("--store-root", required=True)
    run.add_argument("--authorization-id", required=True)
    run.add_argument("--profile-name", default=_DEFAULT_PROFILE)
    run.add_argument("--inventory-json", required=True)
    run.add_argument("--data-dir", default=None)
    run.add_argument("--repo-root", default=None)
    run.add_argument("--artifact-provenance-dates", nargs="*", default=None)
    run.add_argument(
        "--allow-artifact-rebuild",
        action="store_true",
        help="build missing day artifacts from the authorized source partitions "
        "(default: cached artifacts only)",
    )
    run.add_argument(
        "--receipt-out",
        default=None,
        help="also write the result JSON to this file (the Verification Center's "
        "refresh pickup); nothing is written on a refusal",
    )

    register = sub.add_parser(
        "register-authorization",
        help="persist a COMPLETED seed-production packet (owner fields filled)",
    )
    register.add_argument("--store-root", required=True)
    register.add_argument("--packet-json", required=True)
    register.add_argument("--receipt-out", default=None)

    args = parser.parse_args(argv)
    if args.command == "packet":
        return _packet(args)
    if args.command == "verify-authorization":
        return _verify(args)
    if args.command == "register-authorization":
        return _register(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
