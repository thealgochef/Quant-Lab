"""UI-2: the seed-production CLI's operator seams the Verification Center
relies on — ``register-authorization`` (persist a COMPLETED packet; a
placeholder-bearing packet is a typed refusal) and ``--receipt-out`` (the
exact ids written for the center's refresh pickup). Nothing here signs an
owner artifact for real or replays a chain: the completed packet is a test
fixture in an isolated ``test`` namespace; ``run`` is proven to refuse before
any path and to write no receipt on refusal."""

from __future__ import annotations

import importlib
import json

from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.seed_production import (
    SEED_PRODUCTION_AUTHORIZATION_STORE,
    SeedProductionAuthorizationEnvelope,
    build_seed_production_packet,
)
from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import initialize_test_namespace
from tests.agents.ifvg_search.verification_center_fixture import (
    PROFILE,
    QL_IDENTITY,
    SC_COMMIT,
    SC_IDENTITY,
    WINDOW,
    inventory_for_chain,
    write_inventory_manifest,
)


def _packet(root, inventory) -> dict:
    return build_seed_production_packet(
        root,
        baseline_profile_name=PROFILE,
        resolved_section_config_hash=resolve_profile_config(
            {"profile_name": PROFILE}
        ).section_config_hash,
        first_intended_verification_day=WINDOW[0],
        inventory=inventory,
        quant_lab_source_identity=QL_IDENTITY,
        strategy_core_commit=SC_COMMIT,
        strategy_core_source_identity=SC_IDENTITY,
    )


def test_register_authorization_persists_a_completed_packet_and_writes_the_receipt(
    tmp_path, capsys
) -> None:
    script = importlib.import_module("scripts.ifvg_seed_production")
    root = tmp_path / "store"
    initialize_test_namespace(root)
    inventory = inventory_for_chain()
    packet = _packet(root, inventory)
    unsigned_path = tmp_path / "SEED_PRODUCTION_PACKET.json"
    unsigned_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    receipt_path = tmp_path / "center" / "seed_production_authorization.receipt.json"
    # an UNSIGNED packet (owner placeholders) is a typed refusal; no receipt is written
    code = script.main(
        [
            "register-authorization",
            "--store-root",
            str(root),
            "--packet-json",
            str(unsigned_path),
            "--receipt-out",
            str(receipt_path),
        ]
    )
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 2 and printed["status"] == "refused"
    assert printed["reason"] == "packet_not_signed"
    assert not receipt_path.exists()
    # the owner completes the fields (a test fixture — no real signature exists)
    completed = json.loads(json.dumps(packet))
    completed["payload"].update(
        {
            "approved_by": "test-owner",
            "approved_at": "2026-09-04T00:00:00+00:00",
            "effective_from": "2026-09-04T00:00:00+00:00",
            "owner_decision_refs": ["21/R-5:test-fixture"],
        }
    )
    signed_path = tmp_path / "SEED_PRODUCTION_PACKET.signed.json"
    signed_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    code = script.main(
        [
            "register-authorization",
            "--store-root",
            str(root),
            "--packet-json",
            str(signed_path),
            "--receipt-out",
            str(receipt_path),
        ]
    )
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0, printed
    assert printed["status"] == "authorization_registered"
    assert printed["reused"] is False
    authorization_id = printed["seed_production_authorization_id"]
    assert len(authorization_id) == 64
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt == printed
    stored = load_verified_envelope(
        root,
        SEED_PRODUCTION_AUTHORIZATION_STORE,
        authorization_id,
        SeedProductionAuthorizationEnvelope,
    )
    assert stored.payload.provenance == "owner_signed"
    assert stored.payload.first_intended_verification_day == WINDOW[0]
    assert stored.payload.ordered_seed_chain_replay_days[0] == "2026-01-01"
    # a second registration of the same packet is verified reuse
    code = script.main(
        ["register-authorization", "--store-root", str(root), "--packet-json", str(signed_path)]
    )
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and printed["reused"] is True
    assert printed["seed_production_authorization_id"] == authorization_id
    # a payload that names another store namespace is the backend's typed refusal
    foreign = json.loads(json.dumps(completed))
    foreign["payload"]["store_namespace_id"] = "9" * 64
    foreign_path = tmp_path / "foreign.json"
    foreign_path.write_text(json.dumps(foreign), encoding="utf-8")
    code = script.main(
        ["register-authorization", "--store-root", str(root), "--packet-json", str(foreign_path)]
    )
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 2 and printed["reason"] == "store_namespace_mismatch"


def test_run_refuses_before_any_path_and_writes_no_receipt(tmp_path, capsys) -> None:
    script = importlib.import_module("scripts.ifvg_seed_production")
    root = tmp_path / "store"
    initialize_test_namespace(root)
    inventory_path = write_inventory_manifest(tmp_path / "inventory.json", inventory_for_chain())
    receipt_path = tmp_path / "center" / "seed_production_run.receipt.json"
    code = script.main(
        [
            "run",
            "--store-root",
            str(root),
            "--authorization-id",
            "f" * 64,
            "--inventory-json",
            str(inventory_path),
            "--receipt-out",
            str(receipt_path),
        ]
    )
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 2 and printed["status"] == "refused"
    assert printed["reason"] == "authorization_not_found"
    assert not receipt_path.exists()
    assert not (root / "seed_snapshots").exists()
