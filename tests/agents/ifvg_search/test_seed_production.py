"""HARDENING-BACKEND Phase 3 §5.2–§5.5 (F-16 / F-21) — separately authorized
seed production, proven SYNTHETICALLY over the conftest chain.

No real source path is constructed, no real replay runs, no owner signature
is created: the authorizations persisted here carry the synthetic
provenance literal (confined to ``test`` namespaces) and the packets are
unsigned drafts whose placeholders fail validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date

import pytest
from strategy_core.strategies.ifvg_smc.state import IFVG_SEED_SCHEMA_VERSION, seed_hash

from alpha_lab.agents.data_infra.ifvg.data_access import require_fixed_exploration_allowlist
from alpha_lab.agents.data_infra.ifvg.dataset import ChainStart, build_ifvg_v2_capture
from alpha_lab.agents.data_infra.ifvg.day_artifacts import DaySeeds, write_day_artifacts
from alpha_lab.agents.data_infra.ifvg.development_access import VerificationReplayPolicy
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search import seed_production as module
from alpha_lab.agents.data_infra.ifvg.search.authorization import VerificationAuthorizationRef
from alpha_lab.agents.data_infra.ifvg.search.child_replay import load_seed_snapshot
from alpha_lab.agents.data_infra.ifvg.search.seed_production import (
    PERMITTED_SEED_OUTPUTS,
    PROHIBITED_SEED_OUTPUTS,
    SEED_PRODUCTION_ACCESS_POLICY_ID,
    SEED_PRODUCTION_AUTHORIZATION_STORE,
    SEED_PRODUCTION_CHAIN_POLICY_ID,
    SEED_PRODUCTION_RUN_STORE,
    SeedProductionAuthorizationError,
    SeedProductionAuthorizationPayload,
    SeedProductionAuthorizationRef,
    SeedProductionReplayPolicy,
    build_seed_production_packet,
    build_verification_authorization_packet,
    persist_seed_production_authorization,
    render_seed_production_packet_markdown,
    render_verification_packet_markdown,
    run_seed_production_chain,
    seed_chain_source_inventory_hash,
    synthetic_seed_production_authorization,
    verify_seed_production_authorization,
)
from alpha_lab.agents.data_infra.ifvg.search.store import load_sidecar_bytes
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    initialize_store_namespace,
    initialize_test_namespace,
    load_store_namespace,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import publish_supersession
from tests.agents.ifvg_search.conftest import SYNTHETIC_DAYS

_CHAIN = ("2026-01-13", "2026-01-14")
_FIRST_VERIFICATION_DAY = "2026-01-15"
_NOW = "2026-09-02T12:00:00+00:00"
_QL = "1" * 64
_SC_COMMIT = "c" * 40
_SC_IDENTITY = "2" * 64


def _inventory(days) -> dict[str, tuple[str, str]]:
    return {day: ("mbp10", hashlib.sha256(day.encode()).hexdigest()) for day in days}


# ── the access policy ────────────────────────────────────────────────────────


def test_seed_production_policy_fails_before_path() -> None:
    policy = SeedProductionReplayPolicy(("2026-02-05", "2026-02-06", "2026-02-08", "2026-02-09"))
    assert policy.policy_id == SEED_PRODUCTION_ACCESS_POLICY_ID
    require_fixed_exploration_allowlist(policy)  # a trusted development-window class
    with pytest.raises(PermissionError):
        policy.authorize_date("2026-02-10")  # off-chain inside the window
    assert policy.audit.denied_dates.get("2026-02-10") == 1
    with pytest.raises(PermissionError, match="before path construction"):
        SeedProductionReplayPolicy(("2026-06-10", "2026-06-11"))
    with pytest.raises(PermissionError, match="before path construction"):
        SeedProductionReplayPolicy(("2026-06-12",))
    with pytest.raises(ValueError, match="Saturday"):
        SeedProductionReplayPolicy(("2026-02-06", "2026-02-07", "2026-02-08"))
    with pytest.raises(ValueError, match="store-day chain"):
        SeedProductionReplayPolicy(("2026-02-05", "2026-02-09"))  # a gap in the chain
    with pytest.raises(ValueError, match="chronological"):
        SeedProductionReplayPolicy(("2026-02-06", "2026-02-05"))
    with pytest.raises(ValueError, match="empty"):
        SeedProductionReplayPolicy(())
    audit = policy.audit_dict()
    assert audit["policy"] == SEED_PRODUCTION_ACCESS_POLICY_ID
    assert (
        audit["authorized_date_sha256"]
        == hashlib.sha256(
            "\n".join(("2026-02-05", "2026-02-06", "2026-02-08", "2026-02-09")).encode()
        ).hexdigest()
    )


# ── the authorization contract ───────────────────────────────────────────────


def _payload(namespace, witness, **overrides) -> SeedProductionAuthorizationPayload:
    inventory = _inventory(_CHAIN)
    fields = dict(
        store_namespace_id=namespace.store_namespace_id,
        supersession_head_witness=witness,
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        resolved_section_config_hash=resolve_profile_config({}).section_config_hash,
        ordered_seed_chain_replay_days=_CHAIN,
        ordered_seed_chain_logical_trading_days=_CHAIN,
        snapshot_through_day=_CHAIN[-1],
        first_intended_verification_day=_FIRST_VERIFICATION_DAY,
        expected_source_inventory_hash=seed_chain_source_inventory_hash(_CHAIN, inventory),
        quant_lab_source_identity=_QL,
        strategy_core_commit=_SC_COMMIT,
        strategy_core_source_identity=_SC_IDENTITY,
        seed_schema_version=IFVG_SEED_SCHEMA_VERSION,
        provenance="synthetic_test_authorization_v1",
        owner_decision_refs=("21/R-5:synthetic",),
        approved_by="synthetic_fixture",
        approved_at="2026-09-01T00:00:00+00:00",
        effective_from="2026-09-01T00:00:00+00:00",
    )
    fields.update(overrides)
    return SeedProductionAuthorizationPayload(**fields)


def test_authorization_payload_binds_the_chain_semantics(tmp_path) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
        current_supersession_head_witness,
    )

    root = tmp_path / "store"
    namespace = initialize_test_namespace(root)
    witness = current_supersession_head_witness(root)
    payload = _payload(namespace, witness)
    assert payload.access_policy_id == SEED_PRODUCTION_ACCESS_POLICY_ID
    assert payload.chain_policy_id == SEED_PRODUCTION_CHAIN_POLICY_ID
    assert payload.final_day_exhausts_dataset is False
    assert payload.permitted_outputs == PERMITTED_SEED_OUTPUTS
    assert payload.prohibited_outputs == PROHIBITED_SEED_OUTPUTS
    assert "prop_simulation" in PROHIBITED_SEED_OUTPUTS
    assert "research_catalog_publication" in PROHIBITED_SEED_OUTPUTS
    with pytest.raises(ValueError, match="snapshot_through_day"):
        _payload(namespace, witness, snapshot_through_day="2026-01-13")
    with pytest.raises(ValueError, match="first_intended_verification_day"):
        _payload(namespace, witness, first_intended_verification_day="2026-01-16")
    with pytest.raises(ValueError, match="logical"):
        _payload(namespace, witness, ordered_seed_chain_logical_trading_days=("2026-01-13",))
    with pytest.raises(ValueError, match="store-day chain"):
        _payload(
            namespace,
            witness,
            ordered_seed_chain_replay_days=("2026-01-12", "2026-01-14"),
            ordered_seed_chain_logical_trading_days=("2026-01-12", "2026-01-14"),
        )
    with pytest.raises(ValueError, match="permitted development window"):
        _payload(
            namespace,
            witness,
            ordered_seed_chain_replay_days=("2026-06-10", "2026-06-11"),
            ordered_seed_chain_logical_trading_days=("2026-06-10", "2026-06-11"),
            snapshot_through_day="2026-06-11",
            first_intended_verification_day="2026-06-12",
        )
    # an owner-signed authorization must start the chain at the canonical
    # cold start (the first store day of the accepted chain)
    with pytest.raises(ValueError, match="canonical chain start"):
        _payload(namespace, witness, provenance="owner_signed", approved_by="owner")
    with pytest.raises(ValueError, match="placeholder"):
        _payload(namespace, witness, approved_by="<OWNER_TO_FILL>")
    with pytest.raises(ValueError, match="effective_from"):
        _payload(namespace, witness, effective_from="2026-08-31T00:00:00+00:00")
    with pytest.raises(ValueError):
        _payload(namespace, witness, final_day_exhausts_dataset=True)
    ref = SeedProductionAuthorizationRef(
        seed_production_authorization_id="a" * 64,
        approved_by="owner",
        approved_at="2026-09-01T00:00:00+00:00",
        content_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="content_hash"):
        SeedProductionAuthorizationRef(
            seed_production_authorization_id="a" * 64,
            approved_by="owner",
            approved_at="2026-09-01T00:00:00+00:00",
            content_hash="b" * 64,
        )
    assert ref.content_hash == ref.seed_production_authorization_id


def test_verify_authorization_refuses_before_any_path(tmp_path) -> None:
    root = tmp_path / "store"
    initialize_test_namespace(root)
    inventory = _inventory(_CHAIN)
    envelope = synthetic_seed_production_authorization(
        root,
        chain_replay_days=_CHAIN,
        first_intended_verification_day=_FIRST_VERIFICATION_DAY,
        inventory=inventory,
        quant_lab_source_identity=_QL,
        strategy_core_commit=_SC_COMMIT,
        strategy_core_source_identity=_SC_IDENTITY,
    )
    expected = dict(
        expected_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        expected_section_config_hash=resolve_profile_config({}).section_config_hash,
        expected_chain_replay_days=_CHAIN,
        expected_source_inventory_hash=seed_chain_source_inventory_hash(_CHAIN, inventory),
        expected_quant_lab_source_identity=_QL,
        expected_strategy_core=(_SC_COMMIT, _SC_IDENTITY),
        now=_NOW,
    )
    verified = verify_seed_production_authorization(
        root, envelope.seed_production_authorization_id, **expected
    )
    assert verified.seed_production_authorization_id == envelope.seed_production_authorization_id
    # the bare ref verifies too — and must agree with the persisted envelope
    ref = envelope.ref()
    verify_seed_production_authorization(root, ref, **expected)
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        verify_seed_production_authorization(
            root, ref.model_copy(update={"approved_by": "someone-else"}), **expected
        )
    assert excinfo.value.reason == "authorization_ref_mismatch"
    # every divergence is a typed refusal
    for key, value, reason in (
        ("expected_profile_name", "other_profile", "profile_mismatch"),
        ("expected_section_config_hash", "9" * 64, "profile_mismatch"),
        ("expected_chain_replay_days", ("2026-01-13",), "chain_mismatch"),
        ("expected_source_inventory_hash", "8" * 64, "source_inventory_mismatch"),
        ("expected_quant_lab_source_identity", "7" * 64, "code_identity_mismatch"),
        ("expected_strategy_core", ("d" * 40, _SC_IDENTITY), "code_identity_mismatch"),
        ("now", "2026-08-01T00:00:00+00:00", "authorization_not_effective"),
    ):
        with pytest.raises(SeedProductionAuthorizationError) as excinfo:
            verify_seed_production_authorization(
                root, envelope.seed_production_authorization_id, **{**expected, key: value}
            )
        assert excinfo.value.reason == reason
    # an unknown id is a typed refusal, not a store listing
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        verify_seed_production_authorization(root, "f" * 64, **expected)
    assert excinfo.value.reason == "authorization_not_found"
    # a moved head (a supersession recorded after signing) invalidates the witness
    publish_supersession(
        root,
        superseded_decision_id="a" * 64,
        replacement_decision_id="b" * 64,
        reason="test",
        effective_at="2026-09-01T01:00:00+00:00",
        owner_evidence_ref="b" * 64,
    )
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        verify_seed_production_authorization(
            root, envelope.seed_production_authorization_id, **expected
        )
    assert excinfo.value.reason == "supersession_head_witness_mismatch"


def test_synthetic_provenance_is_confined_to_test_namespaces(tmp_path) -> None:
    inventory = _inventory(_CHAIN)
    kwargs = dict(
        chain_replay_days=_CHAIN,
        first_intended_verification_day=_FIRST_VERIFICATION_DAY,
        inventory=inventory,
        quant_lab_source_identity=_QL,
        strategy_core_commit=_SC_COMMIT,
        strategy_core_source_identity=_SC_IDENTITY,
    )
    unmarked = tmp_path / "unmarked"
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        synthetic_seed_production_authorization(unmarked, **kwargs)
    assert excinfo.value.reason == "store_namespace_missing"
    research = tmp_path / "research_store"
    initialize_store_namespace(research, namespace_class="research")
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        synthetic_seed_production_authorization(research, **kwargs)
    assert excinfo.value.reason == "synthetic_provenance_confined_to_test_namespaces"
    # a namespace id of ANOTHER store is refused at persist
    test_root = tmp_path / "test_store"
    other_root = tmp_path / "other_store"
    initialize_test_namespace(test_root)
    other = initialize_test_namespace(other_root)
    from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
        current_supersession_head_witness,
    )

    foreign = _payload(other, current_supersession_head_witness(other_root))
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        persist_seed_production_authorization(test_root, foreign)
    assert excinfo.value.reason == "store_namespace_mismatch"


# ── the synthetic seed-production proof ──────────────────────────────────────


@pytest.fixture()
def seed_lane(tmp_path, doc_default_cfg, synthetic_artifacts):
    root = tmp_path / "search_test" / "v1"
    initialize_test_namespace(root)
    data_dir = tmp_path / "databento"
    resolved = resolve_profile_config({})
    cfg = replace(doc_default_cfg, data_dir=data_dir)
    chain_policy = SeedProductionReplayPolicy(_CHAIN)
    for artifacts in synthetic_artifacts[:2]:
        write_day_artifacts(artifacts, cfg, access_policy=chain_policy)
    # the verification day's artifacts are cached under the verification policy
    write_day_artifacts(
        synthetic_artifacts[2], cfg, access_policy=VerificationReplayPolicy((SYNTHETIC_DAYS[2],))
    )
    inventory = _inventory(_CHAIN)
    authorization = synthetic_seed_production_authorization(
        root,
        chain_replay_days=_CHAIN,
        first_intended_verification_day=_FIRST_VERIFICATION_DAY,
        inventory=inventory,
        quant_lab_source_identity=_QL,
        strategy_core_commit=_SC_COMMIT,
        strategy_core_source_identity=_SC_IDENTITY,
    )
    return {
        "root": root,
        "cfg": cfg,
        "resolved": resolved,
        "authorization": authorization,
        "inventory": inventory,
    }


def _run(lane, **overrides):
    kwargs = dict(
        root=lane["root"],
        authorization_id=lane["authorization"].seed_production_authorization_id,
        cfg=lane["cfg"],
        resolved_profile=lane["resolved"],
        source_inventory=lane["inventory"],
        quant_lab_source_identity=_QL,
        strategy_core_identity=(_SC_COMMIT, _SC_IDENTITY),
        now=_NOW,
        cached_artifacts_only=True,
    )
    kwargs.update(overrides)
    return run_seed_production_chain(**kwargs)


def test_seed_production_chain_produces_only_the_permitted_outputs(
    seed_lane, synthetic_chain, synthetic_artifacts
) -> None:
    result = _run(seed_lane)
    root = seed_lane["root"]
    # the seed is the continuous chain's end-of-day-2 state (final day NOT exhausted)
    assert seed_hash(result.seed) == seed_hash(synthetic_chain[1].end_seed)
    snapshot = result.snapshot
    assert snapshot.payload.snapshot_through_day == _CHAIN[-1]
    assert snapshot.payload.first_replay_day == _FIRST_VERIFICATION_DAY
    assert snapshot.payload.chain_policy_id == SEED_PRODUCTION_CHAIN_POLICY_ID
    assert snapshot.payload.chain_date_count == 2
    assert snapshot.payload.strategy_core_commit == _SC_COMMIT
    day2 = synthetic_artifacts[1]
    assert snapshot.payload.entering_day_seeds.prev_day == day2.date_str
    assert tuple(snapshot.payload.entering_day_seeds.prev_full_hl) == tuple(day2.day_hl)
    # reload-verified, profile-bound
    loaded, chain_start = load_seed_snapshot(
        root,
        snapshot.seed_snapshot_id,
        expected_section_config_hash=seed_lane["resolved"].section_config_hash,
    )
    assert loaded.seed_snapshot_id == snapshot.seed_snapshot_id
    assert seed_hash(chain_start.seed) == seed_hash(result.seed)
    # the receipt binds the authorization, the seed, the chain and the audit
    receipt = result.receipt.payload
    assert receipt.seed_production_authorization_id == (
        seed_lane["authorization"].seed_production_authorization_id
    )
    assert receipt.seed_snapshot_id == snapshot.seed_snapshot_id
    assert receipt.chain_replay_day_count == 2
    assert receipt.verification_evidence_footprint_days == 0
    assert receipt.separately_authorized_preparation is True
    assert receipt.output_namespace == "search_test/v1"
    assert receipt.stores_written == ("seed_snapshots", "seed_production_runs")
    audit_bytes = load_sidecar_bytes(
        root, SEED_PRODUCTION_RUN_STORE, result.receipt.seed_production_run_id, "access_audit.json"
    )
    assert hashlib.sha256(audit_bytes).hexdigest() == receipt.access_audit_sha256
    audit = json.loads(audit_bytes)
    assert audit["policy"] == SEED_PRODUCTION_ACCESS_POLICY_ID
    assert not audit["denied_dates"]
    assert all(not any(v.values()) for v in audit["protected_counters"].values())
    # ONLY the permitted outputs exist: no research, pipeline, model, feature,
    # prop, frontier or catalog store gained an entry
    present = sorted(p.name for p in root.iterdir())
    assert set(present) <= {
        "STORE_NAMESPACE.json",
        "owner_decisions",
        "seed_snapshots",
        SEED_PRODUCTION_AUTHORIZATION_STORE,
        SEED_PRODUCTION_RUN_STORE,
    }
    assert not (root / "search_results").exists()
    assert not (root / "catalog_events.jsonl").exists()
    # the restart from the snapshot over the verification day is emission-identical
    verification_policy = VerificationReplayPolicy((SYNTHETIC_DAYS[2],))
    restart = build_ifvg_v2_capture(
        [SYNTHETIC_DAYS[2]],
        seed_lane["cfg"],
        seed_lane["resolved"],
        access_policy=verification_policy,
        cached_artifacts_only=True,
        start_after_artifact=ChainStart(
            seed=chain_start.seed,
            day_seeds=DaySeeds(
                prev_day=date.fromisoformat(day2.date_str),
                prev_full_hl=day2.day_hl,
                prev_ny_day=None,
                prev_ny_hl=None,
            ),
        ),
    )
    assert restart.day_funnels[SYNTHETIC_DAYS[2]] == synthetic_chain[2].funnel
    assert seed_hash(restart.end_seed) == seed_hash(synthetic_chain[2].end_seed)
    # a second run under the same authorization is verified reuse
    again = _run(seed_lane)
    assert again.snapshot.seed_snapshot_id == snapshot.seed_snapshot_id
    assert again.receipt.seed_production_run_id == result.receipt.seed_production_run_id
    assert again.reused is True


def test_seed_production_refuses_before_any_path_on_divergence(seed_lane, monkeypatch) -> None:
    calls: list = []

    def _never(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("a source path must never be constructed on a refused authorization")

    monkeypatch.setattr(module, "build_ifvg_v2_capture", _never)
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        _run(seed_lane, quant_lab_source_identity="9" * 64)
    assert excinfo.value.reason == "code_identity_mismatch"
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        _run(
            seed_lane,
            source_inventory={"2026-01-13": ("mbp10", "0" * 64), "2026-01-14": ("mbp10", "0" * 64)},
        )
    assert excinfo.value.reason == "source_inventory_mismatch"
    other = resolve_profile_config(
        {
            "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
            "section_overrides": {"min_gap_ticks_capture": 3},
        }
    )
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        _run(seed_lane, resolved_profile=other)
    assert excinfo.value.reason == "profile_mismatch"
    assert calls == []
    # a research-namespace destination refuses the synthetic authorization at verify
    root = seed_lane["root"]
    research = root.parent / "research" / "search" / "v1"
    initialize_store_namespace(research, namespace_class="research")
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        _run(seed_lane, root=research)
    assert excinfo.value.reason == "authorization_not_found"


# ── unsigned packets ────────────────────────────────────────────────────────


def test_seed_production_packet_is_unsigned_and_cannot_validate(tmp_path) -> None:
    root = tmp_path / "store"
    initialize_test_namespace(root)
    packet = build_seed_production_packet(
        root,
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        resolved_section_config_hash=resolve_profile_config({}).section_config_hash,
        first_intended_verification_day="2026-06-04",
        inventory=_inventory(module.store_day_chain("2026-01-01", "2026-06-03")),
        quant_lab_source_identity=_QL,
        strategy_core_commit=_SC_COMMIT,
        strategy_core_source_identity=_SC_IDENTITY,
    )
    payload = packet["payload"]
    assert payload["ordered_seed_chain_replay_days"][0] == "2026-01-01"
    assert payload["snapshot_through_day"] == "2026-06-03"
    assert payload["first_intended_verification_day"] == "2026-06-04"
    assert payload["provenance"] == "owner_signed"
    assert payload["approved_by"] == "<OWNER_TO_FILL>"
    assert payload["store_namespace_id"] == load_store_namespace(root).store_namespace_id
    assert packet["seed_chain_replay_day_count"] == len(payload["ordered_seed_chain_replay_days"])
    assert packet["separately_authorized_preparation"] is True
    assert packet["verification_evidence_footprint_days"] == 0
    with pytest.raises(ValueError):
        SeedProductionAuthorizationPayload.model_validate(payload)
    markdown = render_seed_production_packet_markdown(packet, title="Seed production packet")
    assert "nothing here is an authorization" in markdown
    assert "separately authorized preparation" in markdown
    assert str(packet["seed_chain_replay_day_count"]) in markdown


def test_verification_packet_requires_a_verified_seed(seed_lane) -> None:
    root = seed_lane["root"]
    window = (SYNTHETIC_DAYS[2],)
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        build_verification_authorization_packet(
            root,
            seed_snapshot_id="e" * 64,
            logical_window=window,
            resolved_section_config_hash=seed_lane["resolved"].section_config_hash,
            baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
            shortlist_id="5" * 64,
            coverage_matrix_artifact_id="6" * 64,
            expected_source_inventory_hash="7" * 64,
        )
    assert excinfo.value.reason == "seed_snapshot_unverified"
    result = _run(seed_lane)
    with pytest.raises(SeedProductionAuthorizationError) as excinfo:
        build_verification_authorization_packet(
            root,
            seed_snapshot_id=result.snapshot.seed_snapshot_id,
            logical_window=("2026-01-16",),  # not continuous with the seed
            resolved_section_config_hash=seed_lane["resolved"].section_config_hash,
            baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
            shortlist_id="5" * 64,
            coverage_matrix_artifact_id="6" * 64,
            expected_source_inventory_hash="7" * 64,
        )
    assert excinfo.value.reason == "seed_not_continuous_with_window"
    packet = build_verification_authorization_packet(
        root,
        seed_snapshot_id=result.snapshot.seed_snapshot_id,
        logical_window=window,
        resolved_section_config_hash=seed_lane["resolved"].section_config_hash,
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        shortlist_id="5" * 64,
        coverage_matrix_artifact_id="6" * 64,
        expected_source_inventory_hash="7" * 64,
        seed_production_run_id=result.receipt.seed_production_run_id,
    )
    ref = packet["verification_authorization_ref"]
    assert ref["seed_snapshot_id"] == result.snapshot.seed_snapshot_id
    assert ref["approved_by"] == "<OWNER_TO_FILL>"
    assert ref["approved_at"] == "<OWNER_TO_FILL>"
    assert ref["content_hash"] == "<OWNER_TO_FILL>"
    assert ref["store_namespace_id"] == load_store_namespace(root).store_namespace_id
    assert ref["supersession_head_witness"]["line_count"] == 0
    assert packet["stamps"] == {
        "verification_only": True,
        "not_for_research_interpretation": True,
        "full_pipeline_not_run": True,
    }
    assert packet["zero_protected_sealed_requirements"] is True
    assert packet["seed"]["seed_hash"] == result.snapshot.payload.seed_hash
    assert packet["seed"]["provenance"] == "synthetic_test_authorization_v1"
    assert packet["corrected_logical_allowlist"] == list(window)
    assert packet["pipeline_semantic_id"] == "<PENDING: frozen pipeline specification>"
    with pytest.raises(ValueError):
        VerificationAuthorizationRef.model_validate(
            {k: v for k, v in ref.items() if k in VerificationAuthorizationRef.model_fields}
        )
    markdown = render_verification_packet_markdown(packet, title="Verification packet")
    assert "Only the owner may create" in markdown
    assert "full_pipeline_not_run" in markdown


def test_script_imports_launch_nothing_and_packet_subcommand_writes_files(tmp_path, capsys) -> None:
    import importlib

    script = importlib.import_module("scripts.ifvg_seed_production")
    assert hasattr(script, "main")
    root = tmp_path / "store"
    initialize_test_namespace(root)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(_inventory(module.store_day_chain("2026-01-01", "2026-06-03"))),
        encoding="utf-8",
    )
    out_dir = tmp_path / "packet"
    code = script.main(
        [
            "packet",
            "--store-root",
            str(root),
            "--first-verification-day",
            "2026-06-04",
            "--inventory-json",
            str(inventory_path),
            "--quant-lab-source-identity",
            _QL,
            "--strategy-core-commit",
            _SC_COMMIT,
            "--strategy-core-source-identity",
            _SC_IDENTITY,
            "--out-dir",
            str(out_dir),
        ]
    )
    assert code == 0
    assert (out_dir / "SEED_PRODUCTION_PACKET.json").exists()
    assert (out_dir / "SEED_PRODUCTION_PACKET.md").exists()
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["status"] == "unsigned_packet_written"
    # the run subcommand refuses without a persisted authorization (no path built)
    code = script.main(
        [
            "run",
            "--store-root",
            str(root),
            "--authorization-id",
            "f" * 64,
            "--inventory-json",
            str(inventory_path),
        ]
    )
    assert code == 2
