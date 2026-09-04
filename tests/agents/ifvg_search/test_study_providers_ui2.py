"""UI-2 Verification Center read models (plan §5.3 / §8 / §9 Phase 2):

* the shortlist document loads typed (missing / corrupt / available) and a
  window exposes its LOGICAL days separately from its PHYSICAL partitions;
* the coverage matrix derives from the shortlist window (content-addressed);
* the mutable center record round-trips and picks up external receipts;
* the seed-production authorization, the seed snapshot and the run receipt
  each resolve to typed states by EXACT id (never a listing);
* the owner's completed ``VerificationAuthorizationRef`` validates typed
  against this store, the seed and the window;
* the verification bundle derives from the signed ref (run-independent);
* registering the verification run validates before persisting and turns
  the typed readiness ``ready``; the bounded preflight passes on the bound
  store and refuses typed otherwise; monitor rows are the plan's stages only.
"""

from __future__ import annotations

import json

import pytest

from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    VERIFICATION_FIXTURE_DECISION_KEY,
    SyntheticAuthorizationMarker,
    derive_authorization_requirements,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import publish_supersession
from alpha_lab.agents.data_infra.ifvg.study_providers import (
    SEED_AUTHORIZATION_RECEIPT,
    SEED_RUN_RECEIPT,
    VerificationCenterRecord,
    bounded_preflight_state,
    coverage_matrix_from_shortlist_window,
    load_signed_verification_ref,
    load_source_inventory,
    load_verification_center_record,
    load_window_shortlist,
    pick_up_seed_receipts,
    register_verification_run,
    save_verification_center_record,
    seed_authorization_state,
    seed_receipt_state,
    seed_snapshot_state,
    shortlist_window,
    verification_authorization_readiness,
    verification_bundle_from_signed_ref,
    verification_owner_bundle,
    verification_stage_rows,
)
from tests.agents.ifvg_search.verification_center_fixture import (
    PROFILE,
    QL_IDENTITY,
    SC_COMMIT,
    SC_IDENTITY,
    SEED_CHAIN,
    WINDOW,
    build_verification_store,
    inventory_for_chain,
    write_inventory_manifest,
    write_shortlist_document,
    write_signed_ref,
)


@pytest.fixture()
def center(tmp_path, synthetic_chain) -> dict:
    fixture = build_verification_store(tmp_path / "repo", synthetic_chain)
    fixture["center_root"] = tmp_path / "center"
    fixture["shortlist_path"] = tmp_path / "evidence" / "LOGICAL_WINDOW_COVERAGE_SCAN.json"
    fixture["document"] = write_shortlist_document(fixture["shortlist_path"])
    return fixture


# ── shortlist / window / coverage matrix ─────────────────────────────────────


def test_shortlist_loads_typed_and_separates_logical_days_from_partitions(center, tmp_path):
    missing = load_window_shortlist(tmp_path / "nope.json")
    assert missing.status == "missing" and missing.shortlist is None
    corrupt_path = tmp_path / "corrupt.json"
    corrupt_path.write_text("{not json", encoding="utf-8")
    assert load_window_shortlist(corrupt_path).status == "corrupt"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps({"shortlist": {"bogus": 1}}), encoding="utf-8")
    assert load_window_shortlist(tampered_path).status == "corrupt"
    state = load_window_shortlist(center["shortlist_path"])
    assert state.status == "available"
    assert state.shortlist_id == center["document"]["shortlist_id"]
    assert state.shortlist.owner_selection == "NOT PERFORMED"
    assert state.shortlist.register_program_allowlist_called is False
    assert state.shortlist.eligible_window_count >= 1
    window = shortlist_window(state.shortlist, WINDOW)
    assert window is not None and window.eligible
    assert window.days == WINDOW  # logical trading days
    partitions = [
        (ref.physical_utc_date, ref.relative_logical_partition_key)
        for day_ref in window.trading_day_refs
        for ref in day_ref.ordered_source_partition_refs
    ]
    # each logical day maps to (td−1, td): the physical partitions are a
    # DIFFERENT set from the logical days (2026-01-14 appears only as a partition)
    assert partitions == [
        ("2026-01-14", "prev_utc_date"),
        ("2026-01-15", "utc_date"),
        ("2026-01-15", "prev_utc_date"),
        ("2026-01-16", "utc_date"),
    ]
    assert window.seed_chain_replay_days[0] == "2026-01-01"
    assert window.seed_chain_replay_days[-1] == "2026-01-14"
    assert shortlist_window(state.shortlist, ("2026-01-15",)) is None  # exact window only
    assert shortlist_window(state.shortlist, ("2026-03-02", "2026-03-03")) is None


def test_coverage_matrix_derives_from_the_shortlist_window(center) -> None:
    state = load_window_shortlist(center["shortlist_path"])
    window = shortlist_window(state.shortlist, WINDOW)
    matrix = coverage_matrix_from_shortlist_window(state.shortlist, window)
    assert len(matrix.coverage_matrix_id) == 64
    payload = matrix.payload
    assert payload.candidate_allowlist == WINDOW
    assert payload.candidate_allowlist_hash == allowlist_sha256(WINDOW)
    assert payload.evidence_source_dataset_id == state.shortlist.evidence_source_dataset_id
    assert [row.trading_day for row in payload.rows] == list(WINDOW)
    assert all(row.source_partition_recorded for row in payload.rows)
    assert payload.rows[0].executed_trade_rows == 5
    assert payload.lifecycle_paths_covered["execution_resolution"] is True
    assert payload.rows[0].audit_event_rows is None  # not evaluated → never fabricated
    again = coverage_matrix_from_shortlist_window(state.shortlist, window)
    assert again.coverage_matrix_id == matrix.coverage_matrix_id  # deterministic


# ── the mutable center record and receipt pickup ─────────────────────────────


def test_center_record_round_trips_and_picks_up_receipts(center) -> None:
    root = center["center_root"]
    empty = load_verification_center_record(root)
    assert empty.provisional_window is None and empty.seed_snapshot_id is None
    record = VerificationCenterRecord(
        store_namespace_id=center["namespace_id"],
        provisional_window={
            "shortlist_id": center["document"]["shortlist_id"],
            "days": list(WINDOW),
            "label": "top_ranked",
            "recorded_at_utc": "2026-09-04T00:00:00+00:00",
        },
    )
    path = save_verification_center_record(root, record, now_fn=lambda: "2026-09-04T00:00:01+00:00")
    assert path == root / "verification_center.json"
    loaded = load_verification_center_record(root)
    assert loaded.provisional_window["days"] == list(WINDOW)
    assert loaded.updated_at_utc == "2026-09-04T00:00:01+00:00"
    # receipts written by the external CLI are picked up (exact ids only)
    (root / SEED_AUTHORIZATION_RECEIPT).write_text(
        json.dumps(
            {
                "status": "authorization_registered",
                "seed_production_authorization_id": (
                    center["authorization"].seed_production_authorization_id
                ),
            }
        ),
        encoding="utf-8",
    )
    (root / SEED_RUN_RECEIPT).write_text(
        json.dumps(
            {
                "status": "seed_production_completed",
                "seed_snapshot_id": center["snapshot"].seed_snapshot_id,
                "seed_production_run_id": center["receipt"].seed_production_run_id,
            }
        ),
        encoding="utf-8",
    )
    updated, notes = pick_up_seed_receipts(root, loaded)
    assert updated.seed_production_authorization_id == (
        center["authorization"].seed_production_authorization_id
    )
    assert updated.seed_snapshot_id == center["snapshot"].seed_snapshot_id
    assert updated.seed_production_run_id == center["receipt"].seed_production_run_id
    assert len(notes) == 2 and all("picked up" in note for note in notes)
    # a malformed receipt is reported, never trusted, never fatal
    (root / SEED_RUN_RECEIPT).write_text("{broken", encoding="utf-8")
    again, notes = pick_up_seed_receipts(root, updated)
    assert again.seed_snapshot_id == center["snapshot"].seed_snapshot_id
    assert any("unreadable" in note for note in notes)
    corrupt = load_verification_center_record(root)  # the record itself is intact
    assert corrupt.provisional_window["days"] == list(WINDOW)
    (root / "verification_center.json").write_text("{broken", encoding="utf-8")
    assert load_verification_center_record(root).provisional_window is None


# ── seed states ──────────────────────────────────────────────────────────────


def test_seed_authorization_state_is_typed_by_exact_id(center) -> None:
    store = center["store_root"]
    authorization_id = center["authorization"].seed_production_authorization_id
    common = dict(
        baseline_profile_name=PROFILE,
        section_hash=center["section_hash"],
        first_intended_verification_day=WINDOW[0],
    )
    assert seed_authorization_state(store, None, **common).status == "not_recorded"
    assert seed_authorization_state(store, "f" * 64, **common).status == "not_found"
    # envelope-level facts without the inventory: verified_envelope (inventory not checked)
    state = seed_authorization_state(store, authorization_id, **common)
    assert state.status == "verified_envelope"
    assert state.provenance == "synthetic_test_authorization_v1"
    assert state.chain_day_count == len(SEED_CHAIN)
    assert state.first_intended_verification_day == WINDOW[0]
    assert "inventory" in state.detail
    # the full backend verification with the inventory and the code identities
    verified = seed_authorization_state(
        store,
        authorization_id,
        inventory=center["inventory"],
        quant_lab_source_identity=QL_IDENTITY,
        strategy_core=(SC_COMMIT, SC_IDENTITY),
        now="2026-09-04T00:00:00+00:00",
        **common,
    )
    assert verified.status == "verified"
    diverged = seed_authorization_state(
        store,
        authorization_id,
        inventory=inventory_for_chain(),
        quant_lab_source_identity="9" * 64,
        strategy_core=(SC_COMMIT, SC_IDENTITY),
        now="2026-09-04T00:00:00+00:00",
        **common,
    )
    assert diverged.status == "code_identity_mismatch"
    wrong_window = seed_authorization_state(
        store, authorization_id, **{**common, "first_intended_verification_day": "2026-01-20"}
    )
    assert wrong_window.status == "window_mismatch"
    wrong_profile = seed_authorization_state(
        store, authorization_id, **{**common, "section_hash": "9" * 64}
    )
    assert wrong_profile.status == "profile_mismatch"
    # a moved head after registration: the typed backend reason survives
    publish_supersession(
        store,
        superseded_decision_id="1" * 64,
        replacement_decision_id="2" * 64,
        reason="moved after registration",
        effective_at="2026-09-04T01:00:00+00:00",
        owner_evidence_ref="2" * 64,
    )
    stale = seed_authorization_state(
        store,
        authorization_id,
        inventory=center["inventory"],
        quant_lab_source_identity=QL_IDENTITY,
        strategy_core=(SC_COMMIT, SC_IDENTITY),
        now="2026-09-04T02:00:00+00:00",
        **common,
    )
    assert stale.status == "supersession_head_witness_mismatch"


def test_seed_snapshot_and_receipt_states_are_typed(center, tmp_path, synthetic_chain) -> None:
    store = center["store_root"]
    snapshot_id = center["snapshot"].seed_snapshot_id
    common = dict(
        section_hash=center["section_hash"], first_replay_day=WINDOW[0], profile_name=PROFILE
    )
    assert seed_snapshot_state(store, None, **common).status == "not_recorded"
    assert seed_snapshot_state(store, "e" * 64, **common).status == "missing"
    verified = seed_snapshot_state(store, snapshot_id, **common)
    assert verified.status == "verified"
    assert verified.first_replay_day == WINDOW[0]
    assert verified.snapshot_through_day == SEED_CHAIN[-1]
    assert verified.seed_hash == center["snapshot"].payload.seed_hash
    mismatch = seed_snapshot_state(store, snapshot_id, **{**common, "section_hash": "9" * 64})
    assert mismatch.status == "profile_mismatch"
    discontinuous = seed_snapshot_state(
        store, snapshot_id, **{**common, "first_replay_day": "2026-01-20"}
    )
    assert discontinuous.status == "discontinuous"
    # the run receipt: exact id, binds the snapshot and the authorization
    run_id = center["receipt"].seed_production_run_id
    assert seed_receipt_state(store, None).status == "not_recorded"
    assert seed_receipt_state(store, "e" * 64).status == "missing"
    receipt = seed_receipt_state(store, run_id, expected_seed_snapshot_id=snapshot_id)
    assert receipt.status == "verified"
    assert receipt.seed_snapshot_id == snapshot_id
    assert receipt.authorization_id == center["authorization"].seed_production_authorization_id
    assert receipt.provenance == "synthetic_test_authorization_v1"
    assert receipt.chain_replay_day_count == len(SEED_CHAIN)
    assert receipt.verification_evidence_footprint_days == 0
    assert (
        seed_receipt_state(store, run_id, expected_seed_snapshot_id="e" * 64).status == "mismatch"
    )
    # a store without the seed lane
    bare = build_verification_store(
        tmp_path / "bare", synthetic_chain, with_seed=False, with_authorization=False
    )
    assert seed_snapshot_state(bare["store_root"], snapshot_id, **common).status == "missing"


# ── the signed reference and the run-independent bundle ─────────────────────


def test_signed_ref_validation_is_typed(center, tmp_path) -> None:
    store = center["store_root"]
    snapshot_id = center["snapshot"].seed_snapshot_id
    expected = dict(
        store_root=store,
        expected_seed_snapshot_id=snapshot_id,
        expected_allowlist_hash=allowlist_sha256(WINDOW),
    )
    path = tmp_path / "center" / "VERIFICATION_AUTHORIZATION_REF.signed.json"
    assert load_signed_verification_ref(path, **expected).status == "file_missing"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{broken", encoding="utf-8")
    assert load_signed_verification_ref(path, **expected).status == "malformed"
    unsigned = {
        "verification_policy_id": "verification_fixed_allowlist_max5_v1",
        "approved_allowlist_hash": allowlist_sha256(WINDOW),
        "coverage_matrix_artifact_id": "b" * 64,
        "seed_snapshot_id": snapshot_id,
        "store_namespace_id": center["namespace_id"],
        "supersession_head_witness": {
            "store_namespace_id": center["namespace_id"],
            "line_count": 0,
            "head_sha256": "0" * 64,
        },
        "approved_by": "<OWNER_TO_FILL>",
        "approved_at": "<OWNER_TO_FILL>",
        "content_hash": "<OWNER_TO_FILL>",
    }
    path.write_text(json.dumps(unsigned), encoding="utf-8")
    state = load_signed_verification_ref(path, **expected)
    assert state.status == "unsigned" and "placeholder" in state.detail
    # a packet wrapper (the file the center wrote) is accepted through its ref
    write_signed_ref(path, store, seed_snapshot_id=snapshot_id)
    valid = load_signed_verification_ref(path, **expected)
    assert valid.status == "valid" and valid.ref is not None
    assert valid.ref.approved_by == "test-owner"
    # seed / allowlist / namespace mismatches are typed
    assert (
        load_signed_verification_ref(
            path, **{**expected, "expected_seed_snapshot_id": "e" * 64}
        ).status
        == "seed_mismatch"
    )
    assert (
        load_signed_verification_ref(
            path, **{**expected, "expected_allowlist_hash": allowlist_sha256(("2026-01-20",))}
        ).status
        == "allowlist_mismatch"
    )
    other = build_verification_store(
        tmp_path / "other", None, with_seed=False, with_authorization=False
    )
    foreign = tmp_path / "foreign.json"
    write_signed_ref(foreign, other["store_root"], seed_snapshot_id=snapshot_id)
    assert load_signed_verification_ref(foreign, **expected).status == "wrong_namespace"
    # the head moved after signing → the backend's typed reason
    publish_supersession(
        store,
        superseded_decision_id="1" * 64,
        replacement_decision_id="2" * 64,
        reason="moved after signing",
        effective_at="2026-09-04T01:00:00+00:00",
        owner_evidence_ref="2" * 64,
    )
    assert load_signed_verification_ref(path, **expected).status == "stale_head"


def test_verification_bundle_derives_from_the_signed_ref_not_the_run(center, tmp_path) -> None:
    store = center["store_root"]
    snapshot_id = center["snapshot"].seed_snapshot_id
    path = tmp_path / "ref.json"
    write_signed_ref(path, store, seed_snapshot_id=snapshot_id)
    ref = load_signed_verification_ref(
        path,
        store_root=store,
        expected_seed_snapshot_id=snapshot_id,
        expected_allowlist_hash=allowlist_sha256(WINDOW),
    ).ref
    requirement_set = derive_authorization_requirements("verification_5d", (), None, (), ())
    bundle = verification_bundle_from_signed_ref(ref, requirement_set)
    evidence = bundle.decision_refs[VERIFICATION_FIXTURE_DECISION_KEY]
    assert evidence.decision_artifact_id == ref.content_hash  # the ref IS the artifact
    assert evidence.content_hash == ref.content_hash
    assert evidence.author == "test-owner"
    assert evidence.reviewed_evidence_refs == (ref.coverage_matrix_artifact_id, snapshot_id)
    assert bundle.store_namespace_id == center["namespace_id"]
    # registering a run over ANY pipeline id yields the same bundle from the
    # persisted run — the charter identity never depends on the run id
    run = register_verification_run(
        store,
        ref=ref,
        pipeline_semantic_id="a" * 64,
        allowlist=WINDOW,
        seed_snapshot_id=snapshot_id,
        baseline_profile_name=PROFILE,
        baseline_section_config_hash=center["section_hash"],
        coverage_matrix_artifact_id=ref.coverage_matrix_artifact_id,
        display_name="synthetic bound run",
    )
    readiness = verification_authorization_readiness(store)
    assert readiness.status == "ready"
    assert readiness.evidence_ids == (run.verification_run_id,)
    from_run = verification_owner_bundle(store, readiness, requirement_set)
    assert from_run == bundle


def test_register_verification_run_validates_before_persisting_and_preflight_passes(
    center, tmp_path
) -> None:
    store = center["store_root"]
    snapshot_id = center["snapshot"].seed_snapshot_id
    path = tmp_path / "ref.json"
    write_signed_ref(path, store, seed_snapshot_id=snapshot_id)
    ref = load_signed_verification_ref(
        path,
        store_root=store,
        expected_seed_snapshot_id=snapshot_id,
        expected_allowlist_hash=allowlist_sha256(WINDOW),
    ).ref
    kwargs = dict(
        ref=ref,
        pipeline_semantic_id="a" * 64,
        allowlist=WINDOW,
        seed_snapshot_id=snapshot_id,
        baseline_profile_name=PROFILE,
        baseline_section_config_hash=center["section_hash"],
        coverage_matrix_artifact_id=ref.coverage_matrix_artifact_id,
        display_name="synthetic bound run",
    )
    # a run whose seed disagrees with the signed ref is refused BEFORE any write
    with pytest.raises(PermissionError, match="seed snapshot"):
        register_verification_run(store, **{**kwargs, "seed_snapshot_id": "e" * 64})
    assert verification_authorization_readiness(store).status == "missing"
    run = register_verification_run(store, **kwargs)
    again = register_verification_run(store, **kwargs)  # idempotent (verified reuse)
    assert again.verification_run_id == run.verification_run_id
    assert verification_authorization_readiness(store).status == "ready"
    preflight = bounded_preflight_state(
        store, center["repo_root"], run, section_hash=center["section_hash"]
    )
    assert preflight.status == "passed", preflight.detail
    assert preflight.record is not None
    assert preflight.record.logical_trading_days == WINDOW
    assert preflight.record.physical_partition_dates == ("2026-01-14", "2026-01-15", "2026-01-16")
    refused = bounded_preflight_state(
        store,
        center["repo_root"],
        run,
        section_hash=center["section_hash"],
        authorization=SyntheticAuthorizationMarker(),
    )
    assert refused.status == "refused" and refused.reason == "synthetic_marker_refused"
    elsewhere = bounded_preflight_state(
        store, tmp_path / "other_repo", run, section_hash=center["section_hash"]
    )
    assert elsewhere.status == "refused" and elsewhere.reason == "output_namespace_not_locked"


# ── inventory and monitor rows ───────────────────────────────────────────────


def test_source_inventory_loads_from_the_manifest_or_reports_unavailable(center, tmp_path):
    missing = load_source_inventory(tmp_path / "nope.json")
    assert missing.status == "missing" and missing.inventory is None
    path = write_inventory_manifest(tmp_path / "manifest.json", center["inventory"])
    state = load_source_inventory(path)
    assert state.status == "available"
    assert state.inventory == center["inventory"]
    assert state.partition_count == len(center["inventory"])
    path.write_text("{broken", encoding="utf-8")
    assert load_source_inventory(path).status == "corrupt"


def test_verification_stage_rows_list_only_the_planned_stages() -> None:
    state = {
        "stages": {
            "00_validate_inputs": {"in_plan": True, "status": "completed"},
            "01_prepare_strategy_profiles": {"in_plan": True, "status": "running"},
            "05_materialize_feature_views": {"in_plan": False},
            "11_run_frozen_model_gated_replays": {"in_plan": True, "status": "blocked"},
            "15_verify_and_publish": {"in_plan": True, "status": "pending"},
        }
    }
    rows = verification_stage_rows(state)
    assert [row.stage_value for row in rows] == [
        "00_validate_inputs",
        "01_prepare_strategy_profiles",
        "11_run_frozen_model_gated_replays",
        "15_verify_and_publish",
    ]
    assert all(row.in_plan for row in rows)
    assert verification_stage_rows(None) == ()
