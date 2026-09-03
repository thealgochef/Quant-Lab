"""HARDENING-BACKEND-FIX §10 (HB-FIX-12) — the COMPLETE supersession-chain
authority proof at every real authority seam.

A current head witness alone is insufficient: when the chain names a
replacement (or superseded) owner decision that is missing, hash-invalid,
predecessor-inconsistent, or divergent, every seam — the owner authorization
bundle, the seed-production authorization, the verification authorization,
the regime promotion / activation / gate / study chain loader, and the
pipeline launch — refuses before any data path or publication, even when
the head witness still matches the current head.
"""

from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    persist_regime_assessment,
    persist_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.search import (
    authorization as authorization_module,
)
from alpha_lab.agents.data_infra.ifvg.search import (
    bounded_verification as bounded_module,
)
from alpha_lab.agents.data_infra.ifvg.search import (
    charter as charter_module,
)
from alpha_lab.agents.data_infra.ifvg.search import (
    executors as executors_module,
)
from alpha_lab.agents.data_infra.ifvg.search import (
    pipeline as pipeline_module,
)
from alpha_lab.agents.data_infra.ifvg.search import (
    seed_production as seed_module,
)
from alpha_lab.agents.data_infra.ifvg.search import (
    verification as verification_module,
)
from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    AuthorizationError,
    AuthorizationRequirementSetEnvelope,
    AuthorizationRequirementSetPayload,
    assert_authorization_bound_to_store,
    validate_owner_authorization,
)
from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (
    OWNER_DECISION_STORE,
    OwnerDecisionRefusalError,
    load_supersession_chain,
    synthetic_owner_decision_fixture,
    verify_complete_owner_authority_chain,
)
from alpha_lab.agents.data_infra.ifvg.search.seed_production import (
    SeedProductionAuthorizationError,
    seed_chain_source_inventory_hash,
    synthetic_seed_production_authorization,
    verify_seed_production_authorization,
)
from alpha_lab.agents.data_infra.ifvg.search.store import envelope_destination
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    StoreNamespaceError,
    initialize_test_namespace,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
    assert_head_witness_current,
    current_supersession_head_witness,
    publish_supersession,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import known_cluster_fixture
from tests.agents.ifvg_search.namespace_fixture import (
    owner_authorization_bundle,
    verification_authorization_ref,
)

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id
_CHAIN = ("2026-01-13", "2026-01-14")
_NOW = "2026-09-02T12:00:00+00:00"
_INVENTORY = {
    day: ("legacy_verified_replay_source", f"{index:02d}" * 32)
    for index, day in enumerate(_CHAIN)
}


@pytest.fixture(scope="module")
def regime_lane(tmp_path_factory):
    """One synthetic regime run whose protocol + assessment every decision
    artifact of these tests ratifies (persisted per store by ``_store``)."""

    fixture = known_cluster_fixture(k=3, n=600)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0,
        resolved_input_features=fixture.regime_input_features,
    )
    run = run_regime_protocol(
        fixture.view.frame,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=2,
    )
    assert run.assessment.payload.gates_passed
    return run


def _store(tmp_path: Path, regime_lane) -> dict:
    """A test namespace with a lawful ONE-record chain: prior A superseded by
    replacement B (both verified artifacts), and the witness signed at that
    head."""

    root = tmp_path / "search_test" / "v1"
    initialize_test_namespace(root)
    persist_regime_protocol(root, regime_lane.protocol)
    persist_regime_assessment(root, regime_lane.assessment)
    prior = synthetic_owner_decision_fixture(
        root, protocol=regime_lane.protocol, assessment=regime_lane.assessment
    )
    replacement = synthetic_owner_decision_fixture(
        root,
        protocol=regime_lane.protocol,
        assessment=regime_lane.assessment,
        approved_at="2026-08-28T01:00:00+00:00",
        effective_from="2026-08-28T01:00:00+00:00",
        supersedes=prior.owner_decision_artifact_id,
    )
    witness = current_supersession_head_witness(root)
    assert witness.line_count == 1
    return {
        "root": root,
        "prior": prior,
        "replacement": replacement,
        "witness": witness,
        "run": regime_lane,
    }


def _artifact_dir(root: Path, artifact_id: str) -> Path:
    return envelope_destination(root, OWNER_DECISION_STORE, artifact_id)


def _corrupt(store: dict, mode: str) -> str:
    """Corrupt the chain WITHOUT moving the head (the witness stays current)
    or, for the transition cases, extend it lawfully-looking and re-sign.
    Returns the typed reason the complete proof must raise."""

    root = store["root"]
    run = store["run"]
    replacement_id = store["replacement"].owner_decision_artifact_id
    prior_id = store["prior"].owner_decision_artifact_id
    if mode == "replacement_missing":
        shutil.rmtree(_artifact_dir(root, replacement_id))
        return "supersession_decision_unverifiable"
    if mode == "replacement_manifest_missing":
        (_artifact_dir(root, replacement_id) / "manifest.json").unlink()
        return "supersession_decision_unverifiable"
    if mode == "replacement_hash_invalid":
        envelope_path = _artifact_dir(root, replacement_id) / "envelope.json"
        document = json.loads(envelope_path.read_text(encoding="utf-8"))
        document["payload"]["rationale"] = "rewritten after the fact"
        envelope_path.write_text(json.dumps(document), encoding="utf-8")
        return "supersession_decision_unverifiable"
    if mode == "superseded_missing":
        shutil.rmtree(_artifact_dir(root, prior_id))
        return "supersession_decision_unverifiable"
    if mode == "predecessor_inconsistent":
        # a persisted decision that supersedes NOTHING, then a hand-published
        # record claiming it replaced B: the head advances lawfully-looking
        stray = synthetic_owner_decision_fixture(
            root,
            protocol=run.protocol,
            assessment=run.assessment,
            approved_at="2026-08-28T02:00:00+00:00",
            effective_from="2026-08-28T02:00:00+00:00",
        )
        publish_supersession(
            root,
            superseded_decision_id=replacement_id,
            replacement_decision_id=stray.owner_decision_artifact_id,
            reason="forged transition",
            effective_at="2026-08-28T02:00:00+00:00",
            owner_evidence_ref=stray.owner_decision_artifact_id,
        )
        store["witness"] = current_supersession_head_witness(root)  # re-signed at the new head
        return "supersession_transition_unlawful"
    if mode == "divergent_transition":
        # a SECOND lawful-looking replacement of A: A is superseded twice
        second = synthetic_owner_decision_fixture(
            root,
            protocol=run.protocol,
            assessment=run.assessment,
            approved_at="2026-08-28T03:00:00+00:00",
            effective_from="2026-08-28T03:00:00+00:00",
            supersedes=prior_id,
            value_overrides={},
            persist=False,
        )
        # persist the artifact without the chain step, then publish the record
        from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope

        save_or_reuse_envelope(root, OWNER_DECISION_STORE, second)
        publish_supersession(
            root,
            superseded_decision_id=prior_id,
            replacement_decision_id=second.owner_decision_artifact_id,
            reason="divergent",
            effective_at="2026-08-28T03:00:00+00:00",
            owner_evidence_ref=second.owner_decision_artifact_id,
        )
        store["witness"] = current_supersession_head_witness(root)
        return "supersession_chain_divergent"
    raise AssertionError(mode)


_MODES = (
    "replacement_missing",
    "replacement_manifest_missing",
    "replacement_hash_invalid",
    "superseded_missing",
    "predecessor_inconsistent",
    "divergent_transition",
)


def _seam_bundle(store: dict):
    """OwnerAuthorizationBundle freeze/load (charter freeze, pipeline launch,
    activation and every other seam that binds a bundle to the store)."""

    requirement_set = AuthorizationRequirementSetEnvelope.from_payload(
        AuthorizationRequirementSetPayload(requirements=())
    )
    bundle = owner_authorization_bundle(
        None, requirement_set_id=requirement_set.requirement_set_id, decision_refs={}
    ).model_copy(
        update={
            "store_namespace_id": store["witness"].store_namespace_id,
            "supersession_head_witness": store["witness"],
        }
    )
    validate_owner_authorization(
        bundle, requirement_set, as_of_utc="2026-08-28T12:00:00Z", store_root=store["root"]
    )


def _seam_verification_ref(store: dict):
    """VerificationAuthorizationRef freeze/load (the verification run, the
    executor's real context, the MBP-1 diagnostic and the bounded preflight
    all bind it through the same helper)."""

    ref = verification_authorization_ref(
        None,
        approved_allowlist_hash="a" * 64,
        coverage_matrix_artifact_id="b" * 64,
        seed_snapshot_id="c" * 64,
    ).model_copy(
        update={
            "store_namespace_id": store["witness"].store_namespace_id,
            "supersession_head_witness": store["witness"],
        }
    )
    assert_authorization_bound_to_store(
        store["root"],
        store_namespace_id=ref.store_namespace_id,
        supersession_head_witness=ref.supersession_head_witness,
        expected_namespace_class="test",
    )


def _seam_seed_authorization(store: dict):
    """SeedProductionAuthorizationRef freeze/load."""

    root = store["root"]
    envelope = synthetic_seed_production_authorization(
        root,
        chain_replay_days=_CHAIN,
        first_intended_verification_day="2026-01-15",
        inventory=_INVENTORY,
        quant_lab_source_identity="1" * 64,
        strategy_core_commit="c" * 40,
        strategy_core_source_identity="2" * 64,
    )
    return lambda: verify_seed_production_authorization(
        root,
        envelope.seed_production_authorization_id,
        expected_profile_name=envelope.payload.baseline_profile_name,
        expected_section_config_hash=envelope.payload.resolved_section_config_hash,
        expected_chain_replay_days=_CHAIN,
        expected_source_inventory_hash=seed_chain_source_inventory_hash(_CHAIN, _INVENTORY),
        expected_quant_lab_source_identity="1" * 64,
        expected_strategy_core=("c" * 40, "2" * 64),
        now=_NOW,
    )


def _seam_regime_chain(store: dict):
    """The chain loader every regime promotion / activation / stratification
    gate / study seam runs before accepting owner evidence."""

    load_supersession_chain(store["root"])


@pytest.mark.parametrize("mode", _MODES)
def test_every_seam_refuses_an_incomplete_or_corrupt_replacement_chain(
    tmp_path, regime_lane, mode
) -> None:
    """HB-FIX-12: with the witness CURRENT (or re-signed at the advanced
    head), a chain whose replacement / superseded artifact is missing,
    hash-invalid, predecessor-inconsistent or divergent is refused by every
    seam — the head-witness rule alone would have passed the first four."""

    store = _store(tmp_path, regime_lane)
    root = store["root"]
    # the lawful chain passes every seam
    proof = verify_complete_owner_authority_chain(root, expected_head_witness=store["witness"])
    assert proof.witness == store["witness"] and len(proof.records) == 1
    assert proof.superseded_decision_ids == (store["prior"].owner_decision_artifact_id,)
    assert proof.replacement_decision_ids == (store["replacement"].owner_decision_artifact_id,)
    _seam_bundle(store)
    _seam_verification_ref(store)
    seed_check = _seam_seed_authorization(store)
    seed_check()
    _seam_regime_chain(store)
    expected_reason = _corrupt(store, mode)
    witness = store["witness"]
    if mode in ("replacement_missing", "replacement_manifest_missing",
                "replacement_hash_invalid", "superseded_missing"):
        # the head did not move: the structure-only witness rule still passes …
        assert_head_witness_current(root, witness)
    # … but the complete proof refuses
    with pytest.raises(StoreNamespaceError) as proof_refusal:
        verify_complete_owner_authority_chain(root, expected_head_witness=witness)
    assert proof_refusal.value.reason == expected_reason, str(proof_refusal.value)
    # every seam refuses with the typed reason before any data path
    with pytest.raises(AuthorizationError) as bundle_refusal:
        _seam_bundle(store)
    assert bundle_refusal.value.reason == expected_reason
    with pytest.raises(AuthorizationError) as ref_refusal:
        _seam_verification_ref(store)
    assert ref_refusal.value.reason == expected_reason
    if mode in ("predecessor_inconsistent", "divergent_transition"):
        # the seed authorization was signed at the earlier head: it refuses on
        # the witness first — a re-signed authorization refuses on the chain
        with pytest.raises(SeedProductionAuthorizationError) as stale:
            seed_check()
        assert stale.value.reason == "supersession_head_witness_mismatch"
        with pytest.raises(SeedProductionAuthorizationError) as resigned:
            synthetic_seed_production_authorization(
                root,
                chain_replay_days=_CHAIN,
                first_intended_verification_day="2026-01-15",
                inventory=_INVENTORY,
                quant_lab_source_identity="1" * 64,
                strategy_core_commit="c" * 40,
                strategy_core_source_identity="2" * 64,
            )
        assert resigned.value.reason == expected_reason
    else:
        with pytest.raises(SeedProductionAuthorizationError) as seed_refusal:
            seed_check()
        assert seed_refusal.value.reason == expected_reason
    with pytest.raises(OwnerDecisionRefusalError) as chain_refusal:
        _seam_regime_chain(store)
    assert getattr(chain_refusal.value, "reason", None) == expected_reason


def test_effective_time_and_provenance_rules_are_proven_from_the_artifacts(
    tmp_path, regime_lane
) -> None:
    """A replacement approved BEFORE the decision it supersedes, or a record
    recorded before its replacement was approved, is refused by the proof
    even though the record chain itself is structurally sound."""

    store = _store(tmp_path, regime_lane)
    root = store["root"]
    run = regime_lane
    # an earlier-approved decision hand-published as the replacement of B
    earlier = synthetic_owner_decision_fixture(
        root,
        protocol=run.protocol,
        assessment=run.assessment,
        approved_at="2026-08-27T00:00:00+00:00",
        effective_from="2026-08-27T00:00:00+00:00",
        supersedes=store["replacement"].owner_decision_artifact_id,
        persist=False,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope

    save_or_reuse_envelope(root, OWNER_DECISION_STORE, earlier)
    publish_supersession(
        root,
        superseded_decision_id=store["replacement"].owner_decision_artifact_id,
        replacement_decision_id=earlier.owner_decision_artifact_id,
        reason="earlier approval",
        effective_at="2026-08-28T05:00:00+00:00",
        owner_evidence_ref=earlier.owner_decision_artifact_id,
    )
    with pytest.raises(StoreNamespaceError) as refused:
        verify_complete_owner_authority_chain(root)
    assert refused.value.reason == "supersession_transition_unlawful"
    assert "approved before" in str(refused.value)
    with pytest.raises(OwnerDecisionRefusalError):
        load_supersession_chain(root)


def test_a_record_recorded_before_its_replacement_was_approved_is_refused(
    tmp_path, regime_lane
) -> None:
    store = _store(tmp_path, regime_lane)
    root = store["root"]
    run = regime_lane
    later = synthetic_owner_decision_fixture(
        root,
        protocol=run.protocol,
        assessment=run.assessment,
        approved_at="2026-08-29T00:00:00+00:00",
        effective_from="2026-08-29T00:00:00+00:00",
        supersedes=store["replacement"].owner_decision_artifact_id,
        persist=False,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope

    save_or_reuse_envelope(root, OWNER_DECISION_STORE, later)
    publish_supersession(
        root,
        superseded_decision_id=store["replacement"].owner_decision_artifact_id,
        replacement_decision_id=later.owner_decision_artifact_id,
        reason="recorded too early",
        effective_at="2026-08-28T02:00:00+00:00",  # before the replacement's approval
        owner_evidence_ref=later.owner_decision_artifact_id,
    )
    with pytest.raises(StoreNamespaceError) as refused:
        verify_complete_owner_authority_chain(root)
    assert refused.value.reason == "supersession_transition_unlawful"
    assert "recorded" in str(refused.value)


def test_pipeline_launch_refuses_a_corrupt_replacement_before_any_stage(tmp_path, regime_lane):
    """Pipeline freeze / launch: S00 refuses with the complete-proof reason
    when the chain names a hash-invalid replacement — the bundle is signed at
    the CURRENT head, so the head-witness rule alone would have passed."""

    from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
    from alpha_lab.agents.data_infra.ifvg.development_access import FROZEN_WARMUP_DATES
    from alpha_lab.agents.data_infra.ifvg.search.charter import (
        DatePolicy,
        SearchCharterEnvelope,
        SearchCharterPayload,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
        PipelineSemanticIdentity,
        PipelineSemanticSpecPayload,
    )
    from tests.agents.ifvg_search.pipeline_fixture import (
        STRATEGY_ONLY_STAGE_PLAN,
        build_pipeline_fixture,
    )
    from tests.agents.ifvg_search.test_pipeline_authority_seams import (
        _FULL_DATES,
        _assert_refused_at_s00,
        _launch,
    )

    fixture = build_pipeline_fixture(tmp_path / "launch")
    root = fixture["store_root"]
    initialize_test_namespace(root)
    persist_regime_protocol(root, regime_lane.protocol)
    persist_regime_assessment(root, regime_lane.assessment)
    prior = synthetic_owner_decision_fixture(
        root, protocol=regime_lane.protocol, assessment=regime_lane.assessment
    )
    replacement = synthetic_owner_decision_fixture(
        root,
        protocol=regime_lane.protocol,
        assessment=regime_lane.assessment,
        approved_at="2026-08-28T01:00:00+00:00",
        effective_from="2026-08-28T01:00:00+00:00",
        supersedes=prior.owner_decision_artifact_id,
    )
    # the charter bundle is signed at the current (one-record) head
    bundle = owner_authorization_bundle(root, requirement_set_id="1" * 64, decision_refs={})
    assert bundle.supersession_head_witness.line_count == 1
    charter = SearchCharterEnvelope.from_payload(
        SearchCharterPayload.model_validate(
            {
                **fixture["charter"].payload.model_dump(mode="json"),
                "date_policy": DatePolicy(
                    replay_dates=_FULL_DATES,
                    warmup_dates=FROZEN_WARMUP_DATES,
                    access_policy_id="development_explicit_dates_before_path_v2",
                ).model_dump(mode="json"),
                "owner_authorization": bundle.model_dump(mode="json"),
            }
        )
    )
    semantic = PipelineSemanticIdentity.from_payload(
        PipelineSemanticSpecPayload.model_validate(
            {
                **fixture["semantic"].payload.model_dump(mode="json"),
                "run_scope": "full_authorized_development",
                "date_allowlist": list(_FULL_DATES),
                "allowlist_hash": allowlist_sha256(_FULL_DATES),
                "search_charter_id": charter.search_id,
                "stage_plan": [stage.value for stage in STRATEGY_ONLY_STAGE_PLAN],
                "feature_bundle_ids": [],
                "label_policy_id": None,
                "fold_protocol_id": None,
                "model_protocol_id": None,
            }
        )
    )
    # the replacement artifact is rewritten after signing: the head does not
    # move, the witness is still current, and S00 must still refuse
    envelope_path = _artifact_dir(root, replacement.owner_decision_artifact_id) / "envelope.json"
    document = json.loads(envelope_path.read_text(encoding="utf-8"))
    document["payload"]["rationale"] = "rewritten after signing"
    envelope_path.write_text(json.dumps(document), encoding="utf-8")
    assert_head_witness_current(root, bundle.supersession_head_witness)
    _result, state = _launch(fixture, charter, semantic)
    _assert_refused_at_s00(state, "supersession_decision_unverifiable")


def test_every_real_seam_shares_the_complete_proof_call_path() -> None:
    """The plan's demonstrably shared call path: the bound-to-store helper
    runs the complete proof, and every real seam goes through it (or through
    the chain loader / the seed and bounded seams that call the proof)."""

    bound = inspect.getsource(authorization_module.assert_authorization_bound_to_store)
    assert "verify_complete_owner_authority_chain(" in bound
    assert "assert_head_witness_current(" not in bound
    for module, seam in (
        (charter_module, "assert_authorization_bound_to_store("),
        (charter_module, "validate_owner_authorization("),
        (executors_module, "assert_authorization_bound_to_store("),
        (pipeline_module, "assert_authorization_bound_to_store("),
        (pipeline_module, "validate_verification_run("),
        (verification_module, "assert_authorization_bound_to_store("),
    ):
        assert seam in inspect.getsource(module), (module.__name__, seam)
    from alpha_lab.agents.data_infra.ifvg.features import mbp1_coverage_diagnostic
    from alpha_lab.agents.data_infra.ifvg.ml import (
        regime_block_activation,
        regime_store,
        regime_stratification_gate,
        regime_study,
    )

    assert "assert_authorization_bound_to_store(" in inspect.getsource(mbp1_coverage_diagnostic)
    for module in (regime_block_activation, regime_store, regime_stratification_gate, regime_study):
        assert "load_supersession_chain(root)" in inspect.getsource(module), module.__name__
    assert "verify_complete_owner_authority_chain(" in inspect.getsource(
        seed_module._witness_current
    )
    assert "verify_complete_owner_authority_chain(" in inspect.getsource(bounded_module)
    # the chain loader IS the complete proof
    from alpha_lab.agents.data_infra.ifvg.search import owner_decisions as owner_module

    assert "verify_complete_owner_authority_chain(root).records" in inspect.getsource(
        owner_module.load_supersession_chain
    )
