"""R6.1 workstream F — verified owner-decision evidence (plan §6.F / D5;
§9.1 ``test_decision_25_algorithm_snapshot_is_authorized``; §9.2).

A bare 64-hex reference is not evidence: FEATURE_ELIGIBLE persists only
against a store-verified owner-decision artifact that binds the exact
protocol + assessment, states decisions 25/28/29/30 with the registry /
protocol / assessment values, authorizes exactly this transition, is
effective at the decision instant, and is not superseded (store-owned
chain, fail closed). Verification never mutates a decision payload.

HARDENING-BACKEND (§4.1–§4.3): authority is the store's VERIFIED semantic
namespace (never the path), supersession is an immutable record chain with
a mandatory head, and the writer lock is liveness-aware — the R6.1 JSONL-log
tests below were rewritten against those contracts.
"""

from __future__ import annotations

import json
import shutil

import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    MODEL_FEATURE_PROMOTION_REFUSAL,
    RegimePromotionDecision,
    RegimePromotionDecisionEnvelope,
    RegimeRole,
    RegimeStatus,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    load_regime_promotion,
    persist_regime_assessment,
    persist_regime_promotion,
    persist_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.search import owner_decisions as owner_decisions_module
from alpha_lab.agents.data_infra.ifvg.search import supersession_chain as chain_module
from alpha_lab.agents.data_infra.ifvg.search.owner_decision_lock import (
    OWNER_DECISION_LOCK_FILE,
    OwnerDecisionLock,
    OwnerDecisionLockError,
    current_process_start_token,
)
from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (
    AUTHORIZED_TRANSITIONS,
    OWNER_DECISION_STORE,
    OWNER_DECISION_SUPERSESSION_STORE,
    REGIME_DECISION_KEYS,
    SUPERSESSIONS_HEAD_FILE,
    OwnerDecisionArtifactEnvelope,
    OwnerDecisionArtifactPayload,
    OwnerDecisionRefusalError,
    build_owner_decision_proposal,
    expected_decision_values,
    freeze_decision_values,
    load_owner_decision,
    load_supersession_chain,
    persist_owner_decision,
    render_owner_decision_proposal_markdown,
    synthetic_owner_decision_fixture,
)
from alpha_lab.agents.data_infra.ifvg.search.store import SearchStoreError, envelope_destination
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    StoreNamespaceError,
    initialize_store_namespace,
    initialize_test_namespace,
    load_store_namespace,
    write_supersession_head_atomic,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
    current_supersession_head_witness,
    load_supersession_records,
    publish_supersession,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import known_cluster_fixture

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id
_DECIDED_AT = "2026-08-28T12:00:00+00:00"


def _run(n: int, **protocol_overrides):
    fixture = known_cluster_fixture(k=3, n=n)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0,
        resolved_input_features=fixture.regime_input_features,
        **protocol_overrides,
    )
    return run_regime_protocol(
        fixture.view.frame,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=3,
    )


def _decision(run, **overrides) -> RegimePromotionDecisionEnvelope:
    defaults = dict(
        resolved_regime_protocol_id=run.protocol.resolved_regime_protocol_id,
        role=RegimeRole.DESCRIPTIVE_ONLY,
        status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_status=RegimeStatus.PLANNED,
        previous_decision_ref=None,
        capability_assessment_ref=run.assessment.regime_capability_assessment_id,
        owner_ratification_ref=None,
        decided_at=_DECIDED_AT,
    )
    defaults.update(overrides)
    return RegimePromotionDecisionEnvelope.from_payload(RegimePromotionDecision(**defaults))


@pytest.fixture(scope="module")
def lane(tmp_path_factory):
    root = tmp_path_factory.mktemp("owner_decisions")
    initialize_test_namespace(root)  # HARDENING-BACKEND §4.1: an explicit test namespace
    run = _run(600)
    persist_regime_protocol(root, run.protocol)
    persist_regime_assessment(root, run.assessment)
    assert run.assessment.payload.gates_passed
    small = _run(170, winsorization_policy="clip_p01_p99_train_fitted_v1")
    persist_regime_protocol(root, small.protocol)
    persist_regime_assessment(root, small.assessment)
    first = _decision(run)
    persist_regime_promotion(root, first)
    ready = _decision(
        run,
        role=RegimeRole.STRATIFICATION_ONLY,
        status=RegimeStatus.STRATIFICATION_READY,
        previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_decision_ref=first.regime_promotion_decision_id,
    )
    persist_regime_promotion(root, ready)
    return {"root": root, "run": run, "small": small, "ready": ready}


def _eligible(lane, ref: str) -> RegimePromotionDecisionEnvelope:
    return _decision(
        lane["run"],
        role=RegimeRole.FEATURE_GENERATOR,
        status=RegimeStatus.FEATURE_ELIGIBLE,
        previous_status=RegimeStatus.STRATIFICATION_READY,
        previous_decision_ref=lane["ready"].regime_promotion_decision_id,
        owner_ratification_ref=ref,
    )


def test_bare_64_hex_reference_is_not_evidence(lane):
    with pytest.raises(ValueError, match="not a verified owner-decision artifact"):
        persist_regime_promotion(lane["root"], _eligible(lane, "f" * 64))
    with pytest.raises(ValueError, match="not a verified owner-decision artifact"):
        persist_regime_promotion(
            lane["root"], _eligible(lane, "f" * 64), run_scope="synthetic_fixture"
        )


def test_lawful_synthetic_chain_in_synthetic_scope_and_verification_mutates_nothing(lane):
    root, run = lane["root"], lane["run"]
    artifact = synthetic_owner_decision_fixture(
        root, protocol=run.protocol, assessment=run.assessment
    )
    assert artifact.payload.provenance == "synthetic_test_authorization_v1"
    assert set(artifact.payload.decision_keys) >= set(REGIME_DECISION_KEYS)
    before = artifact.model_dump(mode="json")
    eligible = _eligible(lane, artifact.owner_decision_artifact_id)
    persist_regime_promotion(root, eligible, run_scope="synthetic_fixture")
    stored = load_regime_promotion(root, eligible.regime_promotion_decision_id)
    assert stored.payload.status is RegimeStatus.FEATURE_ELIGIBLE
    assert stored.payload.owner_ratification_ref == artifact.owner_decision_artifact_id
    # verification changed neither the decision nor the owner artifact
    assert stored.model_dump(mode="json") == eligible.model_dump(mode="json")
    assert load_owner_decision(root, artifact.owner_decision_artifact_id).model_dump(
        mode="json"
    ) == before
    ref = artifact.evidence_ref()
    assert ref.content_hash == artifact.owner_decision_artifact_id
    assert ref.decision_artifact_id == artifact.owner_decision_artifact_id
    assert run.assessment.regime_capability_assessment_id in ref.reviewed_evidence_refs


def test_synthetic_provenance_is_refused_outside_the_synthetic_scope(lane):
    root, run = lane["root"], lane["run"]
    artifact = synthetic_owner_decision_fixture(
        root, protocol=run.protocol, assessment=run.assessment
    )
    for scope in ("verification_5d", "full_authorized_development"):
        with pytest.raises(ValueError, match="lawful only in the synthetic_fixture run scope"):
            persist_regime_promotion(
                root, _eligible(lane, artifact.owner_decision_artifact_id), run_scope=scope
            )


@pytest.mark.parametrize(
    ("key", "wrong"),
    [
        ("algorithm_key", "gmm_v1"),
        ("algorithm_parameters_hash", "0" * 64),
        ("algorithm_parameters", {"n_clusters": 2}),
        ("observation_granularity", "context_bar_panel"),
        ("panel_interval_seconds", 300),
        ("observation_stage", "htf_tap"),
        ("fixed_cluster_count", 4),
        ("minimum_cluster_occupancy_fraction", 0.5),
        ("minimum_cluster_rows_per_fold", 1),
        ("minimum_bootstrap_aligned_ami_mean", 0.1),
        ("minimum_training_observations", 1),
    ],
)
def test_decision_25_28_29_30_values_are_verified_against_registry_protocol_assessment(
    lane, key, wrong
):
    """§9.1 ``test_decision_25_algorithm_snapshot_is_authorized`` (and its
    28/29/30 siblings): a wrong algorithm key / pinned-parameter snapshot
    is refused even when every other decision matches."""

    root, run = lane["root"], lane["run"]
    artifact = synthetic_owner_decision_fixture(
        root, protocol=run.protocol, assessment=run.assessment, value_overrides={key: wrong}
    )
    with pytest.raises(ValueError, match=f"decision values disagree.*{key}"):
        persist_regime_promotion(
            root,
            _eligible(lane, artifact.owner_decision_artifact_id),
            run_scope="synthetic_fixture",
        )


def test_expected_values_bind_the_exact_pinned_snapshot(lane):
    run = lane["run"]
    values = expected_decision_values(run.protocol, run.assessment)
    assert values["algorithm_key"] == "kmeans_v1"
    assert values["algorithm_parameters"]["n_init"] == 10
    assert values["algorithm_parameters"]["random_state"] == 7
    assert values["fixed_cluster_count"] == 3
    assert values["minimum_bootstrap_aligned_ami_mean"] == pytest.approx(0.5)
    assert values["minimum_training_observations"] == 150


def test_wrong_protocol_or_wrong_assessment_is_refused(lane):
    root, run, small = lane["root"], lane["run"], lane["small"]
    other_protocol = synthetic_owner_decision_fixture(
        root, protocol=small.protocol, assessment=small.assessment
    )
    with pytest.raises(ValueError, match="ratifies a different regime protocol"):
        persist_regime_promotion(
            root,
            _eligible(lane, other_protocol.owner_decision_artifact_id),
            run_scope="synthetic_fixture",
        )
    other_assessment = synthetic_owner_decision_fixture(
        root, protocol=run.protocol, assessment=small.assessment
    )
    with pytest.raises(ValueError, match="reviewed a different capability assessment"):
        persist_regime_promotion(
            root,
            _eligible(lane, other_assessment.owner_decision_artifact_id),
            run_scope="synthetic_fixture",
        )


def test_assessment_must_be_among_the_reviewed_evidence(lane):
    run = lane["run"]
    values = expected_decision_values(run.protocol, run.assessment)
    with pytest.raises(ValueError, match="must contain the capability assessment id"):
        OwnerDecisionArtifactPayload(
            store_namespace_id="c" * 64,
            decision_keys=REGIME_DECISION_KEYS,
            decision_values=values,
            resolved_regime_protocol_id=run.protocol.resolved_regime_protocol_id,
            capability_assessment_id=run.assessment.regime_capability_assessment_id,
            authorized_transitions=AUTHORIZED_TRANSITIONS,
            provenance="synthetic_test_authorization_v1",
            author="x",
            approved_at="2026-08-28T00:00:00+00:00",
            effective_from="2026-08-28T00:00:00+00:00",
            effective_to=None,
            reviewed_evidence_refs=("e" * 64,),
            supersedes=None,
            rationale="x",
        )


def test_transition_effectivity_and_expiry_are_exact(lane):
    root, run = lane["root"], lane["run"]
    only_model_feature = synthetic_owner_decision_fixture(
        root,
        protocol=run.protocol,
        assessment=run.assessment,
        transitions=("feature_eligible->model_feature",),
    )
    with pytest.raises(ValueError, match="is not authorized by the artifact"):
        persist_regime_promotion(
            root,
            _eligible(lane, only_model_feature.owner_decision_artifact_id),
            run_scope="synthetic_fixture",
        )
    future = synthetic_owner_decision_fixture(
        root,
        protocol=run.protocol,
        assessment=run.assessment,
        approved_at="2027-01-01T00:00:00+00:00",
        effective_from="2027-01-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="not yet effective"):
        persist_regime_promotion(
            root, _eligible(lane, future.owner_decision_artifact_id), run_scope="synthetic_fixture"
        )
    expired = synthetic_owner_decision_fixture(
        root,
        protocol=run.protocol,
        assessment=run.assessment,
        effective_to="2026-08-28T06:00:00+00:00",
    )
    with pytest.raises(ValueError, match="has expired"):
        persist_regime_promotion(
            root, _eligible(lane, expired.owner_decision_artifact_id), run_scope="synthetic_fixture"
        )


def test_supersession_is_store_owned_and_fails_closed(lane, tmp_path):
    root = tmp_path / "supersession"
    run = lane["run"]
    persist_regime_protocol(root, run.protocol)
    persist_regime_assessment(root, run.assessment)
    first = _decision(run)
    persist_regime_promotion(root, first)
    ready = _decision(
        run,
        role=RegimeRole.STRATIFICATION_ONLY,
        status=RegimeStatus.STRATIFICATION_READY,
        previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_decision_ref=first.regime_promotion_decision_id,
    )
    persist_regime_promotion(root, ready)
    local = {"root": root, "run": run, "ready": ready}
    prior = synthetic_owner_decision_fixture(root, protocol=run.protocol, assessment=run.assessment)
    # a supersession must name a VERIFIED prior of the same protocol
    with pytest.raises(OwnerDecisionRefusalError, match="prior decision is not a verified"):
        synthetic_owner_decision_fixture(
            root, protocol=run.protocol, assessment=run.assessment, supersedes="a" * 64
        )
    replacement = synthetic_owner_decision_fixture(
        root,
        protocol=run.protocol,
        assessment=run.assessment,
        supersedes=prior.owner_decision_artifact_id,
        approved_at="2026-08-28T01:00:00+00:00",
        effective_from="2026-08-28T01:00:00+00:00",
    )
    chain = load_supersession_chain(root)
    assert [(r.superseded_artifact_id, r.replacement_artifact_id) for r in chain] == [
        (prior.owner_decision_artifact_id, replacement.owner_decision_artifact_id)
    ]
    with pytest.raises(ValueError, match="superseded by a later decision"):
        persist_regime_promotion(
            root, _eligible(local, prior.owner_decision_artifact_id), run_scope="synthetic_fixture"
        )
    persist_regime_promotion(
        root,
        _eligible(local, replacement.owner_decision_artifact_id),
        run_scope="synthetic_fixture",
    )
    # a forged supersession RECORD (no verified replacement artifact) fails
    # the whole chain closed — the chain is store-owned, never a log line
    publish_supersession(
        root,
        superseded_decision_id=replacement.owner_decision_artifact_id,
        replacement_decision_id="b" * 64,
        reason="forged",
        effective_at="2026-08-28T02:00:00+00:00",
        owner_evidence_ref="b" * 64,
    )
    with pytest.raises(OwnerDecisionRefusalError, match="chain fails closed"):
        load_supersession_chain(root)
    later = _decision(
        run,
        role=RegimeRole.FEATURE_GENERATOR,
        status=RegimeStatus.FEATURE_ELIGIBLE,
        previous_status=RegimeStatus.STRATIFICATION_READY,
        previous_decision_ref=ready.regime_promotion_decision_id,
        owner_ratification_ref=replacement.owner_decision_artifact_id,
        decided_at="2026-08-28T13:00:00+00:00",
    )
    with pytest.raises(ValueError, match="chain fails closed"):
        persist_regime_promotion(root, later, run_scope="synthetic_fixture")


def test_tampered_artifact_is_refused(lane, tmp_path):
    root = tmp_path / "tamper"
    run = lane["run"]
    persist_regime_protocol(root, run.protocol)
    persist_regime_assessment(root, run.assessment)
    artifact = synthetic_owner_decision_fixture(
        root, protocol=run.protocol, assessment=run.assessment
    )
    envelope_path = (
        envelope_destination(root, OWNER_DECISION_STORE, artifact.owner_decision_artifact_id)
        / "envelope.json"
    )
    payload = json.loads(envelope_path.read_text(encoding="utf-8"))
    payload["payload"]["rationale"] = "tampered after the fact"
    envelope_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SearchStoreError):
        load_owner_decision(root, artifact.owner_decision_artifact_id)
    first = _decision(run)
    persist_regime_promotion(root, first)
    ready = _decision(
        run,
        role=RegimeRole.STRATIFICATION_ONLY,
        status=RegimeStatus.STRATIFICATION_READY,
        previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_decision_ref=first.regime_promotion_decision_id,
    )
    persist_regime_promotion(root, ready)
    with pytest.raises(ValueError, match="not a verified owner-decision artifact"):
        persist_regime_promotion(
            root,
            _eligible(
                {"root": root, "run": run, "ready": ready}, artifact.owner_decision_artifact_id
            ),
            run_scope="synthetic_fixture",
        )


def test_proposals_carry_placeholders_that_cannot_persist(lane):
    run = lane["run"]
    proposal = build_owner_decision_proposal(run.protocol, run.assessment)
    assert proposal["decision_values"] == expected_decision_values(run.protocol, run.assessment)
    with pytest.raises(ValueError):
        OwnerDecisionArtifactPayload.model_validate(
            {k: v for k, v in proposal.items() if not k.startswith("_")}
        )
    text = render_owner_decision_proposal_markdown(proposal, title="Draft")
    assert "PROPOSAL — nothing here is an authorization" in text
    assert "| 25 | algorithm_key | `kmeans_v1` |" in text
    assert "<OWNER_TO_FILL>" in text
    with pytest.raises(ValueError, match="unregistered transition"):
        build_owner_decision_proposal(run.protocol, run.assessment, transition="x->y")


def test_persist_is_idempotent_and_the_envelope_is_the_evidence_ref(lane, tmp_path):
    root = tmp_path / "idempotent"
    run = lane["run"]
    artifact = synthetic_owner_decision_fixture(
        root, protocol=run.protocol, assessment=run.assessment
    )
    again = persist_owner_decision(root, artifact, recorded_at="2026-08-28T00:00:00+00:00")
    assert again.owner_decision_artifact_id == artifact.owner_decision_artifact_id
    assert isinstance(again, OwnerDecisionArtifactEnvelope)
    assert load_supersession_chain(root) == ()


# ── adversarial round: S2 / F8 (supersession crash safety + integrity), F2 (D6
# at persistence), S3 (namespace guard), S5 (MODEL_FEATURE unpersistable),
# S11 (monotone chain), S10 (stale lock) ─────────────────────────────────────


def _chain_root(tmp_path, lane, name: str):
    root = tmp_path / name
    initialize_test_namespace(root)
    run = lane["run"]
    persist_regime_protocol(root, run.protocol)
    persist_regime_assessment(root, run.assessment)
    first = _decision(run)
    persist_regime_promotion(root, first)
    ready = _decision(
        run,
        role=RegimeRole.STRATIFICATION_ONLY,
        status=RegimeStatus.STRATIFICATION_READY,
        previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_decision_ref=first.regime_promotion_decision_id,
    )
    persist_regime_promotion(root, ready)
    return root, {"root": root, "run": run, "ready": ready}


def _owner_signed(root, run, *, approved_at="2026-08-28T00:00:00+00:00", supersedes=None):
    payload = OwnerDecisionArtifactPayload(
        store_namespace_id=load_store_namespace(root).store_namespace_id,
        decision_keys=REGIME_DECISION_KEYS,
        decision_values=freeze_decision_values(
            expected_decision_values(run.protocol, run.assessment)
        ),
        resolved_regime_protocol_id=run.protocol.resolved_regime_protocol_id,
        capability_assessment_id=run.assessment.regime_capability_assessment_id,
        authorized_transitions=AUTHORIZED_TRANSITIONS,
        provenance="owner_signed",
        author="owner",
        approved_at=approved_at,
        effective_from=approved_at,
        effective_to=None,
        reviewed_evidence_refs=(run.assessment.regime_capability_assessment_id,),
        supersedes=supersedes,
        rationale="signed test artifact (synthetic evidence; test namespace only)",
    )
    return persist_owner_decision(
        root, OwnerDecisionArtifactEnvelope.from_payload(payload), recorded_at=approved_at
    )


def test_supersession_line_precedes_publication_and_a_crash_fails_closed(
    lane, tmp_path, monkeypatch
):
    """S2 (probe P1): the chained line is written BEFORE the replacement is
    published; a crash in between leaves the chain fail-closed (the prior
    can no longer authorize) until the replacement is re-persisted, which
    is idempotent (exactly one line)."""

    root, local = _chain_root(tmp_path, lane, "crash")
    run = lane["run"]
    prior = synthetic_owner_decision_fixture(root, protocol=run.protocol, assessment=run.assessment)
    persist_regime_promotion(
        root, _eligible(local, prior.owner_decision_artifact_id), run_scope="synthetic_fixture"
    )

    def _crash(*_args, **_kwargs):
        raise RuntimeError("simulated crash between the supersession line and publication")

    monkeypatch.setattr(owner_decisions_module, "save_or_reuse_envelope", _crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        synthetic_owner_decision_fixture(
            root,
            protocol=run.protocol,
            assessment=run.assessment,
            supersedes=prior.owner_decision_artifact_id,
            approved_at="2026-08-28T01:00:00+00:00",
            effective_from="2026-08-28T01:00:00+00:00",
        )
    monkeypatch.undo()
    # the immutable record exists and the head names it, but the replacement
    # was never published: the chain fails closed (the prior cannot authorize)
    assert len(load_supersession_records(root)) == 1
    assert current_supersession_head_witness(root).line_count == 1
    with pytest.raises(OwnerDecisionRefusalError, match="not a verified store entry"):
        load_supersession_chain(root)
    later = _decision(
        run,
        role=RegimeRole.FEATURE_GENERATOR,
        status=RegimeStatus.FEATURE_ELIGIBLE,
        previous_status=RegimeStatus.STRATIFICATION_READY,
        previous_decision_ref=local["ready"].regime_promotion_decision_id,
        owner_ratification_ref=prior.owner_decision_artifact_id,
        decided_at="2026-08-28T13:00:00+00:00",
    )
    with pytest.raises(ValueError, match="chain fails closed"):
        persist_regime_promotion(root, later, run_scope="synthetic_fixture")
    # re-persisting the replacement repairs the chain without a second line
    replacement = synthetic_owner_decision_fixture(
        root,
        protocol=run.protocol,
        assessment=run.assessment,
        supersedes=prior.owner_decision_artifact_id,
        approved_at="2026-08-28T01:00:00+00:00",
        effective_from="2026-08-28T01:00:00+00:00",
    )
    persist_owner_decision(root, replacement, recorded_at="2026-08-28T01:00:00+00:00")
    assert len(load_supersession_records(root)) == 1  # no second record
    assert current_supersession_head_witness(root).line_count == 1
    chain = load_supersession_chain(root)
    assert [(r.superseded_artifact_id, r.replacement_artifact_id) for r in chain] == [
        (prior.owner_decision_artifact_id, replacement.owner_decision_artifact_id)
    ]
    with pytest.raises(ValueError, match="superseded by a later decision"):
        persist_regime_promotion(root, later, run_scope="synthetic_fixture")


def test_deleted_edited_rolled_back_or_headless_chain_fails_closed(lane, tmp_path):
    """S2 (probe P2) under HARDENING-BACKEND §4.2: records are immutable
    store entries and the head commits to the whole chain — deleting a
    record, editing it in place, rolling the head back, or deleting the head
    can never restore a revoked decision's authority; every state is a typed
    refusal."""

    root, _local = _chain_root(tmp_path, lane, "tamper_chain")
    run = lane["run"]
    prior = synthetic_owner_decision_fixture(root, protocol=run.protocol, assessment=run.assessment)
    replacement = synthetic_owner_decision_fixture(
        root,
        protocol=run.protocol,
        assessment=run.assessment,
        supersedes=prior.owner_decision_artifact_id,
        approved_at="2026-08-28T01:00:00+00:00",
        effective_from="2026-08-28T01:00:00+00:00",
    )
    second = synthetic_owner_decision_fixture(
        root,
        protocol=run.protocol,
        assessment=run.assessment,
        supersedes=replacement.owner_decision_artifact_id,
        approved_at="2026-08-28T02:00:00+00:00",
        effective_from="2026-08-28T02:00:00+00:00",
    )
    records = load_supersession_records(root)
    assert [r.replacement_artifact_id for r in load_supersession_chain(root)] == [
        replacement.owner_decision_artifact_id,
        second.owner_decision_artifact_id,
    ]
    witness = current_supersession_head_witness(root)
    namespace = load_store_namespace(root)
    head = root / OWNER_DECISION_STORE / SUPERSESSIONS_HEAD_FILE
    original_head = head.read_text(encoding="utf-8")
    first_dir = envelope_destination(
        root, OWNER_DECISION_SUPERSESSION_STORE, records[0].supersession_record_id
    )
    # (i) the first record deleted -> the chain no longer reaches genesis
    shutil.move(first_dir, tmp_path / "parked_record")
    with pytest.raises(OwnerDecisionRefusalError, match="chain fails closed"):
        load_supersession_chain(root)
    shutil.move(tmp_path / "parked_record", first_dir)
    # (ii) the head rolled back to the first record: structurally valid but
    # SHORTER than the witnessed head — refused by every witness holder
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import chain_head_digest
    from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
        assert_head_witness_current,
    )

    write_supersession_head_atomic(
        root,
        store_namespace_id=namespace.store_namespace_id,
        record_id=records[0].supersession_record_id,
        line_count=1,
        head_sha256=chain_head_digest(
            namespace.payload.authority_genesis_id, records[0].supersession_record_id
        ),
    )
    with pytest.raises(StoreNamespaceError) as shorter:
        assert_head_witness_current(root, witness)
    assert shorter.value.reason == "supersession_head_shorter_than_witness"
    # ...and the rolled-back chain still names `replacement` as superseded, so
    # the ORIGINAL prior cannot authorize either way
    with pytest.raises(ValueError, match="superseded by a later decision"):
        persist_regime_promotion(
            root, _eligible(_local, prior.owner_decision_artifact_id), run_scope="synthetic_fixture"
        )
    # (iii) a record edited in place -> the store manifest refuses it
    head.write_text(original_head, encoding="utf-8")
    envelope_file = (
        envelope_destination(
            root, OWNER_DECISION_SUPERSESSION_STORE, records[1].supersession_record_id
        )
        / "envelope.json"
    )
    edited_original = envelope_file.read_text(encoding="utf-8")
    edited = json.loads(edited_original)
    edited["payload"]["effective_at"] = "2026-08-28T09:00:00+00:00"
    envelope_file.write_text(json.dumps(edited, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(OwnerDecisionRefusalError, match="chain fails closed"):
        load_supersession_chain(root)
    envelope_file.write_text(edited_original, encoding="utf-8")
    # (iv) the head deleted -> corruption, never "no supersessions"
    head.unlink()
    with pytest.raises(OwnerDecisionRefusalError, match="supersession_head_missing"):
        load_supersession_chain(root)
    # restored -> verifies again; every refusal above was fail-closed
    head.write_text(original_head, encoding="utf-8")
    assert len(load_supersession_chain(root)) == 2
    assert_head_witness_current(root, witness)
    # nothing ever wrote into an existing decision directory: the superseded
    # artifacts' bytes are untouched
    for artifact in (prior, replacement):
        directory = envelope_destination(
            root, OWNER_DECISION_STORE, artifact.owner_decision_artifact_id
        )
        assert sorted(p.name for p in directory.iterdir()) == ["envelope.json", "manifest.json"]
        assert load_owner_decision(root, artifact.owner_decision_artifact_id) == artifact


def test_a_replacement_may_never_carry_weaker_provenance(lane, tmp_path):
    """S2: a synthetic artifact cannot revoke an owner_signed one (in any
    scope); the stronger direction is lawful."""

    root, _local = _chain_root(tmp_path, lane, "provenance")
    run = lane["run"]
    signed = _owner_signed(root, run)
    with pytest.raises(OwnerDecisionRefusalError, match="weaker provenance"):
        synthetic_owner_decision_fixture(
            root,
            protocol=run.protocol,
            assessment=run.assessment,
            supersedes=signed.owner_decision_artifact_id,
            approved_at="2026-08-28T01:00:00+00:00",
            effective_from="2026-08-28T01:00:00+00:00",
        )
    assert load_supersession_chain(root) == ()
    synthetic = synthetic_owner_decision_fixture(
        root, protocol=run.protocol, assessment=run.assessment
    )
    stronger = _owner_signed(
        root,
        run,
        approved_at="2026-08-28T01:00:00+00:00",
        supersedes=synthetic.owner_decision_artifact_id,
    )
    assert [r.replacement_artifact_id for r in load_supersession_chain(root)] == [
        stronger.owner_decision_artifact_id
    ]


def test_dead_writer_lock_is_reclaimed_and_a_live_holder_is_never_reclaimed(
    lane, tmp_path, monkeypatch
):
    """S10 under HARDENING-BACKEND §4.3: a lock left by a DEAD writer (stale
    heartbeat + demonstrably dead pid) is reclaimed; a LIVE holder — even
    with an ancient heartbeat — is never reclaimed and the waiter times out
    with a typed reason; a lost lock aborts before the head moves."""

    import functools
    import os
    import platform

    root, _local = _chain_root(tmp_path, lane, "lock")
    run = lane["run"]
    prior = synthetic_owner_decision_fixture(root, protocol=run.protocol, assessment=run.assessment)
    lock = root / OWNER_DECISION_STORE / OWNER_DECISION_LOCK_FILE
    monkeypatch.setattr(
        chain_module,
        "OwnerDecisionLock",
        functools.partial(OwnerDecisionLock, wait_seconds=0.3, heartbeat_timeout_seconds=1.0),
    )

    def _lock_body(pid, token, start_token=None):
        return json.dumps(
            {
                "lock_schema_version": 1,
                "pid": pid,
                "process_start_token": start_token,
                "lock_token": token,
                "host": platform.node(),
                "created_at": "2026-01-01T00:00:00+00:00",
                "heartbeat_at": "2026-01-01T00:00:00+00:00",
            }
        )

    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(_lock_body(999_999_999, "d" * 32), encoding="utf-8")  # a dead writer
    replacement = synthetic_owner_decision_fixture(
        root,
        protocol=run.protocol,
        assessment=run.assessment,
        supersedes=prior.owner_decision_artifact_id,
        approved_at="2026-08-28T01:00:00+00:00",
        effective_from="2026-08-28T01:00:00+00:00",
    )
    assert not lock.exists()
    assert [r.replacement_artifact_id for r in load_supersession_chain(root)] == [
        replacement.owner_decision_artifact_id
    ]
    # a LIVE holder (this process) with an ancient heartbeat: never reclaimed
    lock.write_text(
        _lock_body(os.getpid(), "1" * 32, current_process_start_token()), encoding="utf-8"
    )
    with pytest.raises(OwnerDecisionLockError) as held:
        synthetic_owner_decision_fixture(
            root,
            protocol=run.protocol,
            assessment=run.assessment,
            supersedes=replacement.owner_decision_artifact_id,
            approved_at="2026-08-28T02:00:00+00:00",
            effective_from="2026-08-28T02:00:00+00:00",
        )
    assert held.value.reason == "lock_held_by_live_holder"
    assert json.loads(lock.read_text(encoding="utf-8"))["lock_token"] == "1" * 32
    assert len(load_supersession_records(root)) == 1  # nothing moved
    lock.unlink()


def test_stratification_ready_needs_coverage_gates_and_oos_at_persistence(lane):
    """Plan 9.2 / F2 (D6): STRATIFICATION_READY over an assessment whose
    coverage gates fail is unpersistable whoever mints it; the BLOCKED state
    is the lawful record, and the ladder cannot climb from it."""

    root, small = lane["root"], lane["small"]
    assert not small.assessment.payload.coverage.coverage_gates_passed
    small_first = _decision(small)
    persist_regime_promotion(root, small_first)
    with pytest.raises(ValueError, match=r"structurally requires .* \(D6\)"):
        persist_regime_promotion(
            root,
            _decision(
                small,
                role=RegimeRole.STRATIFICATION_ONLY,
                status=RegimeStatus.STRATIFICATION_READY,
                previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
                previous_decision_ref=small_first.regime_promotion_decision_id,
            ),
        )
    blocked = _decision(
        small,
        status=RegimeStatus.BLOCKED_INSUFFICIENT_COVERAGE,
        previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_decision_ref=small_first.regime_promotion_decision_id,
    )
    persist_regime_promotion(root, blocked)
    with pytest.raises(ValueError, match="skips the sequence"):
        _decision(
            small,
            role=RegimeRole.FEATURE_GENERATOR,
            status=RegimeStatus.FEATURE_ELIGIBLE,
            previous_status=RegimeStatus.BLOCKED_INSUFFICIENT_COVERAGE,
            previous_decision_ref=blocked.regime_promotion_decision_id,
            owner_ratification_ref="f" * 64,
        )


def test_decision_time_is_monotone_along_the_chain(lane):
    """S11: a ratified decision dated before its predecessor is refused."""

    root, run = lane["root"], lane["run"]
    artifact = synthetic_owner_decision_fixture(
        root, protocol=run.protocol, assessment=run.assessment
    )
    with pytest.raises(ValueError, match="monotone in decision time"):
        persist_regime_promotion(
            root,
            _decision(
                run,
                role=RegimeRole.FEATURE_GENERATOR,
                status=RegimeStatus.FEATURE_ELIGIBLE,
                previous_status=RegimeStatus.STRATIFICATION_READY,
                previous_decision_ref=lane["ready"].regime_promotion_decision_id,
                owner_ratification_ref=artifact.owner_decision_artifact_id,
                decided_at="2026-08-28T11:00:00+00:00",
            ),
            run_scope="synthetic_fixture",
        )


def test_model_feature_is_unpersistable_through_the_store(lane):
    """S5 (probe P4): the top of the ladder is refused by the store with the
    CLI's exact text — no activation / controlled-study precondition can be
    forged by a helper, a notebook, or a future seam."""

    root, run = lane["root"], lane["run"]
    artifact = synthetic_owner_decision_fixture(
        root, protocol=run.protocol, assessment=run.assessment
    )
    eligible = _eligible(lane, artifact.owner_decision_artifact_id)
    persist_regime_promotion(root, eligible, run_scope="synthetic_fixture")
    top = _decision(
        run,
        role=RegimeRole.FEATURE_GENERATOR,
        status=RegimeStatus.MODEL_FEATURE,
        previous_status=RegimeStatus.FEATURE_ELIGIBLE,
        previous_decision_ref=eligible.regime_promotion_decision_id,
        owner_ratification_ref=artifact.owner_decision_artifact_id,
        decided_at="2026-08-28T13:00:00+00:00",
    )
    with pytest.raises(PermissionError) as refused:
        persist_regime_promotion(root, top, run_scope="synthetic_fixture")
    assert str(refused.value) == MODEL_FEATURE_PROMOTION_REFUSAL
    with pytest.raises(SearchStoreError):
        load_regime_promotion(root, top.regime_promotion_decision_id)


def test_synthetic_scope_and_provenance_are_confined_to_test_namespaces(lane, tmp_path):
    """S3 (probe P3, P0-4 mirror) under HARDENING-BACKEND §4.1: authority is
    the store's VERIFIED namespace class — a marked ``research`` namespace at
    a PLAIN path refuses the synthetic scope at persistence and refuses a
    synthetic owner artifact at persist AND at load (an artifact copied in
    from a test namespace names another namespace); an UNMARKED store carries
    no owner authority at all; the pathname heuristic survives only as
    defense in depth; a ``test`` namespace under ``search_test`` admits
    both."""

    run = lane["run"]
    # (a) a marked RESEARCH namespace at a plain path (the path grants nothing)
    research = tmp_path / "plain_research_store"
    initialize_store_namespace(research, namespace_class="research", store_instance_id="a3" * 16)
    persist_regime_protocol(research, run.protocol)
    persist_regime_assessment(research, run.assessment)
    first = _decision(run)
    persist_regime_promotion(research, first)  # the real scope is lawful here
    with pytest.raises(PermissionError, match="confined to test namespaces"):
        persist_regime_promotion(research, first, run_scope="synthetic_fixture")
    with pytest.raises(OwnerDecisionRefusalError, match="confined to test namespaces"):
        synthetic_owner_decision_fixture(research, protocol=run.protocol, assessment=run.assessment)
    # a synthetic artifact copied in from a test root names ANOTHER namespace
    artifact = synthetic_owner_decision_fixture(
        lane["root"], protocol=run.protocol, assessment=run.assessment
    )
    artifact_id = artifact.owner_decision_artifact_id
    shutil.copytree(
        envelope_destination(lane["root"], OWNER_DECISION_STORE, artifact_id),
        envelope_destination(research, OWNER_DECISION_STORE, artifact_id),
    )
    with pytest.raises(OwnerDecisionRefusalError, match="another store namespace"):
        load_owner_decision(research, artifact_id)
    # (b) an UNMARKED store: no owner authority (typed), even under a test path
    unmarked = tmp_path / "unmarked" / "search_test" / "v1"
    persist_regime_protocol(unmarked, run.protocol)
    persist_regime_assessment(unmarked, run.assessment)
    persist_regime_promotion(unmarked, first, run_scope="synthetic_fixture")  # structural
    shutil.copytree(
        envelope_destination(lane["root"], OWNER_DECISION_STORE, artifact_id),
        envelope_destination(unmarked, OWNER_DECISION_STORE, artifact_id),
    )
    with pytest.raises(OwnerDecisionRefusalError, match="store_namespace_missing"):
        load_owner_decision(unmarked, artifact_id)
    # (c) defense in depth: a research-LOOKING unmarked path refuses the scope
    looks_research = tmp_path / "data" / "ifvg_datasets" / "search" / "v1"
    persist_regime_protocol(looks_research, run.protocol)
    persist_regime_assessment(looks_research, run.assessment)
    with pytest.raises(PermissionError, match="defense in depth"):
        persist_regime_promotion(looks_research, first, run_scope="synthetic_fixture")
    # (d) the verification namespace (a TEST class under search_test) admits both
    verification = tmp_path / "data" / "ifvg_datasets" / "search_test" / "v1"
    initialize_test_namespace(verification)
    persist_regime_protocol(verification, run.protocol)
    persist_regime_assessment(verification, run.assessment)
    persist_regime_promotion(verification, first, run_scope="synthetic_fixture")
    synthetic_owner_decision_fixture(verification, protocol=run.protocol, assessment=run.assessment)
