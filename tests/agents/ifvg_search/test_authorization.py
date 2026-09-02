"""Computation-path-scoped authorization suites (TEST_MATRIX §3.8, P0-E)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    VERIFICATION_FIXTURE_DECISION_KEY,
    AuthorizationError,
    OwnerAuthorizationBundle,
    OwnerDecisionEvidenceRef,
    SyntheticAuthorizationMarker,
    derive_authorization_requirements,
    validate_owner_authorization,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    initialize_test_namespace,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import publish_supersession
from alpha_lab.agents.data_infra.ifvg.study.computation_path import ComputationPath
from tests.agents.ifvg_search.namespace_fixture import (
    dummy_witness,
    owner_authorization_bundle,
)


def _path(**overrides) -> ComputationPath:
    base = dict(
        full_strategy_replay=False,
        feature_materialization=False,
        label_recomputation=False,
        model_refit=False,
        model_gated_sequential_replay=False,
        cost_recomputation=False,
        prop_resimulation=False,
        bootstrap_resimulation=False,
        reuse_trade_stream_hash=True,
    )
    base.update(overrides)
    return ComputationPath(**base)


_STRATEGY = ("strategy_profile.parent_retest_timeout_1m_bars",)
_PROP = ("prop_contract.topstep_50k", "risk_policy.fixed_dollar")
_REGIME_FEATURE = ("regime.feature_eligibility",)
_REGIME_DESCRIPTIVE = ("regime.descriptive_occupancy",)


def _keys(envelope) -> set[str]:
    return {req.decision_key for req in envelope.payload.requirements}


@pytest.mark.parametrize(
    ("scope", "dimensions", "path", "stages", "firms", "expected"),
    [
        # 1 — synthetic fixture: nothing (the typed marker gates it instead)
        ("synthetic_fixture", _STRATEGY, _path(full_strategy_replay=True), (), (), set()),
        # 2 — verification slice: exactly the fixture authorization
        (
            "verification_5d",
            (),
            _path(),
            (),
            (),
            {VERIFICATION_FIXTURE_DECISION_KEY},
        ),
        # 3 — strategy-only real search: axes/values/gates/workflow, NO prop/regime
        (
            "full_authorized_development",
            _STRATEGY,
            _path(full_strategy_replay=True),
            (),
            (),
            {
                "1:first_search_axes",
                "2:axis_values",
                "7:strategy_gate_thresholds",
                "R-2:authorization_workflow",
            },
        ),
        # 4 — prop benchmark on a frozen stream: prop set only, no strategy axes
        (
            "full_authorized_development",
            _PROP,
            _path(prop_resimulation=True, bootstrap_resimulation=True),
            (),
            ("topstep_50k",),
            {
                "5:firms_in_v1",
                "6:contract_evidence_compiler",
                "8:prop_gate_thresholds",
                "R-3:withdrawal_policies",
                "R-4:path_capability_matrix",
                "10:clock_policy",
            },
        ),
        # 5 — strategy search + prop realization: the union
        (
            "full_authorized_development",
            (*_STRATEGY, *_PROP),
            _path(
                full_strategy_replay=True,
                prop_resimulation=True,
                bootstrap_resimulation=True,
            ),
            (),
            ("topstep_50k",),
            {
                "1:first_search_axes",
                "2:axis_values",
                "7:strategy_gate_thresholds",
                "R-2:authorization_workflow",
                "5:firms_in_v1",
                "6:contract_evidence_compiler",
                "8:prop_gate_thresholds",
                "R-3:withdrawal_policies",
                "R-4:path_capability_matrix",
                "10:clock_policy",
            },
        ),
        # 6 — regime-descriptive study: NO RejectedCandidatePolicy, no regime gates
        (
            "full_authorized_development",
            _REGIME_DESCRIPTIVE,
            _path(model_refit=True),
            (),
            (),
            set(),
        ),
        # 7 — regime feature-eligibility: grain/count/stability decisions
        (
            "full_authorized_development",
            _REGIME_FEATURE,
            _path(feature_materialization=True, model_refit=True),
            (),
            (),
            {"28:regime_grain", "29:cluster_counts", "30:occupancy_stability_gates"},
        ),
        # 8 — model-gated replay: R-1 (via the computation path)
        (
            "full_authorized_development",
            (),
            _path(model_gated_sequential_replay=True),
            (),
            (),
            {"R-1:rejected_candidate_policy"},
        ),
        # 9 — model-gated replay requested via the stage plan alone
        (
            "full_authorized_development",
            (),
            _path(),
            ("11_run_frozen_model_gated_replays",),
            (),
            {"R-1:rejected_candidate_policy"},
        ),
        # 10 — cost-only recompute: no decisions at all
        (
            "full_authorized_development",
            ("cost_policy.round_turn_cost",),
            _path(cost_recomputation=True),
            (),
            (),
            set(),
        ),
    ],
)
def test_requirements_match_the_actual_computation_path(
    scope, dimensions, path, stages, firms, expected
) -> None:
    envelope = derive_authorization_requirements(scope, dimensions, path, stages, firms)
    assert _keys(envelope) == expected


def _ref(
    key: str, *, effective="2026-08-01T00:00:00Z", artifact="a" * 8
) -> OwnerDecisionEvidenceRef:
    return OwnerDecisionEvidenceRef(
        decision_id=key,
        decision_artifact_id=artifact,
        content_hash="c" * 64,
        author="owner",
        approved_at="2026-08-01T00:00:00Z",
        effective_from=effective,
        reviewed_evidence_refs=(),
    )


def test_owner_authorization_fails_closed(tmp_path) -> None:
    envelope = derive_authorization_requirements(
        "full_authorized_development",
        _STRATEGY,
        _path(full_strategy_replay=True),
        (),
        (),
    )
    root = tmp_path / "store"
    complete = owner_authorization_bundle(
        root,
        requirement_set_id=envelope.requirement_set_id,
        decision_refs={key: _ref(key) for key in _keys(envelope)},
    )
    validate_owner_authorization(complete, envelope, as_of_utc="2026-08-18T00:00:00Z")
    validate_owner_authorization(
        complete, envelope, as_of_utc="2026-08-18T00:00:00Z", store_root=root
    )

    # absent evidence
    partial = owner_authorization_bundle(
        root,
        requirement_set_id=envelope.requirement_set_id,
        decision_refs={"1:first_search_axes": _ref("1:first_search_axes")},
    )
    with pytest.raises(AuthorizationError, match="missing evidence"):
        validate_owner_authorization(partial, envelope, as_of_utc="2026-08-18T00:00:00Z")

    # superseded evidence
    with pytest.raises(AuthorizationError, match="superseded"):
        validate_owner_authorization(
            complete,
            envelope,
            as_of_utc="2026-08-18T00:00:00Z",
            superseded_artifact_ids=("a" * 8,),
        )

    # stale (not yet effective)
    future = owner_authorization_bundle(
        root,
        requirement_set_id=envelope.requirement_set_id,
        decision_refs={
            key: _ref(key, effective="2027-01-01T00:00:00Z") for key in _keys(envelope)
        },
    )
    with pytest.raises(AuthorizationError, match="not yet effective"):
        validate_owner_authorization(future, envelope, as_of_utc="2026-08-18T00:00:00Z")

    # requirement-set mismatch (charter-inconsistent)
    with pytest.raises(AuthorizationError, match="different requirement set"):
        validate_owner_authorization(
            complete.model_copy(update={"requirement_set_id": "9" * 64}),
            envelope,
            as_of_utc="2026-08-18T00:00:00Z",
        )


def test_synthetic_marker_is_typed() -> None:
    marker = SyntheticAuthorizationMarker()
    assert marker.kind == "synthetic_test_authorization_v1"


def test_bundle_binds_the_namespace_and_the_current_supersession_head(tmp_path) -> None:
    """HARDENING-BACKEND §4.1 / §4.2: the bundle names the store's verified
    namespace and the head it was signed against; an unmarked store, another
    namespace, a rolled-back (shorter) head and a moved (different) head all
    refuse; the witness must name the bundle's own namespace."""

    envelope = derive_authorization_requirements(
        "full_authorized_development", _STRATEGY, _path(full_strategy_replay=True), (), ()
    )
    refs = {key: _ref(key) for key in _keys(envelope)}
    root = tmp_path / "store"
    bundle = owner_authorization_bundle(
        root, requirement_set_id=envelope.requirement_set_id, decision_refs=refs
    )
    validate_owner_authorization(bundle, envelope, as_of_utc="2026-08-18T00:00:00Z",
                                 store_root=root)
    # a witness of another namespace cannot be paired with the bundle
    with pytest.raises(ValueError, match="another store namespace"):
        OwnerAuthorizationBundle(
            requirement_set_id=envelope.requirement_set_id,
            decision_refs=refs,
            store_namespace_id="1" * 64,
            supersession_head_witness=dummy_witness("2" * 64),
        )
    # an unmarked store
    with pytest.raises(AuthorizationError) as unmarked:
        validate_owner_authorization(
            bundle, envelope, as_of_utc="2026-08-18T00:00:00Z", store_root=tmp_path / "none"
        )
    assert unmarked.value.reason == "store_namespace_missing"
    # another (marked) store
    other = tmp_path / "other"
    initialize_test_namespace(other)
    with pytest.raises(AuthorizationError) as foreign:
        validate_owner_authorization(
            bundle, envelope, as_of_utc="2026-08-18T00:00:00Z", store_root=other
        )
    assert foreign.value.reason == "store_namespace_identity_mismatch"
    # the head moved after signing: a different current head refuses
    publish_supersession(
        root,
        superseded_decision_id="a" * 64,
        replacement_decision_id="b" * 64,
        reason="moved after signing",
        effective_at="2026-08-18T01:00:00+00:00",
        owner_evidence_ref="b" * 64,
    )
    with pytest.raises(AuthorizationError) as moved:
        validate_owner_authorization(
            bundle, envelope, as_of_utc="2026-08-18T00:00:00Z", store_root=root
        )
    assert moved.value.reason == "supersession_head_witness_mismatch"
    # re-signed against the current head: accepted; a rollback then refuses
    resigned = owner_authorization_bundle(
        root, requirement_set_id=envelope.requirement_set_id, decision_refs=refs
    )
    validate_owner_authorization(resigned, envelope, as_of_utc="2026-08-18T00:00:00Z",
                                 store_root=root)
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
        load_store_namespace,
        write_supersession_head_atomic,
    )

    namespace = load_store_namespace(root)
    write_supersession_head_atomic(
        root,
        store_namespace_id=namespace.store_namespace_id,
        record_id=None,
        line_count=0,
        head_sha256=namespace.payload.authority_genesis_id,
    )
    with pytest.raises(AuthorizationError) as rolled_back:
        validate_owner_authorization(
            resigned, envelope, as_of_utc="2026-08-18T00:00:00Z", store_root=root
        )
    assert rolled_back.value.reason == "supersession_head_shorter_than_witness"
