"""R6.1 workstream G (S09c + D9) — the supervised regime studies.

``feature_only`` (the controlled Baseline vs Baseline+Regime study) runs
every rung of the bundle ladder (prevalence + logistic + the CatBoost bundle
rung) on both arms and shares rows/labels/folds on ``comparison_row_id``:
the two arms are DIFFERENT resolved bundles / ladders over ONE static view
(the base bundle's frame — DEV-R6.1-7), and the cross-arm identity check
also pairs a challenger carrying a DIFFERENT ``view_id`` (only
``comparison_row_id`` pairs the arms; the view-scoped ``oos_row_id`` never
does); a planted radial signal that only the fit-local distance features
carry makes the challenger's paired Brier delta negative; a fold-schedule
mismatch refuses; below FEATURE_ELIGIBLE refuses with the exact text; the
status-gated activation changes the registry hash and refuses every
mismatch naming the frozen decision; ``cohort_model`` pairs pooled and
specialized rows and types thin / single-class strata; the study payloads
are deep-immutable and identity-stable (F15); the descriptive OOS artifact
loader is never consulted by any S09c path.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (
    build_bundle_feature_view,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
    FEATURE_BLOCK_REGISTRY,
    FEATURE_BLOCK_RESOLUTION_REGISTRY,
    BlockUnavailableError,
    FeatureBlockStatus,
    feature_block_registry_hash,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import (
    FEATURE_BUNDLE_REGISTRY,
    FeatureBundleDefinition,
    resolve_bundle,
)
from alpha_lab.agents.data_infra.ifvg.fold_schedules import derive_fold_schedule
from alpha_lab.agents.data_infra.ifvg.ml import regime_oos_assignment as oos_module
from alpha_lab.agents.data_infra.ifvg.ml.comparison_rows import (
    COMPARISON_ROW_IDENTITY_KEY,
    label_artifact_content_id,
)
from alpha_lab.agents.data_infra.ifvg.ml.fold_set_artifact import (
    build_fold_set_artifact,
    persist_fold_schedule,
    persist_fold_set_artifact,
)
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import (
    CATBOOST_BUNDLE_PROTOCOL_ID,
    LOGISTIC_PROTOCOL_ID,
    PREVALENCE_PROTOCOL_ID,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_block_activation import (
    REGIME_ACTIVATION_AS_OF_POLICY,
    REGIME_BLOCK_KEY,
    RegimeActivationRefusalError,
    activate_regime_context_block,
    regime_activation_resolution_payload,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_cohort_model import (
    REGIME_COHORT_MODEL_STORE,
    RegimeCohortStratumRecord,
    load_regime_cohort_model_study,
    load_regime_cohort_model_study_detail,
    run_regime_cohort_model_study,
    save_regime_cohort_model_study,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    ObservationGranularity,
    RegimePromotionDecision,
    RegimePromotionDecisionEnvelope,
    RegimeRole,
    RegimeStatus,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_controlled_study import (
    FEATURE_ELIGIBLE_REFUSAL,
    REGIME_CONTROLLED_STUDY_STORE,
    FrozenTree,
    RegimeControlledStudyPayload,
    RegimeSupervisedStudyRefusalError,
    assert_regime_cross_arm_identity,
    load_regime_controlled_study,
    load_regime_controlled_study_detail,
    run_controlled_regime_study,
    save_regime_controlled_study,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_fold_features import (
    RegimeFoldFeatureFrameSource,
    build_regime_fold_features,
    load_regime_fold_feature_source,
    save_regime_fold_features,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    persist_regime_assessment,
    persist_regime_fit,
    persist_regime_promotion,
    persist_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.supervised_ladder import (
    DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    run_supervised_ladder,
)
from alpha_lab.agents.data_infra.ifvg.search import store as store_module
from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (
    synthetic_owner_decision_fixture,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
    CLUSTER_CENTERS,
    REGIME_INPUT_FEATURES,
    known_cluster_fixture,
)

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id
_DECIDED_AT = "2026-08-28T12:00:00+00:00"
#: the controlled regime study runs the WHOLE bundle ladder on both arms
#: (plan §6.G: prevalence + logistic + the CatBoost bundle rung)
_LABEL_POLICY = "synthetic_fixture_labels_v1"
_STUDY_PROTOCOLS = DEFAULT_BUNDLE_LADDER_PROTOCOLS
#: the cohort study fits one ladder per fold × stratum; the two linear rungs
#: keep the unit test bounded (the bundle rung is proven on the pooled arm above)
_COHORT_PROTOCOLS = (PREVALENCE_PROTOCOL_ID, LOGISTIC_PROTOCOL_ID)


def _b7_registry() -> dict:
    """The regime-bearing bundle the hub will register (B0 + regime)."""

    registry = dict(FEATURE_BUNDLE_REGISTRY)
    registry["B7_CORE_REGIME"] = FeatureBundleDefinition(
        feature_bundle_key="B7_CORE_REGIME",
        bundle_version=1,
        human_name="Core + fold-local regime features (status-gated)",
        base_bundle_key="B0_CORE",
        included_block_keys=(REGIME_BLOCK_KEY,),
        required_coverage_gates={},
    )
    return registry


def _planted_labels(fixture) -> pd.DataFrame:
    """A RADIAL signal: inside each cluster the label is 1 for the inner half
    (raw distance to the true center below the cluster median), 0 for the
    outer half — a linear model on the two raw coordinates cannot see it,
    the fit-local assigned distance carries it. Cluster 2 is constant 0 so a
    single-class stratum exists for the cohort study."""

    frame = fixture.view.frame.set_index("candidate_id")
    memberships = pd.Series(fixture.true_memberships)
    centers = np.asarray(CLUSTER_CENTERS[:3], dtype=float)
    points = frame.loc[memberships.index, list(REGIME_INPUT_FEATURES)].to_numpy(dtype=float)
    distance = np.linalg.norm(points - centers[memberships.to_numpy()], axis=1)
    labels = fixture.labeled_candidates.copy()
    target = np.zeros(len(labels), dtype=int)
    for cluster in (0, 1):
        mask = (memberships.to_numpy() == cluster)
        median = np.median(distance[mask])
        inner = mask & (distance < median)
        target[np.isin(labels["candidate_id"].to_numpy(), memberships.index[inner])] = 1
    labels["binary_target"] = target
    labels["gross_r"] = np.where(target == 1, 1.0, -1.0)
    labels["net_r"] = labels["gross_r"]
    return labels


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


def _verified_fits(root, run, observation_frame) -> dict:
    """R6.1-FIX §3.2: persist the run's fits and exact-load their VERIFIED
    assignment evidence — the only fold-feature assignment source."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
        load_regime_fit_assignments,
    )

    for fold_fit in run.fold_fits:
        persist_regime_fit(
            root,
            fold_fit,
            run.assignments[run.assignments["fold_index"] == fold_fit.fold_index],
            observation_frame=observation_frame,
        )
    return {
        fit.fold_index: load_regime_fit_assignments(root, fit.fit_envelope.regime_fit_id)
        for fit in run.fold_fits
    }


@pytest.fixture(scope="module")
def lane(tmp_path_factory, request):
    # the two study stores are hub registrations (search/store.py); until the
    # hub adds them the module extends the registered names for this module
    patcher = pytest.MonkeyPatch()
    request.addfinalizer(patcher.undo)
    pending = tuple(
        name
        for name in (REGIME_CONTROLLED_STUDY_STORE, REGIME_COHORT_MODEL_STORE)
        if name not in store_module.SEARCH_STORE_NAMES
    )
    if pending:
        patcher.setattr(
            store_module, "SEARCH_STORE_NAMES", (*store_module.SEARCH_STORE_NAMES, *pending)
        )
    root = tmp_path_factory.mktemp("supervised_regime")
    fixture = known_cluster_fixture(k=3, n=600)
    labels = _planted_labels(fixture)
    folds = build_context_folds(labels, authorized_trading_days=fixture.trading_days)
    assert any(fold.valid for fold in folds.folds)
    schedule = derive_fold_schedule(fixture.trading_days)
    persist_fold_schedule(root, schedule)
    fold_set, definitions = build_fold_set_artifact(
        folds,
        schedule=schedule,
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id=fixture.view.view_id,
        minimum_train_observations=30,
        labeled=True,
    )
    persist_fold_set_artifact(root, fold_set, definitions)
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0, resolved_input_features=fixture.regime_input_features
    )
    run = run_regime_protocol(
        fixture.view.frame,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=3,
    )
    assert run.assessment.payload.gates_passed
    persist_regime_protocol(root, protocol)
    for fold_fit in run.fold_fits:
        persist_regime_fit(
            root,
            fold_fit,
            run.assignments[run.assignments["fold_index"] == fold_fit.fold_index],
            observation_frame=fixture.view.frame,
        )
    persist_regime_assessment(root, run.assessment)
    envelope, frame = build_regime_fold_features(
        protocol=protocol,
        regime_run=run,
        fit_assignments=_verified_fits(root, run, fixture.view.frame),
        candidate_fold_set=fold_set,
        candidate_folds=folds,
        regime_fold_set=fold_set,
        schedule=schedule,
        candidate_view_frame=fixture.view.frame,
        candidate_view_id=fixture.view.view_id,
    )
    save_regime_fold_features(root, envelope, frame)
    source = load_regime_fold_feature_source(root, envelope.regime_fold_feature_artifact_id)
    # the lawful chain: descriptive → stratification_ready → FEATURE_ELIGIBLE
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
    owner = synthetic_owner_decision_fixture(root, protocol=protocol, assessment=run.assessment)
    eligible = _decision(
        run,
        role=RegimeRole.FEATURE_GENERATOR,
        status=RegimeStatus.FEATURE_ELIGIBLE,
        previous_status=RegimeStatus.STRATIFICATION_READY,
        previous_decision_ref=ready.regime_promotion_decision_id,
        owner_ratification_ref=owner.owner_decision_artifact_id,
    )
    persist_regime_promotion(root, eligible, run_scope="synthetic_fixture")
    activation = activate_regime_context_block(
        root,
        resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
        regime_fold_feature_artifact_id=envelope.regime_fold_feature_artifact_id,
        regime_promotion_decision_id=eligible.regime_promotion_decision_id,
        owner_decision_artifact_id=owner.owner_decision_artifact_id,
        run_scope="synthetic_fixture",
    )
    return {
        "root": root,
        "fixture": fixture,
        "labels": labels,
        "folds": folds,
        "schedule": schedule,
        "fold_set": fold_set,
        "protocol": protocol,
        "run": run,
        "fold_features": envelope,
        "source": source,
        "ready": ready,
        "owner": owner,
        "eligible": eligible,
        "activation": activation,
        # review RB-01 (§7.2): the exact, policy-bearing label artifact id + its policy —
        # the persisting study seams prove the pair before any run can be saved
        "label_artifact_id": label_artifact_content_id(_LABEL_POLICY, labels),
        "label_policy_id": _LABEL_POLICY,
    }


@pytest.fixture(autouse=True)
def _descriptive_loader_never_consulted(monkeypatch):
    def _never(*args, **kwargs):
        raise AssertionError("the S09c path must never open the descriptive OOS artifact")

    monkeypatch.setattr(oos_module, "load_regime_oos_assignment", _never)
    monkeypatch.setattr(oos_module, "load_regime_oos_assignment_frame", _never)


# ── D9: status-gated, versioned activation ───────────────────────────────────


def test_activation_is_a_pure_versioned_event_that_binds_the_frozen_authority(lane):
    activation = lane["activation"]
    envelope = activation.envelope
    payload = envelope.payload
    assert payload.feature_block_key == REGIME_BLOCK_KEY
    assert payload.block_version == FEATURE_BLOCK_REGISTRY[REGIME_BLOCK_KEY].block_version + 1
    assert payload.as_of_policy == REGIME_ACTIVATION_AS_OF_POLICY
    assert payload.join_keys == ("candidate_id", "fold_index")
    assert payload.feature_names == lane["fold_features"].payload.model_feature_names
    assert "canonical_reporting_cluster_id" not in payload.feature_names
    assert payload.source_artifact_refs == (
        lane["protocol"].resolved_regime_protocol_id,
        lane["fold_features"].regime_fold_feature_artifact_id,
        lane["run"].assessment.regime_capability_assessment_id,
        lane["eligible"].regime_promotion_decision_id,
        lane["owner"].owner_decision_artifact_id,
    )
    # the module registries are untouched: the block stays PLANNED at import
    assert FEATURE_BLOCK_REGISTRY[REGIME_BLOCK_KEY].status is FeatureBlockStatus.PLANNED
    assert REGIME_BLOCK_KEY not in FEATURE_BLOCK_RESOLUTION_REGISTRY
    assert activation.definitions[REGIME_BLOCK_KEY].status is FeatureBlockStatus.AVAILABLE
    assert activation.registry_hash != feature_block_registry_hash()
    # the regime-bearing bundle resolves ONLY under the activated mappings
    registry = _b7_registry()
    with pytest.raises(BlockUnavailableError):
        resolve_bundle("B7_CORE_REGIME", bundle_registry=registry)
    b7 = resolve_bundle(
        "B7_CORE_REGIME",
        bundle_registry=registry,
        definitions=activation.definitions,
        resolutions=activation.resolutions,
    )
    assert envelope.resolved_feature_block_id in b7.payload.resolved_block_ids
    assert set(payload.feature_names) <= set(b7.payload.resolved_feature_names)
    # replaying the payload builder reproduces the minted resolution exactly
    replay = regime_activation_resolution_payload(
        protocol=activation.protocol,
        fold_feature_artifact=activation.fold_feature_artifact,
        promotion_decision=activation.promotion_decision,
        owner_decision=activation.owner_decision,
    )
    assert replay.model_dump(mode="json") == payload.model_dump(mode="json")


def test_activation_refuses_every_mismatch_naming_the_frozen_decision(lane, tmp_path):
    root, run, protocol = lane["root"], lane["run"], lane["protocol"]
    eligible, owner = lane["eligible"], lane["owner"]
    fold_feature_id = lane["fold_features"].regime_fold_feature_artifact_id
    frozen = eligible.regime_promotion_decision_id[:12]

    def _activate(**overrides):
        kwargs = dict(
            resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
            regime_fold_feature_artifact_id=fold_feature_id,
            regime_promotion_decision_id=eligible.regime_promotion_decision_id,
            owner_decision_artifact_id=owner.owner_decision_artifact_id,
            run_scope="synthetic_fixture",
        )
        kwargs.update(overrides)
        return activate_regime_context_block(root, **kwargs)

    # below FEATURE_ELIGIBLE: the STRATIFICATION_READY decision cannot activate
    with pytest.raises(RegimeActivationRefusalError, match="below FEATURE_ELIGIBLE") as info:
        _activate(regime_promotion_decision_id=lane["ready"].regime_promotion_decision_id)
    assert lane["ready"].regime_promotion_decision_id[:12] in str(info.value)
    # a different owner artifact than the one that ratified the frozen decision
    other_owner = synthetic_owner_decision_fixture(
        root,
        protocol=protocol,
        assessment=run.assessment,
        approved_at="2026-08-27T00:00:00+00:00",
        effective_from="2026-08-27T00:00:00+00:00",
    )
    with pytest.raises(RegimeActivationRefusalError, match="different owner-decision") as info:
        _activate(owner_decision_artifact_id=other_owner.owner_decision_artifact_id)
    assert frozen in str(info.value)
    # a different protocol than the frozen decision names
    other_protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0,
        resolved_input_features=lane["fixture"].regime_input_features,
        winsorization_policy="clip_p01_p99_train_fitted_v1",
    )
    persist_regime_protocol(root, other_protocol)
    with pytest.raises(RegimeActivationRefusalError, match="different regime protocol") as info:
        _activate(resolved_regime_protocol_id=other_protocol.resolved_regime_protocol_id)
    assert frozen in str(info.value)
    # synthetic provenance outside the synthetic scope (the owner check)
    with pytest.raises(RegimeActivationRefusalError, match="synthetic_fixture run scope"):
        _activate(run_scope="verification_5d")
    # an unverified store entry
    with pytest.raises(RegimeActivationRefusalError, match="not a verified store entry"):
        _activate(regime_fold_feature_artifact_id="9" * 64)
    # a fold-feature artifact of another protocol (built + saved for that protocol)
    other_run = run_regime_protocol(
        lane["fixture"].view.frame,
        lane["folds"],
        other_protocol,
        source_artifact_ids=(lane["fixture"].view.view_id,),
        bootstrap_refits=2,
    )
    other_envelope, other_frame = build_regime_fold_features(
        protocol=other_protocol,
        regime_run=other_run,
        fit_assignments=_verified_fits(root, other_run, lane["fixture"].view.frame),
        candidate_fold_set=lane["fold_set"],
        candidate_folds=lane["folds"],
        regime_fold_set=lane["fold_set"],
        schedule=lane["schedule"],
        candidate_view_frame=lane["fixture"].view.frame,
        candidate_view_id=lane["fixture"].view.view_id,
    )
    save_regime_fold_features(root, other_envelope, other_frame)
    with pytest.raises(RegimeActivationRefusalError, match="belongs to another protocol") as info:
        _activate(regime_fold_feature_artifact_id=other_envelope.regime_fold_feature_artifact_id)
    assert frozen in str(info.value)


# ── feature_only: the controlled Baseline vs Baseline+Regime study ──────────


@pytest.fixture(scope="module")
def controlled(lane):
    return run_controlled_regime_study(
        lane["fixture"].view,
        lane["labels"],
        lane["folds"],
        activation=lane["activation"],
        challenger_bundle_key="B7_CORE_REGIME",
        fold_features=lane["source"],
        candidate_fold_set=lane["fold_set"],
        label_artifact_id=lane["label_artifact_id"],
        label_policy_id=lane["label_policy_id"],
        bundle_registry=_b7_registry(),
        protocols=_STUDY_PROTOCOLS,
    )


def test_arms_share_rows_labels_folds_on_comparison_row_id_across_distinct_bundles(
    controlled, lane
):
    baseline, challenger = controlled.baseline, controlled.challenger
    # ONE static view (the base bundle's frame, DEV-R6.1-7); the arms differ by
    # resolved bundle and ladder — the distinct-VIEW pairing is proven below
    assert baseline.view_id == challenger.view_id
    assert baseline.ladder_id != challenger.ladder_id
    assert set(_STUDY_PROTOCOLS) == {
        PREVALENCE_PROTOCOL_ID,
        LOGISTIC_PROTOCOL_ID,
        CATBOOST_BUNDLE_PROTOCOL_ID,
    }
    assert {rung.protocol_id for rung in challenger.rungs} == set(_STUDY_PROTOCOLS)
    assert controlled.envelope.payload.model_protocol_ids == tuple(_STUDY_PROTOCOLS)
    assert baseline.feature_source["resolved_feature_bundle_id"] != (
        challenger.feature_source["resolved_feature_bundle_id"]
    )
    assert challenger.feature_source["evidence_ref"] == lane["source"].artifact_id
    regime_names = set(lane["source"].feature_names)
    assert regime_names <= set(challenger.feature_source["feature_names"])
    assert not regime_names & set(baseline.feature_source["feature_names"])
    for protocol_id in _STUDY_PROTOCOLS:
        left = baseline.rung(protocol_id).predictions
        right = challenger.rung(protocol_id).predictions
        assert not left.empty
        assert set(left[COMPARISON_ROW_IDENTITY_KEY]) == set(right[COMPARISON_ROW_IDENTITY_KEY])
    payload = controlled.envelope.payload
    assert payload.comparison_class == "feature_only"
    assert payload.parity_status == "held" and payload.oos_row_count > 0
    assert payload.regime_promotion_decision_id == lane["eligible"].regime_promotion_decision_id
    assert payload.owner_decision_artifact_id == lane["owner"].owner_decision_artifact_id
    assert payload.resolved_regime_block_id == lane["activation"].resolved_feature_block_id
    assert payload.fold_set_hash == lane["fold_set"].payload.fold_set_id
    assert payload.research_boundary == "research_only_offline"


def test_challenger_arm_with_a_distinct_view_id_pairs_only_on_comparison_row_id(lane):
    """The genuinely distinct-view case at the controlled-regime-study level:
    the challenger ladder is run over a view carrying ANOTHER ``view_id`` (the
    fold-local regime features joined per fold from the real artifact); the
    study's cross-arm identity check pairs the arms on ``comparison_row_id``
    while every view-scoped ``oos_row_id`` differs."""

    activation, source = lane["activation"], lane["source"]
    registry = _b7_registry()
    challenger_env = resolve_bundle(
        "B7_CORE_REGIME",
        bundle_registry=registry,
        definitions=activation.definitions,
        resolutions=activation.resolutions,
    )
    baseline_env = resolve_bundle(
        "B0_CORE",
        bundle_registry=registry,
        definitions=activation.definitions,
        resolutions=activation.resolutions,
    )
    _view_env, baseline_frame = build_bundle_feature_view(lane["fixture"].view, baseline_env)
    arm_view = replace(lane["fixture"].view, frame=baseline_frame)
    other_view = replace(arm_view, view_id="c" * 64)
    assert other_view.view_id != arm_view.view_id
    schedule_id = source.envelope.payload.fold_schedule_id
    common = dict(
        protocols=_STUDY_PROTOCOLS,
        fold_schedule_id=schedule_id,
        label_artifact_id=lane["label_artifact_id"],
    )
    baseline = run_supervised_ladder(
        arm_view,
        lane["labels"],
        lane["folds"],
        bundle_features=tuple(baseline_env.payload.resolved_feature_names),
        bundle_ref=baseline_env.resolved_feature_bundle_id,
        **common,
    )
    challenger = run_supervised_ladder(
        other_view,
        lane["labels"],
        lane["folds"],
        bundle_features=tuple(challenger_env.payload.resolved_feature_names),
        bundle_ref=challenger_env.resolved_feature_bundle_id,
        bundle_evidence_ref=source.artifact_id,
        bundle_categorical_features=tuple(activation.envelope.payload.categorical_features),
        fold_local_features=source,
        **common,
    )
    assert baseline.view_id != challenger.view_id
    assert_regime_cross_arm_identity(baseline, challenger)
    for protocol_id in _STUDY_PROTOCOLS:
        left = baseline.rung(protocol_id).predictions
        right = challenger.rung(protocol_id).predictions
        assert not left.empty and not right.empty
        assert set(left[COMPARISON_ROW_IDENTITY_KEY]) == set(right[COMPARISON_ROW_IDENTITY_KEY])
        assert set(left["oos_row_id"]).isdisjoint(set(right["oos_row_id"]))
    # a challenger over a different fold set is refused by the same check
    other_days = lane["fixture"].trading_days[:-5]
    other_folds = build_context_folds(
        lane["labels"][lane["labels"]["trading_day"].isin(other_days)],
        authorized_trading_days=other_days,
    )
    narrower = run_supervised_ladder(
        other_view,
        lane["labels"],
        other_folds,
        bundle_features=tuple(baseline_env.payload.resolved_feature_names),
        bundle_ref=baseline_env.resolved_feature_bundle_id,
        **common,
    )
    with pytest.raises(ValueError, match="disagree on the exact OOS rows"):
        assert_regime_cross_arm_identity(baseline, narrower)


def test_planted_signal_in_the_fold_features_lowers_the_challenger_brier(controlled):
    delta = controlled.envelope.payload.paired_deltas[LOGISTIC_PROTOCOL_ID]["brier"]
    assert delta["available"] is True
    assert delta["estimate"] < 0, delta
    assert delta["upper"] < 0, delta
    summary = controlled.envelope.payload.challenger_summary["rungs"][LOGISTIC_PROTOCOL_ID]
    baseline = controlled.envelope.payload.baseline_summary["rungs"][LOGISTIC_PROTOCOL_ID]
    assert summary["brier_score"] < baseline["brier_score"]
    log_loss = controlled.envelope.payload.paired_deltas[LOGISTIC_PROTOCOL_ID]["log_loss"]
    assert log_loss["estimate"] < 0


def test_controlled_study_persists_and_reloads(controlled, lane):
    root = lane["root"]
    # review RB-01 (§7.2): only a run whose label identity was PROVEN exact persists
    assert controlled.label_identity_proof == "exact"
    with pytest.raises(PermissionError, match="not proven exact"):
        save_regime_controlled_study(
            root, replace(controlled, label_identity_proof="caller_supplied_unproven")
        )
    save_regime_controlled_study(root, controlled)
    save_regime_controlled_study(root, controlled)  # verified reuse
    reloaded = load_regime_controlled_study(root, controlled.envelope.regime_controlled_study_id)
    assert reloaded.model_dump(mode="json") == controlled.envelope.model_dump(mode="json")
    detail = load_regime_controlled_study_detail(root, reloaded)
    assert set(detail["challenger"]["rungs"]) == set(_STUDY_PROTOCOLS)
    assert detail["regime_block_resolution"]["payload"]["feature_block_key"] == REGIME_BLOCK_KEY
    # the CatBoost bundle rung ran on BOTH arms and carries a paired delta
    bundle_delta = controlled.envelope.payload.paired_deltas[CATBOOST_BUNDLE_PROTOCOL_ID]
    assert bundle_delta is not None and set(bundle_delta) == {"brier", "log_loss"}
    assert set(detail["baseline"]["rungs"]) == set(_STUDY_PROTOCOLS)


# ── F15: deep-immutable, identity-stable study payloads ─────────────────────


def test_study_payload_values_are_deep_immutable_and_identity_stable(controlled, cohort):
    payload = controlled.envelope.payload
    deltas = payload.paired_deltas[LOGISTIC_PROTOCOL_ID]
    assert isinstance(deltas, FrozenTree) and isinstance(deltas["brier"], FrozenTree)
    with pytest.raises(TypeError):
        deltas["brier"]["estimate"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        payload.baseline_summary["rungs"][LOGISTIC_PROTOCOL_ID]["brier_score"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        payload.paired_deltas["forged"] = {}  # type: ignore[index]
    # the serialized form is the plain JSON object tree (dicts, lists) — the
    # canonical bytes and therefore the study id are those of the plain form
    dumped = payload.model_dump(mode="json")
    assert all(
        isinstance(value, dict) for _key, value in dumped["paired_deltas"] if value is not None
    )
    assert isinstance(dict(dumped["baseline_summary"])["rungs"], dict)
    reloaded = RegimeControlledStudyPayload.model_validate(dumped)
    assert canonical_contract_sha256(reloaded) == controlled.envelope.regime_controlled_study_id
    # cohort strata records freeze their paired deltas the same way
    modeled = next(r for r in cohort.envelope.payload.strata if r.status == "modeled")
    assert isinstance(modeled.paired_deltas[LOGISTIC_PROTOCOL_ID], FrozenTree)
    with pytest.raises(TypeError):
        modeled.paired_deltas[LOGISTIC_PROTOCOL_ID]["brier"] = None  # type: ignore[index]
    # GOLDEN: the ids of nested-summary payloads are exactly what the plain
    # nested-dict form minted before F15 (computed on the pre-F15 tree)
    nested = {
        "rungs": {
            "p1": {"brier_score": 0.25, "log_loss": 0.69, "rows": 10, "nan": None, "flag": True}
        },
        "list": [1, 2, {"a": [3, 4]}],
        "scalar": "x",
    }
    probe_deltas = {
        "p1": {"brier": {"estimate": -0.01, "available": True}, "log_loss": {"estimate": -0.02}}
    }
    from alpha_lab.agents.data_infra.ifvg.ml.regime_controlled_study import _example_payload

    base = _example_payload().model_dump(mode="json")
    base.update(baseline_summary=nested, challenger_summary=nested, paired_deltas=probe_deltas)
    probe = RegimeControlledStudyPayload(**base)
    assert canonical_contract_sha256(probe) == (
        "d7626a4263760b4ff8ef2e18014b1281d3a9860b6f3890f30e1e766a8977abdf"
    )
    assert probe.baseline_summary["list"] == (1, 2, {"a": (3, 4)})
    record = RegimeCohortStratumRecord(
        fold_index=0,
        local_id="0",
        regime_fit_id="a" * 64,
        train_rows=1,
        test_rows=1,
        status="modeled",
        reason=None,
        specialized_ladder_id="b" * 64,
        paired_deltas=probe_deltas,
        pooled_brier_on_stratum=0.2,
        specialized_brier=0.1,
    )
    assert canonical_contract_sha256(record) == (
        "6a2a12479706185c97c82ca0a8b9adcc1bb8adb15c18071c3e8fc480001003d8"
    )


def test_feature_only_refusals_below_status_schedule_mismatch_and_bundle_shape(lane):
    run = lane["run"]
    activation = lane["activation"]
    kwargs = dict(
        activation=activation,
        challenger_bundle_key="B7_CORE_REGIME",
        fold_features=lane["source"],
        candidate_fold_set=lane["fold_set"],
        label_artifact_id=lane["label_artifact_id"],
        label_policy_id=lane["label_policy_id"],
        bundle_registry=_b7_registry(),
        protocols=_STUDY_PROTOCOLS,
    )
    view, labels, folds = lane["fixture"].view, lane["labels"], lane["folds"]
    # below FEATURE_ELIGIBLE: an activation carrying the STRATIFICATION_READY decision
    below = activation.__class__(
        **{**activation.__dict__, "promotion_decision": lane["ready"]}
    )
    with pytest.raises(RegimeSupervisedStudyRefusalError) as info:
        run_controlled_regime_study(view, labels, folds, **{**kwargs, "activation": below})
    assert FEATURE_ELIGIBLE_REFUSAL in str(info.value)
    assert lane["ready"].regime_promotion_decision_id[:12] in str(info.value)
    # a fold-feature artifact the activation does not bind
    other = known_cluster_fixture(k=3, n=500)
    other_folds = build_context_folds(
        other.labeled_candidates, authorized_trading_days=other.trading_days
    )
    other_fold_set, _ = build_fold_set_artifact(
        other_folds,
        schedule=lane["schedule"],
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id=other.view.view_id,
        minimum_train_observations=30,
        labeled=True,
    )
    other_run = run_regime_protocol(
        other.view.frame,
        other_folds,
        lane["protocol"],
        source_artifact_ids=(other.view.view_id,),
        bootstrap_refits=2,
    )
    other_envelope, other_frame = build_regime_fold_features(
        protocol=lane["protocol"],
        regime_run=other_run,
        fit_assignments=_verified_fits(lane["root"], other_run, other.view.frame),
        candidate_fold_set=other_fold_set,
        candidate_folds=other_folds,
        regime_fold_set=other_fold_set,
        schedule=lane["schedule"],
        candidate_view_frame=other.view.frame,
        candidate_view_id=other.view.view_id,
    )
    other_source = RegimeFoldFeatureFrameSource(other_envelope, other_frame)
    with pytest.raises(RegimeSupervisedStudyRefusalError, match="binds fold-feature artifact"):
        run_controlled_regime_study(
            view, labels, folds, **{**kwargs, "fold_features": other_source}
        )
    # fold_schedule / fold-set mismatch: the study's folds are not the artifact's population
    with pytest.raises(ValueError, match="fold_schedule mismatch"):
        run_controlled_regime_study(view, labels, other_folds, **kwargs)
    with pytest.raises(ValueError, match="fold_schedule mismatch"):
        run_controlled_regime_study(
            view, labels, folds, **{**kwargs, "candidate_fold_set": other_fold_set}
        )
    # a challenger without the regime block, and a base that carries it
    with pytest.raises(ValueError, match="carries no regime block"):
        run_controlled_regime_study(
            view, labels, folds, **{**kwargs, "challenger_bundle_key": "B1_CORE_STRUCTURE"}
        )
    registry = _b7_registry()
    registry["B8_REGIME_TWICE"] = FeatureBundleDefinition(
        feature_bundle_key="B8_REGIME_TWICE",
        bundle_version=1,
        human_name="regime on regime (unlawful contrast)",
        base_bundle_key="B7_CORE_REGIME",
        included_block_keys=(REGIME_BLOCK_KEY,),
        required_coverage_gates={},
    )
    with pytest.raises(ValueError, match="base bundle also carries the regime block"):
        run_controlled_regime_study(
            view,
            labels,
            folds,
            **{**kwargs, "challenger_bundle_key": "B8_REGIME_TWICE", "bundle_registry": registry},
        )
    assert run.assessment.payload.gates_passed  # the fixture's authority is intact


# ── cohort_model: per-regime specialized vs pooled ──────────────────────────


@pytest.fixture(scope="module")
def cohort(lane):
    return run_regime_cohort_model_study(
        lane["fixture"].view,
        lane["labels"],
        lane["folds"],
        activation=lane["activation"],
        bundle_key="B0_CORE",
        fold_features=lane["source"],
        candidate_fold_set=lane["fold_set"],
        label_artifact_id=lane["label_artifact_id"],
        label_policy_id=lane["label_policy_id"],
        bundle_registry=_b7_registry(),
        protocols=_COHORT_PROTOCOLS,
    )


def test_cohort_model_pairs_rows_and_types_thin_and_single_class_strata(cohort, lane):
    payload = cohort.envelope.payload
    assert payload.comparison_class == "cohort_model"
    assert payload.minimum_training_rows == 60
    modeled = [record for record in payload.strata if record.status == "modeled"]
    typed = [record for record in payload.strata if record.status != "modeled"]
    assert modeled, "the fixture must model at least one stratum"
    assert {record.reason for record in typed} >= {"single_class_training"}
    assert all(record.reason is None for record in modeled)
    assert all(record.specialized_ladder_id for record in modeled)
    # every specialized OOS row pairs with a pooled row on comparison_row_id
    pooled = cohort.pooled.rung(LOGISTIC_PROTOCOL_ID).predictions
    specialized = cohort.specialized_predictions
    logistic_rows = specialized[specialized["model_protocol_id"] == LOGISTIC_PROTOCOL_ID]
    assert set(logistic_rows[COMPARISON_ROW_IDENTITY_KEY]) <= set(
        pooled[COMPARISON_ROW_IDENTITY_KEY]
    )
    assert not logistic_rows[COMPARISON_ROW_IDENTITY_KEY].duplicated().any()
    assert payload.specialized_oos_row_count == logistic_rows[COMPARISON_ROW_IDENTITY_KEY].nunique()
    assert payload.pooled_oos_row_count == pooled[COMPARISON_ROW_IDENTITY_KEY].nunique()
    assert 0.0 < payload.specialized_coverage_fraction <= 1.0
    for record in modeled:
        assert record.train_rows >= 60 and record.test_rows > 0
        assert record.paired_deltas is not None
        assert record.pooled_brier_on_stratum is not None
        assert record.specialized_brier is not None
        assert record.regime_fit_id in {
            ref.regime_fit_id for ref in lane["fold_features"].payload.regime_fit_ids_by_fold
        }
    # the stratum row counts come from the fold-feature artifact's local ids
    fold = next(fold for fold in lane["folds"].folds if fold.valid)
    ids = lane["source"].local_ids_for_fold(fold.fold_index)
    for record in payload.strata:
        if record.fold_index != fold.fold_index:
            continue
        stratum = ids[ids["local_id"] == record.local_id]
        assert record.train_rows <= int((stratum["partition"] == "train").sum())
        assert record.test_rows == int((stratum["partition"] == "test").sum())


def test_cohort_model_floor_refusals_persistence_and_bundle_shape(cohort, lane):
    root = lane["root"]
    # review RB-01 (§7.2): only a run whose label identity was PROVEN exact persists
    assert cohort.label_identity_proof == "exact"
    with pytest.raises(PermissionError, match="not proven exact"):
        save_regime_cohort_model_study(
            root, replace(cohort, label_identity_proof="caller_supplied_unproven")
        )
    save_regime_cohort_model_study(root, cohort)
    reloaded = load_regime_cohort_model_study(root, cohort.envelope.regime_cohort_model_study_id)
    assert reloaded.model_dump(mode="json") == cohort.envelope.model_dump(mode="json")
    detail = load_regime_cohort_model_study_detail(root, reloaded)
    assert set(detail) == {"pooled", "strata"}
    kwargs = dict(
        activation=lane["activation"],
        bundle_key="B0_CORE",
        fold_features=lane["source"],
        candidate_fold_set=lane["fold_set"],
        label_artifact_id=lane["label_artifact_id"],
        label_policy_id=lane["label_policy_id"],
        bundle_registry=_b7_registry(),
        protocols=_COHORT_PROTOCOLS,
    )
    view, labels, folds = lane["fixture"].view, lane["labels"], lane["folds"]
    # an unreachable floor types EVERY stratum below_training_floor
    starved = run_regime_cohort_model_study(
        view, labels, folds, **{**kwargs, "minimum_training_rows": 10_000}
    )
    assert starved.envelope.payload.strata
    assert {record.reason for record in starved.envelope.payload.strata} == {
        "below_training_floor"
    }
    assert starved.envelope.payload.specialized_coverage_fraction == 0.0
    assert starved.specialized_predictions.empty
    # the model bundle must not carry the regime block (stratification key, not input)
    with pytest.raises(ValueError, match="must not carry the regime block"):
        run_regime_cohort_model_study(
            view, labels, folds, **{**kwargs, "bundle_key": "B7_CORE_REGIME"}
        )
    # the same frozen-authority gate as feature_only
    below = lane["activation"].__class__(
        **{**lane["activation"].__dict__, "promotion_decision": lane["ready"]}
    )
    with pytest.raises(RegimeSupervisedStudyRefusalError, match="nothing here promotes"):
        run_regime_cohort_model_study(view, labels, folds, **{**kwargs, "activation": below})
