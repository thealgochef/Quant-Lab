"""R6 regime-contract suites (ML plan §3–§5; V3 P1-3; Amendment P1-B/P1-C;
TEST_MATRIX §3.8 panel row, §3.9 protocol/fit/promotion separation, §3.10
regime fit provenance) — incl. the adversarial-round closures: the
structural promotion + role ladders (F5/S1/S2), fail-closed protocol
policies (S3), default-on bundle/stage input permission (F2/S4), pinned
parameters in the protocol identity (F12), and the stamped decision-row
minimum (F14)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import AvailabilityStage
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml.regime_algorithms import (
    KMEANS_ALGORITHM_KEY,
    POST_V1_REGIME_EXPANSION_KEYS,
    PROTOCOL_POLICY_FIELDS,
    REGIME_ALGORITHM_REGISTRY,
    RegimeAlgorithmUnavailableError,
    assert_protocol_executable,
    assert_regime_algorithm_fittable,
    pinned_parameters_hash,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    INITIALIZATION_POLICIES,
    PROMOTION_SEQUENCE,
    REGIME_PROPOSED_DEFAULTS,
    V1_UNREPRESENTABLE_ROLES,
    ObservationGranularity,
    RegimeFitEnvelope,
    RegimeFitPayload,
    RegimeLeakageError,
    RegimePromotionDecision,
    RegimePromotionDecisionEnvelope,
    RegimeProtocolEnvelope,
    RegimeProtocolPayload,
    RegimeRole,
    RegimeStatus,
    assert_lawful_promotion,
    assert_lawful_role,
    assert_no_regime_leakage,
    sample_adequacy_minimum,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    assert_inputs_permitted,
    resolve_kmeans_protocol,
)

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id
_B0_NAMES = set(resolve_bundle("B0_CORE").payload.resolved_feature_names)
_B1_ONLY = sorted(
    set(resolve_bundle("B1_CORE_STRUCTURE").payload.resolved_feature_names) - _B0_NAMES
)[0]
_RATIFICATION = "f" * 64


def _protocol(**overrides):
    defaults = dict(
        input_feature_bundle_ref=_B0,
        resolved_input_features=("distance_to_htf_ticks", "opposing_size_ticks"),
    )
    defaults.update(overrides)
    return resolve_kmeans_protocol(**defaults)


def _decision(**overrides) -> RegimePromotionDecision:
    defaults = dict(
        resolved_regime_protocol_id="a" * 64,
        role=RegimeRole.DESCRIPTIVE_ONLY,
        status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_status=RegimeStatus.PLANNED,
        previous_decision_ref=None,
        capability_assessment_ref="e" * 64,
        owner_ratification_ref=None,
        decided_at="2026-08-26T00:00:00Z",
    )
    defaults.update(overrides)
    return RegimePromotionDecision(**defaults)


# ── panel-grain validation (Amendment P1-B; §3.8 row) ────────────────────────


def test_panel_grain_requires_all_three_panel_fields():
    for interval in (300, 900):  # the 5m and 15m panels
        envelope = _protocol(
            observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
            panel_interval_seconds=interval,
            panel_source_artifact_id="b" * 64,
            panel_as_of_policy_id="completed_bars_last_at_or_before_v1",
        )
        assert envelope.payload.panel_interval_seconds == interval
    # every invalid combination refuses
    for missing in ("panel_interval_seconds", "panel_source_artifact_id", "panel_as_of_policy_id"):
        kwargs = dict(
            observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
            panel_interval_seconds=300,
            panel_source_artifact_id="b" * 64,
            panel_as_of_policy_id="completed_bars_last_at_or_before_v1",
        )
        kwargs[missing] = None
        with pytest.raises(ValueError, match="CONTEXT_BAR_PANEL requires"):
            _protocol(**kwargs)
    with pytest.raises(ValueError, match="at least 60"):
        _protocol(
            observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
            panel_interval_seconds=1,
            panel_source_artifact_id="b" * 64,
            panel_as_of_policy_id="completed_bars_last_at_or_before_v1",
        )


def test_non_panel_grains_refuse_panel_fields():
    for grain in (
        ObservationGranularity.CANDIDATE_STAGE_ROW,
        ObservationGranularity.DECISION_ROW,
    ):
        envelope = _protocol(observation_granularity=grain)
        assert envelope.payload.panel_interval_seconds is None
        with pytest.raises(ValueError, match="carries no panel fields"):
            _protocol(
                observation_granularity=grain,
                panel_interval_seconds=300,
            )


# ── protocol/fit/promotion separation (V3 P1-3; §3.9/§3.10 rows) ─────────────


def test_promotion_changes_no_protocol_or_fit_identity():
    protocol = _protocol()
    fit = RegimeFitEnvelope.from_payload(
        RegimeFitPayload(
            resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
            source_artifact_ids=("c" * 64,),
            fold_index=0,
            fit_start="2026-01-05",
            fit_end="2026-02-27",
            training_row_ids_hash="d" * 64,
            training_feature_matrix_hash="f" * 64,
        )
    )
    first = RegimePromotionDecisionEnvelope.from_payload(
        _decision(resolved_regime_protocol_id=protocol.resolved_regime_protocol_id)
    )
    _decision(
        resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
        role=RegimeRole.STRATIFICATION_ONLY,
        status=RegimeStatus.STRATIFICATION_READY,
        previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
        previous_decision_ref=first.regime_promotion_decision_id,
    )
    # the numerical identities are untouched by any status change
    assert (
        RegimeProtocolEnvelope.from_payload(protocol.payload).resolved_regime_protocol_id
        == protocol.resolved_regime_protocol_id
    )
    assert RegimeFitEnvelope.from_payload(fit.payload).regime_fit_id == fit.regime_fit_id
    # role/status fields are structurally ABSENT from protocol and fit
    for field in ("role", "status"):
        assert field not in type(protocol.payload).model_fields
        assert field not in RegimeFitPayload.model_fields


def test_fit_identity_requires_verified_sources_and_binds_training_values():
    base = dict(
        resolved_regime_protocol_id="a" * 64,
        fold_index=0,
        fit_start=None,
        fit_end=None,
        training_row_ids_hash="d" * 64,
        training_feature_matrix_hash="f" * 64,
    )
    with pytest.raises(ValueError):
        RegimeFitPayload(source_artifact_ids=(), **base)
    with pytest.raises(ValueError, match="verified 64-hex"):
        RegimeFitPayload(source_artifact_ids=("not-an-id",), **base)
    one = RegimeFitEnvelope.from_payload(
        RegimeFitPayload(source_artifact_ids=("c" * 64,), **base)
    )
    other = RegimeFitEnvelope.from_payload(
        RegimeFitPayload(
            source_artifact_ids=("c" * 64,),
            **{**base, "training_feature_matrix_hash": "e" * 64},
        )
    )
    assert one.regime_fit_id != other.regime_fit_id


def test_promotion_ladder_is_structural_at_the_contract():
    assert PROMOTION_SEQUENCE[-2:] == (
        RegimeStatus.FEATURE_ELIGIBLE,
        RegimeStatus.MODEL_FEATURE,
    )
    # one forward step at a time — refused at CONSTRUCTION
    with pytest.raises(ValueError, match="skips the sequence"):
        _decision(status=RegimeStatus.FEATURE_ELIGIBLE, owner_ratification_ref=_RATIFICATION)
    # a chained decision must reference its predecessor; a first decision must not
    with pytest.raises(ValueError, match="previous_decision_ref"):
        _decision(
            status=RegimeStatus.STRATIFICATION_READY,
            previous_status=RegimeStatus.DESCRIPTIVE_ONLY,
        )
    with pytest.raises(ValueError, match="previous_decision_ref"):
        _decision(previous_decision_ref="b" * 64)
    # feature-eligible requires the owner's 64-hex ratification reference
    with pytest.raises(ValueError, match="ratification"):
        _decision(
            role=RegimeRole.FEATURE_GENERATOR,
            status=RegimeStatus.FEATURE_ELIGIBLE,
            previous_status=RegimeStatus.STRATIFICATION_READY,
            previous_decision_ref="b" * 64,
        )
    with pytest.raises(ValueError):
        _decision(
            role=RegimeRole.FEATURE_GENERATOR,
            status=RegimeStatus.FEATURE_ELIGIBLE,
            previous_status=RegimeStatus.STRATIFICATION_READY,
            previous_decision_ref="b" * 64,
            owner_ratification_ref="x",  # not an OwnerDecisionEvidenceRef hash
        )
    lawful = _decision(
        role=RegimeRole.FEATURE_GENERATOR,
        status=RegimeStatus.FEATURE_ELIGIBLE,
        previous_status=RegimeStatus.STRATIFICATION_READY,
        previous_decision_ref="b" * 64,
        owner_ratification_ref=_RATIFICATION,
    )
    assert lawful.status is RegimeStatus.FEATURE_ELIGIBLE
    # decision time is ISO-8601 with an explicit offset
    with pytest.raises(ValueError, match="ISO-8601"):
        _decision(decided_at="whenever")
    with pytest.raises(ValueError, match="offset"):
        _decision(decided_at="2026-08-26T00:00:00")
    # SUPERSEDED is terminal; PLANNED is never a target
    with pytest.raises(ValueError, match="terminal"):
        _decision(previous_status=RegimeStatus.SUPERSEDED, previous_decision_ref="b" * 64)
    with pytest.raises(ValueError, match="never a decision target"):
        _decision(status=RegimeStatus.PLANNED)
    # the gate half (helper contract): passing gates are required too
    with pytest.raises(ValueError, match="PASSING capability"):
        assert_lawful_promotion(
            RegimeStatus.STRATIFICATION_READY,
            RegimeStatus.FEATURE_ELIGIBLE,
            owner_ratification_ref=_RATIFICATION,
            gates_passed=False,
        )
    assert_lawful_promotion(
        RegimeStatus.STRATIFICATION_READY,
        RegimeStatus.FEATURE_ELIGIBLE,
        owner_ratification_ref=_RATIFICATION,
        gates_passed=True,
    )
    # demotion/blocking is always lawful; EXPERIMENTAL is a non-promoting
    # state that earns no role above descriptive/monitoring
    assert_lawful_promotion(
        RegimeStatus.FEATURE_ELIGIBLE,
        RegimeStatus.BLOCKED_INSUFFICIENT_COVERAGE,
        owner_ratification_ref=None,
        gates_passed=False,
    )
    experimental = _decision(status=RegimeStatus.EXPERIMENTAL)
    assert experimental.status is RegimeStatus.EXPERIMENTAL
    with pytest.raises(ValueError, match="requires status"):
        _decision(role=RegimeRole.STRATIFICATION_ONLY, status=RegimeStatus.EXPERIMENTAL)
    # model_copy cannot escape the contract (pydantic re-validates)
    with pytest.raises(ValueError):
        RegimePromotionDecisionEnvelope.from_payload(
            _decision().model_copy(update={"status": RegimeStatus.MODEL_FEATURE})
        )


def test_role_ladder_makes_execution_roles_unrepresentable_in_v1():
    from alpha_lab.agents.data_infra.ifvg.ml.decision_policies import S11_BLOCKED_REASON

    for role in V1_UNREPRESENTABLE_ROLES:
        with pytest.raises(ValueError, match="unrepresentable in V1"):
            assert_lawful_role(role, RegimeStatus.MODEL_FEATURE)
        with pytest.raises(ValueError) as caught:
            _decision(role=role)
        assert S11_BLOCKED_REASON in str(caught.value)
    # a role is lawful only at or above the status that earns it
    with pytest.raises(ValueError, match="requires status"):
        _decision(role=RegimeRole.FEATURE_GENERATOR)
    with pytest.raises(ValueError, match="requires status"):
        _decision(
            role=RegimeRole.PREDICTIVE_MODEL,
            status=RegimeStatus.FEATURE_ELIGIBLE,
            previous_status=RegimeStatus.STRATIFICATION_READY,
            previous_decision_ref="b" * 64,
            owner_ratification_ref=_RATIFICATION,
        )
    assert_lawful_role(RegimeRole.MONITORING_ONLY, RegimeStatus.BLOCKED_NO_OOS_ASSIGNMENT)
    assert_lawful_role(RegimeRole.STRATIFICATION_ONLY, RegimeStatus.STRATIFICATION_READY)


# ── leakage + input permission (7B.22-3; §5.2) ───────────────────────────────


def test_leakage_validator_is_default_on_from_the_registries():
    with pytest.raises(RegimeLeakageError, match="prohibited"):
        assert_no_regime_leakage(
            ("distance_to_htf_ticks", "binary_target"),
            observation_stage=AvailabilityStage.ENTRY_DECISION,
        )
    with pytest.raises(RegimeLeakageError, match="prohibited"):
        assert_no_regime_leakage(
            ("expected_payout_share",),
            observation_stage=AvailabilityStage.ENTRY_DECISION,
        )
    # the stage rule applies WITHOUT any caller-supplied map: an
    # entry_decision block feature is refused at an earlier stage
    with pytest.raises(RegimeLeakageError, match="available only after"):
        assert_no_regime_leakage(
            ("direction",), observation_stage=AvailabilityStage.HTF_TAP
        )
    # an unregistered feature has no provable stage → refused
    with pytest.raises(RegimeLeakageError, match="no registered availability stage"):
        assert_no_regime_leakage(
            ("made_up_feature",), observation_stage=AvailabilityStage.ENTRY_DECISION
        )
    # an explicit map is honored (and still refuses late features)
    with pytest.raises(RegimeLeakageError, match="available only after"):
        assert_no_regime_leakage(
            ("late_feature",),
            observation_stage=AvailabilityStage.PARENT_LOCK,
            feature_stage_for={"late_feature": AvailabilityStage.ENTRY_DECISION},
        )
    assert_no_regime_leakage(
        ("distance_to_htf_ticks",),
        observation_stage=AvailabilityStage.ENTRY_DECISION,
    )
    # the protocol resolver runs the validator at construction — both halves
    with pytest.raises(RegimeLeakageError):
        _protocol(resolved_input_features=("net_r",))
    with pytest.raises(RegimeLeakageError, match="available only after"):
        _protocol(observation_stage=AvailabilityStage.HTF_TAP)


def test_inputs_must_belong_to_the_referenced_bundle():
    with pytest.raises(RegimeLeakageError, match="outside the referenced feature bundle"):
        _protocol(resolved_input_features=("distance_to_htf_ticks", _B1_ONLY))
    with pytest.raises(ValueError, match="does not resolve"):
        _protocol(input_feature_bundle_ref="a" * 64)
    # a hand-built payload is re-checked by the same permission seam
    hacked = _protocol().payload.model_copy(
        update={"resolved_input_features": ("distance_to_htf_ticks", _B1_ONLY)}
    )
    with pytest.raises(RegimeLeakageError, match="outside"):
        assert_inputs_permitted(hacked)


# ── planned refusals + registry distinctness (P1-C; 7B.22-18; S3) ────────────


def test_planned_algorithms_refuse_with_the_registered_status_and_reason():
    for key in POST_V1_REGIME_EXPANSION_KEYS:
        with pytest.raises(RegimeAlgorithmUnavailableError, match="planned"):
            assert_regime_algorithm_fittable(key)
    assert_regime_algorithm_fittable(KMEANS_ALGORITHM_KEY)
    with pytest.raises(ValueError, match="unknown regime algorithm"):
        assert_regime_algorithm_fittable("made_up_v1")


_PLANNED_POLICY_VALUES = {
    "cluster_count_policy": "inner_train_only_selection",
    "dimensionality_reduction_policy": "pca_fixed_components_v1",
    "kernel_or_affinity_policy": "rbf",
    "out_of_sample_assignment_policy": "none_training_only",
}


def test_planned_protocol_policies_fail_closed_before_any_fit():
    """Safety S3: every hashed policy value the registry cannot execute is
    refused with the planned reason — a protocol never silently runs as
    fixed-k centroid-predict KMeans."""

    payload = _protocol().payload
    assert_protocol_executable(payload)
    for field, value in _PLANNED_POLICY_VALUES.items():
        hacked = payload.model_copy(update={field: value})
        with pytest.raises(RegimeAlgorithmUnavailableError, match="planned"):
            assert_protocol_executable(hacked)
    assert set(_PLANNED_POLICY_VALUES) < set(PROTOCOL_POLICY_FIELDS)
    # registry drift: version or pinned-parameter hash → refused
    with pytest.raises(RegimeAlgorithmUnavailableError, match="algorithm_version"):
        assert_protocol_executable(payload.model_copy(update={"algorithm_version": "2"}))
    with pytest.raises(RegimeAlgorithmUnavailableError, match="pinned_parameters_hash"):
        assert_protocol_executable(
            payload.model_copy(update={"pinned_parameters_hash": "0" * 64})
        )
    # a planned entry's initialization policy is EXPRESSIBLE but not executable
    gmm = payload.model_copy(
        update={
            "algorithm_key": "gaussian_mixture_v1",
            "initialization_policy": "gmm_kmeans_init_n_init_5_v1",
            "out_of_sample_assignment_policy": "gmm_posterior_v1",
            "pinned_parameters_hash": pinned_parameters_hash(
                REGIME_ALGORITHM_REGISTRY["gaussian_mixture_v1"]
            ),
        }
    )
    assert RegimeProtocolEnvelope.from_payload(gmm).payload.algorithm_key == "gaussian_mixture_v1"
    with pytest.raises(RegimeAlgorithmUnavailableError, match="planned"):
        assert_protocol_executable(gmm)
    with pytest.raises(RegimeAlgorithmUnavailableError, match="initialization_policy"):
        assert_protocol_executable(
            payload.model_copy(update={"initialization_policy": "surrogate_logistic_v1"})
        )
    with pytest.raises(ValueError, match="not registered"):
        RegimeProtocolPayload.model_validate(
            {**payload.model_dump(mode="json"), "initialization_policy": "made_up"}
        )


def test_algorithm_identities_are_distinct_with_distinct_pinned_payloads():
    keys = tuple(REGIME_ALGORITHM_REGISTRY)
    assert len(set(keys)) == len(keys)
    hashes = {pinned_parameters_hash(entry) for entry in REGIME_ALGORITHM_REGISTRY.values()}
    assert len(hashes) == len(keys)
    policies = [entry.initialization_policy for entry in REGIME_ALGORITHM_REGISTRY.values()]
    assert len(set(policies)) == len(policies)
    assert set(policies) == set(INITIALIZATION_POLICIES)
    spectral = REGIME_ALGORITHM_REGISTRY["spectral_clustering_train_only_v1"]
    assert spectral.oos_capable is False
    assert spectral.default_status_on_fit is RegimeStatus.BLOCKED_NO_OOS_ASSIGNMENT
    assert spectral.mandatory_warning_text.startswith("Training-only")
    assert dict(spectral.constraints)["train_fold_only_affinity"] is True
    nystrom = REGIME_ALGORITHM_REGISTRY["nystrom_kmeans_v1"]
    pinned = dict(nystrom.pinned_parameters)
    assert "nystroem_random_state" in pinned
    # nested registry values are immutable (CS §0.3)
    assert isinstance(pinned["kmeans"], tuple)
    for entry in REGIME_ALGORITHM_REGISTRY.values():
        if entry.implementation_status == "planned":
            assert dict(entry.executable_policies) == {}


def test_regime_feature_block_still_refuses_in_v1():
    """Acceptance 7B.22-6 (V1, status-driven): training-only spectral
    results cannot enter a predictive bundle because the regime feature
    block itself stays planned — the resolver refuses it."""

    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
        BlockUnavailableError,
        resolve_available_block,
    )

    with pytest.raises(BlockUnavailableError, match="planned"):
        resolve_available_block("IFVG_REGIME_CONTEXT_V1")


# ── proposal stamps (P1-3) ───────────────────────────────────────────────────


def test_every_scientific_default_is_stamped_proposed():
    expected = {
        "algorithm_baseline",
        "fixed_cluster_count",
        "minimum_cluster_occupancy_fraction",
        "minimum_cluster_rows_per_fold",
        "bootstrap_aligned_ami_advisory_floor",
        "minimum_training_observations_candidate_stage",
        "minimum_training_observations_decision_row",
        "minimum_training_observations_panel",
        "observation_grain_baseline",
    }
    assert set(REGIME_PROPOSED_DEFAULTS) == expected
    for entry in REGIME_PROPOSED_DEFAULTS.values():
        assert entry["stamp"] == "proposed_protocol_default"
        assert entry["owner_ratification_required_before_feature_eligible"] is True
    # the service reads its gates from the SAME stamped source
    assert sample_adequacy_minimum(ObservationGranularity.CANDIDATE_STAGE_ROW) == 150
    assert sample_adequacy_minimum(ObservationGranularity.DECISION_ROW) == 150
    assert sample_adequacy_minimum(ObservationGranularity.CONTEXT_BAR_PANEL) == 300


def test_protocol_identity_is_sensitive_to_science_and_pins_the_seed():
    base = _protocol()
    assert base.payload.random_seed == 7
    assert base.payload.pinned_parameters_hash == pinned_parameters_hash(
        REGIME_ALGORITHM_REGISTRY[KMEANS_ALGORITHM_KEY]
    )
    changed_k = _protocol(resolved_cluster_count=4)
    assert changed_k.resolved_regime_protocol_id != base.resolved_regime_protocol_id
    changed_features = _protocol(resolved_input_features=("distance_to_htf_ticks",))
    assert changed_features.resolved_regime_protocol_id != base.resolved_regime_protocol_id
    # software versions and pinned parameters are identity (CS §0.2)
    versions = RegimeProtocolEnvelope.from_payload(
        base.payload.model_copy(update={"software_versions": {"scikit-learn": "0.0"}})
    )
    assert versions.resolved_regime_protocol_id != base.resolved_regime_protocol_id
    pins = RegimeProtocolEnvelope.from_payload(
        base.payload.model_copy(update={"pinned_parameters_hash": "0" * 64})
    )
    assert pins.resolved_regime_protocol_id != base.resolved_regime_protocol_id
    assert base.payload.fit_scope == "per_training_fold"
