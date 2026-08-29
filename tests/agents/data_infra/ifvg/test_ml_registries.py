"""Registry, decision-policy, calibration, and scope-boundary tests
(ML plan §6/§7/§12; TEST_MATRIX §3.10 "Decision-policy and schedule
envelopes"; acceptance 7B.22-13/14/15/19)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.ml.calibration_policies import (
    CALIBRATION_POLICY_REGISTRY,
    CalibrationPolicyUnavailableError,
    assert_calibration_policy_executable,
)
from alpha_lab.agents.data_infra.ifvg.ml.decision_policies import (
    DECISION_POLICY_REGISTRY,
    S11_BLOCKED_REASON,
    DecisionPolicyEnvelope,
    DecisionPolicyPayload,
    ModelGatedReplayRequest,
    RejectedCandidatePolicy,
    WalkForwardModelScheduleEntry,
    WalkForwardModelScheduleEnvelope,
    WalkForwardModelSchedulePayload,
    resolve_baseline_decision_policy,
)
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import (
    MODEL_PROTOCOL_REGISTRY,
    ModelProtocolStatus,
)
from alpha_lab.agents.data_infra.ifvg.study.study_cell import (
    REGISTERED_CALIBRATION_POLICY_IDS,
    REGISTERED_DECISION_POLICY_KEYS,
    REGISTERED_MODEL_PROTOCOL_KEYS,
)

_ML_SOURCE_DIR = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "alpha_lab"
    / "agents"
    / "data_infra"
    / "ifvg"
    / "ml"
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry alignment and V1 availability boundary
# ─────────────────────────────────────────────────────────────────────────────


def test_model_protocol_registry_aligns_with_study_cell_vocabulary():
    assert tuple(MODEL_PROTOCOL_REGISTRY) == REGISTERED_MODEL_PROTOCOL_KEYS


def test_v1_available_protocols_are_exactly_the_ruled_scope():
    available = tuple(
        key
        for key, entry in MODEL_PROTOCOL_REGISTRY.items()
        if entry.status is ModelProtocolStatus.AVAILABLE
    )
    assert available == (
        "reference_prevalence_v1",
        "ifvg_context_logistic_l2_v1",
        "ifvg_context_catboost_binary_v1",
        # R6.1 (§6.J): the bundle-aware CatBoost rung (research-only)
        "ifvg_context_catboost_bundle_v1",
    )
    assert MODEL_PROTOCOL_REGISTRY["ifvg_context_catboost_bundle_v1"].kind == (
        "nonlinear_challenger_bundle"
    )
    assert MODEL_PROTOCOL_REGISTRY["ifvg_context_gam_v1"].reason == (
        "preregistered_basis_penalty_protocol_not_ratified"
    )


def test_calibration_registry_aligns_and_only_raw_executes():
    assert tuple(CALIBRATION_POLICY_REGISTRY) == REGISTERED_CALIBRATION_POLICY_IDS
    assert_calibration_policy_executable("raw_probability_diagnostics_v1")
    for planned in ("platt_sigmoid_train_fold_v1", "isotonic_train_fold_v1"):
        with pytest.raises(CalibrationPolicyUnavailableError):
            assert_calibration_policy_executable(planned)
        assert CALIBRATION_POLICY_REGISTRY[planned].fit_scope == "train_fold_rows_only"


def test_decision_registry_aligns_and_only_diagnostic_is_available():
    assert tuple(DECISION_POLICY_REGISTRY) == REGISTERED_DECISION_POLICY_KEYS
    for key, entry in DECISION_POLICY_REGISTRY.items():
        if key == "none_diagnostic_only_v1":
            assert entry.status.value == "available"
            assert entry.execution_affecting is False
        else:
            assert entry.status.value == "planned"
            assert entry.execution_affecting is True
            assert entry.reason == S11_BLOCKED_REASON


def test_s11_blocked_reason_matches_the_authorization_layer_verbatim():
    """One sentence, three surfaces: the ml constant, the computation-path
    requirement, and the dimension registry must never drift apart."""

    from alpha_lab.agents.data_infra.ifvg.search import authorization
    from alpha_lab.agents.data_infra.ifvg.study import dimension_contracts

    authorization_source = Path(authorization.__file__).read_text(encoding="utf-8")
    dimension_source = Path(dimension_contracts.__file__).read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", S11_BLOCKED_REASON)
    assert normalized in re.sub(r"[\"\s]+", " ", authorization_source)
    assert normalized in re.sub(r"[\"\s]+", " ", dimension_source)


# ─────────────────────────────────────────────────────────────────────────────
# Decision-policy payload validator (7B.22-13; CS §9)
# ─────────────────────────────────────────────────────────────────────────────


def _execution_payload(**overrides):
    base = {
        "decision_policy_key": "fixed_probability_threshold_v1",
        "parameters": {"threshold": 0.6},
        "resolved_regime_protocol_id": None,
        "rejected_candidate_policy": RejectedCandidatePolicy.REJECT_TERMINATE_SETUP_MISSED,
        "rejected_candidate_policy_ratification_ref": "owner_decision_R1_ref",
        "requires_frozen_model": True,
        "requires_model_gated_sequential_replay": True,
        "produces_new_trade_stream_hash": True,
    }
    base.update(overrides)
    return base


def test_baseline_policy_resolves_to_a_real_64hex_id():
    envelope = resolve_baseline_decision_policy()
    assert re.fullmatch(r"[0-9a-f]{64}", envelope.resolved_decision_policy_id)
    assert envelope.payload.decision_policy_key == "none_diagnostic_only_v1"
    again = resolve_baseline_decision_policy()
    assert again.resolved_decision_policy_id == envelope.resolved_decision_policy_id


def test_diagnostic_policy_refuses_execution_shaped_fields():
    with pytest.raises(ValueError, match="no rejected-candidate policy"):
        DecisionPolicyPayload(
            decision_policy_key="none_diagnostic_only_v1",
            parameters={},
            resolved_regime_protocol_id=None,
            rejected_candidate_policy=RejectedCandidatePolicy.REJECT_RESET_SETUP,
            rejected_candidate_policy_ratification_ref=None,
            requires_frozen_model=False,
            requires_model_gated_sequential_replay=False,
            produces_new_trade_stream_hash=False,
        )
    with pytest.raises(ValueError, match="cannot require"):
        DecisionPolicyPayload(
            decision_policy_key="none_diagnostic_only_v1",
            parameters={},
            resolved_regime_protocol_id=None,
            rejected_candidate_policy=None,
            rejected_candidate_policy_ratification_ref=None,
            requires_frozen_model=True,
            requires_model_gated_sequential_replay=False,
            produces_new_trade_stream_hash=False,
        )


def test_execution_policy_requires_ratified_rejection_semantics():
    DecisionPolicyPayload(**_execution_payload())  # complete form is valid
    with pytest.raises(ValueError, match="rejected-candidate policy"):
        DecisionPolicyPayload(**_execution_payload(rejected_candidate_policy=None))
    with pytest.raises(ValueError, match="ratification reference"):
        DecisionPolicyPayload(
            **_execution_payload(rejected_candidate_policy_ratification_ref=None)
        )
    with pytest.raises(ValueError, match="requires a frozen model"):
        DecisionPolicyPayload(**_execution_payload(requires_frozen_model=False))


def test_regime_conditioned_policy_requires_a_regime_protocol():
    with pytest.raises(ValueError, match="resolved regime protocol id"):
        DecisionPolicyPayload(
            **_execution_payload(
                decision_policy_key="regime_conditioned_threshold_v1",
                resolved_regime_protocol_id=None,
            )
        )
    DecisionPolicyPayload(
        **_execution_payload(
            decision_policy_key="regime_conditioned_threshold_v1",
            resolved_regime_protocol_id="a" * 64,
        )
    )


def test_unregistered_decision_policy_key_is_refused():
    with pytest.raises(ValueError, match="not registered"):
        DecisionPolicyPayload(**_execution_payload(decision_policy_key="made_up_v1"))


# ─────────────────────────────────────────────────────────────────────────────
# §3.10 row: registry key ≠ resolved id; schedule non-self-referential
# ─────────────────────────────────────────────────────────────────────────────


def test_registry_key_is_distinct_from_resolved_policy_id():
    envelope = resolve_baseline_decision_policy()
    assert envelope.payload.decision_policy_key != envelope.resolved_decision_policy_id
    payload_fields = set(DecisionPolicyPayload.model_fields)
    assert "resolved_decision_policy_id" not in payload_fields
    assert "decision_policy_id" not in payload_fields  # the old self-id name


def test_schedule_payload_is_non_self_referential_and_chronological():
    schedule_fields = set(WalkForwardModelSchedulePayload.model_fields)
    assert "resolved_model_schedule_id" not in schedule_fields
    assert "model_schedule_id" not in schedule_fields
    payload = WalkForwardModelSchedulePayload(
        entries=(
            WalkForwardModelScheduleEntry(
                date_from="2026-03-02", date_to="2026-03-06", frozen_model_fit_id="a" * 64
            ),
            WalkForwardModelScheduleEntry(
                date_from="2026-03-09", date_to="2026-03-13", frozen_model_fit_id="b" * 64
            ),
        ),
        coverage_policy_id="uncovered_dates_refuse_v1",
    )
    envelope = WalkForwardModelScheduleEnvelope.from_payload(payload)
    assert re.fullmatch(r"[0-9a-f]{64}", envelope.resolved_model_schedule_id)
    with pytest.raises(ValueError, match="chronological"):
        WalkForwardModelSchedulePayload(
            entries=tuple(reversed(payload.entries)),
            coverage_policy_id="uncovered_dates_refuse_v1",
        )
    with pytest.raises(ValueError, match="reversed"):
        WalkForwardModelScheduleEntry(
            date_from="2026-03-06", date_to="2026-03-02", frozen_model_fit_id="a" * 64
        )


def test_model_gated_replay_request_references_resolved_ids_only():
    request = ModelGatedReplayRequest(
        cell_id="a" * 64,
        frozen_model_fit_id="b" * 64,
        resolved_decision_policy_id="c" * 64,
        resolved_model_schedule_id="d" * 64,
    )
    assert set(type(request).model_fields) == {
        "cell_id",
        "frozen_model_fit_id",
        "resolved_decision_policy_id",
        "resolved_model_schedule_id",
    }


def test_tampered_decision_envelope_fails_closed_on_reload():
    envelope = resolve_baseline_decision_policy()
    with pytest.raises(ValueError, match="does not hash its payload"):
        DecisionPolicyEnvelope(
            resolved_decision_policy_id="0" * 64,
            payload=envelope.payload,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Scope boundaries (7B.22-19 + ML/prop separation)
# ─────────────────────────────────────────────────────────────────────────────


def test_no_deep_learning_or_rl_imports_in_the_ml_lane():
    forbidden = re.compile(
        r"^\s*(?:import|from)\s+(torch|tensorflow|keras|gym|stable_baselines3)\b"
    )
    for source_file in sorted(_ML_SOURCE_DIR.glob("*.py")):
        for line in source_file.read_text(encoding="utf-8").splitlines():
            assert not forbidden.match(line), f"{source_file.name}: {line.strip()}"


def test_drift_module_exports_report_builders_only():
    """The drift lane has no consumer API: nothing retrains, disables,
    promotes, or gates from a drift alarm (ML plan §8)."""

    from alpha_lab.agents.data_infra.ifvg.ml import drift_monitoring

    forbidden_tokens = ("retrain", "disable", "promote", "activate", "gate_from_drift")
    for name in drift_monitoring.__all__:
        lowered = name.lower()
        for token in forbidden_tokens:
            assert token not in lowered
    source = Path(drift_monitoring.__file__).read_text(encoding="utf-8")
    assert "def retrain" not in source
    assert "def promote" not in source


def test_prop_outcomes_cannot_be_model_targets():
    """The supervised/regime target whitelist is ('binary_target',): the
    ladder consumes labels only through the fixed prediction-row schema, and
    no ml module references payout/pass/breach columns as fit targets."""

    payout_shaped = re.compile(
        r"(payout|breach|passed)[\"']?\s*\]\s*(?:\.astype)?.*(fit|target)",
        re.IGNORECASE,
    )
    for source_file in sorted(_ML_SOURCE_DIR.glob("*.py")):
        source = source_file.read_text(encoding="utf-8")
        assert not payout_shaped.search(source), source_file.name
