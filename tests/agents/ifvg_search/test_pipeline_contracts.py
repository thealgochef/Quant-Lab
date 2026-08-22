"""Pipeline contract tests (CS §7; TEST_MATRIX §3.1 P0-3, §3.9
capability-gated operator readiness, §3.10 closure rows)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.search.charter import SimulationProtocol
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    _FORBIDDEN_PAYLOAD_FIELDS,
    registered_identity_pairs,
)
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    CANONICAL_STAGE_ORDER,
    S11_BLOCKED_REASON,
    ExecutionAttemptIdentity,
    PipelineSemanticSpecPayload,
    QuantLabPipelineStage,
    StagePlanBlockedError,
    StageStatus,
    WorkerPolicy,
    assert_stage_plan_launchable,
    derive_stage_plan_readiness,
)


def _spec(**overrides) -> PipelineSemanticSpecPayload:
    base = {
        "run_scope": "verification_5d",
        "date_allowlist": ("2026-06-04", "2026-06-05"),
        "allowlist_hash": allowlist_sha256(("2026-06-04", "2026-06-05")),
        "warmup_policy_id": "zero_real_warmup_seed_snapshot_v1",
        "search_charter_id": "a" * 64,
        "source_artifact_ids": (),
        "feature_bundle_ids": (),
        "label_policy_id": None,
        "fold_protocol_id": None,
        "model_protocol_id": None,
        "cost_policy_sha256": "b" * 64,
        "account_policy_set_ids": (),
        "portfolio_policy_ids": (),
        "simulation_protocol": SimulationProtocol(
            modes=("historical_closed_trade",),
            stress_scenario_ids=(),
            trade_path_capability_policy_id="path_capability_policy_v1",
            clock_policy_id="simulated_clock_v1",
        ),
        "software_commits": {"quant_lab": "c" * 40},
        "stage_plan": (
            QuantLabPipelineStage.S00_VALIDATE_INPUTS,
            QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES,
            QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
        ),
    }
    base.update(overrides)
    return PipelineSemanticSpecPayload(**base)


# ─────────────────────────────────────────────────────────────────────────────
# CS §7 contract shapes
# ─────────────────────────────────────────────────────────────────────────────


def test_canonical_stage_order_is_the_sixteen_stage_contract():
    assert len(CANONICAL_STAGE_ORDER) == 16
    assert CANONICAL_STAGE_ORDER[0] is QuantLabPipelineStage.S00_VALIDATE_INPUTS
    assert CANONICAL_STAGE_ORDER[-1] is QuantLabPipelineStage.S15_VERIFY_AND_PUBLISH
    assert QuantLabPipelineStage.S11_RUN_FROZEN_MODEL_GATED_REPLAYS.value == (
        "11_run_frozen_model_gated_replays"
    )
    assert len(StageStatus) == 10


def test_attempt_identity_is_operational_and_never_registered():
    """P0-3: attempt facts live in the forbidden-payload vocabulary; the
    attempt contract must never appear in the identity-projection registry."""

    attempt_fields = set(ExecutionAttemptIdentity.model_fields)
    assert attempt_fields & _FORBIDDEN_PAYLOAD_FIELDS  # by design
    names = {pair.name for pair in registered_identity_pairs()}
    assert "ExecutionAttempt" not in names
    for pair in registered_identity_pairs():
        assert pair.payload_cls is not ExecutionAttemptIdentity


def test_worker_policy_rejects_more_than_four_workers():
    with pytest.raises(ValueError):
        WorkerPolicy(max_workers=8, max_tasks_per_child=1, memory_budget_bytes=0)


def test_spec_refuses_unordered_or_incoherent_plans():
    with pytest.raises(ValueError, match="canonical stage order"):
        _spec(
            stage_plan=(
                QuantLabPipelineStage.S00_VALIDATE_INPUTS,
                QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
                QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES,
            )
        )
    with pytest.raises(ValueError, match="requires 01_prepare"):
        _spec(
            stage_plan=(
                QuantLabPipelineStage.S00_VALIDATE_INPUTS,
                QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
            )
        )
    with pytest.raises(ValueError, match="begins with 00_validate_inputs"):
        _spec(
            stage_plan=(
                QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES,
                QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
            )
        )
    with pytest.raises(ValueError, match="at most five real days"):
        _spec(
            date_allowlist=tuple(f"2026-06-{day:02d}" for day in (1, 2, 3, 4, 5, 8)),
            allowlist_hash=allowlist_sha256(
                tuple(f"2026-06-{day:02d}" for day in (1, 2, 3, 4, 5, 8))
            ),
        )
    with pytest.raises(ValueError, match="does not hash"):
        _spec(allowlist_hash="0" * 64)


def test_spec_requires_protocol_ids_for_ml_stages():
    ml_plan = (
        QuantLabPipelineStage.S00_VALIDATE_INPUTS,
        QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES,
        QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
        QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS,
        QuantLabPipelineStage.S07_DERIVE_LABELS,
        QuantLabPipelineStage.S08_BUILD_FOLDS,
        QuantLabPipelineStage.S09_TRAIN_MODELS,
    )
    with pytest.raises(ValueError, match="require at least one feature bundle"):
        _spec(stage_plan=ml_plan)
    with pytest.raises(ValueError, match="requires a label policy id"):
        _spec(stage_plan=ml_plan, feature_bundle_ids=("B0_CORE",))
    with pytest.raises(ValueError, match="requires a fold protocol id"):
        _spec(
            stage_plan=ml_plan,
            feature_bundle_ids=("B0_CORE",),
            label_policy_id="x_v1",
        )
    with pytest.raises(ValueError, match="requires a model protocol id"):
        _spec(
            stage_plan=ml_plan,
            feature_bundle_ids=("B0_CORE",),
            label_policy_id="x_v1",
            fold_protocol_id="ifvg_context_walkforward_40_5_5_2_v1",
        )


# ─────────────────────────────────────────────────────────────────────────────
# §3.9 capability-gated operator readiness
# ─────────────────────────────────────────────────────────────────────────────

_FULL_ML_PLAN = (
    QuantLabPipelineStage.S00_VALIDATE_INPUTS,
    QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES,
    QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
    QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS,
    QuantLabPipelineStage.S07_DERIVE_LABELS,
    QuantLabPipelineStage.S08_BUILD_FOLDS,
    QuantLabPipelineStage.S09_TRAIN_MODELS,
    QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS,
)


def test_strategy_only_plan_is_launchable_without_r5b_or_r6():
    spec = _spec()
    report = assert_stage_plan_launchable(spec)
    assert report.launchable
    assert all(entry.state == "available" for entry in report.entries)


def test_mbp1_study_plan_refuses_before_r5b():
    """`IFVG_ORDER_FLOW_MBP1_V1` is planned through R5: a bundle carrying it
    (B2_CORE_ORDER_FLOW) blocks every feature/model stage and the launch —
    no baseline-vs-MBP-1 study is constructible."""

    spec = _spec(
        stage_plan=_FULL_ML_PLAN,
        feature_bundle_ids=("B2_CORE_ORDER_FLOW",),
        label_policy_id="x_v1",
        fold_protocol_id="ifvg_context_walkforward_40_5_5_2_v1",
        model_protocol_id="ifvg_context_catboost_binary_v1",
    )
    report = derive_stage_plan_readiness(spec)
    assert not report.launchable
    blocked = report.entry(QuantLabPipelineStage.S05_MATERIALIZE_FEATURE_VIEWS)
    assert blocked.state == "blocked_capability"
    assert "planned" in blocked.reason
    with pytest.raises(StagePlanBlockedError, match="unavailable capabilities"):
        assert_stage_plan_launchable(spec)


def test_spectral_plan_refuses_before_the_post_v1_expansion():
    spec = _spec(
        stage_plan=_FULL_ML_PLAN,
        feature_bundle_ids=("B0_CORE",),
        label_policy_id="x_v1",
        fold_protocol_id="ifvg_context_walkforward_40_5_5_2_v1",
        model_protocol_id="spectral_clustering_train_only_v1",
    )
    report = derive_stage_plan_readiness(spec)
    assert not report.launchable
    blocked = report.entry(QuantLabPipelineStage.S09_TRAIN_MODELS)
    assert blocked.state == "blocked_capability"
    assert "post-V1" in blocked.reason
    with pytest.raises(StagePlanBlockedError):
        assert_stage_plan_launchable(spec)


def test_planned_gam_protocol_blocks_the_training_stage():
    spec = _spec(
        stage_plan=_FULL_ML_PLAN,
        feature_bundle_ids=("B0_CORE",),
        label_policy_id="x_v1",
        fold_protocol_id="ifvg_context_walkforward_40_5_5_2_v1",
        model_protocol_id="ifvg_context_gam_v1",
    )
    report = derive_stage_plan_readiness(spec)
    assert not report.launchable
    assert "planned" in report.entry(QuantLabPipelineStage.S09_TRAIN_MODELS).reason


def test_s11_is_blocked_terminal_and_never_prevents_launch():
    plan = (
        *_FULL_ML_PLAN,
        QuantLabPipelineStage.S11_RUN_FROZEN_MODEL_GATED_REPLAYS,
    )
    spec = _spec(
        stage_plan=plan,
        feature_bundle_ids=("B0_CORE",),
        label_policy_id="x_v1",
        fold_protocol_id="ifvg_context_walkforward_40_5_5_2_v1",
        model_protocol_id="ifvg_context_catboost_binary_v1",
    )
    report = assert_stage_plan_launchable(spec)  # launchable despite S11
    entry = report.entry(QuantLabPipelineStage.S11_RUN_FROZEN_MODEL_GATED_REPLAYS)
    assert entry.state == "blocked_terminal"
    assert entry.reason == S11_BLOCKED_REASON


def test_semantic_id_ignores_worker_policy_by_construction():
    """P0-3 static half: no worker/resource field exists on the semantic
    payload, so a resource change cannot change the scientific identity."""

    fields = set(PipelineSemanticSpecPayload.model_fields)
    assert not fields & {
        "worker_policy",
        "max_workers",
        "memory_budget_bytes",
        "attempt_number",
        "host_environment",
    }
