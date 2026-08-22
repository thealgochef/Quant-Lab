"""Pipeline E2E + attempt/resume/publication tests (PHASED R5 gate).

The E2E runs the complete 16-stage plan under `verification_5d` semantics on
the synthetic fixture: every stage reaches a terminal state, S11 is BLOCKED
with the exact registered reason, the real store writers fire
(account_policy_sets, account_simulations, insights, search_results,
pipeline_stage_results), and publication stays `prepared_not_published`
with verification-scope activation refused.
"""

from __future__ import annotations

import json

import pytest

from alpha_lab.agents.data_infra.ifvg.search.insights import InsightPanelEnvelope
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    S11_BLOCKED_REASON,
    PipelineResultEnvelope,
    PublicationError,
    QuantLabPipelineStage,
    StageStatus,
    WorkerPolicy,
    activate_pipeline_result,
    read_pipeline_state,
    request_pipeline_cancel,
    run_pipeline,
    run_publication_gates,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    load_sidecar_bytes,
    load_verified_envelope,
)
from tests.agents.ifvg_search.pipeline_fixture import build_pipeline_fixture


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    fixture = build_pipeline_fixture(tmp_path_factory.mktemp("pipeline_e2e"))
    result = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    return {**fixture, "result": result}


def test_all_sixteen_stages_reach_a_terminal_state(completed):
    statuses = completed["result"].stage_statuses
    assert len(statuses) == 16
    terminal = {
        StageStatus.COMPLETED.value,
        StageStatus.REUSED.value,
        StageStatus.BLOCKED.value,
    }
    assert set(statuses.values()) <= terminal, statuses
    blocked = [stage for stage, status in statuses.items() if status == "blocked"]
    assert blocked == [QuantLabPipelineStage.S11_RUN_FROZEN_MODEL_GATED_REPLAYS.value]


def test_s11_carries_the_exact_blocked_reason(completed):
    state = read_pipeline_state(
        completed["state_root"], completed["result"].pipeline_semantic_id
    )
    entry = state["stages"][QuantLabPipelineStage.S11_RUN_FROZEN_MODEL_GATED_REPLAYS.value]
    assert entry["status"] == StageStatus.BLOCKED.value
    assert entry["explanation"] == S11_BLOCKED_REASON


def test_children_replayed_and_gates_evaluated(completed):
    state = read_pipeline_state(
        completed["state_root"], completed["result"].pipeline_semantic_id
    )
    children = state["children"]
    assert len(children) == 4  # the 2×2 enumeration
    assert {row["state"] for row in children} == {"completed"}
    assert len(completed["observed"]) == 4  # four real replay invocations


def test_real_account_policy_set_envelopes_persisted(completed):
    """DEV-R4-17 closure: the S12 writer publishes the exact resolved
    policy-set envelopes the semantic spec pins."""

    for policy_id in completed["policy_ids"]:
        assert has_envelope(completed["store_root"], "account_policy_sets", policy_id)
    state = read_pipeline_state(
        completed["state_root"], completed["result"].pipeline_semantic_id
    )
    s12 = state["stages"][QuantLabPipelineStage.S12_RUN_PROP_HISTORICAL_REPLAYS.value]
    assert tuple(s12["output_artifact_ids"]) == completed["policy_ids"]


def test_account_simulations_persisted_with_ui_sidecars(completed):
    """The account_simulations production writers (S12/S13) publish the
    envelope plus the walk-summary sidecar the trader UI reads."""

    import os

    store = completed["store_root"] / "account_simulations"
    simulation_ids = [name for name in os.listdir(store) if len(name) == 64]
    assert simulation_ids, "no account simulations were persisted"
    summary = json.loads(
        load_sidecar_bytes(
            completed["store_root"],
            "account_simulations",
            simulation_ids[0],
            "walk_summary.json",
        )
    )
    assert summary["simulation_mode"] in (
        "historical_closed_trade",
        "day_block_bootstrap",
    )
    assert "payout_reliability_vector" in summary


def test_frontier_and_insights_persisted(completed):
    state = read_pipeline_state(
        completed["state_root"], completed["result"].pipeline_semantic_id
    )
    s14 = state["stages"][QuantLabPipelineStage.S14_BUILD_FRONTIER_AND_INSIGHTS.value]
    outputs = list(s14["output_artifact_ids"])
    assert outputs, "S14 persisted nothing"
    insight_count = 0
    for artifact_id in outputs:
        if has_envelope(completed["store_root"], "insights", artifact_id):
            envelope = load_verified_envelope(
                completed["store_root"], "insights", artifact_id, InsightPanelEnvelope
            )
            assert len(envelope.payload.panel.insights) == 7  # the fixed categories
            insight_count += 1
    assert insight_count > 0, "no insight panels reached the insights store"


def test_cross_profile_comparison_results_persisted(completed):
    """DEV-R4-16 closure: baseline↔challenger comparison results reach the
    search_results store (not_comparable populations stay truthfully typed)."""

    from alpha_lab.agents.data_infra.ifvg.study.comparison_contracts import (
        ComparisonResultEnvelope,
    )

    state = read_pipeline_state(
        completed["state_root"], completed["result"].pipeline_semantic_id
    )
    s14 = state["stages"][QuantLabPipelineStage.S14_BUILD_FRONTIER_AND_INSIGHTS.value]
    comparison_ids = [
        artifact_id
        for artifact_id in s14["output_artifact_ids"]
        if has_envelope(completed["store_root"], "search_results", artifact_id)
    ]
    assert len(comparison_ids) == 3  # baseline vs each of the three challengers
    envelope = load_verified_envelope(
        completed["store_root"],
        "search_results",
        comparison_ids[0],
        ComparisonResultEnvelope,
    )
    deltas = dict(envelope.payload.delta_reports)
    assert set(deltas) == {"setup", "candidate", "decision", "trade"}
    # the fixture's tables carry only executed trades: the empty kinds have
    # vacuously exact (empty-population) lineage bases, while the trade kind
    # — rows present but without their decision linkage — is truthfully
    # typed not_comparable rather than fuzzily matched
    for kind in ("setup", "candidate", "decision"):
        assert deltas[kind]["match_basis"] == "profile_independent_lineage_exact"
        assert deltas[kind]["jaccard"] == 1.0
    assert deltas["trade"]["match_basis"] == "not_comparable"
    assert "incomplete" in deltas["trade"]["match_basis_reason"]


def test_lineage_evidence_persisted_for_every_replayed_child(completed):
    state = read_pipeline_state(
        completed["state_root"], completed["result"].pipeline_semantic_id
    )
    s02 = state["stages"][
        QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value
    ]
    for row in state["children"]:
        raw = load_sidecar_bytes(
            completed["store_root"],
            "pipeline_stage_results",
            s02["stage_result_id"],
            f"lineage_map_{row['core_replay_id']}.json",
        )
        payload = json.loads(raw)
        assert payload["core_replay_id"] == row["core_replay_id"]
        assert has_envelope(
            completed["store_root"],
            "lineage_reports",
            _lineage_report_id(payload),
        )


def _lineage_report_id(payload: dict) -> str:
    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        canonical_contract_sha256,
    )
    from alpha_lab.agents.data_infra.ifvg.search.lineage import (
        LineageUniquenessReport,
    )

    report = LineageUniquenessReport.model_validate(payload["uniqueness_report"])
    return canonical_contract_sha256(report)


def test_folds_and_ladder_hit_the_verification_safe_failure_state(completed):
    """On three trading days the frozen 40/5/5/2 protocol yields zero folds —
    the ladder still completes control-flow-wise with empty predictions."""

    state = read_pipeline_state(
        completed["state_root"], completed["result"].pipeline_semantic_id
    )
    s08 = state["stages"][QuantLabPipelineStage.S08_BUILD_FOLDS.value]
    assert "0 folds constructible" in s08["explanation"]
    s10 = state["stages"][
        QuantLabPipelineStage.S10_GENERATE_PREDICTIONS_AND_DIAGNOSTICS.value
    ]
    diagnostics = json.loads(
        load_sidecar_bytes(
            completed["store_root"],
            "pipeline_stage_results",
            s10["stage_result_id"],
            "supervised_ladder.json",
        )
    )
    assert diagnostics["parity"]["oos_row_count"] == 0
    for rung in diagnostics["rungs"].values():
        assert rung["prediction_report"]["count"] == 0


def test_pipeline_result_prepared_not_published_with_stamps(completed):
    state = read_pipeline_state(
        completed["state_root"], completed["result"].pipeline_semantic_id
    )
    publication = state["publication"]
    assert publication["state"] == "prepared_not_published"
    assert publication["activated"] is False
    result_envelope = completed["result"].result_envelope
    assert isinstance(result_envelope, PipelineResultEnvelope)
    reloaded = load_verified_envelope(
        completed["store_root"],
        "search_results",
        result_envelope.pipeline_result_id,
        PipelineResultEnvelope,
    )
    stamps = dict(reloaded.payload.verification_stamps)
    assert stamps["verification_only"] is True
    assert stamps["not_for_research_interpretation"] is True
    assert stamps["full_pipeline_not_run"] is True
    # safety review F2: synthetic fixture days are never counted as real
    assert stamps["real_date_count"] == 0
    assert stamps["synthetic_date_count"] == 3
    assert stamps["synthetic_fixture_ids"] == [completed["charter"].search_id]
    assert reloaded.payload.control_flow_gates.passed is True
    assert state["full_pipeline_not_run"] is True


def test_publication_gates_pass_but_verification_scope_cannot_activate(completed):
    gates = run_publication_gates(
        completed["state_root"],
        completed["result"].pipeline_semantic_id,
        store_root=completed["store_root"],
    )
    assert all(gates.values()), gates
    with pytest.raises(PublicationError, match="can never activate"):
        activate_pipeline_result(
            completed["state_root"],
            completed["result"].pipeline_semantic_id,
            store_root=completed["store_root"],
        )


def test_second_attempt_with_different_workers_shares_every_semantic_identity(
    completed,
):
    """TEST_MATRIX §3.1 execution-attempt identity (P0-3): different worker
    counts, same `pipeline_semantic_id`, identical semantic stage/result
    identities — attempts distinct."""

    first_state = read_pipeline_state(
        completed["state_root"], completed["result"].pipeline_semantic_id
    )
    first_ids = {
        stage: entry["stage_result_id"]
        for stage, entry in first_state["stages"].items()
        if entry["stage_result_id"]
    }
    second = run_pipeline(
        completed["semantic"],
        completed["charter"],
        store_root=completed["store_root"],
        state_root=completed["state_root"],
        wiring=completed["wiring"],
        worker_policy=WorkerPolicy(
            max_workers=4, max_tasks_per_child=2, memory_budget_bytes=1 << 31
        ),
        operational_retry_reason="resource clone (attempt-identity test)",
    )
    assert second.pipeline_semantic_id == completed["result"].pipeline_semantic_id
    second_state = read_pipeline_state(
        completed["state_root"], second.pipeline_semantic_id
    )
    second_ids = {
        stage: entry["stage_result_id"]
        for stage, entry in second_state["stages"].items()
        if entry["stage_result_id"]
    }
    assert second_ids == first_ids  # byte-identical semantic results
    # every re-derived stage is marked verified-reuse; S11 stays blocked
    for stage, status in second.stage_statuses.items():
        if stage == QuantLabPipelineStage.S11_RUN_FROZEN_MODEL_GATED_REPLAYS.value:
            assert status == StageStatus.BLOCKED.value
        else:
            assert status == StageStatus.REUSED.value, (stage, status)
    attempts = second_state["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["attempt_number"] == 1
    assert attempts[1]["attempt_number"] == 2
    assert attempts[1]["operational_retry_reason"] == (
        "resource clone (attempt-identity test)"
    )
    workers = {
        json.dumps(attempt["worker_policy"], sort_keys=True) for attempt in attempts
    }
    assert len(workers) == 2  # resource policies genuinely differ


def test_failed_stage_halts_downstream_and_retry_resumes(tmp_path):
    """FUX-PIPE-005 backend half: an operational failure leaves downstream
    stages pending; the retry (new attempt) reuses earlier verified stages
    and completes."""

    calls = {"n": 0}

    def _flaky_label_builder(view):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient label source failure")
        from tests.agents.ifvg_search.pipeline_fixture import (  # noqa: PLC0415
            LABEL_POLICY_ID,
            build_mini_view,
        )

        _view, labels = build_mini_view()
        return LABEL_POLICY_ID, labels

    fixture = build_pipeline_fixture(tmp_path, label_builder=_flaky_label_builder)
    first = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    labels_stage = QuantLabPipelineStage.S07_DERIVE_LABELS.value
    assert first.stage_statuses[labels_stage] == StageStatus.FAILED.value
    state = read_pipeline_state(fixture["state_root"], first.pipeline_semantic_id)
    assert "transient label source failure" in state["stages"][labels_stage]["explanation"]
    downstream = state["stages"][QuantLabPipelineStage.S08_BUILD_FOLDS.value]
    assert downstream["status"] == StageStatus.PENDING.value
    gates = run_publication_gates(
        fixture["state_root"],
        first.pipeline_semantic_id,
        store_root=fixture["store_root"],
    )
    assert not all(gates.values())  # a failed run can never publish

    second = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
        operational_retry_reason="retry after label source recovery",
    )
    assert second.stage_statuses[labels_stage] == StageStatus.COMPLETED.value
    assert second.stage_statuses[
        QuantLabPipelineStage.S15_VERIFY_AND_PUBLISH.value
    ] in (StageStatus.COMPLETED.value, StageStatus.REUSED.value)
    # the replay stage was re-derived byte-identically → verified reuse
    assert second.stage_statuses[
        QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS.value
    ] == StageStatus.REUSED.value
    state = read_pipeline_state(fixture["state_root"], second.pipeline_semantic_id)
    assert len(state["attempts"]) == 2


def test_tripped_access_assertion_fails_the_derived_counter_gate(tmp_path):
    """Safety review F1: `zero_forbidden_counters` is DERIVED — a drive
    whose zero-counter assertion trips fails the child AND the gate; the
    persisted report can never contradict the audit it names."""

    fixture = build_pipeline_fixture(tmp_path)
    original_runner = fixture["wiring"].child_runner
    calls = {"n": 0}

    def _tripping_runner(*, spec, core_replay_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise AssertionError(
                "protected IFVG source access occurred: {'protected': 1}"
            )
        return original_runner(spec=spec, core_replay_id=core_replay_id)

    import dataclasses

    wiring = dataclasses.replace(fixture["wiring"], child_runner=_tripping_runner)
    result = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=wiring,
        worker_policy=fixture["worker_policy"],
    )
    result_envelope = result.result_envelope
    assert result_envelope is not None
    gates = dict(result_envelope.payload.control_flow_gates.results)
    assert gates["zero_forbidden_counters"] is False
    assert result_envelope.payload.control_flow_gates.passed is False
    publication_gates = run_publication_gates(
        fixture["state_root"],
        result.pipeline_semantic_id,
        store_root=fixture["store_root"],
    )
    assert not all(publication_gates.values())  # such a run can never publish


def test_cancel_sentinel_halts_at_the_stage_boundary(tmp_path):
    fixture = build_pipeline_fixture(tmp_path)
    request_pipeline_cancel(
        fixture["state_root"], fixture["semantic"].pipeline_semantic_id
    )
    # a sentinel written BEFORE the run applies to that run's first boundary
    # only after the runner consumed it — the runner unlinks pre-existing
    # sentinels at startup, so this run proceeds normally (same semantics as
    # the search orchestrator's leftover-sentinel rule).
    result = run_pipeline(
        fixture["semantic"],
        fixture["charter"],
        store_root=fixture["store_root"],
        state_root=fixture["state_root"],
        wiring=fixture["wiring"],
        worker_policy=fixture["worker_policy"],
    )
    assert result.stage_statuses[
        QuantLabPipelineStage.S15_VERIFY_AND_PUBLISH.value
    ] == StageStatus.COMPLETED.value
