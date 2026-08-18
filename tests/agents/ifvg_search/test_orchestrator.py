"""Synthetic multi-child orchestration E2E (TEST_MATRIX §3.5; PHASED R2 gate).

The 2×2 grid uses two lawful APPROVED_SEARCH_AXIS dimensions
(``parent_retest_timeout_1m_bars`` × ``entry_near_parent``) under the typed
synthetic authorization marker — implying no owner ratification of anything
(P0-20). Replays are stubbed at the ``child_runner`` seam with fully typed
synthetic tables; identities are deterministic synthetic 64-hex ids. The run
proves: exact enumeration + dedup, generated-profile capability enforcement
before any replay, store-identity reuse (within AND across studies), gate
pass/fail explanations, frontier + insights over the passing child, safe
cancel, lock contention, and resume-after-kill via the stale-heartbeat break.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    SyntheticAuthorizationMarker,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    CostPolicy,
    DatePolicy,
    ObjectivePolicy,
    ResolvedPropGateThresholds,
    ResolvedRobustnessGateThresholds,
    ResolvedStrategyGateThresholds,
    SearchCharterEnvelope,
    SearchCharterPayload,
    SearchMode,
    SimulationProtocol,
    validate_charter,
)
from alpha_lab.agents.data_infra.ifvg.search.failure import FailureReason
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    CoreStrategyReplayIdentity,
    CoreStrategyReplayPayload,
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.insights import render_insight_panel
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (
    SearchLockError,
    enumerate_children,
    read_search_state,
    request_safe_cancel,
    run_search,
)
from alpha_lab.agents.data_infra.ifvg.search.robustness import evaluate_robustness
from alpha_lab.agents.data_infra.ifvg.search.store import has_envelope
from tests.agents.ifvg_search.conftest import (
    SYNTHETIC_DAYS,
    make_resolved_trades_frame,
)

_AXES = {
    "parent_retest_timeout_1m_bars": (
        "parent_retest_timeout_1m_bars.none",
        "parent_retest_timeout_1m_bars.480",
    ),
    "entry_near_parent": (
        "entry_near_parent.disabled",
        "entry_near_parent.enabled_40t",
    ),
}


def _axis_values() -> dict[str, tuple[str, ...]]:
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
        SEARCH_AXIS_REGISTRY_V1,
    )

    values = {}
    for axis in _AXES:
        spec = SEARCH_AXIS_REGISTRY_V1[axis]
        values[axis] = tuple(spec.registered_values[:2])
        assert spec.baseline_value_id == spec.registered_values[0]
    return values


def _charter(*, max_children: int = 4, seed: int = 7) -> SearchCharterEnvelope:
    payload = SearchCharterPayload(
        search_mode=SearchMode.FSM_CONFIG_SEARCH,
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        baseline_section_config_hash="0" * 64,
        axes=_axis_values(),
        locked_invariants_registry_sha256="1" * 64,
        measured_only_fields=(),
        blocked_capabilities=("parent_full_fill_invalidation",),
        authorized_firm_contract_ids=(),
        authorized_risk_policy_ids=(),
        authorized_withdrawal_policy_ids=(),
        objective_policy=ObjectivePolicy(
            feasibility_gates=ResolvedStrategyGateThresholds(
                min_executed_trades=10,
                min_independent_days=3,
                min_session_stability_score=0.5,
            ),
            prop_feasibility_gates=ResolvedPropGateThresholds(
                minimum_first_payout_probability_60d=0.5,
                maximum_breach_probability_90d=0.35,
                minimum_expected_net_payout_90d=None,
                minimum_p10_net_payout_90d=None,
                maximum_p90_payout_drought_days=None,
                minimum_three_payout_probability=None,
            ),
            robustness_gates=ResolvedRobustnessGateThresholds(
                maximum_neighbor_expectancy_degradation_r=None,
                minimum_plateau_width=None,
                maximum_worst_firm_breach_probability_90d=None,
                minimum_time_block_sign_consistency=None,
            ),
            pareto_objectives=("net_expectancy_r", "profit_factor"),
            lexicographic_tie_breaks=("net_expectancy_r", "core_replay_id"),
        ),
        date_policy=DatePolicy(
            replay_dates=SYNTHETIC_DAYS,
            warmup_dates=(),
            access_policy_id="verification_fixed_allowlist_max5_v1",
        ),
        simulation_protocol=SimulationProtocol(
            modes=("historical_closed_trade",),
            stress_scenario_ids=(),
            trade_path_capability_policy_id="path_capability_policy_v1",
            clock_policy_id="simulated_clock_v1",
        ),
        max_child_count=max_children,
        seed=seed,
        cost_policy=CostPolicy(),
        strategy_core_commit="a" * 40,
        quant_lab_commit="b" * 40,
        source_artifact_ids=(),
        owner_authorization=SyntheticAuthorizationMarker(),
    )
    validate_charter(payload, as_of_utc="2026-08-18T00:00:00Z")
    return SearchCharterEnvelope.from_payload(payload)


def _identity_resolver(spec) -> CoreStrategyReplayIdentity:
    """A REAL replay-identity envelope with deterministic synthetic inputs —
    exercises the true store publication/reuse path, not a string shortcut."""

    return CoreStrategyReplayIdentity.from_payload(
        CoreStrategyReplayPayload(
            replay_input_bundle_id=canonical_contract_sha256(
                {"dates": list(SYNTHETIC_DAYS), "fixture": "synthetic_bundle_v1"}
            ),
            quant_lab_replay_source_identity="e" * 64,
            strategy_core_commit="f" * 40,
            strategy_core_source_identity="f" * 64,
            resolved_section_config_hash=spec.resolved_section_config_hash,
            canonical_profile_id=spec.canonical_profile_id,
            warmup_seed_identity="synthetic_cold_start_v1",
            anchor_policy="trading_day_18et_elapsed_v1",
            capture_schema_version=2,
            record_schema_version=2,
        )
    )


def _typed_empty_tables() -> dict:
    trades = make_resolved_trades_frame(SYNTHETIC_DAYS).head(0)
    return {RecordTable.EXECUTED_TRADE: trades}


def _runner_factory(observed: list, passing_section_hashes: set[str]):
    def _runner(*, spec, core_replay_id):
        observed.append((spec.ordinal, core_replay_id))
        if spec.resolved_section_config_hash in passing_section_hashes:
            tables = {
                RecordTable.EXECUTED_TRADE: make_resolved_trades_frame(SYNTHETIC_DAYS)
            }
        else:
            tables = _typed_empty_tables()
        return SimpleNamespace(
            tables=tables, gross_trade_stream_hash=canonical_contract_sha256(
                {"child": core_replay_id}
            )
        )

    return _runner


def test_two_by_two_end_to_end(tmp_path) -> None:
    charter = _charter()
    specs = enumerate_children(
        charter,
        identity_resolver=lambda spec: _identity_resolver(spec).core_replay_id,
    )
    assert len(specs) == 4  # 2×2, zero dedup on distinct sections
    baseline = [spec for spec in specs if spec.comparison_role == "baseline"]
    assert len(baseline) == 1 and baseline[0].capability is None
    challengers = [spec for spec in specs if spec.comparison_role == "challenger"]
    assert all(
        spec.capability is not None and spec.capability.status == "generated_runnable"
        for spec in challengers
    )
    # every generated child gets the canonical study-independent name
    assert all(
        spec.canonical_profile_id.startswith("ifvg_search_profile_")
        for spec in challengers
    )

    observed: list = []
    passing = {baseline[0].resolved_section_config_hash}
    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory(observed, passing),
        stale_lock_seconds=3600.0,
    )
    assert result.phase == "search_complete"
    assert len(result.children) == 4
    assert len(observed) == 4  # four replay invocations, none skipped
    states = {outcome.spec.ordinal: outcome.state for outcome in result.children}
    assert set(states.values()) == {"completed"}
    # every child persisted its membership immutably
    for outcome in result.children:
        assert has_envelope(tmp_path / "store", "core_replays", outcome.core_replay_id) or (
            outcome.state != "completed"
        )
    # gates: the baseline passes, the empty children fail with explanations
    passing_children = [c for c in result.children if c.gate_report and c.gate_report.passed]
    failing_children = [
        c for c in result.children if c.gate_report and not c.gate_report.passed
    ]
    assert len(passing_children) == 1
    assert len(failing_children) == 3
    for child in failing_children:
        assert child.failure_reason is FailureReason.INSUFFICIENT_TRADES
        assert "executed trades" in child.explanation
        assert child.gate_report.human_explanation.startswith("failed")
    # frontier over the single feasible child
    assert result.frontier is not None
    representative = result.frontier.development_exploratory_representative_id
    assert representative == passing_children[0].core_replay_id
    trace_lines = result.frontier.tie_break_trace
    assert any("Development Exploratory Representative" in line for line in trace_lines)

    # insights render deterministically for the representative
    metrics = passing_children[0].metrics
    robustness = evaluate_robustness(
        representative,
        axis_grids={axis: values for axis, values in _axis_values().items()},
        child_positions={
            outcome.core_replay_id: dict(outcome.spec.axis_value_ids)
            for outcome in result.children
        },
        metric_by_child={
            outcome.core_replay_id: (
                outcome.metrics.net_expectancy_r
                if outcome.metrics and outcome.metrics.net_expectancy_r is not None
                else float("nan")
            )
            for outcome in result.children
            if outcome.metrics is not None
        },
    )
    panel = render_insight_panel(
        changed_axis_labels=(),
        child_id=representative,
        metrics=metrics,
        baseline_metrics=None,
        robustness=robustness,
    )
    assert len(panel.insights) >= 7

    # the atomic state file reflects the completed run + the honest skip notes
    state = read_search_state(tmp_path / "state", charter.search_id)
    assert state["phase"] == "search_complete"
    assert len(state["children"]) == 4
    assert "R3" in state["phase_notes"]["prop_simulations"]
    # every evaluated child published its costed evaluation; the frontier is
    # persisted into the frontiers store (F5)
    from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (
        SearchFrontierEnvelope,
        SearchFrontierPayload,
        _child_evaluation_envelope,
    )

    for outcome in result.children:
        evaluation = _child_evaluation_envelope(
            outcome.core_replay_id, charter.payload.cost_policy
        )
        assert has_envelope(
            tmp_path / "store", "costed_evaluations", evaluation.costed_evaluation_id
        )
    frontier_envelope = SearchFrontierEnvelope.from_payload(
        SearchFrontierPayload(search_id=charter.search_id, frontier=result.frontier)
    )
    assert has_envelope(
        tmp_path / "store", "frontiers", frontier_envelope.frontier_id
    )
    # the challenger capabilities are bound to the REAL registry digest (F6)
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import registry_sha256

    assert all(
        spec.capability.registry_hash == registry_sha256()
        for spec in challengers
    )


def test_reuse_within_and_across_studies(tmp_path) -> None:
    charter = _charter()
    observed_first: list = []
    run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory(observed_first, set()),
    )
    assert len(observed_first) == 4

    # the SAME resolved configurations in a DIFFERENT parent study (new seed →
    # new search_id) reuse the stored replays: zero replay invocations.
    second = _charter(seed=11)
    assert second.search_id != charter.search_id
    observed_second: list = []
    result = run_search(
        second,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory(observed_second, set()),
    )
    assert observed_second == []
    assert all(outcome.state == "reused" for outcome in result.children)
    assert all(outcome.replay_invocations == 0 for outcome in result.children)
    # F5/MAJOR-3: reused children RELOAD their published evaluations — gates
    # are evaluated identically to the original run, never silently skipped
    assert all(outcome.metrics is not None for outcome in result.children)
    assert all(outcome.gate_report is not None for outcome in result.children)


def test_repeat_identical_gates_and_frontier_across_full_reuse(tmp_path) -> None:
    """§3.5 repeat-identical at the RESULT level: a fully-reused second run of
    the same charter reproduces the identical gate outcomes and the identical
    persisted frontier (F5)."""

    charter = _charter()
    specs = enumerate_children(
        charter,
        identity_resolver=lambda spec: _identity_resolver(spec).core_replay_id,
    )
    passing = {spec.resolved_section_config_hash for spec in specs}
    first = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], passing),
    )
    assert first.frontier is not None
    assert len(first.frontier.feasible_ids) == 4
    second = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], passing),
    )
    assert all(outcome.state == "reused" for outcome in second.children)
    assert second.frontier == first.frontier  # byte-identical result
    for before, after in zip(first.children, second.children, strict=True):
        assert (before.gate_report is None) == (after.gate_report is None)
        if before.gate_report is not None:
            assert after.gate_report.passed == before.gate_report.passed
            assert after.metrics == before.metrics


def test_failed_neutrality_child_never_publishes(tmp_path) -> None:
    """F1: a runner result carrying a FAILED neutrality report ends the child
    ``failed`` with the invariant reason — no envelope, no membership, and the
    child is NOT reusable on resume."""

    from alpha_lab.agents.data_infra.ifvg.search.child_replay import (
        ChildAuditNeutralityReport,
    )

    charter = _charter()

    def _failed_neutrality_runner(*, spec, core_replay_id):
        return SimpleNamespace(
            tables=_typed_empty_tables(),
            gross_trade_stream_hash=canonical_contract_sha256({"c": core_replay_id}),
            neutrality=ChildAuditNeutralityReport(
                core_replay_id=core_replay_id,
                mechanism="dual_drive_ab_v1",
                audit_disabled_core_table_hashes={},
                audit_enabled_core_table_hashes={},
                tables_equal=False,
                mechanism_evidence_refs=(),
                core_trace_content_hash="9" * 64,
                audit_stamp_referential_integrity=True,
                passed=False,
            ),
        )

    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_failed_neutrality_runner,
    )
    assert all(outcome.state == "failed" for outcome in result.children)
    assert all(
        outcome.failure_reason is FailureReason.INVARIANT
        for outcome in result.children
    )
    for outcome in result.children:
        assert not has_envelope(
            tmp_path / "store", "core_replays", outcome.core_replay_id
        )
    # a rerun REPLAYS them (nothing reusable was published)
    observed: list = []
    rerun = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory(observed, set()),
    )
    assert len(observed) == 4
    assert all(outcome.state == "completed" for outcome in rerun.children)


def test_failed_replay_child_is_typed_and_sanitized(tmp_path) -> None:
    """m4: a raising runner ends the child failed/REPLAY with path-free text."""

    charter = _charter()

    def _raising_runner(*, spec, core_replay_id):
        raise RuntimeError(r"boom at C:\Users\gonza\secret\path.parquet")

    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_raising_runner,
    )
    assert all(outcome.state == "failed" for outcome in result.children)
    assert all(
        outcome.failure_reason is FailureReason.REPLAY for outcome in result.children
    )
    assert all("Users" not in outcome.explanation for outcome in result.children)


def test_enumeration_refuses_blocked_values_even_without_charter_validation(
    tmp_path,
) -> None:
    """F6: a hand-built (never-validated) charter with a blocked value cannot
    enumerate — the value-level authorization re-runs per combination."""

    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
        AXIS_VALUE_REGISTRY_V1,
        SEARCH_AXIS_REGISTRY_V1,
    )

    blocked_value = next(
        value_id
        for value_id, value in AXIS_VALUE_REGISTRY_V1.items()
        if getattr(value, "capability_status", "available") != "available"
        and getattr(value, "axis_technical_key", None) in SEARCH_AXIS_REGISTRY_V1
    )
    blocked_axis = AXIS_VALUE_REGISTRY_V1[blocked_value].axis_technical_key
    charter = _charter()
    hand_built = SearchCharterEnvelope.from_payload(
        charter.payload.model_copy(update={"axes": {blocked_axis: (blocked_value,)}})
    )
    with pytest.raises((PermissionError, ValueError)):
        run_search(
            hand_built,
            store_root=tmp_path / "store",
            state_root=tmp_path / "state",
            identity_resolver=_identity_resolver,
            child_runner=_runner_factory([], set()),
        )


def test_blocked_generated_child_never_reaches_the_runner(tmp_path) -> None:
    charter = _charter()
    specs = enumerate_children(
        charter,
        identity_resolver=lambda spec: _identity_resolver(spec).core_replay_id,
    )
    blocked_hash = next(
        spec.resolved_section_config_hash
        for spec in specs
        if spec.comparison_role == "challenger"
    )

    observed: list = []

    def _blocking_resolver(spec):
        return _identity_resolver(spec)

    # simulate a blocked capability by monkeypatching the enumeration output:
    # re-run run_search with a wrapper runner that fails the test if a blocked
    # child is executed
    import alpha_lab.agents.data_infra.ifvg.search.orchestrator as orchestrator_module

    original = orchestrator_module.evaluate_generated_profile_capability

    def _force_block(**kwargs):
        capability = original(**kwargs)
        if kwargs["axis_value_ids"].get("parent_retest_timeout_1m_bars", "").endswith(
            ".240"
        ):
            return capability.model_copy(
                update={
                    "status": "blocked_owner_decision",
                    "reason": "forced block for the P0-D orchestration test",
                }
            )
        return capability

    orchestrator_module.evaluate_generated_profile_capability = _force_block
    try:
        result = run_search(
            charter,
            store_root=tmp_path / "store",
            state_root=tmp_path / "state",
            identity_resolver=_blocking_resolver,
            child_runner=_runner_factory(observed, set()),
        )
    finally:
        orchestrator_module.evaluate_generated_profile_capability = original

    blocked = [c for c in result.children if c.state == "blocked"]
    ran = [c for c in result.children if c.replay_invocations]
    assert len(blocked) == 2  # the two .240 children
    assert all(c.failure_reason is FailureReason.BLOCKED_AXIS for c in blocked)
    assert all("blocked before replay" in c.explanation for c in blocked)
    blocked_ids = {c.core_replay_id for c in blocked}
    assert blocked_ids.isdisjoint(identifier for _, identifier in observed)
    assert len(ran) == 2
    del blocked_hash


def test_safe_cancel_at_child_boundary(tmp_path) -> None:
    charter = _charter()

    cancelled_after: list = []

    def _cancelling_runner(*, spec, core_replay_id):
        cancelled_after.append(core_replay_id)
        if len(cancelled_after) == 2:
            request_safe_cancel(tmp_path / "state", charter.search_id)
        return SimpleNamespace(
            tables=_typed_empty_tables(),
            gross_trade_stream_hash=canonical_contract_sha256({"c": core_replay_id}),
        )

    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_cancelling_runner,
    )
    assert result.phase == "cancelled"
    assert len(cancelled_after) == 2  # the boundary honored the sentinel
    # the sentinel is CONSUMED when honored (F13) — a later resume is not
    # poisoned into an eternal re-cancel
    assert not (
        tmp_path / "state" / charter.search_id / "cancel.requested"
    ).exists()
    states = [outcome.state for outcome in result.children]
    assert states.count("completed") == 2
    assert states.count("cancelled_at_safe_boundary") == 2
    completed = [c for c in result.children if c.state == "completed"]
    # completed children remain immutable and reusable
    for child in completed:
        assert has_envelope(tmp_path / "store", "core_replays", child.core_replay_id)


def test_lock_contention_and_stale_break_resume(tmp_path) -> None:
    charter = _charter()
    state_root = tmp_path / "state"
    state_root.mkdir(parents=True)
    lock = state_root / f"{charter.search_id}.lock"
    lock.write_text("12345", encoding="ascii")

    # a live (fresh-heartbeat) lock refuses a second runner
    with pytest.raises(SearchLockError, match="already locked"):
        run_search(
            charter,
            store_root=tmp_path / "store",
            state_root=state_root,
            identity_resolver=_identity_resolver,
            child_runner=_runner_factory([], set()),
            stale_lock_seconds=3600.0,
        )

    # resume-after-kill: the orphaned lock's heartbeat is provably stale →
    # broken exactly once, and the search resumes to completion
    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=state_root,
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], set()),
        stale_lock_seconds=0.0,
    )
    assert result.phase == "search_complete"
    assert not lock.exists()


def test_resume_after_kill_reuses_completed_children(tmp_path) -> None:
    charter = _charter()
    observed_first: list = []

    def _crashing_runner(*, spec, core_replay_id):
        observed_first.append(core_replay_id)
        if len(observed_first) == 3:
            raise KeyboardInterrupt  # simulate a kill mid-run
        return SimpleNamespace(
            tables=_typed_empty_tables(),
            gross_trade_stream_hash=canonical_contract_sha256({"c": core_replay_id}),
        )

    with pytest.raises(KeyboardInterrupt):
        run_search(
            charter,
            store_root=tmp_path / "store",
            state_root=tmp_path / "state",
            identity_resolver=_identity_resolver,
            child_runner=_crashing_runner,
        )
    # the killed process left its lock behind; a resume breaks it once stale
    observed_second: list = []
    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory(observed_second, set()),
        stale_lock_seconds=0.0,
    )
    assert result.phase == "search_complete"
    # the two completed-before-kill children were reused, not replayed
    assert len(observed_second) == 2
    reused = [c for c in result.children if c.state == "reused"]
    assert len(reused) == 2


def test_enumeration_dedupes_identical_resolved_identities() -> None:
    charter = _charter()
    specs = enumerate_children(
        charter, identity_resolver=lambda spec: "d" * 64  # all children collide
    )
    assert len(specs) == 1  # deduped on the resolved replay identity


def test_child_ceiling_enforced(tmp_path) -> None:
    charter = _charter()
    constrained = SearchCharterEnvelope.from_payload(
        charter.payload.model_copy(update={"max_child_count": 2})
    )
    with pytest.raises(ValueError, match="exceed the charter ceiling"):
        run_search(
            constrained,
            store_root=tmp_path / "store",
            state_root=tmp_path / "state",
            identity_resolver=_identity_resolver,
            child_runner=_runner_factory([], set()),
        )


def test_state_file_is_valid_json_after_every_phase(tmp_path) -> None:
    charter = _charter()
    run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], set()),
    )
    raw = (tmp_path / "state" / charter.search_id / "search_state.json").read_text(
        encoding="utf-8"
    )
    state = json.loads(raw)
    assert state["schema_version"] == 1
    assert {child["state"] for child in state["children"]} <= {
        "queued",
        "running",
        "completed",
        "failed",
        "reused",
        "cancelled_at_safe_boundary",
        "blocked",
    }
