"""Prop-gate + frontier seam over the search orchestrator (PHASED R3 gate).

Proves the R3 wiring items on synthetic fixtures: `evaluate_prop_gates`
fail-closed rows over a real `PayoutReliabilityVector` (the cross-package
field contract), the conservative ALL-legs feasibility rule (one failing
simulation rejects the child), worst-firm merge semantics into the frontier
(min per maximize metric, max per minimize metric — proven by a
representative flip fixture), simulator-call scoping to gates-passing
children only, and the explicit exclusion paths (no simulations returned;
prop objective unavailable on at least one simulation).
"""

from __future__ import annotations

from alpha_lab.agents.data_infra.ifvg.search.charter import (
    ResolvedPropGateThresholds,
)
from alpha_lab.agents.data_infra.ifvg.search.failure import FailureReason
from alpha_lab.agents.data_infra.ifvg.search.gates import evaluate_prop_gates
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (
    enumerate_children,
    read_search_state,
    run_search,
)
from alpha_lab.propsim.prop_metrics import PayoutReliabilityVector
from tests.agents.ifvg_search.test_orchestrator import (
    _charter,
    _identity_resolver,
    _runner_factory,
)

_BASE_THRESHOLDS = ResolvedPropGateThresholds(
    minimum_first_payout_probability_60d=0.5,
    maximum_breach_probability_90d=0.35,
    minimum_expected_net_payout_90d=None,
    minimum_p10_net_payout_90d=None,
    maximum_p90_payout_drought_days=None,
    minimum_three_payout_probability=None,
)


def _vector(
    *,
    net_90: float = 50.0,
    breach_90: float = 0.1,
    first_60: float = 0.9,
    drought: float | None = 20.0,
) -> PayoutReliabilityVector:
    """A gate-healthy reliability vector unless a caller degrades a field."""

    return PayoutReliabilityVector(
        first_payout_probability_30d=0.6,
        first_payout_probability_60d=first_60,
        three_payout_probability=0.4,
        payout_probability_per_rolling_30d=0.5,
        median_days_between_payouts=14.0,
        p90_payout_drought_days=drought,
        expected_net_payout_90d=net_90,
        p10_net_payout_90d=net_90 / 2.0,
        breach_probability_90d=breach_90,
        expected_replacement_cost=0.0,
    )


# ---------------------------------------------------------------------------
# evaluate_prop_gates unit rows (fail-closed shape mirrors the strategy gates)
# ---------------------------------------------------------------------------


def test_prop_gates_pass_with_explicit_not_required_rows() -> None:
    report = evaluate_prop_gates(_vector(), _BASE_THRESHOLDS)
    assert report.passed
    assert report.failure_reason is None
    assert len(report.checks) == 6  # every configured threshold is covered
    by_id = {check.gate_id: check for check in report.checks}
    # the four None-configured optional thresholds are explicit passes
    for gate_id in (
        "minimum_expected_net_payout_90d",
        "minimum_p10_net_payout_90d",
        "maximum_p90_payout_drought_days",
        "minimum_three_payout_probability",
    ):
        assert by_id[gate_id].passed
        assert "not required" in by_id[gate_id].explanation
    # the two required thresholds carry observed-vs-threshold explanations
    assert "0.9" in by_id["minimum_first_payout_probability_60d"].explanation
    assert "0.35" in by_id["maximum_breach_probability_90d"].explanation


def test_prop_gates_failure_reason_mapping() -> None:
    breach = evaluate_prop_gates(_vector(breach_90=0.9), _BASE_THRESHOLDS)
    assert not breach.passed
    assert breach.failure_reason is FailureReason.BREACH
    assert "maximum_breach_probability_90d" in breach.human_explanation

    fees_thresholds = _BASE_THRESHOLDS.model_copy(
        update={"minimum_expected_net_payout_90d": 1_000.0}
    )
    fees = evaluate_prop_gates(_vector(net_90=10.0), fees_thresholds)
    assert not fees.passed
    assert fees.failure_reason is FailureReason.FEES

    survival = evaluate_prop_gates(_vector(first_60=0.1), _BASE_THRESHOLDS)
    assert not survival.passed
    assert survival.failure_reason is FailureReason.FUNDED_SURVIVAL


def test_prop_gates_fail_closed_on_missing_observed_value() -> None:
    thresholds = _BASE_THRESHOLDS.model_copy(
        update={"maximum_p90_payout_drought_days": 10}
    )
    report = evaluate_prop_gates(_vector(drought=None), thresholds)
    assert not report.passed
    by_id = {check.gate_id: check for check in report.checks}
    drought = by_id["maximum_p90_payout_drought_days"]
    assert not drought.passed
    assert "unavailable" in drought.explanation
    assert report.failure_reason is FailureReason.FUNDED_SURVIVAL


# ---------------------------------------------------------------------------
# Orchestrator seam: simulator scoping, ALL-legs rule, worst-firm frontier
# ---------------------------------------------------------------------------


def _passing_hashes(charter, count: int) -> set[str]:
    specs = enumerate_children(
        charter,
        identity_resolver=lambda spec: _identity_resolver(spec).core_replay_id,
    )
    ordered = sorted(specs, key=lambda spec: spec.ordinal)
    baseline = [s for s in ordered if s.comparison_role == "baseline"]
    challengers = [s for s in ordered if s.comparison_role == "challenger"]
    chosen = (baseline + challengers)[:count]
    return {spec.resolved_section_config_hash for spec in chosen}


def test_prop_seam_worst_firm_merge_flips_representative(tmp_path) -> None:
    charter = _charter(
        pareto_objectives=("net_expectancy_r", "expected_net_payout_90d"),
        lexicographic_tie_breaks=("core_replay_id",),
    )
    passing = _passing_hashes(charter, 2)
    simulator_calls: list[str] = []
    hashes = sorted(passing)
    # child A: one great firm, one terrible firm -> worst 10
    # child B: two middling firms -> worst 40; B must win under worst-firm
    vectors_by_hash = {
        hashes[0]: {
            "firm_hi": _vector(net_90=100.0),
            "firm_lo": _vector(net_90=10.0),
        },
        hashes[1]: {
            "firm_1": _vector(net_90=50.0),
            "firm_2": _vector(net_90=40.0),
        },
    }

    def _simulator(*, outcome, result):
        section_hash = outcome.spec.resolved_section_config_hash
        simulator_calls.append(section_hash)
        return vectors_by_hash[section_hash]

    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], passing),
        prop_simulator=_simulator,
        stale_lock_seconds=3600.0,
    )
    assert result.phase == "search_complete"
    # the simulator ran exactly once per strategy-gate-passing child
    assert sorted(simulator_calls) == hashes
    passing_children = {
        outcome.spec.resolved_section_config_hash: outcome
        for outcome in result.children
        if outcome.gate_report is not None and outcome.gate_report.passed
    }
    assert set(passing_children) == passing
    child_a = passing_children[hashes[0]]
    child_b = passing_children[hashes[1]]
    assert result.frontier is not None
    # both children are feasible; only B is on the frontier (worst 40 > worst 10
    # with equal strategy metrics). A best-firm merge would flip this to A.
    assert set(result.frontier.feasible_ids) == {
        child_a.core_replay_id,
        child_b.core_replay_id,
    }
    assert result.frontier.frontier_ids == (child_b.core_replay_id,)
    assert (
        result.frontier.development_exploratory_representative_id
        == child_b.core_replay_id
    )
    champions = dict(result.frontier.per_objective_champions)
    assert champions["expected_net_payout_90d"] == child_b.core_replay_id
    # with a simulator wired the prop phases are no longer skip-annotated
    state = read_search_state(tmp_path / "state", charter.search_id)
    assert "prop_simulations" not in state["phase_notes"]
    assert "robustness_passed" in state["phase_notes"]


def test_prop_seam_all_legs_rule_rejects_on_one_failing_firm(tmp_path) -> None:
    charter = _charter(
        pareto_objectives=("net_expectancy_r", "expected_net_payout_90d"),
        lexicographic_tie_breaks=("core_replay_id",),
    )
    passing = _passing_hashes(charter, 2)
    hashes = sorted(passing)
    vectors_by_hash = {
        # child A: healthy on one firm, breaches the gate on the other leg
        hashes[0]: {
            "good": _vector(net_90=100.0),
            "breachy": _vector(net_90=100.0, breach_90=0.9),
        },
        hashes[1]: {"firm_1": _vector(net_90=40.0)},
    }

    def _simulator(*, outcome, result):
        return vectors_by_hash[outcome.spec.resolved_section_config_hash]

    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], passing),
        prop_simulator=_simulator,
        stale_lock_seconds=3600.0,
    )
    outcomes = {
        outcome.spec.resolved_section_config_hash: outcome
        for outcome in result.children
    }
    rejected = outcomes[hashes[0]]
    survivor = outcomes[hashes[1]]
    assert rejected.failure_reason is FailureReason.BREACH
    assert "prop gates failed on breachy" in rejected.explanation
    assert result.frontier is not None
    assert result.frontier.feasible_ids == (survivor.core_replay_id,)
    assert (
        result.frontier.development_exploratory_representative_id
        == survivor.core_replay_id
    )


def test_prop_seam_empty_simulations_exclude_child(tmp_path) -> None:
    charter = _charter(
        pareto_objectives=("net_expectancy_r", "expected_net_payout_90d"),
        lexicographic_tie_breaks=("core_replay_id",),
    )
    passing = _passing_hashes(charter, 1)

    def _simulator(*, outcome, result):
        return {}

    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], passing),
        prop_simulator=_simulator,
        stale_lock_seconds=3600.0,
    )
    assert result.phase == "search_complete"
    assert result.frontier is None
    (excluded,) = [
        outcome
        for outcome in result.children
        if outcome.gate_report is not None and outcome.gate_report.passed
    ]
    assert "no simulations" in excluded.explanation


def test_prop_seam_unavailable_objective_excludes_child(tmp_path) -> None:
    charter = _charter(
        pareto_objectives=("net_expectancy_r", "p90_payout_drought_days"),
        lexicographic_tie_breaks=("core_replay_id",),
    )
    passing = _passing_hashes(charter, 1)

    def _simulator(*, outcome, result):
        return {"firm_1": _vector(drought=None)}

    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], passing),
        prop_simulator=_simulator,
        stale_lock_seconds=3600.0,
    )
    assert result.frontier is None
    (excluded,) = [
        outcome
        for outcome in result.children
        if outcome.gate_report is not None and outcome.gate_report.passed
    ]
    assert "p90_payout_drought_days" in excluded.explanation
    assert "unavailable" in excluded.explanation


def test_prop_simulator_exception_is_contained_per_child(tmp_path) -> None:
    """A crashing simulator excludes THAT child with a sanitized reason —
    the search itself still completes (never a raw traceback / frozen state)."""

    charter = _charter(
        pareto_objectives=("net_expectancy_r", "expected_net_payout_90d"),
        lexicographic_tie_breaks=("core_replay_id",),
    )
    passing = _passing_hashes(charter, 1)

    def _simulator(*, outcome, result):
        raise RuntimeError(r"boom C:\secret\path Traceback (most recent call last)")

    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], passing),
        prop_simulator=_simulator,
        stale_lock_seconds=3600.0,
    )
    assert result.phase == "search_complete"
    assert result.frontier is None
    (excluded,) = [
        outcome
        for outcome in result.children
        if outcome.gate_report is not None and outcome.gate_report.passed
    ]
    assert excluded.failure_reason is FailureReason.REPLAY
    assert "prop simulation failed" in excluded.explanation
    assert "C:\\" not in excluded.explanation  # sanitized
    assert "Traceback" not in excluded.explanation


def test_prop_objective_without_simulator_excludes_explicitly(tmp_path) -> None:
    """A prop-owned pareto objective with NO simulator wired can never be
    supplied: the child is excluded with the explicit reason, never silently
    dropped or wrongly optimized."""

    charter = _charter(
        pareto_objectives=("net_expectancy_r", "expected_net_payout_90d"),
        lexicographic_tie_breaks=("core_replay_id",),
    )
    passing = _passing_hashes(charter, 1)
    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], passing),
        stale_lock_seconds=3600.0,
    )
    assert result.frontier is None
    (excluded,) = [
        outcome
        for outcome in result.children
        if outcome.gate_report is not None and outcome.gate_report.passed
    ]
    assert "expected_net_payout_90d" in excluded.explanation
    assert "unavailable" in excluded.explanation
    state = read_search_state(tmp_path / "state", charter.search_id)
    assert "skipped" in state["phase_notes"]["prop_simulations"]
