"""Synthetic pause/recovery and bounded-retention regressions; no market replay."""

from __future__ import annotations

import weakref
from datetime import datetime

import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search import orchestrator as module
from alpha_lab.agents.data_infra.ifvg.search.failure import FailureReason
from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    save_or_reuse_envelope,
)
from tests.agents.ifvg_search.test_orchestrator import (
    _charter,
    _identity_resolver,
    _runner_factory,
)
from tests.agents.ifvg_search.test_prop_seam import _vector


def _specs(charter):
    return module.enumerate_children(
        charter, identity_resolver=lambda spec: _identity_resolver(spec).core_replay_id
    )


def _all_passing(charter):
    return {spec.resolved_section_config_hash for spec in _specs(charter)}


def _assert_same_evaluation(left, right):
    assert right.frontier == left.frontier
    for before, after in zip(left.children, right.children, strict=True):
        assert before.core_replay_id == after.core_replay_id
        assert before.metrics == after.metrics
        assert before.gate_report == after.gate_report


class _WeakResult:
    def __init__(self, result):
        self.tables = result.tables
        self.gross_trade_stream_hash = result.gross_trade_stream_hash


@pytest.mark.parametrize("with_prop", [False, True])
def test_full_result_released_before_next_child_even_with_legacy_prop(tmp_path, with_prop):
    charter = _charter()
    references = []
    table_references = []
    observed = []
    base_runner = _runner_factory(observed, _all_passing(charter))
    prop_calls = []

    def runner(**kwargs):
        assert all(ref() is None for ref in references)
        assert all(ref() is None for ref in table_references)
        result = _WeakResult(base_runner(**kwargs))
        references.append(weakref.ref(result))
        table_references.append(weakref.ref(result.tables[RecordTable.EXECUTED_TRADE]))
        return result

    def simulator(*, outcome, result):
        assert isinstance(result, _WeakResult)  # full legacy object contract survives
        assert result.tables[RecordTable.EXECUTED_TRADE].shape[0] > 0
        prop_calls.append(outcome.core_replay_id)
        return {"firm": _vector()}

    result = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver, child_runner=runner,
        prop_simulator=simulator if with_prop else None,
    )
    assert result.phase == "search_complete"
    assert len(observed) == 4
    assert len(prop_calls) == (4 if with_prop else 0)
    assert all(ref() is None for ref in references)
    assert all(ref() is None for ref in table_references)


@pytest.mark.parametrize("stop_after", [2, 4])
def test_pause_publishes_metrics_and_resume_matches_uninterrupted(tmp_path, stop_after):
    charter = _charter()
    passing = _all_passing(charter)
    control = module.run_search(
        charter, store_root=tmp_path / "control", state_root=tmp_path / "control_state",
        identity_resolver=_identity_resolver, child_runner=_runner_factory([], passing),
    )
    observed = []
    base_runner = _runner_factory(observed, passing)

    def runner(**kwargs):
        result = base_runner(**kwargs)
        if len(observed) == stop_after:
            module.request_safe_cancel(tmp_path / "state", charter.search_id)
        return result

    paused = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver, child_runner=runner,
    )
    assert paused.phase == "cancelled"  # includes a stop during the last child
    assert len(observed) == stop_after
    assert not module.cancellation_requested(tmp_path / "state", charter.search_id)
    for outcome in paused.children[:stop_after]:
        assert outcome.metrics is not None
        assert outcome.gate_report is not None
        evaluation = module._child_evaluation_envelope(
            outcome.core_replay_id, charter.payload.cost_policy
        )
        assert has_envelope(tmp_path / "store", "costed_evaluations",
                            evaluation.costed_evaluation_id)
    saved = module.read_search_state(tmp_path / "state", charter.search_id)
    assert all(child["metrics_available"] and child["gates_evaluated"]
               for child in saved["children"][:stop_after])
    assert saved["attempt_finished_at_utc"] is not None
    resumed_calls = []
    resumed = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory(resumed_calls, passing),
    )
    assert len(resumed_calls) == 4 - stop_after
    _assert_same_evaluation(control, resumed)
    final = module.read_search_state(tmp_path / "state", charter.search_id)
    assert final["started_at_utc"] == saved["started_at_utc"]
    assert final["attempt_started_at_utc"] > saved["attempt_started_at_utc"]
    assert datetime.fromisoformat(final["attempt_finished_at_utc"]) >= datetime.fromisoformat(
        final["attempt_started_at_utc"]
    )


def test_legacy_replays_recover_missing_evaluations_without_replaying(tmp_path):
    charter = _charter()
    passing = _all_passing(charter)
    for spec in _specs(charter):
        save_or_reuse_envelope(tmp_path / "store", "core_replays", _identity_resolver(spec))
    references = []
    loaded = []
    runner = _runner_factory(loaded, passing)

    def loader(**kwargs):
        assert all(ref() is None for ref in references)
        result = _WeakResult(runner(**kwargs))
        references.append(weakref.ref(result))
        return result

    def forbidden_runner(**_kwargs):
        pytest.fail("a saved replay must never be rerun to recover its metrics")

    recovered = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver, child_runner=forbidden_runner,
        result_loader=loader,
    )
    assert len(loaded) == 4
    assert all(child.state == "reused" and child.replay_invocations == 0
               for child in recovered.children)
    assert all(ref() is None for ref in references)
    control = module.run_search(
        charter, store_root=tmp_path / "control", state_root=tmp_path / "control_state",
        identity_resolver=_identity_resolver, child_runner=_runner_factory([], passing),
    )
    _assert_same_evaluation(control, recovered)
    again = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver, child_runner=forbidden_runner,
    )
    _assert_same_evaluation(recovered, again)


@pytest.mark.parametrize("corrupt_loader", [False, True])
def test_reused_missing_or_unverified_metric_evidence_fails_closed(tmp_path, corrupt_loader):
    charter = _charter()
    for spec in _specs(charter):
        save_or_reuse_envelope(tmp_path / "store", "core_replays", _identity_resolver(spec))

    def refused(**_kwargs):
        raise ValueError("persisted table failed its manifest hash verification")

    result = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=lambda **_kwargs: pytest.fail("unexpected replay"),
        result_loader=refused if corrupt_loader else None,
    )
    assert result.frontier is None
    assert all(child.state == "failed" and child.failure_reason is FailureReason.INVARIANT
               for child in result.children)
    assert all(child.metrics is None and child.gate_report is None for child in result.children)
    assert all(child.replay_invocations == 0 for child in result.children)
    assert all("manifest hash" in child.explanation if corrupt_loader
               else "verified result loader" in child.explanation for child in result.children)


@pytest.mark.parametrize("phase", ["underlying_edge_passed", "prop_simulations",
                                  "prop_feasible", "robustness_passed", "frontier_complete"])
def test_stop_honored_at_every_finalization_boundary(tmp_path, monkeypatch, phase):
    charter = _charter()
    checkpoint = module._checkpoint

    def request_after_checkpoint(*args, **kwargs):
        path = checkpoint(*args, **kwargs)
        if args[2] == phase:
            module.request_safe_cancel(tmp_path / "state", charter.search_id)
        return path

    monkeypatch.setattr(module, "_checkpoint", request_after_checkpoint)
    result = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], _all_passing(charter)),
    )
    assert result.phase == "cancelled"
    assert all(child.state == "completed" for child in result.children)
    assert all(child.metrics is not None and child.gate_report is not None
               for child in result.children)
    assert module.read_search_state(tmp_path / "state", charter.search_id)["phase"] == "cancelled"


def test_stop_during_preparation_and_prewarm_launches_no_replays(tmp_path):
    charter = _charter()
    checked = []

    def resolver(spec):
        checked.append(spec.ordinal)
        if len(checked) == 2:
            module.request_safe_cancel(tmp_path / "state", charter.search_id)
        return _identity_resolver(spec)

    result = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=resolver,
        child_runner=lambda **_kwargs: pytest.fail("unexpected replay"),
    )
    assert result.phase == "cancelled"
    assert len(checked) == 2
    state = module.read_search_state(tmp_path / "state", charter.search_id)
    assert state["phase_notes"]["preparation_completed_configurations"] == "2"
    assert state["phase_notes"]["preparation_planned_configurations"] == "4"
    result = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=lambda **_kwargs: pytest.fail("unexpected replay"),
        prewarm=lambda _specs: module.request_safe_cancel(tmp_path / "state", charter.search_id),
    )
    assert result.phase == "cancelled"
    assert all(child.state == "cancelled_at_safe_boundary" for child in result.children)


def test_identity_preparation_runs_once_per_section(tmp_path):
    charter = _charter()
    checked = []

    def resolver(spec):
        checked.append(spec.resolved_section_config_hash)
        return _identity_resolver(spec)

    module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=resolver, child_runner=_runner_factory([], set()),
    )
    assert len(checked) == len(set(checked)) == 4


def test_new_stop_after_resume_before_worker_start_is_not_erased(tmp_path):
    charter = _charter()
    module._checkpoint(tmp_path / "state", charter.search_id, "cancelled", [])
    module.request_safe_cancel(tmp_path / "state", charter.search_id)
    result = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=lambda _spec: pytest.fail("new stop must precede preparation"),
        child_runner=lambda **_kwargs: pytest.fail("unexpected replay"),
    )
    assert result.phase == "cancelled"
    assert not module.cancellation_requested(tmp_path / "state", charter.search_id)


def test_preparation_stop_preserves_saved_completed_cohort_and_metric_flags(tmp_path):
    charter = _charter()
    passing = _all_passing(charter)
    first_calls = []
    base_runner = _runner_factory(first_calls, passing)

    def runner(**kwargs):
        result = base_runner(**kwargs)
        if len(first_calls) == 2:
            module.request_safe_cancel(tmp_path / "state", charter.search_id)
        return result

    module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver, child_runner=runner,
    )

    def stop_after_preparation(_specs):
        state = module.read_search_state(tmp_path / "state", charter.search_id)
        assert [row["state"] for row in state["children"]].count("completed") == 2
        assert all(row["metrics_available"] and row["gates_evaluated"]
                   for row in state["children"][:2])
        module.request_safe_cancel(tmp_path / "state", charter.search_id)

    paused = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=lambda **_kwargs: pytest.fail("preparation stop must not replay"),
        prewarm=stop_after_preparation,
    )
    assert paused.phase == "cancelled"
    state = module.read_search_state(tmp_path / "state", charter.search_id)
    assert [row["state"] for row in state["children"]] == [
        "completed", "completed", "cancelled_at_safe_boundary", "cancelled_at_safe_boundary"
    ]
    assert all(row["metrics_available"] and row["gates_evaluated"]
               for row in state["children"][:2])
    later_calls = []
    final = module.run_search(
        charter, store_root=tmp_path / "store", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver, child_runner=_runner_factory(later_calls, passing),
    )
    assert len(later_calls) == 2
    control = module.run_search(
        charter, store_root=tmp_path / "control", state_root=tmp_path / "control_state",
        identity_resolver=_identity_resolver, child_runner=_runner_factory([], passing),
    )
    _assert_same_evaluation(control, final)


def test_prop_loader_releases_each_result_and_matches_legacy_vectors(tmp_path):
    charter = _charter(pareto_objectives=("net_expectancy_r", "expected_net_payout_90d"))
    passing = _all_passing(charter)
    references = []
    simulated = []

    def simulator(*, outcome, result):
        assert result.tables[RecordTable.EXECUTED_TRADE].shape[0] > 0
        simulated.append(outcome.core_replay_id)
        return {"firm": _vector(net_90=50 + outcome.spec.ordinal)}

    control = module.run_search(
        charter, store_root=tmp_path / "control", state_root=tmp_path / "control_state",
        identity_resolver=_identity_resolver, child_runner=_runner_factory([], passing),
        prop_simulator=simulator,
    )
    base_loader = _runner_factory([], passing)

    def loader(**kwargs):
        assert all(ref() is None for ref in references)
        result = _WeakResult(base_loader(**kwargs))
        references.append(weakref.ref(result))
        return result

    # Existing exact evaluations need no table loading until the prop stage.
    again = module.run_search(
        charter, store_root=tmp_path / "control", state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=lambda **_kwargs: pytest.fail("unexpected replay"),
        prop_simulator=simulator, result_loader=loader,
    )
    assert len(references) == 4
    assert len(simulated) == 8
    assert all(ref() is None for ref in references)
    _assert_same_evaluation(control, again)
