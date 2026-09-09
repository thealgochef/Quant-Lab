"""S02 resume uses actual serialized strategy metrics and scoped identities."""

from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.dataset import table_content_hash
from alpha_lab.agents.data_infra.ifvg.search.charter import CostPolicy
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import _load_child_evaluation
from alpha_lab.agents.data_infra.ifvg.search.research_executor import run_research_source_stage
from tests.agents.ifvg_search.conftest import make_resolved_trades_frame
from tests.agents.ifvg_search.test_research_subject_data import subject_fixture


@pytest.mark.parametrize("empty", [False, True])
def test_source_stage_resumes_persisted_metrics_with_exact_r_and_tick_size(
    tmp_path, monkeypatch, empty
):
    from alpha_lab.agents.data_infra.ifvg.search import (
        lineage,
        pipeline,
        research_artifacts,
        research_executor,
    )

    subject = subject_fixture()
    trades = make_resolved_trades_frame(subject.evaluation_dates)
    trades["is_warmup"] = False
    if empty:
        trades = trades.iloc[:0].copy()
    core = {RecordTable.EXECUTED_TRADE: trades}
    report = SimpleNamespace(
        payload=SimpleNamespace(
            passed=True,
            audit_disabled_core_table_hashes={
                RecordTable.EXECUTED_TRADE.value: table_content_hash(
                    RecordTable.EXECUTED_TRADE, trades
                )
            },
        )
    )
    monkeypatch.setattr(research_executor, "load_verified_envelope", lambda *a: report)
    monkeypatch.setattr(pipeline, "_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_record_children", lambda *a: None)
    monkeypatch.setattr(
        research_artifacts,
        "save_research_cohort",
        lambda *a, **k: SimpleNamespace(
            research_cohort_id="a" * 64, model_dump=lambda **k: {"fixture": "cohort"}
        ),
    )
    monkeypatch.setattr(
        lineage, "build_native_lineage_map", lambda *a, **k: SimpleNamespace(uniqueness_report=None)
    )
    monkeypatch.setattr(lineage, "persist_lineage_uniqueness", lambda *a: None)
    monkeypatch.setattr(lineage, "serialize_native_lineage_map", lambda *a: {})
    monkeypatch.setattr(
        research_executor,
        "evaluate_strategy_gates",
        lambda *a: SimpleNamespace(passed=False, human_explanation="fixture"),
    )
    preparation = SimpleNamespace(
        v2_tables=core,
        candidate_view=SimpleNamespace(frame=trades[["candidate_id"]]),
        replay_invocations=0,
        label_source_reference={"artifact_id": "b" * 64},
    )
    preparation.prepare = lambda: preparation
    cost = CostPolicy(cost_points_round_turn=0.5, tick_size=0.5)
    context = SimpleNamespace(
        wiring=SimpleNamespace(
            research_subject=subject,
            research_authorization_check=lambda: None,
            research_preparation=preparation,
            cost_points=cost.cost_points_round_turn,
        ),
        children=[
            {
                "core_replay_id": subject.core_replay_id,
                "ordinal": 0,
                "axis_value_ids": {},
                "comparison_role": "baseline",
            }
        ],
        charter=SimpleNamespace(
            search_id="c" * 64,
            payload=SimpleNamespace(
                cost_policy=cost,
                objective_policy=SimpleNamespace(feasibility_gates={}, pareto_objectives=()),
            ),
        ),
        store_root=tmp_path,
        executed_trades_by_child={},
        neutrality_by_child={},
        tables_by_child={},
        metrics_by_child={},
        gates_passed={},
        lineage_sidecars={},
        stage_sidecars={},
    )
    first, _note = run_research_source_stage(context)
    loaded = _load_child_evaluation(tmp_path, context.children[0]["costed_evaluation_id"])
    assert loaded.model_dump(mode="json") == context.metrics_by_child[
        subject.core_replay_id
    ].model_dump(mode="json")
    if not empty:
        assert loaded.planned_vs_realized.planned_rrr == 2.5
        risk = float(trades.iloc[0].risk_ticks)
        assert loaded.planned_vs_realized.cost_r == pytest.approx(0.5 / (risk * 0.5))
    monkeypatch.setattr(
        research_executor,
        "compute_strategy_metrics",
        lambda *a, **k: pytest.fail("resume recomputed metrics"),
    )
    resumed, _note = run_research_source_stage(context)
    assert resumed == first
    assert context.metrics_by_child[subject.core_replay_id] == loaded


def test_trade_cohort_uses_entry_candidates_and_cutoff_preserving_resolution_day(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (
        build_research_executed_trade_table,
        load_executed_trade_table,
        research_trade_cohort_masks,
        save_executed_trade_table,
    )
    from alpha_lab.agents.data_infra.ifvg.search.research_artifacts import save_research_cohort

    subject = subject_fixture()
    trades = make_resolved_trades_frame(("2026-01-13",), trades_per_day=5)
    trades["is_warmup"] = [False, False, False, True, False]
    before, inside, late, warm, unresolved = trades.candidate_id.tolist()
    trades["trading_day"] = ["2026-01-13", "2026-01-14", "2026-01-14", "2026-01-13", "2026-01-14"]
    trades["entry_ts_utc"] = [
        "2026-01-12T21:00:00Z",
        "2026-01-13T21:00:00Z",
        "2026-01-14T21:00:00Z",
        "2026-01-12T20:00:00Z",
        "2026-01-14T21:30:00Z",
    ]
    trades["resolution_ts_utc"] = [
        "2026-01-13T12:00:00Z",
        "2026-01-14T12:00:00Z",
        subject.cutoff_ts_utc,
        "2026-01-13T11:00:00Z",
        None,
    ]
    trades.loc[4, "status"] = "open"
    ids = (inside, late, unresolved)
    masks = research_trade_cohort_masks(
        trades, candidate_ids=ids, cutoff_ts_utc=subject.cutoff_ts_utc
    )
    assert {name: trades.loc[mask, "candidate_id"].tolist() for name, mask in masks.items()} == {
        "included": [inside],
        "warmup": [warm],
        "out_of_cohort": [before],
        "cutoff_censored": [late],
        "unresolved": [unresolved],
    }
    envelope, data = build_research_executed_trade_table(
        subject.core_replay_id,
        trades,
        record_schema_version=2,
        research_subject_id=subject.subject_id,
        evaluation_dates=subject.evaluation_dates,
        candidate_ids=ids,
        cutoff_ts_utc=subject.cutoff_ts_utc,
    )
    save_executed_trade_table(tmp_path, envelope, data)
    loaded = load_executed_trade_table(tmp_path, envelope.executed_trade_table_id)
    assert loaded.frame.candidate_id.tolist() == [inside]
    assert loaded.frame.trading_day.astype(str).tolist() == ["2026-01-14"]
    assert envelope.source_core_table_hash == table_content_hash(RecordTable.EXECUTED_TRADE, trades)
    candidates = pd.DataFrame(
        {
            "candidate_id": trades.candidate_id.tolist(),
            "trading_day": ["2026-01-12", "2026-01-13", "2026-01-14", "2026-01-12", "2026-01-14"],
            "is_warmup": [False, False, False, True, False],
        }
    )
    cohort = save_research_cohort(
        tmp_path,
        subject=subject,
        view=SimpleNamespace(view_id="a" * 64, frame=candidates[candidates.candidate_id.isin(ids)]),
        raw_tables={RecordTable.ENTRY_CANDIDATE: candidates, RecordTable.EXECUTED_TRADE: trades},
        scoped_trades=loaded.frame,
    )
    assert cohort.payload.executed_trade_ids == (trades.iloc[1].trade_id,)
    assert cohort.payload.cutoff_censored_trade_ids == (trades.iloc[2].trade_id,)
    assert cohort.payload.excluded_unresolved_trade_ids == (trades.iloc[4].trade_id,)
    assert cohort.payload.excluded_window_trade_ids == (trades.iloc[0].trade_id,)
