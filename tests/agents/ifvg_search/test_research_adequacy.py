"""Research execution, statistical adequacy and strategy selection are distinct."""

from types import SimpleNamespace

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.features.mbp1_arrow_schemas import mbp1_window_validity_fields
from alpha_lab.agents.data_infra.ifvg.ml import regime_report_stage
from alpha_lab.agents.data_infra.ifvg.ml.regime_study import RegimeStudyRequest
from alpha_lab.agents.data_infra.ifvg.search.research_artifacts import research_evidence_status
from tests.agents.ifvg_search.test_orchestrator import _charter


def context_for_status(indices=(0, 1)):
    predictions = pd.DataFrame({"fold_index": list(indices)})
    return SimpleNamespace(
        ladder=SimpleNamespace(
            parity={"oos_row_count": len(indices)},
            rungs=(SimpleNamespace(predictions=predictions),),
        ),
        regime={},
        mbp1_evidence={},
    )


def test_execution_does_not_imply_enough_held_out_evidence():
    assert research_evidence_status(context_for_status())["status"] == "evaluable"
    result = research_evidence_status(context_for_status((0,)))
    assert result["status"] == "insufficient_evidence"
    assert result["valid_oos_folds"] == 1
    assert research_evidence_status(context_for_status(()))["oos_rows"] == 0


def test_mbp_all_null_populations_remain_present_but_not_useful():
    context = context_for_status()
    frame = pd.DataFrame(
        {
            "candidate_id": ["train", "test"],
            **{key: [False, False] for key in mbp1_window_validity_fields()},
        }
    )
    context.mbp1_evidence = {"feature_frame": frame}
    context.folds = SimpleNamespace(
        folds=(
            SimpleNamespace(
                valid=True,
                fold_index=0,
                train_candidate_ids=("train",),
                test_candidate_ids=("test",),
            ),
        )
    )
    result = research_evidence_status(context)
    assert result["status"] == "insufficient_evidence"
    assert result["mbp_fold_coverage"][0]["train_rows_with_valid_windows"] == 0
    assert len(frame) == 2


def test_failed_regime_gates_are_not_a_successful_assessment():
    context = context_for_status()
    context.regime["execution"] = SimpleNamespace(
        run=SimpleNamespace(
            assessment=SimpleNamespace(
                payload=SimpleNamespace(
                    gates_passed=False, gate_failures=("stability_unavailable",)
                )
            )
        )
    )
    assert research_evidence_status(context)["status"] == "insufficient_evidence"


def test_real_descriptive_report_keeps_unpromoted_strategy_cohort(tmp_path, monkeypatch):
    core, table, subject = "a" * 64, "b" * 64, "c" * 64
    loaded = SimpleNamespace(
        envelope=SimpleNamespace(
            executed_trade_table_id=table, executed_trade_table_sha256="d" * 64, row_count=3
        ),
        frame=pd.DataFrame(),
    )
    monkeypatch.setattr(regime_report_stage, "_child_table_evidence", lambda *a: (loaded, None))
    monkeypatch.setattr(regime_report_stage, "panel_event_assigner", lambda *a: None)
    monkeypatch.setattr(regime_report_stage, "_delivered_by_s09c", lambda *a: {})
    captured = []

    def report(inputs):
        captured.append(inputs)
        return SimpleNamespace(report_ids=(), reports_by_class={}, refusals={}, delivered_by={})

    monkeypatch.setattr(regime_report_stage, "build_regime_stratified_reports", report)
    request = RegimeStudyRequest(
        input_feature_bundle_key="B0_CORE",
        resolved_input_features=("candidate_risk_ticks",),
        stratified_reporting_requested=True,
        comparison_classes_requested=("cohort_descriptive",),
    )
    context = SimpleNamespace(
        semantic=SimpleNamespace(
            payload=SimpleNamespace(
                regime_study=request, run_scope=SimpleNamespace(value="full_authorized_development")
            )
        ),
        charter=_charter(),
        wiring=SimpleNamespace(
            cost_points=None, research_subject=SimpleNamespace(subject_id=subject)
        ),
        regime={
            "decision": SimpleNamespace(regime_promotion_decision_id="e" * 64),
            "execution": SimpleNamespace(
                protocol=SimpleNamespace(resolved_regime_protocol_id="f" * 64),
                oos_assignment=SimpleNamespace(regime_oos_assignment_id="1" * 64),
            ),
        },
        children=[{"core_replay_id": core, "state": "reused"}],
        gates_passed={},
        store_root=tmp_path,
        frontier_id=None,
    )
    _, record, _ = regime_report_stage.build_reports(context)
    assert core in captured[0].children
    assert record["strategy_selection_gates"] == {core: False}
    assert record["children_skipped"] == {}


def test_real_regime_followup_runs_each_ladder_arm_once(monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.search import pipeline

    context = context_for_status()
    context.semantic = SimpleNamespace(payload=SimpleNamespace(model_protocol_id="headline"))
    context.wiring = SimpleNamespace(research_subject=object())
    context.stage_sidecars, context.state = {}, {}
    context.research_model_run_ids = ["a" * 64]
    challenger = context.ladder
    challenger.ladder_id = "b" * 64
    context.ladder = None
    monkeypatch.setattr(
        pipeline._regime, "regime_request", lambda *a: SimpleNamespace(requires_supervision=True)
    )

    def forbidden(*args):
        raise AssertionError("separate B0 ladder would duplicate the requested controlled arm")

    monkeypatch.setattr(pipeline, "_stage_s09_supervised_ladder", forbidden)
    calls = []

    def controlled(ctx):
        calls.append(True)
        ctx.regime["controlled_study"] = SimpleNamespace(
            baseline=SimpleNamespace(ladder_id="c" * 64), challenger=challenger
        )
        return ("d" * 64,), {"S09c": {}}, "; controlled comparison"

    monkeypatch.setattr(pipeline._regime, "s09_regime_fit", controlled)
    outputs, note = pipeline._stage_s09_train(context)
    assert calls == [True]
    assert context.ladder is challenger
    assert "b" * 64 in outputs and "c" * 64 in outputs
    assert "once per arm" in note
    assert "research_model_runs.json" in context.stage_sidecars
