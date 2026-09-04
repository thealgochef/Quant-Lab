"""UI-3 Phase 3 — Context Research presentation (F-07 full, F-09): the pure
reading / roll-up assembly over the persisted reports, the adapters' reference
fields, the separate-axis coverage chart, the named reliability diagonal, and
the AppTest of the decision-summary-first renderer with detail levels."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from alpha_lab.agents.data_infra.ifvg.context_report_adapters import (  # noqa: E402
    adapt_candidate_research_report,
)
from alpha_lab.agents.data_infra.ifvg.presentation.context_research import (  # noqa: E402
    candidate_readings,
    compatibility_reasons,
    coverage_readings,
    decision_summary,
    execution_readings,
    fold_chips,
    importance_top,
    m3_status_chip,
    reconciliation_readings,
    sample_adequacy_readings,
)
from alpha_lab.agents.data_infra.ifvg.presentation.rollups import RollupSection  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.presentation.status_vocabulary import (  # noqa: E402
    UiStatus,
)
from tests.agents.test_ifvg_lab_tab import _apptest_result  # noqa: E402


def _complete_candidate_report() -> dict:
    report = _apptest_result("complete")["candidate_research_report"]
    metrics = report["model"]["metrics"]
    metrics.update(
        {
            "count": 2,
            "prevalence": 0.5,
            "mean_probability": 0.5,
            "reference_brier_score": 0.25,
            "calibration": {
                "intercept": -0.1,
                "slope": 1.2,
                "available": True,
                "reason": None,
            },
        }
    )
    report["uncertainty"] = {
        "setup_cluster_net_r_mean": {
            "available": True,
            "reason": None,
            "cluster_count": 6,
            "repetitions": 10_000,
            "seed": 7,
            "estimate": 0.2,
            "lower": 0.05,
            "upper": 0.35,
        },
        "trading_day_block_net_r_mean": {
            "available": False,
            "reason": "fewer_than_two_usable_clusters",
            "cluster_count": 1,
            "repetitions": 10_000,
            "seed": 7,
            "lower": None,
            "upper": None,
        },
    }
    report["fold_definitions"] = [
        {"fold_index": 0, "valid": True, "invalid_reason": None, "train_candidate_ids": ["a"] * 40},
        {
            "fold_index": 1,
            "valid": False,
            "invalid_reason": "insufficient_class_coverage",
            "train_candidate_ids": ["a"] * 12,
        },
    ]
    report["model"]["feature_importance"] = [
        {
            "feature": f"ctx_feature_{index}",
            "permutation_importance_mean": 0.01 * index,
            "fold_coverage_fraction": 1.0 if index % 2 else 0.5,
            "permutation_importance_between_fold_variance": 0.001 * index,
            "use": "descriptive_only",
        }
        for index in range(14)
    ]
    return report


def test_adapter_exposes_reference_calibration_and_fold_fields() -> None:
    adapted = adapt_candidate_research_report(_complete_candidate_report())
    assert adapted["metrics"]["reference_brier_score"] == pytest.approx(0.25)
    assert adapted["metrics"]["prevalence"] == pytest.approx(0.5)
    assert adapted["calibration"]["slope"] == pytest.approx(1.2)
    assert adapted["calibration"]["available"] is True
    assert adapted["oos_count"] == 2
    assert adapted["fold_summary"] == {
        "total": 2,
        "valid": 1,
        "invalid_reasons": {"insufficient_class_coverage": 1},
    }
    legacy = adapt_candidate_research_report({"candidate_count": 0, "model": {}})
    assert legacy["calibration"]["available"] is False
    assert legacy["oos_count"] == 0
    assert legacy["fold_summary"] == {"total": 0, "valid": 0, "invalid_reasons": {}}


def test_candidate_readings_follow_the_registry_rules() -> None:
    adapted = adapt_candidate_research_report(_complete_candidate_report())
    readings = candidate_readings(adapted)
    assert readings["brier_score"].status is UiStatus.PASS  # 0.2 beats the 0.25 reference
    assert readings["brier_skill_score"].status is UiStatus.PASS  # 0.1 > 0
    assert readings["auc"].status is UiStatus.INFORMATIONAL  # 0.5 line: direction only
    assert "chance" in readings["auc"].interpretation
    assert readings["log_loss"].status is UiStatus.INFORMATIONAL
    assert readings["calibration_slope"].status is UiStatus.INFORMATIONAL
    assert "+0.200" in readings["calibration_slope"].interpretation
    assert readings["mean_probability"].reference_value == pytest.approx(0.5)
    undefined = adapt_candidate_research_report(
        _apptest_result("undefined_auc")["candidate_research_report"]
    )
    auc = candidate_readings(undefined)["auc"]
    assert auc.status is UiStatus.UNAVAILABLE
    assert "single_class_oos" in auc.interpretation
    absent = candidate_readings(adapt_candidate_research_report({"candidate_count": 0}))
    assert {reading.status for reading in absent.values()} == {UiStatus.UNAVAILABLE}


def test_sample_adequacy_folds_importance_and_intervals() -> None:
    adapted = adapt_candidate_research_report(_complete_candidate_report())
    adequacy = {reading.technical_key: reading for reading in sample_adequacy_readings(adapted)}
    assert adequacy["candidate_count"].status is UiStatus.INFORMATIONAL
    assert adequacy["valid_fold_count"].status is UiStatus.PASS  # one valid fold ≥ 1
    assert adequacy["bootstrap_cluster_count"].status is UiStatus.INCONCLUSIVE  # min over intervals
    assert adequacy["oos_prediction_count"].value == 2
    chips = fold_chips(adapted)
    assert [chip["fold"] for chip in chips] == [0, 1]
    assert chips[0]["status"] is UiStatus.PASS and chips[1]["status"] is UiStatus.INCONCLUSIVE
    assert "insufficient_class_coverage" in chips[1]["reason"]
    assert chips[1]["train_candidates"].status is UiStatus.INCONCLUSIVE  # 12 < 30
    top = importance_top(adapted, n=10)
    assert len(top) == 10
    assert list(top["feature"])[0] == "ctx_feature_13"  # highest first
    assert "fold_coverage_fraction" in top.columns
    assert "permutation_importance_between_fold_variance" in top.columns


def test_decision_summary_orders_the_four_rollups_and_never_greens_unknown_evidence() -> None:
    result = _apptest_result("complete")
    result["candidate_research_report"] = _complete_candidate_report()
    result["reconciliation_audit_report"] = {
        "passed": True,
        "evaluated": True,
        "unevaluated_reports": ["data_access_audit"],
        "reports": {
            "validity_report.json": {"passed": True, "violations": {}},
            "reconciliation_report.json": {"passed": True, "violations": []},
            "data_access_audit.json": {
                "protected_file_opens": 0,
                "file_opens": 4,
                "denied_dates": {},
            },
        },
    }
    summary = decision_summary(result)
    assert [rollup.section for rollup in summary] == [
        RollupSection.DATA_INTEGRITY,
        RollupSection.PROBABILITY_SKILL,
        RollupSection.CALIBRATION,
        RollupSection.STABILITY,
    ]
    by_section = {rollup.section: rollup for rollup in summary}
    assert by_section[RollupSection.DATA_INTEGRITY].status is UiStatus.PASS
    assert by_section[RollupSection.PROBABILITY_SKILL].status is UiStatus.PASS
    assert by_section[RollupSection.CALIBRATION].status is UiStatus.INFORMATIONAL
    # one interval is unavailable (a single usable cluster) → the section is inconclusive
    assert by_section[RollupSection.STABILITY].status is UiStatus.INCONCLUSIVE
    # a legacy pass flag without evaluated gates is never green
    legacy = dict(result, reconciliation_audit_report={"passed": True})
    integrity = decision_summary(legacy)[0]
    assert integrity.status is not UiStatus.PASS
    # no model result → probability skill is inconclusive, never blocked or failed
    empty = _apptest_result("empty")
    assert decision_summary(empty)[1].status is UiStatus.INCONCLUSIVE


def test_reconciliation_execution_and_coverage_readings() -> None:
    from alpha_lab.agents.data_infra.ifvg.context_report_adapters import (
        adapt_actual_execution_report,
        adapt_feature_coverage_report,
        adapt_reconciliation_report,
    )

    reconciliation = adapt_reconciliation_report(
        {
            "passed": False,
            "reports": {
                "validity_report.json": {"passed": False, "violations": {"schema": 1}},
                "capacity_report.json": {
                    "passed": True,
                    "observed": {"terminal_seed_bytes": 1000, "max_transition_bytes": 100},
                    "limits": {"terminal_seed_bytes": 838_860, "max_transition_bytes": 26_214},
                },
                "performance_report.json": {
                    "passed": True,
                    "replay_slowdown_fraction": 0.05,
                    "repeated_run_p95_slowdown_fraction": None,
                },
                "data_access_audit.json": {
                    "protected_file_opens": 0,
                    "file_opens": 3,
                    "denied_dates": {"2026-06-12": 2},
                },
            },
        }
    )
    readings = {r.technical_key: r for r in reconciliation_readings(reconciliation)}
    assert readings["validity_report"].status is UiStatus.FAIL
    assert readings["capacity_report"].status is UiStatus.PASS
    assert readings["terminal_seed_bytes"].status is UiStatus.PASS
    assert readings["terminal_seed_bytes"].reference_value == pytest.approx(838_860)
    assert readings["replay_slowdown_fraction"].status is UiStatus.PASS
    # an unmeasured optional figure is omitted, never fabricated as unavailable
    assert "repeated_run_p95_slowdown_fraction" not in readings
    assert readings["denied_attempt_count"].status is UiStatus.FAIL  # measured nonzero
    assert readings["protected_file_opens"].status is UiStatus.INFORMATIONAL  # policy zero
    assert readings["file_opens"].status is UiStatus.INFORMATIONAL
    execution = execution_readings(
        adapt_actual_execution_report(
            {
                "eligible_decision_count": 5,
                "executed_trade_count": 3,
                "resolved_trade_count": 3,
                "win_rate": 0.667,
                "total_realized_r": 1.5,
                "max_drawdown_r": -1.0,
                "total_realized_dollars": 60.0,
                "max_drawdown_dollars": -20.0,
                "uncertainty": {
                    "trading_day_block_realized_r_mean": {
                        "available": True,
                        "lower": -0.2,
                        "upper": 0.9,
                        "cluster_count": 2,
                    }
                },
            }
        )
    )
    assert execution["win_rate"].display == "66.7%"
    assert execution["max_drawdown_r"].status is UiStatus.INFORMATIONAL  # no gate on this surface
    assert execution["trading_day_block_realized_r_mean"].status is UiStatus.INCONCLUSIVE
    coverage = coverage_readings(
        adapt_feature_coverage_report(
            {
                "candidate_count": 10,
                "feature_count": 2,
                "features": [
                    {
                        "feature": "dense",
                        "non_null_count": 10,
                        "missing_count": 0,
                        "coverage_fraction": 1.0,
                        "unique_non_null_values": 7,
                        "constant": False,
                        "low_coverage": False,
                    },
                    {
                        "feature": "sparse",
                        "non_null_count": 0,
                        "missing_count": 10,
                        "coverage_fraction": 0.0,
                        "unique_non_null_values": 0,
                        "constant": False,
                        "low_coverage": True,
                    },
                ],
                "m3_status": "descriptive_only_no_positive_qualification_coverage",
            }
        )
    )
    assert coverage["feature_count"].value == 2
    per_feature = {row["feature"]: row for row in coverage["features"]}
    assert per_feature["dense"]["status"] is UiStatus.PASS
    assert per_feature["sparse"]["status"] is UiStatus.INCONCLUSIVE  # below the 0.10 rule
    status, text = m3_status_chip("descriptive_only_no_positive_qualification_coverage")
    assert status is UiStatus.WARNING and "descriptive" in text.lower()
    assert m3_status_chip("model_eligible")[0] is UiStatus.INFORMATIONAL


def test_compatibility_reasons_are_words() -> None:
    reconciliation = SimpleNamespace(
        compatible_for_metric_delta=False,
        registered_tier_delta_only=False,
        differing_fields=("cohort", "label", "oos_row_ids"),
    )
    reasons = compatibility_reasons(reconciliation)
    assert reasons == (
        "different observation cohorts",
        "different label targets",
        "different out-of-sample rows",
    )
    tier_only = SimpleNamespace(
        compatible_for_metric_delta=True,
        registered_tier_delta_only=True,
        differing_fields=(),
    )
    assert compatibility_reasons(tier_only) == (
        "only the feature tier differs (a registered tier delta)",
    )


def test_coverage_figure_uses_separate_axes_and_reliability_names_the_diagonal() -> None:
    import ifvg_lab_charts as charts

    rows = [
        {
            "threshold": 0.4,
            "coverage_count": 8,
            "coverage_fraction": 0.8,
            "net_r_sum": 3.0,
            "net_r_mean": 0.375,
        },
        {
            "threshold": 0.6,
            "coverage_count": 3,
            "coverage_fraction": 0.3,
            "net_r_sum": 2.1,
            "net_r_mean": 0.7,
        },
    ]
    figure = charts.build_coverage_figure(rows)
    assert len(figure.data) == 2
    assert figure.data[0].yaxis == "y" and figure.data[1].yaxis == "y2"  # separate scales
    assert figure.layout.yaxis2 is not None
    names = {trace.name for trace in figure.data}
    assert "coverage" in names and "net R" in names
    legacy = charts.build_coverage_figure(
        {"coverage": [{"thr": 0.5, "coverage": 0.5, "n": 4, "mean_net_r": 0.2, "win_rate": 0.5}]}
    )
    assert len(legacy.data) == 2
    reliability = charts.build_reliability_figure(
        pd.DataFrame(
            {"mean_probability": [0.25, 0.75], "observed_rate": [0.2, 0.8], "count": [4, 6]}
        )
    )
    names = [trace.name for trace in reliability.data]
    assert "perfect calibration (diagonal)" in names
    assert "observed rate" in names


@pytest.mark.parametrize("state", ("complete", "empty", "undefined_auc", "censored"))
def test_render_result_leads_with_the_decision_summary(monkeypatch, state: str) -> None:
    apptest = pytest.importorskip("streamlit.testing.v1")
    import ifvg_lab_tab as tab

    result = _apptest_result(state)
    if state == "complete":
        result["candidate_research_report"] = _complete_candidate_report()
    monkeypatch.setattr(tab, "_APPTEST_RESULT", result, raising=False)

    def _app() -> None:
        import ifvg_lab_tab
        import streamlit as st

        ifvg_lab_tab._render_result(st, ifvg_lab_tab._APPTEST_RESULT)

    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    markdown = "\n".join(str(block.value) for block in at.markdown)
    captions = "\n".join(str(block.value) for block in at.caption)
    assert "Decision summary" in markdown
    for section in ("Data integrity", "Probability skill", "Calibration", "Stability"):
        assert section in markdown
    assert "Sample adequacy" in markdown
    assert "Candidate Research" in captions and "Actual Executed Strategy" in captions
    labels = [metric.label for metric in at.metric]
    assert "Candidates" in labels and "Brier score" in labels
    detail = next(r for r in at.radio if r.key.startswith("ifvg_context_v1_detail_"))
    assert detail.options == ["Summary", "Research details", "Technical identity & audit"]
    assert [item.label for item in at.tabs] == [
        "Candidate research",
        "Actual execution",
        "Feature coverage",
        "Reconciliation",
    ]
    if state == "complete":
        assert "✓ Pass" in captions  # Brier beats its reference
        assert "✕ Fail" not in markdown.split("Decision summary")[1].split("Sample adequacy")[0]
    if state == "undefined_auc":
        assert "single_class_oos" in captions
        assert "∅ Unavailable" in captions
    # raw JSON only under Technical identity & audit
    assert not any(e.label.startswith("Raw candidate report") for e in at.expander)
    detail.set_value("Technical identity & audit").run()
    assert not at.exception
    assert any(e.label.startswith("Raw candidate report") for e in at.expander)


def test_run_history_explains_incompatibility_in_words(monkeypatch) -> None:
    apptest = pytest.importorskip("streamlit.testing.v1")
    import ifvg_lab_tab as tab

    result_payload = _apptest_result("empty")
    config = SimpleNamespace(
        feature_tier=SimpleNamespace(value="M2"),
        label=SimpleNamespace(label_family="r_multiple", reward_r=1.0, fixed_stop_ticks=None),
        observation_filters={},
        model_dump=lambda **_kwargs: {"feature_tier": "M2"},
    )
    runs = {
        "a" * 64: SimpleNamespace(
            config=config,
            result=SimpleNamespace(
                run_id="a" * 64, status="complete", model_dump=lambda **_k: result_payload
            ),
            predictions=pd.DataFrame(),
        ),
        "b" * 64: SimpleNamespace(
            config=config,
            result=SimpleNamespace(
                run_id="b" * 64, status="complete", model_dump=lambda **_k: result_payload
            ),
            predictions=pd.DataFrame(),
        ),
    }
    monkeypatch.setattr(
        tab,
        "list_context_run_catalog",
        lambda **_kwargs: [
            {"run_id": "a" * 64, "display_name": "Left"},
            {"run_id": "b" * 64, "display_name": "Right"},
        ],
    )
    monkeypatch.setattr(
        tab, "load_context_experiment_run", lambda run_id, **_kwargs: runs[run_id]
    )
    monkeypatch.setattr(tab, "describe_observation_cohort", lambda _filters: "all")
    monkeypatch.setattr(tab, "context_cohort_label", lambda _cohort: "All development")
    monkeypatch.setattr(
        tab,
        "reconcile_context_runs",
        lambda _left, _right: SimpleNamespace(
            compatible_for_metric_delta=False,
            registered_tier_delta_only=False,
            differing_fields=("label", "folds"),
            model_dump=lambda **_k: {"compatible_for_metric_delta": False},
        ),
    )

    def _app() -> None:
        import ifvg_lab_tab
        import streamlit as st

        ifvg_lab_tab._run_history(st)

    at = apptest.AppTest.from_function(_app, default_timeout=60)
    at.run()
    assert not at.exception
    warnings = "\n".join(str(block.value) for block in at.warning)
    assert "different label targets" in warnings and "different fold schedules" in warnings
    assert "suppressed" in warnings
