"""Pure contract-to-report adapter coverage for the restored IFVG Lab."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_report_adapters import (
    adapt_actual_execution_report,
    adapt_candidate_research_report,
    adapt_feature_coverage_report,
    adapt_reconciliation_report,
)
from alpha_lab.agents.data_infra.ifvg.context_reporting import (
    build_actual_execution_report,
)
from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable


def _complete_candidate_report() -> dict:
    return {
        "candidate_count": 12,
        "resolved_candidate_count": 10,
        "censored_candidate_count": 2,
        "labels": {"0": 4, "1": 6},
        "censoring": {"development_cutoff": 2},
        "uncertainty": {
            "setup_cluster_net_r_mean": {
                "available": True,
                "lower": -0.1,
                "upper": 0.3,
            }
        },
        "model": {
            "metrics": {
                "status": "complete",
                "brier_score": 0.21,
                "brier_skill_score": 0.08,
                "log_loss": 0.61,
                "auc": 0.67,
                "auc_reason": None,
                "reliability_bins": [
                    {
                        "bin": 0,
                        "mean_probability": 0.3,
                        "observed_rate": 0.25,
                        "count": 4,
                    }
                ],
                "thresholds": [
                    {
                        "threshold": 0.5,
                        "coverage_count": 5,
                        "coverage_fraction": 0.5,
                        "r": {
                            "gross_r_sum": 2.0,
                            "net_r_sum": 2.0,
                            "net_r_mean": 0.4,
                        },
                    }
                ],
            },
            "folds": [{"fold_index": 0, "status": "complete"}],
            "feature_importance": [
                {
                    "feature": "ctx_gap_count",
                    "permutation_importance_mean": 0.12,
                    "use": "descriptive_only",
                }
            ],
        },
    }


def test_complete_candidate_adapter_exposes_visual_tables() -> None:
    adapted = adapt_candidate_research_report(_complete_candidate_report())

    assert adapted["status"] == "complete"
    assert adapted["kpis"] == {
        "candidate_count": 12,
        "resolved_candidate_count": 10,
        "censored_candidate_count": 2,
    }
    assert adapted["metrics"]["auc"] == pytest.approx(0.67)
    assert adapted["reliability"].iloc[0]["observed_rate"] == pytest.approx(0.25)
    assert adapted["thresholds"].iloc[0]["net_r_sum"] == pytest.approx(2.0)
    assert adapted["folds"].iloc[0]["fold_index"] == 0
    assert adapted["uncertainty"].iloc[0]["lower"] == pytest.approx(-0.1)
    assert adapted["feature_importance"].iloc[0]["use"] == "descriptive_only"


@pytest.mark.parametrize(
    ("report", "expected_status", "expected_censored", "auc_reason"),
    [
        (
            {
                "candidate_count": 0,
                "censoring": {},
                "model": {"status": "insufficient_class_coverage"},
            },
            "insufficient_class_coverage",
            0,
            None,
        ),
        (
            {
                "candidate_count": 3,
                "resolved_candidate_count": 0,
                "censored_candidate_count": 3,
                "censoring": {"development_cutoff": 3},
                "model": {"status": "insufficient_class_coverage"},
            },
            "insufficient_class_coverage",
            3,
            None,
        ),
        (
            {
                "candidate_count": 20,
                "resolved_candidate_count": 20,
                "censored_candidate_count": 0,
                "model": {
                    "metrics": {
                        "status": "complete",
                        "auc": None,
                        "auc_reason": "single_class_oos",
                    }
                },
            },
            "complete",
            0,
            "single_class_oos",
        ),
    ],
)
def test_candidate_edge_states_remain_truthful(
    report: dict,
    expected_status: str,
    expected_censored: int,
    auc_reason: str | None,
) -> None:
    adapted = adapt_candidate_research_report(report)

    assert adapted["status"] == expected_status
    assert adapted["kpis"]["censored_candidate_count"] == expected_censored
    assert adapted["metrics"]["auc_reason"] == auc_reason


def test_final_descriptive_status_and_invalid_folds_override_model_fallback() -> None:
    adapted = adapt_candidate_research_report(
        {
            "candidate_count": 20,
            "experiment_status": (
                "descriptive_only_no_positive_qualification_coverage"
            ),
            "fold_definitions": [
                {
                    "fold_index": 0,
                    "valid": False,
                    "invalid_reason": "insufficient_class_coverage",
                }
            ],
            "model": {"status": "insufficient_class_coverage"},
        }
    )

    assert adapted["status"].startswith("descriptive_only")
    assert adapted["folds"].iloc[0]["valid"] == False  # noqa: E712


def test_feature_adapter_reports_missingness_flags_m3_and_240m() -> None:
    adapted = adapt_feature_coverage_report(
        {
            "candidate_count": 10,
            "feature_count": 2,
            "features": [
                {
                    "feature": "constant_context",
                    "non_null_count": 10,
                    "missing_count": 0,
                    "coverage_fraction": 1.0,
                    "constant": True,
                    "low_coverage": False,
                },
                {
                    "feature": "sparse_context",
                    "non_null_count": 0,
                    "missing_count": 10,
                    "coverage_fraction": 0.0,
                    "constant": False,
                    "low_coverage": True,
                },
            ],
            "m3_status": "descriptive_only_no_positive_qualification_coverage",
            "anchor_240m_status": "experimental_q40_open",
            "primary_tier_contains_240m": False,
            "pool_width_ticks": {"count": 1, "median": 1.0},
            "pool_member_separation_ticks": {"count": 1, "median": 0.5},
            "opposing_leg_qualifying_counts": {"False": 10},
        }
    )

    features = adapted["features"].set_index("feature")
    assert features.loc["sparse_context", "missing_fraction"] == pytest.approx(1.0)
    assert set(adapted["flags"]["feature"]) == {
        "constant_context",
        "sparse_context",
    }
    assert adapted["kpis"]["m3_status"].startswith("descriptive_only")
    assert adapted["kpis"]["primary_tier_contains_240m"] is False


def test_empty_actual_and_failed_reconciliation_adapters_are_safe() -> None:
    actual = adapt_actual_execution_report({})
    reconciliation = adapt_reconciliation_report(
        {
            "passed": False,
            "reports": {
                "validity_report.json": {
                    "passed": False,
                    "violations": {"schema": 1},
                }
            },
        }
    )

    assert actual["kpis"]["executed_trade_count"] == 0
    assert actual["equity"].empty
    assert reconciliation["passed"] is False
    assert reconciliation["status"] == "fail"
    assert reconciliation["gates"].iloc[0]["violation_count"] == 1
    assert reconciliation["gates"].iloc[0]["status"] == "fail"


def test_unknown_access_evidence_is_never_pass() -> None:
    """UI-1 (plan F-07): a report without an evaluated pass flag is
    UNAVAILABLE; protected counters are policy-enforced zeros — informational,
    never PASS; the roll-up is green only for evaluated passing gates."""

    unevaluated = adapt_reconciliation_report(
        {
            "passed": None,
            "evaluated": False,
            "unevaluated_reports": ["data_access_audit"],
            "reports": {
                "data_access_audit.json": {
                    "protected_file_opens": 0,
                    "file_opens": 12,
                    "denied_dates": {},
                }
            },
        }
    )
    assert unevaluated["passed"] is None
    assert unevaluated["status"] == "unavailable"
    assert unevaluated["evaluated"] is False
    gate = unevaluated["gates"].iloc[0]
    assert gate["gate"] == "data_access_audit"
    assert not bool(gate["evaluated"]) and gate["status"] == "unavailable"
    counters = unevaluated["access_counters"].set_index("counter")
    assert counters.loc["protected_file_opens", "status"] == "informational"
    assert counters.loc["file_opens", "status"] == "informational"
    assert "pass" not in set(counters["status"])
    # a legacy True flag without evidence is still rendered as it was persisted
    legacy = adapt_reconciliation_report({"passed": True})
    assert legacy["passed"] is True and legacy["status"] == "pass"
    assert legacy["evaluated"] is False  # … but it is not an evaluated gate


def test_actual_execution_drawdown_includes_zero_starting_equity() -> None:
    decisions = pd.DataFrame({"decision_id": ["d1", "d2"]})
    trades = pd.DataFrame(
        {
            "trade_id": ["t1", "t2"],
            "trading_day": ["2026-01-02", "2026-01-03"],
            "resolution_ts_utc": pd.to_datetime(
                ["2026-01-02T15:00:00Z", "2026-01-03T15:00:00Z"], utc=True
            ),
            "realized_r": [-1.0, 0.5],
            "realized_ticks": [-4, 2],
        }
    )
    pair = SimpleNamespace(
        v2=SimpleNamespace(
            reference=SimpleNamespace(artifact_id="a" * 64),
            tables={
                RecordTable.ELIGIBLE_DECISION: decisions,
                RecordTable.EXECUTED_TRADE: trades,
            },
            reports={},
        )
    )

    report = build_actual_execution_report(pair)

    assert report["max_drawdown_r"] == pytest.approx(-1.0)
    assert report["max_drawdown_dollars"] == pytest.approx(-20.0)
    assert report["equity"][0]["drawdown_r"] == pytest.approx(-1.0)
    assert report["surface"] == "actual_v2_execution_only"
