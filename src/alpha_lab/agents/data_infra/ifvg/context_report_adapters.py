"""Pure adapters from immutable IFVG result contracts to report-ready tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = [
    "adapt_actual_execution_report",
    "adapt_candidate_research_report",
    "adapt_feature_coverage_report",
    "adapt_reconciliation_report",
]


def _records(value: Any) -> pd.DataFrame:
    if not isinstance(value, list) or not value:
        return pd.DataFrame()
    return pd.json_normalize(value)


def _flatten_intervals(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    rows: list[dict[str, Any]] = []
    for name, child in sorted(value.items()):
        key = f"{prefix}.{name}" if prefix else str(name)
        if isinstance(child, dict) and {
            "available",
            "lower",
            "upper",
        }.intersection(child):
            rows.append({"interval": key, **child})
        elif isinstance(child, dict):
            rows.extend(_flatten_intervals(child, key))
    return rows


def adapt_candidate_research_report(report: dict[str, Any]) -> dict[str, Any]:
    model = report.get("model") if isinstance(report.get("model"), dict) else {}
    metrics = model.get("metrics") if isinstance(model.get("metrics"), dict) else {}
    threshold_rows = []
    for row in metrics.get("thresholds", ()) or ():
        if not isinstance(row, dict):
            continue
        r_values = row.get("r") if isinstance(row.get("r"), dict) else {}
        threshold_rows.append(
            {
                "threshold": row.get("threshold"),
                "coverage_count": row.get("coverage_count"),
                "coverage_fraction": row.get("coverage_fraction"),
                "gross_r_sum": r_values.get("gross_r_sum"),
                "net_r_sum": r_values.get("net_r_sum"),
                "net_r_mean": r_values.get("net_r_mean"),
            }
        )
    uncertainty_rows = _flatten_intervals(report.get("uncertainty", {}))
    uncertainty_rows.extend(_flatten_intervals(model.get("uncertainty", {}), "model"))
    return {
        "status": report.get(
            "experiment_status",
            model.get("status", metrics.get("status", "unavailable")),
        ),
        "kpis": {
            "candidate_count": int(report.get("candidate_count", 0) or 0),
            "resolved_candidate_count": int(
                report.get("resolved_candidate_count", 0) or 0
            ),
            "censored_candidate_count": int(
                report.get("censored_candidate_count", 0) or 0
            ),
        },
        "labels": pd.DataFrame(
            [
                {"label": key, "count": value}
                for key, value in sorted((report.get("labels") or {}).items())
            ]
        ),
        "censoring": pd.DataFrame(
            [
                {"reason": key, "count": value}
                for key, value in sorted((report.get("censoring") or {}).items())
            ]
        ),
        "metrics": {
            name: metrics.get(name)
            for name in (
                "brier_score",
                "brier_skill_score",
                "log_loss",
                "auc",
                "auc_reason",
                "prevalence",
                "mean_probability",
            )
        },
        "reliability": _records(metrics.get("reliability_bins")),
        "thresholds": pd.DataFrame(threshold_rows),
        "folds": _records(report.get("fold_definitions") or model.get("folds")),
        "uncertainty": pd.DataFrame(uncertainty_rows),
        "feature_importance": _records(model.get("feature_importance")),
    }


def adapt_actual_execution_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "kpis": {
            "eligible_decision_count": int(report.get("eligible_decision_count", 0) or 0),
            "executed_trade_count": int(report.get("executed_trade_count", 0) or 0),
            "resolved_trade_count": int(report.get("resolved_trade_count", 0) or 0),
            "win_rate": report.get("win_rate"),
            "total_realized_r": report.get("total_realized_r"),
            "max_drawdown_r": report.get("max_drawdown_r"),
            "total_realized_dollars": report.get("total_realized_dollars"),
            "max_drawdown_dollars": report.get("max_drawdown_dollars"),
        },
        "equity": _records(report.get("equity")),
        "r_distribution": pd.DataFrame(
            {"realized_r": report.get("realized_r_distribution", ()) or ()}
        ),
        "uncertainty": pd.DataFrame(_flatten_intervals(report.get("uncertainty", {}))),
        "dollar_conversion": report.get("dollar_conversion", {}),
    }


def adapt_feature_coverage_report(report: dict[str, Any]) -> dict[str, Any]:
    features = _records(report.get("features"))
    if not features.empty:
        denominator = int(report.get("candidate_count", 0) or 0)
        features = features.copy()
        features["missing_fraction"] = (
            pd.to_numeric(features["missing_count"], errors="coerce") / denominator
            if denominator
            else None
        )
    return {
        "kpis": {
            "candidate_count": int(report.get("candidate_count", 0) or 0),
            "feature_count": int(report.get("feature_count", 0) or 0),
            "m3_status": report.get("m3_status"),
            "anchor_240m_status": report.get("anchor_240m_status"),
            "primary_tier_contains_240m": bool(
                report.get("primary_tier_contains_240m", False)
            ),
        },
        "features": features,
        "flags": (
            features.loc[
                features.get("constant", pd.Series(False, index=features.index)).fillna(False)
                | features.get(
                    "low_coverage", pd.Series(False, index=features.index)
                ).fillna(False)
            ].copy()
            if not features.empty
            else pd.DataFrame()
        ),
        "pool_summary": pd.DataFrame(
            [
                {"measure": name, **value}
                for name, value in (
                    ("pool_width_ticks", report.get("pool_width_ticks", {})),
                    (
                        "pool_member_separation_ticks",
                        report.get("pool_member_separation_ticks", {}),
                    ),
                )
                if isinstance(value, dict)
            ]
        ),
        "opposing_leg_coverage": pd.DataFrame(
            [
                {"qualifies": key, "count": value}
                for key, value in sorted(
                    (report.get("opposing_leg_qualifying_counts") or {}).items()
                )
            ]
        ),
    }


def adapt_reconciliation_report(report: dict[str, Any]) -> dict[str, Any]:
    reports = report.get("reports") if isinstance(report.get("reports"), dict) else {}
    gates = []
    for name, payload in sorted(reports.items()):
        if not isinstance(payload, dict):
            continue
        gates.append(
            {
                "gate": name.removesuffix("_report.json").removesuffix(".json"),
                "passed": payload.get("passed"),
                "violation_count": len(payload.get("violations", {}) or {}),
            }
        )
    access = reports.get("data_access_audit.json", {})
    access_rows = []
    if isinstance(access, dict):
        for name, value in sorted(access.items()):
            if isinstance(value, (int, float, bool)):
                access_rows.append({"counter": name, "value": value})
    return {
        "passed": bool(report.get("passed", False)),
        "identities": pd.DataFrame(
            [
                {"artifact": "v2", **(report.get("v2") or {})},
                {"artifact": "v3", **(report.get("v3") or {})},
            ]
        ),
        "table_rows": pd.DataFrame(
            [
                {"table": key, "rows": value}
                for key, value in sorted((report.get("context_table_rows") or {}).items())
            ]
        ),
        "gates": pd.DataFrame(gates),
        "access_counters": pd.DataFrame(access_rows),
        "performance": reports.get("performance_report.json", {}),
        "capacity": reports.get("capacity_report.json", {}),
    }
