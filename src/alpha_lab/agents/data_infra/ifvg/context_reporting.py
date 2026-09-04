"""Separated reporting surfaces for IFVG context research and actual execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .artifact_io import VerifiedIfvgPair
from .context_contracts import ContextRecordTable
from .context_experiment_contracts import (
    IfvgContextExperimentConfig,
    IfvgContextExperimentResult,
    context_run_identity,
)
from .context_feature_view import (
    CandidateFeatureView,
    apply_observation_filters,
    build_candidate_feature_view,
    features_for_tier,
    m3_cohort_status,
)
from .context_folds import ContextFoldSet, build_context_folds
from .context_labels import ContextLabelDerivation, derive_context_candidate_labels
from .context_model import ContextModelRun, run_context_fold_models
from .context_statistics import (
    binary_prediction_report,
    candidate_uncertainty_report,
    executed_trade_uncertainty_report,
    feature_importance_report,
)
from .contracts import RecordTable

__all__ = [
    "ContextExperimentExecution",
    "execute_context_experiment",
    "build_candidate_research_report",
    "build_actual_execution_report",
    "build_context_feature_coverage_report",
    "build_context_reconciliation_audit_report",
]


@dataclass(frozen=True, slots=True)
class ContextExperimentExecution:
    result: IfvgContextExperimentResult
    view: CandidateFeatureView
    labels: ContextLabelDerivation
    folds: ContextFoldSet
    model_run: ContextModelRun | None


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("__MISSING__").value_counts().sort_index().items()
    }


def build_candidate_research_report(
    labels: pd.DataFrame,
    model_run: ContextModelRun | None,
) -> dict[str, Any]:
    if labels.empty:
        return {
            "surface": "candidate_research_counterfactual",
            "candidate_count": 0,
            "censoring": {},
            "model": {"status": "insufficient_class_coverage"},
        }
    resolved = labels.loc[labels["resolution_available"].fillna(False).astype(bool)]
    gross = pd.to_numeric(resolved["gross_r"], errors="raise")
    net = pd.to_numeric(resolved["net_r"], errors="raise")
    report: dict[str, Any] = {
        "surface": "candidate_research_counterfactual",
        "candidate_count": len(labels),
        "resolved_candidate_count": int(labels["resolution_available"].astype(bool).sum()),
        "censored_candidate_count": int(labels["censored"].astype(bool).sum()),
        "labels": _counts(labels, "label"),
        "censoring": _counts(labels.loc[labels["censored"].astype(bool)], "censor_reason"),
        "label_config_hashes": sorted(labels["label_config_hash"].astype(str).unique()),
        "cost_per_trade_r": sorted(
            pd.to_numeric(labels["cost_per_trade_r"], errors="raise").unique().tolist()
        ),
        "gross_net_r": {
            "count": len(resolved),
            "gross_r_sum": float(gross.sum()),
            "gross_r_mean": float(gross.mean()) if len(gross) else None,
            "net_r_sum": float(net.sum()),
            "net_r_mean": float(net.mean()) if len(net) else None,
        },
        "uncertainty": (
            candidate_uncertainty_report(resolved)
            if not resolved.empty
            else {
                "status": "unavailable",
                "reason": "no_resolved_candidates",
            }
        ),
    }
    if model_run is None:
        report["model"] = {"status": "insufficient_class_coverage"}
        return report
    predictions = model_run.predictions
    report["model"] = {
        "protocol_id": model_run.protocol.protocol_id,
        "resolved_protocol_hash": model_run.protocol.resolved_hash,
        "metrics": binary_prediction_report(predictions),
        "folds": list(model_run.fold_reports),
        "uncertainty": (
            candidate_uncertainty_report(predictions)
            if not predictions.empty
            else {
                "status": "unavailable",
                "reason": "no_oos_predictions",
            }
        ),
        "feature_importance": feature_importance_report(model_run.feature_importance),
        "feature_importance_use": "descriptive_only",
    }
    return report


def build_actual_execution_report(pair: VerifiedIfvgPair) -> dict[str, Any]:
    decisions = pair.v2.tables[RecordTable.ELIGIBLE_DECISION]
    trades = pair.v2.tables[RecordTable.EXECUTED_TRADE]
    source_report = pair.v2.reports.get("executed_trade_report.json", {})
    # The source v2 report is already exact-execution-only.  Keep its payload
    # intact and add only identity/count reconciliation, never counterfactual labels.
    resolved = trades.iloc[0:0].copy()
    if not trades.empty and "realized_r" in trades:
        resolved = trades.loc[
            pd.to_numeric(trades["realized_r"], errors="coerce").notna()
        ].copy()
    uncertainty: dict[str, Any] = {
        "status": "unavailable",
        "reason": "no_resolved_executed_trades",
    }
    if not resolved.empty:
        if "trading_day" not in resolved and "envelope_trading_day" in resolved:
            resolved["trading_day"] = resolved["envelope_trading_day"]
        uncertainty = executed_trade_uncertainty_report(resolved)
    order_columns = [
        column
        for column in ("resolution_ts_utc", "entry_ts_utc", "trace_ordinal")
        if column in resolved
    ]
    if order_columns:
        resolved = resolved.sort_values(order_columns, kind="mergesort")
    realized_r = pd.to_numeric(
        resolved.get("realized_r", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    realized_ticks = pd.to_numeric(
        resolved.get("realized_ticks", pd.Series(dtype=float)), errors="coerce"
    )
    # The verified artifact is NQ on the integer 0.25-point grid. One NQ
    # contract is $20/point, so each source tick is $5. This is a display
    # conversion only; it never changes source execution records or labels.
    realized_dollars = realized_ticks * 5.0
    cumulative_r = realized_r.cumsum()
    cumulative_dollars = realized_dollars.fillna(0.0).cumsum()
    drawdown_r = cumulative_r - cumulative_r.cummax().clip(lower=0.0)
    drawdown_dollars = (
        cumulative_dollars - cumulative_dollars.cummax().clip(lower=0.0)
    )
    equity = []
    for position, (_, row) in enumerate(resolved.iterrows()):
        equity.append(
            {
                "trade_id": str(row.get("trade_id", "")),
                "resolution_ts_utc": row.get("resolution_ts_utc"),
                "realized_r": float(realized_r.iloc[position]),
                "cumulative_r": float(cumulative_r.iloc[position]),
                "drawdown_r": float(drawdown_r.iloc[position]),
                "realized_dollars": (
                    None
                    if pd.isna(realized_dollars.iloc[position])
                    else float(realized_dollars.iloc[position])
                ),
                "cumulative_dollars": float(cumulative_dollars.iloc[position]),
                "drawdown_dollars": float(drawdown_dollars.iloc[position]),
            }
        )
    return {
        "surface": "actual_v2_execution_only",
        "source_v2_artifact_id": pair.v2.reference.artifact_id,
        "eligible_decision_count": len(decisions),
        "executed_trade_count": len(trades),
        "decision_ids": sorted(decisions.get("decision_id", pd.Series(dtype=str)).astype(str)),
        "trade_ids": sorted(trades.get("trade_id", pd.Series(dtype=str)).astype(str)),
        "source_report": source_report,
        "uncertainty": uncertainty,
        "resolved_trade_count": len(resolved),
        "winning_trade_count": int((realized_r > 0).sum()),
        "win_rate": float((realized_r > 0).mean()) if len(realized_r) else None,
        "total_realized_r": float(realized_r.sum()),
        "mean_realized_r": float(realized_r.mean()) if len(realized_r) else None,
        "max_drawdown_r": float(drawdown_r.min()) if len(drawdown_r) else None,
        "total_realized_dollars": (
            float(realized_dollars.sum()) if realized_dollars.notna().any() else None
        ),
        "max_drawdown_dollars": (
            float(drawdown_dollars.min()) if len(drawdown_dollars) else None
        ),
        "dollar_conversion": {
            "instrument": "NQ",
            "contracts": 1,
            "dollars_per_tick": 5.0,
            "use": "display_only_from_source_v2_realized_ticks",
        },
        "equity": equity,
        "realized_r_distribution": [float(value) for value in realized_r],
    }


def _distribution(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty or column not in frame:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "min": float(values.min()),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def build_context_feature_coverage_report(
    pair: VerifiedIfvgPair,
    view: CandidateFeatureView,
    *,
    tier,
) -> dict[str, Any]:
    features = features_for_tier(tier)
    frame = view.frame
    coverage = []
    for feature in features:
        values = frame[feature]
        non_null = values.notna()
        unique = values.loc[non_null].astype(str).nunique()
        coverage.append(
            {
                "feature": feature,
                "non_null_count": int(non_null.sum()),
                "missing_count": int((~non_null).sum()),
                "coverage_fraction": float(non_null.mean()) if len(values) else None,
                "unique_non_null_values": int(unique),
                "constant": bool(non_null.any() and unique <= 1),
                "low_coverage": bool(len(values) and non_null.mean() < 0.10),
            }
        )
    lifecycle = pair.v3.tables[ContextRecordTable.EQUAL_LEVEL_POOL_LIFECYCLE]
    sweeps = pair.v3.tables[ContextRecordTable.EQUAL_LEVEL_SWEEP_LINK]
    primary_has_240 = any("14400s" in feature or "240m" in feature for feature in features)
    return {
        "surface": "feature_coverage",
        "tier": str(getattr(tier, "value", tier)),
        "candidate_count": len(frame),
        "feature_count": len(features),
        "features": coverage,
        "pool_width_ticks": _distribution(
            lifecycle.assign(
                _pool_width=pd.to_numeric(
                    lifecycle["pool_upper_bound_ticks"], errors="coerce"
                )
                - pd.to_numeric(lifecycle["pool_lower_bound_ticks"], errors="coerce")
            )
            if {
                "pool_upper_bound_ticks",
                "pool_lower_bound_ticks",
            }.issubset(lifecycle)
            else lifecycle,
            "_pool_width",
        ),
        "pool_member_separation_ticks": _distribution(
            lifecycle, "pool_absolute_separation_ticks"
        ),
        "opposing_leg_qualifying_counts": _counts(sweeps, "qualifies_opposing_leg"),
        "m3_status": view.m3_status,
        "anchor_240m_status": "experimental_q40_open",
        "primary_tier_contains_240m": primary_has_240,
    }


_RECONCILIATION_REPORT_NAMES: tuple[str, ...] = (
    "reconciliation_report.json",
    "identity_report.json",
    "data_access_audit.json",
    "capacity_report.json",
    "performance_report.json",
    "validity_report.json",
)


def build_context_reconciliation_audit_report(pair: VerifiedIfvgPair) -> dict[str, Any]:
    """UI-1 (plan F-07): ``passed`` is DERIVED from the evaluated gate
    evidence of the pair's persisted reports — ``True`` only when every
    report carrying a Boolean ``passed`` flag passed, ``False`` when any
    evaluated gate failed, ``None`` when no report was evaluated at all.
    Reports without a ``passed`` flag (the data-access audit carries
    counters, not a verdict) are listed as unevaluated, never counted green.
    """

    v3_registry = pair.v3.manifest.get("context_arrow_registry", {})
    reports = {name: pair.v3.reports.get(name, {}) for name in _RECONCILIATION_REPORT_NAMES}
    evaluations: dict[str, bool | None] = {}
    for name, payload in reports.items():
        flag = payload.get("passed") if isinstance(payload, dict) else None
        evaluations[name] = flag if isinstance(flag, bool) else None
    evaluated = {name: flag for name, flag in evaluations.items() if flag is not None}
    unevaluated = tuple(sorted(name for name, flag in evaluations.items() if flag is None))
    return {
        "surface": "reconciliation_and_audit",
        "passed": all(evaluated.values()) if evaluated else None,
        "evaluated": bool(evaluated),
        "evaluated_gate_count": len(evaluated),
        "gate_evaluations": evaluations,
        "unevaluated_reports": unevaluated,
        "v2": {
            "artifact_id": pair.v2.reference.artifact_id,
            "manifest_payload_sha256": pair.v2.reference.manifest_payload_sha256,
            "dataset_schema_version": pair.v2.reference.dataset_schema_version,
            "preparation_status": pair.v2.reference.preparation_status.value,
        },
        "v3": {
            "artifact_id": pair.v3.reference.artifact_id,
            "manifest_payload_sha256": pair.v3.reference.manifest_payload_sha256,
            "dataset_schema_version": pair.v3.reference.dataset_schema_version,
            "feature_formula_version": pair.v3.reference.feature_formula_version,
            "preparation_status": pair.v3.reference.preparation_status.value,
            "arrow_registry": v3_registry,
        },
        "exact_pair_reference": pair.v3.manifest.get("accepted_v2_reference"),
        "context_table_rows": {
            table.value: len(pair.v3.tables[table]) for table in ContextRecordTable
        },
        "reports": reports,
    }


def execute_context_experiment(
    pair: VerifiedIfvgPair,
    config: IfvgContextExperimentConfig,
    bars_1m: pd.DataFrame,
) -> ContextExperimentExecution:
    """Execute one registered tier without search, selection, or promotion."""

    if config.dataset.artifact_pair != pair.reference:
        raise ValueError("experiment config does not reference the verified artifact pair")
    view = build_candidate_feature_view(pair)
    cohort = apply_observation_filters(view, config.observation_filters)
    cohort = cohort.loc[~cohort["is_warmup"].astype(bool)].reset_index(drop=True)
    labels = derive_context_candidate_labels(
        cohort,
        bars_1m,
        config.label,
        cutoff_ts_utc=config.dataset.cutoff_ts_utc,
    )
    evidence_days = tuple(
        str(day)
        for day in (pair.v3.manifest.get("identity", {}).get("evidence_dates") or ())
    )
    if not evidence_days:
        evidence_days = tuple(sorted(cohort["trading_day"].astype(str).unique()))
    folds = build_context_folds(
        labels.labels,
        authorized_trading_days=evidence_days,
        train_days=config.train_days,
        test_days=config.test_days,
        step_days=config.step_days,
        embargo_days=config.embargo_days,
        minimum_train_candidates=config.minimum_train_candidates,
    )
    cohort_view = CandidateFeatureView(
        view_id=view.view_id,
        artifact_pair_hash=view.artifact_pair_hash,
        feature_registry_hash=view.feature_registry_hash,
        frame=cohort,
        tier_features=view.tier_features,
        m3_status=m3_cohort_status(cohort),
    )
    model_run = None
    status = folds.status
    if any(fold.valid for fold in folds.folds):
        if (
            config.feature_tier.value == "M3"
            and cohort_view.m3_status != "model_eligible"
        ):
            status = cohort_view.m3_status
        else:
            model_run = run_context_fold_models(
                view,
                labels.labels,
                folds,
                tier=config.feature_tier,
            )
            status = "complete"
    candidate_report = build_candidate_research_report(labels.labels, model_run)
    candidate_report["experiment_status"] = status
    candidate_report["fold_definitions"] = [
        fold.model_dump(mode="json") for fold in folds.folds
    ]
    execution_report = build_actual_execution_report(pair)
    coverage_report = build_context_feature_coverage_report(
        pair,
        cohort_view,
        tier=config.feature_tier,
    )
    audit_report = build_context_reconciliation_audit_report(pair)
    predictions = model_run.predictions if model_run is not None else pd.DataFrame()
    model_protocol_hash = (
        model_run.protocol.resolved_hash if model_run is not None else None
    )
    run_id = context_run_identity(
        config_hash=config.identity,
        view_id=view.view_id,
        label_derivation_id=labels.derivation_id,
        folds=[fold.model_dump(mode="json") for fold in folds.folds],
        model_protocol_hash=model_protocol_hash,
        predictions=predictions.to_dict("records"),
        status=status,
    )
    result = IfvgContextExperimentResult(
        run_id=run_id,
        config_hash=config.identity,
        view_id=view.view_id,
        label_derivation_id=labels.derivation_id,
        model_protocol_hash=model_protocol_hash,
        status=status,
        folds=folds.folds,
        oos_row_ids=(
            tuple(predictions["oos_row_id"].astype(str))
            if not predictions.empty
            else ()
        ),
        candidate_research_report=candidate_report,
        actual_execution_report=execution_report,
        feature_coverage_report=coverage_report,
        reconciliation_audit_report=audit_report,
        model_report=(candidate_report.get("model") if model_run is not None else None),
    )
    return ContextExperimentExecution(
        result=result,
        view=view,
        labels=labels,
        folds=folds,
        model_run=model_run,
    )
