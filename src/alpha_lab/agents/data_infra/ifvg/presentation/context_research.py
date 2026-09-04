"""Context Research presentation (UI-3; plan §7 "Context Research") — pure.

The typed readings and section roll-ups of one immutable context run, built
from the persisted reports through the pure adapters
(``context_report_adapters``) and the metric registry — the frozen M0–M3
computation is never touched. Every value is read against a REGISTERED
reference (the prevalence-reference Brier, the 0 skill boundary, the 0.5
chance line, the calibration targets, the walk-forward minimum, the
two-cluster bootstrap rule, the 0.10 low-coverage rule, the report limits,
the measured access counters); unknown or unevaluated evidence is
UNAVAILABLE and a section with unavailable evidence is INCONCLUSIVE — never
PASS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import pandas as pd

from .metric_registry import (
    MetricReading,
    evaluate_gate_flag,
    evaluate_interval,
    evaluate_metric,
    unavailable_reading,
)
from .rollups import RollupSection, SectionRollup, rollup_section
from .status_vocabulary import UiStatus, status_chip

__all__ = [
    "candidate_readings",
    "compatibility_reasons",
    "coverage_readings",
    "decision_summary",
    "execution_readings",
    "fold_chips",
    "importance_top",
    "m3_status_chip",
    "reconciliation_readings",
    "sample_adequacy_readings",
]


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


# ── candidate research (counterfactual) ─────────────────────────────────────


def candidate_readings(adapted: Mapping[str, Any]) -> Mapping[str, MetricReading]:
    """The probability-skill and calibration readings of the candidate report."""

    metrics = dict(adapted.get("metrics") or {})
    calibration = dict(adapted.get("calibration") or {})
    prevalence = _number(metrics.get("prevalence"))
    reference = _number(metrics.get("reference_brier_score"))
    readings: dict[str, MetricReading] = {
        "brier_score": evaluate_metric(
            "brier_score", metrics.get("brier_score"), reference=reference
        ),
        "brier_skill_score": evaluate_metric("brier_skill_score", metrics.get("brier_skill_score")),
        "log_loss": evaluate_metric("log_loss", metrics.get("log_loss")),
    }
    auc = metrics.get("auc")
    if _number(auc) is None:
        reason = metrics.get("auc_reason") or "no out-of-sample predictions"
        readings["auc"] = unavailable_reading("auc", f"AUC is undefined: {reason}")
    else:
        readings["auc"] = evaluate_metric("auc", auc)
    readings["prevalence"] = evaluate_metric("prevalence", prevalence)
    readings["mean_probability"] = evaluate_metric(
        "mean_probability", metrics.get("mean_probability"), reference=prevalence
    )
    readings["reference_brier_score"] = evaluate_metric("reference_brier_score", reference)
    if calibration.get("available"):
        readings["calibration_slope"] = evaluate_metric(
            "calibration_slope", calibration.get("slope")
        )
        readings["calibration_intercept"] = evaluate_metric(
            "calibration_intercept", calibration.get("intercept")
        )
    else:
        reason = str(calibration.get("reason") or "calibration not available")
        readings["calibration_slope"] = unavailable_reading(
            "calibration_slope", f"calibration slope unavailable: {reason}"
        )
        readings["calibration_intercept"] = unavailable_reading(
            "calibration_intercept", f"calibration intercept unavailable: {reason}"
        )
    readings["oos_prediction_count"] = evaluate_metric("oos_prediction_count", metrics.get("count"))
    return MappingProxyType(readings)


def _interval_readings(frame: pd.DataFrame, *, keys: Mapping[str, str]) -> list[MetricReading]:
    readings: list[MetricReading] = []
    if frame is None or frame.empty or "interval" not in frame:
        return readings
    for _, row in frame.iterrows():
        name = str(row.get("interval", ""))
        key = keys.get(name.split(".")[-1])
        if key is None:
            continue
        readings.append(
            evaluate_interval(
                key,
                lower=_number(row.get("lower")),
                upper=_number(row.get("upper")),
                available=bool(row.get("available", False)),
                reason=(None if pd.isna(row.get("reason")) else str(row.get("reason")))
                if "reason" in row
                else None,
                estimate=_number(row.get("estimate")) if "estimate" in row else None,
                sample=(
                    int(row.get("cluster_count"))
                    if "cluster_count" in row and _number(row.get("cluster_count")) is not None
                    else None
                ),
            )
        )
    return readings


_CANDIDATE_INTERVALS: Mapping[str, str] = MappingProxyType(
    {
        "setup_cluster_net_r_mean": "setup_cluster_net_r_mean",
        "trading_day_block_net_r_mean": "trading_day_block_net_r_mean",
    }
)


def sample_adequacy_readings(adapted: Mapping[str, Any]) -> tuple[MetricReading, ...]:
    """The sample-adequacy card: counts, out-of-sample predictions, valid
    folds and the bootstrap clusters — each against its registered rule."""

    kpis = dict(adapted.get("kpis") or {})
    metrics = dict(adapted.get("metrics") or {})
    folds = dict(adapted.get("fold_summary") or {})
    uncertainty = adapted.get("uncertainty")
    clusters: float | None = None
    if isinstance(uncertainty, pd.DataFrame) and "cluster_count" in uncertainty:
        counts = [c for c in (_number(v) for v in uncertainty["cluster_count"]) if c is not None]
        clusters = min(counts) if counts else None
    return (
        evaluate_metric("candidate_count", kpis.get("candidate_count")),
        evaluate_metric("resolved_candidate_count", kpis.get("resolved_candidate_count")),
        evaluate_metric("censored_candidate_count", kpis.get("censored_candidate_count")),
        evaluate_metric("oos_prediction_count", metrics.get("count")),
        evaluate_metric("valid_fold_count", folds.get("valid") if folds else None),
        evaluate_metric("bootstrap_cluster_count", clusters),
    )


def fold_chips(adapted: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One chip per walk-forward fold: valid → PASS, invalid → INCONCLUSIVE
    with its typed reason, plus the fold's training-candidate adequacy."""

    frame = adapted.get("folds")
    chips: list[dict[str, Any]] = []
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return chips
    for _, row in frame.iterrows():
        valid = bool(row.get("valid", False)) if "valid" in row else False
        reason = row.get("invalid_reason") if "invalid_reason" in row else None
        reason_text = "valid" if valid else str(reason or "invalid")
        train_ids = row.get("train_candidate_ids") if "train_candidate_ids" in row else None
        train_count = len(train_ids) if isinstance(train_ids, (list, tuple)) else None
        status = UiStatus.PASS if valid else UiStatus.INCONCLUSIVE
        chips.append(
            {
                "fold": int(row.get("fold_index", len(chips))),
                "status": status,
                "chip": status_chip(status),
                "reason": reason_text,
                "train_candidates": evaluate_metric("fold_train_candidate_count", train_count),
            }
        )
    return chips


def importance_top(adapted: Mapping[str, Any], *, n: int = 10) -> pd.DataFrame:
    """The top-N permutation importances with their fold stability columns."""

    frame = adapted.get("feature_importance")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    columns = [
        column
        for column in (
            "feature",
            "permutation_importance_mean",
            "fold_coverage_fraction",
            "permutation_importance_between_fold_variance",
            "catboost_importance_mean",
        )
        if column in frame.columns
    ]
    ordered = frame.sort_values("permutation_importance_mean", ascending=False, kind="mergesort")
    return ordered[columns].head(n).reset_index(drop=True)


# ── actual execution (source v2) ────────────────────────────────────────────


def execution_readings(adapted: Mapping[str, Any]) -> Mapping[str, MetricReading]:
    kpis = dict(adapted.get("kpis") or {})
    readings: dict[str, MetricReading] = {
        key: evaluate_metric(key, kpis.get(key))
        for key in (
            "eligible_decision_count",
            "executed_trade_count",
            "resolved_trade_count",
            "winning_trade_count",
            "win_rate",
            "total_realized_r",
            "mean_realized_r",
            "max_drawdown_r",
            "total_realized_dollars",
            "max_drawdown_dollars",
        )
        if key in kpis
    }
    for reading in _interval_readings(
        adapted.get("uncertainty"),
        keys={"trading_day_block_realized_r_mean": "trading_day_block_realized_r_mean"},
    ):
        readings[reading.technical_key] = reading
    return MappingProxyType(readings)


# ── feature coverage ────────────────────────────────────────────────────────


def m3_status_chip(status: str | None) -> tuple[UiStatus, str]:
    value = str(status or "")
    if value == "model_eligible":
        return UiStatus.INFORMATIONAL, "Model-eligible: positive qualification coverage exists"
    if value.startswith("descriptive_only"):
        return (
            UiStatus.WARNING,
            "Descriptive only — no positive qualification coverage in this cohort",
        )
    if not value:
        return UiStatus.UNAVAILABLE, "M3 status not persisted"
    return UiStatus.INFORMATIONAL, value


def coverage_readings(adapted: Mapping[str, Any]) -> dict[str, Any]:
    kpis = dict(adapted.get("kpis") or {})
    features = adapted.get("features")
    rows: list[dict[str, Any]] = []
    if isinstance(features, pd.DataFrame) and not features.empty:
        for _, row in features.iterrows():
            reading = evaluate_metric("feature_coverage_fraction", row.get("coverage_fraction"))
            rows.append(
                {
                    "feature": str(row.get("feature")),
                    "coverage": reading,
                    "status": reading.status,
                    "chip": reading.chip,
                    "missing_fraction": _number(row.get("missing_fraction"))
                    if "missing_fraction" in row
                    else None,
                    "constant": bool(row.get("constant", False)),
                    "low_coverage": bool(row.get("low_coverage", False)),
                }
            )
    return {
        "feature_count": evaluate_metric("feature_count", kpis.get("feature_count")),
        "candidate_count": evaluate_metric("candidate_count", kpis.get("candidate_count")),
        "features": rows,
        "m3": m3_status_chip(kpis.get("m3_status")),
    }


# ── reconciliation / audit ──────────────────────────────────────────────────

_GATE_KEYS: Mapping[str, str] = MappingProxyType(
    {
        "validity": "validity_report",
        "reconciliation": "reconciliation_report",
        "identity": "identity_report",
        "capacity": "capacity_report",
        "performance": "performance_report",
        "invariant_audit": "invariant_audit",
    }
)
_PERFORMANCE_KEYS: tuple[str, ...] = (
    "replay_slowdown_fraction",
    "repeated_run_p95_slowdown_fraction",
    "completed_1m_step_p99_ms",
    "multi_timeframe_callback_p99_ms",
)
_CAPACITY_KEYS: tuple[str, ...] = (
    "terminal_state_bytes",
    "terminal_seed_bytes",
    "max_transition_bytes",
    "max_members_per_pool",
)
_ACCESS_COUNTERS: tuple[str, ...] = (
    "protected_path_constructions",
    "protected_metadata_accesses",
    "protected_file_opens",
    "protected_rows_read",
    "path_constructions",
    "metadata_accesses",
    "file_opens",
    "rows_read",
)


def reconciliation_readings(adapted: Mapping[str, Any]) -> tuple[MetricReading, ...]:
    """Gate flags (evaluated only), capacity vs limits, performance vs limits,
    and the access counters (measured denials vs policy zeros)."""

    readings: list[MetricReading] = []
    gates = adapted.get("gates")
    if isinstance(gates, pd.DataFrame) and not gates.empty:
        for _, row in gates.iterrows():
            key = _GATE_KEYS.get(str(row.get("gate")))
            if key is None:
                continue
            evaluated = bool(row.get("evaluated", False))
            readings.append(
                evaluate_gate_flag(
                    key, row.get("passed") if evaluated else None, evaluated=evaluated
                )
            )
    capacity = dict(adapted.get("capacity") or {})
    observed = dict(capacity.get("observed") or {})
    limits = dict(capacity.get("limits") or {})
    for key in _CAPACITY_KEYS:
        if observed.get(key) is not None:
            readings.append(
                evaluate_metric(key, observed.get(key), reference=_number(limits.get(key)))
            )
    performance = dict(adapted.get("performance") or {})
    # an optional figure the report did not measure is omitted (its report's own
    # evaluated flag covers it) — never fabricated as an unavailable reading
    for key in _PERFORMANCE_KEYS:
        if performance.get(key) is not None:
            readings.append(evaluate_metric(key, performance.get(key)))
    access = dict(adapted.get("access") or {})
    if access:
        denied = access.get("denied_dates")
        if isinstance(denied, dict):  # the measured counter, when the audit persists it
            denied_count = float(sum(_number(v) or 0.0 for v in denied.values()))
            readings.append(evaluate_metric("denied_attempt_count", denied_count))
        for key in _ACCESS_COUNTERS:
            if key in access:
                readings.append(evaluate_metric(key, access.get(key)))
    return tuple(readings)


# ── the decision summary ────────────────────────────────────────────────────


def decision_summary(result: Mapping[str, Any]) -> tuple[SectionRollup, ...]:
    """Data integrity → Probability skill → Calibration → Stability."""

    from ..context_report_adapters import (  # noqa: PLC0415 — pure adapters
        adapt_candidate_research_report,
        adapt_reconciliation_report,
    )

    candidate = adapt_candidate_research_report(
        dict(result.get("candidate_research_report") or {})
    )
    reconciliation = adapt_reconciliation_report(
        dict(result.get("reconciliation_audit_report") or {})
    )
    skill = candidate_readings(candidate)
    intervals = _interval_readings(candidate.get("uncertainty"), keys=_CANDIDATE_INTERVALS)
    adequacy = {reading.technical_key: reading for reading in sample_adequacy_readings(candidate)}
    return (
        rollup_section(
            RollupSection.DATA_INTEGRITY,
            reconciliation_readings(reconciliation),
            inspect_next="the gate cards and access counters on the Reconciliation tab",
        ),
        rollup_section(
            RollupSection.PROBABILITY_SKILL,
            [skill["brier_score"], skill["brier_skill_score"], skill["log_loss"], skill["auc"]],
            inspect_next=(
                "the Brier / skill / AUC cards and the reference Brier on the Candidate tab"
            ),
        ),
        rollup_section(
            RollupSection.CALIBRATION,
            [skill["calibration_slope"], skill["calibration_intercept"], skill["mean_probability"]],
            inspect_next="the reliability chart and the calibration targets on the Candidate tab",
        ),
        rollup_section(
            RollupSection.STABILITY,
            [*intervals, adequacy["valid_fold_count"]],
            inspect_next="the bootstrap intervals and the fold validity chips on the Candidate tab",
        ),
    )


# ── run compatibility ───────────────────────────────────────────────────────

_DIFFERENCE_WORDS: Mapping[str, str] = MappingProxyType(
    {
        "artifact_pair": "different artifact pairs",
        "cohort": "different observation cohorts",
        "label": "different label targets",
        "folds": "different fold schedules",
        "model_protocol": "different model protocols",
        "calibration_protocol": "different calibration protocols",
        "bootstrap_protocol": "different bootstrap protocols",
        "oos_row_ids": "different out-of-sample rows",
    }
)


def compatibility_reasons(reconciliation: Any) -> tuple[str, ...]:
    """The run-compatibility verdict in words (plan §7 Context Research)."""

    fields: Sequence[str] = tuple(getattr(reconciliation, "differing_fields", ()) or ())
    reasons = tuple(_DIFFERENCE_WORDS.get(str(field), f"different {field}") for field in fields)
    if not reasons and bool(getattr(reconciliation, "registered_tier_delta_only", False)):
        return ("only the feature tier differs (a registered tier delta)",)
    return reasons
