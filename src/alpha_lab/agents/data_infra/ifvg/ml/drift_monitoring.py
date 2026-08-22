"""Core drift-monitoring report builders (`ML_REGIME_CONTRACT_PLAN.md` §8).

V1 core scope (§12): feature distributions (PSI / KS / Wasserstein),
prediction-distribution drift, calibration drift (per-window Brier +
recalibration intercept/slope), and regime occupancy where a regime
exists. The module exports REPORT BUILDERS ONLY — there is no consumer API
that retrains, disables, promotes, or gates anything from a drift alarm;
V1 automation is structurally impossible. The expanded metric set
(funnel-conversion, payoff-compression, MBP-1 coverage drift) is the
post-V1 regime-expansion release.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from ..search.identities import FrozenContract, ImmutableMap

__all__ = [
    "DriftWindow",
    "FeatureDriftRow",
    "PredictionDriftReport",
    "CalibrationDriftReport",
    "RegimeOccupancyDriftRow",
    "DriftReport",
    "build_drift_report",
]

_PSI_BINS = 10
_EPSILON = 1e-6


class DriftWindow(FrozenContract):
    label: str
    date_from: str
    date_to: str
    sample_count: int


class FeatureDriftRow(FrozenContract):
    feature: str
    psi: float | None
    ks_statistic: float | None
    ks_pvalue: float | None
    wasserstein: float | None
    reference_count: int
    current_count: int
    unavailable_reason: str | None


class PredictionDriftReport(FrozenContract):
    reference_mean: float | None
    current_mean: float | None
    reference_std: float | None
    current_std: float | None
    ks_statistic: float | None
    ks_pvalue: float | None
    wasserstein: float | None
    unavailable_reason: str | None


class CalibrationDriftReport(FrozenContract):
    reference_brier: float | None
    current_brier: float | None
    reference_intercept: float | None
    reference_slope: float | None
    current_intercept: float | None
    current_slope: float | None
    unavailable_reason: str | None


class RegimeOccupancyDriftRow(FrozenContract):
    canonical_reporting_cluster_id: int
    reference_occupancy: float
    current_occupancy: float


class DriftReport(FrozenContract):
    """Descriptive drift evidence — monitoring only, never a control input."""

    role: Literal["monitoring_only"] = "monitoring_only"
    reference_window: DriftWindow
    current_window: DriftWindow
    feature_rows: tuple[FeatureDriftRow, ...]
    prediction: PredictionDriftReport
    calibration: CalibrationDriftReport
    regime_occupancy: tuple[RegimeOccupancyDriftRow, ...] | None
    notes: ImmutableMap[str, str]


def _clean_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)


def _psi(reference: np.ndarray, current: np.ndarray) -> float | None:
    """Population-stability index on reference-quantile bins."""

    if len(reference) < _PSI_BINS or len(current) == 0:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, _PSI_BINS + 1)))
    if len(edges) < 3:
        return None  # a near-constant feature has no informative bins
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)
    reference_share = np.clip(reference_counts / len(reference), _EPSILON, None)
    current_share = np.clip(current_counts / len(current), _EPSILON, None)
    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def _feature_row(
    feature: str, reference: pd.DataFrame, current: pd.DataFrame
) -> FeatureDriftRow:
    if feature not in reference or feature not in current:
        return FeatureDriftRow(
            feature=feature,
            psi=None,
            ks_statistic=None,
            ks_pvalue=None,
            wasserstein=None,
            reference_count=0,
            current_count=0,
            unavailable_reason="feature_absent_from_window",
        )
    reference_values = _clean_numeric(reference[feature])
    current_values = _clean_numeric(current[feature])
    if len(reference_values) == 0 or len(current_values) == 0:
        return FeatureDriftRow(
            feature=feature,
            psi=None,
            ks_statistic=None,
            ks_pvalue=None,
            wasserstein=None,
            reference_count=int(len(reference_values)),
            current_count=int(len(current_values)),
            unavailable_reason="no_numeric_observations",
        )
    ks = stats.ks_2samp(reference_values, current_values)
    return FeatureDriftRow(
        feature=feature,
        psi=_psi(reference_values, current_values),
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        wasserstein=float(stats.wasserstein_distance(reference_values, current_values)),
        reference_count=int(len(reference_values)),
        current_count=int(len(current_values)),
        unavailable_reason=None,
    )


def _prediction_report(
    reference: pd.DataFrame, current: pd.DataFrame
) -> PredictionDriftReport:
    empty = PredictionDriftReport(
        reference_mean=None,
        current_mean=None,
        reference_std=None,
        current_std=None,
        ks_statistic=None,
        ks_pvalue=None,
        wasserstein=None,
        unavailable_reason="no_predictions_in_at_least_one_window",
    )
    if "probability" not in reference or "probability" not in current:
        return empty
    reference_values = _clean_numeric(reference["probability"])
    current_values = _clean_numeric(current["probability"])
    if len(reference_values) == 0 or len(current_values) == 0:
        return empty
    ks = stats.ks_2samp(reference_values, current_values)
    return PredictionDriftReport(
        reference_mean=float(np.mean(reference_values)),
        current_mean=float(np.mean(current_values)),
        reference_std=float(np.std(reference_values, ddof=0)),
        current_std=float(np.std(current_values, ddof=0)),
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        wasserstein=float(stats.wasserstein_distance(reference_values, current_values)),
        unavailable_reason=None,
    )


def _calibration_fit(frame: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    """(brier, intercept, slope) for one window; Nones when undefined."""

    if "probability" not in frame or "target" not in frame:
        return None, None, None
    work = frame[["probability", "target"]].dropna()
    if work.empty:
        return None, None, None
    probability = np.clip(
        pd.to_numeric(work["probability"], errors="raise").to_numpy(dtype=float),
        1e-12,
        1 - 1e-12,
    )
    target = pd.to_numeric(work["target"], errors="raise").astype(int).to_numpy()
    brier = float(np.mean((target - probability) ** 2))
    if len(set(target)) < 2:
        return brier, None, None
    logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10_000)
    model.fit(logits, target)
    return brier, float(model.intercept_[0]), float(model.coef_[0][0])


def _calibration_report(
    reference: pd.DataFrame, current: pd.DataFrame
) -> CalibrationDriftReport:
    reference_brier, reference_intercept, reference_slope = _calibration_fit(reference)
    current_brier, current_intercept, current_slope = _calibration_fit(current)
    unavailable = None
    if reference_brier is None or current_brier is None:
        unavailable = "no_labeled_predictions_in_at_least_one_window"
    return CalibrationDriftReport(
        reference_brier=reference_brier,
        current_brier=current_brier,
        reference_intercept=reference_intercept,
        reference_slope=reference_slope,
        current_intercept=current_intercept,
        current_slope=current_slope,
        unavailable_reason=unavailable,
    )


def _occupancy(assignments: pd.DataFrame) -> dict[int, float]:
    valid = assignments[
        assignments.get("valid", pd.Series(dtype=bool)).astype(bool)
        & assignments["canonical_reporting_cluster_id"].notna()
    ]
    if valid.empty:
        return {}
    counts = valid["canonical_reporting_cluster_id"].astype(int).value_counts()
    total = int(counts.sum())
    return {int(cluster): float(count / total) for cluster, count in counts.items()}


def build_drift_report(
    *,
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    features: tuple[str, ...],
    reference_window: DriftWindow,
    current_window: DriftWindow,
    regime_reference_assignments: pd.DataFrame | None = None,
    regime_current_assignments: pd.DataFrame | None = None,
    notes: dict[str, str] | None = None,
) -> DriftReport:
    """Build the immutable descriptive drift report for two explicit windows.

    Regime occupancy rows appear only when BOTH windows carry regime
    assignments (a regime exists); otherwise the field is ``None`` — never a
    fabricated zero-occupancy table.
    """

    regime_rows: tuple[RegimeOccupancyDriftRow, ...] | None = None
    if (
        regime_reference_assignments is not None
        and regime_current_assignments is not None
    ):
        reference_occupancy = _occupancy(regime_reference_assignments)
        current_occupancy = _occupancy(regime_current_assignments)
        clusters = sorted(set(reference_occupancy) | set(current_occupancy))
        regime_rows = tuple(
            RegimeOccupancyDriftRow(
                canonical_reporting_cluster_id=cluster,
                reference_occupancy=reference_occupancy.get(cluster, 0.0),
                current_occupancy=current_occupancy.get(cluster, 0.0),
            )
            for cluster in clusters
        )
    return DriftReport(
        reference_window=reference_window,
        current_window=current_window,
        feature_rows=tuple(
            _feature_row(feature, reference_frame, current_frame)
            for feature in features
        ),
        prediction=_prediction_report(reference_frame, current_frame),
        calibration=_calibration_report(reference_frame, current_frame),
        regime_occupancy=regime_rows,
        notes=dict(notes or {}),
    )
