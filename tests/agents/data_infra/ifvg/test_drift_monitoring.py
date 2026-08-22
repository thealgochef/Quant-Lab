"""Core drift-report builder tests (ML plan §8; V1 core scope §12)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from alpha_lab.agents.data_infra.ifvg.ml.drift_monitoring import (
    DriftWindow,
    build_drift_report,
)


def _window(label: str, start: str, end: str, count: int) -> DriftWindow:
    return DriftWindow(label=label, date_from=start, date_to=end, sample_count=count)


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(7)
    n = 300
    reference = pd.DataFrame(
        {
            "feature_stable": rng.normal(0.0, 1.0, size=n),
            "feature_shifted": rng.normal(0.0, 1.0, size=n),
            "probability": rng.uniform(0.2, 0.8, size=n),
        }
    )
    reference["target"] = (rng.uniform(size=n) < reference["probability"]).astype(int)
    current = pd.DataFrame(
        {
            "feature_stable": rng.normal(0.0, 1.0, size=n),
            "feature_shifted": rng.normal(2.5, 1.4, size=n),  # a real shift
            "probability": rng.uniform(0.2, 0.8, size=n),
        }
    )
    current["target"] = (rng.uniform(size=n) < current["probability"]).astype(int)
    return reference, current


def test_feature_drift_separates_stable_from_shifted(frames):
    reference, current = frames
    report = build_drift_report(
        reference_frame=reference,
        current_frame=current,
        features=("feature_stable", "feature_shifted"),
        reference_window=_window("reference", "2026-01-05", "2026-02-27", len(reference)),
        current_window=_window("current", "2026-03-02", "2026-04-24", len(current)),
    )
    rows = {row.feature: row for row in report.feature_rows}
    assert rows["feature_shifted"].psi > rows["feature_stable"].psi
    assert rows["feature_shifted"].psi > 0.5
    assert rows["feature_stable"].psi < 0.25
    assert rows["feature_shifted"].wasserstein > rows["feature_stable"].wasserstein
    assert rows["feature_shifted"].ks_pvalue < 0.01


def test_missing_feature_yields_typed_unavailability(frames):
    reference, current = frames
    report = build_drift_report(
        reference_frame=reference,
        current_frame=current,
        features=("not_a_feature",),
        reference_window=_window("reference", "2026-01-05", "2026-02-27", len(reference)),
        current_window=_window("current", "2026-03-02", "2026-04-24", len(current)),
    )
    row = report.feature_rows[0]
    assert row.unavailable_reason == "feature_absent_from_window"
    assert row.psi is None


def test_prediction_and_calibration_sections_populate(frames):
    reference, current = frames
    report = build_drift_report(
        reference_frame=reference,
        current_frame=current,
        features=("feature_stable",),
        reference_window=_window("reference", "2026-01-05", "2026-02-27", len(reference)),
        current_window=_window("current", "2026-03-02", "2026-04-24", len(current)),
    )
    assert report.prediction.unavailable_reason is None
    assert 0.0 < report.prediction.reference_mean < 1.0
    assert report.calibration.reference_brier is not None
    assert report.calibration.current_slope is not None
    assert report.role == "monitoring_only"


def test_regime_occupancy_is_none_without_a_regime_and_rows_with_one(frames):
    reference, current = frames
    windows = {
        "reference_window": _window("reference", "2026-01-05", "2026-02-27", len(reference)),
        "current_window": _window("current", "2026-03-02", "2026-04-24", len(current)),
    }
    without = build_drift_report(
        reference_frame=reference,
        current_frame=current,
        features=("feature_stable",),
        **windows,
    )
    assert without.regime_occupancy is None

    assignments = pd.DataFrame(
        {
            "canonical_reporting_cluster_id": [0, 0, 1, 1, 2, None],
            "valid": [True, True, True, True, True, False],
        }
    )
    shifted = pd.DataFrame(
        {
            "canonical_reporting_cluster_id": [0, 1, 1, 1, 2, 2],
            "valid": [True] * 6,
        }
    )
    with_regime = build_drift_report(
        reference_frame=reference,
        current_frame=current,
        features=("feature_stable",),
        regime_reference_assignments=assignments,
        regime_current_assignments=shifted,
        **windows,
    )
    occupancy = {row.canonical_reporting_cluster_id: row for row in with_regime.regime_occupancy}
    assert set(occupancy) == {0, 1, 2}
    assert occupancy[0].reference_occupancy == pytest.approx(0.4)
    assert occupancy[1].current_occupancy == pytest.approx(0.5)


def test_report_is_deeply_frozen(frames):
    reference, current = frames
    report = build_drift_report(
        reference_frame=reference,
        current_frame=current,
        features=("feature_stable",),
        reference_window=_window("reference", "2026-01-05", "2026-02-27", len(reference)),
        current_window=_window("current", "2026-03-02", "2026-04-24", len(current)),
    )
    with pytest.raises(ValidationError):
        report.role = "something_else"  # type: ignore[misc]
    with pytest.raises(TypeError):
        report.notes["new"] = "mutation"  # type: ignore[index]
