"""Actual labeled-fold accounting without preparation or fitting."""

from dataclasses import replace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_evidence import (
    authorized_session_span_ns,
)
from alpha_lab.agents.data_infra.ifvg.search.research_readiness import (
    build_research_fold_readiness,
    summarize_research_folds,
)


def _labels():
    days = tuple(pd.bdate_range("2026-01-05", periods=55).strftime("%Y-%m-%d"))
    rows = []
    for index, day in enumerate(days):
        if index == 40:
            continue  # The first test day is still part of the logical calendar.
        for offset in range(2):
            entry = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=15, minutes=offset)
            rows.append({
                "candidate_id": f"candidate_{index}_{offset}",
                "setup_id": f"setup_{index}_{offset}",
                "trading_day": day,
                "entry_ts_utc": entry,
                "resolution_ts_utc": entry + pd.Timedelta(minutes=1),
                "entry_available": True,
                "resolution_available": True,
                "binary_target": offset,
                "censored": False,
                "feature": float(index),
            })
    labels = pd.DataFrame(rows)
    for candidate in ("candidate_3_0", "candidate_42_0"):
        mask = labels.candidate_id == candidate
        labels.loc[mask, ["binary_target", "resolution_ts_utc"]] = None
        labels.loc[mask, "resolution_available"] = False
        labels.loc[mask, "censored"] = True
    labels.loc[labels.candidate_id.isin(("candidate_5_0", "candidate_41_0")), "setup_id"] = (
        "boundary_setup"
    )
    boundary, _end = authorized_session_span_ns(days[40])
    labels.loc[labels.candidate_id == "candidate_20_0", "resolution_ts_utc"] = pd.Timestamp(
        boundary, tz="UTC"
    )
    return labels, days


def test_actual_strict_calendar_fold_counts_reconcile_without_fitting(monkeypatch):
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer

    def no_fit(*args, **kwargs):
        raise AssertionError("readiness must not fit")

    monkeypatch.setattr(KMeans, "fit", no_fit)
    monkeypatch.setattr(SimpleImputer, "fit", no_fit)
    labels, days = _labels()
    folds, facts = build_research_fold_readiness(
        labels, authorized_trading_days=days, resolved_input_features=("feature",)
    )
    first = facts["per_fold"][0]
    assert first["train"] == {
        "raw_count": 80, "unavailable_or_censored_count": 1, "censored_count": 1,
        "boundary_setup_excluded_count": 1, "purged_count": 1, "embargoed_count": 4,
        "final_count": 73, "class_counts": {"0": 35, "1": 38}, "counts_reconcile": True,
    }
    assert first["test"]["raw_count"] == 8
    assert first["test"]["final_count"] == 6
    assert first["test"]["class_counts"] == {"0": 2, "1": 4}
    assert first["valid"] and first["invalid_reason"] is None
    assert first["training_prevalence"] == pytest.approx(38 / 73)
    assert folds.folds[0].purged_candidate_ids == ("candidate_20_0",)
    assert facts["days_without_candidates"] == [days[40]]
    assert facts["calendar_day_count"] == 55
    assert facts["candidate_regime_sample_adequacy"]["observed_minimum_training_rows"] == 73
    assert not facts["statistical_power_established"]


def test_corrupt_fold_populations_refuse_instead_of_reporting_reconciled():
    labels, days = _labels()
    folds, _facts = build_research_fold_readiness(labels, authorized_trading_days=days)
    first = folds.folds[0]
    incorrect = first.model_copy(update={"train_candidate_ids": first.train_candidate_ids[1:]})
    with pytest.raises(ValueError, match="final population does not reconcile"):
        summarize_research_folds(labels, replace(folds, folds=(incorrect,)))


def test_no_calendar_window_and_single_class_are_explicit_insufficiencies():
    labels, days = _labels()
    short = labels.loc[labels.trading_day.isin(days[:40])]
    folds, facts = build_research_fold_readiness(short, authorized_trading_days=days[:40])
    assert not folds.folds and facts["fold_count"] == facts["valid_fold_count"] == 0
    labels["binary_target"] = labels.binary_target.where(labels.binary_target.isna(), 1)
    _folds, facts = build_research_fold_readiness(labels, authorized_trading_days=days)
    assert facts["valid_fold_count"] == 0
    assert facts["invalid_reasons"] == {"insufficient_class_coverage": 3}
    assert all(row["train"]["class_counts"]["0"] == 0 for row in facts["per_fold"])
