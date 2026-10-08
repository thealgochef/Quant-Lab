"""Finite protocol, real estimator round trips and chronological isolation."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.propsim.funded.ml_phase.models import RegressionFit, fit_validity, select_fold
from alpha_lab.propsim.funded.ml_phase.protocol import (
    action_for,
    load_contracts,
    validate_contracts,
)

PACKET = Path(__file__).resolve().parents[3] / "docs/ifsm-mffu-ml-phase-v01"


@pytest.fixture
def contracts():
    return load_contracts(PACKET)


def training(contracts):
    names = contracts["FEATURE_LISTS"]["ENTRY"]["F0"]
    frame = pd.DataFrame({n: np.arange(40, dtype=float) for n in names})
    frame["htf_timeframe"] = "3600"
    frame["parent_timeframe"] = "60"
    frame["trading_weekday"] = "Mon"
    frame["parent_width_r"] = np.nan
    frame["row_id"] = [f"r{i}" for i in range(40)]
    frame["label"] = np.arange(40) * 0.1 - 2
    frame["trading_day"] = [f"2025-06-{i // 2 + 1:02}" for i in range(40)]
    return frame


def test_exact_contract_dates_and_cells(contracts):
    assert len(contracts["DATE_AND_FOLD_PLAN"]["scored_dates"]) == 171
    poisoned = copy.deepcopy(contracts)
    poisoned["DATE_AND_FOLD_PLAN"]["folds"][0]["test_dates"].append("2026-06-11")
    with pytest.raises(ValueError, match="fold"):
        validate_contracts(poisoned)


@pytest.mark.parametrize(
    "score,action",
    [
        (-1e-9, "change"),
        (0, "baseline"),
        (1e-9, "baseline"),
        (None, "baseline_unavailable"),
        (float("nan"), "baseline_unavailable"),
        (float("inf"), "baseline_unavailable"),
    ],
)
def test_fixed_zero_action(score, action):
    assert action_for(score) == action


@pytest.mark.parametrize("model", ["RIDGE", "CATBOOST"])
def test_real_fit_roundtrip_train_only_and_f0_isolation(contracts, tmp_path, model):
    frame = training(contracts)
    protocol = contracts["MODEL_PROTOCOL"]
    assert fit_validity(frame, "ENTRY", protocol) is None
    fit = RegressionFit.fit(
        frame,
        names=contracts["FEATURE_LISTS"]["ENTRY"]["F0"],
        definitions=contracts["FEATURE_DEFINITIONS"]["rows"],
        model=model,
        protocol=protocol,
        identity={"fixture": "causal"},
    )
    before = copy.deepcopy(fit.state)
    test = frame.iloc[:4].copy()
    test.loc[:, "initial_risk_points"] = 1e8
    test["total_net_gamma"] = -1e99
    test["label"] = 1e50
    test.loc[:, "parent_timeframe"] = [None, "novel", "60", "180"]
    predicted = fit.predict(test)
    assert fit.state == before
    assert np.isfinite(predicted).all()
    assert all("gamma" not in n for n in fit.state["transformed_names"])
    assert "parent_width_r" in fit.state["all_null"]
    folder = tmp_path / "fit"
    fit.save(folder)
    restored = RegressionFit.load(folder)
    np.testing.assert_allclose(restored.predict(test), predicted, rtol=1e-14, atol=1e-14)
    assert np.mean((fit.predict(frame) - frame.label) ** 2) < np.var(frame.label)
    with pytest.raises(ValueError, match="missing declared"):
        fit.predict(test.drop(columns="parent_width_r"))


def test_purge_label_maturity_intervals_and_full_test_episodes():
    frame = pd.DataFrame(
        [
            ["ok", "a", "2025-06-01", 1.0, "exact", 1, 4, 4],
            ["late", "b", "2025-06-01", 2.0, "exact", 1, 4, 12],
            ["overlap", "c", "2025-06-01", 3.0, "exact", 1, 15, 15],
            ["episode", "shared", "2025-06-01", 4.0, "exact", 1, 4, 4],
            ["test", "shared", "2025-06-03", np.nan, "censored", 11, None, None],
        ],
        columns=[
            "row_id",
            "episode_id",
            "trading_day",
            "label",
            "label_status",
            "label_start_ns",
            "label_end_ns",
            "label_available_ns",
        ],
    )
    fold = {"train_candidate_dates": ["2025-06-01"], "test_dates": ["2025-06-03"]}
    train, test, reasons = select_fold(frame, fold, cutoff_ns=9, test_start_ns=10, test_end_ns=20)
    assert train.row_id.tolist() == ["ok"]
    assert test.row_id.tolist() == ["test"]
    assert reasons["episode_overlap"] == 1
    assert reasons["interval_overlap"] == 1


def test_invalid_fold_is_not_a_zero_score(contracts):
    frame = training(contracts)
    assert (
        fit_validity(frame.iloc[:10], "ENTRY", contracts["MODEL_PROTOCOL"])
        == "insufficient_training_rows"
    )
    frame["label"] = 2.0
    assert fit_validity(frame, "ENTRY", contracts["MODEL_PROTOCOL"]) == "constant_training_target"
