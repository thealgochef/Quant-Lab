"""Whole finite synthetic benchmark and unseen online-event inference."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from alpha_lab.propsim.funded.clock import to_ns
from alpha_lab.propsim.funded.ml_phase.benchmark import OnlineScorer, fit_benchmark, save_json
from alpha_lab.propsim.funded.ml_phase.protocol import load_contracts


def fixture_data():
    contracts = load_contracts(Path(__file__).resolve().parents[3] / "docs/ifsm-mffu-ml-phase-v01")
    definitions = contracts["FEATURE_DEFINITIONS"]["rows"]
    dates = contracts["DATE_AND_FOLD_PLAN"]["original_evaluation_dates"]
    datasets = {}
    for reference in ("MCB062", "MCB025"):
        datasets[reference] = {}
        for job in ("ENTRY", "CONTINUATION"):
            rows = []
            for index, day in enumerate(dates):
                ns = to_ns(day + "T12:00:00Z")
                row = {
                    r["name"]: (
                        r["domain"][0] if r["kind"] == "categorical" else float(index % 13 + 1)
                    )
                    for r in definitions
                    if job in r["jobs"]
                }
                key = f"{reference}/{job}/{day}"
                row.update(
                    row_id=key,
                    candidate=key,
                    episode_id=key,
                    trading_day=day,
                    decision_ns=ns,
                    label_start_ns=ns,
                    label_end_ns=ns + 60_000_000_000,
                    label_available_ns=ns + 60_000_000_000,
                    label_status="exact",
                    label=(index % 13 - 6) / 3,
                )
                rows.append(row)
            datasets[reference][job] = rows
    boundaries = {
        fold["fold_id"]: {
            "cutoff_ns": to_ns(fold["train_candidate_dates"][-1] + "T22:00:00Z"),
            "test_start_ns": to_ns(fold["test_dates"][0] + "T00:00:00Z"),
            "test_end_ns": to_ns(fold["test_dates"][-1] + "T22:00:00Z"),
        }
        for fold in contracts["DATE_AND_FOLD_PLAN"]["folds"]
    }
    return contracts, datasets, boundaries


def test_full_finite_benchmark_serialization_resume_and_online_new_event(tmp_path):
    contracts, datasets, boundaries = fixture_data()
    kwargs = dict(
        datasets=datasets,
        contracts=contracts,
        boundaries=boundaries,
        identity={"synthetic_only": True},
        folder=tmp_path,
    )
    result = fit_benchmark(**kwargs)
    assert len(result["fits"]) == 216
    assert all(r["status"] == "valid" for r in result["fits"])
    assert len(result["predictions"]) == 24 * 171
    assert len(result["mean_controls"]) == 4 * 171
    keys = [(r["cell_id"], r["row_id"]) for r in result["predictions"]]
    assert len(keys) == len(set(keys))
    assert all(
        r["training_cutoff_ns"] < r["activation_ns"] <= r["decision_ns"]
        for r in result["predictions"]
    )
    save_json(tmp_path / "benchmark.json", result)
    assert fit_benchmark(**kwargs) == result
    cell = contracts["MODEL_MATRIX"]["cells"][0]
    scorer = OnlineScorer(cell, result, tmp_path, contracts)
    row = dict(datasets[cell["reference"]][cell["job"]][82])
    row["candidate"] = row["episode_id"] = "new-on-policy-opportunity"
    scored = scorer(cell["job"], row)
    assert scored["new_opportunity"] and scored["score"] is not None
    assert scorer("CONTINUATION", row)["reason"] == "other_job_unchanged"


def test_empty_populations_remain_24_named_fallback_cells(tmp_path):
    contracts, _, boundaries = fixture_data()
    empty = {r: {job: [] for job in ("ENTRY", "CONTINUATION")} for r in ("MCB062", "MCB025")}
    result = fit_benchmark(
        datasets=empty,
        contracts=contracts,
        boundaries=boundaries,
        identity={"synthetic_only": True},
        folder=tmp_path,
    )
    assert len(result["pooled_metrics"]) == 24
    assert len(result["fits"]) == 216
    assert all(r["status"] == "baseline_fallback" for r in result["fits"])
    assert not result["predictions"]


def test_offline_unavailable_parity_and_later_model_substitution_rejected(tmp_path):
    contracts, datasets, boundaries = fixture_data()
    contracts["MODEL_MATRIX"]["cells"] = contracts["MODEL_MATRIX"]["cells"][:1]
    contracts["DATE_AND_FOLD_PLAN"]["folds"] = contracts["DATE_AND_FOLD_PLAN"]["folds"][:2]
    cell = contracts["MODEL_MATRIX"]["cells"][0]
    row = datasets[cell["reference"]][cell["job"]][82]
    required = next(r["name"] for r in contracts["FEATURE_DEFINITIONS"]["rows"]
                    if cell["job"] in r["jobs"] and r["required_for_valid_decision"])
    row[required] = None
    result = fit_benchmark(datasets=datasets, contracts=contracts, boundaries=boundaries,
                           identity={"synthetic_only": True}, folder=tmp_path)
    prediction = next(p for p in result["predictions"] if p["row_id"] == row["row_id"])
    online = OnlineScorer(cell, result, tmp_path, contracts)(cell["job"], row)
    assert prediction["prediction"] is online["score"] is None
    assert prediction["action"] == online["action"] == "baseline_unavailable"
    assert prediction["unavailable_reason"] == online["reason"]
    changed = copy.deepcopy(result)
    changed["fits"][0]["fit_path"] = changed["fits"][1]["fit_path"]
    with pytest.raises(PermissionError, match="loaded model identity"):
        OnlineScorer(cell, changed, tmp_path, contracts)
    changed = copy.deepcopy(result)
    changed["fits"].pop()
    with pytest.raises(PermissionError, match="fold membership"):
        OnlineScorer(cell, changed, tmp_path, contracts)
