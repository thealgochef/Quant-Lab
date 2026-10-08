"""The predeclared chronological benchmark and the identical online scorer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import RegressionFit, fit_validity, select_fold
from .protocol import action_for, digest


def frame_from_rows(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for name in ("decision_ns", "label_start_ns", "label_end_ns", "label_available_ns"):
        if name in frame:
            frame[name] = pd.array([row.get(name) for row in rows], dtype="Int64")
    return frame


def current_feature_reason(row, job, contracts):
    required = [r["name"] for r in contracts["FEATURE_DEFINITIONS"]["rows"]
                if job in r["jobs"] and r["required_for_valid_decision"]]
    if any(row.get(name) is None or not np.isfinite(row[name]) for name in required):
        return "required_current_feature_unavailable"
    return None


def fit_benchmark(
    *, datasets: dict, contracts: dict, boundaries: dict, identity: dict, folder: Path
) -> dict:
    """Every fit gets only its earlier training rows; test data is predict-only."""
    records, predictions, metrics, controls = [], [], [], []
    for cell in contracts["MODEL_MATRIX"]["cells"]:
        rows = datasets[cell["reference"]][cell["job"]]
        frame = frame_from_rows(rows)
        if frame.empty:
            frame = pd.DataFrame(
                columns=[
                    "row_id",
                    "trading_day",
                    "label",
                    "label_status",
                    "episode_id",
                    "decision_ns",
                    "label_start_ns",
                    "label_end_ns",
                    "label_available_ns",
                    *contracts["FEATURE_LISTS"][cell["job"]]["F2"],
                ]
            )
        for fold in contracts["DATE_AND_FOLD_PLAN"]["folds"]:
            boundary = boundaries[fold["fold_id"]]
            train, test, purge = select_fold(frame, fold, **boundary)
            reason = fit_validity(train, cell["job"], contracts["MODEL_PROTOCOL"])
            fit_identity = {
                **identity,
                "cell": cell,
                "fold": fold,
                "boundaries": boundary,
                "dataset_sha256": digest(rows),
            }
            record = {
                "cell_id": cell["cell_id"],
                "fold_id": fold["fold_id"],
                "reference": cell["reference"],
                "job": cell["job"],
                "feature_set": cell["feature_set"],
                "model": cell["model"],
                "train_rows": len(train),
                "train_dates": int(train.trading_day.nunique()),
                "test_rows": len(test),
                "purge": purge,
                "unavailable_reason": reason,
                "training_cutoff_ns": boundary["cutoff_ns"],
                "activation_ns": boundary["test_start_ns"],
                "model_id": None,
            }
            mean = float(train.label.mean()) if len(train) else None
            scores = [None] * len(test)
            if reason is None:
                destination = folder / "fits" / cell["cell_id"] / fold["fold_id"]
                if (destination / "fit.json").exists():
                    model = RegressionFit.load(destination)
                    if model.state["identity"] != fit_identity:
                        raise PermissionError("fit resume identity differs from the frozen inputs")
                else:
                    if destination.exists():
                        import uuid

                        destination = destination.with_name(
                            destination.name + ".retry-" + uuid.uuid4().hex
                        )
                    model = RegressionFit.fit(
                        train,
                        names=contracts["FEATURE_LISTS"][cell["job"]][cell["feature_set"]],
                        definitions=contracts["FEATURE_DEFINITIONS"]["rows"],
                        model=cell["model"],
                        protocol=contracts["MODEL_PROTOCOL"],
                        identity=fit_identity,
                    )
                    model.save(destination)
                scores = model.predict(test).tolist() if len(test) else []
                # Serialization is part of every real fit, not merely a synthetic check.
                restored = RegressionFit.load(destination)
                if len(test):
                    np.testing.assert_allclose(
                        restored.predict(test), scores, rtol=1e-13, atol=1e-13
                    )
                record["model_id"] = model.state["fit_id"]
                record["fit_path"] = destination.relative_to(folder).as_posix()
            record["status"] = "valid" if reason is None else "baseline_fallback"
            records.append(record)
            subset = []
            for row, score in zip(test.to_dict("records"), scores, strict=True):
                unavailable = reason or current_feature_reason(row, cell["job"], contracts)
                if unavailable:
                    score = None
                elif score is not None and not np.isfinite(score):
                    unavailable, score = "nonfinite_model_score", None
                label = float(row["label"]) if row["label_status"] == "exact" else None
                prediction = {
                    k: row[k]
                    for k in ("row_id", "trading_day", "decision_ns", "episode_id", "candidate")
                }
                prediction.update(
                    cell_id=cell["cell_id"],
                    fold_id=fold["fold_id"],
                    model_id=record["model_id"],
                    training_cutoff_ns=boundary["cutoff_ns"],
                    activation_ns=boundary["test_start_ns"],
                    prediction=score if score is None or np.isfinite(score) else None,
                    action=action_for(score),
                    unavailable_reason=unavailable,
                    label=label,
                    label_status=row["label_status"],
                    train_mean=mean,
                )
                predictions.append(prediction)
                subset.append(prediction)
                if cell["feature_set"] == "F0" and cell["model"] == "RIDGE":
                    controls.append(
                        {
                            **prediction,
                            "cell_id": f"MEAN_{cell['reference']}_{cell['job']}",
                            "prediction": mean,
                            "action": "predictive_control_only",
                        }
                    )
            metrics.append(
                {
                    "cell_id": cell["cell_id"],
                    "fold_id": fold["fold_id"],
                    **prediction_metrics(subset),
                }
            )
    pooled = [
        {
            "cell_id": cell["cell_id"],
            **prediction_metrics([r for r in predictions if r["cell_id"] == cell["cell_id"]]),
        }
        for cell in contracts["MODEL_MATRIX"]["cells"]
    ]
    result = {
        "fits": records,
        "predictions": predictions,
        "fold_metrics": metrics,
        "pooled_metrics": pooled,
        "mean_controls": controls,
        "identity": identity,
        "cells": contracts["MODEL_MATRIX"]["cells"],
    }
    return result


def prediction_metrics(rows):
    valid = [
        r
        for r in rows
        if r["prediction"] is not None
        and r["label"] is not None
        and np.isfinite(r["prediction"])
        and np.isfinite(r["label"])
    ]
    if not valid:
        return {
            "observations": 0,
            "status": "no_valid_predictions",
            "rmse": None,
            "mae": None,
            "skill": None,
            "bias": None,
            "association": None,
        }
    y = np.array([r["label"] for r in valid])
    p = np.array([r["prediction"] for r in valid])
    mean = np.array([r["train_mean"] for r in valid])
    denominator = np.sum((y - mean) ** 2)
    return {
        "observations": len(valid),
        "status": "available",
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(abs(y - p))),
        "bias": float(np.mean(p - y)),
        "skill": float(1 - np.sum((y - p) ** 2) / denominator) if denominator > 0 else None,
        "skill_null_reason": "constant_control_zero_error" if denominator == 0 else None,
        "association": float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 0 and np.std(y) > 0 else None,
        "negative_count": int((p < 0).sum()),
        "nonnegative_count": int((p >= 0).sum()),
        "negative_mean_net_r": float(y[p < 0].mean()) if (p < 0).any() else None,
        "nonnegative_mean_net_r": float(y[p >= 0].mean()) if (p >= 0).any() else None,
    }


class OnlineScorer:
    """Event-time model selection and feature inference, never a saved-ID lookup."""

    def __init__(self, cell, benchmark: dict, model_root: Path, contracts: dict):
        self.cell, self.contracts = cell, contracts
        records = [r for r in benchmark["fits"] if r["cell_id"] == cell["cell_id"]]
        self.records = {r["fold_id"]: r for r in records}
        folds = {f["fold_id"]: f for f in contracts["DATE_AND_FOLD_PLAN"]["folds"]}
        if len(records) != len(folds) or set(self.records) != set(folds):
            raise PermissionError("model fold membership differs from the frozen contract")
        self.dates = {
            day: fold["fold_id"]
            for fold in contracts["DATE_AND_FOLD_PLAN"]["folds"]
            for day in fold["test_dates"]
        }
        self.models = {
            key: RegressionFit.load(model_root / r["fit_path"])
            for key, r in self.records.items()
            if r["model_id"] is not None
        }
        for key, model in self.models.items():
            record = self.records[key]
            identity = model.state["identity"]
            boundary = identity["boundaries"]
            if (
                model.state["fit_id"] != record["model_id"]
                or identity["cell"] != cell
                or identity["fold"] != folds[key]
                or any(identity.get(k) != v for k, v in benchmark["identity"].items())
                or boundary["cutoff_ns"] != record["training_cutoff_ns"]
                or boundary["test_start_ns"] != record["activation_ns"]
                or boundary["cutoff_ns"] >= boundary["test_start_ns"]
            ):
                raise PermissionError(
                    "loaded model identity or time boundary differs from its fold"
                )
        self.shadow_keys = {
            (r["decision_ns"], r["episode_id"], r["candidate"])
            for r in benchmark["predictions"]
            if r["cell_id"] == cell["cell_id"]
        }

    def __call__(self, job, row):
        if job != self.cell["job"]:
            return {"action": "baseline", "score": None, "reason": "other_job_unchanged"}
        fold_id = self.dates[row["trading_day"]]
        record = self.records[fold_id]
        if row["decision_ns"] < record["activation_ns"]:
            raise PermissionError("model used before activation")
        common = {
            "cell_id": self.cell["cell_id"],
            "fold_id": fold_id,
            "model_id": record["model_id"],
            "training_cutoff_ns": record["training_cutoff_ns"],
            "activation_ns": record["activation_ns"],
            "new_opportunity": (row["decision_ns"], row["episode_id"], row["candidate"])
            not in self.shadow_keys,
        }
        if record["model_id"] is None:
            return {
                **common,
                "score": None,
                "action": "baseline_unavailable",
                "reason": record["unavailable_reason"],
            }
        if current_feature_reason(row, job, self.contracts):
            return {
                **common,
                "score": None,
                "action": "baseline_unavailable",
                "reason": "required_current_feature_unavailable",
            }
        model = self.models[fold_id]
        value = float(model.predict(pd.DataFrame([row]))[0])
        out_of_range = [
            name
            for name, lo, hi in zip(
                model.state["numeric"],
                model.state["train_min"],
                model.state["train_max"],
                strict=True,
            )
            if row.get(name) is not None and lo is not None and not lo <= row[name] <= hi
        ]
        return {
            **common,
            "score": value if np.isfinite(value) else None,
            "action": action_for(value),
            "reason": None if np.isfinite(value) else "nonfinite_model_score",
            "out_of_training_range": out_of_range,
        }


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
