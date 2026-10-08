"""Frozen regressors with portable Ridge state and train-only preprocessing.

No test input is accepted by fit. Chronological selection is explicit, and both
estimators receive exactly the same training labels and fixed feature schema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import digest


def select_fold(
    rows: pd.DataFrame, fold: dict, *, cutoff_ns: int, test_start_ns: int, test_end_ns: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Purge complete episodes and intervals, including unresolved test rows."""
    if rows["row_id"].duplicated().any():
        raise ValueError("duplicate decision identity")
    test = rows.loc[rows.trading_day.isin(fold["test_dates"])].copy()
    train = rows.loc[rows.trading_day.isin(fold["train_candidate_dates"])].copy()
    if cutoff_ns >= test_start_ns:
        raise ValueError("training cutoff must precede model activation")
    reasons = {}
    conditions = {
        "unavailable_label": train.label_status.ne("exact"),
        "nonfinite_label": ~np.isfinite(train.label.to_numpy(dtype=float)),
        "not_matured": train.label_available_ns.isna() | train.label_available_ns.gt(cutoff_ns),
        "episode_overlap": train.episode_id.isin(set(test.episode_id)),
        "interval_overlap": train.label_start_ns.le(test_end_ns)
        & (train.label_end_ns.isna() | train.label_end_ns.ge(test_start_ns)),
    }
    rejected = pd.Series(False, index=train.index)
    for reason, mask in conditions.items():
        reasons[reason] = int(mask.sum())
        rejected |= mask
    return train.loc[~rejected].copy(), test, reasons


def fit_validity(train: pd.DataFrame, job: str, protocol: dict) -> str | None:
    minimum = protocol["validity"][job]
    if len(train) < minimum["minimum_training_rows"]:
        return "insufficient_training_rows"
    if train.trading_day.nunique() < minimum["minimum_distinct_training_dates"]:
        return "insufficient_training_dates"
    y = train.label.to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("eligible training labels must be finite")
    if np.var(y) <= protocol["validity"]["minimum_target_population_variance"]:
        return "constant_training_target"
    return None


@dataclass
class RegressionFit:
    state: dict
    estimator: object | None = None

    @classmethod
    def fit(
        cls,
        train: pd.DataFrame,
        *,
        names: list[str],
        definitions: list[dict],
        model: str,
        protocol: dict,
        identity: dict,
    ) -> RegressionFit:
        if train.empty or train.row_id.duplicated().any():
            raise ValueError("training rows must be nonempty and unique")
        by_name = {r["name"]: r for r in definitions}
        numeric = [n for n in names if by_name[n]["kind"] == "numeric"]
        categories = {n: by_name[n]["domain"] for n in names if by_name[n]["kind"] == "categorical"}
        x = _numeric(train, numeric)
        y = train.label.to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("nonfinite training target")
        medians = [
            float(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else 0.0
            for col in x.T
        ]
        filled = np.where(np.isnan(x), medians, x)
        mean, scale = filled.mean(axis=0), filled.std(axis=0, ddof=0)
        scale[scale == 0] = 1
        state = {
            "schema": "ifsm_ml_regression_fit_v1",
            "model": model,
            "identity": identity,
            "names": names,
            "numeric": numeric,
            "categories": categories,
            "medians": medians,
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "all_null": [n for n, col in zip(numeric, x.T, strict=True) if np.isnan(col).all()],
            "constant": [n for n, col in zip(numeric, filled.T, strict=True) if np.ptp(col) == 0],
            "train_min": [
                float(np.min(c[np.isfinite(c)])) if np.isfinite(c).any() else None for c in x.T
            ],
            "train_max": [
                float(np.max(c[np.isfinite(c)])) if np.isfinite(c).any() else None for c in x.T
            ],
            "train_row_ids": train.row_id.tolist(),
            "train_labels_sha256": digest(y.tolist()),
            "train_features_sha256": digest(
                [
                    [None if pd.isna(value) else value for value in row]
                    for row in train[names].itertuples(index=False, name=None)
                ]
            ),
            "train_mean": float(y.mean()),
            "parameters": dict(protocol[model.lower()]["parameters"]),
        }
        result = cls(state)
        features = result.transform(train)
        if model == "RIDGE":
            from sklearn.linear_model import Ridge

            estimator = Ridge(**state["parameters"]).fit(features, y)
            state["coefficients"] = estimator.coef_.tolist()
            state["intercept"] = float(estimator.intercept_)
        elif model == "CATBOOST":
            from catboost import CatBoostRegressor

            estimator = CatBoostRegressor(**state["parameters"])
            estimator.fit(features, y, cat_features=list(categories))
            result.estimator = estimator
        else:
            raise ValueError("unsupported phase model")
        state["transformed_names"] = (
            features.columns.tolist()
            if isinstance(features, pd.DataFrame)
            else result.transformed_names()
        )
        state["fit_id"] = digest(state)
        return result

    def transformed_names(self) -> list[str]:
        state = self.state
        return (
            state["numeric"]
            + [f"{n}__missing" for n in state["numeric"]]
            + [f"{n}={v}" for n, values in state["categories"].items() for v in values]
        )

    def transform(self, rows: pd.DataFrame):
        state = self.state
        # Missing columns are a schema/program error, not legitimate source missingness.
        absent = set(state["names"]) - set(rows.columns)
        if absent:
            raise ValueError(f"missing declared feature columns: {sorted(absent)}")
        x = _numeric(rows, state["numeric"])
        flags = np.isnan(x).astype(float)
        categoricals = {
            n: [_category(v, domain) for v in rows[n]] for n, domain in state["categories"].items()
        }
        if state["model"] == "CATBOOST":
            frame = pd.DataFrame(x, columns=state["numeric"])
            for index, name in enumerate(state["numeric"]):
                frame[f"{name}__missing"] = flags[:, index]
            for name, values in categoricals.items():
                frame[name] = values
            return frame
        filled = np.where(np.isnan(x), state["medians"], x)
        scaled = (filled - state["mean"]) / state["scale"]
        columns = [scaled, flags]
        for name, domain in state["categories"].items():
            columns.append(
                np.array(
                    [[float(v == token) for token in domain] for v in categoricals[name]]
                ).reshape(len(rows), len(domain))
            )
        return np.concatenate(columns, axis=1)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        features = self.transform(rows)
        if self.state["model"] == "RIDGE":
            return features @ np.array(self.state["coefficients"]) + self.state["intercept"]
        return np.asarray(self.estimator.predict(features), dtype=np.float64)

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=False)
        state = dict(self.state)
        if state["model"] == "CATBOOST":
            import hashlib

            self.estimator.save_model(str(folder / "model.cbm"))
            state["model_sha256"] = hashlib.sha256((folder / "model.cbm").read_bytes()).hexdigest()
        (folder / "fit.json").write_text(
            json.dumps(state, indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, folder: Path) -> RegressionFit:
        import hashlib

        state = json.loads((folder / "fit.json").read_text(encoding="utf-8"))
        check = {k: v for k, v in state.items() if k not in {"fit_id", "model_sha256"}}
        if digest(check) != state["fit_id"]:
            raise ValueError("fit record identity mismatch")
        result = cls(state)
        if state["model"] == "CATBOOST":
            from catboost import CatBoostRegressor

            if (
                hashlib.sha256((folder / "model.cbm").read_bytes()).hexdigest()
                != state["model_sha256"]
            ):
                raise ValueError("native model hash mismatch")
            result.estimator = CatBoostRegressor()
            result.estimator.load_model(str(folder / "model.cbm"))
        return result


def _numeric(rows: pd.DataFrame, names: list[str]) -> np.ndarray:
    x = rows[names].to_numpy(dtype=np.float64, copy=True)
    if np.isinf(x).any():
        raise ValueError("infinite source feature must have a typed invalid-source reason")
    return x


def _category(value, domain: list[str]) -> str:
    if pd.isna(value):
        return "__MISSING__"
    return str(value) if str(value) in domain else "__UNSEEN__"
