"""Fold-local regime preprocessing (ML_REGIME_CONTRACT_PLAN §5.3).

The fixed per-fold pipeline: ``SimpleImputer(median, add_indicator)`` →
optional train-fitted winsorizer (p01/p99 clips) → ``StandardScaler``.
Every statistic is fitted on TRAINING-FOLD rows only, and the fit API
accepts an ``IfvgContextFoldDefinition`` and slices internally — there is
no row-subset parameter (test-asserted), so a pooled or future-leaking fit
has no representation at THIS seam (the fold definitions themselves are
the caller's evidence: the service binds their exact ids into
``fold_set_id`` and the training input values into every fit identity).

Rows whose regime inputs are ALL missing never enter a fit: they are typed
``source_feature_missing`` and excluded from the training matrix, so the
fit identity (``training_row_ids_hash`` + ``training_feature_matrix_hash``)
and the typed-null coverage report agree exactly (review F10).

The fitted pipeline persists dual-format (canonical JSON parameter payload
whose hash is ``fitted_parameter_payload_hash``, plus a joblib binary for
reuse); persistence and reload verification live in ``regime_store``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..context_folds import IfvgContextFoldDefinition
from ..search.identities import canonical_contract_sha256

__all__ = [
    "WINSORIZATION_POLICIES",
    "OBSERVATION_KEY_COLUMNS",
    "OBSERVATION_TS_COLUMNS",
    "observation_key_column",
    "observation_ts_column",
    "keyed_observations",
    "training_feature_matrix_hash",
    "FittedRegimePreprocessing",
    "fit_regime_preprocessing",
]

WINSORIZATION_POLICIES: tuple[str, ...] = ("none", "clip_p01_p99_train_fitted_v1")

#: The observation key column (candidate views key by ``candidate_id``; a
#: completed-bar panel keys by ``row_id``) — first present wins.
OBSERVATION_KEY_COLUMNS: tuple[str, ...] = ("candidate_id", "row_id")

#: The observation's as-of timestamp — first present wins (candidate views
#: carry ``entry_ts_utc``/``feature_as_of_ts``; panels ``bar_close_ts_utc``).
OBSERVATION_TS_COLUMNS: tuple[str, ...] = (
    "entry_ts_utc",
    "feature_as_of_ts",
    "bar_close_ts_utc",
)


def observation_key_column(frame: pd.DataFrame) -> str:
    for column in OBSERVATION_KEY_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(
        f"observation frame requires one of the key columns {OBSERVATION_KEY_COLUMNS}"
    )


def observation_ts_column(frame: pd.DataFrame) -> str | None:
    for column in OBSERVATION_TS_COLUMNS:
        if column in frame.columns:
            return column
    return None


def keyed_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame indexed by its string observation key; duplicates refused
    (a duplicated key would silently reweight every fitted statistic)."""

    column = observation_key_column(frame)
    keys = frame[column].astype(str)
    duplicates = sorted(keys[keys.duplicated()].unique().tolist())
    if duplicates:
        raise ValueError(
            f"observation frame has {len(duplicates)} duplicated {column} values "
            f"(e.g. {duplicates[:3]}); every observation must be unique"
        )
    indexed = frame.copy()
    indexed.index = pd.Index(keys, name=column)
    return indexed


def _jsonable_array(values: np.ndarray | None) -> list | None:
    if values is None:
        return None
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(values, dtype=float)]


def training_feature_matrix_hash(
    indexed: pd.DataFrame, row_ids: tuple[str, ...], features: tuple[str, ...]
) -> str:
    """Canonical hash of the training INPUT values (row id → raw feature
    values, NaN as null) — the content half of every fit identity."""

    matrix = indexed.loc[list(row_ids), list(features)].to_numpy(dtype=float)
    rows = {
        str(row_id): [None if not np.isfinite(v) else float(v) for v in values]
        for row_id, values in zip(row_ids, matrix, strict=True)
    }
    return canonical_contract_sha256({"features": list(features), "rows": rows})


@dataclass(frozen=True, slots=True)
class FittedRegimePreprocessing:
    """One fold's fitted pipeline + its canonical parameter payload."""

    fold_index: int
    features: tuple[str, ...]
    winsorization_policy: str
    imputer: SimpleImputer
    scaler: StandardScaler
    clip_lower: np.ndarray | None
    clip_upper: np.ndarray | None
    output_feature_names: tuple[str, ...]
    parameter_payload: dict
    fitted_parameter_payload_hash: str
    training_row_ids: tuple[str, ...]
    training_rows_all_missing: tuple[str, ...]
    training_feature_matrix_hash: str

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Deterministic transform of any rows carrying the feature columns."""

        matrix = frame.loc[:, list(self.features)].to_numpy(dtype=float)
        imputed = self.imputer.transform(matrix)
        n = len(self.features)
        if self.clip_lower is not None and self.clip_upper is not None:
            imputed[:, :n] = np.clip(imputed[:, :n], self.clip_lower, self.clip_upper)
        return self.scaler.transform(imputed)


def fit_regime_preprocessing(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    fold: IfvgContextFoldDefinition,
    *,
    winsorization_policy: str = "none",
) -> FittedRegimePreprocessing:
    """Fit the fixed pipeline on the fold's TRAINING rows only.

    The frame is keyed by ``candidate_id`` (or ``row_id`` for a panel); the
    training slice is derived from ``fold.train_candidate_ids`` INTERNALLY —
    no caller-supplied row subset exists. All-missing training rows are
    excluded (typed ``source_feature_missing`` downstream).
    """

    if winsorization_policy not in WINSORIZATION_POLICIES:
        raise ValueError(
            f"unknown winsorization policy {winsorization_policy!r}; "
            f"registered: {WINSORIZATION_POLICIES}"
        )
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"preprocessing input lacks features: {missing}")
    if not fold.valid:
        raise ValueError(
            f"fold {fold.fold_index} is invalid ({fold.invalid_reason}); "
            "nothing is fitted on invalid folds"
        )
    indexed = keyed_observations(frame)
    fold_train_ids = tuple(sorted(str(cid) for cid in fold.train_candidate_ids))
    absent = sorted(set(fold_train_ids) - set(indexed.index))
    if absent:
        raise ValueError(
            f"{len(absent)} training candidates are missing from the input frame"
        )
    raw = indexed.loc[list(fold_train_ids), list(features)].to_numpy(dtype=float)
    present_mask = ~np.isnan(raw).all(axis=1)
    all_missing = tuple(
        row_id for row_id, present in zip(fold_train_ids, present_mask, strict=True)
        if not present
    )
    train_ids = tuple(
        row_id for row_id, present in zip(fold_train_ids, present_mask, strict=True)
        if present
    )
    if not train_ids:
        raise ValueError(
            f"fold {fold.fold_index} has no training row with any regime input present"
        )
    train = raw[present_mask]

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    imputed = imputer.fit_transform(train)
    n = len(features)
    clip_lower = clip_upper = None
    if winsorization_policy == "clip_p01_p99_train_fitted_v1":
        clip_lower = np.nanpercentile(imputed[:, :n], 1, axis=0)
        clip_upper = np.nanpercentile(imputed[:, :n], 99, axis=0)
        imputed[:, :n] = np.clip(imputed[:, :n], clip_lower, clip_upper)
    scaler = StandardScaler()
    scaler.fit(imputed)

    indicator_names = tuple(
        f"missing_indicator__{features[i]}"
        for i in (
            imputer.indicator_.features_
            if getattr(imputer, "indicator_", None) is not None
            else ()
        )
    )
    output_names = (*features, *indicator_names)
    parameter_payload = {
        "pipeline": "median_impute_indicator__winsorize__standard_scale_v1",
        "features": list(features),
        "winsorization_policy": winsorization_policy,
        "imputer_medians": _jsonable_array(imputer.statistics_),
        "clip_lower": _jsonable_array(clip_lower),
        "clip_upper": _jsonable_array(clip_upper),
        "scaler_mean": _jsonable_array(scaler.mean_),
        "scaler_scale": _jsonable_array(scaler.scale_),
        "output_feature_names": list(output_names),
    }
    return FittedRegimePreprocessing(
        fold_index=fold.fold_index,
        features=tuple(features),
        winsorization_policy=winsorization_policy,
        imputer=imputer,
        scaler=scaler,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        output_feature_names=output_names,
        parameter_payload=parameter_payload,
        fitted_parameter_payload_hash=canonical_contract_sha256(parameter_payload),
        training_row_ids=train_ids,
        training_rows_all_missing=all_missing,
        training_feature_matrix_hash=training_feature_matrix_hash(
            indexed, train_ids, tuple(features)
        ),
    )
