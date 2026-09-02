"""Bundle-independent comparison-row identity and the fold-local feature seam
(R6.1 §6.J / D13; owner plan-review correction 7).

Baseline and challenger arms of a controlled comparison are different
bundle views with DIFFERENT ``view_id``s, so the legacy view-scoped
``oos_row_id`` (``hash{view_id, fold_index, candidate_id}``) can never pair
their rows. Every bundle-parametrized rung therefore also emits

    comparison_row_id = hash{fold_schedule_id, candidate_fold_set_id,
                             fold_index, candidate_id, label_artifact_id}

— the identity of the OUT-OF-SAMPLE ROW itself (which fold of which schedule
over which fold population scored which candidate under which label
artifact), independent of the feature bundle that fed the model. The
identical-rows gate and every paired delta key on it; the legacy
``oos_row_id`` stays as a column for M0–M3 compatibility and never pairs arms.

The module also hosts :class:`RegimeFoldFeatureSource` — the seam through
which fold-LOCAL regime features (R6.1 §6.G, D7) reach a bundle-parametrized
rung: for fold *k* the runner left-joins the source's fold-*k* frame onto the
static view rows on ``candidate_id`` before fitting on the training rows and
predicting the test rows. A candidate without a fold-feature row keeps NaN
values (logistic: the fold-fitted imputer + indicator; CatBoost: native NaN;
categoricals: the registered missing token). No row of another fold is ever
consulted.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from ..context_experiment_contracts import canonical_contract_sha256
from ..context_folds import ContextFoldSet
from ..fold_schedules import derive_fold_schedule
from .fold_set_artifact import fold_set_id

__all__ = [
    "COMPARISON_ROW_IDENTITY_KEY",
    "LEGACY_ROW_IDENTITY_KEY",
    "LABEL_CONSUMED_COLUMNS",
    "LABEL_ARTIFACT_FORMULA_VERSION",
    "RegimeFoldFeatureSource",
    "comparison_row_id",
    "label_content_hash",
    "label_artifact_content_id",
    "default_fold_schedule_id",
    "candidate_fold_set_id",
    "assert_fold_local_source_coherent",
    "join_fold_local_features",
    "with_comparison_row_ids",
]

COMPARISON_ROW_IDENTITY_KEY = "comparison_row_id"
LEGACY_ROW_IDENTITY_KEY = "oos_row_id"

#: R6.1-FIX §3.6 (F-08): EVERY label / economic column a supervised study
#: consumes — the exact label artifact binds all of them (plus the registered
#: label policy), never only the ``(candidate_id, binary_target)`` pairs.
LABEL_CONSUMED_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "setup_id",
    "trading_day",
    "entry_ts_utc",
    "resolution_ts_utc",
    "entry_available",
    "resolution_available",
    "binary_target",
    "gross_r",
    "net_r",
)
LABEL_ARTIFACT_FORMULA_VERSION = "label_artifact_consumed_columns_v2"


def _label_cell(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return bool(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return float(value)
    return str(value)


def label_artifact_content_id(label_policy_id: str | None, labeled_candidates: pd.DataFrame) -> str:
    """The exact label artifact identity (R6.1-FIX §3.6): the registered
    label policy plus EVERY consumed label/economic column of every row,
    order-invariantly (rows sorted by candidate id). ``label_policy_id=None``
    is the UNPERSISTED helper form (``label_identity_source =
    content_hash_unpersisted``) — it can never name a persisted study."""

    missing = sorted(set(LABEL_CONSUMED_COLUMNS) - set(labeled_candidates.columns))
    if missing:
        raise ValueError(f"labeled candidates lack consumed label columns: {missing}")
    rows = sorted(
        (
            [_label_cell(row[column]) for column in LABEL_CONSUMED_COLUMNS]
            for row in labeled_candidates.loc[:, list(LABEL_CONSUMED_COLUMNS)].to_dict("records")
        ),
        key=lambda cells: str(cells[0]),
    )
    return canonical_contract_sha256(
        {
            "formula": LABEL_ARTIFACT_FORMULA_VERSION,
            "label_policy_id": None if label_policy_id is None else str(label_policy_id),
            "columns": list(LABEL_CONSUMED_COLUMNS),
            "rows": rows,
        }
    )


@runtime_checkable
class RegimeFoldFeatureSource(Protocol):
    """A per-fold feature source (the persisted ``RegimeFoldFeatureArtifact``).

    ``frame_for_fold(k)`` returns ONLY fold *k*'s rows (train + test
    partitions) with a ``candidate_id`` column plus exactly
    ``feature_names``; ``categorical_features`` names the block-declared
    categoricals among them (e.g. the fit-local hard regime id).
    """

    @property
    def artifact_id(self) -> str: ...

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    @property
    def categorical_features(self) -> tuple[str, ...]: ...

    def frame_for_fold(self, fold_index: int) -> pd.DataFrame: ...


def comparison_row_id(
    *,
    fold_schedule_id: str,
    candidate_fold_set_id: str,
    fold_index: int,
    candidate_id: str,
    label_artifact_id: str,
) -> str:
    """D13: the bundle-independent identity of one out-of-sample row."""

    return canonical_contract_sha256(
        {
            "fold_schedule_id": str(fold_schedule_id),
            "candidate_fold_set_id": str(candidate_fold_set_id),
            "fold_index": int(fold_index),
            "candidate_id": str(candidate_id),
            "label_artifact_id": str(label_artifact_id),
        }
    )


def label_content_hash(labeled_candidates: pd.DataFrame) -> str:
    """The ladder's label content id: sorted ``(candidate_id, target)`` pairs
    (the pipeline's S07 passes its own policy-bearing ``label_artifact_id``
    explicitly; this is the default when a ladder runs outside S07)."""

    return canonical_contract_sha256(
        {
            "labeled_pairs": sorted(
                (str(candidate_id), None if pd.isna(target) else int(target))
                for candidate_id, target in zip(
                    labeled_candidates["candidate_id"],
                    labeled_candidates["binary_target"],
                    strict=True,
                )
            )
        }
    )


def candidate_fold_set_id(folds: ContextFoldSet) -> str:
    """The ONE legacy row-population hash of the candidate fold set."""

    return fold_set_id(folds)


def default_fold_schedule_id(
    folds: ContextFoldSet, labeled_candidates: pd.DataFrame | None = None
) -> str:
    """The frozen schedule id derived from the days the fold set was built
    over: the labeled candidates' trading days (exactly what S08 authorizes)
    when labels are supplied, else the union of the fold windows' days; with
    no days at all (an empty fold set on an empty label view) a deterministic
    sentinel — never a caller string."""

    days: tuple[str, ...] = ()
    if labeled_candidates is not None and "trading_day" in labeled_candidates:
        days = tuple(sorted(set(labeled_candidates["trading_day"].astype(str))))
    if not days:
        pool: set[str] = set()
        for fold in folds.folds:
            pool.update(str(day) for day in fold.train_days)
            pool.update(str(day) for day in fold.test_days)
        days = tuple(sorted(pool))
    if not days:
        return canonical_contract_sha256({"fold_schedule": "no_authorized_days_v1"})
    return derive_fold_schedule(days).fold_schedule_id


def assert_fold_local_source_coherent(
    source: RegimeFoldFeatureSource,
    *,
    features: tuple[str, ...],
    static_columns: tuple[str, ...] | list[str],
) -> None:
    """Refuse an inert or colliding fold-local feature source."""

    names = tuple(source.feature_names)
    if not names:
        raise ValueError("fold-local feature source declares no feature names")
    if len(set(names)) != len(names):
        raise ValueError("fold-local feature source repeats a feature name")
    colliding = sorted(set(names) & set(static_columns))
    if colliding:
        raise ValueError(
            f"fold-local features collide with static view columns: {colliding}"
        )
    inert = sorted(set(names) - set(features))
    if inert:
        raise ValueError(
            "fold-local feature source names features outside the ordered model "
            f"features (an inert evidence claim is refused): {inert}"
        )
    unknown_categorical = sorted(set(source.categorical_features) - set(names))
    if unknown_categorical:
        raise ValueError(
            "fold-local categorical features must be among the source's feature "
            f"names: {unknown_categorical}"
        )


def join_fold_local_features(
    rows: pd.DataFrame,
    source: RegimeFoldFeatureSource,
    *,
    fold_index: int,
) -> pd.DataFrame:
    """LEFT-join fold ``fold_index``'s feature rows onto ``rows`` (indexed by
    ``candidate_id``); a candidate without a fold row keeps NaN values."""

    extra = source.frame_for_fold(int(fold_index))
    names = list(source.feature_names)
    if "candidate_id" not in extra.columns:
        raise ValueError("fold-local feature frame lacks candidate_id")
    missing = sorted(set(names) - set(extra.columns))
    if missing:
        raise ValueError(f"fold-local feature frame lacks declared features {missing}")
    keyed = extra.assign(candidate_id=extra["candidate_id"].astype(str))
    if keyed["candidate_id"].duplicated().any():
        raise ValueError(
            f"fold-local feature frame repeats a candidate in fold {fold_index}"
        )
    keyed = keyed.set_index("candidate_id")[names]
    joined = rows.join(keyed, how="left", validate="one_to_one")
    return joined


def with_comparison_row_ids(
    predictions: pd.DataFrame,
    *,
    fold_schedule_id: str,
    candidate_fold_set_id: str,
    label_artifact_id: str,
) -> pd.DataFrame:
    """Add (or verify) the ``comparison_row_id`` column of a prediction frame
    from its own ``fold_index`` / ``candidate_id`` columns."""

    if predictions.empty:
        frame = predictions.copy()
        if COMPARISON_ROW_IDENTITY_KEY not in frame.columns:
            frame[COMPARISON_ROW_IDENTITY_KEY] = pd.Series(dtype="object")
        return frame
    expected = [
        comparison_row_id(
            fold_schedule_id=fold_schedule_id,
            candidate_fold_set_id=candidate_fold_set_id,
            fold_index=int(fold_index),
            candidate_id=str(candidate_id),
            label_artifact_id=label_artifact_id,
        )
        for fold_index, candidate_id in zip(
            predictions["fold_index"], predictions["candidate_id"], strict=True
        )
    ]
    frame = predictions.copy()
    if COMPARISON_ROW_IDENTITY_KEY in frame.columns:
        if list(frame[COMPARISON_ROW_IDENTITY_KEY].astype(str)) != expected:
            raise ValueError(
                "prediction rows carry comparison_row_ids that do not derive from "
                "their own fold/candidate under the ladder's schedule, fold set, and "
                "label artifact"
            )
        return frame
    frame[COMPARISON_ROW_IDENTITY_KEY] = expected
    return frame
