"""Sample-adequacy PREVIEW (R6.1 §6.C) — pure, fit-free.

Before any regime fit exists the pipeline can state, per fold, exactly what
the sample-adequacy gate will see: the training rows that carry at least one
regime input (the rows the fold-local fit actually trains on — all-missing
rows are typed ``source_feature_missing`` and never enter a fit), the
stamped per-grain floor, and the expected gate outcome. The preview mirrors
``regime_service._assess_capability``'s row counting exactly (a test proves
preview == assessment on n=600 / n=170 / a panel).

Honesty fields: the floor is FLAT per grain (it is NOT a function of k —
the audit's "150 for k=3" was a coincidence); the balanced-cluster lower
bound ``k × minimum_cluster_rows_per_fold`` is informational; the occupancy,
rows-per-cluster, empty-cluster, OOS-coverage, and stability gates cannot be
previewed without a fit. A failing preview is a typed fact, never a stage
failure.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ..context_folds import ContextFoldSet
from ..search.identities import FrozenContract
from .regime_contracts import (
    REGIME_PROPOSED_DEFAULTS,
    ObservationGranularity,
    sample_adequacy_minimum,
)
from .regime_preprocessing import keyed_observations

__all__ = [
    "NON_PREVIEWABLE_GATES",
    "FoldSampleAdequacy",
    "SampleAdequacyPreview",
    "sample_adequacy_floor_facts",
    "preview_sample_adequacy",
]

NON_PREVIEWABLE_GATES: tuple[str, ...] = (
    "minimum_cluster_occupancy",
    "minimum_cluster_rows_per_fold",
    "empty_cluster_in_fold",
    "no_oos_assignment_coverage",
    "bootstrap_aligned_ami_below_minimum",
)


class FoldSampleAdequacy(FrozenContract):
    fold_index: int = Field(ge=0)
    fold_valid: bool
    fold_invalid_reason: str | None
    train_rows_total: int = Field(ge=0)
    train_rows_with_inputs: int = Field(ge=0)
    train_rows_all_missing: int = Field(ge=0)
    test_rows_total: int = Field(ge=0)
    expected_sample_adequacy: Literal["pass", "fail", "not_fitted"]


class SampleAdequacyPreview(FrozenContract):
    observation_granularity: ObservationGranularity
    minimum_training_observations: int = Field(ge=0)
    resolved_cluster_count: int = Field(ge=2)
    per_fold: tuple[FoldSampleAdequacy, ...]
    observed_minimum_training_rows: int = Field(ge=0)
    expected_gate_outcome: Literal["pass", "fail", "no_valid_folds"]
    #: honesty: the floor is flat per grain, never a function of k
    floor_is_flat_per_grain: Literal[True] = True
    #: informational lower bound only (k × minimum_cluster_rows_per_fold)
    balanced_cluster_rows_lower_bound: int = Field(ge=0)
    non_previewable_gates: tuple[str, ...] = NON_PREVIEWABLE_GATES
    stamp: Literal["proposed_protocol_default"] = "proposed_protocol_default"


def sample_adequacy_floor_facts(
    observation_granularity: ObservationGranularity, resolved_cluster_count: int
) -> dict:
    """S06 floor facts (before folds exist): the stamped per-grain floor and
    the informational balanced-cluster lower bound."""

    grain = ObservationGranularity(observation_granularity)
    floor = sample_adequacy_minimum(grain)
    cluster_rows = int(REGIME_PROPOSED_DEFAULTS["minimum_cluster_rows_per_fold"]["value"])
    return {
        "observation_granularity": grain.value,
        "minimum_training_observations": floor,
        "floor_is_flat_per_grain": True,
        "resolved_cluster_count": int(resolved_cluster_count),
        "balanced_cluster_rows_lower_bound": int(resolved_cluster_count) * cluster_rows,
        "minimum_cluster_rows_per_fold": cluster_rows,
        "non_previewable_gates": list(NON_PREVIEWABLE_GATES),
        "stamp": "proposed_protocol_default",
    }


def preview_sample_adequacy(
    observation_frame: pd.DataFrame,
    folds: ContextFoldSet,
    *,
    observation_granularity: ObservationGranularity,
    resolved_cluster_count: int,
    resolved_input_features: tuple[str, ...] | None = None,
) -> SampleAdequacyPreview:
    """Per-fold training-row facts under the stamped floor, fit-free."""

    grain = ObservationGranularity(observation_granularity)
    floor = sample_adequacy_minimum(grain)
    cluster_rows = int(REGIME_PROPOSED_DEFAULTS["minimum_cluster_rows_per_fold"]["value"])
    indexed = keyed_observations(observation_frame)
    features = tuple(resolved_input_features or ())
    missing = sorted(set(features) - set(indexed.columns))
    if missing:
        raise ValueError(f"observation frame lacks regime inputs: {missing}")
    per_fold: list[FoldSampleAdequacy] = []
    observed: list[int] = []
    for fold in folds.folds:
        train_ids = [str(value) for value in fold.train_candidate_ids]
        absent = sorted(set(train_ids) - set(indexed.index))
        if absent:
            raise ValueError(
                f"fold {fold.fold_index}: {len(absent)} training observations are "
                "missing from the observation frame"
            )
        if features and train_ids:
            raw = indexed.loc[train_ids, list(features)].to_numpy(dtype=float)
            with_inputs = int((~np.isnan(raw).all(axis=1)).sum())
        else:
            with_inputs = len(train_ids)
        all_missing = len(train_ids) - with_inputs
        if not fold.valid:
            outcome: Literal["pass", "fail", "not_fitted"] = "not_fitted"
        else:
            outcome = "pass" if with_inputs >= floor else "fail"
            observed.append(with_inputs)
        per_fold.append(
            FoldSampleAdequacy(
                fold_index=fold.fold_index,
                fold_valid=fold.valid,
                fold_invalid_reason=fold.invalid_reason,
                train_rows_total=len(train_ids),
                train_rows_with_inputs=with_inputs,
                train_rows_all_missing=all_missing,
                test_rows_total=len(fold.test_candidate_ids),
                expected_sample_adequacy=outcome,
            )
        )
    if not observed:
        outcome_all: Literal["pass", "fail", "no_valid_folds"] = "no_valid_folds"
    else:
        outcome_all = "pass" if min(observed) >= floor else "fail"
    return SampleAdequacyPreview(
        observation_granularity=grain,
        minimum_training_observations=floor,
        resolved_cluster_count=int(resolved_cluster_count),
        per_fold=tuple(per_fold),
        observed_minimum_training_rows=min(observed) if observed else 0,
        expected_gate_outcome=outcome_all,
        balanced_cluster_rows_lower_bound=int(resolved_cluster_count) * cluster_rows,
    )
