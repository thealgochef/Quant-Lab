"""Deterministic setup-grouped, purged IFVG context walk-forward folds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .context_experiment_contracts import IfvgContextFoldDefinition

__all__ = [
    "ContextFoldSet",
    "build_context_folds",
]


@dataclass(frozen=True, slots=True)
class ContextFoldSet:
    folds: tuple[IfvgContextFoldDefinition, ...]
    assignment: pd.DataFrame
    status: str


def _ids(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(sorted(frame["candidate_id"].astype(str)))


def _invalid_reason(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    minimum_train_candidates: int,
) -> str | None:
    if len(train) < minimum_train_candidates:
        return "insufficient_train_candidates"
    classes = set(pd.to_numeric(train["binary_target"], errors="coerce").dropna().astype(int))
    if classes != {0, 1}:
        return "insufficient_class_coverage"
    if test.empty:
        return "no_resolved_test_candidates"
    return None


def build_context_folds(
    labeled_candidates: pd.DataFrame,
    *,
    authorized_trading_days: tuple[str, ...],
    train_days: int = 40,
    test_days: int = 5,
    step_days: int = 5,
    embargo_days: int = 2,
    minimum_train_candidates: int = 30,
    purge_from_logical_test_start: bool = False,
) -> ContextFoldSet:
    required = {
        "candidate_id",
        "setup_id",
        "trading_day",
        "entry_ts_utc",
        "resolution_ts_utc",
        "entry_available",
        "resolution_available",
        "binary_target",
    }
    missing = sorted(required - set(labeled_candidates))
    if missing:
        raise ValueError(f"fold input is missing columns {missing}")
    if train_days != 40 or test_days != 5 or step_days != 5 or embargo_days != 2:
        raise ValueError("IFVG context fold protocol is fixed at 40/5/5 with two-day embargo")
    days = tuple(str(day) for day in authorized_trading_days)
    if days != tuple(sorted(days)) or len(days) != len(set(days)):
        raise ValueError("authorized trading days must be unique and chronological")
    frame = labeled_candidates.copy()
    if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
        raise ValueError("fold input requires unique non-null candidate IDs")
    if not set(frame["trading_day"].astype(str)).issubset(days):
        raise ValueError("fold input contains an unauthorized trading day")
    frame["_entry"] = pd.to_datetime(frame["entry_ts_utc"], utc=True, errors="coerce")
    frame["_resolution"] = pd.to_datetime(frame["resolution_ts_utc"], utc=True, errors="coerce")
    available = (
        frame["entry_available"].fillna(False).astype(bool)
        & frame["resolution_available"].fillna(False).astype(bool)
        & frame["_entry"].notna()
        & frame["_resolution"].notna()
        & frame["binary_target"].notna()
    )
    eligible = frame.loc[available].copy()
    if (eligible["_resolution"] < eligible["_entry"]).any():
        raise ValueError("candidate label interval resolves before entry")

    folds: list[IfvgContextFoldDefinition] = []
    assignment_rows: list[dict[str, Any]] = []
    seen_test_ids: set[str] = set()
    fold_index = 0
    test_start_index = train_days
    while test_start_index + test_days <= len(days):
        train_day_values = days[:test_start_index]
        test_day_values = days[test_start_index : test_start_index + test_days]
        raw_train = eligible[eligible["trading_day"].astype(str).isin(train_day_values)].copy()
        raw_test = eligible[eligible["trading_day"].astype(str).isin(test_day_values)].copy()

        # Setup grouping is determined before censoring/availability exclusions.
        # Otherwise a censored sibling candidate can hide that a setup crosses the
        # train/test boundary and permit an exact-setup leak into both partitions.
        all_train = frame[frame["trading_day"].astype(str).isin(train_day_values)]
        all_test = frame[frame["trading_day"].astype(str).isin(test_day_values)]
        train_setups = set(all_train["setup_id"].astype(str))
        test_setups = set(all_test["setup_id"].astype(str))
        boundary_setups = train_setups & test_setups
        train = raw_train[~raw_train["setup_id"].astype(str).isin(boundary_setups)].copy()
        test = raw_test[~raw_test["setup_id"].astype(str).isin(boundary_setups)].copy()

        purged_ids: tuple[str, ...] = ()
        if purge_from_logical_test_start:
            from .features.mbp1_coverage_evidence import (  # noqa: PLC0415
                authorized_session_span_ns,
            )

            # The first logical test day starts at the previous civil day's
            # 18:00 ET, even when no candidate is emitted on that test day.
            boundary_ns, _end_ns = authorized_session_span_ns(test_day_values[0])
            overlap = train["_resolution"] >= pd.Timestamp(boundary_ns, tz="UTC")
            purged_ids = _ids(train.loc[overlap])
            train = train.loc[~overlap].copy()
        elif not test.empty:
            test_interval_start = test["_entry"].min()
            test_interval_end = test["_resolution"].max()
            overlap = (train["_entry"] <= test_interval_end) & (
                train["_resolution"] >= test_interval_start
            )
            purged_ids = _ids(train.loc[overlap])
            train = train.loc[~overlap].copy()

        embargo_day_values = train_day_values[-embargo_days:]
        embargo = train["trading_day"].astype(str).isin(embargo_day_values)
        embargoed_ids = _ids(train.loc[embargo])
        train = train.loc[~embargo].copy()

        duplicate_oos = set(test["candidate_id"].astype(str)) & seen_test_ids
        if duplicate_oos:
            raise ValueError(f"candidate appears in multiple OOS folds: {sorted(duplicate_oos)}")
        seen_test_ids.update(test["candidate_id"].astype(str))
        reason = _invalid_reason(
            train,
            test,
            minimum_train_candidates=minimum_train_candidates,
        )
        prevalence = (
            float(pd.to_numeric(train["binary_target"], errors="raise").mean())
            if not train.empty
            else None
        )
        definition = IfvgContextFoldDefinition(
            fold_index=fold_index,
            train_days=train_day_values,
            test_days=test_day_values,
            train_candidate_ids=_ids(train),
            test_candidate_ids=_ids(test),
            excluded_boundary_setup_ids=tuple(sorted(boundary_setups)),
            purged_candidate_ids=purged_ids,
            embargoed_candidate_ids=embargoed_ids,
            valid=reason is None,
            invalid_reason=reason,
            training_prevalence=prevalence,
        )
        folds.append(definition)
        for partition, subset in (("train", train), ("test", test)):
            assignment_rows.extend(
                {
                    "fold_index": fold_index,
                    "candidate_id": candidate_id,
                    "partition": partition,
                    "fold_valid": definition.valid,
                    "fold_invalid_reason": definition.invalid_reason,
                }
                for candidate_id in _ids(subset)
            )
        fold_index += 1
        test_start_index += step_days

    assignment = pd.DataFrame(
        assignment_rows,
        columns=(
            "fold_index",
            "candidate_id",
            "partition",
            "fold_valid",
            "fold_invalid_reason",
        ),
    )
    valid_folds = sum(fold.valid for fold in folds)
    status = "ready" if valid_folds else "insufficient_class_coverage"
    return ContextFoldSet(folds=tuple(folds), assignment=assignment, status=status)
