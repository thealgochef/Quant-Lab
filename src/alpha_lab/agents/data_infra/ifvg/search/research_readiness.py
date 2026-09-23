"""Pure labeled-fold readiness facts; no preparation, fitting, or persistence."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from ..context_folds import ContextFoldSet, build_context_folds


def _available_labels(labels: pd.DataFrame) -> pd.Series:
    """Availability accounting matches the canonical fold builder's eligibility mask."""

    return (
        labels["entry_available"].fillna(False).astype(bool)
        & labels["resolution_available"].fillna(False).astype(bool)
        & pd.to_datetime(labels["entry_ts_utc"], utc=True, errors="coerce").notna()
        & pd.to_datetime(labels["resolution_ts_utc"], utc=True, errors="coerce").notna()
        & labels["binary_target"].notna()
    )


def _censored_labels(labels: pd.DataFrame) -> pd.Series:
    if "censored" in labels:
        return labels["censored"].fillna(False).astype(bool)
    if "label" in labels:
        return labels["label"].astype(str).eq("censored")
    return pd.Series(False, index=labels.index)


def _partition_facts(labels, available, censored, fold, partition):
    days = fold.train_days if partition == "train" else fold.test_days
    raw = labels.loc[labels["trading_day"].astype(str).isin(days)]
    raw_ids = set(raw.index)
    eligible = raw.loc[available.loc[raw.index]]
    unavailable_ids = raw_ids - set(eligible.index)
    boundary_ids = set(
        eligible.loc[eligible["setup_id"].astype(str).isin(fold.excluded_boundary_setup_ids)].index
    )
    purged_ids = set(fold.purged_candidate_ids) if partition == "train" else set()
    embargoed_ids = set(fold.embargoed_candidate_ids) if partition == "train" else set()
    final_ids = set(
        fold.train_candidate_ids if partition == "train" else fold.test_candidate_ids
    )
    stages = (unavailable_ids, boundary_ids, purged_ids, embargoed_ids, final_ids)
    covered = set()
    for stage in stages:
        if covered & stage or not stage.issubset(raw_ids):
            raise ValueError(
                f"fold {fold.fold_index} {partition} exclusion stages do not reconcile"
            )
        covered.update(stage)
    if covered != raw_ids:
        raise ValueError(f"fold {fold.fold_index} {partition} final population does not reconcile")
    final = labels.loc[sorted(final_ids)]
    targets = pd.to_numeric(final["binary_target"], errors="raise").astype(int)
    if not set(targets).issubset({0, 1}):
        raise ValueError("readiness requires binary 0/1 final targets")
    return {
        "raw_count": len(raw),
        "unavailable_or_censored_count": len(unavailable_ids),
        # Descriptive subset of the raw population, not another removal stage.
        "censored_count": int(censored.loc[raw.index].sum()),
        "boundary_setup_excluded_count": len(boundary_ids),
        "purged_count": len(purged_ids),
        "embargoed_count": len(embargoed_ids),
        "final_count": len(final),
        "class_counts": {"0": int((targets == 0).sum()), "1": int((targets == 1).sum())},
        "counts_reconcile": True,
    }


def summarize_research_folds(labels: pd.DataFrame, folds: ContextFoldSet) -> dict:
    """Explain the canonical row selections, asserting disjoint accounting per partition."""

    if labels["candidate_id"].isna().any() or labels["candidate_id"].duplicated().any():
        raise ValueError("readiness requires unique non-null candidate IDs")
    indexed = labels.copy()
    indexed.index = indexed["candidate_id"].astype(str)
    available = _available_labels(indexed)
    censored = _censored_labels(indexed)
    per_fold = []
    for fold in folds.folds:
        per_fold.append(
            {
                "fold_index": fold.fold_index,
                "train_days": list(fold.train_days),
                "test_days": list(fold.test_days),
                "valid": fold.valid,
                "invalid_reason": fold.invalid_reason,
                "training_prevalence": fold.training_prevalence,
                "boundary_setup_count": len(fold.excluded_boundary_setup_ids),
                "train": _partition_facts(indexed, available, censored, fold, "train"),
                "test": _partition_facts(indexed, available, censored, fold, "test"),
            }
        )
    return {
        "schema_version": "research_labeled_fold_readiness_v1",
        "candidate_count": len(indexed),
        "available_label_count": int(available.sum()),
        "unavailable_or_censored_count": int((~available).sum()),
        "censored_count": int(censored.sum()),
        "fold_count": len(folds.folds),
        "valid_fold_count": sum(fold.valid for fold in folds.folds),
        "invalid_reasons": dict(Counter(
            fold.invalid_reason for fold in folds.folds if not fold.valid
        )),
        "fold_status": folds.status,
        "per_fold": per_fold,
        "counts_reconcile": True,
        "statistical_power_established": False,
    }


def build_research_fold_readiness(
    labels: pd.DataFrame,
    *,
    authorized_trading_days: tuple[str, ...],
    regime_frame: pd.DataFrame | None = None,
    resolved_input_features: tuple[str, ...] | None = None,
    resolved_cluster_count: int | None = None,
) -> tuple[ContextFoldSet, dict]:
    """Use the real logical-calendar fold path, optionally previewing candidate R6 sample floors."""

    days = tuple(authorized_trading_days)
    folds = build_context_folds(
        labels, authorized_trading_days=days, purge_from_logical_test_start=True
    )
    facts = summarize_research_folds(labels, folds)
    facts.update(
        authorized_trading_days=list(days),
        calendar_day_count=len(days),
        days_without_candidates=sorted(set(days) - set(labels["trading_day"].astype(str))),
        purge_from_logical_test_start=True,
    )
    if resolved_input_features is not None:
        from ..ml.regime_contracts import (
            REGIME_PROPOSED_DEFAULTS,
            ObservationGranularity,
        )
        from ..ml.regime_sample_adequacy import preview_sample_adequacy

        if not resolved_input_features:
            raise ValueError("candidate regime preview requires at least one selected feature")
        count = resolved_cluster_count
        if count is None:
            count = int(REGIME_PROPOSED_DEFAULTS["fixed_cluster_count"]["value"])
        preview = preview_sample_adequacy(
            labels if regime_frame is None else regime_frame,
            folds,
            observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
            resolved_cluster_count=count,
            resolved_input_features=tuple(resolved_input_features),
        )
        facts["candidate_regime_sample_adequacy"] = preview.model_dump(mode="json")
    return folds, facts
