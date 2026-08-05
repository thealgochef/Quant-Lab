"""Candidate-specific geometry joins and quarantine helpers."""

from __future__ import annotations

import pandas as pd

__all__ = ["attach_candidate_geometry", "quarantine_missing_geometry"]


def attach_candidate_geometry(
    candidates: pd.DataFrame,
    dossiers: pd.DataFrame,
) -> pd.DataFrame:
    """One-to-one candidate join. ``setup_id`` is never a fallback key."""
    for name, frame in (("candidates", candidates), ("dossiers", dossiers)):
        if "candidate_id" not in frame.columns:
            raise ValueError(f"{name} must carry candidate_id; setup_id fallback is forbidden")
        if frame["candidate_id"].isna().any():
            raise ValueError(f"{name}.candidate_id contains null values")
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"{name}.candidate_id must be unique")
    geometry_columns = [
        column
        for column in dossiers.columns
        if column != "setup_id" or column not in candidates.columns
    ]
    joined = candidates.merge(
        dossiers[geometry_columns],
        on="candidate_id",
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_geometry"),
    )
    missing = joined["_merge"] != "both"
    if missing.any():
        ids = sorted(joined.loc[missing, "candidate_id"].astype(str))
        raise ValueError(f"missing candidate-specific geometry for {ids[:5]}")
    return joined.drop(columns=["_merge"])


def quarantine_missing_geometry(
    candidates: pd.DataFrame,
    dossiers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete candidates and an explicit incomplete-evidence quarantine."""
    if "candidate_id" not in candidates or "candidate_id" not in dossiers:
        raise ValueError("candidate_id is required for geometry quarantine")
    complete_ids = set(dossiers["candidate_id"].dropna().astype(str))
    missing = ~candidates["candidate_id"].astype(str).isin(complete_ids)
    quarantine = candidates.loc[missing, ["candidate_id", "setup_id"]].copy()
    quarantine["reasons"] = "geometry_incomplete"
    return candidates.loc[~missing].copy(), quarantine
