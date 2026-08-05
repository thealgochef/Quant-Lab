"""Application service for verified, deterministic IFVG context experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from .artifact_io import VerifiedIfvgPair, load_verified_label_source_bars
from .context_experiment_contracts import (
    IfvgContextExperimentConfig,
    canonical_contract_sha256,
)
from .context_feature_view import CandidateFeatureView
from .context_reporting import execute_context_experiment
from .context_run_store import (
    CONTEXT_RUN_CATALOG,
    CONTEXT_RUN_STORE,
    CONTEXT_VIEW_STORE,
    StoredContextRun,
    list_context_run_catalog,
    load_candidate_feature_view_frame,
    load_context_experiment_run,
    save_candidate_feature_view,
    save_context_experiment_run,
    update_context_run_catalog,
)

__all__ = [
    "CONTEXT_COHORT_REGISTRY",
    "CatalogedContextExperiment",
    "context_cohort_filters",
    "context_cohort_label",
    "describe_observation_cohort",
    "run_and_catalog_context_experiment",
]


CONTEXT_COHORT_REGISTRY = MappingProxyType(
    {
        "all_development": {
            "label": "All development evidence",
            "start": None,
            "end": None,
        },
        "prior_research": {
            "label": "Prior research (through Apr 30)",
            "start": None,
            "end": "2026-04-30",
        },
        "exposed_development": {
            "label": "Exposed development (May 1–Jun 10)",
            "start": "2026-05-01",
            "end": "2026-06-10",
        },
    }
)


@dataclass(frozen=True, slots=True)
class CatalogedContextExperiment:
    stored_run: StoredContextRun
    view_manifest: dict[str, Any]
    reused_view: bool
    reused_run: bool

    @property
    def run_manifest_sha256(self) -> str:
        return str(self.stored_run.manifest["manifest_payload_sha256"])

    @property
    def view_manifest_sha256(self) -> str:
        return str(self.view_manifest["manifest_payload_sha256"])


def context_cohort_label(cohort: str) -> str:
    try:
        return str(CONTEXT_COHORT_REGISTRY[cohort]["label"])
    except KeyError as error:
        raise ValueError(f"unregistered IFVG context cohort {cohort!r}") from error


def context_cohort_filters(
    view: CandidateFeatureView,
    cohort: str,
) -> dict[str, tuple[str, ...]]:
    try:
        definition = CONTEXT_COHORT_REGISTRY[cohort]
    except KeyError as error:
        raise ValueError(f"unregistered IFVG context cohort {cohort!r}") from error
    days = tuple(sorted(view.frame["trading_day"].dropna().astype(str).unique()))
    start = definition["start"]
    end = definition["end"]
    if start is not None:
        days = tuple(day for day in days if day >= start)
    if end is not None:
        days = tuple(day for day in days if day <= end)
    return {"trading_day": days}


def describe_observation_cohort(filters: dict[str, tuple[str, ...]]) -> str:
    days = tuple(filters.get("trading_day", ()))
    if not days:
        return "custom"
    for cohort in CONTEXT_COHORT_REGISTRY:
        definition = CONTEXT_COHORT_REGISTRY[cohort]
        start = definition["start"]
        end = definition["end"]
        if start is None and end == "2026-04-30" and max(days) <= end:
            return cohort
        if start == "2026-05-01" and min(days) >= start and max(days) <= str(end):
            return cohort
    return "all_development"


def _assert_frame_matches(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as error:
        raise ValueError(f"existing immutable {label} differs from deterministic output") from error


def _persist_and_verify_view(
    view: CandidateFeatureView,
    *,
    base_dir: Path,
) -> tuple[dict[str, Any], bool]:
    reused = False
    try:
        save_candidate_feature_view(view, base_dir=base_dir)
    except FileExistsError:
        reused = True
    frame, manifest = load_candidate_feature_view_frame(view.view_id, base_dir=base_dir)
    _assert_frame_matches(frame, view.frame, label="candidate feature view")
    if (
        manifest.get("artifact_pair_hash") != view.artifact_pair_hash
        or manifest.get("feature_registry_hash") != view.feature_registry_hash
    ):
        raise ValueError("existing immutable candidate feature view identity differs")
    return manifest, reused


def _persist_and_verify_run(
    config: IfvgContextExperimentConfig,
    execution,
    *,
    base_dir: Path,
) -> tuple[StoredContextRun, bool]:
    predictions = (
        None if execution.model_run is None else execution.model_run.predictions
    )
    importance = (
        None if execution.model_run is None else execution.model_run.feature_importance
    )
    reused = False
    try:
        save_context_experiment_run(
            config,
            execution.result,
            predictions=predictions,
            feature_importance=importance,
            base_dir=base_dir,
        )
    except FileExistsError:
        reused = True
    stored = load_context_experiment_run(execution.result.run_id, base_dir=base_dir)
    if (
        canonical_contract_sha256(stored.config) != canonical_contract_sha256(config)
        or canonical_contract_sha256(stored.result)
        != canonical_contract_sha256(execution.result)
    ):
        raise ValueError("existing immutable context run contract differs")
    _assert_frame_matches(
        stored.predictions,
        pd.DataFrame() if predictions is None else predictions,
        label="OOS prediction stream",
    )
    _assert_frame_matches(
        stored.feature_importance,
        pd.DataFrame() if importance is None else importance,
        label="feature importance",
    )
    return stored, reused


def run_and_catalog_context_experiment(
    pair: VerifiedIfvgPair,
    config: IfvgContextExperimentConfig,
    *,
    display_name: str | None = None,
    notes: str | None = None,
    view_store: Path = CONTEXT_VIEW_STORE,
    run_store: Path = CONTEXT_RUN_STORE,
    catalog_path: Path = CONTEXT_RUN_CATALOG,
) -> CatalogedContextExperiment:
    """Execute, immutably verify, and only then catalog one registered run."""

    if config.dataset.artifact_pair != pair.reference:
        raise ValueError("experiment config does not reference the verified artifact pair")
    bars = load_verified_label_source_bars(pair.v2)
    execution = execute_context_experiment(pair, config, bars)
    view_manifest, reused_view = _persist_and_verify_view(
        execution.view,
        base_dir=Path(view_store),
    )
    stored, reused_run = _persist_and_verify_run(
        config,
        execution,
        base_dir=Path(run_store),
    )
    update_context_run_catalog(
        stored.result.run_id,
        display_name=display_name,
        notes=notes,
        catalog_path=Path(catalog_path),
    )
    cataloged = {
        item["run_id"]: item
        for item in list_context_run_catalog(catalog_path=Path(catalog_path))
    }
    if stored.result.run_id not in cataloged:
        raise ValueError("verified immutable context run was not cataloged")
    return CatalogedContextExperiment(
        stored_run=stored,
        view_manifest=view_manifest,
        reused_view=reused_view,
        reused_run=reused_run,
    )
