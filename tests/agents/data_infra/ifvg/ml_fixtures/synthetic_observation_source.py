"""R6.1 — persisted synthetic observation sources for the regime seam (§6.D).

Every real regime path loads its observations through the verified store
(``bundle_feature_views`` / ``context_bar_panels``) and its folds through a
persisted fold-set artifact. These builders persist the synthetic fixtures
to a caller-supplied tmp root FIRST and hand back the exact ids, so the
tests exercise the same seam the pipeline uses (never an in-memory frame
and never a caller string).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (
    build_bundle_feature_view,
    save_bundle_feature_view,
)
from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_contract import (
    CONTEXT_BAR_PANEL_BUNDLE_KEY,
    CONTEXT_BAR_PANEL_CATEGORICAL_FEATURES,
    CONTEXT_BAR_PANEL_FEATURES,
    PANEL_AS_OF_POLICY_ID_V1,
)
from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_materializer import (
    materialize_context_bar_panel,
    save_context_bar_panel_artifact,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.fold_schedules import (
    build_candidate_folds_from_schedule,
    build_context_bar_panel_folds,
    derive_fold_schedule,
)
from alpha_lab.agents.data_infra.ifvg.ml.fold_set_artifact import (
    build_fold_set_artifact,
    persist_fold_schedule,
    persist_fold_set_artifact,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import ObservationGranularity
from alpha_lab.agents.data_infra.ifvg.ml.regime_observation_source import (
    RegimeObservationSourceRef,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import resolve_kmeans_protocol
from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (
    load_verified_replay_chart_artifact,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
    REGIME_INPUT_FEATURES,
    known_cluster_fixture,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_context_panel import (
    synthetic_label_source_1m,
    synthetic_pair_ref,
    synthetic_trading_days,
    write_synthetic_replay_chart_artifact,
)

__all__ = [
    "PANEL_NUMERIC_FEATURES",
    "PersistedCandidateSource",
    "PersistedPanelSource",
    "persisted_candidate_source",
    "persisted_panel_source",
]

#: The six numeric panel features (the categorical session state is never a
#: KMeans input).
PANEL_NUMERIC_FEATURES: tuple[str, ...] = tuple(
    name
    for name in CONTEXT_BAR_PANEL_FEATURES
    if name not in CONTEXT_BAR_PANEL_CATEGORICAL_FEATURES
)


@dataclass(frozen=True)
class PersistedCandidateSource:
    root: Path
    fixture: object
    bundle: object
    view_envelope: object
    view_frame: pd.DataFrame
    source_ref: RegimeObservationSourceRef
    schedule: object
    folds: object
    fold_set_envelope: object
    protocol: object


@dataclass(frozen=True)
class PersistedPanelSource:
    root: Path
    replay_base: Path
    pair: object
    replay_artifact_id: str
    panel_envelope: object
    panel_frame: pd.DataFrame
    validity_frame: pd.DataFrame
    source_ref: RegimeObservationSourceRef
    schedule: object
    folds: object
    fold_set_envelope: object
    protocol: object
    candidate_frame: pd.DataFrame
    days: tuple[str, ...]
    #: R6.1 F1: the PERSISTED candidate bundle view (known-cluster view over
    #: the same 55 days; 09:MM ET entries inside the panel's bar coverage)
    #: whose as-of instants the panel grain assigns, and its verified ref.
    candidate_view_envelope: object = None
    candidate_source_ref: RegimeObservationSourceRef | None = None


def persisted_candidate_source(
    root: Path,
    *,
    n: int = 600,
    k: int = 3,
    minimum_train_rows: int = 150,
    bundle_key: str = "B0_CORE",
) -> PersistedCandidateSource:
    """Known-cluster candidate view → persisted bundle view + schedule +
    label-free fold-set artifact + the candidate-grain protocol."""

    root = Path(root)
    fixture = known_cluster_fixture(k=k, n=n)
    bundle = resolve_bundle(bundle_key)
    envelope, frame = build_bundle_feature_view(fixture.view, bundle)
    stored = save_bundle_feature_view(root, envelope, frame)
    schedule = derive_fold_schedule(fixture.trading_days)
    persist_fold_schedule(root, schedule)
    folds = build_candidate_folds_from_schedule(
        fixture.view.frame,
        authorized_trading_days=fixture.trading_days,
        minimum_train_rows=minimum_train_rows,
    )
    fold_set, definitions = build_fold_set_artifact(
        folds,
        schedule=schedule,
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id=stored.bundle_feature_view_id,
        minimum_train_observations=minimum_train_rows,
        labeled=False,
    )
    persist_fold_set_artifact(root, fold_set, definitions)
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=bundle.resolved_feature_bundle_id,
        resolved_input_features=REGIME_INPUT_FEATURES,
    )
    return PersistedCandidateSource(
        root=root,
        fixture=fixture,
        bundle=bundle,
        view_envelope=stored,
        view_frame=frame,
        source_ref=RegimeObservationSourceRef(
            source_kind="bundle_feature_view", artifact_id=stored.bundle_feature_view_id
        ),
        schedule=schedule,
        folds=folds,
        fold_set_envelope=fold_set,
        protocol=protocol,
    )


def persisted_panel_source(
    root: Path,
    *,
    interval: int = 300,
    day_count: int = 55,
) -> PersistedPanelSource:
    """Synthetic replay-chart artifact → materialized + persisted 5m/15m
    panel + schedule + panel-native fold-set artifact + the panel-grain
    protocol, plus one candidate as-of instant per trading day (10:03 ET,
    three minutes after a completed bar)."""

    root = Path(root)
    days = synthetic_trading_days(day_count)
    bars = synthetic_label_source_1m(days)
    pair = synthetic_pair_ref(seed=f"panel-{interval}")
    replay_base = root / "replay_chart"
    artifact_id = write_synthetic_replay_chart_artifact(replay_base, bars, pair)
    replay = load_verified_replay_chart_artifact(replay_base, artifact_id, expected_pair=pair)
    envelope, panel, validity = materialize_context_bar_panel(
        replay, panel_interval_seconds=interval
    )
    stored, _reused = save_context_bar_panel_artifact(root, envelope, panel, validity)
    schedule = derive_fold_schedule(days)
    persist_fold_schedule(root, schedule)
    folds = build_context_bar_panel_folds(panel, authorized_trading_days=days)
    fold_set, definitions = build_fold_set_artifact(
        folds,
        schedule=schedule,
        observation_grain=ObservationGranularity.CONTEXT_BAR_PANEL,
        observation_source_artifact_id=stored.context_bar_panel_artifact_id,
        minimum_train_observations=300,
        labeled=False,
    )
    persist_fold_set_artifact(root, fold_set, definitions)
    bundle = resolve_bundle(CONTEXT_BAR_PANEL_BUNDLE_KEY)
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=bundle.resolved_feature_bundle_id,
        resolved_input_features=PANEL_NUMERIC_FEATURES,
        observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
        panel_interval_seconds=interval,
        panel_source_artifact_id=stored.context_bar_panel_artifact_id,
        panel_as_of_policy_id=PANEL_AS_OF_POLICY_ID_V1,
    )
    candidate_frame = pd.DataFrame(
        {
            "candidate_id": [f"pc_{index:03d}" for index in range(len(days))],
            "setup_id": [f"pcsetup_{index:03d}" for index in range(len(days))],
            "trading_day": list(days),
            # 10:03 ET = 15:03 UTC in January (EST): 3 s past a completed 5m bar
            "entry_ts_utc": [f"{day}T15:03:00Z" for day in days],
        }
    )
    # the persisted candidate bundle view the executor verified-loads as the
    # panel grain's as-of source (its days equal the panel's 55 days)
    candidate_fixture = known_cluster_fixture(k=3, n=600)
    if tuple(candidate_fixture.trading_days) != tuple(days):
        raise ValueError("the candidate view days must equal the panel days")
    candidate_bundle = resolve_bundle("B0_CORE")
    candidate_envelope, candidate_view_frame = build_bundle_feature_view(
        candidate_fixture.view, candidate_bundle
    )
    stored_candidate_view = save_bundle_feature_view(
        root, candidate_envelope, candidate_view_frame
    )
    return PersistedPanelSource(
        root=root,
        replay_base=replay_base,
        pair=pair,
        replay_artifact_id=artifact_id,
        panel_envelope=stored,
        panel_frame=panel,
        validity_frame=validity,
        source_ref=RegimeObservationSourceRef(
            source_kind="context_bar_panel", artifact_id=stored.context_bar_panel_artifact_id
        ),
        schedule=schedule,
        folds=folds,
        fold_set_envelope=fold_set,
        protocol=protocol,
        candidate_frame=candidate_frame,
        days=days,
        candidate_view_envelope=stored_candidate_view,
        candidate_source_ref=RegimeObservationSourceRef(
            source_kind="bundle_feature_view",
            artifact_id=stored_candidate_view.bundle_feature_view_id,
        ),
    )
