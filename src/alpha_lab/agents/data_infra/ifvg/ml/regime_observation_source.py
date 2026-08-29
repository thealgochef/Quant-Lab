"""Verified-load provenance for regime observations (R6.1 D2 / §6.D).

Every regime input is a VERIFIED-LOADED artifact: candidate grain → the
persisted ``BundleFeatureViewEnvelope`` (store ``bundle_feature_views``);
panel grain → the persisted ``ContextBarPanelArtifactEnvelope`` (store
``context_bar_panels``). ``source_artifact_ids`` always come from the loaded
envelope — never from a caller string (audit S3; the kickoff's "a
caller-provided string is NOT evidence"). The observation matrix hash binds
the exact values that will be fitted; :func:`run_regime_protocol_from_source`
refuses a grain mismatch and, for the panel grain, a protocol whose pinned
panel id / interval / as-of policy disagree with the loaded artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ..context_folds import ContextFoldSet
from ..features.bundle_feature_view import (
    load_bundle_feature_view,
    load_bundle_feature_view_frame,
)
from ..features.context_bar_panel_materializer import (
    load_context_bar_panel_artifact,
    load_context_bar_panel_frame,
)
from ..search.identities import SHA256_PATTERN, FrozenContract, canonical_contract_sha256
from .regime_contracts import ObservationGranularity, RegimeProtocolEnvelope
from .regime_preprocessing import keyed_observations

__all__ = [
    "RegimeObservationSourceRef",
    "VerifiedRegimeObservations",
    "observation_matrix_hash",
    "load_regime_observations",
    "assert_source_matches_protocol",
    "run_regime_protocol_from_source",
]


class RegimeObservationSourceRef(FrozenContract):
    source_kind: Literal["bundle_feature_view", "context_bar_panel"]
    artifact_id: str = Field(pattern=SHA256_PATTERN)

    @property
    def observation_granularity(self) -> ObservationGranularity:
        if self.source_kind == "context_bar_panel":
            return ObservationGranularity.CONTEXT_BAR_PANEL
        return ObservationGranularity.CANDIDATE_STAGE_ROW


@dataclass(frozen=True, slots=True)
class VerifiedRegimeObservations:
    """Observations whose provenance was LOADED and verified, never asserted."""

    observation_granularity: ObservationGranularity
    source_kind: str
    source_artifact_ids: tuple[str, ...]
    frame: pd.DataFrame
    observation_matrix_hash: str
    provenance: dict
    #: the loaded panel envelope (panel grain only) for coherence checks
    panel_envelope: object | None = None


def observation_matrix_hash(frame: pd.DataFrame, features: tuple[str, ...] | None = None) -> str:
    """Canonical hash of the keyed observation values (all columns when
    ``features`` is None); NaN → null; row order irrelevant."""

    indexed = keyed_observations(frame)
    columns = list(features) if features is not None else [
        column for column in indexed.columns if column != indexed.index.name
    ]
    rows: dict[str, list] = {}
    value_rows = indexed.loc[:, columns].itertuples(index=False)
    for row_id, values in zip(indexed.index, value_rows, strict=True):
        cleaned = []
        for value in values:
            if isinstance(value, float | np.floating):
                cleaned.append(None if not np.isfinite(value) else float(value))
            elif value is None or (isinstance(value, str) and value == ""):
                cleaned.append(value)
            else:
                try:
                    missing = pd.isna(value)
                except (TypeError, ValueError):
                    missing = False
                if missing is True:
                    cleaned.append(None)
                elif isinstance(value, str | int | bool):
                    cleaned.append(value)
                else:
                    cleaned.append(str(value))
        rows[str(row_id)] = cleaned
    return canonical_contract_sha256({"columns": columns, "rows": rows})


def load_regime_observations(
    root: Path, ref: RegimeObservationSourceRef
) -> VerifiedRegimeObservations:
    """Verified store load → frame + provenance (ids from the envelope)."""

    root = Path(root)
    if ref.source_kind == "bundle_feature_view":
        envelope = load_bundle_feature_view(root, ref.artifact_id)
        frame = load_bundle_feature_view_frame(root, envelope)
        payload = envelope.payload
        return VerifiedRegimeObservations(
            observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
            source_kind=ref.source_kind,
            source_artifact_ids=(envelope.bundle_feature_view_id,),
            frame=frame,
            observation_matrix_hash=observation_matrix_hash(frame),
            provenance={
                "bundle_feature_view_id": envelope.bundle_feature_view_id,
                "view_id": payload.view_id,
                "feature_bundle_key": payload.feature_bundle_key,
                "resolved_feature_bundle_id": payload.resolved_feature_bundle_id,
                "mbp1_feature_artifact_id": payload.mbp1_feature_artifact_id,
                "regime_fold_feature_artifact_id": payload.regime_fold_feature_artifact_id,
                "frame_table_sha256": envelope.frame_table_sha256,
            },
        )
    envelope = load_context_bar_panel_artifact(root, ref.artifact_id)
    frame = load_context_bar_panel_frame(root, envelope)
    payload = envelope.payload
    return VerifiedRegimeObservations(
        observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
        source_kind=ref.source_kind,
        source_artifact_ids=(envelope.context_bar_panel_artifact_id,),
        frame=frame,
        observation_matrix_hash=observation_matrix_hash(frame),
        provenance={
            "context_bar_panel_artifact_id": envelope.context_bar_panel_artifact_id,
            "replay_chart_artifact_id": payload.replay_chart_artifact_id,
            "panel_interval_seconds": payload.panel_interval_seconds,
            "panel_as_of_policy_id": payload.panel_as_of_policy_id,
            "resolved_feature_block_id": payload.resolved_feature_block_id,
            "panel_table_sha256": envelope.panel_table_sha256,
        },
        panel_envelope=envelope,
    )


def assert_source_matches_protocol(
    source: VerifiedRegimeObservations, protocol: RegimeProtocolEnvelope
) -> None:
    payload = protocol.payload
    if payload.observation_granularity is not source.observation_granularity:
        raise ValueError(
            f"protocol grain {payload.observation_granularity.value} does not match the "
            f"loaded observation source ({source.observation_granularity.value})"
        )
    if source.observation_granularity is ObservationGranularity.CONTEXT_BAR_PANEL:
        panel = source.panel_envelope
        if panel is None:  # pragma: no cover - loader invariant
            raise ValueError("panel observations carry no panel envelope")
        if payload.panel_source_artifact_id != panel.context_bar_panel_artifact_id:
            raise ValueError("protocol pins a different context_bar_panel_artifact_id")
        if payload.panel_interval_seconds != panel.payload.panel_interval_seconds:
            raise ValueError("protocol pins a different panel interval than the artifact")
        if payload.panel_as_of_policy_id != panel.payload.panel_as_of_policy_id:
            raise ValueError("protocol pins a different panel as-of policy than the artifact")


def run_regime_protocol_from_source(
    source: VerifiedRegimeObservations,
    folds: ContextFoldSet,
    protocol: RegimeProtocolEnvelope,
    *,
    bootstrap_refits: int = 50,
):
    """The kernel run over VERIFIED observations (the seam every real path
    uses); ``source_artifact_ids`` come from the loaded envelope."""

    from .regime_service import run_regime_protocol  # noqa: PLC0415

    assert_source_matches_protocol(source, protocol)
    return run_regime_protocol(
        source.frame,
        folds,
        protocol,
        source_artifact_ids=source.source_artifact_ids,
        bootstrap_refits=bootstrap_refits,
    )
