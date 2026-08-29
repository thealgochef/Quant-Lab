"""The regime executor — S09a's body (R6.1 §6.D / D14).

``execute_regime_protocol`` persists the protocol, verified-loads the fold
set (its grain/key must agree with the protocol), verified-loads the
observations through the provenance seam, runs the kernel, persists every
fold fit (verified reuse by reproduction — ``regime_store.persist_regime_fit``)
and the capability assessment, and builds + saves the DESCRIPTIVE
``RegimeOosAssignmentArtifact`` (candidate grain from the fits' OOS rows;
panel grain through the normative PIT rule at the requested as-of stage).
Every bound id comes from a loaded envelope; the executor never accepts a
frame — the panel grain's candidate as-of instants come from a
VERIFIED-LOADED candidate bundle view named by a
:class:`RegimeObservationSourceRef` (adversarial R6.1 F1: a caller string
paired with an in-memory frame is not provenance). Zero valid folds is the
designed safe failure: the assessment is persisted with its typed gate
failures and no fit exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..features.context_bar_panel_contract import PANEL_ASSIGNMENT_MAX_STALENESS_INTERVALS
from ..features.feature_blocks import AvailabilityStage
from .fold_set_artifact import load_fold_set_artifact, observation_key_for_grain
from .regime_contracts import ObservationGranularity, RegimeProtocolEnvelope
from .regime_observation_source import (
    RegimeObservationSourceRef,
    VerifiedRegimeObservations,
    load_regime_observations,
    run_regime_protocol_from_source,
)
from .regime_oos_assignment import (
    PanelAssignmentContext,
    RegimeOosAssignmentEnvelope,
    assign_panel_regimes_to_candidates,
    build_regime_oos_assignment_artifact,
    candidate_as_of_frame,
    candidate_fold_oos_assignment,
    save_regime_oos_assignment,
)
from .regime_service import RegimeProtocolRun
from .regime_store import (
    persist_regime_assessment,
    persist_regime_fit,
    persist_regime_protocol,
)

__all__ = [
    "RegimeExecutionResult",
    "candidate_as_of_source_reference",
    "execute_regime_protocol",
    "load_candidate_as_of_source",
]


def candidate_as_of_source_reference(observations: VerifiedRegimeObservations) -> str:
    """The provenance line of a loaded observation source —
    ``<source_kind>:<artifact id>`` from the LOADED envelope, never a caller
    string (the payload field is pattern-checked against exactly this form)."""

    return f"{observations.source_kind}:{observations.source_artifact_ids[0]}"


def load_candidate_as_of_source(
    root: Path, candidate_as_of_source: object
) -> VerifiedRegimeObservations:
    """Verified-load the candidate bundle view whose as-of instants the panel
    grain assigns; anything that is not a ``bundle_feature_view``
    :class:`RegimeObservationSourceRef` is refused (typed)."""

    if not isinstance(candidate_as_of_source, RegimeObservationSourceRef):
        raise TypeError(
            "candidate_as_of_source must be a RegimeObservationSourceRef of kind "
            "bundle_feature_view (a caller string / in-memory frame is not evidence); got "
            f"{type(candidate_as_of_source).__name__}"
        )
    if candidate_as_of_source.source_kind != "bundle_feature_view":
        raise ValueError(
            "the candidate as-of source must be a persisted bundle_feature_view; got "
            f"{candidate_as_of_source.source_kind}"
        )
    return load_regime_observations(Path(root), candidate_as_of_source)


@dataclass(frozen=True, slots=True)
class RegimeExecutionResult:
    protocol: RegimeProtocolEnvelope
    run: RegimeProtocolRun
    fold_set_artifact_id: str
    fold_schedule_id: str
    regime_fit_ids: tuple[str, ...]
    regime_capability_assessment_id: str
    oos_assignment: RegimeOosAssignmentEnvelope
    oos_assignment_frame: pd.DataFrame
    source_artifact_ids: tuple[str, ...]
    observation_matrix_hash: str
    fits_reused: tuple[bool, ...]


def execute_regime_protocol(
    store_root: Path,
    *,
    protocol: RegimeProtocolEnvelope,
    observation_source: RegimeObservationSourceRef,
    fold_set_artifact_id: str,
    candidate_as_of_source: RegimeObservationSourceRef | None = None,
    candidate_as_of_stage: AvailabilityStage = AvailabilityStage.ENTRY_DECISION,
    bootstrap_refits: int = 50,
) -> RegimeExecutionResult:
    """Persist → load fold set → load observations → run → persist fits and
    assessment → descriptive OOS assignment artifact.

    ``candidate_as_of_source`` (panel grain only) names the PERSISTED
    candidate bundle view whose as-of instants the panel regimes are assigned
    to; it is verified-loaded through the observation seam and the artifact's
    provenance line is stamped from the loaded envelope. The candidate grain
    derives the as-of instants from the loaded observations themselves and
    refuses the parameter.
    """

    root = Path(store_root)
    payload = protocol.payload
    persist_regime_protocol(root, protocol)
    fold_set, folds = load_fold_set_artifact(root, fold_set_artifact_id)
    grain = ObservationGranularity(payload.observation_granularity)
    if fold_set.payload.observation_grain is not grain:
        raise ValueError(
            f"fold set grain {fold_set.payload.observation_grain.value} does not match "
            f"the protocol grain {grain.value}"
        )
    if fold_set.payload.observation_key != observation_key_for_grain(grain):
        raise ValueError("fold set observation key does not match the protocol grain")
    observations = load_regime_observations(root, observation_source)
    if fold_set.payload.observation_source_artifact_id not in observations.source_artifact_ids:
        raise ValueError(
            "the fold set was built over a different observation source than the one loaded"
        )
    run = run_regime_protocol_from_source(
        observations, folds, protocol, bootstrap_refits=bootstrap_refits
    )
    if run.fold_set_id != fold_set.payload.fold_set_id:
        raise ValueError("the run's fold_set_id disagrees with the loaded fold-set artifact")
    reused: list[bool] = []
    for fold_fit in run.fold_fits:
        _artifact, was_reused = persist_regime_fit(
            root,
            fold_fit,
            run.assignments[run.assignments["fold_index"] == fold_fit.fold_index],
            observation_frame=observations.frame,
            return_reuse=True,
        )
        reused.append(bool(was_reused))
    persist_regime_assessment(root, run.assessment)
    fit_ids = tuple(fit.fit_envelope.regime_fit_id for fit in run.fold_fits)

    if grain is ObservationGranularity.CONTEXT_BAR_PANEL:
        if candidate_as_of_source is None:
            raise ValueError("the panel grain requires the candidate as-of source")
        candidates = load_candidate_as_of_source(root, candidate_as_of_source)
        as_of = candidate_as_of_frame(candidates.frame, stage=candidate_as_of_stage)
        interval = int(payload.panel_interval_seconds or 0)
        max_staleness = interval * PANEL_ASSIGNMENT_MAX_STALENESS_INTERVALS
        frame = assign_panel_regimes_to_candidates(
            observations.frame,
            run.assignments,
            as_of,
            protocol=protocol,
            max_staleness_seconds=max_staleness,
        )
        context = PanelAssignmentContext(
            context_bar_panel_artifact_id=str(payload.panel_source_artifact_id),
            panel_as_of_policy_id=str(payload.panel_as_of_policy_id),
            candidate_as_of_stage=candidate_as_of_stage,
            panel_interval_seconds=interval,
            max_staleness_seconds=max_staleness,
        )
        as_of_ref = candidate_as_of_source_reference(candidates)
    else:
        if candidate_as_of_source is not None:
            raise ValueError(
                "the candidate grain derives its as-of instants from the loaded "
                "observations; candidate_as_of_source is legal for the panel grain only"
            )
        key_column = "candidate_id"
        candidate_ids = tuple(observations.frame[key_column].astype(str))
        frame = candidate_fold_oos_assignment(run.assignments, candidate_ids)
        as_of = candidate_as_of_frame(observations.frame, stage=candidate_as_of_stage)
        context = None
        as_of_ref = candidate_as_of_source_reference(observations)
    envelope, table = build_regime_oos_assignment_artifact(
        frame,
        protocol=protocol,
        regime_fit_ids=fit_ids,
        regime_fold_set_id=fold_set.payload.fold_set_id,
        fold_schedule_id=fold_set.payload.fold_schedule_id,
        consulted_assignments=run.assignments,
        candidate_as_of=as_of,
        candidate_as_of_source_ref=as_of_ref,
        panel_context=context,
    )
    save_regime_oos_assignment(root, envelope, table)
    return RegimeExecutionResult(
        protocol=protocol,
        run=run,
        fold_set_artifact_id=fold_set.fold_set_artifact_id,
        fold_schedule_id=fold_set.payload.fold_schedule_id,
        regime_fit_ids=fit_ids,
        regime_capability_assessment_id=run.assessment.regime_capability_assessment_id,
        oos_assignment=envelope,
        oos_assignment_frame=frame,
        source_artifact_ids=observations.source_artifact_ids,
        observation_matrix_hash=observations.observation_matrix_hash,
        fits_reused=tuple(reused),
    )
