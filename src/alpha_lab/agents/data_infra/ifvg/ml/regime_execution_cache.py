"""Exact completed S09a research reuse through the existing regime serializers."""

from __future__ import annotations

import json
from io import BytesIO
from typing import ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
)
from ..search.store import load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope
from .regime_contracts import ObservationGranularity
from .regime_observation_source import assert_source_matches_protocol
from .regime_oos_assignment import load_regime_oos_assignment, load_regime_oos_assignment_frame
from .regime_preprocessing import (
    FittedRegimePreprocessing,
    keyed_observations,
    training_feature_matrix_hash,
)
from .regime_service import RegimeFoldFit, RegimeProtocolRun
from .regime_store import load_regime_assessment, load_regime_fit, load_regime_fit_assignments
from .research_evidence import _source_identity, frame_content_hash

RESEARCH_REGIME_EXECUTION_STORE = "research_regime_executions"


class ResearchRegimeExecutionPayload(FrozenContract):
    schema_version: Literal[1] = 1
    request_json: str


class ResearchRegimeExecutionEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "regime_execution_request_id"
    regime_execution_request_id: str = Field(pattern=SHA256_PATTERN)
    payload: ResearchRegimeExecutionPayload


def regime_execution_request(
    root,
    *,
    protocol,
    observations,
    fold_set,
    candidate_as_of_source,
    candidate_as_of_stage,
    bootstrap_refits,
):
    from .regime_executor import load_candidate_as_of_source  # noqa: PLC0415

    assert_source_matches_protocol(observations, protocol)
    panel = protocol.payload.observation_granularity is ObservationGranularity.CONTEXT_BAR_PANEL
    candidate = None
    if panel:
        if candidate_as_of_source is None:
            raise ValueError("the panel grain requires the candidate as-of source")
        loaded = load_candidate_as_of_source(root, candidate_as_of_source)
        candidate = {
            "source": candidate_as_of_source.model_dump(mode="json"),
            "values_hash": frame_content_hash(loaded.frame),
        }
    elif candidate_as_of_source is not None:
        raise ValueError("candidate as-of source is legal for the panel grain only")
    request = {
        "protocol_id": protocol.resolved_regime_protocol_id,
        "source_artifact_ids": list(observations.source_artifact_ids),
        "observation_source": {
            "source_kind": observations.source_kind,
            "artifact_id": observations.source_artifact_ids[0],
        },
        "observation_matrix_hash": observations.observation_matrix_hash,
        "observation_values_hash": frame_content_hash(observations.frame),
        "fold_set_artifact_id": fold_set.fold_set_artifact_id,
        "fold_set_id": fold_set.payload.fold_set_id,
        "fold_schedule_id": fold_set.payload.fold_schedule_id,
        "candidate_as_of_source": candidate,
        "candidate_as_of_stage": str(candidate_as_of_stage),
        "bootstrap_refits": int(bootstrap_refits),
        "implementation": _source_identity(),
    }
    return ResearchRegimeExecutionEnvelope.from_payload(
        ResearchRegimeExecutionPayload(
            request_json=json.dumps(request, sort_keys=True, separators=(",", ":"))
        )
    )


def persist_completed_regime_execution(root, envelope, result):
    facts = [
        {
            "regime_fit_id": fit.fit_envelope.regime_fit_id,
            "training_row_ids": list(fit.preprocessing.training_row_ids),
            "training_rows_all_missing": list(fit.preprocessing.training_rows_all_missing),
        }
        for fit in result.run.fold_fits
    ]
    record = {
        "regime_fit_ids": list(result.regime_fit_ids),
        "fold_facts": facts,
        "assessment_id": result.regime_capability_assessment_id,
        "oos_assignment_id": result.oos_assignment.regime_oos_assignment_id,
        "partial_fit_reuse": False,
    }
    save_or_reuse_envelope(
        root,
        RESEARCH_REGIME_EXECUTION_STORE,
        envelope,
        extra_files={
            "execution.json": json.dumps(record, sort_keys=True).encode(),
            "run_assignments.parquet": result.run.assignments.to_parquet(index=False),
        },
    )


def load_completed_regime_execution(root, envelope, *, protocol, observations, fold_set):
    """None means no completed request; corrupt evidence refuses before fitting."""
    from .regime_executor import RegimeExecutionResult  # noqa: PLC0415

    request_id = envelope.regime_execution_request_id
    if not (root / RESEARCH_REGIME_EXECUTION_STORE / request_id).exists():
        return None
    stored = load_verified_envelope(
        root, RESEARCH_REGIME_EXECUTION_STORE, request_id, ResearchRegimeExecutionEnvelope
    )
    if stored != envelope:
        raise ValueError("completed regime execution differs from the exact requested inputs")
    request = json.loads(envelope.payload.request_json)
    for name, value in (
        ("protocol_id", protocol.resolved_regime_protocol_id),
        ("source_artifact_ids", list(observations.source_artifact_ids)),
        ("observation_matrix_hash", observations.observation_matrix_hash),
        ("observation_values_hash", frame_content_hash(observations.frame)),
        ("fold_set_artifact_id", fold_set.fold_set_artifact_id),
        ("fold_set_id", fold_set.payload.fold_set_id),
        ("fold_schedule_id", fold_set.payload.fold_schedule_id),
    ):
        if request[name] != value:
            raise ValueError(f"completed regime request differs from verified {name}")
    record = json.loads(
        load_sidecar_bytes(root, RESEARCH_REGIME_EXECUTION_STORE, request_id, "execution.json")
    )
    assessment = load_regime_assessment(root, record["assessment_id"])
    fit_ids = tuple(record["regime_fit_ids"])
    if (
        assessment.payload.resolved_regime_protocol_id != protocol.resolved_regime_protocol_id
        or assessment.payload.fold_set_id != fold_set.payload.fold_set_id
        or set(assessment.payload.regime_fit_ids) != set(fit_ids)
        or len(set(fit_ids)) != len(fit_ids)
        or assessment.payload.stability.bootstrap_refits_per_fold_requested
        not in ((request["bootstrap_refits"],) if fit_ids else (0, request["bootstrap_refits"]))
    ):
        raise ValueError(
            "completed regime assessment differs from the requested protocol/folds/bootstrap"
        )
    facts = {fact["regime_fit_id"]: fact for fact in record["fold_facts"]}
    if len(facts) != len(record["fold_facts"]) or set(facts) != set(fit_ids):
        raise ValueError("completed regime fit metadata has incomplete model coverage")
    # Verify every fit's manifest and assignment evidence before deserializing
    # any model. Reloaded transforms/distances must reproduce all valid rows.
    verified = {}
    for fit_id in fit_ids:
        fit = load_regime_fit_assignments(root, fit_id)
        if (
            fit.envelope.payload.resolved_regime_protocol_id != protocol.resolved_regime_protocol_id
            or fit.envelope.payload.source_artifact_ids != observations.source_artifact_ids
            or fit.fold_index in verified
        ):
            raise ValueError("completed regime fit has another protocol/source or repeated fold")
        verified[fit.fold_index] = fit
    indexed = keyed_observations(observations.frame)
    fold_fits = []
    for fit_id in fit_ids:
        loaded = load_regime_fit(root, fit_id)
        fact = facts[fit_id]
        bundle, parameters = loaded._bundle, loaded.parameters
        features = tuple(bundle["features"])
        training_ids = tuple(fact["training_row_ids"])
        if (
            features != protocol.payload.resolved_input_features
            or canonical_contract_sha256({"training_row_ids": sorted(training_ids)})
            != loaded.envelope.payload.training_row_ids_hash
            or training_feature_matrix_hash(indexed, training_ids, features)
            != loaded.envelope.payload.training_feature_matrix_hash
        ):
            raise ValueError(
                "completed regime model preprocessing differs from its training evidence"
            )
        preprocessing = FittedRegimePreprocessing(
            fold_index=loaded.artifact.fold_index,
            features=features,
            winsorization_policy=parameters["winsorization_policy"],
            imputer=bundle["imputer"],
            scaler=bundle["scaler"],
            clip_lower=bundle["clip_lower"],
            clip_upper=bundle["clip_upper"],
            output_feature_names=tuple(parameters["output_feature_names"]),
            parameter_payload=parameters,
            fitted_parameter_payload_hash=loaded.artifact.fitted_parameter_payload_hash,
            training_row_ids=training_ids,
            training_rows_all_missing=tuple(fact["training_rows_all_missing"]),
            training_feature_matrix_hash=loaded.envelope.payload.training_feature_matrix_hash,
        )
        assignments = verified[loaded.artifact.fold_index].frame
        valid = assignments.loc[assignments["valid"].astype(bool)]
        transformed = preprocessing.transform(indexed.loc[valid["row_id"].astype(str)])
        centroids = np.asarray(bundle["estimator"].cluster_centers_, dtype=float)
        distances = np.linalg.norm(transformed[:, None, :] - centroids[None, :, :], axis=2)
        if not np.allclose(
            distances, np.asarray(valid["distances"].tolist()), rtol=1e-12, atol=1e-12
        ) or not np.array_equal(
            np.argmin(distances, axis=1), valid["fold_local_cluster_id"].astype(int)
        ):
            raise ValueError(
                "reloaded regime model does not reproduce its saved assignment distances"
            )
        fold_fits.append(
            RegimeFoldFit(
                fit_envelope=loaded.envelope,
                fold_index=loaded.artifact.fold_index,
                preprocessing=preprocessing,
                estimator=bundle["estimator"],
                centroids_scaled=centroids,
                inertia=float(loaded.artifact.inertia_or_loglik),
                training_row_count=loaded.artifact.training_row_count,
            )
        )
    oos = load_regime_oos_assignment(root, record["oos_assignment_id"])
    oos_frame = load_regime_oos_assignment_frame(root, oos)
    if (
        oos.payload.resolved_regime_protocol_id != protocol.resolved_regime_protocol_id
        or oos.payload.fold_schedule_id != fold_set.payload.fold_schedule_id
        or oos.payload.regime_fold_set_id != fold_set.payload.fold_set_id
        or set(oos.payload.regime_fit_ids) != set(fit_ids)
        or oos.payload.candidate_as_of_stage.value != request["candidate_as_of_stage"]
    ):
        raise ValueError("completed regime OOS artifact differs from the requested execution")
    assignments = pd.read_parquet(
        BytesIO(
            load_sidecar_bytes(
                root, RESEARCH_REGIME_EXECUTION_STORE, request_id, "run_assignments.parquet"
            )
        )
    )
    run = RegimeProtocolRun(
        protocol=protocol,
        fold_fits=tuple(fold_fits),
        assignments=assignments,
        fold_set_id=fold_set.payload.fold_set_id,
        assessment=assessment,
    )
    return RegimeExecutionResult(
        protocol=protocol,
        run=run,
        fold_set_artifact_id=fold_set.fold_set_artifact_id,
        fold_schedule_id=fold_set.payload.fold_schedule_id,
        regime_fit_ids=fit_ids,
        regime_capability_assessment_id=assessment.regime_capability_assessment_id,
        oos_assignment=oos,
        oos_assignment_frame=oos_frame,
        source_artifact_ids=observations.source_artifact_ids,
        observation_matrix_hash=observations.observation_matrix_hash,
        fits_reused=tuple(True for _ in fit_ids),
        verified_fit_assignments=verified,
        regime_execution_request_id=request_id,
    )


def verify_cached_execution(root, request_id):
    """Exact-ID S15 reload, including verified model prediction reproduction."""
    from pathlib import Path  # noqa: PLC0415

    from ..features.feature_blocks import AvailabilityStage  # noqa: PLC0415
    from .fold_set_artifact import load_fold_set_artifact  # noqa: PLC0415
    from .regime_executor import load_candidate_as_of_source  # noqa: PLC0415
    from .regime_observation_source import (  # noqa: PLC0415
        RegimeObservationSourceRef,
        load_regime_observations,
    )
    from .regime_oos_assignment import (  # noqa: PLC0415
        candidate_as_of_frame,
        candidate_as_of_source_hash,
    )
    from .regime_store import load_regime_protocol  # noqa: PLC0415

    root = Path(root)
    envelope = load_verified_envelope(
        root, RESEARCH_REGIME_EXECUTION_STORE, request_id, ResearchRegimeExecutionEnvelope
    )
    request = json.loads(envelope.payload.request_json)
    protocol = load_regime_protocol(root, request["protocol_id"])
    observations = load_regime_observations(
        root, RegimeObservationSourceRef(**request["observation_source"])
    )
    fold_set, _ = load_fold_set_artifact(root, request["fold_set_artifact_id"])
    candidate = request["candidate_as_of_source"]
    if candidate is not None:
        candidates = load_candidate_as_of_source(
            root, RegimeObservationSourceRef(**candidate["source"])
        )
        if frame_content_hash(candidates.frame) != candidate["values_hash"]:
            raise ValueError("cached candidate as-of source values changed")
    else:
        candidates = observations
    result = load_completed_regime_execution(
        root, envelope, protocol=protocol, observations=observations, fold_set=fold_set
    )
    as_of = candidate_as_of_frame(
        candidates.frame, stage=AvailabilityStage(request["candidate_as_of_stage"])
    )
    if result.oos_assignment.payload.candidate_as_of_source_hash != candidate_as_of_source_hash(
        as_of
    ):
        raise ValueError("cached OOS assignment differs from its exact candidate as-of values")
    return result
