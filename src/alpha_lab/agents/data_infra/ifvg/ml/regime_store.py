"""Regime persistence on the verified store protocol (ML plan §5.3, P1-4).

Every fitted artifact persists DUAL-FORMAT beside its identity envelope:
(a) the canonical JSON parameter payload (medians, clip bounds, scaler
mean/scale) whose hash is ``fitted_parameter_payload_hash``; (b) a joblib
binary of the fitted sklearn objects for reuse. ``persist_regime_fit``
binds the assignment frame to the fit (identity, fold, protocol, and the
estimator's own labels — review F6), VERIFIES the exact bytes it is about
to publish (reload from bytes → re-transform ``np.allclose`` → re-predict
equals the persisted labels) BEFORE anything reaches the store (review
S8), publishes, then re-verifies through the manifest-checked store —
every reference is MANIFEST-RELATIVE (a relocation test copies the
artifact directory, reloads by relative refs + checksums, re-transforms,
allclose).

Promotion decisions are persisted only through
``persist_regime_promotion``, which loads the REFERENCED capability
assessment from the same store (it must exist, verify, and name the same
protocol), loads the chained previous decision when one is referenced,
and re-runs ``assert_lawful_promotion`` with the assessment's own
``gates_passed`` — a FEATURE_ELIGIBLE+ decision over a failing or absent
assessment is unpersistable (reviews F5/S1).

Trust boundary (review S7): the store verifies INTEGRITY (manifest +
sha256 of every sidecar before any byte is interpreted), not authenticity
— a store root is a trusted local directory; the joblib sidecar is
unpickled only after its hash matched the verified manifest, and the UI
never unpickles anything (it loads JSON envelopes and the Arrow frame).
"""

from __future__ import annotations

import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..search.identities import canonical_contract_sha256
from ..search.store import (
    envelope_destination,
    load_sidecar_bytes,
    load_verified_envelope,
    save_or_reuse_envelope,
)
from .regime_contracts import (
    RegimeCapabilityAssessmentEnvelope,
    RegimeFitArtifact,
    RegimeFitEnvelope,
    RegimePromotionDecisionEnvelope,
    RegimeProtocolEnvelope,
    assert_lawful_promotion,
)
from .regime_preprocessing import keyed_observations

__all__ = [
    "REGIME_PROTOCOL_STORE",
    "REGIME_FIT_STORE",
    "REGIME_ASSESSMENT_STORE",
    "REGIME_PROMOTION_STORE",
    "PIPELINE_SIDECAR",
    "PARAMETERS_SIDECAR",
    "ASSIGNMENTS_SIDECAR",
    "ARTIFACT_SIDECAR",
    "persist_regime_protocol",
    "persist_regime_fit",
    "persist_regime_assessment",
    "persist_regime_promotion",
    "load_regime_protocol",
    "load_regime_fit",
    "load_regime_fit_assignments",
    "load_regime_assessment",
    "load_regime_promotion",
    "ReloadedRegimeFit",
]

REGIME_PROTOCOL_STORE = "regime_protocols"
REGIME_FIT_STORE = "regime_fits"
REGIME_ASSESSMENT_STORE = "regime_assessments"
REGIME_PROMOTION_STORE = "regime_promotions"

PIPELINE_SIDECAR = "pipeline.joblib"
PARAMETERS_SIDECAR = "parameters.json"
ASSIGNMENTS_SIDECAR = "assignments.arrow"
ARTIFACT_SIDECAR = "artifact.json"

_BINDING_COLUMNS = (
    "resolved_regime_protocol_id",
    "regime_fit_id",
    "row_id",
    "fold_index",
    "partition",
    "fold_local_cluster_id",
    "valid",
)


def persist_regime_protocol(root: Path, envelope: RegimeProtocolEnvelope):
    return save_or_reuse_envelope(Path(root), REGIME_PROTOCOL_STORE, envelope)


def _assignments_bytes(assignments: pd.DataFrame) -> bytes:
    import pyarrow as pa
    import pyarrow.ipc

    table = pa.Table.from_pandas(assignments, preserve_index=False)
    sink = io.BytesIO()
    with pyarrow.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def _assignments_from_bytes(data: bytes) -> pd.DataFrame:
    import pyarrow.ipc

    with pyarrow.ipc.open_file(io.BytesIO(data)) as reader:
        return reader.read_all().to_pandas()


def _transform_with_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    features = list(bundle["features"])
    matrix = frame.loc[:, features].to_numpy(dtype=float)
    imputed = bundle["imputer"].transform(matrix)
    lower = bundle["clip_lower"]
    upper = bundle["clip_upper"]
    if lower is not None and upper is not None:
        n = len(features)
        imputed[:, :n] = np.clip(imputed[:, :n], lower, upper)
    return bundle["scaler"].transform(imputed)


def _software_versions() -> dict[str, str]:
    import sklearn

    return {
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _bound_assignments(fold_fit, fold_assignments: pd.DataFrame) -> pd.DataFrame:
    """Refuse an assignment frame that is not THIS fit's (review F6)."""

    envelope = fold_fit.fit_envelope
    missing = sorted(set(_BINDING_COLUMNS) - set(fold_assignments.columns))
    if missing:
        raise ValueError(f"assignment frame lacks binding columns: {missing}")
    if fold_assignments.empty:
        raise ValueError("assignment frame is empty; a fit always assigns its rows")
    if (fold_assignments["regime_fit_id"].astype(str) != envelope.regime_fit_id).any():
        raise ValueError("assignment frame carries rows of a different regime_fit_id")
    if (fold_assignments["fold_index"].astype(int) != fold_fit.fold_index).any():
        raise ValueError("assignment frame carries rows of a different fold")
    if (
        fold_assignments["resolved_regime_protocol_id"].astype(str)
        != envelope.payload.resolved_regime_protocol_id
    ).any():
        raise ValueError("assignment frame carries rows of a different protocol")
    if fold_assignments["row_id"].astype(str).duplicated().any():
        raise ValueError("assignment frame repeats a row id")
    return fold_assignments.reset_index(drop=True)


def _verify_bundle(
    bundle: dict,
    parameters: dict,
    artifact: RegimeFitArtifact,
    rows: pd.DataFrame,
    expected_transform: np.ndarray,
    expected_labels: np.ndarray,
) -> None:
    if canonical_contract_sha256(parameters) != artifact.fitted_parameter_payload_hash:
        raise ValueError("parameter payload does not hash to fitted_parameter_payload_hash")
    actual = _transform_with_bundle(bundle, rows)
    if not np.allclose(actual, expected_transform, equal_nan=True):
        raise ValueError(
            "the reloaded regime pipeline does not reproduce the fit-time "
            "transform; refusing to treat the persisted fit as verified"
        )
    predicted = np.asarray(bundle["estimator"].predict(actual), dtype=int)
    if not np.array_equal(predicted, expected_labels):
        raise ValueError(
            "the reloaded estimator does not reproduce the persisted fold-local "
            "cluster ids; refusing to treat the persisted fit as verified"
        )


def persist_regime_fit(
    root: Path,
    fold_fit,
    fold_assignments: pd.DataFrame,
    *,
    observation_frame: pd.DataFrame,
) -> RegimeFitArtifact:
    """Bind → verify the exact bytes → publish → re-verify through the store.

    ``observation_frame`` is the run's observation frame; every VALID
    assignment row is re-transformed and re-predicted from the bytes about
    to be published and must reproduce the persisted ``fold_local_cluster_id``
    exactly — a fit that cannot reproduce itself never reaches the store.
    """

    assignments = _bound_assignments(fold_fit, fold_assignments)
    preprocessing = fold_fit.preprocessing
    indexed = keyed_observations(observation_frame)
    valid = assignments[assignments["valid"].astype(bool)]
    if valid.empty:
        raise ValueError("a fit must carry at least one valid assignment row")
    row_ids = [str(row_id) for row_id in valid["row_id"]]
    absent = sorted(set(row_ids) - set(indexed.index))
    if absent:
        raise ValueError(f"{len(absent)} assigned rows are absent from the observation frame")
    rows = indexed.loc[row_ids]
    expected_transform = preprocessing.transform(rows)
    expected_labels = valid["fold_local_cluster_id"].astype(int).to_numpy()
    if not np.array_equal(
        np.asarray(fold_fit.estimator.predict(expected_transform), dtype=int),
        expected_labels,
    ):
        raise ValueError(
            "assignment frame labels do not match the fit's own estimator; "
            "refusing to persist a frame that is not this fit's output"
        )

    bundle = {
        "imputer": preprocessing.imputer,
        "scaler": preprocessing.scaler,
        "clip_lower": preprocessing.clip_lower,
        "clip_upper": preprocessing.clip_upper,
        "features": list(preprocessing.features),
        "estimator": fold_fit.estimator,
    }
    pipeline_blob = io.BytesIO()
    joblib.dump(bundle, pipeline_blob)
    pipeline_bytes = pipeline_blob.getvalue()
    parameters_bytes = (
        json.dumps(preprocessing.parameter_payload, sort_keys=True) + "\n"
    ).encode("utf-8")
    artifact = RegimeFitArtifact(
        regime_fit_id=fold_fit.fit_envelope.regime_fit_id,
        fold_index=fold_fit.fold_index,
        fitted_parameter_payload_hash=preprocessing.fitted_parameter_payload_hash,
        preprocessing_pipeline_ref=PIPELINE_SIDECAR,
        fold_local_cluster_ids=tuple(range(len(fold_fit.centroids_scaled))),
        centroid_or_component_descriptors={
            int(cluster): tuple(
                (name, round(float(value), 6))
                for name, value in zip(
                    preprocessing.output_feature_names,
                    fold_fit.centroids_scaled[cluster],
                    strict=True,
                )
            )
            for cluster in range(len(fold_fit.centroids_scaled))
        },
        training_row_ids_hash=canonical_contract_sha256(
            {"training_row_ids": sorted(preprocessing.training_row_ids)}
        ),
        training_feature_matrix_hash=preprocessing.training_feature_matrix_hash,
        training_row_count=fold_fit.training_row_count,
        inertia_or_loglik=float(fold_fit.inertia),
        software_versions=_software_versions(),
    )
    artifact_bytes = (
        json.dumps(artifact.model_dump(mode="json"), sort_keys=True) + "\n"
    ).encode("utf-8")
    assignments_bytes = _assignments_bytes(assignments)

    # PRE-PUBLISH verification of the exact bytes (review S8)
    _verify_bundle(
        joblib.load(io.BytesIO(pipeline_bytes)),
        json.loads(parameters_bytes.decode("utf-8")),
        RegimeFitArtifact.model_validate(json.loads(artifact_bytes.decode("utf-8"))),
        rows,
        expected_transform,
        expected_labels,
    )
    if len(_assignments_from_bytes(assignments_bytes)) != len(assignments):
        raise ValueError("assignment frame does not round-trip through Arrow")

    _stored, reused = save_or_reuse_envelope(
        Path(root),
        REGIME_FIT_STORE,
        fold_fit.fit_envelope,
        extra_files={
            PIPELINE_SIDECAR: pipeline_bytes,
            PARAMETERS_SIDECAR: parameters_bytes,
            ASSIGNMENTS_SIDECAR: assignments_bytes,
            ARTIFACT_SIDECAR: artifact_bytes,
        },
    )
    # POST-PUBLISH verification through the manifest-checked store; a fresh
    # entry that fails here is withdrawn (it never existed as verified)
    try:
        reloaded = load_regime_fit(Path(root), fold_fit.fit_envelope.regime_fit_id)
        _verify_bundle(
            reloaded._bundle,
            reloaded.parameters,
            reloaded.artifact,
            rows,
            expected_transform,
            expected_labels,
        )
    except Exception:
        if not reused:
            shutil.rmtree(
                envelope_destination(
                    Path(root), REGIME_FIT_STORE, fold_fit.fit_envelope.regime_fit_id
                ),
                ignore_errors=True,
            )
        raise
    return artifact


@dataclass(frozen=True, slots=True)
class ReloadedRegimeFit:
    """A store-verified fit: envelope, artifact, transformer, assignments."""

    envelope: RegimeFitEnvelope
    artifact: RegimeFitArtifact
    parameters: dict
    assignments: pd.DataFrame
    _bundle: dict

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return _transform_with_bundle(self._bundle, frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._bundle["estimator"].predict(self.transform(frame))


def load_regime_fit(root: Path, regime_fit_id: str) -> ReloadedRegimeFit:
    """Manifest-verified reload by MANIFEST-RELATIVE references only.

    Every sidecar loads through the store's hash-checked reader relative to
    the artifact directory — nothing depends on the fitting machine's paths
    (revision P1-4); the parameter payload is re-hashed against the
    artifact's ``fitted_parameter_payload_hash``; the artifact must name
    this very fit; the joblib sidecar is unpickled only after its bytes
    matched the verified manifest.
    """

    envelope = load_verified_envelope(
        Path(root), REGIME_FIT_STORE, regime_fit_id, RegimeFitEnvelope
    )
    artifact = RegimeFitArtifact.model_validate(
        json.loads(
            load_sidecar_bytes(
                Path(root), REGIME_FIT_STORE, regime_fit_id, ARTIFACT_SIDECAR
            ).decode("utf-8")
        )
    )
    if artifact.regime_fit_id != regime_fit_id or (
        artifact.fold_index != envelope.payload.fold_index
    ):
        raise ValueError("artifact sidecar does not describe this regime fit; refusing")
    if artifact.training_feature_matrix_hash != (
        envelope.payload.training_feature_matrix_hash
    ):
        raise ValueError("artifact training matrix hash disagrees with the fit identity")
    parameters = json.loads(
        load_sidecar_bytes(
            Path(root), REGIME_FIT_STORE, regime_fit_id, PARAMETERS_SIDECAR
        ).decode("utf-8")
    )
    if canonical_contract_sha256(parameters) != artifact.fitted_parameter_payload_hash:
        raise ValueError(
            "reloaded parameter payload does not hash to the artifact's "
            "fitted_parameter_payload_hash; refusing"
        )
    bundle = joblib.load(
        io.BytesIO(
            load_sidecar_bytes(
                Path(root),
                REGIME_FIT_STORE,
                regime_fit_id,
                artifact.preprocessing_pipeline_ref,
            )
        )
    )
    assignments = _assignments_from_bytes(
        load_sidecar_bytes(
            Path(root), REGIME_FIT_STORE, regime_fit_id, ASSIGNMENTS_SIDECAR
        )
    )
    if (assignments["regime_fit_id"].astype(str) != regime_fit_id).any():
        raise ValueError("assignment sidecar carries rows of a different regime fit")
    return ReloadedRegimeFit(
        envelope=envelope,
        artifact=artifact,
        parameters=parameters,
        assignments=assignments,
        _bundle=bundle,
    )


def load_regime_fit_assignments(
    root: Path, regime_fit_id: str
) -> tuple[RegimeFitEnvelope, RegimeFitArtifact, pd.DataFrame]:
    """The UI path: envelope + artifact JSON + the Arrow assignment frame,
    all manifest-verified — the joblib sidecar is never unpickled here."""

    envelope = load_verified_envelope(
        Path(root), REGIME_FIT_STORE, regime_fit_id, RegimeFitEnvelope
    )
    artifact = RegimeFitArtifact.model_validate(
        json.loads(
            load_sidecar_bytes(
                Path(root), REGIME_FIT_STORE, regime_fit_id, ARTIFACT_SIDECAR
            ).decode("utf-8")
        )
    )
    if artifact.regime_fit_id != regime_fit_id:
        raise ValueError("artifact sidecar does not describe this regime fit; refusing")
    assignments = _assignments_from_bytes(
        load_sidecar_bytes(
            Path(root), REGIME_FIT_STORE, regime_fit_id, ASSIGNMENTS_SIDECAR
        )
    )
    if (assignments["regime_fit_id"].astype(str) != regime_fit_id).any():
        raise ValueError("assignment sidecar carries rows of a different regime fit")
    return envelope, artifact, assignments


def persist_regime_assessment(
    root: Path, envelope: RegimeCapabilityAssessmentEnvelope
):
    return save_or_reuse_envelope(Path(root), REGIME_ASSESSMENT_STORE, envelope)


def load_regime_assessment(
    root: Path, assessment_id: str
) -> RegimeCapabilityAssessmentEnvelope:
    return load_verified_envelope(
        Path(root),
        REGIME_ASSESSMENT_STORE,
        assessment_id,
        RegimeCapabilityAssessmentEnvelope,
    )


def load_regime_promotion(root: Path, decision_id: str) -> RegimePromotionDecisionEnvelope:
    return load_verified_envelope(
        Path(root), REGIME_PROMOTION_STORE, decision_id, RegimePromotionDecisionEnvelope
    )


def persist_regime_promotion(root: Path, envelope: RegimePromotionDecisionEnvelope):
    """Persist a promotion decision ONLY against its verified evidence.

    The referenced capability assessment must exist in this store, verify,
    and name the decision's protocol; the chained previous decision (when
    referenced) must exist, name the same protocol, and carry exactly the
    declared ``previous_status``; and the ladder is re-checked with the
    assessment's OWN ``gates_passed`` — so FEATURE_ELIGIBLE+ over a failing
    or absent assessment is unpersistable (reviews F5/S1).
    """

    decision = envelope.payload
    try:
        assessment = load_regime_assessment(Path(root), decision.capability_assessment_ref)
    except Exception as error:
        raise ValueError(
            "promotion refused: the referenced capability assessment is not a "
            "verified entry of this store"
        ) from error
    if assessment.payload.resolved_regime_protocol_id != decision.resolved_regime_protocol_id:
        raise ValueError(
            "promotion refused: the capability assessment names a different regime protocol"
        )
    if decision.previous_decision_ref is not None:
        try:
            previous = load_regime_promotion(Path(root), decision.previous_decision_ref)
        except Exception as error:
            raise ValueError(
                "promotion refused: the referenced previous decision is not a "
                "verified entry of this store"
            ) from error
        if previous.payload.resolved_regime_protocol_id != decision.resolved_regime_protocol_id:
            raise ValueError("promotion refused: the previous decision names another protocol")
        if previous.payload.status != decision.previous_status:
            raise ValueError(
                "promotion refused: previous_status does not match the referenced "
                "previous decision's status"
            )
    assert_lawful_promotion(
        decision.previous_status,
        decision.status,
        owner_ratification_ref=decision.owner_ratification_ref,
        gates_passed=assessment.payload.gates_passed,
    )
    return save_or_reuse_envelope(Path(root), REGIME_PROMOTION_STORE, envelope)


def load_regime_protocol(root: Path, protocol_id: str) -> RegimeProtocolEnvelope:
    return load_verified_envelope(
        Path(root), REGIME_PROTOCOL_STORE, protocol_id, RegimeProtocolEnvelope
    )
