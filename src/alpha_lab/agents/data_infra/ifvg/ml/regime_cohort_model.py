"""``cohort_model`` — per-regime specialized models vs the pooled ladder
(R6.1 §6.G "S09c"; owner planning decision 4).

For every valid supervised fold ``k`` and every fit-local regime ``c`` of
fit ``k``: the TRAIN stratum is fold ``k``'s training rows whose fold-local
feature row (partition ``train``, fit ``k``) carries id ``c``; the TEST
stratum is fold ``k``'s test rows whose row (partition ``test``, fit ``k``)
carries ``c``. A stratum with fewer training rows than the stamped floor,
a single training class, or no test rows is typed
``insufficient_regime_partition`` (``below_training_floor`` /
``single_class_training`` / ``no_test_rows``) — never modeled, never
silently dropped. Each modeled stratum runs the SAME bundle-parametrized
ladder as the pooled model through a one-fold ``ContextFoldSet``; its rows
are re-keyed onto the pooled study's D13 identity (schedule, the pooled
fold set's hash, the label artifact) so pooled and specialized predictions
pair exactly on ``comparison_row_id``. The pooled ladder is retained; the
per-regime pooled-vs-specialized paired deltas and
``specialized_coverage_fraction`` are the study's outputs.

The regime id is the STRATIFICATION key, never a model input here: the
model bundle must not carry the regime block. Nothing here promotes: the
study requires the same verified FEATURE_ELIGIBLE frozen authority as
``feature_only`` (``assert_supervised_regime_authority``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import Field

from ..context_experiment_contracts import IfvgContextFoldDefinition
from ..context_feature_view import CandidateFeatureView
from ..context_folds import ContextFoldSet
from ..features.bundle_feature_view import build_bundle_feature_view
from ..features.feature_bundles import FEATURE_BUNDLE_REGISTRY, resolve_bundle
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from ..search.store import (
    load_sidecar_bytes,
    load_verified_envelope,
    save_or_reuse_envelope,
)
from .comparison_rows import (
    COMPARISON_ROW_IDENTITY_KEY,
    label_artifact_content_id,
    with_comparison_row_ids,
)
from .fold_set_artifact import FoldSetArtifactEnvelope
from .model_protocols import LOGISTIC_PROTOCOL_ID, PREVALENCE_PROTOCOL_ID
from .regime_block_activation import RegimeBlockActivation
from .regime_contracts import REGIME_PROPOSED_DEFAULTS
from .regime_controlled_study import (
    FrozenJson,
    assert_supervised_regime_authority,
    jsonable,
    with_losses,
)
from .regime_fold_features import RegimeFoldFeatureFrameSource
from .supervised_ladder import (
    DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    SupervisedLadderRun,
    paired_cell_delta_report,
    run_supervised_ladder,
)

__all__ = [
    "REGIME_COHORT_MODEL_STORE",
    "MINIMUM_TRAINING_ROWS_PER_REGIME_STRATUM",
    "COHORT_STRATUM_REASONS",
    "RegimeCohortStratumRecord",
    "RegimeCohortModelStudyPayload",
    "RegimeCohortModelStudyEnvelope",
    "RegimeCohortModelStudyRun",
    "run_regime_cohort_model_study",
    "save_regime_cohort_model_study",
    "load_regime_cohort_model_study",
    "load_regime_cohort_model_study_detail",
]

REGIME_COHORT_MODEL_STORE = "regime_cohort_model_studies"
_DETAIL_SIDECAR = "regime_cohort_model_study_detail.json"
#: Plan §8 stamp (``minimum_training_rows_per_regime_stratum``,
#: proposed_protocol_default) — single-sourced from ``REGIME_PROPOSED_DEFAULTS``
#: once the hub registers the entry; the value is the plan's 60.
MINIMUM_TRAINING_ROWS_PER_REGIME_STRATUM = int(
    REGIME_PROPOSED_DEFAULTS["minimum_training_rows_per_regime_stratum"]["value"]
)
COHORT_STRATUM_REASONS: tuple[str, ...] = (
    "below_training_floor",
    "single_class_training",
    "no_test_rows",
)
_ONE_FOLD_STATUS = "regime_cohort_one_fold_v1"


class RegimeCohortStratumRecord(FrozenContract):
    fold_index: int = Field(ge=0)
    local_id: str
    regime_fit_id: str = Field(pattern=SHA256_PATTERN)
    train_rows: int = Field(ge=0)
    test_rows: int = Field(ge=0)
    status: Literal["modeled", "insufficient_regime_partition"]
    reason: str | None
    specialized_ladder_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    #: pooled-vs-specialized paired deltas per modeled rung (Brier / log-loss);
    #: deep-immutable, serialized as plain JSON objects (identity-stable)
    paired_deltas: ImmutableMap[str, FrozenJson] | None
    pooled_brier_on_stratum: float | None
    specialized_brier: float | None


class RegimeCohortModelStudyPayload(FrozenContract):
    comparison_class: Literal["cohort_model"] = "cohort_model"
    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    regime_fold_feature_artifact_id: str = Field(pattern=SHA256_PATTERN)
    regime_promotion_decision_id: str = Field(pattern=SHA256_PATTERN)
    owner_decision_artifact_id: str = Field(pattern=SHA256_PATTERN)
    capability_assessment_id: str = Field(pattern=SHA256_PATTERN)
    bundle_key: str
    resolved_feature_bundle_id: str = Field(pattern=SHA256_PATTERN)
    view_id: str = Field(pattern=SHA256_PATTERN)
    label_artifact_id: str = Field(pattern=SHA256_PATTERN)
    label_content_hash: str = Field(pattern=SHA256_PATTERN)
    fold_schedule_id: str = Field(pattern=SHA256_PATTERN)
    candidate_fold_set_id: str = Field(pattern=SHA256_PATTERN)
    fold_set_hash: str = Field(pattern=SHA256_PATTERN)
    model_protocol_ids: tuple[str, ...] = Field(min_length=2)
    calibration_policy_id: str
    minimum_training_rows: int = Field(ge=1)
    pooled_ladder_id: str = Field(pattern=SHA256_PATTERN)
    strata: tuple[RegimeCohortStratumRecord, ...]
    pooled_oos_row_count: int = Field(ge=0)
    specialized_oos_row_count: int = Field(ge=0)
    specialized_coverage_fraction: float = Field(ge=0.0, le=1.0)
    research_boundary: Literal["research_only_offline"] = "research_only_offline"


class RegimeCohortModelStudyEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "regime_cohort_model_study_id"

    regime_cohort_model_study_id: str = Field(pattern=SHA256_PATTERN)
    payload: RegimeCohortModelStudyPayload
    detail_sha256: str = Field(pattern=SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class RegimeCohortModelStudyRun:
    envelope: RegimeCohortModelStudyEnvelope
    pooled: SupervisedLadderRun
    specialized: dict[tuple[int, str], SupervisedLadderRun]
    #: every specialized OOS prediction row (re-keyed onto the pooled D13 identity)
    specialized_predictions: pd.DataFrame
    detail_bytes: bytes
    #: review RB-01 (§7.2): ``"exact"`` ONLY when the label artifact id was proven to
    #: derive from the registered policy and these labels — the one persistable state
    label_identity_proof: str = "caller_supplied_unproven"


def _one_fold_set(
    fold: IfvgContextFoldDefinition,
    *,
    train_ids: tuple[str, ...],
    test_ids: tuple[str, ...],
    training_prevalence: float,
) -> ContextFoldSet:
    definition = IfvgContextFoldDefinition(
        fold_index=fold.fold_index,
        train_days=fold.train_days,
        test_days=fold.test_days,
        train_candidate_ids=train_ids,
        test_candidate_ids=test_ids,
        excluded_boundary_setup_ids=fold.excluded_boundary_setup_ids,
        purged_candidate_ids=fold.purged_candidate_ids,
        embargoed_candidate_ids=fold.embargoed_candidate_ids,
        valid=True,
        invalid_reason=None,
        training_prevalence=float(training_prevalence),
    )
    return ContextFoldSet(folds=(definition,), assignment=pd.DataFrame(), status=_ONE_FOLD_STATUS)


def _rekey(
    predictions: pd.DataFrame,
    *,
    fold_schedule_id: str,
    pooled_fold_set_hash: str,
    label_artifact_id: str,
) -> pd.DataFrame:
    """The one-fold ladder keys its rows on the one-fold population; the
    study pairs on the POOLED identity, so the rows are re-keyed."""

    frame = predictions.drop(columns=[COMPARISON_ROW_IDENTITY_KEY], errors="ignore")
    return with_comparison_row_ids(
        frame,
        fold_schedule_id=fold_schedule_id,
        candidate_fold_set_id=pooled_fold_set_hash,
        label_artifact_id=label_artifact_id,
    )


def _mean_brier(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    return float(with_losses(frame)["brier_loss"].mean())


def run_regime_cohort_model_study(
    view: CandidateFeatureView,
    labeled_candidates: pd.DataFrame,
    folds: ContextFoldSet,
    *,
    activation: RegimeBlockActivation,
    bundle_key: str,
    fold_features: RegimeFoldFeatureFrameSource,
    candidate_fold_set: FoldSetArtifactEnvelope,
    label_artifact_id: str,
    bundle_registry=None,
    protocols: tuple[str, ...] = DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    minimum_training_rows: int = MINIMUM_TRAINING_ROWS_PER_REGIME_STRATUM,
    calibration_policy_id: str = "raw_probability_diagnostics_v1",
    label_policy_id: str | None = None,
) -> RegimeCohortModelStudyRun:
    """Module docstring."""

    if label_policy_id is not None:
        # HARDENING-BACKEND-FIX §7.2: the persisting seam proves the exact label identity
        from .comparison_rows import assert_exact_label_artifact  # noqa: PLC0415

        label_artifact_id = assert_exact_label_artifact(
            label_artifact_id, label_policy_id, labeled_candidates
        )
        label_identity_proof = "exact"
    else:
        # review RB-01: trusted in memory only — never persistable
        label_identity_proof = "caller_supplied_unproven"
    assert_supervised_regime_authority(
        activation, fold_features, candidate_fold_set=candidate_fold_set, folds=folds
    )
    if int(minimum_training_rows) < 1:
        raise ValueError("minimum_training_rows must be positive")
    registry = bundle_registry if bundle_registry is not None else FEATURE_BUNDLE_REGISTRY
    bundle_env = resolve_bundle(
        bundle_key,
        bundle_registry=registry,
        definitions=activation.definitions,
        resolutions=activation.resolutions,
    )
    if activation.resolved_feature_block_id in bundle_env.payload.resolved_block_ids:
        raise ValueError(
            "cohort_model stratifies on the fit-local regime id; the model bundle "
            f"{bundle_key!r} must not carry the regime block (use feature_only for that)"
        )
    names = tuple(bundle_env.payload.resolved_feature_names)
    _view_env, frame = build_bundle_feature_view(view, bundle_env)
    arm_view = replace(view, frame=frame)
    schedule_id = fold_features.envelope.payload.fold_schedule_id
    pooled_hash = candidate_fold_set.payload.fold_set_id
    ladder_kwargs: dict[str, Any] = dict(
        bundle_features=names,
        bundle_ref=bundle_env.resolved_feature_bundle_id,
        protocols=protocols,
        calibration_policy_id=calibration_policy_id,
        fold_schedule_id=schedule_id,
        label_artifact_id=label_artifact_id,
    )
    pooled = run_supervised_ladder(arm_view, labeled_candidates, folds, **ladder_kwargs)
    pooled_logistic = pooled.rung(LOGISTIC_PROTOCOL_ID).predictions

    labels = labeled_candidates.set_index("candidate_id", verify_integrity=True)
    fit_by_fold = {
        ref.fold_index: ref.regime_fit_id
        for ref in fold_features.envelope.payload.regime_fit_ids_by_fold
    }
    records: list[RegimeCohortStratumRecord] = []
    specialized: dict[tuple[int, str], SupervisedLadderRun] = {}
    specialized_frames: list[pd.DataFrame] = []
    detail_strata: dict[str, Any] = {}
    for fold in folds.folds:
        if not fold.valid:
            continue
        fit_id = fit_by_fold.get(int(fold.fold_index))
        if fit_id is None:
            continue  # no valid regime fit for this fold: no strata exist
        ids = fold_features.local_ids_for_fold(fold.fold_index)
        train_pool = set(str(cid) for cid in fold.train_candidate_ids)
        test_pool = set(str(cid) for cid in fold.test_candidate_ids)
        for local_id in sorted(ids["local_id"].unique()):
            stratum = ids[ids["local_id"] == local_id]
            train_ids = tuple(
                sorted(
                    set(stratum.loc[stratum["partition"] == "train", "candidate_id"]) & train_pool
                )
            )
            test_ids = tuple(
                sorted(set(stratum.loc[stratum["partition"] == "test", "candidate_id"]) & test_pool)
            )
            key = (int(fold.fold_index), str(local_id))
            train_targets = pd.to_numeric(
                labels.reindex(list(train_ids))["binary_target"], errors="coerce"
            ).dropna()
            reason: str | None = None
            if len(train_targets) < int(minimum_training_rows):
                reason = "below_training_floor"
            elif train_targets.nunique() < 2:
                reason = "single_class_training"
            elif not test_ids:
                reason = "no_test_rows"
            if reason is not None:
                records.append(
                    RegimeCohortStratumRecord(
                        fold_index=fold.fold_index,
                        local_id=str(local_id),
                        regime_fit_id=fit_id,
                        train_rows=int(len(train_targets)),
                        test_rows=int(len(test_ids)),
                        status="insufficient_regime_partition",
                        reason=reason,
                        specialized_ladder_id=None,
                        paired_deltas=None,
                        pooled_brier_on_stratum=None,
                        specialized_brier=None,
                    )
                )
                continue
            one_fold = _one_fold_set(
                fold,
                train_ids=tuple(train_targets.index.astype(str)),
                test_ids=test_ids,
                training_prevalence=float(train_targets.mean()),
            )
            run = run_supervised_ladder(arm_view, labeled_candidates, one_fold, **ladder_kwargs)
            specialized[key] = run
            deltas: dict[str, Any] = {}
            stratum_pooled_brier: float | None = None
            stratum_specialized_brier: float | None = None
            for rung in run.rungs:
                if rung.protocol_id == PREVALENCE_PROTOCOL_ID or rung.predictions.empty:
                    continue
                own = _rekey(
                    rung.predictions,
                    fold_schedule_id=schedule_id,
                    pooled_fold_set_hash=pooled_hash,
                    label_artifact_id=label_artifact_id,
                )
                pooled_rung = pooled.rung(rung.protocol_id).predictions
                pooled_subset = pooled_rung[
                    pooled_rung[COMPARISON_ROW_IDENTITY_KEY].isin(
                        set(own[COMPARISON_ROW_IDENTITY_KEY])
                    )
                ]
                if set(pooled_subset[COMPARISON_ROW_IDENTITY_KEY]) != set(
                    own[COMPARISON_ROW_IDENTITY_KEY]
                ):
                    raise ValueError(
                        "the specialized stratum's OOS rows are not a subset of the pooled "
                        f"ladder's rows for {rung.protocol_id} (fold {fold.fold_index})"
                    )
                deltas[rung.protocol_id] = {
                    "brier": jsonable(
                        paired_cell_delta_report(
                            with_losses(pooled_subset),
                            with_losses(own),
                            value_column="brier_loss",
                            key_column=COMPARISON_ROW_IDENTITY_KEY,
                        )
                    ),
                    "log_loss": jsonable(
                        paired_cell_delta_report(
                            with_losses(pooled_subset),
                            with_losses(own),
                            value_column="log_loss",
                            key_column=COMPARISON_ROW_IDENTITY_KEY,
                        )
                    ),
                }
                if rung.protocol_id == LOGISTIC_PROTOCOL_ID:
                    stratum_pooled_brier = _mean_brier(pooled_subset)
                    stratum_specialized_brier = _mean_brier(own)
                    tagged = own.assign(
                        regime_local_id=str(local_id), model_protocol_id=rung.protocol_id
                    )
                    specialized_frames.append(tagged)
                else:
                    specialized_frames.append(
                        own.assign(
                            regime_local_id=str(local_id), model_protocol_id=rung.protocol_id
                        )
                    )
            records.append(
                RegimeCohortStratumRecord(
                    fold_index=fold.fold_index,
                    local_id=str(local_id),
                    regime_fit_id=fit_id,
                    train_rows=int(len(train_targets)),
                    test_rows=int(len(test_ids)),
                    status="modeled",
                    reason=None,
                    specialized_ladder_id=run.ladder_id,
                    paired_deltas=deltas,
                    pooled_brier_on_stratum=stratum_pooled_brier,
                    specialized_brier=stratum_specialized_brier,
                )
            )
            detail_strata[f"{fold.fold_index}:{local_id}"] = {
                "parity": jsonable(run.parity),
                "rungs": {
                    rung.protocol_id: jsonable(rung.prediction_report) for rung in run.rungs
                },
            }

    specialized_predictions = (
        pd.concat(specialized_frames, ignore_index=True)
        if specialized_frames
        else pd.DataFrame(
            columns=[*pooled_logistic.columns, "regime_local_id", "model_protocol_id"]
        )
    )
    pooled_oos = int(pooled.parity.get("oos_row_count", 0))
    logistic_specialized = specialized_predictions[
        specialized_predictions.get("model_protocol_id", pd.Series(dtype=object))
        == LOGISTIC_PROTOCOL_ID
    ] if not specialized_predictions.empty else specialized_predictions
    specialized_oos = int(logistic_specialized[COMPARISON_ROW_IDENTITY_KEY].nunique()) if (
        not logistic_specialized.empty
    ) else 0
    coverage = float(specialized_oos / pooled_oos) if pooled_oos else 0.0

    payload = RegimeCohortModelStudyPayload(
        resolved_regime_protocol_id=activation.protocol.resolved_regime_protocol_id,
        regime_fold_feature_artifact_id=fold_features.artifact_id,
        regime_promotion_decision_id=activation.promotion_decision.regime_promotion_decision_id,
        owner_decision_artifact_id=activation.owner_decision.owner_decision_artifact_id,
        capability_assessment_id=activation.assessment.regime_capability_assessment_id,
        bundle_key=bundle_key,
        resolved_feature_bundle_id=bundle_env.resolved_feature_bundle_id,
        view_id=view.view_id,
        label_artifact_id=label_artifact_id,
        label_content_hash=label_artifact_content_id(None, labeled_candidates),
        fold_schedule_id=schedule_id,
        candidate_fold_set_id=candidate_fold_set.fold_set_artifact_id,
        fold_set_hash=pooled_hash,
        model_protocol_ids=tuple(protocols),
        calibration_policy_id=calibration_policy_id,
        minimum_training_rows=int(minimum_training_rows),
        pooled_ladder_id=pooled.ladder_id,
        strata=tuple(records),
        pooled_oos_row_count=pooled_oos,
        specialized_oos_row_count=specialized_oos,
        specialized_coverage_fraction=min(1.0, coverage),
    )
    detail = {
        "pooled": {
            "parity": jsonable(pooled.parity),
            "paired_deltas": jsonable(pooled.paired_deltas),
            "rungs": {rung.protocol_id: jsonable(rung.prediction_report) for rung in pooled.rungs},
        },
        "strata": detail_strata,
    }
    detail_bytes = (json.dumps(detail, sort_keys=True) + "\n").encode("utf-8")
    envelope = RegimeCohortModelStudyEnvelope.from_payload(
        payload, detail_sha256=hashlib.sha256(detail_bytes).hexdigest()
    )
    return RegimeCohortModelStudyRun(
        envelope=envelope,
        pooled=pooled,
        specialized=specialized,
        specialized_predictions=specialized_predictions,
        detail_bytes=detail_bytes,
        label_identity_proof=label_identity_proof,
    )


# ── persistence ──────────────────────────────────────────────────────────────


def save_regime_cohort_model_study(root: Path, run: RegimeCohortModelStudyRun) -> tuple:
    # review RB-01 (§7.2): only a run whose label identity was PROVEN exact persists
    from .comparison_rows import assert_persistable_label_proof  # noqa: PLC0415

    assert_persistable_label_proof(run.label_identity_proof, runner="run_regime_cohort_model_study")
    return save_or_reuse_envelope(
        Path(root),
        REGIME_COHORT_MODEL_STORE,
        run.envelope,
        extra_files={_DETAIL_SIDECAR: run.detail_bytes},
    )


def load_regime_cohort_model_study(root: Path, study_id: str) -> RegimeCohortModelStudyEnvelope:
    return load_verified_envelope(
        Path(root), REGIME_COHORT_MODEL_STORE, study_id, RegimeCohortModelStudyEnvelope
    )


def load_regime_cohort_model_study_detail(
    root: Path, envelope: RegimeCohortModelStudyEnvelope
) -> dict[str, Any]:
    data = load_sidecar_bytes(
        Path(root),
        REGIME_COHORT_MODEL_STORE,
        envelope.regime_cohort_model_study_id,
        _DETAIL_SIDECAR,
    )
    if hashlib.sha256(data).hexdigest() != envelope.detail_sha256:
        raise ValueError("regime cohort-model study detail sidecar fails the envelope hash check")
    return json.loads(data.decode("utf-8"))


def _example_payload() -> RegimeCohortModelStudyPayload:
    return RegimeCohortModelStudyPayload(
        resolved_regime_protocol_id="a" * 64,
        regime_fold_feature_artifact_id="b" * 64,
        regime_promotion_decision_id="c" * 64,
        owner_decision_artifact_id="d" * 64,
        capability_assessment_id="e" * 64,
        bundle_key="B0_CORE",
        resolved_feature_bundle_id="f" * 64,
        view_id="1" * 64,
        label_artifact_id="2" * 64,
        label_content_hash="3" * 64,
        fold_schedule_id="4" * 64,
        candidate_fold_set_id="5" * 64,
        fold_set_hash="6" * 64,
        model_protocol_ids=tuple(DEFAULT_BUNDLE_LADDER_PROTOCOLS),
        calibration_policy_id="raw_probability_diagnostics_v1",
        minimum_training_rows=MINIMUM_TRAINING_ROWS_PER_REGIME_STRATUM,
        pooled_ladder_id="7" * 64,
        strata=(
            RegimeCohortStratumRecord(
                fold_index=0,
                local_id="0",
                regime_fit_id="8" * 64,
                train_rows=10,
                test_rows=0,
                status="insufficient_regime_partition",
                reason="below_training_floor",
                specialized_ladder_id=None,
                paired_deltas=None,
                pooled_brier_on_stratum=None,
                specialized_brier=None,
            ),
        ),
        pooled_oos_row_count=0,
        specialized_oos_row_count=0,
        specialized_coverage_fraction=0.0,
    )


register_identity_pair(
    name="RegimeCohortModelStudy",
    envelope_cls=RegimeCohortModelStudyEnvelope,
    payload_cls=RegimeCohortModelStudyPayload,
    id_field="regime_cohort_model_study_id",
    example_factory=_example_payload,
    extra_envelope_fields=("detail_sha256",),
)
