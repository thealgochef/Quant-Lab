"""``feature_only`` — the controlled Baseline vs Baseline+Regime study
(R6.1 §6.G "S09c"; D7 / D8 / D9 / D13).

Two arms over IDENTICAL profile/candidate rows, labels, folds, purge,
embargo, costs, seed, and model protocols: the challenger bundle carries the
STATUS-GATED regime block (``IFVG_REGIME_CONTEXT_V1`` activated by the
exact frozen FEATURE_ELIGIBLE authority — ``regime_block_activation``); its
baseline is the challenger's OWN base bundle (composition by extension —
the bundle graph is the preregistered block ablation). The regime columns
are FOLD-LOCAL (D7): they never live in the static candidate view; the
challenger ladder joins them per fold through the fold-local feature seam
(``comparison_rows.RegimeFoldFeatureSource``), and the ladder's identity
binds the fold-feature artifact id as the challenger's evidence ref.

Both arms run the bundle-parametrized supervised ladder (prevalence +
logistic + the bundle-aware CatBoost rung) and are gated by the identical
rows assertion on the bundle-independent ``comparison_row_id`` (D13) before
any paired delta (Brier, log-loss) is computed. Nothing here promotes: the
study REQUIRES a verified FEATURE_ELIGIBLE (or later) frozen decision and
refuses below it with the exact text ``FEATURE_ELIGIBLE_REFUSAL``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field, PlainSerializer, PlainValidator

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
from .comparison_rows import COMPARISON_ROW_IDENTITY_KEY, label_content_hash
from .fold_set_artifact import FoldSetArtifactEnvelope
from .fold_set_artifact import fold_set_id as _legacy_fold_set_id
from .model_protocols import PREVALENCE_PROTOCOL_ID
from .regime_block_activation import FEATURE_ELIGIBLE_STATUSES, RegimeBlockActivation
from .regime_contracts import RegimeStatus
from .regime_fold_features import RegimeFoldFeatureFrameSource
from .supervised_ladder import (
    DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    SupervisedLadderRun,
    paired_cell_delta_report,
    run_supervised_ladder,
)

__all__ = [
    "REGIME_CONTROLLED_STUDY_STORE",
    "FEATURE_ELIGIBLE_REFUSAL",
    "RegimeSupervisedStudyRefusalError",
    "RegimeControlledStudyPayload",
    "RegimeControlledStudyEnvelope",
    "RegimeControlledStudyRun",
    "jsonable",
    "FrozenTree",
    "FrozenJson",
    "freeze_json",
    "thaw_json",
    "assert_regime_cross_arm_identity",
    "with_losses",
    "assert_supervised_regime_authority",
    "run_controlled_regime_study",
    "save_regime_controlled_study",
    "load_regime_controlled_study",
    "load_regime_controlled_study_detail",
]

REGIME_CONTROLLED_STUDY_STORE = "regime_controlled_studies"
_DETAIL_SIDECAR = "regime_controlled_study_detail.json"

#: The exact below-status refusal (owner planning decision 4).
FEATURE_ELIGIBLE_REFUSAL = (
    "feature_only and cohort_model require a verified FEATURE_ELIGIBLE (or later) "
    "frozen promotion decision bound to this exact regime protocol and fold-feature "
    "artifact; nothing here promotes"
)


class RegimeSupervisedStudyRefusalError(PermissionError):
    """A supervised regime study was requested without its frozen authority."""


def jsonable(value: Any) -> Any:
    """NaN/Inf-safe JSON projection (typed nulls, never invalid JSON)."""

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and (pd.isna(value) or math.isinf(value)):
        return None
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


class FrozenTree(Mapping):
    """A read-only JSON object tree (R6.1 contract review F15): nested
    mappings are ``FrozenTree``, sequences are tuples, scalars pass through —
    nothing below an identity payload can be mutated after the identity is
    calculated. It SERIALIZES as the plain JSON object it froze, so the
    canonical payload bytes (and every study id) are exactly what the plain
    nested dicts produced."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping) -> None:
        items = sorted(
            ((key, freeze_json(value)) for key, value in dict(data).items()),
            key=lambda kv: json.dumps(kv[0], sort_keys=True, default=str),
        )
        object.__setattr__(self, "_data", dict(items))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenTree({thaw_json(self)!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Mapping):
            return thaw_json(self) == thaw_json(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self._data.items()))

    def __deepcopy__(self, memo):  # immutable: the copy IS the object
        return self

    def __copy__(self):
        return self


def freeze_json(value: Any) -> Any:
    """Deep-freeze a JSON-like value: mappings → :class:`FrozenTree`, lists /
    tuples → tuples, JSON scalars pass through; anything else refuses."""

    if isinstance(value, FrozenTree):
        return value
    if isinstance(value, Mapping):
        return FrozenTree(value)
    if isinstance(value, list | tuple):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, set | frozenset):
        raise ValueError("sets have no canonical order; use a sorted tuple")
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    raise TypeError(f"{type(value).__name__} is not a JSON value")


def thaw_json(value: Any) -> Any:
    """The plain JSON form (dicts / lists / scalars) of a frozen tree."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [thaw_json(item) for item in value]
    return value


#: The deep-immutable value type of the study summaries / paired deltas:
#: validated into ``FrozenTree`` / tuples, serialized as plain JSON objects
#: (identity-stable with the plain-dict form).
FrozenJson = Annotated[
    Any,
    PlainValidator(freeze_json),
    PlainSerializer(thaw_json, when_used="always"),
]


def with_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-row Brier loss and log-loss (probabilities clipped at 1e-15)."""

    out = predictions.copy()
    target = pd.to_numeric(out["target"], errors="raise").astype(float)
    probability = pd.to_numeric(out["probability"], errors="raise").astype(float)
    clipped = probability.clip(1e-15, 1 - 1e-15)
    out["brier_loss"] = (target - probability) ** 2
    out["log_loss"] = -(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped))
    return out


# ── contracts ────────────────────────────────────────────────────────────────


class RegimeControlledStudyPayload(FrozenContract):
    comparison_class: Literal["feature_only"] = "feature_only"
    resolved_regime_protocol_id: str = Field(pattern=SHA256_PATTERN)
    regime_fold_feature_artifact_id: str = Field(pattern=SHA256_PATTERN)
    regime_promotion_decision_id: str = Field(pattern=SHA256_PATTERN)
    owner_decision_artifact_id: str = Field(pattern=SHA256_PATTERN)
    capability_assessment_id: str = Field(pattern=SHA256_PATTERN)
    resolved_regime_block_id: str = Field(pattern=SHA256_PATTERN)
    block_registry_hash: str = Field(pattern=SHA256_PATTERN)
    baseline_bundle_key: str
    challenger_bundle_key: str
    baseline_resolved_bundle_id: str = Field(pattern=SHA256_PATTERN)
    challenger_resolved_bundle_id: str = Field(pattern=SHA256_PATTERN)
    view_id: str = Field(pattern=SHA256_PATTERN)
    label_artifact_id: str = Field(pattern=SHA256_PATTERN)
    label_content_hash: str = Field(pattern=SHA256_PATTERN)
    fold_schedule_id: str = Field(pattern=SHA256_PATTERN)
    candidate_fold_set_id: str = Field(pattern=SHA256_PATTERN)
    fold_set_hash: str = Field(pattern=SHA256_PATTERN)
    model_protocol_ids: tuple[str, ...] = Field(min_length=2)
    calibration_policy_id: str
    baseline_ladder_id: str = Field(pattern=SHA256_PATTERN)
    challenger_ladder_id: str = Field(pattern=SHA256_PATTERN)
    oos_row_count: int = Field(ge=0)
    parity_status: Literal["held", "not_evaluable"]
    #: deep-immutable (``FrozenTree`` / tuples below the map); serialized as
    #: the plain JSON objects they froze — the study id is unchanged
    baseline_summary: ImmutableMap[str, FrozenJson]
    challenger_summary: ImmutableMap[str, FrozenJson]
    paired_deltas: ImmutableMap[str, FrozenJson]
    research_boundary: Literal["research_only_offline"] = "research_only_offline"


class RegimeControlledStudyEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "regime_controlled_study_id"

    regime_controlled_study_id: str = Field(pattern=SHA256_PATTERN)
    payload: RegimeControlledStudyPayload
    #: post-materialization fact binding the full diagnostics sidecar
    detail_sha256: str = Field(pattern=SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class RegimeControlledStudyRun:
    envelope: RegimeControlledStudyEnvelope
    baseline: SupervisedLadderRun
    challenger: SupervisedLadderRun
    detail_bytes: bytes
    #: review RB-01 (§7.2): ``"exact"`` ONLY when the label artifact id was proven to
    #: derive from the registered policy and these labels — the one persistable state
    label_identity_proof: str = "caller_supplied_unproven"


# ── the frozen authority gate (shared with cohort_model) ────────────────────


def assert_supervised_regime_authority(
    activation: RegimeBlockActivation,
    fold_features: RegimeFoldFeatureFrameSource,
    *,
    candidate_fold_set: FoldSetArtifactEnvelope,
    folds: ContextFoldSet,
) -> None:
    """A model-bearing study runs only under the exact frozen FEATURE_ELIGIBLE
    authority its activation verified, over the fold-feature artifact that
    authority binds, on the candidate fold set that artifact was built for."""

    decision = activation.promotion_decision
    status = RegimeStatus(decision.payload.status)
    if status not in FEATURE_ELIGIBLE_STATUSES:
        raise RegimeSupervisedStudyRefusalError(
            f"{FEATURE_ELIGIBLE_REFUSAL} (frozen decision "
            f"{decision.regime_promotion_decision_id[:12]}… carries {status.value})"
        )
    bound = activation.fold_feature_artifact.regime_fold_feature_artifact_id
    if fold_features.artifact_id != bound:
        raise RegimeSupervisedStudyRefusalError(
            f"{FEATURE_ELIGIBLE_REFUSAL} (the activation binds fold-feature artifact "
            f"{bound[:12]}…, the study supplied {fold_features.artifact_id[:12]}…)"
        )
    artifact = fold_features.envelope.payload
    if artifact.resolved_regime_protocol_id != activation.protocol.resolved_regime_protocol_id:
        raise RegimeSupervisedStudyRefusalError(
            f"{FEATURE_ELIGIBLE_REFUSAL} (fold-feature artifact of another protocol)"
        )
    if artifact.candidate_fold_set_id != candidate_fold_set.fold_set_artifact_id:
        raise ValueError(
            "fold_schedule mismatch: the fold-feature artifact was built for candidate fold "
            f"set {artifact.candidate_fold_set_id[:12]}…, the study runs on "
            f"{candidate_fold_set.fold_set_artifact_id[:12]}… (one schedule, one fold set)"
        )
    if artifact.fold_schedule_id != candidate_fold_set.payload.fold_schedule_id:
        raise ValueError(
            "fold_schedule mismatch between the fold-feature artifact and the candidate fold set"
        )
    if _legacy_fold_set_id(folds) != candidate_fold_set.payload.fold_set_id:
        raise ValueError(
            "fold_schedule mismatch: the supplied folds are not the candidate fold-set "
            "artifact's population"
        )
    if artifact.candidate_fold_set_hash != candidate_fold_set.payload.fold_set_id:
        raise ValueError(
            "fold_schedule mismatch: the fold-feature artifact's fold-set hash differs from the "
            "candidate fold set"
        )


# ── the study ────────────────────────────────────────────────────────────────


def _summary(run: SupervisedLadderRun) -> dict[str, Any]:
    out: dict[str, Any] = {"oos_row_count": run.parity.get("oos_row_count", 0), "rungs": {}}
    for rung in run.rungs:
        report = dict(rung.prediction_report or {})
        out["rungs"][rung.protocol_id] = {
            "status": report.get("status"),
            "brier_score": report.get("brier_score"),
            "reference_brier_score": report.get("reference_brier_score"),
            "brier_skill_score": report.get("brier_skill_score"),
            "log_loss": report.get("log_loss"),
            "auc": report.get("auc"),
            "resolved_protocol_hash": rung.resolved_protocol_hash,
        }
    return jsonable(out)


def _row_key(row) -> tuple[str, str, int, float]:
    return (
        str(row.comparison_row_id),
        str(row.candidate_id),
        int(row.target),
        float(row.training_prevalence),
    )


def assert_regime_cross_arm_identity(
    baseline: SupervisedLadderRun, challenger: SupervisedLadderRun
) -> None:
    """Identical ``comparison_row_id`` populations and (candidate, target,
    prevalence) tuples per rung across the arms (D13) — and a numerically
    identical prevalence reference. The arms may carry different
    ``view_id``s / ``oos_row_id``s: only ``comparison_row_id`` pairs them."""

    key = COMPARISON_ROW_IDENTITY_KEY
    baseline_ids = {rung.protocol_id for rung in baseline.rungs}
    challenger_ids = {rung.protocol_id for rung in challenger.rungs}
    if baseline_ids != challenger_ids:
        raise ValueError("controlled-study arms ran different rung sets")
    for protocol_id in sorted(baseline_ids):
        left = baseline.rung(protocol_id).predictions
        right = challenger.rung(protocol_id).predictions
        if left.empty and right.empty:
            continue
        if left.empty != right.empty or key not in left or key not in right:
            raise ValueError(
                f"controlled-study arms disagree on the exact OOS rows for {protocol_id}"
            )
        left_keys = {_row_key(r) for r in left.itertuples()}
        right_keys = {_row_key(r) for r in right.itertuples()}
        if left_keys != right_keys:
            raise ValueError(
                f"controlled-study arms disagree on the exact OOS rows for {protocol_id}; "
                "the paired delta is undefined (identical rows/labels/folds are the premise)"
            )
    left = baseline.rung(PREVALENCE_PROTOCOL_ID).predictions
    right = challenger.rung(PREVALENCE_PROTOCOL_ID).predictions
    if not left.empty:
        merged = left[[key, "probability"]].merge(
            right[[key, "probability"]],
            on=key,
            validate="one_to_one",
            suffixes=("_baseline", "_challenger"),
        )
        if not (merged["probability_baseline"] == merged["probability_challenger"]).all():
            raise ValueError(
                "the prevalence reference differs across arms — the arms did not share "
                "fold-local training rows"
            )


def _paired_deltas(
    baseline: SupervisedLadderRun, challenger: SupervisedLadderRun
) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for rung in baseline.rungs:
        if rung.protocol_id == PREVALENCE_PROTOCOL_ID:
            continue
        left = rung.predictions
        right = challenger.rung(rung.protocol_id).predictions
        if left.empty or right.empty:
            deltas[rung.protocol_id] = None
            continue
        deltas[rung.protocol_id] = {
            "brier": jsonable(
                paired_cell_delta_report(
                    with_losses(left),
                    with_losses(right),
                    value_column="brier_loss",
                    key_column=COMPARISON_ROW_IDENTITY_KEY,
                )
            ),
            "log_loss": jsonable(
                paired_cell_delta_report(
                    with_losses(left),
                    with_losses(right),
                    value_column="log_loss",
                    key_column=COMPARISON_ROW_IDENTITY_KEY,
                )
            ),
        }
    return deltas


def run_controlled_regime_study(
    view: CandidateFeatureView,
    labeled_candidates: pd.DataFrame,
    folds: ContextFoldSet,
    *,
    activation: RegimeBlockActivation,
    challenger_bundle_key: str,
    fold_features: RegimeFoldFeatureFrameSource,
    candidate_fold_set: FoldSetArtifactEnvelope,
    label_artifact_id: str,
    bundle_registry=None,
    protocols: tuple[str, ...] = DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    calibration_policy_id: str = "raw_probability_diagnostics_v1",
    label_policy_id: str | None = None,
) -> RegimeControlledStudyRun:
    """Module docstring. ``bundle_registry`` (default: the published bundle
    registry) must carry the regime-bearing challenger bundle; both bundles
    are resolved under the ACTIVATION's registries (the pure event's
    mappings), never the module registries."""

    assert_supervised_regime_authority(
        activation, fold_features, candidate_fold_set=candidate_fold_set, folds=folds
    )
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
    registry = bundle_registry if bundle_registry is not None else FEATURE_BUNDLE_REGISTRY
    definition = registry.get(challenger_bundle_key)
    if definition is None:
        raise ValueError(f"unregistered feature bundle {challenger_bundle_key!r}")
    challenger_env = resolve_bundle(
        challenger_bundle_key,
        bundle_registry=registry,
        definitions=activation.definitions,
        resolutions=activation.resolutions,
    )
    regime_block_id = activation.resolved_feature_block_id
    if regime_block_id not in challenger_env.payload.resolved_block_ids:
        raise ValueError(
            f"bundle {challenger_bundle_key!r} carries no regime block under the activated "
            "registry; the controlled regime study requires a regime-bearing challenger"
        )
    if definition.base_bundle_key is None:
        raise ValueError(
            f"bundle {challenger_bundle_key!r} has no base bundle; the controlled study "
            "baseline is the challenger's own base"
        )
    baseline_env = resolve_bundle(
        definition.base_bundle_key,
        bundle_registry=registry,
        definitions=activation.definitions,
        resolutions=activation.resolutions,
    )
    if regime_block_id in baseline_env.payload.resolved_block_ids:
        raise ValueError(
            "the challenger's base bundle also carries the regime block; the controlled "
            "contrast would not isolate the block"
        )
    challenger_names = tuple(challenger_env.payload.resolved_feature_names)
    baseline_names = tuple(baseline_env.payload.resolved_feature_names)
    regime_names = tuple(fold_features.feature_names)
    if set(regime_names) - set(challenger_names):
        raise ValueError(
            "the fold-feature artifact names model features the challenger bundle does not "
            "resolve (the block resolution and the artifact disagree)"
        )
    if tuple(name for name in challenger_names if name not in set(regime_names)) != (
        baseline_names
    ):
        raise ValueError(
            "the challenger bundle is not its base bundle extended by exactly the regime "
            "features (composition by extension is the study's premise)"
        )

    # the regime columns are FOLD-LOCAL: the static frame of BOTH arms is the
    # base bundle's view; the challenger ladder joins fit-k rows per fold
    baseline_view_env, baseline_frame = build_bundle_feature_view(view, baseline_env)
    arm_view = replace(view, frame=baseline_frame)
    schedule_id = fold_features.envelope.payload.fold_schedule_id
    baseline_run = run_supervised_ladder(
        arm_view,
        labeled_candidates,
        folds,
        bundle_features=baseline_names,
        bundle_ref=baseline_env.resolved_feature_bundle_id,
        protocols=protocols,
        calibration_policy_id=calibration_policy_id,
        fold_schedule_id=schedule_id,
        label_artifact_id=label_artifact_id,
    )
    challenger_run = run_supervised_ladder(
        arm_view,
        labeled_candidates,
        folds,
        bundle_features=challenger_names,
        bundle_ref=challenger_env.resolved_feature_bundle_id,
        bundle_evidence_ref=fold_features.artifact_id,
        protocols=protocols,
        calibration_policy_id=calibration_policy_id,
        bundle_categorical_features=tuple(activation.envelope.payload.categorical_features),
        fold_local_features=fold_features,
        fold_schedule_id=schedule_id,
        label_artifact_id=label_artifact_id,
    )
    assert_regime_cross_arm_identity(baseline_run, challenger_run)
    deltas = _paired_deltas(baseline_run, challenger_run)

    payload = RegimeControlledStudyPayload(
        resolved_regime_protocol_id=activation.protocol.resolved_regime_protocol_id,
        regime_fold_feature_artifact_id=fold_features.artifact_id,
        regime_promotion_decision_id=activation.promotion_decision.regime_promotion_decision_id,
        owner_decision_artifact_id=activation.owner_decision.owner_decision_artifact_id,
        capability_assessment_id=activation.assessment.regime_capability_assessment_id,
        resolved_regime_block_id=regime_block_id,
        block_registry_hash=activation.registry_hash,
        baseline_bundle_key=baseline_env.payload.feature_bundle_key,
        challenger_bundle_key=challenger_bundle_key,
        baseline_resolved_bundle_id=baseline_env.resolved_feature_bundle_id,
        challenger_resolved_bundle_id=challenger_env.resolved_feature_bundle_id,
        view_id=view.view_id,
        label_artifact_id=label_artifact_id,
        label_content_hash=label_content_hash(labeled_candidates),
        fold_schedule_id=schedule_id,
        candidate_fold_set_id=candidate_fold_set.fold_set_artifact_id,
        fold_set_hash=candidate_fold_set.payload.fold_set_id,
        model_protocol_ids=tuple(protocols),
        calibration_policy_id=calibration_policy_id,
        baseline_ladder_id=baseline_run.ladder_id,
        challenger_ladder_id=challenger_run.ladder_id,
        oos_row_count=int(baseline_run.parity.get("oos_row_count", 0)),
        parity_status=baseline_run.parity.get("status", "not_evaluable"),
        baseline_summary=_summary(baseline_run),
        challenger_summary=_summary(challenger_run),
        paired_deltas=deltas,
    )
    detail = {
        "baseline_view_envelope": baseline_view_env.model_dump(mode="json"),
        "regime_block_resolution": activation.envelope.model_dump(mode="json"),
        "baseline": {
            "parity": jsonable(baseline_run.parity),
            "paired_deltas": jsonable(baseline_run.paired_deltas),
            "rungs": {
                rung.protocol_id: jsonable(rung.prediction_report) for rung in baseline_run.rungs
            },
        },
        "challenger": {
            "parity": jsonable(challenger_run.parity),
            "paired_deltas": jsonable(challenger_run.paired_deltas),
            "rungs": {
                rung.protocol_id: jsonable(rung.prediction_report)
                for rung in challenger_run.rungs
            },
        },
    }
    detail_bytes = (json.dumps(detail, sort_keys=True) + "\n").encode("utf-8")
    envelope = RegimeControlledStudyEnvelope.from_payload(
        payload, detail_sha256=hashlib.sha256(detail_bytes).hexdigest()
    )
    return RegimeControlledStudyRun(
        envelope=envelope,
        baseline=baseline_run,
        challenger=challenger_run,
        detail_bytes=detail_bytes,
        label_identity_proof=label_identity_proof,
    )


# ── persistence ──────────────────────────────────────────────────────────────


def save_regime_controlled_study(root: Path, run: RegimeControlledStudyRun) -> tuple:
    # review RB-01 (§7.2): only a run whose label identity was PROVEN exact persists
    from .comparison_rows import assert_persistable_label_proof  # noqa: PLC0415

    assert_persistable_label_proof(run.label_identity_proof, runner="run_controlled_regime_study")
    return save_or_reuse_envelope(
        Path(root),
        REGIME_CONTROLLED_STUDY_STORE,
        run.envelope,
        extra_files={_DETAIL_SIDECAR: run.detail_bytes},
    )


def load_regime_controlled_study(root: Path, study_id: str) -> RegimeControlledStudyEnvelope:
    return load_verified_envelope(
        Path(root), REGIME_CONTROLLED_STUDY_STORE, study_id, RegimeControlledStudyEnvelope
    )


def load_regime_controlled_study_detail(
    root: Path, envelope: RegimeControlledStudyEnvelope
) -> dict[str, Any]:
    data = load_sidecar_bytes(
        Path(root),
        REGIME_CONTROLLED_STUDY_STORE,
        envelope.regime_controlled_study_id,
        _DETAIL_SIDECAR,
    )
    if hashlib.sha256(data).hexdigest() != envelope.detail_sha256:
        raise ValueError("regime controlled-study detail sidecar fails the envelope hash check")
    return json.loads(data.decode("utf-8"))


def _example_payload() -> RegimeControlledStudyPayload:
    return RegimeControlledStudyPayload(
        resolved_regime_protocol_id="a" * 64,
        regime_fold_feature_artifact_id="b" * 64,
        regime_promotion_decision_id="c" * 64,
        owner_decision_artifact_id="d" * 64,
        capability_assessment_id="e" * 64,
        resolved_regime_block_id="f" * 64,
        block_registry_hash="1" * 64,
        baseline_bundle_key="B0_CORE",
        challenger_bundle_key="B7_CORE_REGIME",
        baseline_resolved_bundle_id="2" * 64,
        challenger_resolved_bundle_id="3" * 64,
        view_id="4" * 64,
        label_artifact_id="5" * 64,
        label_content_hash="6" * 64,
        fold_schedule_id="7" * 64,
        candidate_fold_set_id="8" * 64,
        fold_set_hash="9" * 64,
        model_protocol_ids=tuple(DEFAULT_BUNDLE_LADDER_PROTOCOLS),
        calibration_policy_id="raw_probability_diagnostics_v1",
        baseline_ladder_id="0" * 64,
        challenger_ladder_id="a1" * 32,
        oos_row_count=0,
        parity_status="not_evaluable",
        baseline_summary={},
        challenger_summary={},
        paired_deltas={},
    )


register_identity_pair(
    name="RegimeControlledStudy",
    envelope_cls=RegimeControlledStudyEnvelope,
    payload_cls=RegimeControlledStudyPayload,
    id_field="regime_controlled_study_id",
    example_factory=_example_payload,
    extra_envelope_fields=("detail_sha256",),
)
