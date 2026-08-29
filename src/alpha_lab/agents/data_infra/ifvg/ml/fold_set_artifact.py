"""Persisted fold-set artifacts (R6.1 D3 / §6.B).

ONE legacy row-population hash (:func:`fold_set_id`) now serves every lane —
``regime_service``, the supervised ladder, and the controlled feature study
delegate to it, and a three-way equality test pins that no existing identity
moved. The :class:`FoldSetArtifactEnvelope` adds the persisted, verified form
S08 emits: its payload binds the grain-agnostic ``fold_schedule_id``, the
observation grain and key, the observation source artifact, the training
floor, the labeled flag, the legacy ``fold_set_id``, and one
:class:`FoldSetMember` per fold (windows, counts, id hashes); the envelope's
``fold_definitions_sha256`` binds the ``fold_definitions.json`` sidecar that
carries the complete id lists (train/test/purged/embargoed/boundary setups).
Reload rebuilds the ``IfvgContextFoldDefinition``s and refuses unless the
rebuilt legacy hash equals the payload's ``fold_set_id`` and every member's
windows equal the persisted schedule's.

Cross-grain rule (:func:`assert_same_fold_schedule`): two fold sets are
comparable when their ``fold_schedule_id``s AND per-fold ``(train_days,
test_days)`` windows agree; their ``fold_set_id``s are EXPECTED to differ
across grains and are never compared.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import Field, model_validator

from ..context_experiment_contracts import IfvgContextFoldDefinition
from ..context_folds import ContextFoldSet
from ..fold_schedules import (
    FoldScheduleEnvelope,
    FoldWindow,
    derive_fold_schedule,
)
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    register_identity_pair,
)
from ..search.store import load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope
from .regime_contracts import ObservationGranularity

__all__ = [
    "FOLD_SCHEDULE_STORE",
    "FOLD_SET_STORE",
    "FOLD_DEFINITIONS_SIDECAR",
    "fold_set_hash_body",
    "fold_set_id",
    "FoldSetMember",
    "FoldSetArtifactPayload",
    "FoldSetArtifactEnvelope",
    "build_fold_set_artifact",
    "persist_fold_schedule",
    "load_fold_schedule",
    "persist_fold_set_artifact",
    "load_fold_set_artifact",
    "assert_same_fold_schedule",
    "observation_key_for_grain",
]

FOLD_SCHEDULE_STORE = "fold_schedules"
FOLD_SET_STORE = "fold_sets"
FOLD_DEFINITIONS_SIDECAR = "fold_definitions.json"


def fold_set_hash_body(folds: ContextFoldSet) -> dict[str, Any]:
    """The ONE legacy row-population body every lane hashes."""

    return {
        "folds": [
            {
                "fold_index": fold.fold_index,
                "valid": fold.valid,
                "train_candidate_ids": list(fold.train_candidate_ids),
                "test_candidate_ids": list(fold.test_candidate_ids),
            }
            for fold in folds.folds
        ]
    }


def fold_set_id(folds: ContextFoldSet) -> str:
    return canonical_contract_sha256(fold_set_hash_body(folds))


def observation_key_for_grain(grain: ObservationGranularity) -> str:
    grain = ObservationGranularity(grain)
    if grain is ObservationGranularity.CONTEXT_BAR_PANEL:
        return "row_id"
    return "candidate_id"


class FoldSetMember(FrozenContract):
    fold_index: int = Field(ge=0)
    valid: bool
    invalid_reason: str | None
    train_days: tuple[str, ...]
    test_days: tuple[str, ...]
    train_ids_hash: str = Field(pattern=SHA256_PATTERN)
    test_ids_hash: str = Field(pattern=SHA256_PATTERN)
    train_count: int = Field(ge=0)
    test_count: int = Field(ge=0)
    purged_count: int = Field(ge=0)
    embargoed_count: int = Field(ge=0)
    excluded_boundary_setup_count: int = Field(ge=0)


class FoldSetArtifactPayload(FrozenContract):
    fold_schedule_id: str = Field(pattern=SHA256_PATTERN)
    fold_protocol_id: str
    observation_grain: ObservationGranularity
    observation_key: Literal["candidate_id", "row_id"]
    observation_source_artifact_id: str = Field(pattern=SHA256_PATTERN)
    authorized_trading_days: tuple[str, ...] = Field(min_length=1)
    minimum_train_observations: int = Field(ge=0)
    labeled: bool
    #: the legacy row-population hash (``fold_set_id(folds)``)
    fold_set_id: str = Field(pattern=SHA256_PATTERN)
    folds: tuple[FoldSetMember, ...]

    @model_validator(mode="after")
    def _key_matches_grain(self):
        if self.observation_key != observation_key_for_grain(self.observation_grain):
            raise ValueError(
                f"{self.observation_grain.value} folds are keyed by "
                f"{observation_key_for_grain(self.observation_grain)!r}"
            )
        indices = [member.fold_index for member in self.folds]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ValueError("fold members must be unique and ordered by fold index")
        return self


class FoldSetArtifactEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "fold_set_artifact_id"

    fold_set_artifact_id: str = Field(pattern=SHA256_PATTERN)
    payload: FoldSetArtifactPayload
    #: post-materialization fact binding the ``fold_definitions.json`` sidecar
    fold_definitions_sha256: str = Field(pattern=SHA256_PATTERN)


def _ids_hash(ids: tuple[str, ...]) -> str:
    return canonical_contract_sha256({"ids": sorted(str(value) for value in ids)})


def _member(fold: IfvgContextFoldDefinition) -> FoldSetMember:
    return FoldSetMember(
        fold_index=fold.fold_index,
        valid=fold.valid,
        invalid_reason=fold.invalid_reason,
        train_days=tuple(fold.train_days),
        test_days=tuple(fold.test_days),
        train_ids_hash=_ids_hash(fold.train_candidate_ids),
        test_ids_hash=_ids_hash(fold.test_candidate_ids),
        train_count=len(fold.train_candidate_ids),
        test_count=len(fold.test_candidate_ids),
        purged_count=len(fold.purged_candidate_ids),
        embargoed_count=len(fold.embargoed_candidate_ids),
        excluded_boundary_setup_count=len(fold.excluded_boundary_setup_ids),
    )


def _definitions_bytes(folds: ContextFoldSet) -> bytes:
    payload = [fold.model_dump(mode="json") for fold in folds.folds]
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _assert_windows_match(
    members: tuple[FoldSetMember, ...] | tuple[IfvgContextFoldDefinition, ...],
    windows: tuple[FoldWindow, ...],
) -> None:
    by_index = {window.fold_index: window for window in windows}
    if len(members) != len(windows):
        raise ValueError(
            f"fold set carries {len(members)} folds but the schedule has {len(windows)}"
        )
    for member in members:
        window = by_index.get(member.fold_index)
        if window is None:
            raise ValueError(f"fold {member.fold_index} is not in the schedule")
        if tuple(member.train_days) != tuple(window.train_days) or (
            tuple(member.test_days) != tuple(window.test_days)
        ):
            raise ValueError(
                f"fold {member.fold_index} windows disagree with the schedule"
            )


def build_fold_set_artifact(
    folds: ContextFoldSet,
    *,
    schedule: FoldScheduleEnvelope,
    observation_grain: ObservationGranularity,
    observation_source_artifact_id: str,
    minimum_train_observations: int,
    labeled: bool,
) -> tuple[FoldSetArtifactEnvelope, bytes]:
    """The persisted form of one fold set under one schedule."""

    grain = ObservationGranularity(observation_grain)
    _assert_windows_match(tuple(folds.folds), schedule.payload.windows)
    definitions = _definitions_bytes(folds)
    payload = FoldSetArtifactPayload(
        fold_schedule_id=schedule.fold_schedule_id,
        fold_protocol_id=schedule.payload.fold_protocol_id,
        observation_grain=grain,
        observation_key=observation_key_for_grain(grain),
        observation_source_artifact_id=observation_source_artifact_id,
        authorized_trading_days=schedule.payload.authorized_trading_days,
        minimum_train_observations=int(minimum_train_observations),
        labeled=bool(labeled),
        fold_set_id=fold_set_id(folds),
        folds=tuple(_member(fold) for fold in folds.folds),
    )
    envelope = FoldSetArtifactEnvelope.from_payload(
        payload, fold_definitions_sha256=hashlib.sha256(definitions).hexdigest()
    )
    return envelope, definitions


def persist_fold_schedule(root: Path, envelope: FoldScheduleEnvelope):
    return save_or_reuse_envelope(Path(root), FOLD_SCHEDULE_STORE, envelope)


def load_fold_schedule(root: Path, fold_schedule_id: str) -> FoldScheduleEnvelope:
    envelope = load_verified_envelope(
        Path(root), FOLD_SCHEDULE_STORE, fold_schedule_id, FoldScheduleEnvelope
    )
    # the schedule must still derive from its own days under its protocol
    rederived = derive_fold_schedule(
        envelope.payload.authorized_trading_days, protocol=envelope.payload.fold_protocol_id
    )
    if rederived.fold_schedule_id != fold_schedule_id:
        raise ValueError("stored fold schedule does not re-derive from its own days")
    return envelope


def persist_fold_set_artifact(
    root: Path, envelope: FoldSetArtifactEnvelope, definitions: bytes
):
    if hashlib.sha256(definitions).hexdigest() != envelope.fold_definitions_sha256:
        raise ValueError("fold definitions do not hash to the envelope's sidecar hash")
    return save_or_reuse_envelope(
        Path(root),
        FOLD_SET_STORE,
        envelope,
        extra_files={FOLD_DEFINITIONS_SIDECAR: definitions},
    )


def _rebuild_assignment(folds: tuple[IfvgContextFoldDefinition, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        for partition, ids in (
            ("train", fold.train_candidate_ids),
            ("test", fold.test_candidate_ids),
        ):
            rows.extend(
                {
                    "fold_index": fold.fold_index,
                    "candidate_id": str(value),
                    "partition": partition,
                    "fold_valid": fold.valid,
                    "fold_invalid_reason": fold.invalid_reason,
                }
                for value in sorted(str(v) for v in ids)
            )
    return pd.DataFrame(
        rows,
        columns=("fold_index", "candidate_id", "partition", "fold_valid", "fold_invalid_reason"),
    )


def load_fold_set_artifact(
    root: Path, fold_set_artifact_id: str
) -> tuple[FoldSetArtifactEnvelope, ContextFoldSet]:
    """Verified reload: rebuilds the fold definitions from the sidecar and
    refuses unless the legacy hash and the schedule windows agree."""

    envelope = load_verified_envelope(
        Path(root), FOLD_SET_STORE, fold_set_artifact_id, FoldSetArtifactEnvelope
    )
    data = load_sidecar_bytes(
        Path(root), FOLD_SET_STORE, fold_set_artifact_id, FOLD_DEFINITIONS_SIDECAR
    )
    if hashlib.sha256(data).hexdigest() != envelope.fold_definitions_sha256:
        raise ValueError("fold definitions sidecar fails the envelope hash check")
    definitions = tuple(
        IfvgContextFoldDefinition.model_validate(item)
        for item in json.loads(data.decode("utf-8"))
    )
    rebuilt = ContextFoldSet(
        folds=definitions,
        assignment=_rebuild_assignment(definitions),
        status="ready" if any(fold.valid for fold in definitions) else "no_valid_folds",
    )
    if fold_set_id(rebuilt) != envelope.payload.fold_set_id:
        raise ValueError(
            "rebuilt fold definitions do not hash to the artifact's fold_set_id; refusing"
        )
    for fold, member in zip(definitions, envelope.payload.folds, strict=True):
        if _member(fold) != member:
            raise ValueError(f"fold {fold.fold_index} member facts disagree with the sidecar")
    schedule = load_fold_schedule(Path(root), envelope.payload.fold_schedule_id)
    _assert_windows_match(envelope.payload.folds, schedule.payload.windows)
    return envelope, rebuilt


def assert_same_fold_schedule(
    left: FoldSetArtifactPayload, right: FoldSetArtifactPayload
) -> None:
    """Cross-grain comparability: equal schedule ids AND equal per-fold
    windows; ``fold_set_id``s are never compared (they differ across grains)."""

    if left.fold_schedule_id != right.fold_schedule_id:
        raise ValueError(
            "fold sets derive from different fold schedules "
            f"({left.fold_schedule_id[:12]}… vs {right.fold_schedule_id[:12]}…); "
            "cross-grain studies require one schedule"
        )
    if len(left.folds) != len(right.folds):
        raise ValueError("fold sets carry different fold counts under one schedule")
    for a, b in zip(left.folds, right.folds, strict=True):
        if (a.fold_index, a.train_days, a.test_days) != (b.fold_index, b.train_days, b.test_days):
            raise ValueError(f"fold {a.fold_index} windows differ between the fold sets")


def _example_fold_set_payload() -> FoldSetArtifactPayload:
    days = tuple(day.strftime("%Y-%m-%d") for day in pd.bdate_range("2026-01-05", periods=45))
    schedule = derive_fold_schedule(days)
    window = schedule.payload.windows[0]
    return FoldSetArtifactPayload(
        fold_schedule_id=schedule.fold_schedule_id,
        fold_protocol_id=schedule.payload.fold_protocol_id,
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_key="candidate_id",
        observation_source_artifact_id="a" * 64,
        authorized_trading_days=days,
        minimum_train_observations=150,
        labeled=False,
        fold_set_id="b" * 64,
        folds=(
            FoldSetMember(
                fold_index=0,
                valid=False,
                invalid_reason="insufficient_train_candidates",
                train_days=window.train_days,
                test_days=window.test_days,
                train_ids_hash="c" * 64,
                test_ids_hash="d" * 64,
                train_count=0,
                test_count=0,
                purged_count=0,
                embargoed_count=0,
                excluded_boundary_setup_count=0,
            ),
        ),
    )


register_identity_pair(
    name="FoldSetArtifact",
    envelope_cls=FoldSetArtifactEnvelope,
    payload_cls=FoldSetArtifactPayload,
    id_field="fold_set_artifact_id",
    example_factory=_example_fold_set_payload,
    extra_envelope_fields=("fold_definitions_sha256",),
)
