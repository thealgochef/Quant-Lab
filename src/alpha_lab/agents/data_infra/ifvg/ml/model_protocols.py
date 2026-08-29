"""Supervised model-protocol registry (`ML_REGIME_CONTRACT_PLAN.md` §2).

The registry is the ONLY route to a ladder rung: `run_supervised_ladder`
accepts protocol ids, never model objects or parameter dicts. Planned
entries (`ifvg_context_gam_v1`) refuse execution fail-closed with their
registered reason, and every enumeration-shaped request outside a frozen,
owner-ratified charter raises :class:`ProhibitedSelectionError`
(brief §7B.3; acceptance 7B.22-15).
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from ..search.identities import FrozenContract
from ..study.study_cell import REGISTERED_MODEL_PROTOCOL_KEYS

__all__ = [
    "ModelProtocolStatus",
    "ModelProtocolEntry",
    "MODEL_PROTOCOL_REGISTRY",
    "PREVALENCE_PROTOCOL_ID",
    "LOGISTIC_PROTOCOL_ID",
    "CATBOOST_PROTOCOL_ID",
    "CATBOOST_BUNDLE_PROTOCOL_ID",
    "GAM_PROTOCOL_ID",
    "ModelProtocolUnavailableError",
    "ProhibitedSelectionError",
    "resolve_model_protocol_entry",
    "assert_model_protocol_available",
    "assert_single_frozen_selection",
]

PREVALENCE_PROTOCOL_ID = "reference_prevalence_v1"
LOGISTIC_PROTOCOL_ID = "ifvg_context_logistic_l2_v1"
CATBOOST_PROTOCOL_ID = "ifvg_context_catboost_binary_v1"
#: R6.1 (§6.J): the bundle-aware CatBoost rung — the frozen lane's parameters
#: reused by value over the exact resolved features of ONE bundle (+ optional
#: fold-local regime features); research-only; no selection surface.
CATBOOST_BUNDLE_PROTOCOL_ID = "ifvg_context_catboost_bundle_v1"
GAM_PROTOCOL_ID = "ifvg_context_gam_v1"


class ModelProtocolStatus(StrEnum):
    AVAILABLE = "available"
    PLANNED = "planned"


class ModelProtocolEntry(FrozenContract):
    protocol_id: str
    status: ModelProtocolStatus
    kind: Literal[
        "reference",
        "interpretable",
        "nonlinear_challenger",
        "nonlinear_challenger_bundle",
        "planned_basis_expansion",
    ]
    reason: str | None = None


MODEL_PROTOCOL_REGISTRY: MappingProxyType[str, ModelProtocolEntry] = MappingProxyType(
    {
        PREVALENCE_PROTOCOL_ID: ModelProtocolEntry(
            protocol_id=PREVALENCE_PROTOCOL_ID,
            status=ModelProtocolStatus.AVAILABLE,
            kind="reference",
        ),
        LOGISTIC_PROTOCOL_ID: ModelProtocolEntry(
            protocol_id=LOGISTIC_PROTOCOL_ID,
            status=ModelProtocolStatus.AVAILABLE,
            kind="interpretable",
        ),
        CATBOOST_PROTOCOL_ID: ModelProtocolEntry(
            protocol_id=CATBOOST_PROTOCOL_ID,
            status=ModelProtocolStatus.AVAILABLE,
            kind="nonlinear_challenger",
        ),
        CATBOOST_BUNDLE_PROTOCOL_ID: ModelProtocolEntry(
            protocol_id=CATBOOST_BUNDLE_PROTOCOL_ID,
            status=ModelProtocolStatus.AVAILABLE,
            kind="nonlinear_challenger_bundle",
        ),
        GAM_PROTOCOL_ID: ModelProtocolEntry(
            protocol_id=GAM_PROTOCOL_ID,
            status=ModelProtocolStatus.PLANNED,
            kind="planned_basis_expansion",
            reason="preregistered_basis_penalty_protocol_not_ratified",
        ),
    }
)

if tuple(MODEL_PROTOCOL_REGISTRY) != REGISTERED_MODEL_PROTOCOL_KEYS:
    raise AssertionError(
        "the ml model-protocol registry must cover exactly the study-cell "
        "registered model protocol keys, in order"
    )


class ModelProtocolUnavailableError(PermissionError):
    """A registered-but-planned protocol was requested for execution."""


class ProhibitedSelectionError(PermissionError):
    """An automated-selection request outside a frozen, owner-ratified charter.

    Brief §7B.3: enumerating more than one value for a threshold, cluster
    count, calibrator, or feature subset is prohibited unless the request
    rides a frozen charter carrying owner ratification for that exact
    enumeration.
    """


def resolve_model_protocol_entry(protocol_id: str) -> ModelProtocolEntry:
    """The registered entry for ``protocol_id`` (unknown ids are refused)."""

    entry = MODEL_PROTOCOL_REGISTRY.get(protocol_id)
    if entry is None:
        raise ValueError(
            f"unknown model protocol id {protocol_id!r}; registered ids: "
            f"{sorted(MODEL_PROTOCOL_REGISTRY)}"
        )
    return entry


def assert_model_protocol_available(protocol_id: str) -> ModelProtocolEntry:
    """Fail closed on planned protocols; the reason is the registered one."""

    entry = resolve_model_protocol_entry(protocol_id)
    if entry.status is not ModelProtocolStatus.AVAILABLE:
        raise ModelProtocolUnavailableError(
            f"model protocol {protocol_id!r} is {entry.status.value}: {entry.reason}"
        )
    return entry


def assert_single_frozen_selection(
    kind: Literal["threshold", "cluster_count", "calibrator", "feature_subset"],
    values: Iterable[object],
    *,
    owner_ratification_ref: str | None = None,
) -> None:
    """Refuse enumeration of selection-bearing values outside a ratified charter.

    A single value is always lawful. More than one distinct value requires an
    owner ratification reference carried by a frozen charter; without it the
    request is a prohibited automated selection (acceptance 7B.22-15).
    """

    distinct = {repr(value) for value in values}
    if len(distinct) <= 1:
        return
    if owner_ratification_ref:
        return
    raise ProhibitedSelectionError(
        f"enumerating {len(distinct)} {kind} values is an automated selection; "
        "it requires a frozen charter carrying owner ratification for exactly "
        "this enumeration (brief §7B.3)"
    )
