"""Public CohortSpec contract (CONTRACTS_AND_SCHEMAS.md §9.1; brief §7A.13).

The three interpretation modes stay distinct: only
``sequential_strategy_profile`` creates an executable counterfactual child —
never a filter over an existing candidate table.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import Field

from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)

__all__ = [
    "InterpretationMode",
    "RegimeFilterRef",
    "CohortPayload",
    "CohortEnvelope",
    "BASELINE_COHORT",
    "interpretation_to_comparison_class",
]


class InterpretationMode(StrEnum):
    DESCRIPTIVE_SLICE = "descriptive_slice"
    SPECIALIZED_MODEL = "specialized_model"
    SEQUENTIAL_STRATEGY_PROFILE = "sequential_strategy_profile"


class RegimeFilterRef(FrozenContract):
    resolved_regime_protocol_id: str
    regime_fit_ids: tuple[str, ...]
    canonical_reporting_cluster_ids: tuple[int, ...]
    assignment_partition: Literal["test_oos"] = "test_oos"


class CohortPayload(FrozenContract):
    interpretation_mode: InterpretationMode
    date_policy_id: str
    warmup_policy: Literal["exclude_warmup_v1", "include_warmup_v1"]
    row_kind: Literal["entry_candidate", "eligible_decision", "executed_trade", "context_bar"]
    session_filter: tuple[str, ...] | None
    direction_filter: Literal["long", "short"] | None
    entry_family_filter: tuple[str, ...] | None
    htf_timeframe_filter: tuple[int, ...] | None
    parent_timeframe_filter: tuple[int, ...] | None
    regime_filter: RegimeFilterRef | None
    evidence_quality_filter: tuple[str, ...] | None
    executed_counterfactual_scope: Literal["executed_only", "counterfactual_only", "both"]
    minimum_coverage: ImmutableMap[str, float]


class CohortEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "cohort_id"

    cohort_id: str = Field(pattern=SHA256_PATTERN)
    payload: CohortPayload


def _baseline_cohort_payload() -> CohortPayload:
    return CohortPayload(
        interpretation_mode=InterpretationMode.DESCRIPTIVE_SLICE,
        date_policy_id="development_explicit_dates_before_path_v2",
        warmup_policy="exclude_warmup_v1",
        row_kind="entry_candidate",
        session_filter=None,
        direction_filter=None,
        entry_family_filter=None,
        htf_timeframe_filter=None,
        parent_timeframe_filter=None,
        regime_filter=None,
        evidence_quality_filter=None,
        executed_counterfactual_scope="both",
        minimum_coverage={},
    )


#: §7A.2 baseline cohort: all post-warmup candidates, unfiltered.
BASELINE_COHORT: CohortEnvelope = CohortEnvelope.from_payload(_baseline_cohort_payload())


def interpretation_to_comparison_class(mode: InterpretationMode) -> str:
    """§7A.14 selector mapping (DT §8) — 1:1, no fourth option."""

    return {
        InterpretationMode.DESCRIPTIVE_SLICE: "cohort_descriptive",
        InterpretationMode.SPECIALIZED_MODEL: "cohort_model",
        InterpretationMode.SEQUENTIAL_STRATEGY_PROFILE: "strategy_counterfactual",
    }[mode]


register_identity_pair(
    name="Cohort",
    envelope_cls=CohortEnvelope,
    payload_cls=CohortPayload,
    id_field="cohort_id",
    example_factory=_baseline_cohort_payload,
)
