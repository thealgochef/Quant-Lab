"""Comparison contract (DELTA_TAXONOMY.md §3; brief §7A.4–7A.5, §7A.15).

``build_comparison`` verifies the declared class against the observed
changed-dimension set (refusing mismatches), evaluates the class's required
equalities, and on failure returns ``config_diff_only`` with a full field diff
and NO delta families beyond identity + compatibility.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, Literal

from pydantic import Field

from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)
from .computation_path import ComputationPath
from .delta_outputs import select_delta_families
from .study_cell import StudyCellIdentity

__all__ = [
    "ComparisonClass",
    "ComparisonPayload",
    "ComparisonEnvelope",
    "ComparisonCompatibility",
    "ComparisonResult",
    "ComparisonResultEnvelope",
    "REQUIRED_EQUALITIES",
    "REQUIRED_DIFFERENCES",
    "ComparisonClassMismatchError",
    "build_comparison",
    "build_config_diff",
]


class ComparisonClass(StrEnum):
    FEATURE_ONLY = "feature_only"
    COHORT_DESCRIPTIVE = "cohort_descriptive"
    COHORT_MODEL = "cohort_model"
    STRATEGY_COUNTERFACTUAL = "strategy_counterfactual"
    LABEL_COUNTERFACTUAL = "label_counterfactual"
    MODEL_PROTOCOL = "model_protocol"
    MODEL_GATE_EXECUTION = "model_gate_execution"
    EXECUTION_POLICY = "execution_policy"
    COST_POLICY = "cost_policy"
    RISK_POLICY = "risk_policy"
    PROP_REALIZATION = "prop_realization"
    PAYOUT_POLICY = "payout_policy"
    PORTFOLIO_POLICY = "portfolio_policy"
    DATA_LINEAGE = "data_lineage"
    VALIDATION_PROTOCOL = "validation_protocol"
    STRESS_SCENARIO = "stress_scenario"
    COMPOSITE_PREREGISTERED = "composite_preregistered"
    CONFIGURATION_NEIGHBOR = "configuration_neighbor"
    BASELINE_TO_CHILD = "baseline_to_child"
    CHILD_TO_CHILD = "child_to_child"


class ComparisonPayload(FrozenContract):
    """The hashed payload — no self-id (V3 P0-1)."""

    baseline_cell_id: str = Field(pattern=SHA256_PATTERN)
    challenger_cell_ids: tuple[str, ...]
    changed_dimension_ids: tuple[str, ...]
    frozen_dimension_ids: tuple[str, ...]
    derived_dimension_ids: tuple[str, ...]
    comparison_class: ComparisonClass
    required_equalities: tuple[str, ...]
    permitted_differences: tuple[str, ...]
    computation_path: ComputationPath
    metric_registry: tuple[str, ...]
    uncertainty_protocol: Literal[
        "setup_and_day_block_bootstrap_10000_seed7_v1",
        "day_block_bootstrap_10000_seed7_v1",
    ]
    interpretation: Literal["descriptive", "modeled", "strategy_counterfactual"]


class ComparisonEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "comparison_id"

    comparison_id: str = Field(pattern=SHA256_PATTERN)
    payload: ComparisonPayload


class ComparisonCompatibility(FrozenContract):
    per_dimension_match: ImmutableMap[str, bool]
    required_equalities_satisfied: bool
    registered_single_axis_delta_only: bool
    compatible_for_metric_delta: bool
    compatibility_status: Literal["compatible", "config_diff_only"]
    incompatibility_reasons: tuple[str, ...]
    differing_fields: tuple[str, ...]


class ComparisonResult(FrozenContract):
    comparison_id: str = Field(pattern=SHA256_PATTERN)
    compatibility: ComparisonCompatibility
    delta_reports: ImmutableMap[str, ImmutableMap[str, Any]]
    config_diff: ImmutableMap[str, Any] | None
    evidence_links: tuple[str, ...]


class ComparisonResultEnvelope(EnvelopeBase):
    """Persisted comparison result (R5 S14; DEV-R4-16 closure).

    ``comparison_id`` inside the payload is a FOREIGN reference: for
    study-cell comparisons it names the frozen ``ComparisonEnvelope``; for
    the search lane it is the typed cross-profile derivation id of
    DECISIONS_TAKEN #42 (study cells require published v2 dataset
    references synthetic control-flow children do not have — DEV-R5-8).
    The envelope's own id hashes the complete result content, so
    re-computed identical deltas reuse one immutable artifact
    (``search_results`` store) and the UI consumes persisted contracts
    instead of rebuilding deltas at render time.
    """

    _ID_FIELD: ClassVar[str] = "comparison_result_id"

    comparison_result_id: str = Field(pattern=SHA256_PATTERN)
    payload: ComparisonResult


#: Required equalities per class (§7A.15) — technical keys resolved against
#: each side's equality-evidence mapping (cell fields + run-artifact facts).
REQUIRED_EQUALITIES: MappingProxyType[ComparisonClass, tuple[str, ...]] = MappingProxyType(
    {
        ComparisonClass.FEATURE_ONLY: (
            "artifact_pair_hash",
            "strategy_profile",
            "cohort_id",
            "observation_filters_hash",
            "label_policy_id",
            "label_derivation_id",
            "fold_set_hash",
            "resolved_model_protocol_id",
            "cost_policy_id",
            "oos_row_ids",
        ),
        ComparisonClass.COHORT_DESCRIPTIVE: (
            "artifact_pair_hash",
            "strategy_profile",
            "label_policy_id",
            "data_lineage",
        ),
        ComparisonClass.COHORT_MODEL: (
            "artifact_pair_hash",
            "strategy_profile",
            "label_policy_id",
            "label_derivation_id",
            "fold_set_hash",
            "resolved_model_protocol_id",
            "cost_policy_id",
        ),
        ComparisonClass.LABEL_COUNTERFACTUAL: (
            "artifact_pair_hash",
            "strategy_profile",
            "cohort_id",
            "fold_protocol_id",
            "candidate_ids",
        ),
        ComparisonClass.MODEL_PROTOCOL: (
            "oos_row_ids",
            "label_derivation_id",
            "fold_set_hash",
            "resolved_feature_bundle_id",
            "cost_policy_id",
        ),
        ComparisonClass.MODEL_GATE_EXECUTION: ("frozen_model_fit_id",),
        ComparisonClass.EXECUTION_POLICY: ("cost_policy_id",),
        ComparisonClass.COST_POLICY: ("gross_trade_stream_hash",),
        ComparisonClass.RISK_POLICY: (
            "gross_trade_stream_hash",
            "prop_contract_id",
            "withdrawal_policy_id",
            "replacement_policy",
            "clock_policy_id",
            "trade_path_bundle_id",
            "path_capability_report_id",
        ),
        ComparisonClass.PROP_REALIZATION: (
            "gross_trade_stream_hash",
            "risk_policy_id",
            "withdrawal_policy_id",
            "replacement_policy",
            "clock_policy_id",
            "trade_path_bundle_id",
            "path_capability_report_id",
        ),
        ComparisonClass.PAYOUT_POLICY: (
            "gross_trade_stream_hash",
            "prop_contract_id",
            "risk_policy_id",
        ),
        ComparisonClass.PORTFOLIO_POLICY: ("common_market_path_id",),
        ComparisonClass.STRATEGY_COUNTERFACTUAL: (
            "authorized_date_set_id",
            "cost_policy_id",
            "label_policy_id",
            "execution_policy_id",
            "fold_protocol_id",
        ),
        ComparisonClass.CONFIGURATION_NEIGHBOR: (
            "authorized_date_set_id",
            "cost_policy_id",
            "label_policy_id",
            "execution_policy_id",
            "fold_protocol_id",
        ),
        ComparisonClass.BASELINE_TO_CHILD: (
            "authorized_date_set_id",
            "cost_policy_id",
            "label_policy_id",
            "execution_policy_id",
            "fold_protocol_id",
        ),
        ComparisonClass.CHILD_TO_CHILD: (
            "authorized_date_set_id",
            "cost_policy_id",
            "label_policy_id",
            "execution_policy_id",
            "fold_protocol_id",
        ),
        ComparisonClass.DATA_LINEAGE: ("everything_else_equal",),
        ComparisonClass.VALIDATION_PROTOCOL: ("everything_else_equal",),
        ComparisonClass.STRESS_SCENARIO: ("everything_else_equal",),
        ComparisonClass.COMPOSITE_PREREGISTERED: ("charter_frozen_equality_list",),
    }
)

#: Which top-level cell dimensions each class REQUIRES changed, and which it
#: PERMITS changed. ``data_lineage`` is always permitted as a derived change
#: (derived artifact ids ride inside it).
_CLASS_DIMENSIONS: MappingProxyType[ComparisonClass, tuple[tuple[str, ...], tuple[str, ...]]] = (
    MappingProxyType(
        {
            ComparisonClass.FEATURE_ONLY: (("feature_bundle",), ("feature_bundle",)),
            ComparisonClass.COHORT_DESCRIPTIVE: (
                ("observation_cohort",),
                ("observation_cohort",),
            ),
            ComparisonClass.COHORT_MODEL: (
                ("observation_cohort",),
                ("observation_cohort",),
            ),
            ComparisonClass.STRATEGY_COUNTERFACTUAL: (
                ("strategy_profile",),
                ("strategy_profile",),
            ),
            ComparisonClass.LABEL_COUNTERFACTUAL: (("label_policy",), ("label_policy",)),
            ComparisonClass.MODEL_PROTOCOL: (("model_protocol",), ("model_protocol",)),
            ComparisonClass.MODEL_GATE_EXECUTION: (
                ("decision_policy",),
                ("decision_policy",),
            ),
            ComparisonClass.EXECUTION_POLICY: (
                ("execution_policy",),
                ("execution_policy",),
            ),
            ComparisonClass.COST_POLICY: (("cost_policy",), ("cost_policy",)),
            ComparisonClass.RISK_POLICY: (("risk_policy",), ("risk_policy",)),
            ComparisonClass.PROP_REALIZATION: (
                ("prop_contract",),
                ("prop_contract", "risk_policy"),
            ),
            ComparisonClass.PAYOUT_POLICY: (("payout_policy",), ("payout_policy",)),
            ComparisonClass.PORTFOLIO_POLICY: (
                ("portfolio_policy",),
                ("portfolio_policy",),
            ),
            ComparisonClass.DATA_LINEAGE: (("data_lineage",), ("data_lineage",)),
            ComparisonClass.VALIDATION_PROTOCOL: (
                ("validation_protocol",),
                ("validation_protocol",),
            ),
            ComparisonClass.STRESS_SCENARIO: (
                ("stress_scenario",),
                ("stress_scenario",),
            ),
            ComparisonClass.COMPOSITE_PREREGISTERED: ((), tuple()),
            ComparisonClass.CONFIGURATION_NEIGHBOR: (
                ("strategy_profile",),
                ("strategy_profile",),
            ),
            ComparisonClass.BASELINE_TO_CHILD: (
                ("strategy_profile",),
                ("strategy_profile",),
            ),
            ComparisonClass.CHILD_TO_CHILD: (
                ("strategy_profile",),
                ("strategy_profile",),
            ),
        }
    )
)


class ComparisonClassMismatchError(ValueError):
    """The declared class does not match the observed changed-dimension set."""


def _changed_dimensions(
    baseline: StudyCellIdentity, challenger: StudyCellIdentity
) -> tuple[str, ...]:
    changed = []
    for name in type(baseline.payload).model_fields:
        left = getattr(baseline.payload, name)
        right = getattr(challenger.payload, name)
        if left.model_dump(mode="json") != right.model_dump(mode="json"):
            changed.append(name)
    return tuple(changed)


#: Classes whose required-EQUALITY semantics are "every dimension other than
#: the declared one is identical" — evaluated structurally from the cell diff,
#: never from caller-supplied evidence tokens.
_EVERYTHING_ELSE_EQUAL_CLASSES = frozenset(
    {
        ComparisonClass.DATA_LINEAGE,
        ComparisonClass.VALIDATION_PROTOCOL,
        ComparisonClass.STRESS_SCENARIO,
    }
)

#: Required INEQUALITIES per class: model-gated execution must produce a NEW
#: trade stream — equal stream hashes are an incompatibility (§7A.15).
REQUIRED_DIFFERENCES: MappingProxyType[ComparisonClass, tuple[str, ...]] = MappingProxyType(
    {ComparisonClass.MODEL_GATE_EXECUTION: ("gross_trade_stream_hash",)}
)


def build_config_diff(
    baseline: StudyCellIdentity, challengers: tuple[StudyCellIdentity, ...]
) -> dict[str, dict[str, Any]]:
    """Full per-dimension, per-field diff for `config_diff_only` results."""

    diff: dict[str, dict[str, Any]] = {}
    base_dump = baseline.payload.model_dump(mode="json")
    for challenger in challengers:
        challenger_dump = challenger.payload.model_dump(mode="json")
        for dimension, base_value in base_dump.items():
            other_value = challenger_dump[dimension]
            if base_value == other_value:
                continue
            entry = diff.setdefault(dimension, {})
            for field in set(base_value) | set(other_value):
                if base_value.get(field) != other_value.get(field):
                    entry.setdefault(field, {"baseline": base_value.get(field), "challengers": {}})
                    entry[field]["challengers"][challenger.cell_id] = other_value.get(field)
    return diff


def build_comparison(
    *,
    baseline: StudyCellIdentity,
    challengers: tuple[StudyCellIdentity, ...],
    comparison_class: ComparisonClass,
    computation_path: ComputationPath,
    uncertainty_protocol: str = "setup_and_day_block_bootstrap_10000_seed7_v1",
    interpretation: str,
    equality_evidence: dict[str, dict[str, Any]],
    permitted_differences: tuple[str, ...] = (),
    charter_equalities: tuple[str, ...] = (),
) -> tuple[ComparisonEnvelope, ComparisonCompatibility]:
    """Declare, verify, and freeze one comparison (§7A.15).

    ``equality_evidence`` maps ``cell_id`` → flat technical-key evidence used
    to evaluate the class's required equalities (cell fields plus run-artifact
    facts like ``oos_row_ids`` or ``gross_trade_stream_hash``).
    ``charter_equalities`` is the charter-frozen equality list required for —
    and only lawful with — ``composite_preregistered`` comparisons.
    """

    if not challengers:
        raise ValueError("a comparison requires at least one challenger cell")
    if charter_equalities and comparison_class is not ComparisonClass.COMPOSITE_PREREGISTERED:
        raise ValueError(
            "charter_equalities apply only to composite_preregistered comparisons"
        )
    observed: set[str] = set()
    for challenger in challengers:
        observed.update(_changed_dimensions(baseline, challenger))
    required_changed, permitted_changed = _CLASS_DIMENSIONS[comparison_class]
    non_derived = observed - {"data_lineage"}
    if comparison_class is not ComparisonClass.COMPOSITE_PREREGISTERED:
        missing = set(required_changed) - observed
        if missing:
            raise ComparisonClassMismatchError(
                f"declared class {comparison_class.value} requires changed "
                f"dimensions {sorted(missing)} but they are identical"
            )
        unexpected = non_derived - set(permitted_changed)
        if unexpected:
            raise ComparisonClassMismatchError(
                f"declared class {comparison_class.value} does not permit changed "
                f"dimensions {sorted(unexpected)}"
            )

    reasons: list[str] = []
    differing: list[str] = []
    if comparison_class in _EVERYTHING_ELSE_EQUAL_CLASSES:
        # structural: nothing outside the declared dimension may differ
        required = REQUIRED_EQUALITIES[comparison_class]
        stray = observed - set(required_changed)
        if stray:
            reasons.append(
                "everything-else-equal violated by changed dimensions "
                + ", ".join(sorted(stray))
            )
            differing.extend(sorted(stray))
    elif comparison_class is ComparisonClass.COMPOSITE_PREREGISTERED:
        if not charter_equalities:
            reasons.append(
                "composite_preregistered requires the charter-frozen equality list"
            )
        required = charter_equalities
    else:
        required = REQUIRED_EQUALITIES[comparison_class]
    if comparison_class not in _EVERYTHING_ELSE_EQUAL_CLASSES:
        baseline_evidence = equality_evidence.get(baseline.cell_id, {})
        for challenger in challengers:
            challenger_evidence = equality_evidence.get(challenger.cell_id, {})
            for key in required:
                left = baseline_evidence.get(key)
                right = challenger_evidence.get(key)
                if left is None or right is None:
                    reasons.append(f"required equality {key} lacks evidence")
                    differing.append(key)
                elif left != right:
                    reasons.append(f"required equality {key} differs")
                    differing.append(key)
            for key in REQUIRED_DIFFERENCES.get(comparison_class, ()):
                left = baseline_evidence.get(key)
                right = challenger_evidence.get(key)
                if left is None or right is None:
                    reasons.append(f"required difference {key} lacks evidence")
                    differing.append(key)
                elif left == right:
                    reasons.append(
                        f"required difference {key} was not observed (values equal)"
                    )
                    differing.append(key)

    satisfied = not reasons
    if not satisfied:
        # config_diff_only carries the FULL field diff (§7A.15 last rule)
        differing.extend(
            f"{dimension}.{field}"
            for dimension, fields in build_config_diff(baseline, challengers).items()
            for field in fields
        )
    dimension_names = tuple(type(baseline.payload).model_fields)
    compatibility = ComparisonCompatibility(
        per_dimension_match={name: name not in observed for name in dimension_names},
        required_equalities_satisfied=satisfied,
        registered_single_axis_delta_only=len(non_derived) <= 1,
        compatible_for_metric_delta=satisfied,
        compatibility_status="compatible" if satisfied else "config_diff_only",
        incompatibility_reasons=tuple(sorted(set(reasons))),
        differing_fields=tuple(sorted(set(differing))),
    )
    families = (
        tuple(family.value for family in select_delta_families(comparison_class.value))
        if satisfied
        else ("identity_delta", "compatibility_delta")
    )
    payload = ComparisonPayload(
        baseline_cell_id=baseline.cell_id,
        challenger_cell_ids=tuple(cell.cell_id for cell in challengers),
        changed_dimension_ids=tuple(sorted(observed)),
        frozen_dimension_ids=tuple(
            name for name in dimension_names if name not in observed
        ),
        derived_dimension_ids=("data_lineage",) if "data_lineage" in observed else (),
        comparison_class=comparison_class,
        required_equalities=required,
        permitted_differences=permitted_differences,
        computation_path=computation_path,
        metric_registry=families,
        uncertainty_protocol=uncertainty_protocol,  # type: ignore[arg-type]
        interpretation=interpretation,  # type: ignore[arg-type]
    )
    return ComparisonEnvelope.from_payload(payload), compatibility


def _example_comparison_payload() -> ComparisonPayload:
    from .study_cell import BASELINE_STUDY_CELL  # noqa: PLC0415

    return ComparisonPayload(
        baseline_cell_id=BASELINE_STUDY_CELL.cell_id,
        challenger_cell_ids=("f" * 64,),
        changed_dimension_ids=("strategy_profile",),
        frozen_dimension_ids=("cost_policy",),
        derived_dimension_ids=(),
        comparison_class=ComparisonClass.STRATEGY_COUNTERFACTUAL,
        required_equalities=REQUIRED_EQUALITIES[ComparisonClass.STRATEGY_COUNTERFACTUAL],
        permitted_differences=(),
        computation_path=ComputationPath(
            full_strategy_replay=True,
            feature_materialization=False,
            label_recomputation=False,
            model_refit=False,
            model_gated_sequential_replay=False,
            cost_recomputation=True,
            prop_resimulation=True,
            bootstrap_resimulation=True,
            reuse_trade_stream_hash=False,
        ),
        metric_registry=("identity_delta",),
        uncertainty_protocol="setup_and_day_block_bootstrap_10000_seed7_v1",
        interpretation="strategy_counterfactual",
    )


register_identity_pair(
    name="Comparison",
    envelope_cls=ComparisonEnvelope,
    payload_cls=ComparisonPayload,
    id_field="comparison_id",
    example_factory=_example_comparison_payload,
)


def _example_comparison_result() -> ComparisonResult:
    return ComparisonResult(
        comparison_id="a" * 64,
        compatibility=ComparisonCompatibility(
            per_dimension_match={"strategy_profile": False},
            required_equalities_satisfied=True,
            registered_single_axis_delta_only=True,
            compatible_for_metric_delta=True,
            compatibility_status="compatible",
            incompatibility_reasons=(),
            differing_fields=(),
        ),
        delta_reports={"population_delta": {"entity_kind": "setup"}},
        config_diff=None,
        evidence_links=("b" * 64,),
    )


register_identity_pair(
    name="ComparisonResult",
    envelope_cls=ComparisonResultEnvelope,
    payload_cls=ComparisonResult,
    id_field="comparison_result_id",
    example_factory=_example_comparison_result,
)
