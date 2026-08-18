"""Experiment-dimension registry (DELTA_TAXONOMY.md §1; brief §7A.3).

The registry fails closed: an unknown, blocked, or incompatible dimension can
never enter a study. Seeding encodes the brief §7A.5 computation table
row-by-row (verified by ``derive_computation_path`` tests).
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType
from typing import Literal

from ..search.identities import FrozenContract

__all__ = [
    "DimensionType",
    "DimensionCapabilityStatus",
    "OwnerRatificationStatus",
    "ExperimentDimensionSpec",
    "EXPERIMENT_DIMENSION_REGISTRY",
    "resolve_dimension",
    "assert_dimension_usable",
    "DimensionBlockedError",
]


class DimensionType(StrEnum):
    DATA_LINEAGE = "data_lineage"
    MARKET_UNIVERSE = "market_universe"
    STRATEGY_PROFILE = "strategy_profile"
    OBSERVATION_COHORT = "observation_cohort"
    LABEL_POLICY = "label_policy"
    FEATURE_BLOCK = "feature_block"
    MODEL_PROTOCOL = "model_protocol"
    DECISION_POLICY = "decision_policy"
    EXECUTION_POLICY = "execution_policy"
    COST_POLICY = "cost_policy"
    RISK_POLICY = "risk_policy"
    PROP_CONTRACT = "prop_contract"
    PAYOUT_POLICY = "payout_policy"
    PORTFOLIO_POLICY = "portfolio_policy"
    VALIDATION_PROTOCOL = "validation_protocol"
    STRESS_SCENARIO = "stress_scenario"
    ENGINEERING_PROTOCOL = "engineering_protocol"


class DimensionCapabilityStatus(StrEnum):
    AVAILABLE = "available"
    PLANNED = "planned"
    BLOCKED_MISSING_SOURCE = "blocked_missing_source"
    BLOCKED_OWNER_DECISION = "blocked_owner_decision"
    EXPERIMENTAL = "experimental"
    SUPERSEDED = "superseded"


class OwnerRatificationStatus(StrEnum):
    RATIFIED = "ratified"
    PENDING = "pending"
    NOT_REQUIRED = "not_required"


class DimensionBlockedError(PermissionError):
    """A blocked/unknown/incompatible dimension entered a study (fail-closed)."""


class ExperimentDimensionSpec(FrozenContract):
    dimension_id: str
    dimension_type: DimensionType
    human_name: str
    technical_key: str
    description: str
    baseline_value: str
    allowed_values: tuple[str, ...]
    value_schema: str
    capability_status: DimensionCapabilityStatus
    capability_reason: str | None = None
    owner_ratification_status: OwnerRatificationStatus
    # required-computation flags
    requires_full_strategy_replay: bool
    requires_feature_materialization: bool
    requires_label_recomputation: bool
    requires_model_refit: bool
    requires_model_gated_sequential_replay: bool
    requires_cost_recomputation: bool
    requires_prop_resimulation: bool
    requires_bootstrap_resimulation: bool
    # population/order effect flags
    can_change_setup_population: bool
    can_change_candidate_population: bool
    can_change_trade_order: bool
    can_change_account_state: bool
    can_change_label_meaning: bool
    compatible_dimensions: tuple[str, ...] = ()
    incompatible_dimensions: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    identity_effect: Literal["new_study_cell", "annotation_only"]


def _spec(
    dimension_id: str,
    dimension_type: DimensionType,
    human_name: str,
    technical_key: str,
    description: str,
    *,
    replay: bool = False,
    features: bool = False,
    labels: bool = False,
    refit: bool = False,
    gated: bool = False,
    cost: bool = False,
    prop: bool = False,
    bootstrap: bool = False,
    setup_pop: bool = False,
    cand_pop: bool = False,
    trade_order: bool = False,
    account: bool = False,
    label_meaning: bool = False,
    capability: DimensionCapabilityStatus = DimensionCapabilityStatus.AVAILABLE,
    capability_reason: str | None = None,
    ratification: OwnerRatificationStatus = OwnerRatificationStatus.PENDING,
    incompatible: tuple[str, ...] = (),
    identity_effect: str = "new_study_cell",
) -> ExperimentDimensionSpec:
    return ExperimentDimensionSpec(
        dimension_id=dimension_id,
        dimension_type=dimension_type,
        human_name=human_name,
        technical_key=technical_key,
        description=description,
        baseline_value="baseline",
        allowed_values=(),
        value_schema="registered_value_v1",
        capability_status=capability,
        capability_reason=capability_reason,
        owner_ratification_status=ratification,
        requires_full_strategy_replay=replay,
        requires_feature_materialization=features,
        requires_label_recomputation=labels,
        requires_model_refit=refit,
        requires_model_gated_sequential_replay=gated,
        requires_cost_recomputation=cost,
        requires_prop_resimulation=prop,
        requires_bootstrap_resimulation=bootstrap,
        can_change_setup_population=setup_pop,
        can_change_candidate_population=cand_pop,
        can_change_trade_order=trade_order,
        can_change_account_state=account,
        can_change_label_meaning=label_meaning,
        incompatible_dimensions=incompatible,
        identity_effect=identity_effect,  # type: ignore[arg-type]
    )


#: Seeded per the brief §7A.5 computation table (row-by-row test coverage).
EXPERIMENT_DIMENSION_REGISTRY: MappingProxyType[str, ExperimentDimensionSpec] = (
    MappingProxyType(
        {
            spec.dimension_id: spec
            for spec in (
                _spec(
                    "data_lineage.source_dataset",
                    DimensionType.DATA_LINEAGE,
                    "Source data / formula / bar policy",
                    "data_lineage",
                    "data, formula, or bar-policy change regenerates everything",
                    replay=True, features=True, labels=True, refit=True,
                    cost=True, prop=True, bootstrap=True,
                    setup_pop=True, cand_pop=True, trade_order=True,
                    account=True, label_meaning=True,
                ),
                _spec(
                    "market_universe.instrument",
                    DimensionType.MARKET_UNIVERSE,
                    "Instrument universe",
                    "market_universe",
                    "NQ-only in v1; a different universe regenerates everything",
                    replay=True, features=True, labels=True, refit=True,
                    cost=True, prop=True, bootstrap=True,
                    setup_pop=True, cand_pop=True, trade_order=True, account=True,
                    capability=DimensionCapabilityStatus.BLOCKED_OWNER_DECISION,
                    capability_reason="nq_only_v1 is the only registered universe",
                ),
                _spec(
                    "strategy_profile.parent_retest_timeout_1m_bars",
                    DimensionType.STRATEGY_PROFILE,
                    "Parent retest staleness timeout",
                    "strategy_profile.parent_retest_timeout_1m_bars",
                    "FSM occupancy change → full sequential replay",
                    replay=True, setup_pop=True, cand_pop=True, trade_order=True,
                ),
                _spec(
                    "strategy_profile.fsm_axis",
                    DimensionType.STRATEGY_PROFILE,
                    "Registered FSM configuration axis",
                    "strategy_profile.fsm_axis",
                    "any occupancy/order-changing FSM axis → full sequential replay",
                    replay=True, setup_pop=True, cand_pop=True, trade_order=True,
                ),
                _spec(
                    "strategy_profile.session_policy",
                    DimensionType.STRATEGY_PROFILE,
                    "Session policy (as strategy change)",
                    "strategy_profile.session_policy",
                    "session strategy change → new sequential profile",
                    replay=True, setup_pop=True, cand_pop=True, trade_order=True,
                ),
                _spec(
                    "strategy_profile.direction_policy",
                    DimensionType.STRATEGY_PROFILE,
                    "Direction policy (as strategy change)",
                    "strategy_profile.direction_policy",
                    "direction change → new sequential profile (shorts unratified)",
                    replay=True, setup_pop=True, cand_pop=True, trade_order=True,
                    capability=DimensionCapabilityStatus.BLOCKED_OWNER_DECISION,
                    capability_reason="canonical_short_profile_not_ratified",
                ),
                _spec(
                    "strategy_profile.entry_family",
                    DimensionType.STRATEGY_PROFILE,
                    "Entry family (as strategy change)",
                    "strategy_profile.entry_family",
                    "entry-family change → new sequential profile (retest blocked)",
                    replay=True, setup_pop=True, cand_pop=True, trade_order=True,
                    capability=DimensionCapabilityStatus.BLOCKED_OWNER_DECISION,
                    capability_reason="ifvg_retest execution is unratified",
                ),
                _spec(
                    "observation_cohort.descriptive_slice",
                    DimensionType.OBSERVATION_COHORT,
                    "Descriptive cohort slice",
                    "observation_cohort",
                    "analysis filter only — no recomputation of any kind",
                ),
                _spec(
                    "observation_cohort.specialized_model",
                    DimensionType.OBSERVATION_COHORT,
                    "Specialized-model cohort",
                    "observation_cohort",
                    "fold-local refit on the declared cohort",
                    refit=True,
                ),
                _spec(
                    "label_policy.candidate_static_r",
                    DimensionType.LABEL_POLICY,
                    "Candidate label policy",
                    "label_policy",
                    "label recomputation on unchanged replay evidence",
                    labels=True, refit=True, label_meaning=True,
                ),
                _spec(
                    "label_policy.lifetime_changing",
                    DimensionType.LABEL_POLICY,
                    "Trade-lifetime-changing label policy",
                    "label_policy",
                    "a label change that alters trade lifetime/occupancy → full replay",
                    replay=True, labels=True, refit=True,
                    setup_pop=True, cand_pop=True, trade_order=True,
                    label_meaning=True,
                ),
                _spec(
                    "feature_block.membership",
                    DimensionType.FEATURE_BLOCK,
                    "Feature-block membership",
                    "feature_bundle",
                    "feature blocks → materialize + refit; replay unchanged",
                    features=True, refit=True,
                ),
                _spec(
                    "model_protocol.algorithm",
                    DimensionType.MODEL_PROTOCOL,
                    "Model algorithm / hyperparameters",
                    "model_protocol",
                    "refit on identical rows/folds",
                    refit=True,
                ),
                _spec(
                    "decision_policy.execution_gate",
                    DimensionType.DECISION_POLICY,
                    "Execution-affecting decision policy",
                    "decision_policy",
                    "frozen model + model-gated sequential replay (S11 blocked until "
                    "RejectedCandidatePolicy is owner-ratified)",
                    gated=True, setup_pop=True, cand_pop=True, trade_order=True,
                    account=True,
                    capability=DimensionCapabilityStatus.BLOCKED_OWNER_DECISION,
                    capability_reason=(
                        "model-gated execution is unavailable until "
                        "RejectedCandidatePolicy is owner-ratified and sequential "
                        "golden tests pass"
                    ),
                ),
                _spec(
                    "execution_policy.fill_or_exit",
                    DimensionType.EXECUTION_POLICY,
                    "Lifetime-changing fill/exit/management",
                    "execution_policy",
                    "changes trade lifetime/occupancy → full replay",
                    replay=True, trade_order=True, account=True,
                    capability=DimensionCapabilityStatus.BLOCKED_MISSING_SOURCE,
                    capability_reason="fill/exit models are reducer-hardcoded in v1",
                ),
                _spec(
                    "cost_policy.round_turn_cost",
                    DimensionType.COST_POLICY,
                    "Cost policy",
                    "cost_policy",
                    "cost-only change with unchanged trade sequence → cost recompute",
                    cost=True,
                ),
                _spec(
                    "risk_policy.sizing",
                    DimensionType.RISK_POLICY,
                    "Account risk policy",
                    "risk_policy",
                    "account resimulation on the identical trade stream",
                    prop=True, bootstrap=True, account=True,
                ),
                _spec(
                    "prop_contract.firm_rules",
                    DimensionType.PROP_CONTRACT,
                    "Prop firm contract",
                    "prop_contract",
                    "prop resimulation on the identical trade stream",
                    prop=True, bootstrap=True, account=True,
                ),
                _spec(
                    "payout_policy.withdrawal_behavior",
                    DimensionType.PAYOUT_POLICY,
                    "Withdrawal policy (trader behavior)",
                    "payout_policy",
                    "payout resimulation; separate from the firm contract",
                    prop=True, bootstrap=True, account=True,
                ),
                _spec(
                    "portfolio_policy.legs",
                    DimensionType.PORTFOLIO_POLICY,
                    "Portfolio topology",
                    "portfolio_policy",
                    "portfolio resimulation on one common correlated path",
                    prop=True, bootstrap=True, account=True,
                ),
                _spec(
                    "validation_protocol.fold_protocol",
                    DimensionType.VALIDATION_PROTOCOL,
                    "Fold protocol",
                    "validation_protocol",
                    "model refit under the changed fold protocol",
                    refit=True,
                ),
                _spec(
                    "validation_protocol.bootstrap",
                    DimensionType.VALIDATION_PROTOCOL,
                    "Bootstrap protocol",
                    "validation_protocol",
                    "simulation rerun only",
                    bootstrap=True,
                ),
                _spec(
                    "stress_scenario.scenario_set",
                    DimensionType.STRESS_SCENARIO,
                    "Stress scenarios",
                    "stress_scenario",
                    "simulation rerun only",
                    bootstrap=True,
                ),
                _spec(
                    "engineering_protocol.runtime",
                    DimensionType.ENGINEERING_PROTOCOL,
                    "Runtime/storage operation details",
                    "engineering_protocol",
                    "workers, storage roots, page sizes, runtime estimates — "
                    "excluded from the semantic payload (annotation only)",
                    ratification=OwnerRatificationStatus.NOT_REQUIRED,
                    identity_effect="annotation_only",
                ),
            )
        }
    )
)


def resolve_dimension(dimension_id: str) -> ExperimentDimensionSpec:
    spec = EXPERIMENT_DIMENSION_REGISTRY.get(dimension_id)
    if spec is None:
        raise ValueError(f"unregistered experiment dimension {dimension_id!r}")
    return spec


def assert_dimension_usable(
    spec: ExperimentDimensionSpec, *, allow_experimental: bool = False
) -> None:
    if spec.capability_status is DimensionCapabilityStatus.AVAILABLE:
        return
    if (
        spec.capability_status is DimensionCapabilityStatus.EXPERIMENTAL
        and allow_experimental
    ):
        return
    raise DimensionBlockedError(
        f"dimension {spec.dimension_id!r} is {spec.capability_status.value}"
        + (f": {spec.capability_reason}" if spec.capability_reason else "")
    )
