"""Search charter, gate-threshold, and objective contracts (§3.1/§3.4).

Freeze means immutable save: any semantic change creates a new ``search_id``;
presentation changes remain catalog-only. ``validate_charter`` enforces
registered axes and values, a runnable baseline or generated-profile
capability, the child-count ceiling, the date policy, and computation-path-
scoped authorization — all fail-closed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from ..context_experiment_contracts import (
    PROFILE_CAPABILITY_REGISTRY,
    ProfileCapabilityStatus,
)
from ..development_access import FROZEN_WARMUP_DATES
from .authorization import (
    AuthorizationError,
    OwnerAuthorizationBundle,
    SyntheticAuthorizationMarker,
    derive_authorization_requirements,
    validate_owner_authorization,
)
from .axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
    assert_axes_authorized,
)
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)

__all__ = [
    "SearchMode",
    "ResolvedStrategyGateThresholds",
    "ResolvedPropGateThresholds",
    "ResolvedRobustnessGateThresholds",
    "PlannedVsRealizedEdge",
    "ObjectivePolicy",
    "DatePolicy",
    "SimulationProtocol",
    "CostPolicy",
    "SearchCharterPayload",
    "SearchCharterEnvelope",
    "CharterValidationError",
    "validate_charter",
    "save_charter",
    "MAX_CHILD_COUNT_CEILING",
]

MAX_CHILD_COUNT_CEILING = 256
_DEVELOPMENT_CUTOFF_DAY = "2026-06-10"
_PROTECTED_BUFFER_DAY = "2026-06-11"


class SearchMode(StrEnum):
    SINGLE_CONFIGURATION = "single_configuration"
    FSM_CONFIG_SEARCH = "fsm_config_search"
    PROP_BENCHMARK = "prop_benchmark"
    UNIVERSAL_PROP_SEARCH = "universal_prop_search"


class ResolvedStrategyGateThresholds(FrozenContract):
    min_executed_trades: int = 30
    min_independent_days: int = 20
    min_net_expectancy_r: float = 0.0
    min_profit_factor: float = 1.1
    max_drawdown_r: float = 15.0
    max_time_under_water_days: int = 45
    min_session_stability_score: float
    min_time_block_sign_consistency: float = 0.6
    max_top_day_pnl_share: float = 0.40
    max_top_setup_pnl_share: float = 0.25
    require_bootstrap_ci_excludes_zero: bool = False


class ResolvedPropGateThresholds(FrozenContract):
    minimum_first_payout_probability_60d: float
    maximum_breach_probability_90d: float
    minimum_expected_net_payout_90d: float | None
    minimum_p10_net_payout_90d: float | None
    maximum_p90_payout_drought_days: int | None
    minimum_three_payout_probability: float | None


class ResolvedRobustnessGateThresholds(FrozenContract):
    maximum_neighbor_expectancy_degradation_r: float | None
    minimum_plateau_width: int | None
    maximum_worst_firm_breach_probability_90d: float | None
    minimum_time_block_sign_consistency: float | None
    minimum_outer_fold_recurrence: float | None = None  # schema-reserved (exploratory lane: null)


class PlannedVsRealizedEdge(FrozenContract):
    planned_rrr: float
    realized_avg_winner_r: float
    realized_avg_loser_r: float
    realized_payoff_ratio: float
    gross_expectancy_r: float
    cost_r: float
    net_expectancy_r: float
    slippage_commission_drag_r: float


class ObjectivePolicy(FrozenContract):
    """No weight fields exist; a hidden weighted score is structurally impossible."""

    feasibility_gates: ResolvedStrategyGateThresholds
    prop_feasibility_gates: ResolvedPropGateThresholds
    robustness_gates: ResolvedRobustnessGateThresholds
    pareto_objectives: tuple[str, ...]
    lexicographic_tie_breaks: tuple[str, ...]


class DatePolicy(FrozenContract):
    replay_dates: tuple[str, ...]
    warmup_dates: tuple[str, ...]
    development_cutoff_utc: Literal["2026-06-10T21:00:00Z"] = "2026-06-10T21:00:00Z"
    access_policy_id: Literal[
        "development_explicit_dates_before_path_v2",
        "verification_fixed_allowlist_max5_v1",
    ]

    @model_validator(mode="after")
    def _dates_are_lawful(self):
        ordered = self.replay_dates
        if ordered != tuple(sorted(ordered)) or len(ordered) != len(set(ordered)):
            raise ValueError("replay dates must be unique and chronological")
        for day in (*ordered, *self.warmup_dates):
            if day >= _PROTECTED_BUFFER_DAY:
                raise ValueError(
                    "protected or sealed dates can never enter a search date policy"
                )
            if day < "2026-01-01":
                raise ValueError(
                    "dates before the permitted development window can never "
                    "enter a search date policy"
                )
        if self.access_policy_id == "verification_fixed_allowlist_max5_v1":
            if self.warmup_dates:
                raise ValueError("the verification fixture uses zero real warmup days")
            if len(ordered) > 5:
                raise ValueError(
                    "the verification allowlist admits at most five real trading days"
                )
        else:
            if tuple(day for day in ordered if day < "2026-01-13") != FROZEN_WARMUP_DATES:
                raise ValueError(
                    "development replay requires the frozen ten-date warmup prefix"
                )
        return self


class SimulationProtocol(FrozenContract):
    modes: tuple[
        Literal[
            "historical_closed_trade",
            "historical_1m_scenario",
            "historical_ordered_event_replay",
            "day_block_bootstrap",
            "stress",
        ],
        ...,
    ]
    bootstrap_n_paths: int = 10_000
    bootstrap_seed: int = 42
    block_policy: Literal["single_day", "contiguous_5day_block"] = "single_day"
    stress_scenario_ids: tuple[str, ...]
    trade_path_capability_policy_id: str
    clock_policy_id: str


class CostPolicy(FrozenContract):
    cost_points_round_turn: float = 0.514
    dollars_per_point: float = 20.0
    tick_size: float = 0.25


class SearchCharterPayload(FrozenContract):
    contract_name: Literal["ifvg_prop_robust_config_search_v1"] = (
        "ifvg_prop_robust_config_search_v1"
    )
    schema_version: int = 1
    search_mode: SearchMode
    baseline_profile_name: str
    baseline_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    axes: ImmutableMap[str, tuple[str, ...]]
    locked_invariants_registry_sha256: str = Field(pattern=SHA256_PATTERN)
    measured_only_fields: tuple[str, ...]
    blocked_capabilities: tuple[str, ...]
    authorized_firm_contract_ids: tuple[str, ...]
    authorized_risk_policy_ids: tuple[str, ...]
    authorized_withdrawal_policy_ids: tuple[str, ...]
    objective_policy: ObjectivePolicy
    date_policy: DatePolicy
    simulation_protocol: SimulationProtocol
    max_child_count: int = Field(gt=0, le=MAX_CHILD_COUNT_CEILING)
    search_algorithm: Literal["deterministic_exhaustive_v1"] = "deterministic_exhaustive_v1"
    seed: int
    cost_policy: CostPolicy
    strategy_core_commit: str
    quant_lab_commit: str
    source_artifact_ids: tuple[str, ...]
    owner_authorization: OwnerAuthorizationBundle | SyntheticAuthorizationMarker


class SearchCharterEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "search_id"

    search_id: str = Field(pattern=SHA256_PATTERN)
    payload: SearchCharterPayload


class CharterValidationError(ValueError):
    """The charter cannot freeze (fail-closed)."""


def _enumerated_child_count(axes: ImmutableMap[str, tuple[str, ...]]) -> int:
    count = 1
    for _axis, value_ids in axes.items():
        count *= max(len(value_ids), 1)
    return count


def validate_charter(
    payload: SearchCharterPayload,
    *,
    as_of_utc: str,
    superseded_artifact_ids: tuple[str, ...] = (),
    axis_registry=AXIS_VALUE_REGISTRY_V1,
    axis_specs=SEARCH_AXIS_REGISTRY_V1,
    generated_profile_capability=None,
) -> None:
    """Fail-closed charter validation (§3.1)."""

    from ..study.computation_path import ComputationPath  # noqa: PLC0415
    from .identities import GeneratedProfileCapability, is_generated_profile_id  # noqa: PLC0415

    capability = PROFILE_CAPABILITY_REGISTRY.get(payload.baseline_profile_name)
    if capability is not None:
        # A pre-existing named baseline: gated by the fixed registry.
        if capability.status is not ProfileCapabilityStatus.RUNNABLE:
            raise CharterValidationError(
                f"baseline is {capability.status.value}: {capability.reason}"
            )
    elif is_generated_profile_id(payload.baseline_profile_name):
        # A generated canonical profile: gated by its OWN capability contract
        # (§1.4) — it never fails merely for being absent from the fixed
        # registry, and it never anchors a charter without one.
        if not isinstance(generated_profile_capability, GeneratedProfileCapability):
            raise CharterValidationError(
                "a generated baseline profile requires its GeneratedProfileCapability"
            )
        if generated_profile_capability.status != "generated_runnable":
            raise CharterValidationError(
                "generated baseline is "
                f"{generated_profile_capability.status}: "
                f"{generated_profile_capability.reason}"
            )
    else:
        raise CharterValidationError(
            f"baseline {payload.baseline_profile_name!r} is neither a registered "
            "profile nor a generated canonical search profile"
        )

    synthetic = isinstance(payload.owner_authorization, SyntheticAuthorizationMarker)
    for axis_key, value_ids in sorted(payload.axes.items()):
        if not value_ids:
            raise CharterValidationError(f"axis {axis_key!r} registers no values")
        for value_id in value_ids:
            try:
                assert_axes_authorized(
                    {axis_key: value_id},
                    axis_registry,
                    axes=axis_specs,
                    require_ratified=not synthetic,
                )
            except PermissionError as error:
                raise CharterValidationError(str(error)) from error

    if _enumerated_child_count(payload.axes) > payload.max_child_count:
        raise CharterValidationError(
            "enumerated child count exceeds the charter's max_child_count"
        )

    prop_search = payload.search_mode in (
        SearchMode.PROP_BENCHMARK,
        SearchMode.UNIVERSAL_PROP_SEARCH,
    )
    if prop_search and not payload.authorized_firm_contract_ids:
        raise CharterValidationError(
            f"{payload.search_mode.value} requires at least one firm contract"
        )

    prop_modes = {
        "historical_closed_trade",
        "historical_1m_scenario",
        "day_block_bootstrap",
        "stress",
    }
    runs_prop = bool(payload.authorized_firm_contract_ids) and bool(
        prop_modes & set(payload.simulation_protocol.modes)
    )
    computation_path = ComputationPath(
        full_strategy_replay=any(
            axis_specs[axis].requires_full_sequential_replay
            for axis in payload.axes
            if axis in axis_specs
        )
        or payload.search_mode in (SearchMode.FSM_CONFIG_SEARCH, SearchMode.SINGLE_CONFIGURATION),
        feature_materialization=False,
        label_recomputation=False,
        model_refit=False,
        model_gated_sequential_replay=False,
        cost_recomputation=True,
        prop_resimulation=runs_prop,
        bootstrap_resimulation="day_block_bootstrap" in payload.simulation_protocol.modes
        and runs_prop,
        reuse_trade_stream_hash=False,
    )

    if synthetic:
        return

    run_scope = (
        "verification_5d"
        if payload.date_policy.access_policy_id == "verification_fixed_allowlist_max5_v1"
        else "full_authorized_development"
    )
    dimensions = tuple(
        f"strategy_profile.{axis}" for axis in sorted(payload.axes)
    ) + tuple(f"prop_contract.{firm}" for firm in payload.authorized_firm_contract_ids)
    requirement_set = derive_authorization_requirements(
        run_scope,
        dimensions,
        computation_path,
        (),
        payload.authorized_firm_contract_ids,
    )
    try:
        validate_owner_authorization(
            payload.owner_authorization,
            requirement_set,
            as_of_utc=as_of_utc,
            superseded_artifact_ids=superseded_artifact_ids,
        )
    except AuthorizationError as error:
        raise CharterValidationError(str(error)) from error


def _example_charter_payload() -> SearchCharterPayload:
    return SearchCharterPayload(
        search_mode=SearchMode.FSM_CONFIG_SEARCH,
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        baseline_section_config_hash="0" * 64,
        axes={
            "parent_retest_timeout_1m_bars": (
                "parent_retest_timeout_1m_bars.none",
                "parent_retest_timeout_1m_bars.480",
            )
        },
        locked_invariants_registry_sha256="1" * 64,
        measured_only_fields=(),
        blocked_capabilities=("parent_full_fill_invalidation",),
        authorized_firm_contract_ids=(),
        authorized_risk_policy_ids=(),
        authorized_withdrawal_policy_ids=(),
        objective_policy=ObjectivePolicy(
            feasibility_gates=ResolvedStrategyGateThresholds(
                min_session_stability_score=0.5
            ),
            prop_feasibility_gates=ResolvedPropGateThresholds(
                minimum_first_payout_probability_60d=0.5,
                maximum_breach_probability_90d=0.35,
                minimum_expected_net_payout_90d=None,
                minimum_p10_net_payout_90d=None,
                maximum_p90_payout_drought_days=None,
                minimum_three_payout_probability=None,
            ),
            robustness_gates=ResolvedRobustnessGateThresholds(
                maximum_neighbor_expectancy_degradation_r=None,
                minimum_plateau_width=None,
                maximum_worst_firm_breach_probability_90d=None,
                minimum_time_block_sign_consistency=None,
            ),
            pareto_objectives=("expected_net_payout_90d",),
            lexicographic_tie_breaks=("core_replay_id",),
        ),
        date_policy=DatePolicy(
            replay_dates=("2026-06-04", "2026-06-05"),
            warmup_dates=(),
            access_policy_id="verification_fixed_allowlist_max5_v1",
        ),
        simulation_protocol=SimulationProtocol(
            modes=("historical_closed_trade",),
            stress_scenario_ids=(),
            trade_path_capability_policy_id="path_capability_policy_v1",
            clock_policy_id="simulated_clock_v1",
        ),
        max_child_count=4,
        seed=7,
        cost_policy=CostPolicy(),
        strategy_core_commit="a" * 40,
        quant_lab_commit="b" * 40,
        source_artifact_ids=(),
        owner_authorization=SyntheticAuthorizationMarker(),
    )


def save_charter(root, envelope: SearchCharterEnvelope):
    """Namespace-confined charter publication (P0-4).

    A synthetic-authorization charter may never enter the research namespace:
    any root whose path contains a ``search`` store segment that is not
    ``search_test`` is refused for synthetic charters. Test/tmp namespaces
    (no ``search`` segment at all) and ``search_test`` roots are lawful.
    """

    from pathlib import Path  # noqa: PLC0415

    from .store import save_or_reuse_envelope  # noqa: PLC0415

    parts = Path(root).resolve().parts
    research_namespace = "search" in parts and "search_test" not in parts
    if research_namespace and isinstance(
        envelope.payload.owner_authorization, SyntheticAuthorizationMarker
    ):
        raise PermissionError(
            "synthetic-authorization charters are confined to test namespaces; "
            "the research store refuses them (P0-4)"
        )
    return save_or_reuse_envelope(root, "charters", envelope)


register_identity_pair(
    name="SearchCharter",
    envelope_cls=SearchCharterEnvelope,
    payload_cls=SearchCharterPayload,
    id_field="search_id",
    example_factory=_example_charter_payload,
)
