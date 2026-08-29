"""Canonical study-cell identity (DELTA_TAXONOMY.md §2; brief §7A.1–7A.2).

The hashed :class:`StudyCellSemanticPayload` carries exactly the 16 semantic
dimensions; runtime/storage/display details live in the never-hashed
:class:`StudyCellAnnotation`. Strategy-only cells carry typed ``none_*``
identities and no context-formula lineage (V3 P0-6 / Amendment P1-G).
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator
from strategy_core.strategies.ifvg_smc.section import (
    IFVG_STRATEGY_VERSION,
    default_ifvg_smc_section,
    ifvg_profile_hash,
)

from ..config import (
    FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
    FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
)
from ..context_experiment_contracts import PROFILE_CAPABILITY_REGISTRY
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    is_generated_profile_id,
    register_identity_pair,
)
from .cohort import BASELINE_COHORT

__all__ = [
    "LEGACY_REPLAY_PROVENANCE",
    "NONE_CONTEXT_ARTIFACT",
    "NONE_FEATURE_BUNDLE",
    "NONE_LABEL_VIEW",
    "NONE_FOLD_SET",
    "NONE_MODEL_PROTOCOL",
    "NONE_REGIME_MODEL",
    "DataLineagePayload",
    "MarketUniverseIdentity",
    "StrategyProfileIdentity",
    "ObservationCohortIdentity",
    "LabelPolicyIdentity",
    "FeatureBundleIdentity",
    "ModelProtocolIdentity",
    "DecisionPolicyIdentity",
    "ExecutionPolicyIdentity",
    "CostPolicyIdentity",
    "RiskPolicyIdentity",
    "PropContractIdentity",
    "PayoutPolicyIdentity",
    "PortfolioPolicyIdentity",
    "ValidationProtocolIdentity",
    "StressScenarioIdentity",
    "StudyCellSemanticPayload",
    "StudyCellIdentity",
    "EngineeringProtocolAnnotation",
    "StudyCellAnnotation",
    "BASELINE_STUDY_CELL",
    "study_cell_from_context_run",
]

#: Opaque legacy replay provenance (V3 P0-3): permissible ONLY as historical
#: replay-input identity. The feature layer, bundle materialization, dashboard,
#: live-source contracts, and model features can never query it (guard tests).
LEGACY_REPLAY_PROVENANCE = "legacy_verified_replay_source"

NONE_CONTEXT_ARTIFACT = "none_context_artifact_v1"
NONE_FEATURE_BUNDLE = "none_feature_bundle_v1"
NONE_LABEL_VIEW = "none_label_view_v1"
NONE_FOLD_SET = "none_fold_set_v1"
NONE_MODEL_PROTOCOL = "none_model_protocol_v1"
NONE_REGIME_MODEL = "none_regime_model_v1"

_HEX_OR_NONE = r"^([0-9a-f]{64}|none_[a-z0-9_]+_v[0-9]+)$"
_HEX_OR_POLICY_TOKEN = r"^([0-9a-f]{64}|[a-z][a-z0-9_]*_v[0-9]+)$"
_CORE_REPLAY_OR_LEGACY = r"^([0-9a-f]{64}|legacy_verified_replay_source)$"

#: Registered vocabularies the cell validator fails closed against. Later
#: releases extend these when their registries land (R3 stress scenarios, R5
#: decision policies, R6 regime protocols) — extension, never bypass.
REGISTERED_MODEL_PROTOCOL_KEYS: tuple[str, ...] = (
    "reference_prevalence_v1",
    "ifvg_context_logistic_l2_v1",
    "ifvg_context_catboost_binary_v1",
    "ifvg_context_catboost_bundle_v1",
    "ifvg_context_gam_v1",
)
REGISTERED_CALIBRATION_POLICY_IDS: tuple[str, ...] = (
    "raw_probability_diagnostics_v1",
    "platt_sigmoid_train_fold_v1",
    "isotonic_train_fold_v1",
)
REGISTERED_DECISION_POLICY_KEYS: tuple[str, ...] = (
    "none_diagnostic_only_v1",
    "fixed_probability_threshold_v1",
    "expected_r_threshold_v1",
    "abstention_band_v1",
    "top_n_per_day_v1",
    "confidence_margin_v1",
    "regime_conditioned_threshold_v1",
)
REGISTERED_STRESS_SCENARIO_IDS: tuple[str, ...] = ("none_historical_v1",)


def _registered_bundle_keys() -> frozenset[str]:
    from ..features.feature_bundles import (  # noqa: PLC0415
        FEATURE_BUNDLE_REGISTRY,
        FROZEN_TIER_BUNDLES,
    )

    return frozenset(FEATURE_BUNDLE_REGISTRY) | frozenset(FROZEN_TIER_BUNDLES)


def _active_dimension_ids(payload: StudyCellSemanticPayload) -> tuple[str, ...]:
    """Map non-baseline selections onto their fail-closed registry dimensions."""

    active: list[str] = []
    if payload.decision_policy.decision_policy_key != "none_diagnostic_only_v1":
        active.append("decision_policy.execution_gate")
    if payload.model_protocol.resolved_model_protocol_id != NONE_MODEL_PROTOCOL:
        active.append("model_protocol.algorithm")
    if payload.risk_policy.risk_policy_id != "none_v1":
        active.append("risk_policy.sizing")
    if payload.prop_contract.prop_contract_id != "none_personal_account_v1":
        active.append("prop_contract.firm_rules")
    if payload.payout_policy.payout_policy_id != "none_v1":
        active.append("payout_policy.withdrawal_behavior")
    if payload.portfolio_policy.portfolio_policy_id != "single_account_v1":
        active.append("portfolio_policy.legs")
    if payload.stress_scenario.stress_scenario_id != "none_historical_v1":
        active.append("stress_scenario.scenario_set")
    return tuple(active)


class DataLineagePayload(FrozenContract):
    """Concrete strategy-only-capable data lineage (V3 P0-6)."""

    core_replay_id: str = Field(pattern=_CORE_REPLAY_OR_LEGACY)
    v2_artifact_id: str = Field(pattern=SHA256_PATTERN)
    v2_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    data_source_id: Literal["databento_nq_v1"] = "databento_nq_v1"
    dataset_schema_version: Literal[2] = 2
    bar_construction_policy_id: Literal["time_1m_v1"] = "time_1m_v1"
    order_flow_depth_policy: Literal["mbp1_only_v1"] = "mbp1_only_v1"
    # typed optionals — None for strategy-only cells (Amendment P1-G)
    context_artifact_id: str | None = None
    context_manifest_hash: str | None = Field(default=None, pattern=SHA256_PATTERN)
    context_formula_id: str | None = None
    feature_view_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    label_view_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    fold_set_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    model_fit_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    regime_fit_id: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _coherent(self):
        if self.context_formula_id is not None and self.context_artifact_id is None:
            raise ValueError(
                "a context formula identity requires an actual context artifact — "
                "strategy-only cells never carry ifvg_context_formula_v2"
            )
        if self.context_artifact_id is not None and self.context_manifest_hash is None:
            raise ValueError("a context artifact reference requires its manifest hash")
        if self.model_fit_id is not None and (
            self.label_view_id is None or self.fold_set_id is None
        ):
            raise ValueError("a model fit requires exact label and fold references")
        return self

    @property
    def artifact_pair_hash(self) -> str | None:
        """Derived convenience for v2+v3 studies only (never for strategy-only)."""

        if self.context_artifact_id is None:
            return None
        return canonical_contract_sha256(
            {
                "v2": {"artifact_id": self.v2_artifact_id, "manifest": self.v2_manifest_hash},
                "v3": {
                    "artifact_id": self.context_artifact_id,
                    "manifest": self.context_manifest_hash,
                },
            }
        )


class MarketUniverseIdentity(FrozenContract):
    instrument_universe_id: Literal["nq_only_v1"] = "nq_only_v1"
    contract_roll_policy_id: str


class StrategyProfileIdentity(FrozenContract):
    strategy_profile_id: str
    strategy_version: str
    profile_hash: str = Field(pattern=SHA256_PATTERN)
    section_config_hash: str = Field(pattern=SHA256_PATTERN)
    entry_family: str
    direction_policy: str
    session_policy: str
    concurrency_policy: Literal["one_active_setup_one_active_trade_v1"] = (
        "one_active_setup_one_active_trade_v1"
    )

    @model_validator(mode="after")
    def _name_is_gated(self):
        name = self.strategy_profile_id
        if name in PROFILE_CAPABILITY_REGISTRY or is_generated_profile_id(name):
            return self
        raise ValueError(
            f"profile {name!r} is neither a fixed registered baseline nor a "
            "generated canonical search profile (P0-D)"
        )


class ObservationCohortIdentity(FrozenContract):
    cohort_id: str = Field(pattern=SHA256_PATTERN)
    observation_filters_hash: str = Field(pattern=SHA256_PATTERN)
    warmup_policy_id: str
    date_policy_id: str


class LabelPolicyIdentity(FrozenContract):
    label_policy_id: str
    label_family: str
    label_derivation_id: str = Field(pattern=_HEX_OR_NONE)


class FeatureBundleIdentity(FrozenContract):
    feature_bundle_key: str
    resolved_feature_bundle_id: str = Field(pattern=_HEX_OR_NONE)
    block_registry_hash: str = Field(pattern=SHA256_PATTERN)


class ModelProtocolIdentity(FrozenContract):
    model_protocol_key: str
    resolved_model_protocol_id: str = Field(pattern=_HEX_OR_NONE)
    calibration_policy_id: str
    resolved_regime_protocol_id: str | None = None


class DecisionPolicyIdentity(FrozenContract):
    decision_policy_key: str
    resolved_decision_policy_id: str = Field(pattern=_HEX_OR_NONE)


class ExecutionPolicyIdentity(FrozenContract):
    execution_policy_id: Literal["confirmation_close_next1m_stop_first_v1"] = (
        "confirmation_close_next1m_stop_first_v1"
    )
    anchor_policy: str
    resolver_policy: Literal["next_1m_bar_stop_first_v1"] = "next_1m_bar_stop_first_v1"


class CostPolicyIdentity(FrozenContract):
    cost_policy_id: str
    cost_points: float


class RiskPolicyIdentity(FrozenContract):
    risk_policy_id: str = Field(pattern=_HEX_OR_POLICY_TOKEN)


class PropContractIdentity(FrozenContract):
    prop_contract_id: str = Field(pattern=_HEX_OR_POLICY_TOKEN)


class PayoutPolicyIdentity(FrozenContract):
    payout_policy_id: str = Field(pattern=_HEX_OR_POLICY_TOKEN)


class PortfolioPolicyIdentity(FrozenContract):
    portfolio_policy_id: str = Field(pattern=_HEX_OR_POLICY_TOKEN)


class ValidationProtocolIdentity(FrozenContract):
    fold_protocol_id: Literal["ifvg_context_walkforward_40_5_5_2_v1"] = (
        "ifvg_context_walkforward_40_5_5_2_v1"
    )
    bootstrap_protocol_id: Literal["setup_and_day_block_bootstrap_10000_seed7_v1"] = (
        "setup_and_day_block_bootstrap_10000_seed7_v1"
    )
    fold_set_hash: str = Field(pattern=_HEX_OR_NONE)


class StressScenarioIdentity(FrozenContract):
    stress_scenario_id: str = Field(pattern=r"^([a-z0-9_]+_v[0-9]+)$")


class StudyCellSemanticPayload(FrozenContract):
    """Exactly the 16 semantic dimensions (Amendment P0-B) — nothing else."""

    data_lineage: DataLineagePayload
    market_universe: MarketUniverseIdentity
    strategy_profile: StrategyProfileIdentity
    observation_cohort: ObservationCohortIdentity
    label_policy: LabelPolicyIdentity
    feature_bundle: FeatureBundleIdentity
    model_protocol: ModelProtocolIdentity
    decision_policy: DecisionPolicyIdentity
    execution_policy: ExecutionPolicyIdentity
    cost_policy: CostPolicyIdentity
    risk_policy: RiskPolicyIdentity
    prop_contract: PropContractIdentity
    payout_policy: PayoutPolicyIdentity
    portfolio_policy: PortfolioPolicyIdentity
    validation_protocol: ValidationProtocolIdentity
    stress_scenario: StressScenarioIdentity

    @model_validator(mode="after")
    def _all_dimensions_registered_and_usable(self):
        # 1. Every registry-bound axis value must be registered (fail closed on
        #    unknown values — §7A.3 closing requirement).
        problems: list[str] = []
        for label, value, vocabulary in (
            (
                "feature_bundle.feature_bundle_key",
                self.feature_bundle.feature_bundle_key,
                _registered_bundle_keys(),
            ),
            (
                "model_protocol.model_protocol_key",
                self.model_protocol.model_protocol_key,
                REGISTERED_MODEL_PROTOCOL_KEYS,
            ),
            (
                "model_protocol.calibration_policy_id",
                self.model_protocol.calibration_policy_id,
                REGISTERED_CALIBRATION_POLICY_IDS,
            ),
            (
                "decision_policy.decision_policy_key",
                self.decision_policy.decision_policy_key,
                REGISTERED_DECISION_POLICY_KEYS,
            ),
            (
                "stress_scenario.stress_scenario_id",
                self.stress_scenario.stress_scenario_id,
                REGISTERED_STRESS_SCENARIO_IDS,
            ),
            (
                "observation_cohort.warmup_policy_id",
                self.observation_cohort.warmup_policy_id,
                ("exclude_warmup_v1", "include_warmup_v1"),
            ),
            (
                "observation_cohort.date_policy_id",
                self.observation_cohort.date_policy_id,
                (
                    "development_explicit_dates_before_path_v2",
                    "verification_fixed_allowlist_max5_v1",
                ),
            ),
        ):
            if value not in vocabulary:
                problems.append(f"{label}={value!r} is not a registered value")
        if problems:
            raise ValueError(
                "study cell refuses unregistered dimension values: "
                + "; ".join(problems)
            )
        # 2. Every ACTIVE dimension must be usable per the fail-closed registry
        #    (a blocked dimension — e.g. an execution-gating decision policy —
        #    can never enter a study), plus pairwise incompatibility.
        from .dimension_contracts import (  # noqa: PLC0415
            EXPERIMENT_DIMENSION_REGISTRY,
            assert_dimension_usable,
        )

        active = _active_dimension_ids(self)
        for dimension_id in active:
            spec = EXPERIMENT_DIMENSION_REGISTRY.get(dimension_id)
            if spec is None:
                raise ValueError(f"unregistered experiment dimension {dimension_id!r}")
            assert_dimension_usable(spec)
        for dimension_id in active:
            spec = EXPERIMENT_DIMENSION_REGISTRY[dimension_id]
            clash = set(spec.incompatible_dimensions) & set(active)
            if clash:
                raise ValueError(
                    f"dimension {dimension_id!r} is incompatible with {sorted(clash)}"
                )
        # 3. Coherence: a model protocol requires labels + folds; a
        #    strategy-only cell is never blocked by a missing v3 pair (P1-G).
        model_none = self.model_protocol.resolved_model_protocol_id == NONE_MODEL_PROTOCOL
        if not model_none:
            if self.label_policy.label_derivation_id == NONE_LABEL_VIEW:
                raise ValueError("a model study requires an exact label derivation")
            if self.validation_protocol.fold_set_hash == NONE_FOLD_SET:
                raise ValueError("a model study requires an exact fold set")
        if (
            self.feature_bundle.resolved_feature_bundle_id != NONE_FEATURE_BUNDLE
            and self.data_lineage.feature_view_id is None
            and self.data_lineage.context_artifact_id is None
        ):
            raise ValueError(
                "a resolved feature bundle requires its exact source evidence refs"
            )
        return self


class StudyCellIdentity(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "cell_id"

    cell_id: str = Field(pattern=SHA256_PATTERN)
    payload: StudyCellSemanticPayload


class EngineeringProtocolAnnotation(FrozenContract):
    """Runtime/storage operation details ONLY — never hashed (P0-B)."""

    model_config = FrozenContract.model_config | {"extra": "allow"}

    worker_count: int | None = None
    storage_root: str | None = None
    page_size: int | None = None
    runtime_estimate_seconds: float | None = None


class StudyCellAnnotation(FrozenContract):
    cell_id: str = Field(pattern=SHA256_PATTERN)
    engineering_protocol: EngineeringProtocolAnnotation
    display_metadata: dict[str, Any] = Field(default_factory=dict)


_BASELINE_SECTION = default_ifvg_smc_section()
_BASELINE_PROFILE_HASH = ifvg_profile_hash(_BASELINE_SECTION)


def _baseline_cell_payload() -> StudyCellSemanticPayload:
    return StudyCellSemanticPayload(
        data_lineage=DataLineagePayload(
            core_replay_id=LEGACY_REPLAY_PROVENANCE,
            v2_artifact_id=FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
            v2_manifest_hash=FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
        ),
        market_universe=MarketUniverseIdentity(
            contract_roll_policy_id="databento_continuous_front_v1"
        ),
        strategy_profile=StrategyProfileIdentity(
            strategy_profile_id="ifvg_v2_doc_default_fresh_static_1r",
            strategy_version=IFVG_STRATEGY_VERSION,
            profile_hash=_BASELINE_PROFILE_HASH,
            section_config_hash=_BASELINE_PROFILE_HASH,
            entry_family="fresh_fvg_continuation",
            direction_policy="long_only_v1",
            session_policy="doc_sessions_asia_london_ny_v1",
        ),
        observation_cohort=ObservationCohortIdentity(
            cohort_id=BASELINE_COHORT.cohort_id,
            observation_filters_hash=canonical_contract_sha256({}),
            warmup_policy_id="exclude_warmup_v1",
            date_policy_id="development_explicit_dates_before_path_v2",
        ),
        label_policy=LabelPolicyIdentity(
            label_policy_id="candidate_static_r1_v2",
            label_family="candidate_static_r_long_form_v2",
            label_derivation_id=NONE_LABEL_VIEW,
        ),
        # DT §2 names the baseline bundle by its core-block name; the typed
        # bundle registry (DT §7) is the binding key surface, so the lawful
        # baseline bundle key is B0_CORE (= IFVG_CORE_BASELINE_V1 + session).
        # Recorded as DEV-R1-8; the resolved id stays the strategy-only null.
        feature_bundle=FeatureBundleIdentity(
            feature_bundle_key="B0_CORE",
            resolved_feature_bundle_id=NONE_FEATURE_BUNDLE,
            block_registry_hash=canonical_contract_sha256("feature_block_registry_v1"),
        ),
        model_protocol=ModelProtocolIdentity(
            model_protocol_key="reference_prevalence_v1",
            resolved_model_protocol_id=NONE_MODEL_PROTOCOL,
            calibration_policy_id="raw_probability_diagnostics_v1",
        ),
        decision_policy=DecisionPolicyIdentity(
            decision_policy_key="none_diagnostic_only_v1",
            resolved_decision_policy_id="none_decision_policy_v1",
        ),
        execution_policy=ExecutionPolicyIdentity(
            anchor_policy="trading_day_18et_elapsed_v1",
        ),
        cost_policy=CostPolicyIdentity(
            cost_policy_id="gross_zero_cost_v1", cost_points=0.0
        ),
        risk_policy=RiskPolicyIdentity(risk_policy_id="none_v1"),
        prop_contract=PropContractIdentity(prop_contract_id="none_personal_account_v1"),
        payout_policy=PayoutPolicyIdentity(payout_policy_id="none_v1"),
        portfolio_policy=PortfolioPolicyIdentity(portfolio_policy_id="single_account_v1"),
        validation_protocol=ValidationProtocolIdentity(fold_set_hash=NONE_FOLD_SET),
        stress_scenario=StressScenarioIdentity(stress_scenario_id="none_historical_v1"),
    )


#: §7A.2: one distinct baseline per dimension, over the accepted final-review
#: v2 dataset (legacy replay provenance — pre-lane artifact).
BASELINE_STUDY_CELL: StudyCellIdentity = StudyCellIdentity.from_payload(
    _baseline_cell_payload()
)


def study_cell_from_context_run(
    *,
    config,
    resolved_feature_bundle_id: str,
    feature_bundle_key: str,
    block_registry_hash: str,
    view_id: str | None = None,
    label_derivation_id: str | None = None,
    fold_set_hash: str | None = None,
    resolved_model_protocol_id: str | None = None,
) -> StudyCellIdentity:
    """Read-only adapter: map a stored M0–M3 context run into a study cell.

    Legacy runs participate in comparisons without re-execution; the frozen
    tier registry is untouched (tiers map onto frozen bundles, DT §5.4).
    """

    pair = config.dataset.artifact_pair
    base = _baseline_cell_payload()
    payload = base.model_copy(
        update={
            "data_lineage": DataLineagePayload(
                core_replay_id=LEGACY_REPLAY_PROVENANCE,
                v2_artifact_id=pair.v2.artifact_id,
                v2_manifest_hash=pair.v2.manifest_payload_sha256,
                context_artifact_id=pair.v3.artifact_id,
                context_manifest_hash=pair.v3.manifest_payload_sha256,
                context_formula_id=pair.v3.feature_formula_version,
                feature_view_id=view_id,
                label_view_id=label_derivation_id,
                fold_set_id=fold_set_hash,
            ),
            "observation_cohort": base.observation_cohort.model_copy(
                update={
                    "observation_filters_hash": canonical_contract_sha256(
                        {
                            key: list(values)
                            for key, values in sorted(config.observation_filters.items())
                        }
                    )
                }
            ),
            "label_policy": LabelPolicyIdentity(
                label_policy_id=config.label.identity,
                label_family=config.label.label_family,
                label_derivation_id=label_derivation_id or NONE_LABEL_VIEW,
            ),
            "feature_bundle": FeatureBundleIdentity(
                feature_bundle_key=feature_bundle_key,
                resolved_feature_bundle_id=resolved_feature_bundle_id,
                block_registry_hash=block_registry_hash,
            ),
            "model_protocol": ModelProtocolIdentity(
                model_protocol_key=config.model_protocol_id,
                resolved_model_protocol_id=resolved_model_protocol_id
                or NONE_MODEL_PROTOCOL,
                calibration_policy_id="raw_probability_diagnostics_v1",
            ),
            "validation_protocol": ValidationProtocolIdentity(
                fold_set_hash=fold_set_hash or NONE_FOLD_SET
            ),
        }
    )
    return StudyCellIdentity.from_payload(payload)


register_identity_pair(
    name="StudyCell",
    envelope_cls=StudyCellIdentity,
    payload_cls=StudyCellSemanticPayload,
    id_field="cell_id",
    example_factory=_baseline_cell_payload,
)
