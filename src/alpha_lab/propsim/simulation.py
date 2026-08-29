"""Complete account/portfolio simulation identities + deterministic runners.

CS §5.4: every result-changing policy rides the simulation identity — firm,
risk, withdrawal, replacement, clock, trade-path bundle (+ manifest +
capability report), mode, scenario/bootstrap protocol, seed, paths. No
result-changing constructor-only argument exists (the constructor-surface
audit test diffs the runner kwargs against the identity fields). Duplicate
sampled index sequences across bootstrap draws are LEGAL; every draw carries
a unique ``path_instance_id`` and a stored ``sampled_index_sequence_hash``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    register_identity_pair,
)
from alpha_lab.propsim.account import (
    AccountPolicySetPayload,
    AccountTrade,
    AccountWalk,
    AccountWalkResult,
)
from alpha_lab.propsim.calendar import (
    BOOTSTRAP_CLOCK_POLICY,
    HISTORICAL_CLOCK_POLICY,
    SimulatedClockPolicy,
)
from alpha_lab.propsim.contract_evidence import (
    PropContractSupersession,
    assert_contract_not_superseded,
)
from alpha_lab.propsim.event_detail import (
    EVENT_DETAIL_POLICY_NONE,
    EVENT_DETAIL_SCHEMA_VERSION_NONE,
    EVENT_DETAIL_STORAGE_NONE,
    EventDetailBudget,
    EventDetailPersistencePolicy,
    EventDetailStoragePolicy,
    assert_event_detail_policy_coherent,
)
from alpha_lab.propsim.firm_contracts import PropFirmContractPayload
from alpha_lab.propsim.risk import PropRiskPolicyPayload
from alpha_lab.propsim.stress import apply_stress_scenario
from alpha_lab.propsim.trade_path import (
    INTRABAR_SCENARIO_POLICIES,
    SCENARIO_FIDELITIES,
    PathCapabilityReport,
    TradePathArtifactEnvelope,
    TradePathBundleEnvelope,
    TradePathFidelity,
)
from alpha_lab.propsim.withdrawal import WithdrawalPolicyPayload

__all__ = [
    "SIMULATION_MODES",
    "SimulationPathRecord",
    "AccountSimulationPayload",
    "AccountSimulationEnvelope",
    "PortfolioLeg",
    "PortfolioPolicyPayload",
    "PortfolioPolicyEnvelope",
    "PortfolioSimulationPayload",
    "PortfolioSimulationEnvelope",
    "AccountSimulationRun",
    "run_account_simulation",
    "clock_policy_for_mode",
    "bootstrap_horizon_for",
    "UnsupportedSimulationModeError",
    "SimulationIdentityError",
]

SIMULATION_MODES: tuple[str, ...] = (
    "historical_closed_trade",
    "historical_1m_scenario",
    "historical_ordered_event_replay",
    "day_block_bootstrap",
    "stress",
)


class UnsupportedSimulationModeError(PermissionError):
    """The evidence cannot support the requested simulation mode (fail-closed)."""


class SimulationIdentityError(ValueError):
    """A resolved artifact does not hash to the identity that claims to pin it."""


#: The registered day-block bootstrap protocol FAMILY. The horizon (maximum
#: sampled trading days per path) is part of the protocol id itself, so the
#: path length always rides the simulation identity — no result-changing
#: constructor argument exists (P0-16/P0-21).
_BOOTSTRAP_PROTOCOL_PATTERN = re.compile(r"^day_block_bootstrap_h([1-9]\d{0,4})_v1$")


def bootstrap_horizon_for(bootstrap_protocol_id: str) -> int:
    """The sampled-path horizon a registered bootstrap protocol id encodes."""

    match = _BOOTSTRAP_PROTOCOL_PATTERN.match(bootstrap_protocol_id)
    if match is None:
        raise UnsupportedSimulationModeError(
            f"unregistered bootstrap protocol {bootstrap_protocol_id!r} "
            "(registered family: day_block_bootstrap_h<horizon>_v1)"
        )
    return int(match.group(1))


#: Scenario policy → the walk's breach-observation mode. Two registered
#: policies = two modes = (on order-sensitive trades) two different results.
_SCENARIO_BREACH_MODES: dict[str, str] = {
    "bar_adverse_extreme_first_v1": "unrealized_adverse_first",
    "bar_favorable_extreme_first_v1": "unrealized_favorable_first",
}


class SimulationPathRecord(FrozenContract):
    path_instance_id: str
    sampled_index_sequence_hash: str = Field(pattern=SHA256_PATTERN)
    draw_ordinal: int = Field(ge=0)


class AccountSimulationPayload(FrozenContract):
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    gross_trade_stream_hash: str = Field(pattern=SHA256_PATTERN)
    costed_evaluation_id: str = Field(pattern=SHA256_PATTERN)
    trade_path_bundle_id: str = Field(pattern=SHA256_PATTERN)
    trade_path_bundle_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    path_capability_report_id: str = Field(pattern=SHA256_PATTERN)
    account_policy_set_id: str = Field(pattern=SHA256_PATTERN)
    simulation_mode: str
    intrabar_scenario_policy_id: str | None
    bootstrap_protocol_id: str | None
    stress_scenario_id: str | None
    seed: int
    n_paths: int = Field(ge=1)
    #: R6.1 D15 — the prop-event detail representation is part of the
    #: identity: policy, storage representation, schema version, and the
    #: registered budget (``none_v0`` = the R3–R6 representation; changing
    #: any of them mints a new simulation id — persisted artifacts are never
    #: widened).
    event_detail_persistence_policy_id: EventDetailPersistencePolicy = EVENT_DETAIL_POLICY_NONE
    event_detail_storage_policy_id: EventDetailStoragePolicy = EVENT_DETAIL_STORAGE_NONE
    event_detail_schema_version: int = Field(default=EVENT_DETAIL_SCHEMA_VERSION_NONE, ge=0)
    event_detail_budget: EventDetailBudget | None = None

    @model_validator(mode="after")
    def _event_detail_coherent(self):
        assert_event_detail_policy_coherent(
            self.event_detail_persistence_policy_id,
            self.event_detail_storage_policy_id,
            self.event_detail_schema_version,
            self.event_detail_budget,
        )
        return self


class AccountSimulationEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "account_simulation_id"

    account_simulation_id: str = Field(pattern=SHA256_PATTERN)
    payload: AccountSimulationPayload


class PortfolioLeg(FrozenContract):
    leg_id: str
    account_policy_set_id: str = Field(pattern=SHA256_PATTERN)
    n_accounts: int = Field(ge=1)


class PortfolioPolicyPayload(FrozenContract):
    legs: tuple[PortfolioLeg, ...]
    copied_market_path_policy: Literal["one_common_correlated_path_v1"] = (
        "one_common_correlated_path_v1"
    )


class PortfolioPolicyEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "portfolio_policy_id"

    portfolio_policy_id: str = Field(pattern=SHA256_PATTERN)
    payload: PortfolioPolicyPayload


class PortfolioSimulationPayload(FrozenContract):
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    gross_trade_stream_hash: str = Field(pattern=SHA256_PATTERN)
    costed_evaluation_id: str = Field(pattern=SHA256_PATTERN)
    trade_path_bundle_id: str = Field(pattern=SHA256_PATTERN)
    trade_path_bundle_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    path_capability_report_id: str = Field(pattern=SHA256_PATTERN)
    portfolio_policy_id: str = Field(pattern=SHA256_PATTERN)
    resolved_legs: tuple[PortfolioLeg, ...]
    simulation_mode: str
    intrabar_scenario_policy_id: str | None
    bootstrap_protocol_id: str | None
    stress_scenario_id: str | None
    seed: int
    n_paths: int = Field(ge=1)
    #: R6.1 D15 — same identity-bearing representation fields as the account
    #: simulation (no portfolio detail writer is wired in R6.1; the fields
    #: keep the two identities structurally aligned).
    event_detail_persistence_policy_id: EventDetailPersistencePolicy = EVENT_DETAIL_POLICY_NONE
    event_detail_storage_policy_id: EventDetailStoragePolicy = EVENT_DETAIL_STORAGE_NONE
    event_detail_schema_version: int = Field(default=EVENT_DETAIL_SCHEMA_VERSION_NONE, ge=0)
    event_detail_budget: EventDetailBudget | None = None

    @model_validator(mode="after")
    def _event_detail_coherent(self):
        assert_event_detail_policy_coherent(
            self.event_detail_persistence_policy_id,
            self.event_detail_storage_policy_id,
            self.event_detail_schema_version,
            self.event_detail_budget,
        )
        return self


class PortfolioSimulationEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "portfolio_simulation_id"

    portfolio_simulation_id: str = Field(pattern=SHA256_PATTERN)
    payload: PortfolioSimulationPayload


@dataclass(frozen=True)
class AccountSimulationRun:
    envelope: AccountSimulationEnvelope
    walk_results: tuple[AccountWalkResult, ...]
    path_records: tuple[SimulationPathRecord, ...]


def _sequence_hash(indices: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps([int(index) for index in indices], separators=(",", ":")).encode()
    ).hexdigest()


def _path_instance_id(simulation_id: str, draw_ordinal: int) -> str:
    return f"path-{draw_ordinal:05d}-{simulation_id[:16]}"


_SYNTHETIC_DAY0 = date(2020, 1, 6)  # a Monday; bootstrap/stress days are synthetic


def _synthetic_days(count: int) -> list[date]:
    days: list[date] = []
    cursor = _SYNTHETIC_DAY0
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def clock_policy_for_mode(mode: str) -> SimulatedClockPolicy:
    """The simulated clock a mode walks under (historical calendar vs the
    synthetic bootstrap/stress trading-day clock) — the single source both
    the runner and the event-detail writer use."""

    return HISTORICAL_CLOCK_POLICY if mode.startswith("historical") else BOOTSTRAP_CLOCK_POLICY


def _breach_mode_for(mode: str, intrabar_scenario_policy_id: str | None) -> str:
    if mode == "historical_1m_scenario":
        # _require_mode_support has already validated the policy id
        return _SCENARIO_BREACH_MODES[intrabar_scenario_policy_id or ""]
    return "realized_only"


def _require_mode_support(
    mode: str,
    *,
    capability_report: PathCapabilityReport,
    bundle_fidelities: tuple[TradePathFidelity, ...],
    intrabar_scenario_policy_id: str | None,
    bootstrap_protocol_id: str | None,
) -> None:
    if mode not in SIMULATION_MODES:
        raise UnsupportedSimulationModeError(f"unknown simulation mode {mode!r}")
    if mode == "historical_ordered_event_replay":
        ordered = {
            TradePathFidelity.ORDERED_MBP1_EVENT_PATH,
            TradePathFidelity.ORDERED_FILL_EVENT_PATH,
        } & set(bundle_fidelities)
        if not ordered or set(bundle_fidelities) & SCENARIO_FIDELITIES:
            raise UnsupportedSimulationModeError(
                "historical_ordered_event_replay requires ACTUAL ordered event "
                "evidence — assumed scenario paths can never satisfy it"
            )
    if mode == "historical_1m_scenario":
        if intrabar_scenario_policy_id is None:
            raise UnsupportedSimulationModeError(
                "historical_1m_scenario requires an explicit registered "
                "intrabar scenario policy id (the result is a SCENARIO)"
            )
        if intrabar_scenario_policy_id not in INTRABAR_SCENARIO_POLICIES:
            raise UnsupportedSimulationModeError(
                f"unregistered intrabar scenario policy "
                f"{intrabar_scenario_policy_id!r}; registered: "
                f"{INTRABAR_SCENARIO_POLICIES}"
            )
        if TradePathFidelity.ASSUMED_1M_INTRABAR_PATH not in set(bundle_fidelities):
            raise UnsupportedSimulationModeError(
                "historical_1m_scenario requires assumed-intrabar scenario "
                "artifacts in the bundle"
            )
    if mode == "day_block_bootstrap":
        if bootstrap_protocol_id is None:
            raise UnsupportedSimulationModeError(
                "day_block_bootstrap requires a registered bootstrap protocol "
                "id — the sampled-path horizon rides the simulation identity"
            )
        bootstrap_horizon_for(bootstrap_protocol_id)  # validates the family
    unsupported = [
        rule_id
        for rule_id, state in capability_report.per_rule.items()
        if state == "unsupported"
    ]
    if unsupported:
        raise UnsupportedSimulationModeError(
            "the trade-path evidence cannot support these firm rules "
            f"(fail-closed): {sorted(unsupported)}"
        )


def _revalidate_identities(
    payload: AccountSimulationPayload,
    *,
    firm: PropFirmContractPayload,
    risk_policy: PropRiskPolicyPayload,
    withdrawal_policy: WithdrawalPolicyPayload,
    policy_set: AccountPolicySetPayload,
    day_blocks: Sequence[tuple[date, Sequence[AccountTrade]]],
    bundle: TradePathBundleEnvelope,
    artifacts: Sequence[TradePathArtifactEnvelope],
    capability_report: PathCapabilityReport,
    supersessions: tuple[PropContractSupersession, ...],
) -> None:
    """Every resolved artifact must hash to the identity that pins it (CS §0.3c)."""

    checks: tuple[tuple[str, str, str], ...] = (
        (
            "account_policy_set_id",
            canonical_contract_sha256(policy_set),
            payload.account_policy_set_id,
        ),
        (
            "policy_set.firm_contract_id",
            canonical_contract_sha256(firm),
            policy_set.firm_contract_id,
        ),
        (
            "policy_set.risk_policy_id",
            canonical_contract_sha256(risk_policy),
            policy_set.risk_policy_id,
        ),
        (
            "policy_set.withdrawal_policy_id",
            canonical_contract_sha256(withdrawal_policy),
            policy_set.withdrawal_policy_id,
        ),
        (
            "path_capability_report_id",
            canonical_contract_sha256(capability_report),
            payload.path_capability_report_id,
        ),
        (
            "trade_path_bundle_id",
            bundle.trade_path_bundle_id,
            payload.trade_path_bundle_id,
        ),
        (
            "trade_path_bundle_manifest_sha256",
            bundle.manifest_payload_sha256,
            payload.trade_path_bundle_manifest_sha256,
        ),
        ("core_replay_id", bundle.payload.core_replay_id, payload.core_replay_id),
        (
            "gross_trade_stream_hash",
            bundle.payload.gross_trade_stream_hash,
            payload.gross_trade_stream_hash,
        ),
        (
            "capability_report.trade_path_bundle_id",
            capability_report.trade_path_bundle_id,
            bundle.trade_path_bundle_id,
        ),
    )
    for field_name, observed, pinned in checks:
        if observed != pinned:
            raise SimulationIdentityError(
                f"{field_name}: the resolved artifact hashes to "
                f"{observed[:12]}… but the identity pins {pinned[:12]}…"
            )
    assert_contract_not_superseded(policy_set.firm_contract_id, supersessions)
    firm_rule_ids = {req.rule_id for req in firm.rule_path_requirements}
    covered = set(capability_report.per_rule.keys())
    uncovered = sorted(firm_rule_ids - covered)
    if uncovered:
        raise SimulationIdentityError(
            f"the capability report does not cover firm rules {uncovered} — "
            "an empty or partial report cannot authorize a simulation"
        )
    artifact_ids = {artifact.trade_path_artifact_id for artifact in artifacts}
    bundle_artifact_ids = set(bundle.payload.ordered_trade_path_artifact_ids)
    if artifact_ids != bundle_artifact_ids:
        raise SimulationIdentityError(
            "the supplied artifacts do not exactly match the bundle's "
            "ordered_trade_path_artifact_ids"
        )
    if payload.intrabar_scenario_policy_id is not None:
        mismatched = sorted(
            artifact.payload.trade_id
            for artifact in artifacts
            if artifact.payload.intrabar_scenario_policy_id is not None
            and artifact.payload.intrabar_scenario_policy_id
            != payload.intrabar_scenario_policy_id
        )
        if mismatched:
            raise SimulationIdentityError(
                "scenario artifacts were generated under a DIFFERENT intrabar "
                f"policy than the simulation identity claims: {mismatched}"
            )
    walked_trade_ids = {
        trade.trade_id for _day, trades in day_blocks for trade in trades
    }
    uncovered_trades = sorted(
        walked_trade_ids - set(bundle.payload.ordered_trade_ids)
    )
    if uncovered_trades:
        raise SimulationIdentityError(
            f"day blocks contain trades outside the bundle: {uncovered_trades}"
        )


def run_account_simulation(
    payload: AccountSimulationPayload,
    *,
    firm: PropFirmContractPayload,
    risk_policy: PropRiskPolicyPayload,
    withdrawal_policy: WithdrawalPolicyPayload,
    policy_set: AccountPolicySetPayload,
    day_blocks: Sequence[tuple[date, Sequence[AccountTrade]]],
    bundle: TradePathBundleEnvelope,
    artifacts: Sequence[TradePathArtifactEnvelope],
    capability_report: PathCapabilityReport,
    supersessions: tuple[PropContractSupersession, ...] = (),
) -> AccountSimulationRun:
    """Run one complete account simulation from its FROZEN identity payload.

    Every result-changing input is a field of ``payload`` or one of the
    identity-bearing policy objects it references; every supplied artifact is
    REVALIDATED against the identity that pins it before any walk runs
    (fail-closed — caller trust is never sufficient). Fidelities and path
    event ids are DERIVED from the bundle's artifacts, never caller-asserted.
    ``supersessions`` is a pure refusal registry: it can only block a retired
    firm contract id, never change a number.
    """

    _revalidate_identities(
        payload,
        firm=firm,
        risk_policy=risk_policy,
        withdrawal_policy=withdrawal_policy,
        policy_set=policy_set,
        day_blocks=day_blocks,
        bundle=bundle,
        artifacts=artifacts,
        capability_report=capability_report,
        supersessions=supersessions,
    )
    bundle_fidelities = tuple(artifact.payload.fidelity for artifact in artifacts)
    bundle_event_ids = frozenset(
        event.event_id
        for artifact in artifacts
        for event in (artifact.payload.ordered_events or ())
    )
    _require_mode_support(
        payload.simulation_mode,
        capability_report=capability_report,
        bundle_fidelities=bundle_fidelities,
        intrabar_scenario_policy_id=payload.intrabar_scenario_policy_id,
        bootstrap_protocol_id=payload.bootstrap_protocol_id,
    )
    envelope = AccountSimulationEnvelope(
        account_simulation_id=canonical_contract_sha256(payload),
        payload=payload,
    )
    simulation_id = envelope.account_simulation_id
    mode = payload.simulation_mode
    breach_mode = _breach_mode_for(mode, payload.intrabar_scenario_policy_id)
    clock_policy: SimulatedClockPolicy = clock_policy_for_mode(mode)

    def _walk(
        blocks: Sequence[tuple[date, Sequence[AccountTrade]]],
        path_instance_id: str,
    ) -> AccountWalkResult:
        walk = AccountWalk(
            firm=firm,
            firm_contract_id=policy_set.firm_contract_id,
            risk_policy=risk_policy,
            withdrawal_policy=withdrawal_policy,
            policy_set=policy_set,
            path_instance_id=path_instance_id,
            breach_mode=breach_mode,
            clock_policy=clock_policy,
            bundle_event_ids=bundle_event_ids,
        )
        for day, trades in blocks:
            if walk.play_day(day, trades) in (
                "breached_out",
                "expired",
                "evaluation_passed",
            ):
                break
        return walk.result()

    results: list[AccountWalkResult] = []
    records: list[SimulationPathRecord] = []
    if mode in ("historical_closed_trade", "historical_1m_scenario"):
        indices = tuple(range(len(day_blocks)))
        path_id = _path_instance_id(simulation_id, 0)
        results.append(_walk(day_blocks, path_id))
        records.append(
            SimulationPathRecord(
                path_instance_id=path_id,
                sampled_index_sequence_hash=_sequence_hash(indices),
                draw_ordinal=0,
            )
        )
    elif mode == "stress":
        if payload.stress_scenario_id is None:
            raise UnsupportedSimulationModeError(
                "stress mode requires a registered stress_scenario_id"
            )
        stressed = apply_stress_scenario(
            day_blocks, payload.stress_scenario_id, base_seed=payload.seed
        )
        synthetic = _synthetic_days(len(stressed))
        remapped = tuple(
            (synthetic[index], trades) for index, (_day, trades) in enumerate(stressed)
        )
        path_id = _path_instance_id(simulation_id, 0)
        results.append(_walk(remapped, path_id))
        records.append(
            SimulationPathRecord(
                path_instance_id=path_id,
                sampled_index_sequence_hash=_sequence_hash(range(len(stressed))),
                draw_ordinal=0,
            )
        )
    elif mode == "day_block_bootstrap":
        if not day_blocks:
            raise UnsupportedSimulationModeError(
                "day-block bootstrap requires at least one trading day"
            )
        rng = np.random.default_rng(payload.seed)
        n_days = len(day_blocks)
        # the horizon is encoded in the registered protocol id, so the path
        # length rides the simulation identity (P0-16/P0-21)
        horizon = bootstrap_horizon_for(payload.bootstrap_protocol_id or "")
        synthetic = _synthetic_days(horizon)
        for draw in range(payload.n_paths):
            walk = AccountWalk(
                firm=firm,
                firm_contract_id=policy_set.firm_contract_id,
                risk_policy=risk_policy,
                withdrawal_policy=withdrawal_policy,
                policy_set=policy_set,
                path_instance_id=_path_instance_id(simulation_id, draw),
                breach_mode=breach_mode,
                clock_policy=clock_policy,
                bundle_event_ids=bundle_event_ids,
            )
            sampled: list[int] = []
            verdict: str | None = None
            for step in range(horizon):
                index = int(rng.integers(0, n_days))
                sampled.append(index)
                _, trades = day_blocks[index]
                verdict = walk.play_day(synthetic[step], trades)
                if verdict in ("breached_out", "expired", "evaluation_passed"):
                    break
            results.append(walk.result())
            records.append(
                SimulationPathRecord(
                    path_instance_id=_path_instance_id(simulation_id, draw),
                    sampled_index_sequence_hash=_sequence_hash(sampled),
                    draw_ordinal=draw,
                )
            )
    else:  # historical_ordered_event_replay — reachable only with real ordered evidence
        raise UnsupportedSimulationModeError(
            "historical_ordered_event_replay execution lands with real ordered "
            "sources (R5B+); no synthetic path exists"
        )
    return AccountSimulationRun(
        envelope=envelope,
        walk_results=tuple(results),
        path_records=tuple(records),
    )


register_identity_pair(
    name="AccountSimulation",
    envelope_cls=AccountSimulationEnvelope,
    payload_cls=AccountSimulationPayload,
    id_field="account_simulation_id",
    example_factory=lambda: AccountSimulationPayload(
        core_replay_id="a" * 64,
        gross_trade_stream_hash="b" * 64,
        costed_evaluation_id="c" * 64,
        trade_path_bundle_id="d" * 64,
        trade_path_bundle_manifest_sha256="e" * 64,
        path_capability_report_id="f" * 64,
        account_policy_set_id="1" * 64,
        simulation_mode="historical_closed_trade",
        intrabar_scenario_policy_id=None,
        bootstrap_protocol_id=None,
        stress_scenario_id=None,
        seed=42,
        n_paths=1,
    ),
)
register_identity_pair(
    name="PortfolioPolicy",
    envelope_cls=PortfolioPolicyEnvelope,
    payload_cls=PortfolioPolicyPayload,
    id_field="portfolio_policy_id",
    example_factory=lambda: PortfolioPolicyPayload(
        legs=(
            PortfolioLeg(leg_id="leg-1", account_policy_set_id="1" * 64, n_accounts=2),
        ),
    ),
)
register_identity_pair(
    name="PortfolioSimulation",
    envelope_cls=PortfolioSimulationEnvelope,
    payload_cls=PortfolioSimulationPayload,
    id_field="portfolio_simulation_id",
    example_factory=lambda: PortfolioSimulationPayload(
        core_replay_id="a" * 64,
        gross_trade_stream_hash="b" * 64,
        costed_evaluation_id="c" * 64,
        trade_path_bundle_id="d" * 64,
        trade_path_bundle_manifest_sha256="e" * 64,
        path_capability_report_id="f" * 64,
        portfolio_policy_id="2" * 64,
        resolved_legs=(
            PortfolioLeg(leg_id="leg-1", account_policy_set_id="1" * 64, n_accounts=2),
        ),
        simulation_mode="historical_closed_trade",
        intrabar_scenario_policy_id=None,
        bootstrap_protocol_id=None,
        stress_scenario_id=None,
        seed=42,
        n_paths=1,
    ),
)
