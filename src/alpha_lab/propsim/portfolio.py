"""Portfolio replay: copied accounts on ONE common correlated market path.

CS §5.4 + the §9 guardrail: ``run_portfolio_replay`` has NO per-account
resampling path — every account in every leg replays the SAME ordered day
blocks (the one common correlated path per draw). Divergence across copies
comes only from their policy sets (risk/withdrawal/replacement), never from
independent market draws.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
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
from alpha_lab.propsim.firm_contracts import PropFirmContractPayload
from alpha_lab.propsim.risk import PropRiskPolicyPayload
from alpha_lab.propsim.simulation import (
    PortfolioLeg,
    SimulationPathRecord,
    _sequence_hash,
    _synthetic_days,
    bootstrap_horizon_for,
)
from alpha_lab.propsim.withdrawal import WithdrawalPolicyPayload

__all__ = [
    "LegPolicies",
    "PortfolioReplayResult",
    "PortfolioBootstrapResult",
    "run_portfolio_replay",
    "run_portfolio_bootstrap",
]


@dataclass(frozen=True)
class LegPolicies:
    """The resolved policy objects one leg's account-policy-set id points at."""

    firm: PropFirmContractPayload
    risk_policy: PropRiskPolicyPayload
    withdrawal_policy: WithdrawalPolicyPayload
    policy_set: AccountPolicySetPayload


@dataclass(frozen=True)
class PortfolioReplayResult:
    per_account: tuple[tuple[str, AccountWalkResult], ...]  # (account key, result)
    common_path_instance_id: str


def run_portfolio_replay(
    legs: Sequence[PortfolioLeg],
    leg_policies: Mapping[str, LegPolicies],
    *,
    day_blocks: Sequence[tuple[date, Sequence[AccountTrade]]],
    path_instance_id: str,
    breach_mode: str = "realized_only",
    clock_policy: SimulatedClockPolicy = HISTORICAL_CLOCK_POLICY,
    bundle_event_ids: frozenset[str] | None = None,
) -> PortfolioReplayResult:
    """Replay every account of every leg over the ONE common path.

    There is deliberately no per-account day sampling parameter: the same
    ``day_blocks`` sequence drives every copy (copied accounts share one
    sampled sequence per path).
    """

    per_account: list[tuple[str, AccountWalkResult]] = []
    for leg in legs:
        policies = leg_policies.get(leg.leg_id)
        if policies is None:
            raise ValueError(f"leg {leg.leg_id!r} has no resolved policies")
        # the leg's account-policy-set ID must be the hash of the resolved
        # policy set it claims to pin (never a self-comparison)
        resolved_id = canonical_contract_sha256(policies.policy_set)
        if resolved_id != leg.account_policy_set_id:
            raise ValueError(
                f"leg {leg.leg_id!r}: resolved policy set hashes to "
                f"{resolved_id[:12]}… but the leg pins "
                f"{leg.account_policy_set_id[:12]}…"
            )
        for copy_index in range(leg.n_accounts):
            walk = AccountWalk(
                firm=policies.firm,
                firm_contract_id=policies.policy_set.firm_contract_id,
                risk_policy=policies.risk_policy,
                withdrawal_policy=policies.withdrawal_policy,
                policy_set=policies.policy_set,
                path_instance_id=path_instance_id,
                breach_mode=breach_mode,
                clock_policy=clock_policy,
                bundle_event_ids=bundle_event_ids,
                # copies must never collide on account/event identities
                account_namespace=f"{leg.leg_id}#{copy_index}",
            )
            for day, trades in day_blocks:
                if walk.play_day(day, trades) in (
                    "breached_out",
                    "expired",
                    "evaluation_passed",
                ):
                    break
            per_account.append((f"{leg.leg_id}#{copy_index}", walk.result()))
    return PortfolioReplayResult(
        per_account=tuple(per_account),
        common_path_instance_id=path_instance_id,
    )


@dataclass(frozen=True)
class PortfolioBootstrapResult:
    """Per-draw portfolio replays, every copy on the draw's ONE sampled path."""

    per_path: tuple[tuple[SimulationPathRecord, PortfolioReplayResult], ...]


def run_portfolio_bootstrap(
    legs: Sequence[PortfolioLeg],
    leg_policies: Mapping[str, LegPolicies],
    *,
    day_blocks: Sequence[tuple[date, Sequence[AccountTrade]]],
    bootstrap_protocol_id: str,
    seed: int,
    n_paths: int,
    breach_mode: str = "realized_only",
    bundle_event_ids: frozenset[str] | None = None,
) -> PortfolioBootstrapResult:
    """Day-block bootstrap over the whole portfolio (P0-21 shared sequence).

    ONE index sequence is drawn per path and every account of every leg
    replays exactly that sequence — there is structurally no per-account
    resampling surface. The draw sequence depends only on (seed, n_days,
    horizon), so copied accounts share `sampled_index_sequence_hash` by
    construction.
    """

    if not day_blocks:
        raise ValueError("portfolio bootstrap requires at least one trading day")
    horizon = bootstrap_horizon_for(bootstrap_protocol_id)
    rng = np.random.default_rng(seed)
    n_days = len(day_blocks)
    synthetic = _synthetic_days(horizon)
    per_path: list[tuple[SimulationPathRecord, PortfolioReplayResult]] = []
    for draw in range(n_paths):
        sampled = [int(rng.integers(0, n_days)) for _ in range(horizon)]
        path_instance_id = f"portfolio-path-{draw:05d}"
        blocks = tuple(
            (synthetic[step], day_blocks[index][1])
            for step, index in enumerate(sampled)
        )
        replay = run_portfolio_replay(
            legs,
            leg_policies,
            day_blocks=blocks,
            path_instance_id=path_instance_id,
            breach_mode=breach_mode,
            clock_policy=BOOTSTRAP_CLOCK_POLICY,
            bundle_event_ids=bundle_event_ids,
        )
        record = SimulationPathRecord(
            path_instance_id=path_instance_id,
            sampled_index_sequence_hash=_sequence_hash(sampled),
            draw_ordinal=draw,
        )
        per_path.append((record, replay))
    return PortfolioBootstrapResult(per_path=tuple(per_path))
