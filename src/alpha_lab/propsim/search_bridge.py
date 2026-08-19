"""The production ``prop_simulator`` for the search orchestrator seam.

PHASED R3 gate closure: the frontier consumes prop metrics through REAL
simulations of the child's executed-trade stream — adapter → trade-path
artifacts → bundle → capability report → account walk → reliability vector —
never through fabricated vectors. Every identity the simulation payload pins
is built (and revalidated by the runner) from the actual objects consumed.

The default chain runs ``historical_closed_trade`` on closed-trade artifacts:
honest base fidelity with no assumed chronology. A firm whose rules demand
market-price chronology is refused by the capability report, the runner
raises, and the orchestrator's per-child containment excludes that child
with a sanitized reason — fail-closed end to end.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from alpha_lab.propsim.account import AccountPolicySetPayload
from alpha_lab.propsim.adapters import (
    account_trades_from_executed_frame,
    gross_trade_stream_hash,
    group_account_trades_by_day,
)
from alpha_lab.propsim.contract_evidence import PropContractSupersession
from alpha_lab.propsim.firm_contracts import PropFirmContractPayload
from alpha_lab.propsim.prop_metrics import (
    PayoutReliabilityVector,
    build_payout_reliability_vector,
)
from alpha_lab.propsim.risk import PropRiskPolicyPayload
from alpha_lab.propsim.simulation import (
    AccountSimulationPayload,
    run_account_simulation,
)
from alpha_lab.propsim.trade_path import (
    build_closed_trade_artifact,
    build_trade_path_bundle,
    evaluate_path_capabilities,
)
from alpha_lab.propsim.withdrawal import WithdrawalPolicyPayload

__all__ = ["FirmSimulationSpec", "make_prop_simulator"]


@dataclass(frozen=True)
class FirmSimulationSpec:
    """One leg of the per-child prop evaluation: firm + trader policies."""

    label: str
    firm: PropFirmContractPayload
    risk_policy: PropRiskPolicyPayload
    withdrawal_policy: WithdrawalPolicyPayload
    replacement_policy: str = "none"
    max_replacements: int = 0
    clock_policy_id: str = "historical_calendar_clock_v1"

    def policy_set(self) -> AccountPolicySetPayload:
        return AccountPolicySetPayload(
            firm_contract_id=canonical_contract_sha256(self.firm),
            risk_policy_id=canonical_contract_sha256(self.risk_policy),
            withdrawal_policy_id=canonical_contract_sha256(self.withdrawal_policy),
            replacement_policy=self.replacement_policy,  # type: ignore[arg-type]
            max_replacements=self.max_replacements,
            clock_policy_id=self.clock_policy_id,
        )


def make_prop_simulator(
    specs: Sequence[FirmSimulationSpec],
    *,
    tick_size: float,
    costed_evaluation_id_for: Callable[[str], str],
    cost_points_round_turn: float = 0.0,
    seed: int = 7,
    supersessions: tuple[PropContractSupersession, ...] = (),
) -> Callable[..., dict[str, PayoutReliabilityVector]]:
    """Build the orchestrator-seam callable over the real simulation chain.

    The returned callable matches ``run_search(prop_simulator=...)``:
    ``simulator(outcome=..., result=...) -> {label: PayoutReliabilityVector}``.
    ``result`` must carry the child's rebuilt typed tables; a REUSED child
    (``result is None``) raises — wiring a table loader for reused children
    is the caller's explicit choice, never an implicit re-read.
    """

    def _simulator(*, outcome: Any, result: Any) -> dict[str, PayoutReliabilityVector]:
        if result is None:
            raise ValueError(
                "prop simulation needs the child's rebuilt tables; this child "
                "was REUSED without them — load its published tables first"
            )
        frame: pd.DataFrame = result.tables[RecordTable.EXECUTED_TRADE]
        account_trades = account_trades_from_executed_frame(
            frame, tick_size=tick_size, cost_points_round_turn=cost_points_round_turn
        )
        if not account_trades:
            return {}
        core_replay_id = outcome.core_replay_id
        stream_hash = gross_trade_stream_hash(frame)
        prices_by_trade = _entry_exit_prices(frame, tick_size=tick_size)
        artifacts = tuple(
            build_closed_trade_artifact(
                core_replay_id=core_replay_id,
                trade_id=trade.trade_id,
                entry_ts_utc=trade.entry_ts_utc,
                resolution_ts_utc=trade.resolution_ts_utc,
                entry_price=prices_by_trade[trade.trade_id][0],
                exit_price=prices_by_trade[trade.trade_id][1],
            )
            for trade in account_trades
        )
        ordered_trade_ids = tuple(trade.trade_id for trade in account_trades)
        bundle = build_trade_path_bundle(
            artifacts,
            gross_trade_stream_hash=stream_hash,
            ordered_trade_ids=ordered_trade_ids,
        )
        day_blocks = group_account_trades_by_day(account_trades)
        vectors: dict[str, PayoutReliabilityVector] = {}
        for spec in specs:
            policy_set = spec.policy_set()
            report = evaluate_path_capabilities(
                bundle, spec.firm.rule_path_requirements, artifacts=artifacts
            )
            payload = AccountSimulationPayload(
                core_replay_id=core_replay_id,
                gross_trade_stream_hash=stream_hash,
                costed_evaluation_id=costed_evaluation_id_for(core_replay_id),
                trade_path_bundle_id=bundle.trade_path_bundle_id,
                trade_path_bundle_manifest_sha256=bundle.manifest_payload_sha256,
                path_capability_report_id=canonical_contract_sha256(report),
                account_policy_set_id=canonical_contract_sha256(policy_set),
                simulation_mode="historical_closed_trade",
                intrabar_scenario_policy_id=None,
                bootstrap_protocol_id=None,
                stress_scenario_id=None,
                seed=seed,
                n_paths=1,
            )
            run = run_account_simulation(
                payload,
                firm=spec.firm,
                risk_policy=spec.risk_policy,
                withdrawal_policy=spec.withdrawal_policy,
                policy_set=policy_set,
                day_blocks=day_blocks,
                bundle=bundle,
                artifacts=artifacts,
                capability_report=report,
                supersessions=supersessions,
            )
            vectors[spec.label] = build_payout_reliability_vector(run.walk_results)
        return vectors

    return _simulator


def _entry_exit_prices(
    frame: pd.DataFrame, *, tick_size: float
) -> Mapping[str, tuple[float, float]]:
    """Entry/exit evidence prices per trade (points), from the tick columns."""

    prices: dict[str, tuple[float, float]] = {}
    for row in frame.to_dict("records"):
        entry = float(row["entry_ticks"]) * tick_size
        exit_price = entry + float(row["realized_ticks"]) * tick_size
        prices[str(row["trade_id"])] = (entry, exit_price)
    return prices
