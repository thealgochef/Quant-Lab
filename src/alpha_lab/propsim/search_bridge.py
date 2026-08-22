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

R5 closes the two deferred seams (DEV-R3-11, DEV-R4-17) **additively** —
every default reproduces the R3 behavior byte-for-byte:

* **Scenario/bootstrap/stress bridging:** ``simulation_modes`` selects which
  registered modes run per firm spec. ``historical_1m_scenario`` builds
  ASSUMED intrabar scenario artifacts from caller-supplied 1m bar
  observations under a registered intrabar policy (two policies = two
  scenario identities); ``day_block_bootstrap`` rides the
  ``day_block_bootstrap_h<N>_v1`` protocol family with ``n_paths`` draws;
  ``stress`` runs one simulation per registered scenario id.
  ``historical_ordered_event_replay`` is REFUSED at this seam — it is
  reserved for actual ordered event streams, and no ordered source is wired
  here.
* **Production persistence:** with ``store_root`` set, every real
  ``AccountPolicySetEnvelope`` and ``AccountSimulationEnvelope`` (plus the
  walk-summary / historical account-event sidecars the trader UI reads) is
  published to the immutable stores — the S12/S13 production writers the R4
  UI wiring anticipated.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from alpha_lab.propsim.account import AccountPolicySetEnvelope, AccountPolicySetPayload
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
    SIMULATION_MODES,
    AccountSimulationPayload,
    AccountSimulationRun,
    run_account_simulation,
)
from alpha_lab.propsim.trade_path import (
    OhlcBarPathObservation,
    build_assumed_intrabar_artifact,
    build_closed_trade_artifact,
    build_trade_path_bundle,
    evaluate_path_capabilities,
)
from alpha_lab.propsim.withdrawal import WithdrawalPolicyPayload

__all__ = ["FirmSimulationSpec", "make_prop_simulator", "persist_account_simulation"]

_DEFAULT_MODES: tuple[str, ...] = ("historical_closed_trade",)


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

    def policy_set_envelope(self) -> AccountPolicySetEnvelope:
        """The REAL resolved policy-set envelope (DEV-R4-17 closure)."""

        payload = self.policy_set()
        return AccountPolicySetEnvelope(
            account_policy_set_id=canonical_contract_sha256(payload),
            payload=payload,
        )


def _validate_modes(
    simulation_modes: tuple[str, ...],
    *,
    intrabar_scenario_policy_id: str | None,
    bootstrap_protocol_id: str | None,
    stress_scenario_ids: tuple[str, ...],
    bar_observations_for: Callable[[str], Mapping[str, Sequence[OhlcBarPathObservation]]]
    | None,
) -> None:
    if not simulation_modes:
        raise ValueError("at least one simulation mode is required")
    unknown = sorted(set(simulation_modes) - set(SIMULATION_MODES))
    if unknown:
        raise ValueError(f"unregistered simulation modes {unknown}")
    if len(set(simulation_modes)) != len(simulation_modes):
        raise ValueError("simulation modes contain duplicates")
    if "historical_ordered_event_replay" in simulation_modes:
        raise ValueError(
            "historical_ordered_event_replay is reserved for actual ordered "
            "event streams; no ordered source is wired at this seam"
        )
    if "historical_1m_scenario" in simulation_modes:
        if intrabar_scenario_policy_id is None:
            raise ValueError(
                "historical_1m_scenario requires a registered intrabar "
                "scenario policy id"
            )
        if bar_observations_for is None:
            raise ValueError(
                "historical_1m_scenario requires per-trade 1m bar observations "
                "(bar_observations_for); scenario paths are never fabricated"
            )
    if "day_block_bootstrap" in simulation_modes and bootstrap_protocol_id is None:
        raise ValueError(
            "day_block_bootstrap requires the day_block_bootstrap_h<N>_v1 "
            "protocol id (the horizon rides the identity)"
        )
    if "stress" in simulation_modes and not stress_scenario_ids:
        raise ValueError("stress mode requires at least one registered scenario id")


def persist_account_simulation(
    store_root: Path,
    run: AccountSimulationRun,
    *,
    vector: PayoutReliabilityVector,
    risk_policy_label: str,
) -> None:
    """Publish one simulation's envelope + the sidecars the trader UI reads.

    ``walk_summary.json`` always rides; the ordered historical account-event
    stream rides only for historical modes (bootstrap/stress paths advance a
    synthetic clock — their event streams are not a historical timeline).
    """

    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        save_or_reuse_envelope,
    )

    payload = run.envelope.payload
    walk_summary: dict[str, Any] = {
        "simulation_mode": payload.simulation_mode,
        "risk_policy_label": risk_policy_label,
        "n_paths": payload.n_paths,
        "total_fees": (
            float(run.walk_results[0].total_fees) if run.walk_results else None
        ),
        "verdict": run.walk_results[0].verdict if run.walk_results else None,
        "payout_reliability_vector": vector.model_dump(mode="json"),
    }
    extra_files = {
        "walk_summary.json": (
            json.dumps(walk_summary, sort_keys=True) + "\n"
        ).encode("utf-8")
    }
    if payload.simulation_mode.startswith("historical") and run.walk_results:
        events = [
            envelope.model_dump(mode="json")
            for envelope in run.walk_results[0].events
        ]
        extra_files["account_events.json"] = (
            json.dumps(events, sort_keys=True) + "\n"
        ).encode("utf-8")
    save_or_reuse_envelope(
        store_root,
        "account_simulations",
        run.envelope,
        extra_files=extra_files,
    )


def make_prop_simulator(
    specs: Sequence[FirmSimulationSpec],
    *,
    tick_size: float,
    costed_evaluation_id_for: Callable[[str], str],
    cost_points_round_turn: float = 0.0,
    seed: int = 7,
    supersessions: tuple[PropContractSupersession, ...] = (),
    simulation_modes: tuple[str, ...] = _DEFAULT_MODES,
    intrabar_scenario_policy_id: str | None = None,
    bootstrap_protocol_id: str | None = None,
    stress_scenario_ids: tuple[str, ...] = (),
    n_paths: int = 1,
    bar_observations_for: Callable[[str], Mapping[str, Sequence[OhlcBarPathObservation]]]
    | None = None,
    store_root: Path | None = None,
) -> Callable[..., dict[str, PayoutReliabilityVector]]:
    """Build the orchestrator-seam callable over the real simulation chain.

    The returned callable matches ``run_search(prop_simulator=...)``:
    ``simulator(outcome=..., result=...) -> {label: PayoutReliabilityVector}``.
    ``result`` must carry the child's rebuilt typed tables; a REUSED child
    (``result is None``) raises — wiring a table loader for reused children
    is the caller's explicit choice, never an implicit re-read.

    With the default single ``historical_closed_trade`` mode the vector keys
    are the spec labels (R3-identical). With additional modes each key is
    ``"<label>:<mode>"`` (stress adds ``":<scenario id>"``) — every leg
    enters the orchestrator's conservative ALL-legs feasibility rule.
    """

    _validate_modes(
        tuple(simulation_modes),
        intrabar_scenario_policy_id=intrabar_scenario_policy_id,
        bootstrap_protocol_id=bootstrap_protocol_id,
        stress_scenario_ids=tuple(stress_scenario_ids),
        bar_observations_for=bar_observations_for,
    )
    single_mode = len(simulation_modes) == 1

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
        closed_artifacts = tuple(
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
        directions_by_trade = {
            str(row["trade_id"]): str(row["direction"])
            for row in frame.to_dict("records")
        }
        artifacts_by_mode: dict[str, tuple] = {}
        for mode in simulation_modes:
            if mode == "historical_1m_scenario":
                observations_by_trade = bar_observations_for(core_replay_id)  # type: ignore[misc]
                missing = [
                    trade_id
                    for trade_id in ordered_trade_ids
                    if not observations_by_trade.get(trade_id)
                ]
                if missing:
                    raise ValueError(
                        "historical_1m_scenario requires bar observations for "
                        f"every executed trade; missing: {sorted(missing)[:5]}"
                    )
                artifacts_by_mode[mode] = tuple(
                    build_assumed_intrabar_artifact(
                        core_replay_id=core_replay_id,
                        trade_id=trade.trade_id,
                        trade_direction=directions_by_trade[trade.trade_id],  # type: ignore[arg-type]
                        observations=tuple(observations_by_trade[trade.trade_id]),
                        scenario_policy_id=intrabar_scenario_policy_id,  # type: ignore[arg-type]
                    )
                    for trade in account_trades
                )
            else:
                # closed-trade evidence: the honest base fidelity for the
                # historical, bootstrap, and stress walks alike
                artifacts_by_mode[mode] = closed_artifacts
        day_blocks = group_account_trades_by_day(account_trades)
        vectors: dict[str, PayoutReliabilityVector] = {}
        for spec in specs:
            policy_set_envelope = spec.policy_set_envelope()
            policy_set = policy_set_envelope.payload
            if store_root is not None:
                from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
                    save_or_reuse_envelope,
                )

                save_or_reuse_envelope(
                    store_root, "account_policy_sets", policy_set_envelope
                )
            for mode in simulation_modes:
                artifacts = artifacts_by_mode[mode]
                bundle = build_trade_path_bundle(
                    artifacts,
                    gross_trade_stream_hash=stream_hash,
                    ordered_trade_ids=ordered_trade_ids,
                )
                report = evaluate_path_capabilities(
                    bundle, spec.firm.rule_path_requirements, artifacts=artifacts
                )
                scenario_runs: tuple[str | None, ...] = (
                    tuple(stress_scenario_ids) if mode == "stress" else (None,)
                )
                for stress_scenario_id in scenario_runs:
                    payload = AccountSimulationPayload(
                        core_replay_id=core_replay_id,
                        gross_trade_stream_hash=stream_hash,
                        costed_evaluation_id=costed_evaluation_id_for(core_replay_id),
                        trade_path_bundle_id=bundle.trade_path_bundle_id,
                        trade_path_bundle_manifest_sha256=bundle.manifest_payload_sha256,
                        path_capability_report_id=canonical_contract_sha256(report),
                        account_policy_set_id=policy_set_envelope.account_policy_set_id,
                        simulation_mode=mode,  # type: ignore[arg-type]
                        intrabar_scenario_policy_id=(
                            intrabar_scenario_policy_id
                            if mode == "historical_1m_scenario"
                            else None
                        ),
                        bootstrap_protocol_id=(
                            bootstrap_protocol_id
                            if mode == "day_block_bootstrap"
                            else None
                        ),
                        stress_scenario_id=stress_scenario_id,
                        seed=seed,
                        n_paths=n_paths if mode in ("day_block_bootstrap", "stress") else 1,
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
                    vector = build_payout_reliability_vector(run.walk_results)
                    if single_mode:
                        label = spec.label
                    elif stress_scenario_id is not None:
                        label = f"{spec.label}:{mode}:{stress_scenario_id}"
                    else:
                        label = f"{spec.label}:{mode}"
                    vectors[label] = vector
                    if store_root is not None:
                        persist_account_simulation(
                            store_root,
                            run,
                            vector=vector,
                            risk_policy_label=spec.label,
                        )
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
