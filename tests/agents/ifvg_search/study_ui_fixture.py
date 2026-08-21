"""Shared R4 UI fixture: one COMPLETED synthetic 2×2 search on disk.

Runs the R2 synthetic orchestration end-to-end into a temporary store/state
root (real charter, real frontier, real costed evaluations), registers the
search in the catalog, persists one synthetic account simulation (walk
summary + totally ordered event envelopes as manifest-verified sidecars) for
the representative child, and stores the R3 synthetic fixture contract so
the wizard's prop step has a truthful ``synthetic_fixture_verified`` card.

Everything is synthetic-namespace data produced through the production
publication paths — nothing is fabricated past the store layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.search.catalog import append_catalog_event
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (
    enumerate_children,
    run_search,
)
from alpha_lab.agents.data_infra.ifvg.search.store import save_envelope_immutable
from alpha_lab.agents.data_infra.ifvg.study_providers import (
    ACCOUNT_EVENTS_SIDECAR,
    WALK_SUMMARY_SIDECAR,
    costed_evaluation_id_for,
)
from alpha_lab.propsim.account import EVENT_ORDER_POLICY_V1, PropAccountEventEnvelope
from alpha_lab.propsim.firm_contracts import AccountPhase
from alpha_lab.propsim.simulation import (
    AccountSimulationEnvelope,
    AccountSimulationPayload,
)

__all__ = ["build_completed_search"]


def _hex(seed: str) -> str:
    return canonical_contract_sha256({"fixture": seed})


def _event(
    ordinal: int,
    event_type: str,
    payload: dict[str, Any],
    *,
    ts: str,
    phase: AccountPhase = AccountPhase.FUNDED,
    trade: str | None = None,
    setup: str | None = None,
) -> dict[str, Any]:
    envelope = PropAccountEventEnvelope(
        event_id=_hex(f"event-{ordinal}"),
        event_ts_utc=ts,
        event_ordinal=ordinal,
        path_instance_id=_hex("path-0"),
        account_id="fixture-account-1",
        account_ordinal=0,
        firm_contract_id=_hex("firm"),
        account_phase=phase,
        source_trade_id=trade,
        source_decision_id=None,
        source_candidate_id=None,
        source_setup_id=setup,
        source_path_event_id=None,
        event_type=event_type,  # type: ignore[arg-type]
        event_order_policy_id=EVENT_ORDER_POLICY_V1,
        payload=payload,  # type: ignore[arg-type]
    )
    return envelope.model_dump(mode="json")


def _ordered_events() -> list[dict[str, Any]]:
    day = "2026-01-06T"
    return [
        _event(
            0,
            "equity_update",
            {
                "prior_equity": 50_000.0,
                "new_equity": 50_400.0,
                "realized_delta": 400.0,
                "unrealized_delta": 0.0,
            },
            ts=f"{day}15:00:00+00:00",
            trade="trade-0001",
            setup="setup-0001",
        ),
        _event(
            1,
            "threshold_ratchet",
            {
                "prior_floor": 48_000.0,
                "new_floor": 48_400.0,
                "reference_equity": 50_400.0,
            },
            ts=f"{day}15:00:00+00:00",
        ),
        _event(
            2,
            "payout",
            {
                "requested_amount": 300.0,
                "approved_amount": 300.0,
                "trader_amount": 270.0,
                "firm_amount": 30.0,
            },
            ts=f"{day}21:00:00+00:00",
            trade="trade-0001",
            setup="setup-0001",
        ),
        _event(
            3,
            "fee",
            {"fee_kind": "recurring", "amount": 85.0},
            ts=f"{day}21:05:00+00:00",
        ),
        _event(
            4,
            "equity_update",
            {
                "prior_equity": 50_400.0,
                "new_equity": 49_900.0,
                "realized_delta": -500.0,
                "unrealized_delta": 0.0,
            },
            ts="2026-01-07T15:30:00+00:00",
            trade="trade-0002",
            setup="setup-0002",
        ),
    ]


def _walk_summary(mode: str) -> dict[str, Any]:
    return {
        "simulation_mode": mode,
        "risk_policy_label": "fixed_one_nq",
        "total_fees": 85.0,
        "payout_reliability_vector": {
            "first_payout_probability_30d": 0.55,
            "first_payout_probability_60d": 0.70,
            "three_payout_probability": 0.30,
            "payout_probability_per_rolling_30d": 0.60,
            "median_days_between_payouts": 22.0,
            "p90_payout_drought_days": 45.0,
            "expected_net_payout_90d": 1_850.0,
            "p10_net_payout_90d": 250.0,
            "breach_probability_90d": 0.20,
            "expected_replacement_cost": 60.0,
            "median_account_lifetime_days": 75.0,
        },
        "survival_curve": {
            "days": [0, 15, 30, 60, 90],
            "survival": [1.0, 0.95, 0.9, 0.84, 0.8],
        },
        "payout_samples": {
            "30-day": [0.0, 250.0, 400.0, 800.0],
            "60-day": [0.0, 400.0, 900.0, 1500.0],
            "90-day": [0.0, 400.0, 1200.0, 2500.0, 3200.0],
            "lifetime": [0.0, 500.0, 2000.0, 5200.0],
        },
        "payout_eligibility_windows": [
            {"start": "2026-01-06T18:00:00+00:00", "end": "2026-01-07T18:00:00+00:00"}
        ],
    }


def build_completed_search(
    tmp_root: Path, *, with_prop: bool = True, with_contract: bool = True
) -> dict[str, Any]:
    """One completed synthetic search + optional prop/contract artifacts."""

    from tests.agents.ifvg_search.test_orchestrator import (
        _charter,
        _identity_resolver,
        _runner_factory,
    )

    tmp_root = Path(tmp_root)
    store_root = tmp_root / "store"
    state_root = tmp_root / "state"
    draft_root = tmp_root / "drafts"
    charter = _charter()
    specs_observed: list = []
    baseline_hash = next(
        spec.resolved_section_config_hash
        for spec in enumerate_children(
            charter,
            identity_resolver=lambda spec: _identity_resolver(spec).core_replay_id,
        )
        if spec.comparison_role == "baseline"
    )
    result = run_search(
        charter,
        store_root=store_root,
        state_root=state_root,
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory(specs_observed, {baseline_hash}),
        stale_lock_seconds=3600.0,
    )
    assert result.phase == "search_complete"
    assert result.frontier is not None
    representative = result.frontier.development_exploratory_representative_id
    assert representative is not None
    append_catalog_event(
        store_root,
        kind="display_name",
        artifact_id=charter.search_id,
        payload={"display_name": "Synthetic 2x2 study"},
    )
    from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter

    save_charter(store_root, charter)

    fixture: dict[str, Any] = {
        "store_root": store_root,
        "state_root": state_root,
        "draft_root": draft_root,
        "search_id": charter.search_id,
        "charter": charter,
        "representative": representative,
        "frontier": result.frontier,
    }

    if with_prop:
        mode = "historical_closed_trade"
        payload = AccountSimulationPayload(
            core_replay_id=representative,
            gross_trade_stream_hash=_hex("stream"),
            costed_evaluation_id=costed_evaluation_id_for(
                representative, charter.payload.cost_policy
            ),
            trade_path_bundle_id=_hex("bundle"),
            trade_path_bundle_manifest_sha256=_hex("bundle-manifest"),
            path_capability_report_id=_hex("capability"),
            account_policy_set_id=_hex("policy-set"),
            simulation_mode=mode,
            intrabar_scenario_policy_id=None,
            bootstrap_protocol_id="day_block_bootstrap_h90_v1",
            stress_scenario_id=None,
            seed=42,
            n_paths=4,
        )
        envelope = AccountSimulationEnvelope.from_payload(payload)
        import json as _json

        save_envelope_immutable(
            store_root,
            "account_simulations",
            envelope,
            extra_files={
                WALK_SUMMARY_SIDECAR: (
                    _json.dumps(_walk_summary(mode), sort_keys=True) + "\n"
                ).encode("utf-8"),
                ACCOUNT_EVENTS_SIDECAR: (
                    _json.dumps(_ordered_events(), sort_keys=True) + "\n"
                ).encode("utf-8"),
            },
        )
        append_catalog_event(
            store_root,
            kind="display_name",
            artifact_id=envelope.account_simulation_id,
            payload={"display_name": "synthetic_fixture_firm · fixed_one_nq"},
        )
        fixture["simulation_id"] = envelope.account_simulation_id

    if with_contract:
        from tests.propsim.test_synthetic_contract_e2e import (
            _compiled_synthetic_contract,
        )

        contract_envelope, _compilation = _compiled_synthetic_contract()
        save_envelope_immutable(store_root, "prop_contracts", contract_envelope)
        append_catalog_event(
            store_root,
            kind="display_name",
            artifact_id=contract_envelope.firm_contract_id,
            payload={"display_name": "Synthetic fixture firm 50k"},
        )
        fixture["contract_id"] = contract_envelope.firm_contract_id

    return fixture
