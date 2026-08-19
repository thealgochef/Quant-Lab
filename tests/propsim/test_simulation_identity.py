"""Simulation identity, revalidation, modes, bootstrap, portfolio (§3.4/§3.10)."""

from __future__ import annotations

import inspect
from datetime import date

import pytest

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from alpha_lab.propsim.account import AccountPolicySetPayload, AccountTrade
from alpha_lab.propsim.contract_evidence import (
    PropContractSupersession,
    SupersededContractError,
)
from alpha_lab.propsim.firm_contracts import SYNTHETIC_FIXTURE_FIRM
from alpha_lab.propsim.portfolio import (
    LegPolicies,
    run_portfolio_bootstrap,
    run_portfolio_replay,
)
from alpha_lab.propsim.risk import FIXED_ONE_NQ_RISK_POLICY
from alpha_lab.propsim.simulation import (
    AccountSimulationPayload,
    PortfolioLeg,
    SimulationIdentityError,
    UnsupportedSimulationModeError,
    bootstrap_horizon_for,
    run_account_simulation,
)
from alpha_lab.propsim.stress import STRESS_SCENARIOS_V1, apply_stress_scenario
from alpha_lab.propsim.trade_path import (
    OhlcBarPathObservation,
    PathCapabilityReport,
    build_assumed_intrabar_artifact,
    build_closed_trade_artifact,
    build_trade_path_bundle,
    evaluate_path_capabilities,
)
from alpha_lab.propsim.withdrawal import REQUEST_MAX_AT_ELIGIBILITY

_FIRM_ID = canonical_contract_sha256(SYNTHETIC_FIXTURE_FIRM)

_POLICY_SET = AccountPolicySetPayload(
    firm_contract_id=_FIRM_ID,
    risk_policy_id=canonical_contract_sha256(FIXED_ONE_NQ_RISK_POLICY),
    withdrawal_policy_id=canonical_contract_sha256(REQUEST_MAX_AT_ELIGIBILITY),
    replacement_policy="none",
    max_replacements=0,
    clock_policy_id="historical_calendar_clock_v1",
)

_CORE = "a" * 64
_STREAM = "b" * 64


def _trade(day: str, points: float, slot: int = 0) -> AccountTrade:
    return AccountTrade(
        day=date.fromisoformat(day),
        entry_ts_utc=f"{day}T1{4 + slot}:00:00+00:00",
        resolution_ts_utc=f"{day}T1{4 + slot}:05:00+00:00",
        points=points,
        risk_points=10.0,
        mfe_pts=None,
        mae_pts=None,
        trade_id=f"trade-{day}-{slot}",
    )


_DAY_BLOCKS = [
    (date(2026, 1, 13), [_trade("2026-01-13", 40.0)]),
    (date(2026, 1, 14), [_trade("2026-01-14", -20.0)]),
    (date(2026, 1, 15), [_trade("2026-01-15", 30.0)]),
]


def _evidence(day_blocks=_DAY_BLOCKS, *, firm=SYNTHETIC_FIXTURE_FIRM):
    """REAL artifacts + bundle + capability report for the walked stream."""

    trades = [trade for _day, block in day_blocks for trade in block]
    artifacts = tuple(
        build_closed_trade_artifact(
            core_replay_id=_CORE,
            trade_id=trade.trade_id,
            entry_ts_utc=trade.entry_ts_utc,
            resolution_ts_utc=trade.resolution_ts_utc,
            entry_price=20_000.0,
            exit_price=20_000.0 + trade.points,
        )
        for trade in trades
    )
    bundle = build_trade_path_bundle(
        artifacts,
        gross_trade_stream_hash=_STREAM,
        ordered_trade_ids=tuple(trade.trade_id for trade in trades),
    )
    report = evaluate_path_capabilities(
        bundle, firm.rule_path_requirements, artifacts=artifacts
    )
    return artifacts, bundle, report


_ARTIFACTS, _BUNDLE, _REPORT = _evidence()


def _payload(*, bundle=_BUNDLE, report=_REPORT, policy_set=_POLICY_SET, **overrides):
    base = dict(
        core_replay_id=_CORE,
        gross_trade_stream_hash=_STREAM,
        costed_evaluation_id="c" * 64,
        trade_path_bundle_id=bundle.trade_path_bundle_id,
        trade_path_bundle_manifest_sha256=bundle.manifest_payload_sha256,
        path_capability_report_id=canonical_contract_sha256(report),
        account_policy_set_id=canonical_contract_sha256(policy_set),
        simulation_mode="historical_closed_trade",
        intrabar_scenario_policy_id=None,
        bootstrap_protocol_id=None,
        stress_scenario_id=None,
        seed=42,
        n_paths=1,
    )
    base.update(overrides)
    return AccountSimulationPayload(**base)


def _run(payload: AccountSimulationPayload, **overrides):
    kwargs = dict(
        firm=SYNTHETIC_FIXTURE_FIRM,
        risk_policy=FIXED_ONE_NQ_RISK_POLICY,
        withdrawal_policy=REQUEST_MAX_AT_ELIGIBILITY,
        policy_set=_POLICY_SET,
        day_blocks=_DAY_BLOCKS,
        bundle=_BUNDLE,
        artifacts=_ARTIFACTS,
        capability_report=_REPORT,
    )
    kwargs.update(overrides)
    return run_account_simulation(payload, **kwargs)


def test_simulation_identity_is_sensitive_to_every_result_changing_field() -> None:
    base = _payload()
    for field, value in (
        ("trade_path_bundle_id", "1" * 64),
        ("account_policy_set_id", "2" * 64),
        ("simulation_mode", "day_block_bootstrap"),
        ("bootstrap_protocol_id", "day_block_bootstrap_h3_v1"),
        ("seed", 7),
        ("n_paths", 2),
        ("stress_scenario_id", "stress_cost_x2_v1"),
    ):
        changed = base.model_copy(update={field: value})
        assert canonical_contract_sha256(changed) != canonical_contract_sha256(base)


def test_no_result_changing_constructor_only_argument_exists() -> None:
    """P0-16 constructor-surface audit: every runner kwarg is an identity
    field, a resolved artifact one of those identities pins (and the runner
    REVALIDATES), or a pure refusal registry that can only block a run."""

    identity_fields = set(AccountSimulationPayload.model_fields)
    parameters = set(inspect.signature(run_account_simulation).parameters) - {"payload"}
    resolved_artifacts = {
        "firm",  # ← policy_set.firm_contract_id (hash-revalidated)
        "risk_policy",  # ← policy_set.risk_policy_id (hash-revalidated)
        "withdrawal_policy",  # ← policy_set.withdrawal_policy_id (revalidated)
        "policy_set",  # ← account_policy_set_id (hash-revalidated)
        "day_blocks",  # ← bundle.ordered_trade_ids coverage (revalidated)
        "bundle",  # ← trade_path_bundle_id + manifest (revalidated)
        "artifacts",  # ← bundle.ordered_trade_path_artifact_ids (revalidated)
        "capability_report",  # ← path_capability_report_id (hash-revalidated)
    }
    refusal_registries = {"supersessions"}  # can only refuse, never change a number
    unexplained = parameters - resolved_artifacts - refusal_registries - identity_fields
    assert not unexplained, f"unaudited runner arguments: {unexplained}"
    # the former result-shaping constructor argument is gone: the bootstrap
    # horizon rides the registered protocol id inside the identity
    assert "max_days_per_path" not in parameters


def test_runner_revalidates_every_pinned_artifact() -> None:
    """CS §0.3(c): caller trust is never sufficient — mismatches fail closed."""

    tampered_firm = SYNTHETIC_FIXTURE_FIRM.model_copy(
        update={"account_size_label": "100k"}
    )
    with pytest.raises(SimulationIdentityError, match="firm_contract_id"):
        _run(_payload(), firm=tampered_firm)
    empty_report = PathCapabilityReport(
        trade_path_bundle_id=_BUNDLE.trade_path_bundle_id,
        available_capabilities=(),
        per_rule={},
        failure_reasons={},
    )
    with pytest.raises(SimulationIdentityError, match="path_capability_report_id"):
        _run(_payload(), capability_report=empty_report)
    # even correctly-pinned, an empty report cannot cover the firm's rules
    with pytest.raises(SimulationIdentityError, match="does not cover"):
        _run(_payload(report=empty_report), capability_report=empty_report)
    with pytest.raises(SimulationIdentityError, match="trade_path_bundle_id"):
        _run(_payload(trade_path_bundle_id="9" * 64))
    stray = _trade("2026-01-16", 5.0)
    with pytest.raises(SimulationIdentityError, match="outside the bundle"):
        _run(_payload(), day_blocks=[*_DAY_BLOCKS, (date(2026, 1, 16), [stray])])


def test_superseded_contract_is_refused_for_new_simulations() -> None:
    supersession = PropContractSupersession(
        prior_firm_contract_id=_FIRM_ID,
        replacement_firm_contract_id="7" * 64,
        reason="synthetic fixture superseded by v2",
        effective_at="2026-01-20T00:00:00+00:00",
    )
    with pytest.raises(SupersededContractError, match="superseded"):
        _run(_payload(), supersessions=(supersession,))
    # an unrelated supersession does not block
    unrelated = supersession.model_copy(update={"prior_firm_contract_id": "8" * 64})
    run = _run(_payload(), supersessions=(unrelated,))
    assert len(run.walk_results) == 1


def test_bootstrap_paths_are_unique_instances_with_stored_sequence_hashes() -> None:
    """P0-21: duplicate sampled sequences are LEGAL; instances stay unique;
    the horizon rides the registered protocol id (identity, not an argument)."""

    payload = _payload(
        simulation_mode="day_block_bootstrap",
        bootstrap_protocol_id="day_block_bootstrap_h3_v1",
        seed=7,
        n_paths=8,
    )
    # a 1-day universe provably repeats the same sampled sequence
    run = _run(payload, day_blocks=_DAY_BLOCKS[:1])
    hashes = [record.sampled_index_sequence_hash for record in run.path_records]
    instances = [record.path_instance_id for record in run.path_records]
    assert len(set(instances)) == len(instances) == 8
    assert len(set(hashes)) < len(hashes)  # duplicates occurred and were accepted
    assert [record.draw_ordinal for record in run.path_records] == list(range(8))
    # same spec → identical draw sequence
    rerun = _run(payload, day_blocks=_DAY_BLOCKS[:1])
    assert [r.sampled_index_sequence_hash for r in rerun.path_records] == hashes
    # a different protocol id is a different identity AND different results
    longer = _payload(
        simulation_mode="day_block_bootstrap",
        bootstrap_protocol_id="day_block_bootstrap_h40_v1",
        seed=7,
        n_paths=8,
    )
    assert canonical_contract_sha256(longer) != canonical_contract_sha256(payload)
    longer_run = _run(longer, day_blocks=_DAY_BLOCKS[:1])
    assert [
        r.sampled_index_sequence_hash for r in longer_run.path_records
    ] != hashes
    assert bootstrap_horizon_for("day_block_bootstrap_h40_v1") == 40
    with pytest.raises(UnsupportedSimulationModeError, match="unregistered bootstrap"):
        bootstrap_horizon_for("my_bootstrap_v9")
    with pytest.raises(UnsupportedSimulationModeError, match="registered bootstrap"):
        _run(
            _payload(simulation_mode="day_block_bootstrap", bootstrap_protocol_id=None),
            day_blocks=_DAY_BLOCKS[:1],
        )


def test_modes_fail_closed_on_unsupported_evidence() -> None:
    with pytest.raises(UnsupportedSimulationModeError, match="ACTUAL ordered"):
        _run(_payload(simulation_mode="historical_ordered_event_replay"))
    with pytest.raises(UnsupportedSimulationModeError, match="registered"):
        _run(_payload(simulation_mode="historical_1m_scenario"))
    with pytest.raises(UnsupportedSimulationModeError, match="unregistered intrabar"):
        _run(
            _payload(
                simulation_mode="historical_1m_scenario",
                intrabar_scenario_policy_id="my_secret_order_v1",
            )
        )
    with pytest.raises(UnsupportedSimulationModeError, match="scenario artifacts"):
        _run(
            _payload(
                simulation_mode="historical_1m_scenario",
                intrabar_scenario_policy_id="bar_adverse_extreme_first_v1",
            )
        )
    failing_report = PathCapabilityReport(
        trade_path_bundle_id=_BUNDLE.trade_path_bundle_id,
        available_capabilities=(),
        per_rule={
            requirement.rule_id: "unsupported"
            for requirement in SYNTHETIC_FIXTURE_FIRM.rule_path_requirements
        },
        failure_reasons={
            requirement.rule_id: ("missing capabilities",)
            for requirement in SYNTHETIC_FIXTURE_FIRM.rule_path_requirements
        },
    )
    with pytest.raises(UnsupportedSimulationModeError, match="fail-closed"):
        _run(_payload(report=failing_report), capability_report=failing_report)


def _scenario_fixture(policy_id: str):
    """An order-sensitive one-trade stream under an intraday-trail firm."""

    intraday_phase = SYNTHETIC_FIXTURE_FIRM.evaluation.model_copy(
        update={"trail_style": "intraday_peak_trail", "dll_amount": None}
    )
    firm = SYNTHETIC_FIXTURE_FIRM.model_copy(
        update={
            "evaluation": intraday_phase,
            "funded": None,
            "payout": None,
            "rule_path_requirements": tuple(
                req
                for req in SYNTHETIC_FIXTURE_FIRM.rule_path_requirements
                if req.rule_id.startswith("evaluation.breach_observation")
            ),
        }
    )
    policy_set = _POLICY_SET.model_copy(
        update={"firm_contract_id": canonical_contract_sha256(firm)}
    )
    trade = AccountTrade(
        day=date(2026, 1, 13),
        entry_ts_utc="2026-01-13T14:00:00+00:00",
        resolution_ts_utc="2026-01-13T14:30:00+00:00",
        points=10.0,
        risk_points=10.0,
        mfe_pts=60.0,  # favorable excursion raises the trail by $1,200
        mae_pts=90.0,  # adverse excursion: $1,800 against the entry balance
        trade_id="trade-order-sensitive",
    )
    day_blocks = [(trade.day, [trade])]
    bar = OhlcBarPathObservation(
        open_ts_utc=trade.entry_ts_utc,
        close_ts_utc=trade.resolution_ts_utc,
        open_price=20_000.0,
        high_price=20_060.0,
        low_price=19_910.0,
        close_price=20_010.0,
    )
    artifacts = (
        build_assumed_intrabar_artifact(
            core_replay_id=_CORE,
            trade_id=trade.trade_id,
            trade_direction="long",
            observations=(bar,),
            scenario_policy_id=policy_id,
        ),
    )
    bundle = build_trade_path_bundle(
        artifacts,
        gross_trade_stream_hash=_STREAM,
        ordered_trade_ids=(trade.trade_id,),
    )
    report = evaluate_path_capabilities(
        bundle, firm.rule_path_requirements, artifacts=artifacts
    )
    payload = _payload(
        bundle=bundle,
        report=report,
        policy_set=policy_set,
        simulation_mode="historical_1m_scenario",
        intrabar_scenario_policy_id=policy_id,
    )
    return payload, firm, policy_set, day_blocks, bundle, artifacts, report


def test_two_scenario_policies_produce_two_identities_and_two_results() -> None:
    """§3.8 (R3): the scenario id is never inert — on an order-sensitive
    trade the two registered policies walk to DIFFERENT verdicts."""

    outcomes = {}
    for policy_id in (
        "bar_adverse_extreme_first_v1",
        "bar_favorable_extreme_first_v1",
    ):
        payload, firm, policy_set, day_blocks, bundle, artifacts, report = (
            _scenario_fixture(policy_id)
        )
        run = run_account_simulation(
            payload,
            firm=firm,
            risk_policy=FIXED_ONE_NQ_RISK_POLICY,
            withdrawal_policy=REQUEST_MAX_AT_ELIGIBILITY,
            policy_set=policy_set,
            day_blocks=day_blocks,
            bundle=bundle,
            artifacts=artifacts,
            capability_report=report,
        )
        outcomes[policy_id] = (
            run.envelope.account_simulation_id,
            run.walk_results[0].verdict,
            run.walk_results[0].final_state.balance,
        )
    adverse = outcomes["bar_adverse_extreme_first_v1"]
    favorable = outcomes["bar_favorable_extreme_first_v1"]
    assert adverse[0] != favorable[0]  # two scenario identities
    # favorable-first raises the trail from MFE before the MAE test → breach;
    # adverse-first tests MAE against the unraised floor → survives
    assert favorable[1] == "breached_out"
    assert adverse[1] == "evaluation_alive"
    assert adverse[2] != favorable[2]


def test_scenario_artifacts_must_match_the_payload_policy() -> None:
    payload, firm, policy_set, day_blocks, bundle, artifacts, report = (
        _scenario_fixture("bar_adverse_extreme_first_v1")
    )
    mismatched = payload.model_copy(
        update={"intrabar_scenario_policy_id": "bar_favorable_extreme_first_v1"}
    )
    with pytest.raises(SimulationIdentityError, match="DIFFERENT intrabar"):
        run_account_simulation(
            mismatched,
            firm=firm,
            risk_policy=FIXED_ONE_NQ_RISK_POLICY,
            withdrawal_policy=REQUEST_MAX_AT_ELIGIBILITY,
            policy_set=policy_set,
            day_blocks=day_blocks,
            bundle=bundle,
            artifacts=artifacts,
            capability_report=report,
        )


def test_stress_scenarios_are_registered_and_deterministic() -> None:
    assert len(STRESS_SCENARIOS_V1) == 9
    blocks = [(day, tuple(trades)) for day, trades in _DAY_BLOCKS]
    once = apply_stress_scenario(blocks, "stress_shuffle_days_a_v1", base_seed=42)
    twice = apply_stress_scenario(blocks, "stress_shuffle_days_a_v1", base_seed=42)
    assert [d for d, _ in once] == [d for d, _ in twice]
    other = apply_stress_scenario(blocks, "stress_shuffle_days_b_v1", base_seed=42)
    assert once != other  # distinct seed offsets → distinct deterministic orders
    dropped = apply_stress_scenario(blocks, "stress_drop_best_day_v1", base_seed=42)
    assert len(dropped) == len(blocks) - 1
    costed = apply_stress_scenario(blocks, "stress_cost_x2_v1", base_seed=42)
    assert costed[0][1][0].points == pytest.approx(blocks[0][1][0].points - 0.514)
    with pytest.raises(ValueError, match="unregistered stress scenario"):
        apply_stress_scenario(blocks, "my_scenario", base_seed=42)
    run = _run(
        _payload(simulation_mode="stress", stress_scenario_id="stress_cost_x2_v1")
    )
    assert len(run.walk_results) == 1


def _leg_fixture():
    legs = (
        PortfolioLeg(
            leg_id="leg-1",
            account_policy_set_id=canonical_contract_sha256(_POLICY_SET),
            n_accounts=2,
        ),
    )
    policies = {
        "leg-1": LegPolicies(
            firm=SYNTHETIC_FIXTURE_FIRM,
            risk_policy=FIXED_ONE_NQ_RISK_POLICY,
            withdrawal_policy=REQUEST_MAX_AT_ELIGIBILITY,
            policy_set=_POLICY_SET,
        )
    }
    return legs, policies


def test_portfolio_replays_one_common_path_with_no_resampling_surface() -> None:
    """§3.10 + the §9 guardrail: copied accounts share ONE correlated path and
    the replay path has no per-account sampling surface."""

    legs, policies = _leg_fixture()
    result = run_portfolio_replay(
        legs,
        policies,
        day_blocks=_DAY_BLOCKS,
        path_instance_id="path-00000-portfolio",
    )
    assert len(result.per_account) == 2
    (key_a, walk_a), (key_b, walk_b) = result.per_account
    assert {key_a, key_b} == {"leg-1#0", "leg-1#1"}
    # identical policies over the one common path → identical balances,
    # but copies never collide on account/event identities
    assert walk_a.final_state.balance == pytest.approx(walk_b.final_state.balance)
    ids_a = {event.account_id for event in walk_a.events}
    ids_b = {event.account_id for event in walk_b.events}
    assert not ids_a & ids_b
    event_ids = [e.event_id for e in (*walk_a.events, *walk_b.events)]
    assert len(set(event_ids)) == len(event_ids)
    # the REPLAY path has no sampling/resampling surface (the portfolio
    # bootstrap draws once per path at portfolio level, never per account)
    source = inspect.getsource(run_portfolio_replay)
    for token in ("integers(", "choice(", "resample", "default_rng"):
        assert token not in source
    parameters = set(inspect.signature(run_portfolio_replay).parameters)
    assert not any("sample" in name or "seed" in name for name in parameters)


def test_portfolio_leg_pin_binding_and_bootstrap_shared_sequence() -> None:
    """The leg's policy-set id must hash-match its resolved policies, and the
    portfolio bootstrap shares ONE sampled sequence across every copy."""

    legs, policies = _leg_fixture()
    bad_leg = legs[0].model_copy(update={"account_policy_set_id": "9" * 64})
    with pytest.raises(ValueError, match="pins"):
        run_portfolio_replay(
            (bad_leg,),
            policies,
            day_blocks=_DAY_BLOCKS,
            path_instance_id="path-00000-portfolio",
        )
    result = run_portfolio_bootstrap(
        legs,
        policies,
        day_blocks=_DAY_BLOCKS,
        bootstrap_protocol_id="day_block_bootstrap_h3_v1",
        seed=11,
        n_paths=2,
    )
    assert len(result.per_path) == 2
    for record, replay in result.per_path:
        # one sampled sequence per path — every copy replayed exactly it
        assert record.sampled_index_sequence_hash
        (_key_a, walk_a), (_key_b, walk_b) = replay.per_account
        assert walk_a.final_state.balance == pytest.approx(
            walk_b.final_state.balance
        )
    rerun = run_portfolio_bootstrap(
        legs,
        policies,
        day_blocks=_DAY_BLOCKS,
        bootstrap_protocol_id="day_block_bootstrap_h3_v1",
        seed=11,
        n_paths=2,
    )
    assert [r.sampled_index_sequence_hash for r, _ in rerun.per_path] == [
        r.sampled_index_sequence_hash for r, _ in result.per_path
    ]
