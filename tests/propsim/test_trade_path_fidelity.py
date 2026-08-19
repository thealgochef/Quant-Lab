"""Path-fidelity truthfulness + capability-based rule support (TM §3.3/§3.8/§3.10)."""

from __future__ import annotations

import pydantic
import pytest

from alpha_lab.propsim.trade_path import (
    INTRABAR_SCENARIO_POLICIES,
    OhlcBarPathObservation,
    PathCapability,
    PropRulePathRequirement,
    TradePathFidelity,
    build_assumed_intrabar_artifact,
    build_closed_trade_artifact,
    build_ohlc_artifact,
    build_trade_path_bundle,
    evaluate_path_capabilities,
)

_CORE = "a" * 64


def _bar(open_p=20000.0, high=20020.0, low=19970.0, close=20010.0):
    return OhlcBarPathObservation(
        open_ts_utc="2026-01-13T14:00:00+00:00",
        close_ts_utc="2026-01-13T14:01:00+00:00",
        open_price=open_p,
        high_price=high,
        low_price=low,
        close_price=close,
    )


def test_ohlc_observation_cannot_claim_an_intrabar_order() -> None:
    """P0-H: `observed_intrabar_order == "unknown"` is the ONLY representable value."""

    assert _bar().observed_intrabar_order == "unknown"
    with pytest.raises(pydantic.ValidationError):
        OhlcBarPathObservation(
            open_ts_utc="t",
            close_ts_utc="t",
            open_price=1.0,
            high_price=2.0,
            low_price=0.0,
            close_price=1.5,
            observed_intrabar_order="low_first",  # type: ignore[arg-type]
        )


def test_two_scenario_policies_are_two_identities_with_different_paths() -> None:
    """An order-sensitive trade: adverse-first vs favorable-first differ."""

    bar = _bar()
    adverse = build_assumed_intrabar_artifact(
        core_replay_id=_CORE,
        trade_id="trade-1",
        trade_direction="long",
        observations=(bar,),
        scenario_policy_id="bar_adverse_extreme_first_v1",
    )
    favorable = build_assumed_intrabar_artifact(
        core_replay_id=_CORE,
        trade_id="trade-1",
        trade_direction="long",
        observations=(bar,),
        scenario_policy_id="bar_favorable_extreme_first_v1",
    )
    assert adverse.trade_path_artifact_id != favorable.trade_path_artifact_id
    first_adverse = adverse.payload.ordered_events[0].price
    first_favorable = favorable.payload.ordered_events[0].price
    assert first_adverse != first_favorable  # the traversal order genuinely differs
    assert adverse.payload.fidelity is TradePathFidelity.ASSUMED_1M_INTRABAR_PATH
    assert all(
        event.generating_policy_id == "bar_adverse_extreme_first_v1"
        for event in adverse.payload.ordered_events
    )
    with pytest.raises(ValueError, match="unregistered intrabar scenario policy"):
        build_assumed_intrabar_artifact(
            core_replay_id=_CORE,
            trade_id="trade-1",
            trade_direction="long",
            observations=(bar,),
            scenario_policy_id="my_own_ordering",
        )
    assert INTRABAR_SCENARIO_POLICIES == (
        "bar_adverse_extreme_first_v1",
        "bar_favorable_extreme_first_v1",
    )


def test_bundle_identity_pins_every_artifact_and_changes_with_any_of_them() -> None:
    """P0-H/§3.10: changing any per-trade path artifact changes the bundle id."""

    closed_1 = build_closed_trade_artifact(
        core_replay_id=_CORE,
        trade_id="trade-1",
        entry_ts_utc="2026-01-13T14:00:00+00:00",
        resolution_ts_utc="2026-01-13T14:05:00+00:00",
        entry_price=20000.0,
        exit_price=20010.0,
    )
    closed_2 = build_closed_trade_artifact(
        core_replay_id=_CORE,
        trade_id="trade-2",
        entry_ts_utc="2026-01-13T15:00:00+00:00",
        resolution_ts_utc="2026-01-13T15:05:00+00:00",
        entry_price=20005.0,
        exit_price=19995.0,
    )
    bundle = build_trade_path_bundle(
        (closed_1, closed_2),
        gross_trade_stream_hash="b" * 64,
        ordered_trade_ids=("trade-1", "trade-2"),
    )
    changed_2 = build_closed_trade_artifact(
        core_replay_id=_CORE,
        trade_id="trade-2",
        entry_ts_utc="2026-01-13T15:00:00+00:00",
        resolution_ts_utc="2026-01-13T15:05:00+00:00",
        entry_price=20005.0,
        exit_price=19990.0,  # one price changed
    )
    changed_bundle = build_trade_path_bundle(
        (closed_1, changed_2),
        gross_trade_stream_hash="b" * 64,
        ordered_trade_ids=("trade-1", "trade-2"),
    )
    assert bundle.trade_path_bundle_id != changed_bundle.trade_path_bundle_id

    # completeness: every executed trade exactly one artifact (§3.10)
    with pytest.raises(ValueError, match="without a path artifact"):
        build_trade_path_bundle(
            (closed_1,),
            gross_trade_stream_hash="b" * 64,
            ordered_trade_ids=("trade-1", "trade-2"),
        )
    with pytest.raises(ValueError, match="more than one path artifact"):
        build_trade_path_bundle(
            (closed_1, closed_1),
            gross_trade_stream_hash="b" * 64,
            ordered_trade_ids=("trade-1",),
        )
    # exact, deterministic path-event ids
    ids = [event.event_id for event in closed_1.payload.ordered_events]
    rebuilt = build_closed_trade_artifact(
        core_replay_id=_CORE,
        trade_id="trade-1",
        entry_ts_utc="2026-01-13T14:00:00+00:00",
        resolution_ts_utc="2026-01-13T14:05:00+00:00",
        entry_price=20000.0,
        exit_price=20010.0,
    )
    assert [event.event_id for event in rebuilt.payload.ordered_events] == ids


def test_rule_support_is_capability_based_never_enum_ordered() -> None:
    """P0-12/P0-H + §3.10: chronology-sensitive rules refuse OHLC-only;
    insensitive rules accept it; ordered fills do not imply market chronology."""

    ohlc = build_ohlc_artifact(
        core_replay_id=_CORE, trade_id="trade-1", observations=(_bar(),)
    )
    bundle = build_trade_path_bundle(
        (ohlc,),
        gross_trade_stream_hash="b" * 64,
        ordered_trade_ids=("trade-1",),
    )
    chronology_rule = PropRulePathRequirement(
        rule_id="funded.breach_observation_continuous",
        required_path_capabilities=(
            PathCapability.MARKET_PRICE_CHRONOLOGY,
            PathCapability.UNREALIZED_MARK_TO_MARKET,
        ),
        accepted_fidelity_classes=(
            TradePathFidelity.ASSUMED_1M_INTRABAR_PATH,
            TradePathFidelity.ORDERED_MBP1_EVENT_PATH,
        ),
        scenario_use_permitted=True,
    )
    extrema_insensitive_rule = PropRulePathRequirement(
        rule_id="evaluation.breach_observation_eod",
        required_path_capabilities=(PathCapability.CLOSED_TRADE_RESULT,),
        accepted_fidelity_classes=(TradePathFidelity.OHLC_1M_UNORDERED,),
        scenario_use_permitted=False,
    )
    report = evaluate_path_capabilities(
        bundle,
        (chronology_rule, extrema_insensitive_rule),
        artifacts=(ohlc,),
    )
    assert report.per_rule["funded.breach_observation_continuous"] == "unsupported"
    assert any(
        "missing capabilities" in reason
        for reason in report.failure_reasons["funded.breach_observation_continuous"]
    )
    assert report.per_rule["evaluation.breach_observation_eod"] == "supported"

    # scenario evidence satisfies a chronology rule ONLY as scenario_only,
    # and ONLY when the rule permits it
    scenario = build_assumed_intrabar_artifact(
        core_replay_id=_CORE,
        trade_id="trade-1",
        trade_direction="long",
        observations=(_bar(),),
        scenario_policy_id="bar_adverse_extreme_first_v1",
    )
    scenario_bundle = build_trade_path_bundle(
        (scenario,),
        gross_trade_stream_hash="b" * 64,
        ordered_trade_ids=("trade-1",),
    )
    report = evaluate_path_capabilities(
        scenario_bundle,
        (chronology_rule,),
        artifacts=(scenario,),
    )
    assert report.per_rule["funded.breach_observation_continuous"] == "scenario_only"
    forbidden = chronology_rule.model_copy(update={"scenario_use_permitted": False})
    report = evaluate_path_capabilities(
        scenario_bundle,
        (forbidden,),
        artifacts=(scenario,),
    )
    assert report.per_rule["funded.breach_observation_continuous"] == "unsupported"

    # ordered fills provide ORDERED_FILLS but never market-price chronology
    from alpha_lab.propsim.trade_path import _FIDELITY_CAPABILITIES

    fill_caps = _FIDELITY_CAPABILITIES[TradePathFidelity.ORDERED_FILL_EVENT_PATH]
    assert PathCapability.ORDERED_FILLS in fill_caps
    assert PathCapability.MARKET_PRICE_CHRONOLOGY not in fill_caps


def test_scenario_wording_is_never_exact_historical() -> None:
    """Source scan (P0-H): assumed paths are scenario/approximation wording."""

    from pathlib import Path

    import alpha_lab.propsim.simulation as simulation_module
    import alpha_lab.propsim.trade_path as trade_path_module

    for module in (trade_path_module, simulation_module):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        assert "exact historical" not in source


def test_mixed_fidelity_bundle_degrades_per_trade_not_per_bundle() -> None:
    """A chronology rule is never 'supported' for a stream in which ANY
    trade's only evidence is an assumed scenario (per-trade fail-closed)."""

    ordered = build_ohlc_artifact(
        core_replay_id=_CORE, trade_id="trade-1", observations=(_bar(),)
    )
    # trade-2 carries only an assumed scenario path
    scenario = build_assumed_intrabar_artifact(
        core_replay_id=_CORE,
        trade_id="trade-2",
        trade_direction="long",
        observations=(_bar(),),
        scenario_policy_id="bar_adverse_extreme_first_v1",
    )
    bundle = build_trade_path_bundle(
        (ordered, scenario),
        gross_trade_stream_hash="b" * 64,
        ordered_trade_ids=("trade-1", "trade-2"),
    )
    rule = PropRulePathRequirement(
        rule_id="funded.breach_observation_continuous",
        required_path_capabilities=(PathCapability.UNREALIZED_MARK_TO_MARKET,),
        accepted_fidelity_classes=(
            TradePathFidelity.OHLC_1M_UNORDERED,
            TradePathFidelity.ASSUMED_1M_INTRABAR_PATH,
        ),
        scenario_use_permitted=True,
    )
    report = evaluate_path_capabilities(bundle, (rule,), artifacts=(ordered, scenario))
    # one scenario-only trade degrades the WHOLE stream to scenario_only
    assert report.per_rule["funded.breach_observation_continuous"] == "scenario_only"
    forbidden = rule.model_copy(update={"scenario_use_permitted": False})
    report = evaluate_path_capabilities(
        bundle, (forbidden,), artifacts=(ordered, scenario)
    )
    assert report.per_rule["funded.breach_observation_continuous"] == "unsupported"
    assert any(
        "trade-2" in reason
        for reason in report.failure_reasons["funded.breach_observation_continuous"]
    )
    # an empty bundle cannot claim a core replay at all
    with pytest.raises(ValueError, match="at least one artifact"):
        build_trade_path_bundle(
            (), gross_trade_stream_hash="b" * 64, ordered_trade_ids=()
        )


def test_account_events_carry_positive_path_event_links() -> None:
    """CS §5.3 positive direction: a trade citing a REAL bundle event id
    surfaces it on the emitted equity_update envelope."""

    from datetime import date

    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        canonical_contract_sha256,
    )
    from alpha_lab.propsim.account import (
        AccountPolicySetPayload,
        AccountTrade,
        AccountWalk,
    )
    from alpha_lab.propsim.firm_contracts import SYNTHETIC_FIXTURE_FIRM
    from alpha_lab.propsim.risk import FIXED_ONE_NQ_RISK_POLICY
    from alpha_lab.propsim.withdrawal import REQUEST_MAX_AT_ELIGIBILITY

    artifact = build_closed_trade_artifact(
        core_replay_id=_CORE,
        trade_id="trade-linked",
        entry_ts_utc="2026-01-13T14:00:00+00:00",
        resolution_ts_utc="2026-01-13T14:05:00+00:00",
        entry_price=20_000.0,
        exit_price=20_010.0,
    )
    exit_event_id = artifact.payload.ordered_events[-1].event_id
    policy_set = AccountPolicySetPayload(
        firm_contract_id=canonical_contract_sha256(SYNTHETIC_FIXTURE_FIRM),
        risk_policy_id=canonical_contract_sha256(FIXED_ONE_NQ_RISK_POLICY),
        withdrawal_policy_id=canonical_contract_sha256(REQUEST_MAX_AT_ELIGIBILITY),
        replacement_policy="none",
        max_replacements=0,
        clock_policy_id="historical_calendar_clock_v1",
    )
    walk = AccountWalk(
        firm=SYNTHETIC_FIXTURE_FIRM,
        firm_contract_id=policy_set.firm_contract_id,
        risk_policy=FIXED_ONE_NQ_RISK_POLICY,
        withdrawal_policy=REQUEST_MAX_AT_ELIGIBILITY,
        policy_set=policy_set,
        path_instance_id="path-00000-links",
        enable_fees=False,
        bundle_event_ids=frozenset(
            event.event_id for event in artifact.payload.ordered_events
        ),
    )
    walk.play_day(
        date(2026, 1, 13),
        [
            AccountTrade(
                day=date(2026, 1, 13),
                entry_ts_utc="2026-01-13T14:00:00+00:00",
                resolution_ts_utc="2026-01-13T14:05:00+00:00",
                points=10.0,
                risk_points=10.0,
                mfe_pts=None,
                mae_pts=None,
                trade_id="trade-linked",
                exit_path_event_id=exit_event_id,
            )
        ],
    )
    updates = [
        event for event in walk.result().events if event.event_type == "equity_update"
    ]
    assert updates[0].source_path_event_id == exit_event_id
    assert updates[0].source_trade_id == "trade-linked"


def test_scenario_wording_scan_covers_every_propsim_lifecycle_module() -> None:
    """Widened source scan: the untruthful phrase appears in NO R3 module."""

    from pathlib import Path

    import alpha_lab.propsim as propsim_package

    package_dir = Path(propsim_package.__file__).parent
    lifecycle_modules = (
        "trade_path.py",
        "calendar.py",
        "firm_contracts.py",
        "contract_evidence.py",
        "account.py",
        "risk.py",
        "withdrawal.py",
        "adapters.py",
        "portfolio.py",
        "stress.py",
        "simulation.py",
        "prop_metrics.py",
        "search_bridge.py",
    )
    for name in lifecycle_modules:
        source = (package_dir / name).read_text(encoding="utf-8").lower()
        assert "exact historical" not in source, name
