"""Version-2 comparison plans: variations around a base configuration.

Uses the real verified daily-close study (read only; skipped when absent). Under
the pinned Strategy-Core (no ``exit_policy`` field) only fixed-target variations
resolve; scale-out resolution is exercised when the scale-out Core build is the
imported ``strategy_core``.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from alpha_lab.propsim.funded import comparison_runner
from alpha_lab.propsim.funded.comparison_plan import ComparisonVariantRef
from alpha_lab.propsim.funded.comparison_source import (
    discover_comparison_sources,
    variation_configurations,
    variation_name,
)
from alpha_lab.propsim.funded.comparison_study import build_variation_plan, save_plan
from tests.propsim.funded.pair_builders import BASE, SyntheticDay, flat, run_pair, weekdays

BASE_NAME = "S0_D80_W1_P1"
CORE = {"base_commit": "7c7111e398c083cf8e966e2e0c5aac8a41cc12c0", "branch": "test",
        "patch_sha256": "e" * 64}
T2_H1 = {"tp_r_multiple": "tp_r_multiple.2.0", "htf_timeframes": "htf_timeframes.1H",
         "parent_timeframes": "parent_timeframes.1m-5m-10m-15m-30m",
         "enable_shorts": "enable_shorts.true"}


@pytest.fixture(scope="module")
def source():
    found = [s for s in discover_comparison_sources() if BASE_NAME in s.by_name]
    if not found:
        pytest.skip("the verified daily-close study archive is not available locally")
    return found[0]


def _mini(changes):
    return {"changes": changes, "instrument": "mini", "quantity": 1,
            "cost_per_contract_mills": 5140}


def test_names_and_study_members_are_recognized(source):
    base, other, all_hours = variation_configurations(
        source, BASE_NAME,
        [{}, T2_H1, {"enabled_entry_sessions": "enabled_entry_sessions.all_open_market_v1"}])
    assert base.name == BASE_NAME and base is source.by_name[BASE_NAME]
    assert all_hours.name == "S1_D80_W1_P1"  # also a verified study configuration
    assert other.name == "S0-T2-H1-P5-LS-FX" == variation_name(other.axis_value_ids)
    assert "one-hour gaps only" in other.display_name and "long and short" in other.display_name


def test_plan_records_each_variant_with_its_own_size(source):
    envelope = build_variation_plan(source, BASE_NAME, [_mini({}), _mini(T2_H1)],
                                    firm_keys=["takeprofittrader", "myfundedfutures"],
                                    core_source=CORE)
    variants = {v.name: v for v in envelope.payload.variants}
    assert variants[BASE_NAME].in_verified_study
    assert not variants["S0-T2-H1-P5-LS-FX"].in_verified_study
    assert variants["S0-T2-H1-P5-LS-FX"].cache_configuration == BASE_NAME
    assert {v.cost_per_contract_mills for v in variants.values()} == {5140}
    with pytest.raises(ValueError):
        build_variation_plan(source, BASE_NAME, [_mini({}), _mini({})],
                             firm_keys=["takeprofittrader"], core_source=CORE)


def test_scale_out_variant_needs_an_even_size():
    with pytest.raises(ValueError):
        ComparisonVariantRef(
            name="x", display_name="x", axis_value_ids=(), resolved_section_config_hash="a" * 64,
            exit_policy="scale_out_half_breakeven_hold_to_close_v1", instrument="micro",
            quantity=5, cost_per_contract_mills=514, in_verified_study=False,
            cache_configuration=BASE_NAME)


def _fake(requests):
    days = weekdays(date(2026, 3, 2), 4)

    def run(task):
        plan = task["plan"]
        variant = next(v for v in plan["variants"] if v["name"] == task["configuration"])
        requests.append((variant["name"], variant["instrument"], variant["quantity"],
                         variant["cost_per_contract_mills"]))
        specs = [SyntheticDay(d, [[BASE], [BASE + 300, BASE + 600], *flat(4)],
                              {0: ("long", BASE - 100, BASE + 600)}) for d in days]
        pairs, resumed = {}, {}
        for profile in plan["firm_profiles"]:
            key = profile["firm_key"]
            pair_id = f"{variant['name']}|{key}"
            run_, ledger, _ = run_pair(specs, firm_key=key, pair_id=pair_id,
                                       quantity=variant["quantity"])
            pairs[key] = {"pair_id": pair_id, "forced_flat": 0, "trades_not_in_reference": 0,
                          "ledger": json.loads(json.dumps(ledger.snapshot()))}
            resumed[key] = {"identical": True}
        return {"configuration": variant["name"], "display_name": variant["display_name"],
                "axes": {}, "settings_plain": [], "resumed": resumed, "pairs": pairs,
                "reference": {"equivalent": True, "saved_study_trades": 0,
                              "replayed_trades": 0},
                "prints": {"minutes_checked": 0, "minutes_rebuilt_exactly": 0,
                           "missing_utc_days": [], "files": []},
                "sizing": {"instrument": variant["instrument"], "instrument_label": "x",
                           "quantity": variant["quantity"], "tick_value_cents": 500,
                           "cost_per_contract_mills": variant["cost_per_contract_mills"]},
                "exit_policy": variant["exit_policy"], "strategy_trades_no_account": []}

    return run


def test_worker_runs_every_variant_with_its_size_and_checks_the_core(source, tmp_path,
                                                                    monkeypatch):
    envelope = build_variation_plan(source, BASE_NAME, [_mini({}), _mini(T2_H1)],
                                    firm_keys=["takeprofittrader", "myfundedfutures"],
                                    core_source=CORE, purpose="engineering_sample")
    store = tmp_path / "store"
    plan_id = save_plan(store, envelope)
    # a different Core source is refused before anything runs
    monkeypatch.setattr("alpha_lab.propsim.funded.core_identity.core_source_identity",
                        lambda: {**CORE, "patch_sha256": "f" * 64, "root": "x"})
    with pytest.raises(PermissionError):
        comparison_runner.run_comparison_plan(plan_id=plan_id, store_root=store,
                                              state_root=tmp_path / "s1",
                                              reports_root=tmp_path / "r",
                                              run_config_fn=lambda t: pytest.fail("ran"))
    monkeypatch.setattr("alpha_lab.propsim.funded.core_identity.core_source_identity",
                        lambda: {**CORE, "root": "x"})
    requests: list = []
    final = comparison_runner.run_comparison_plan(
        plan_id=plan_id, store_root=store, state_root=tmp_path / "s2",
        reports_root=tmp_path / "r", run_config_fn=_fake(requests))
    assert final["status"] == "Completed", final
    assert sorted(r[0] for r in requests) == ["S0-T2-H1-P5-LS-FX", BASE_NAME]
    result = comparison_runner.load_comparison_result(store, final["result_id"])
    assert set(result["settings"]["sizing_by_configuration"]) == {BASE_NAME,
                                                                   "S0-T2-H1-P5-LS-FX"}
    assert "per trade" in result["settings"]["size_text"]
    assert len(result["tables"]["pair_results"]) == 4



def test_scale_out_variant_must_take_its_half_at_one_r():
    with pytest.raises(ValueError, match="1R"):
        ComparisonVariantRef(
            name="x", display_name="x",
            axis_value_ids=(("tp_r_multiple", "tp_r_multiple.2.0"),),
            resolved_section_config_hash="a" * 64,
            exit_policy="scale_out_half_breakeven_hold_to_close_v1", instrument="micro",
            quantity=10, cost_per_contract_mills=514, in_verified_study=False,
            cache_configuration=BASE_NAME)


def test_a_failing_variant_outside_the_study_is_listed_not_completed(source, tmp_path,
                                                                     monkeypatch):
    # review round 2, B1: one failure must not void the other configurations
    envelope = build_variation_plan(source, BASE_NAME, [_mini({}), _mini(T2_H1)],
                                    firm_keys=["takeprofittrader"], core_source=CORE,
                                    purpose="engineering_sample")
    store = tmp_path / "store"
    plan_id = save_plan(store, envelope)
    monkeypatch.setattr("alpha_lab.propsim.funded.core_identity.core_source_identity",
                        lambda: {**CORE, "root": "x"})
    good = _fake([])

    def run(task):
        if task["configuration"] != BASE_NAME:
            raise RuntimeError("simulated failure")
        return good(task)

    final = comparison_runner.run_comparison_plan(
        plan_id=plan_id, store_root=store, state_root=tmp_path / "s",
        reports_root=tmp_path / "r", run_config_fn=run)
    assert final["status"] == "Incomplete", final
    result = comparison_runner.load_comparison_result(store, final["result_id"])
    rows = {r["configuration"]: r for r in result["tables"]["pair_results"]}
    assert rows["S0-T2-H1-P5-LS-FX"]["status"] == "Not completed"
    assert "one-hour gaps only" in rows["S0-T2-H1-P5-LS-FX"]["configuration_label"]
    assert rows[BASE_NAME]["status"] == "Completed"
