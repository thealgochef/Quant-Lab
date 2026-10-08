"""Synthetic saved reporting checks; no market readers or engines are invoked."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from alpha_lab.propsim.funded.comparison_result import strategy_metrics
from alpha_lab.propsim.funded.full_range_reporting import (
    _ordinary_trade,
    attach_full_range_reports,
    validate_full_range_reports,
)

NAMES = tuple(f"C{index:02d}" for index in range(1, 7))
FIRMS = ("takeprofittrader", "myfundedfutures")


def reporting_fixture(*, failed=False):
    scope = json.loads((Path(__file__).parents[3] /
                        "docs/ifsm-correct-config-full-range-v01/date_scope.json").read_text())
    dates, warmup = tuple(scope["evaluation_dates"]), tuple(scope["warmup_dates"])
    outputs, failures = [], []
    result = {"tables": {"trades": [], "pair_results": [], "cash_ledger": [],
                          "strategy_metrics": [], "account_events": []},
              "summaries_cents": {}}
    context = {"entry_session": "ny", "regime": "unknown", "slot_chicago": "s2_1000_1200",
               "context_available": False, "gate_status": "context_unavailable",
               "levels_source_eod_date": "2025-06-13", "selected_instrument_id": 1001,
               "roll_flag": False}
    for index, name in enumerate(NAMES):
        partial = index < 2
        size = {"instrument": "micro" if partial else "mini", "quantity": 10 if partial else 1,
                "tick_value_cents": 50 if partial else 500,
                "cost_per_contract_mills": 514 if partial else 5140}
        broken = failed and name == NAMES[-1]
        if broken:
            failures.append({"configuration": name, "reason": "Synthetic input unavailable"})
        else:
            ordinary = []
            for num, day in enumerate((warmup[0], dates[1], dates[5])):
                ordinary.append({
                    "trade_id": f"{name}-ordinary-{num}", "direction": "long",
                    "entry_ticks": 80000, "stop_ticks": 79992, "target_ticks": 80008,
                    "exit_ticks": 80000 if partial else 80008,
                    "scale_out_ticks": 80008 if partial else None,
                    "risk_ticks": 8, "realized_r": .5 if partial else 1.,
                    "entry_ts_utc": f"{day}T15:00:00Z",
                    "resolution_ts_utc": f"{day}T15:10:00Z",
                    "is_warmup": num == 0, "entry_context": copy.deepcopy(context),
                })
            outputs.append({"configuration": name, "display_name": name, "sizing": size,
                            "strategy_trades_no_account": ordinary,
                            "section_config_hash": name + "-synthetic"})
        for firm in FIRMS:
            pair = f"{name}|{firm}"
            cost = 10200 if firm == FIRMS[0] else 12500
            summary = {"configuration": name, "firm_key": firm,
                       "status": "Not completed" if broken else "Completed"}
            if not broken:
                summary.update(net_cash_earned_cents=-cost, account_costs_cents=cost,
                               payouts_received_cents=0)
                result["tables"]["cash_ledger"].append({
                    "pair_id": pair, "configuration": name, "firm_key": firm,
                    "kind": "account_purchase", "amount_usd": cost / 100,
                    "ts_utc": "2025-06-15T22:00:00Z",
                })
                gross = 2000 if partial else 4000
                result["tables"]["trades"].append({
                    "pair_id": pair, "configuration": name, "firm_key": firm,
                    "strategy_trade_id": f"{pair}-funded-0", "seq": 1,
                    "entry_utc": f"{dates[3]}T15:00:00Z",
                    "exit_utc": f"{dates[3]}T15:10:00Z",
                    "quantity": size["quantity"], "gross_pnl_usd": gross / 100,
                    "costs_usd": 10.28, "net_pnl_usd": (gross - 1028) / 100,
                    "entry_context": {**context, "regime": "negative"},
                })
            result["summaries_cents"][pair] = summary
            result["tables"]["pair_results"].append({"pair_id": pair, **summary})
    coverage = {day: {"source_status": "available", "source_selected_instrument_id": 1001,
                      "roll_flag": day == dates[3]} for day in dates}
    kwargs = dict(configuration_names=NAMES, evaluation_dates=dates, warmup_dates=warmup,
                  cutoff_utc=scope["cutoff_utc"], coverage_by_day=coverage)
    return result, outputs, failures, kwargs


def test_full_scope_streams_account_money_actual_entries_and_weighted_partial():
    result, outputs, failures, kwargs = reporting_fixture()
    attach_full_range_reports(result, outputs, failures, **kwargs)
    tables = result["tables"]
    assert len(tables["strategy_metrics"]) == 6
    assert len(tables["pair_results"]) == 12
    assert len(tables["daily_activity"]) == 18 * 253
    assert len(tables["roll_days"]) == 18 * 253
    partial = next(row for row in tables["strategy_trades"]
                   if row["configuration"] == "C01" and not row["is_warmup"])
    assert partial["quantity"] == 10
    assert partial["scale_out_quantity"] == partial["final_exit_quantity"] == 5
    assert partial["gross_pnl_cents"] == 2000
    assert partial["costs_cents"] == 1028
    assert partial["net_pnl_cents"] == 972
    assert partial["quantity_weighted_gross_points"] == 1.
    strategy_days = [row for row in tables["daily_activity"]
                     if row["configuration"] == "C01" and row["stream"] == "strategy"]
    funded_days = [row for row in tables["daily_activity"]
                   if row["configuration"] == "C01" and row["firm_key"] == FIRMS[0]]
    assert sum(row["actual_entries"] for row in strategy_days) == 2
    assert sum(row["actual_entries"] for row in funded_days) == 1
    assert all(row["stage_count_status"] == "unavailable" for row in strategy_days)
    assert not any(row["evaluation_date"] in kwargs["warmup_dates"] for row in strategy_days)
    assert all(row["funded_cash_allocated"] is False for row in tables["context_breakdowns"])
    actual_group = next(row for row in tables["context_breakdowns"]
                        if row["configuration"] == "C01" and row["firm_key"] == FIRMS[0]
                        and row["dimension"] == "regime")
    assert actual_group["bucket"] == "negative"
    assert validate_full_range_reports(result)["passed"]


def test_six_micro_ordinary_trade_posts_each_rounded_fill_fee():
    _result, outputs, _failures, _kwargs = reporting_fixture()
    output = copy.deepcopy(outputs[0])
    raw = output["strategy_trades_no_account"][1]
    output["sizing"]["quantity"] = 6
    partial = _ordinary_trade(raw, output)
    assert (partial["quantity"], partial["scale_out_quantity"],
            partial["final_exit_quantity"]) == (6, 3, 3)
    assert partial["costs_cents"] == 308 + 154 + 154 == 616
    assert partial["gross_pnl_cents"] - partial["net_pnl_cents"] == 616

    # QG can select six even when the output's default size is ten.
    output["sizing"]["quantity"] = 10
    raw["quantity"] = 6
    assert _ordinary_trade(raw, output)["costs_cents"] == 616
    raw["quantity"] = 10
    assert _ordinary_trade(raw, output)["costs_cents"] == 1028


def test_strategy_metrics_price_six_and_ten_micro_fills_individually():
    _result, outputs, _failures, _kwargs = reporting_fixture()
    output = copy.deepcopy(outputs[0])
    six = output["strategy_trades_no_account"][1]
    six.update(quantity=6, risk_ticks=1, realized_r=2.5)
    output["strategy_trades_no_account"] = [six]
    six_metrics = strategy_metrics(output)
    assert six_metrics["round_trip_cost_points_per_contract"] == .513
    assert six_metrics["net_r_after_costs"] == .45

    ten = copy.deepcopy(six)
    ten.update(quantity=10, trade_id="ten", resolution_ts_utc="2025-06-18T15:10:00Z")
    output["strategy_trades_no_account"] = [ten]
    ten_metrics = strategy_metrics(output)
    assert ten_metrics["round_trip_cost_points_per_contract"] == .514
    assert ten_metrics["net_r_after_costs"] == .44

    output["strategy_trades_no_account"] = [six, ten]
    mixed_metrics = strategy_metrics(output)
    assert mixed_metrics["round_trip_cost_points_per_contract"] is None
    assert mixed_metrics["net_r_after_costs"] == .89


def test_leading_trailing_and_tied_maximal_spans_are_saved():
    result, outputs, failures, kwargs = reporting_fixture()
    # Four entries delimit equal one-date leading/middle gaps; trailing wins overall.
    attach_full_range_reports(result, outputs, failures, **kwargs)
    spans = [row for row in result["tables"]["no_entry_spans"]
             if row["configuration"] == "C01" and row["stream"] == "strategy"]
    assert len(spans) == 3
    assert any(row["leading_censored"] for row in spans)
    assert any(row["trailing_censored"] for row in spans)
    assert sum(row["evaluation_dates_without_entry"] for row in spans) == 251
    assert spans[0]["is_tied_longest"]
    tied_result, tied_outputs, tied_failures, tied_kwargs = reporting_fixture()
    one = tied_outputs[0]["strategy_trades_no_account"][1]
    one["entry_ts_utc"] = f"{kwargs['evaluation_dates'][126]}T15:00:00Z"
    one["resolution_ts_utc"] = f"{kwargs['evaluation_dates'][126]}T15:10:00Z"
    tied_outputs[0]["strategy_trades_no_account"] = [one]
    attach_full_range_reports(tied_result, tied_outputs, tied_failures, **tied_kwargs)
    longest = [row for row in tied_result["tables"]["no_entry_spans"]
               if row["configuration"] == "C01" and row["stream"] == "strategy"
               and row["is_tied_longest"]]
    assert len(longest) == 2
    assert {row["evaluation_dates_without_entry"] for row in longest} == {126}


def test_failed_rows_and_unavailable_input_are_never_zero_activity():
    result, outputs, failures, kwargs = reporting_fixture(failed=True)
    missing = kwargs["evaluation_dates"][10]
    kwargs["coverage_by_day"][missing]["source_status"] = "unavailable"
    attach_full_range_reports(result, outputs, failures, **kwargs)
    metrics = next(row for row in result["tables"]["strategy_metrics"]
                   if row["configuration"] == "C06")
    assert metrics["status"] == "Not completed"
    assert "profit_after_costs_cents" not in metrics
    unknown = [row for row in result["tables"]["daily_activity"]
               if row["configuration"] == "C06" or row["evaluation_date"] == missing]
    assert all(row["actual_entries"] is None for row in unknown)
    assert len(result["tables"]["pair_results"]) == 12
    assert not result["tables"]["no_entry_spans"]  # complete gaps cannot cross unavailable dates


def test_tampered_activity_cash_and_group_outcomes_fail_reconciliation():
    result, outputs, failures, kwargs = reporting_fixture()
    attach_full_range_reports(result, outputs, failures, **kwargs)
    result["tables"]["daily_activity"][0]["actual_entries"] = 99
    assert not validate_full_range_reports(result)["passed"]
    result, outputs, failures, kwargs = reporting_fixture()
    result["tables"]["trades"][0]["net_pnl_usd"] += .01
    with pytest.raises(ValueError, match="funded saved trade money"):
        attach_full_range_reports(result, outputs, failures, **kwargs)
    result, outputs, failures, kwargs = reporting_fixture()
    outputs[0]["strategy_trades_no_account"][0]["realized_r"] = 99.
    with pytest.raises(ValueError, match="weighted fills differ"):
        attach_full_range_reports(result, outputs, failures, **kwargs)


def test_recorded_partial_timestamp_is_preserved_and_not_scaled_out_is_explicit():
    result, outputs, failures, kwargs = reporting_fixture()
    raw = outputs[0]["strategy_trades_no_account"][1]
    raw["scale_out_ts_utc"] = raw["entry_ts_utc"].replace("15:00", "15:05")
    attach_full_range_reports(result, outputs, failures, **kwargs)
    trade = next(row for row in result["tables"]["strategy_trades"]
                 if row["trade_id"] == raw["trade_id"])
    assert trade["scale_out_timestamp_status"] == "available"
    assert trade["scale_out_utc"] == "2025-06-17T15:05:00+00:00"
    assert "10:05 AM" in trade["scale_out_chicago"]
    whole = next(row for row in result["tables"]["strategy_trades"]
                 if row["configuration"] == "C03")
    assert whole["scale_out_timestamp_status"] == "not_scaled_out"


def test_duplicate_worker_outputs_are_refused_before_reporting():
    result, outputs, failures, kwargs = reporting_fixture()
    outputs.append(copy.deepcopy(outputs[0]))
    with pytest.raises(ValueError, match="exactly one complete or failed output"):
        attach_full_range_reports(result, outputs, failures, **kwargs)


def test_one_firm_matrix_retains_independent_failed_and_completed_rows():
    result, outputs, failures, kwargs = reporting_fixture(failed=True)
    keep = {NAMES[0], NAMES[-1]}
    result["tables"]["trades"] = [
        row for row in result["tables"]["trades"]
        if row["configuration"] in keep and row["firm_key"] == "myfundedfutures"
    ]
    result["tables"]["pair_results"] = [
        row for row in result["tables"]["pair_results"]
        if row["configuration"] in keep and row["firm_key"] == "myfundedfutures"
    ]
    result["tables"]["cash_ledger"] = [
        row for row in result["tables"]["cash_ledger"]
        if row["configuration"] in keep and row["firm_key"] == "myfundedfutures"
    ]
    result["summaries_cents"] = {
        key: value for key, value in result["summaries_cents"].items()
        if value["configuration"] in keep and value["firm_key"] == "myfundedfutures"
    }
    kwargs["configuration_names"] = (NAMES[0], NAMES[-1])
    attach_full_range_reports(
        result, [row for row in outputs if row["configuration"] in keep],
        failures, **kwargs, expected_configuration_count=2, expected_firm_count=1,
    )
    assert result["full_range_reporting"]["schema"] == "ifsm_mffu_context_batch_reporting_v1"
    assert len(result["tables"]["pair_results"]) == 2
    assert result["tables"]["pair_results"][1]["status"] == "Not completed"
    assert validate_full_range_reports(result)["passed"]


def _historical_mffu_price_fixture():
    result, outputs, failures, kwargs = reporting_fixture()
    old, new = NAMES[0], "MCB001"
    outputs[0]["configuration"] = outputs[0]["display_name"] = new
    for rows in result["tables"].values():
        for row in rows:
            if row.get("configuration") == old:
                row["configuration"] = new
                row["pair_id"] = row["pair_id"].replace(old + "|", new + "|", 1)
    result["summaries_cents"] = {
        (key.replace(old + "|", new + "|", 1) if value["configuration"] == old else key):
        ({**value, "configuration": new} if value["configuration"] == old else value)
        for key, value in result["summaries_cents"].items()
    }
    for row in result["tables"]["trades"]:
        if row["configuration"] == new:
            row["initial_risk_cents"] = 10000
            row["price_excursion_status"] = "unavailable_legacy_reuse"
    kwargs["configuration_names"] = (new, *NAMES[1:])
    return result, outputs, failures, kwargs


@pytest.mark.parametrize("status", ["reused_nonimpact", "reused_equivalent_after_verification"])
def test_correction_reuse_retains_explicit_historical_price_unavailability(status):
    result, outputs, failures, kwargs = _historical_mffu_price_fixture()
    outputs[0]["correction_reuse"] = {"status": status, "original_result_id": "a" * 64}
    outputs[0]["historical_reuse"] = {"status": "compatible_reused", "reference_batch_id": "C02"}
    before = copy.deepcopy(outputs)
    attach_full_range_reports(result, outputs, failures, **kwargs)
    historical = [row for row in result["tables"]["trades"] if row["configuration"] == "MCB001"]
    assert len(historical) == 2
    assert all(row["price_excursion_status"] == "unavailable_legacy_reuse" for row in historical)
    assert all(row["price_min_ticks"] is None for row in historical)
    assert outputs == before
    assert validate_full_range_reports(result)["passed"]


@pytest.mark.parametrize("status", [None, "replayed_changed", "replayed_equal"])
def test_genuine_new_mffu_trade_still_rejects_missing_price_path(status):
    result, outputs, failures, kwargs = _historical_mffu_price_fixture()
    if status is not None:
        outputs[0]["correction_reuse"] = {"status": status}
    with pytest.raises(ValueError, match="new MFFU funded trade lacks a complete price path"):
        attach_full_range_reports(result, outputs, failures, **kwargs)
