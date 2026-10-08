"""Synthetic cash and timing checks for the saved 64-intent analysis."""

from __future__ import annotations

import json
from itertools import product

import pytest

from alpha_lab.propsim.funded.mffu_batch_analysis import analyze_mffu_batch


def _variants() -> list[dict]:
    rows = []

    def add(family: str, schedule: str, cap: str, context: str, exit_policy: str,
            sizing: str = "Q10", overhead: str = "O0", geometry: str = "G0") -> None:
        rows.append({
            "variant_id": f"MCB{len(rows) + 1:03d}", "family": family,
            "schedule": schedule, "daily_cap": cap, "entry_context": context,
            "exit": exit_policy, "sizing": sizing, "overhead": overhead,
            "geometry": geometry,
        })

    for schedule, cap, context, exit_policy in product(
        ("S0", "S1"), ("U", "D1"), ("F0", "FE", "FL", "FEL"), ("XP", "XF", "XG")
    ):
        add("core_48", schedule, cap, context, exit_policy)
    for schedule, sizing in product(("S0", "S1"), ("Q6", "QG")):
        add("sizing", schedule, "U", "F0", "XP", sizing=sizing)
    for schedule, cap in product(("S0", "S1"), ("U", "D1")):
        add("overhead", schedule, cap, "F0", "XP", overhead="O8")
    for schedule, geometry in product(("S0", "S1"), ("G05", "G075", "G10")):
        add("geometry", schedule, "U", "F0", "XP", geometry=geometry)
    for schedule in ("S0", "S1"):
        add("state_exit", schedule, "U", "F0", "XE")
    assert len(rows) == 64
    return rows


def _cash(row: dict) -> int:
    schedule = row["schedule"] == "S1"
    cap = row["daily_cap"] == "D1"
    fe = row["entry_context"] in {"FE", "FEL"}
    fl = row["entry_context"] in {"FL", "FEL"}
    exit_policy = row["exit"]
    return (
        100_000 + 10_000 * schedule + 5_000 * cap + 2_000 * fe + 3_000 * fl
        + 700 * fe * fl + {"XP": 0, "XF": 1_000, "XG": 2_000, "XE": 0}[exit_policy]
        + cap * {"XP": 0, "XF": 400, "XG": 600, "XE": 0}[exit_policy]
        + 900 * schedule * cap
    )


def _result(variants: list[dict]) -> dict:
    summaries = {}
    for row in variants:
        cash = _cash(row)
        variant_id = row["variant_id"]
        summaries[f"{variant_id}|myfundedfutures"] = {
            "configuration": variant_id, "firm_key": "myfundedfutures",
            "status": "Completed", "net_cash_earned_cents": cash,
            "payouts_received_cents": cash + 12_500,
            "account_costs_cents": 12_500,
            "payouts_received_count": 1, "trades_taken": 0,
        }
    return {
        "period": {"start_utc": "2025-06-15T22:00:00Z",
                   "cutoff_utc": "2026-06-10T21:00:00Z"},
        "summaries_cents": summaries, "tables": {},
    }


def _analyze(result: dict, variants: list[dict], pairs: list[dict] | None = None) -> dict:
    return analyze_mffu_batch(
        result, variants=variants, comparison_pairs=pairs or [],
        evaluation_dates=["2025-06-16", "2025-07-18", "2026-01-05", "2026-06-10"],
        expected_pair_count=len(pairs or []),
    )


def test_matched_cash_and_declared_interaction_deltas() -> None:
    variants = _variants()
    output = _analyze(
        _result(variants), variants,
        [{"base_id": "MCB001", "challenger_id": "MCB002",
          "changed_axis": "exit", "group": "core_matched"}],
    )
    assert output["variant_count"] == output["completed_count"] == 64
    assert output["matched_pairs"][0]["delta_cents"] == 1_000
    assert {cell["delta_cents"] for cell in output["interactions"]["fe_x_fl"]["cells"]} == {700}
    cap_exit = output["interactions"]["cap_x_exit_vs_xp"]["cells"]
    assert {cell["delta_cents"] for cell in cap_exit
            if cell["challenger_exit"] == "XF"} == {400}
    assert {cell["delta_cents"] for cell in cap_exit
            if cell["challenger_exit"] == "XG"} == {600}
    schedule_cap = output["interactions"]["schedule_x_cap"]["cells"]
    assert {cell["delta_cents"] for cell in schedule_cap} == {900}


def test_failed_financial_row_is_unavailable_not_zero() -> None:
    variants = _variants()
    result = _result(variants)
    result["summaries_cents"]["MCB001|myfundedfutures"] = {
        "configuration": "MCB001", "firm_key": "myfundedfutures",
        "status": "Not completed", "reason": "source_mismatch",
    }
    output = _analyze(
        result, variants,
        [{"base_id": "MCB001", "challenger_id": "MCB002"}],
    )
    assert output["completed_count"] == 63
    assert output["variants"][0]["net_cash_cents"] is None
    assert output["variants"][0]["reason"] == "source_mismatch"
    assert output["matched_pairs"][0]["status"] == "unavailable"
    assert output["matched_pairs"][0]["delta_cents"] is None
    assert "MCB001" in output["matched_pairs"][0]["unavailable_ids"]
    assert any(cell["status"] == "unavailable"
               for cell in output["interactions"]["fe_x_fl"]["cells"])
    assert all(item["variant_id"] != "MCB001" for item in output["ranking"])


def test_waiting_cash_year_month_and_trade_concentration() -> None:
    variants = _variants()
    result = _result(variants)
    first = result["summaries_cents"]["MCB001|myfundedfutures"]
    first.update(payouts_received_cents=112_500, account_costs_cents=12_500,
                 net_cash_earned_cents=100_000, payouts_received_count=2,
                 trades_taken=2)
    second = result["summaries_cents"]["MCB002|myfundedfutures"]
    second.update(payouts_received_cents=0, account_costs_cents=12_500,
                  net_cash_earned_cents=-12_500, payouts_received_count=0)
    result["tables"] = {
        "payout_events": [
            {"configuration": "MCB001", "firm_key": "myfundedfutures",
             "event": "received", "received_utc": "2025-07-18T21:00:00Z"},
            {"configuration": "MCB001", "firm_key": "myfundedfutures",
             "event": "received", "received_utc": "2026-01-05T22:00:00Z"},
        ],
        "cash_ledger": [
            {"configuration": "MCB001", "firm_key": "myfundedfutures",
             "kind": "account_purchase", "ts_utc": "2025-06-15T22:00:00Z",
             "amount_usd": "125.00"},
            {"configuration": "MCB001", "firm_key": "myfundedfutures",
             "kind": "payout_received", "ts_utc": "2025-07-18T21:00:00Z",
             "amount_usd": "500.00"},
            {"configuration": "MCB001", "firm_key": "myfundedfutures",
             "kind": "payout_received", "ts_utc": "2026-01-05T22:00:00Z",
             "amount_usd": "625.00"},
        ],
        "trades": [
            {"configuration": "MCB001", "firm_key": "myfundedfutures",
             "entry_utc": "2025-07-17T12:00:00Z", "entry_session": "asia",
             "scale_out_quantity": 5, "exit_kind": "scheduled_close",
             "net_pnl_cents": 100_000, "trade_ref": "runner"},
            {"configuration": "MCB001", "firm_key": "myfundedfutures",
             "entry_utc": "2026-01-04T23:00:00Z", "entry_session": "ny",
             "scale_out_quantity": 0, "exit_kind": "target",
             "net_pnl_cents": -1_000, "trade_ref": "small"},
        ],
    }
    output = _analyze(result, variants)
    wait = output["waiting_by_variant"]["MCB001"]
    assert wait["initial"]["calendar_days"] == 33
    assert wait["initial"]["evaluated_trading_days"] == 2
    assert wait["max_between_receipts"]["start_chicago_date"] == "2025-07-18"
    assert wait["max_no_receipt_interval"]["kind"] == "between_receipts"
    assert wait["terminal"]["censored_end"] is True
    no_receipt = output["waiting_by_variant"]["MCB002"]
    assert no_receipt["initial"]["censored_end"] is True
    assert no_receipt["max_no_receipt_interval"]["end_chicago_date"] == "2026-06-10"
    cash = output["cash_concentration_by_variant"]["MCB001"]
    assert cash["yearly"]["2025"]["net_cash_cents"] == 37_500
    assert cash["yearly"]["2026"]["net_cash_cents"] == 62_500
    assert "2025-08" in cash["zero_receipt_months"]
    assert output["cash_concentration_by_variant"]["MCB002"]["status"] == "unavailable"
    trades = output["trade_concentration_by_variant"]["MCB001"]
    assert trades["by_entry_session"]["asia"]["net_trade_pnl_cents"] == 100_000
    assert trades["by_resolution"]["partial_to_scheduled_close"]["trades"] == 1
    assert trades["largest_trades"][0]["trade_id"] == "runner"
    assert trades["basis"] == "actual_funded_trade_net_pnl_not_received_cash"


def test_inconsistent_cash_ledger_is_a_validation_error() -> None:
    variants = _variants()
    result = _result(variants)
    result["tables"] = {"cash_ledger": [
        {"configuration": "MCB001", "firm_key": "myfundedfutures",
         "kind": "account_purchase", "ts_utc": "2025-06-15T22:00:00Z",
         "amount_usd": "125.00"},
    ]}
    with pytest.raises(ValueError, match="does not reconcile"):
        _analyze(result, variants)


def test_complete_matrix_and_pair_identity_are_required() -> None:
    variants = _variants()
    with pytest.raises(ValueError, match="MCB001..MCB064"):
        _analyze(_result(variants), variants[:-1])
    with pytest.raises(ValueError, match="duplicate directed"):
        _analyze(_result(variants), variants, [
            {"base_id": "MCB001", "challenger_id": "MCB002"},
            {"base_id": "MCB001", "challenger_id": "MCB002"},
        ])


def test_frozen_plan_intent_json_projects_without_extra_dependencies() -> None:
    variants = _variants()

    class Ref:
        def __init__(self, row: dict):
            self.row = row

        def model_dump(self) -> dict:
            return {"variant_id": self.row["variant_id"],
                    "intent_json": json.dumps(self.row)}

    output = _analyze(_result(variants), [Ref(row) for row in variants])
    assert output["variant_count"] == 64
    json.dumps(output)


def _trade(configuration: str, day: str, hour: int, *, pnl: int,
           stop: int = 79_900, target: int = 80_100,
           exit_ticks: int = 80_100, exit_kind: str = "target",
           partial: int = 0) -> dict:
    return {
        "configuration": configuration, "firm_key": "myfundedfutures",
        "entry_utc": f"{day}T{hour:02d}:00:00Z", "entry_trading_day": day,
        "entry_ticks": 80_000, "direction": "long", "stop_ticks": stop,
        "target_ticks": target, "exit_ticks": exit_ticks, "exit_kind": exit_kind,
        "scale_out_quantity": partial, "net_pnl_cents": pnl,
        "trade_ref": f"{configuration}-{day}-{hour}",
    }


def test_matched_funded_entry_gains_losses_and_changed_exits_remain_trade_diagnostics():
    variants = _variants()
    result = _result(variants)
    for key in ("MCB001", "MCB002"):
        result["summaries_cents"][f"{key}|myfundedfutures"]["trades_taken"] = 2
    result["tables"]["trades"] = [
        _trade("MCB001", "2025-06-16", 15, pnl=5_000),
        _trade("MCB001", "2025-06-17", 15, pnl=-2_000, exit_ticks=79_900,
               exit_kind="stop"),
        _trade("MCB002", "2025-06-16", 15, pnl=7_000, stop=79_800,
               target=80_150, exit_ticks=80_125, exit_kind="scheduled_close"),
        _trade("MCB002", "2025-06-18", 15, pnl=3_000),
    ]
    for trade in result["tables"]["trades"]:
        trade["account_id"] = trade["configuration"] + "#1"
        trade["quantity"] = 10
        trade["final_exit_quantity"] = 10
        trade["costs_cents"] = 1_028
    output = _analyze(result, variants, [
        {"base_id": "MCB001", "challenger_id": "MCB002"},
    ])
    pair = output["matched_pairs"][0]
    assert pair["delta_cents"] == 1_000  # payout cash is a separate account outcome
    diagnostic = pair["trade_diagnostics"]
    assert diagnostic["basis"] == "actual_funded_trade_net_pnl_not_received_cash"
    assert (diagnostic["common_entries"], diagnostic["gained_entries"],
            diagnostic["lost_entries"]) == (1, 1, 1)
    assert diagnostic["common_entry_pnl_delta_cents"] == 2_000
    assert diagnostic["gained_entry_pnl_cents"] == 3_000
    assert diagnostic["lost_entry_pnl_cents"] == -2_000
    changed = diagnostic["changed_common"][0]
    assert {"stop_ticks", "target_ticks", "exit_ticks", "exit_kind"} <= set(
        changed["changed_fields"]
    )
    assert changed["base"]["stop_ticks"] == 79_900
    assert changed["challenger"]["stop_ticks"] == 79_800
    assert changed["base"]["account_id"] == "MCB001#1"
    assert changed["challenger"]["account_id"] == "MCB002#1"
    assert changed["base"]["trading_costs_cents"] == 1_028
    assert diagnostic["gained"][0]["trade_ref"].startswith("MCB002")
    assert diagnostic["lost"][0]["trade_ref"].startswith("MCB001")
    assert diagnostic["gained"][0]["account_id"] == "MCB002#1"
    assert diagnostic["lost"][0]["quantity"] == 10


def test_first_later_entry_and_partial_runner_resolution_use_funded_trades():
    variants = _variants()
    result = _result(variants)
    result["summaries_cents"]["MCB001|myfundedfutures"]["trades_taken"] = 3
    result["tables"]["trades"] = [
        _trade("MCB001", "2025-06-16", 15, pnl=1_000,
               exit_kind="scheduled_close", partial=5),
        _trade("MCB001", "2025-06-16", 16, pnl=-200,
               exit_kind="breakeven_stop", partial=5),
        _trade("MCB001", "2025-06-17", 15, pnl=300),
    ]
    output = _analyze(result, variants)
    breakdown = output["entry_order_runner_by_variant"]["MCB001"]
    first, later = breakdown["by_entry_order"]["first"], breakdown["by_entry_order"]["later"]
    assert (first["trades"], first["net_trade_pnl_cents"]) == (2, 1_300)
    assert (later["trades"], later["net_trade_pnl_cents"]) == (1, -200)
    assert first["by_resolution"]["partial_to_daily_close"]["trades"] == 1
    assert first["by_resolution"]["no_partial"]["trades"] == 1
    assert later["by_resolution"]["partial_to_entry_stop"]["trades"] == 1


def test_decision_context_coverage_and_reused_event_unavailability():
    variants = _variants()
    result = _result(variants)
    result["mffu_batch"] = {
        "dispositions": [
            {"variant_id": "MCB001", "status": "compatible_reused"},
            {"variant_id": "MCB002", "status": "newly_completed"},
        ],
        "decision_context": [
            {"configuration": "MCB001", "stream": "funded",
             "event": "entry_annotation", "action": "annotated",
             "context": {"gamma_status": "available"}},
            {"configuration": "MCB002", "stream": "funded",
             "event": "entry_admission", "action": "reject",
             "reasons": ["ifsm_early_positive", "daily_execution_cap"],
             "quota_before": 1, "quota_after": 1,
             "context": {"gamma_status": "stale", "levels_status": "available",
                         "gamma_sign": "unknown",
                         "decision_ts_utc": "2025-06-17T15:00:00+00:00",
                         "gamma_eligible_from_utc": "2025-06-13T03:00:00+00:00",
                         "gamma_report_date": "2025-06-12",
                         "level_eligible_from_utc": "2025-06-17T03:00:00+00:00",
                         "level_requested_date": "2025-06-16",
                         "level_report_date": "2025-06-16"}},
            {"configuration": "MCB002", "stream": "funded",
             "event": "parent_lock_distance", "action": "freeze_distance",
             "fallback_reason": "missing_or_stale_bands",
             "context": {"gamma_status": "available", "levels_status": "stale",
                         "gamma_sign": "positive"}},
            {"configuration": "MCB002", "stream": "funded",
             "event": "entry_admission", "action": "admit",
             "reasons": [], "quota_before": 0, "quota_after": 1,
             "context": {"gamma_status": "available", "levels_status": "available",
                         "gamma_sign": "positive", "positive_run_age": 6,
                         "positive_run_age_is_lower_bound": True,
                         "decision_ts_utc": "2025-06-17T15:00:00+00:00",
                         "gamma_eligible_from_utc": "2025-06-17T03:00:00+00:00",
                         "gamma_report_date": "2025-06-16",
                         "level_eligible_from_utc": "2025-06-17T03:00:00+00:00",
                         "level_requested_date": "2025-06-16",
                         "level_report_date": "2025-06-13"}},
        ],
    }
    output = _analyze(result, variants)
    coverage = output["decision_context_coverage_by_variant"]
    assert coverage["MCB001"] == {
        "status": "unavailable", "reason": "historical_policy_events_unavailable",
    }
    funded = coverage["MCB002"]["streams"]["funded"]
    assert funded["events"]["entry_admission"] == 2
    assert funded["actions"] == {"reject": 1, "freeze_distance": 1, "admit": 1}
    assert funded["rejection_reasons"]["daily_execution_cap"] == 1
    assert funded["gamma_status"]["stale"] == 1
    assert funded["levels_status"]["stale"] == 1
    assert funded["geometry_fallback_reasons"]["missing_or_stale_bands"] == 1
    assert funded["quota_increments"] == 1
    assert funded["gamma_nominal_timing"]["eligible_by_nominal_schedule"] == 2
    assert funded["gamma_report_carry"]["carried_prior_report"] == 1
    assert funded["level_report_carry"]["carried_prior_report"] == 1
    assert funded["gamma_positive_phase"]["established_positive"] == 1
    assert funded["positive_age_bound"]["lower_bound"] == 1
    assert funded["actions_by_context_class"]["gamma_status:stale"] == {"reject": 1}
    assert funded["timing_basis"] == "nominal_eligibility_not_observed_vendor_publication"


def test_decision_coverage_rejects_future_nominal_eligibility():
    variants = _variants()
    result = _result(variants)
    result["mffu_batch"] = {
        "decision_context": [{
            "configuration": "MCB001", "stream": "funded",
            "event": "entry_admission", "action": "admit",
            "context": {
                "decision_ts_utc": "2025-06-17T15:00:00+00:00",
                "gamma_eligible_from_utc": "2025-06-17T16:00:00+00:00",
            },
        }],
    }
    with pytest.raises(ValueError, match="eligible after its decision"):
        _analyze(result, variants)


def test_htf_zone_concentration_reconciles_known_and_unknown_funded_trades():
    variants = _variants()
    result = _result(variants)
    result["summaries_cents"]["MCB001|myfundedfutures"]["trades_taken"] = 4
    first = _trade("MCB001", "2025-06-16", 15, pnl=1_000)
    first["htf_zone_id"] = "zone-a"
    first["account_id"] = "account-1"
    second = _trade("MCB001", "2025-06-17", 15, pnl=-200)
    second["htf_zone_id"] = "zone-a"
    second["account_id"] = "account-1"
    third = _trade("MCB001", "2025-06-18", 15, pnl=300)
    third["htf_zone_id"] = "zone-b"
    third["account_id"] = "account-2"
    fourth = _trade("MCB001", "2025-06-19", 15, pnl=-50)
    fourth["htf_zone_id"] = None  # Historical reuse or absent signal identity.
    result["tables"]["trades"] = [first, second, third, fourth]

    output = _analyze(result, variants)
    concentration = output["trade_concentration_by_variant"]["MCB001"]
    assert concentration["by_htf_zone_id"] == {
        "zone-a": {"trades": 2, "net_trade_pnl_cents": 800},
        "zone-b": {"trades": 1, "net_trade_pnl_cents": 300},
    }
    assert concentration["unknown_htf_zone"] == {
        "trades": 1, "net_trade_pnl_cents": -50,
    }
    assert concentration["htf_zone_coverage"] == {
        "status": "partial_unknown", "known_trades": 3, "unknown_trades": 1,
        "known_zone_count": 2, "reason": "htf_zone_id_absent_or_null",
    }
    assert concentration["by_account_id"] == {
        "account-1": {"trades": 2, "net_trade_pnl_cents": 800},
        "account-2": {"trades": 1, "net_trade_pnl_cents": 300},
    }
    assert concentration["unknown_account"] == {
        "trades": 1, "net_trade_pnl_cents": -50,
    }
    assert concentration["account_coverage"]["status"] == "partial_unknown"
    assert (sum(item["trades"] for item in concentration["by_htf_zone_id"].values())
            + concentration["unknown_htf_zone"]["trades"] == 4)
    assert (sum(item["net_trade_pnl_cents"] for item in
                concentration["by_htf_zone_id"].values())
            + concentration["unknown_htf_zone"]["net_trade_pnl_cents"] == 1_050)


def test_htf_zone_concentration_is_explicitly_unknown_for_legacy_trades():
    variants = _variants()
    result = _result(variants)
    result["summaries_cents"]["MCB001|myfundedfutures"]["trades_taken"] = 2
    result["tables"]["trades"] = [
        _trade("MCB001", "2025-06-16", 15, pnl=1_000),
        _trade("MCB001", "2025-06-17", 15, pnl=-200),
    ]
    concentration = _analyze(result, variants)["trade_concentration_by_variant"]["MCB001"]
    assert concentration["by_htf_zone_id"] == {}
    assert concentration["unknown_htf_zone"] == {
        "trades": 2, "net_trade_pnl_cents": 800,
    }
    assert concentration["htf_zone_coverage"]["status"] == "unavailable"
    assert concentration["basis"] == "actual_funded_trade_net_pnl_not_received_cash"


def test_htf_zone_concentration_rejects_malformed_source_identity():
    variants = _variants()
    result = _result(variants)
    result["summaries_cents"]["MCB001|myfundedfutures"]["trades_taken"] = 1
    trade = _trade("MCB001", "2025-06-16", 15, pnl=1_000)
    trade["htf_zone_id"] = 42
    result["tables"]["trades"] = [trade]
    with pytest.raises(ValueError, match="invalid HTF zone ID"):
        _analyze(result, variants)
