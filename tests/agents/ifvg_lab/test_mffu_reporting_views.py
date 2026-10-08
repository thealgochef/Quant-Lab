"""Reporting definitions exercise saved events, never a new strategy simulation."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.lab import mffu_gamma as gamma
from alpha_lab.agents.data_infra.ifvg.presentation.lab import mffu_lenses as lenses
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import study_from_result
from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_reporting import (
    compact_review_tables,
    load_report,
    report_folder,
    source_hashes,
)


def _rank_row(key, *, cash=100, wait=1.001, spend=12500, lost=2, activity=0.5, payout=50000):
    return {
        "configuration_id": key,
        "measurement_status": {},
        "net_received_cash_cents": cash,
        "acquisition_spend_cents": spend,
        "first_receipt_elapsed_seconds": wait,
        "first_flat_buffer_elapsed_seconds": wait,
        "accounts_lost": lost,
        "entry_date_coverage": activity,
        "longest_no_entry_evaluated_dates": 2,
        "largest_received_payout_cents": payout,
        "median_received_payout_cents": payout,
        "average_entries_per_evaluated_date": 1.0,
        "first_receipt_calendar_days": 2,
        "schedule": "S0",
    }


@pytest.mark.parametrize(
    "lens,field,direction",
    [
        ("total_cash", "net_received_cash_cents", "desc"),
        ("payout_speed", "first_receipt_elapsed_seconds", "asc"),
        ("cushion_speed", "first_flat_buffer_elapsed_seconds", "asc"),
        ("account_usage", "accounts_lost", "asc"),
        ("activity", "entry_date_coverage", "desc"),
        ("large_payouts", "largest_received_payout_cents", "desc"),
    ],
)
def test_all_six_sorts_use_exact_values_missing_last_stable_id(lens, field, direction):
    a, b, c, missing = [_rank_row(k) for k in ("MCB002", "MCB001", "MCB003", "MCB004")]
    c[field] += 0.00001 if direction == "desc" else -0.00001
    missing[field] = None
    assert [r["configuration_id"] for r in lenses.sort_rows([a, missing, b, c], lens)] == [
        "MCB003",
        "MCB001",
        "MCB002",
        "MCB004",
    ]


def test_targets_are_and_inclusive_keep_empty_and_report_every_miss_and_unknown():
    good = _rank_row("MCB001")
    bad = dict(
        _rank_row("MCB002"), average_entries_per_evaluated_date=2.0, first_receipt_calendar_days=9
    )
    unknown = _rank_row("MCB003")
    unknown["measurement_status"] = {"first_receipt_calendar_days": "not_reached"}
    targets = {"average_entries_per_evaluated_date": [0.8, 1.3], "first_receipt_calendar_days": 2}
    source = deepcopy([good, bad, unknown])
    view = lenses.filter_rows(source, targets=targets)
    assert view["matches"] == 1 and view["unknown"] == 1 and view["failed"] == 1
    assert view["rows"][0]["configuration_id"] == "MCB001"
    empty_targets = {**targets, "net_received_cash_cents": 200}
    view = lenses.filter_rows(source, targets=empty_targets)
    assert view["matches"] == 0 and view["rows"] == []
    closest = lenses.filter_rows(source, targets=empty_targets, closest=True)
    assert closest["matches"] == 0 and len(closest["rows"]) == 3
    failing = next(r for r in closest["rows"] if r["configuration_id"] == "MCB002")
    assert [v["miss"] for v in failing["target_violations"]] == [0.7, 7, 100]
    assert (
        next(r for r in closest["rows"] if r["configuration_id"] == "MCB003")[
            "unverifiable_targets"
        ][0]["status"]
        == "not_reached"
    )
    assert source == [good, bad, unknown]
    reset = lenses.filter_rows(source)
    assert reset["matches"] == 3
    exported = json.loads(
        lenses.export_view(
            closest, lens="total_cash", targets=empty_targets, categories={}, search=""
        )
    )
    assert exported["targets"] == empty_targets and exported["scope"] == "full_study"
    assert exported["rows"][0]["net_received_cash_cents"] == 100


def test_invalid_ranges_and_filters_never_change_saved_settings():
    rows = [_rank_row("MCB001"), _rank_row("MCB002")]
    targets = {"average_entries_per_evaluated_date": [1.3, 0.8]}
    assert lenses.filter_rows(rows, targets=targets)["errors"]
    assert lenses.validate_targets({"entry_date_coverage": 1.1})
    assert lenses.validate_targets({"accounts_bought": float("inf")})
    assert (
        lenses.filter_rows(rows, search="mcb001", categories={"schedule": ["S0"]})["matches"] == 1
    )
    assert lenses.filter_rows(rows, categories={"schedule": ["S1"]})["matches"] == 0
    readable = dict(rows[0], schedule="S1", sizing="Q10", geometry="G10", exit="XP")
    for text in ("ten", "all hours", "partial", "implied"):
        assert lenses.filter_rows([readable], search=text)["matches"] == 1


def _study():
    key = "MCB001|myfundedfutures"
    variant = SimpleNamespace(name="MCB001", intent_json=json.dumps({"schedule": "S0"}))
    plan = SimpleNamespace(
        configurations=(variant,),
        source=SimpleNamespace(
            evaluation_dates=("2025-06-16", "2025-06-17", "2025-06-18"),
            warmup_dates=(),
            cutoff_utc="2025-06-18T21:00:00Z",
        ),
    )

    def scoped(row):
        return dict(row, pair_id=key, configuration="MCB001", firm_key="myfundedfutures")

    trades = [
        scoped(
            {
                "trade_ref": "failed",
                "entry_utc": "2025-06-16T13:00:00Z",
                "exit_utc": "2025-06-16T14:00:00Z",
                "entry_trading_day": "2025-06-16",
                "balance_after_usd": 2200.0,
                "account_failed": True,
                "account_number": 1,
                "seq": 10,
            }
        ),
        scoped(
            {
                "trade_ref": "open_peak",
                "entry_utc": "2025-06-17T12:00:00Z",
                "exit_utc": "2025-06-17T13:00:00Z",
                "entry_trading_day": "2025-06-17",
                "balance_after_usd": 1900.0,
                "max_equity_usd": 3000.0,
                "account_number": 2,
                "seq": 12,
            }
        ),
        scoped(
            {
                "trade_ref": "equality",
                "entry_utc": "2025-06-17T14:00:00Z",
                "exit_utc": "2025-06-17T15:00:00.000000009Z",
                "entry_trading_day": "2025-06-17",
                "balance_after_usd": 2000.0,
                "account_number": 2,
                "seq": 14,
            }
        ),
    ]
    cash = [
        scoped(
            {
                "kind": "account_purchase",
                "account_number": 1,
                "amount_usd": 125.0,
                "ts_utc": "2025-06-15T22:00:00Z",
                "seq": 1,
            }
        ),
        scoped(
            {
                "kind": "account_purchase",
                "account_number": 2,
                "amount_usd": 125.0,
                "ts_utc": "2025-06-16T14:00:00Z",
                "seq": 11,
            }
        ),
        scoped(
            {
                "kind": "payout_received",
                "account_number": 2,
                "amount_usd": 450.0,
                "ts_utc": "2025-06-18T21:00:00Z",
                "seq": 20,
            }
        ),
    ]
    journeys = [
        scoped(
            {
                "account_number": 1,
                "created_utc": "2025-06-15T22:00:00Z",
                "failed_utc": "2025-06-16T14:00:00Z",
            }
        ),
        scoped({"account_number": 2, "created_utc": "2025-06-16T14:00:00Z", "failed_utc": None}),
    ]
    events = [
        scoped(
            {
                "account_number": 2,
                "ts_utc": "2025-06-17T21:00:00Z",
                "seq": 15,
                "status_after": "secured",
            }
        ),
        scoped(
            {
                "account_number": 2,
                "ts_utc": "2025-06-17T21:01:00Z",
                "seq": 16,
                "status_after": "processing",
            }
        ),
        scoped(
            {
                "account_number": 2,
                "ts_utc": "2025-06-18T21:00:00Z",
                "seq": 21,
                "status_after": "ready",
            }
        ),
    ]
    result = {
        "tables": {
            "trades": trades,
            "cash_ledger": cash,
            "account_journeys": journeys,
            "account_events": events,
            "payout_events": [],
        },
        "summaries_cents": {
            key: {
                "configuration": "MCB001",
                "firm_key": "myfundedfutures",
                "status": "Completed",
                "net_cash_earned_cents": 20000,
                "account_costs_cents": 25000,
                "payouts_received_cents": 45000,
            }
        },
    }
    return study_from_result(result, result_id="a" * 64, plan=plan)


def test_milestone_excludes_failed_and_open_peaks_preserves_replacement_clock_and_zero_dates():
    row = lenses.build_rows(_study())[0]
    assert row["first_flat_buffer_trade_ref"] == "equality"
    assert row["first_flat_buffer_calendar_days"] == 2
    assert row["first_flat_buffer_elapsed_seconds"] == 147600.000000009
    assert row["first_receipt_calendar_days"] == 3
    assert row["purchases_before_first_receipt"] == 2
    assert row["replacement_spend_cents"] == 12500
    assert row["funded_entries"] == 3 and row["entry_date_coverage"] == 2 / 3
    assert row["zero_entry_dates"] + row["one_entry_dates"] + row["multiple_entry_dates"] == 3
    assert row["protection_duration_seconds"] == 60
    assert row["processing_duration_seconds"] == 86340
    assert row["account_milestone_statuses"][0]["cushion_status"] == "failed_before_reaching"
    assert row["no_receipt_intervals"][-1]["right_censored"]


def test_pending_or_no_receipt_not_zero_and_longest_includes_initial_terminal():
    study = _study()
    study.result["tables"]["cash_ledger"] = [
        r for r in study.result["tables"]["cash_ledger"] if r["kind"] != "payout_received"
    ]
    row = lenses.build_rows(study)[0]
    assert row["first_receipt_calendar_days"] is None
    assert row["first_receipt_status"] == "not_reached"
    assert row["largest_received_payout_cents"] is None
    assert row["longest_no_receipt_calendar_days"] == 3
    assert row["longest_no_receipt_right_censored"]
    assert not lenses.filter_rows([row], targets={"first_receipt_calendar_days": 100})["matches"]


def test_incomplete_account_history_is_not_a_conditional_median_reaching_sample():
    study = _study()
    study.result["tables"]["account_journeys"][1]["trades"] = 3
    row = lenses.build_rows(study)[0]
    assert row["account_buffer_unavailable"] == 1
    assert row["account_buffer_reached"] == 0
    assert row["conditional_median_account_days_to_buffer"] is None


@pytest.mark.parametrize(
    "value,age,bound,expected",
    [
        (-1, None, 0, "negative"),
        (0, 0, 0, "neutral_zero"),
        (1, 5, 5, "positive_early"),
        (1, 6, 6, "positive_established"),
        (1, None, 3, "positive_age_unknown"),
        (1, None, 6, "positive_established"),
        (None, None, 0, "unknown"),
    ],
)
def test_disjoint_gamma_categories_and_left_censoring(value, age, bound, expected):
    assert (
        gamma.category(
            {
                "status": "selected",
                "value": value,
                "positive_run_age": age,
                "positive_run_lower_bound": bound,
            }
        )
        == expected
    )


def test_point_cursor_hides_future_outcomes_checkpoint_and_map_and_preserves_nanoseconds():
    entry = "2025-06-17T02:59:59.999999999Z"
    target = "2025-06-17T03:00:00.000000001Z"
    context = {"gamma": {"status": "selected", "value": 1, "positive_run_age": 1}}
    trade = {
        "configuration_id": "MCB001",
        "population": "funded",
        "trade_key": "exact",
        "trade_ref": "a",
        "entry_utc": entry,
        "exit_utc": "2025-06-17T04:00:00Z",
        "trading_date": "2025-06-17",
        "first_checkpoint_utc": target,
        "entry_snapshot": context,
        "first_checkpoint_snapshot": context,
        "lock_policy_receipt": None,
        "first_target_policy_receipt": None,
        "entry_policy_receipt": None,
        "first_checkpoint_branch": "actual partial",
        "context_role": "executed_saved_context",
        "quantity": 6,
        "net_pnl_cents": 1784,
        "net_initial_risk_units": 0.37,
        "outcome": "partial_entry_stop",
        "remaining_leg_net_cents": -308,
    }
    data = {"trades": [trade]}
    now = gamma.selected_rows(data, "MCB001", cursor=entry)
    assert len(now) == 1 and now[0]["net_pnl_cents"] is None and now[0]["outcome"] is None
    assert not gamma.selected_rows(data, "MCB001", basis="first_1R_checkpoint", cursor=entry)
    assert len(gamma.checkpoint_cards(trade, cursor=entry)) == 1
    assert len(gamma.checkpoint_cards(trade, cursor=target)) == 2
    data.update(
        level_timeline=[{"eligible_from_utc": "2025-06-17T03:00:00Z", "level_set_id": "map"}],
        level_maps={
            "map": {
                "report_date": "2025-06-16",
                "items": {
                    "HVL": {"price": 100, "gex": None},
                    "HVL 0DTE": {"price": 100, "gex": 20},
                },
            }
        },
    )
    assert (
        gamma.level_segments(
            data, start="2025-06-17T02:00:00Z", end="2025-06-17T05:00:00Z", cursor=entry
        )
        == []
    )
    segments = gamma.level_segments(
        data, start="2025-06-17T02:00:00Z", end="2025-06-17T05:00:00Z", cursor=target
    )
    assert len(segments) == 1 and len(segments[0]["levels"]) == 2
    assert pd.Timestamp(segments[0]["start_utc"]) == pd.Timestamp("2025-06-17T03:00:00Z")
    assert segments[0]["levels"][0]["signed_vendor_gex"] is None


def test_gamma_group_counts_cents_risk_denominators_and_partition_reconcile():
    rows = [
        {
            "gamma_category": "negative",
            "chicago_clock": "London",
            "trading_date": "2025-06-17",
            "net_pnl_cents": 101,
            "net_initial_risk_units": 1.0,
            "outcome": "no_partial",
            "remaining_leg_net_cents": None,
        },
        {
            "gamma_category": "unknown",
            "chicago_clock": "London",
            "trading_date": "2025-06-17",
            "net_pnl_cents": -10,
            "net_initial_risk_units": None,
            "outcome": "partial_entry_stop",
            "remaining_leg_net_cents": -308,
        },
        {
            "gamma_category": "positive_early",
            "chicago_clock": "London",
            "trading_date": "2025-06-17",
            "net_pnl_cents": 0,
            "net_initial_risk_units": 0,
            "outcome": "partial_deadline",
            "remaining_leg_net_cents": 22,
        },
    ]
    groups = gamma.grouped(rows)
    assert sum(r["actual_trades"] for r in groups) == 3
    assert sum(r["net_pnl_cents"] for r in groups) == 91
    assert sum(r["unavailable_risk_count"] for r in groups) == 1
    partitions = gamma.outcome_partition(rows)
    assert sum(r["trades"] for r in partitions) == 3
    assert sum(r["remaining_leg_net_cents"] for r in partitions) == -286


def test_ordinary_partial_leg_is_conserved_and_missing_checkpoint_time_not_never_reached():
    class Index:
        _level_times = ()
        bundle_sha256 = "b" * 64
        table_sha256 = {}

        def snapshot(self, at):
            return {
                "policy_id": "test",
                "gamma": {"status": "selected", "value": 1},
                "levels": {"status": "no_eligible_set"},
            }

    trade = {
        "configuration": "MCB001",
        "trade_id": "ordinary",
        "trading_day": "2025-06-17",
        "entry_utc": "2025-06-17T02:00:00Z",
        "exit_utc": "2025-06-17T04:00:00Z",
        "entry_ticks": 800,
        "exit_ticks": 800,
        "scale_out_ticks": 816,
        "scale_out_quantity": 3,
        "final_exit_quantity": 3,
        "quantity": 6,
        "scale_out_ts_utc": None,
        "scale_out_timestamp_status": "unavailable_legacy_reuse",
        "resolution": "breakeven_stop",
        "direction": "long",
        "cost_per_contract_mills": 514,
        "tick_value_cents": 50,
        "costs_cents": 616,
        "net_pnl_cents": 1784,
        "risk_ticks": 16,
        "entry_context": {},
    }
    study = SimpleNamespace(
        result={"tables": {"strategy_trades": [trade]}},
        configurations=("MCB001",),
        calendar=("2025-06-17",),
        result_id="a" * 64,
    )
    data = gamma.build_gamma(study, Index())
    observed = data["trades"][0]
    assert observed["remaining_leg_net_cents"] == -308
    assert observed["net_pnl_cents"] == 1784
    assert observed["first_checkpoint_status"] == "reached_time_unavailable"
    assert observed["first_checkpoint_time_reason"] == "unavailable_legacy_reuse"
    assert len(gamma.selected_rows(data, "MCB001", "strategy", "entry")) == 1
    assert gamma.selected_rows(data, "MCB001", "strategy", "first_1R_checkpoint") == []


def test_reporting_companion_rejects_tamper_foreign_identity_and_stale_definition(tmp_path):
    rid = "a" * 64
    source_folder = tmp_path / "funded_comparison_results" / rid
    source_folder.mkdir(parents=True)
    (source_folder / "envelope.json").write_text(
        json.dumps({"payload": {"result_json_sha256": "b" * 64}})
    )
    folder = report_folder(tmp_path, rid)
    folder.mkdir(parents=True)
    encoded = json.dumps({"economic_result_id": rid, "reporting_version": lenses.VERSION}).encode()
    envelope = {
        "economic_result_id": rid,
        "reporting_version": lenses.VERSION,
        "schema": "ifsm_mffu_reporting_companion_v1",
        "source_result_json_sha256": "b" * 64,
        "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "report_bytes": len(encoded),
        "reporting_source_sha256": source_hashes(),
    }
    (folder / "report.json").write_bytes(encoded)
    (folder / "envelope.json").write_text(json.dumps(envelope))
    assert load_report(tmp_path, rid)["economic_result_id"] == rid
    (folder / "report.json").write_bytes(encoded + b" ")
    with pytest.raises(ValueError, match="identity or bytes changed"):
        load_report(tmp_path, rid)
    (folder / "report.json").write_bytes(encoded)
    envelope["economic_result_id"] = "c" * 64
    (folder / "envelope.json").write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="identity or bytes changed"):
        load_report(tmp_path, rid)
    envelope["economic_result_id"] = rid
    envelope["reporting_source_sha256"] = {}
    (folder / "envelope.json").write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="identity or bytes changed"):
        load_report(tmp_path, rid)


def test_half_cent_received_median_displays_half_up_without_rounding_sort_value():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from ifvg_lab_mffu_views import _display

    row = {"median_received_payout_cents": 100.5}
    display = _display([row], ["median_received_payout_cents"])
    assert display[0]["Median received payment"] == "$1.01"
    assert row["median_received_payout_cents"] == 100.5


def test_saved_report_requires_exact_producer_witness_after_runtime_change(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import ifvg_lab_mffu_views as ui

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import mffu_reporting as reporting

    producer = tmp_path / "producer.py"
    producer.write_bytes(b"version = 1\n")
    monkeypatch.setattr(reporting, "_source_paths", lambda: {"producer": producer})
    hashes = reporting.source_hashes()
    rid = "a" * 64
    root = tmp_path / "store"
    folder = report_folder(root, rid)
    folder.mkdir(parents=True)
    economic = root / "funded_comparison_results" / rid
    economic.mkdir(parents=True)
    economic.joinpath("envelope.json").write_text(json.dumps({
        "payload": {"result_json_sha256": "b" * 64},
    }))
    payload = {"economic_result_id": rid, "reporting_version": lenses.VERSION}
    encoded = json.dumps(payload).encode()
    folder.joinpath("report.json").write_bytes(encoded)
    folder.joinpath("envelope.json").write_text(json.dumps({
        **payload, "schema": reporting.SCHEMA,
        "source_result_json_sha256": "b" * 64,
        "report_bytes": len(encoded), "report_sha256": hashlib.sha256(encoded).hexdigest(),
        "reporting_source_sha256": hashes,
    }))
    reporting._save_source_witnesses(folder, hashes)
    witness = folder / "producer_sources" / f"{hashes['producer']}.py"
    assert witness.read_bytes() == producer.read_bytes()
    producer.write_bytes(b"version = 2\n")
    assert load_report(root, rid) == payload
    assert ui.reporting(str(root), rid) == payload
    renamed = witness.with_name("renamed.py")
    witness.rename(renamed)
    with pytest.raises(ValueError, match="identity or bytes changed"):
        ui.reporting(str(root), rid)
    renamed.rename(witness)
    witness.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="identity or bytes changed"):
        load_report(root, rid)
    with pytest.raises(ValueError, match="identity or bytes changed"):
        ui.reporting(str(root), rid)
    witness.unlink()
    with pytest.raises(ValueError, match="identity or bytes changed"):
        load_report(root, rid)
    with pytest.raises(ValueError, match="identity or bytes changed"):
        ui.reporting(str(root), rid)
    witness.write_bytes(b"version = 1\n")
    folder.joinpath("report.json").write_bytes(encoded + b" ")
    with pytest.raises(ValueError, match="identity or bytes changed"):
        load_report(root, rid)


def test_original_reporting_regression_if_published():
    root = (
        Path(__file__).resolve().parents[3].parent
        / "Claude-Quant-Lab-Research-Artifacts/ifsm-mffu-context-batch-20261007/study_store"
    )
    try:
        report = load_report(
            root, "7a8cb062d3fc31bcf71e7015a60e4ea389988b749435609fbc15cd1beda9d052"
        )
    except FileNotFoundError:
        pytest.skip("exact original reporting companion is not published in this checkout")
    one, sixty_two = report["lenses"][0], report["lenses"][61]
    assert (
        one["funded_entries"],
        one["entry_dates"],
        one["first_flat_buffer_calendar_days"],
        one["first_receipt_calendar_days"],
        one["accounts_bought"],
        one["accounts_lost"],
    ) == (265, 147, 12, 33, 10, 9)
    assert (
        sixty_two["funded_entries"],
        sixty_two["entry_dates"],
        sixty_two["zero_entry_dates"],
        sixty_two["one_entry_dates"],
        sixty_two["multiple_entry_dates"],
    ) == (302, 153, 100, 80, 73)
    ticks = sorted(
        r["frozen_limit_ticks"]
        for r in report["gamma"]["geometry"]
        if r["configuration_id"] == "MCB062" and r["population"] == "funded"
    )
    assert (len(ticks), min(ticks), max(ticks), (ticks[150] + ticks[151]) / 2) == (
        302,
        17,
        212,
        117,
    )
    assert (sum(t > 80 for t in ticks), ticks.count(80), sum(t < 80 for t in ticks)) == (297, 1, 4)
    assert len(report["lenses"]) == 64
    compact = compact_review_tables(report)
    assert len(compact["configuration_lenses"]) == 64
    for config in ("MCB001", "MCB062"):
        groups = report["gamma"]["precomputed"][config + "|funded|entry"]["groups"]
        assert sum(r["actual_trades"] for r in groups) == next(
            r["funded_entries"] for r in report["lenses"] if r["configuration_id"] == config
        )
