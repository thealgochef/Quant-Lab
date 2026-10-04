"""Task B execution groups and independent cash-event months reconcile."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import pytest
from strategy_core.decisions.sessions import trading_day_for
from strategy_core.strategies.ifvg_smc.menthorq_levels import (
    LEVEL_COLUMN_NAMES,
    MenthorqLevelSnapshot,
)
from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash
from strategy_core.strategies.ifvg_smc.state import IFVG_SEED_SCHEMA_VERSION, IfvgDaySeed, seed_hash
from strategy_core.structures.swings import SwingTracker

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.menthorq_study_reporting import (
    StudyConfigurationInput,
    assert_context_execution_parity,
    build_study_reports,
    write_study_reports,
)
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import resolve_axis_overrides
from alpha_lab.agents.data_infra.ifvg.search.task_b import task_b_configurations
from alpha_lab.propsim.funded.clock import CHICAGO
from tests.agents.test_ifvg_v2_reporting_manifest import _tables

_DAYS = ("2025-06-16", "2025-06-30", "2025-07-01", "2025-07-02", "2026-06-10")
_FIRMS = ("myfundedfutures", "takeprofittrader")


class _Provider:
    schema_version = 1
    formula_version = "menthorq_eod_v1"
    source_file_sha256 = {"fixture": "a" * 64}

    def snapshot(self, stamp):
        local = stamp.astimezone(CHICAGO)
        day = local.date()
        logical_day = trading_day_for(stamp)
        available = 6 <= local.hour < 17 and day != date(2025, 6, 30)
        return MenthorqLevelSnapshot(
            trading_date=day, source_eod_date=day - timedelta(days=1) if available else None,
            source_file_sha256="a" * 64, levels=dict.fromkeys(LEVEL_COLUMN_NAMES, None),
            regime="positive" if available else "unknown", selected_instrument_id=11,
            roll_flag=logical_day is not None and logical_day.isoformat() in {
                "2025-06-16", "2025-07-01", "2025-07-02"},
            context_available=available,
        )

    def prior_cash_close_for(self, stamp):
        return None


def _input_tables():
    tables = _tables()
    entries = [pd.Timestamp("2025-06-30T23:00:00Z"), pd.Timestamp("2025-07-01T15:00:00Z")]
    for kind in (RecordTable.ENTRY_CANDIDATE, RecordTable.ELIGIBLE_DECISION,
                 RecordTable.EXECUTED_TRADE):
        table = tables[kind]
        table["trading_day"] = date(2025, 7, 1)
        table["envelope_trading_day"] = "2025-07-01"
        table["entry_session"] = ["asia", "ny"]
        table["envelope_entry_session"] = ["asia", "ny"]
        table["envelope_ts_utc"] = entries
        table["entry_ts_utc"] = entries if kind is RecordTable.EXECUTED_TRADE else pd.NaT
        table["geometry_entry_bar_open_ticks"] = table.entry_ticks
    trades = tables[RecordTable.EXECUTED_TRADE]
    trades["resolution_ts_utc"] = trades.entry_ts_utc + pd.Timedelta(minutes=2)
    trades["envelope_ts_utc"] = trades.resolution_ts_utc
    candidates = tables[RecordTable.ENTRY_CANDIDATE].copy()
    candidates["is_warmup"] = False
    blocked = candidates.iloc[[0]].copy()
    blocked["candidate_id"] = "blocked-candidate"
    blocked["entry_session"] = "unknown"
    blocked["envelope_entry_session"] = "unknown"
    blocked["envelope_ts_utc"] = pd.Timestamp("2025-07-02T15:00:00Z")
    blocked["trading_day"] = date(2025, 7, 2)
    blocked["block_reasons"] = '["regime_gate", "session", "session"]'
    warmup = candidates.iloc[[0]].copy()
    warmup["candidate_id"] = "warmup-candidate"
    warmup["envelope_ts_utc"] = pd.Timestamp("2025-06-13T15:00:00Z")
    warmup["trading_day"] = date(2025, 6, 13)
    warmup["is_warmup"] = True
    tables[RecordTable.ENTRY_CANDIDATE] = pd.concat(
        [candidates, blocked, warmup], ignore_index=True)
    return tables


def _funded(names):
    summaries, events = {}, []
    for name in names:
        for firm in _FIRMS:
            summaries[f"{name}|{firm}"] = {
                "configuration": name, "firm_key": firm, "status": "Completed",
                "net_cash_earned_cents": 7_501,
                "payouts_received_cents": 20_001, "account_costs_cents": 12_500,
                "payouts_received_count": 1, "accounts_purchased": 2,
            }
            for kind, amount, stamp in (
                ("account_purchase", 102, "2025-06-16T15:00:00Z"),
                ("payout_received", 200.01, "2025-07-01T00:00:00Z"),
                ("account_purchase", 23, "2025-07-02T15:00:00Z"),
            ):
                events.append({"configuration": name, "firm_key": firm, "kind": kind,
                               "amount_usd": amount, "ts_utc": stamp})
    return {"schema_version": "funded_comparison_result_v1",
            "mode": "single_account_configuration_comparison",
            "validation": {"passed": True}, "summaries_cents": summaries,
            "tables": {"cash_ledger": events}}


def _inputs():
    rows = task_b_configurations()
    funded = _funded(row.name for row in rows)
    return tuple(StudyConfigurationInput(
        configuration=row.name, tables=_input_tables(), provider=_Provider(),
        section=resolve_profile_config({
            "section_overrides": resolve_axis_overrides(dict(row.axis_value_ids)),
        }).section,
        evaluation_days=_DAYS, funded_result=funded, cost_points=.5,
        run_identity={"configuration": row.name},
    ) for row in rows)


def test_thirteen_configs_keep_roll_trades_and_reconcile_points_without_cash_allocation():
    reports = build_study_reports(_inputs())
    comparison = reports["comparison"]
    assert len(comparison) == 13
    assert comparison.trades.eq(2).all()
    assert comparison.net_points.eq(-1).all()
    assert comparison["roll_day_trades"].eq(2).all()
    assert comparison["roll_day_net_points"].eq(-1).all()
    assert comparison["ny.trades"].eq(1).all()
    assert comparison["ny.net_points"].eq(-3).all()
    assert not any(column.startswith("ny.") and "cash" in column for column in comparison)
    for firm in _FIRMS:
        assert comparison[f"net_cash_cents.{firm}"].eq(7_501).all()
    for name in ("comparison_by_session", "comparison_by_regime", "comparison_by_slot", "cells"):
        table = reports[name]
        assert not any("cash" in column for column in table)
        grouped = table.groupby("configuration").agg({"trades": "sum", "wins": "sum",
                                                       "gross_points": "sum", "net_points": "sum"})
        assert len(grouped) == (12 if name == "cells" else 13)
        assert grouped.trades.eq(2).all()
        assert grouped.wins.eq(1).all()
        assert grouped.gross_points.eq(0).all()
        assert grouped.net_points.eq(-1).all()
    assert "unknown" in set(reports["comparison_by_regime"].regime)
    assert "outside_cash" in set(reports["comparison_by_slot"].slot_chicago)
    candidates = reports["context_on/grouped_entry_candidate"]
    assert candidates["count"].sum() == 3
    assert "unknown" in set(candidates.entry_session)
    assert len(reports["context_on/context_export"]) == 4
    assert reports["context_on/context_export"].decision_id.notna().sum() == 2
    assert reports["context_on/context_export"].trade_id.notna().sum() == 2
    coverage = reports["coverage"].iloc[0]
    assert coverage.evaluation_days == 5
    assert coverage.days_with_context_available == 4
    assert coverage.candidates == 3 and coverage.candidates_with_context == 2
    assert coverage["candidates_blocked.session"] == 1
    assert coverage["candidates_blocked.regime_gate"] == 1


def test_entry_calendar_month_and_cash_event_calendar_month_are_independent():
    reports = build_study_reports(_inputs())
    months = reports["comparison_by_month"]
    assert "entry_calendar_month" not in months and "net_points" not in months
    assert "calendar_month" in months and "net_points_by_entry_month" in months
    base = months.loc[months.configuration.eq("baseline")].set_index("calendar_month")
    assert len(base) == 13 and base.index.tolist() == [
        str(period) for period in pd.period_range("2025-06", "2026-06", freq="M")]
    assert base.loc["2025-06", "net_points_by_entry_month"] == 2
    assert base.loc["2025-07", "net_points_by_entry_month"] == -3
    comparison = reports["comparison"].set_index("configuration")
    assert base.trades.sum() == comparison.loc["baseline", "trades"] == 2
    assert base.net_points_by_entry_month.sum() == comparison.loc["baseline", "net_points"] == -1
    for metric in ("wins", "gross_points"):
        assert base[metric].sum() == comparison.loc["baseline", metric]
    assert "net_points_by_entry_month" not in reports["comparison"]
    for firm in _FIRMS:
        cash_column = f"net_cash_by_cash_event_month_cents.{firm}"
        assert base.loc["2025-06", cash_column] == 9_801
        assert base.loc["2025-07", cash_column] == -2_300
        assert base[cash_column].sum() == 7_501
    rolls = reports["roll_days"]
    base_rolls = rolls.loc[rolls.configuration.eq("baseline")].set_index("trading_day")
    assert set(base_rolls.index) == {"2025-06-16", "2025-07-01", "2025-07-02"}
    assert base_rolls.loc["2025-06-16", "trades"] == 0
    assert base_rolls.loc["2025-06-16", "cash_events"] == 2
    assert base_rolls.loc["2025-07-01", "trades"] == 2
    assert base_rolls.loc["2025-07-02", "trades"] == 0
    assert base_rolls.loc["2025-07-02", "cash_events"] == 2
    assert base_rolls.trades.sum() == 2 and base_rolls.net_points.sum() == -1
    for firm in _FIRMS:
        assert base_rolls[f"cash_events.{firm}"].sum() == 2
        assert base_rolls[f"net_cash_by_cash_event_day_cents.{firm}"].sum() == -12_500


def test_scheduled_exit_uses_recorded_partial_points_and_subtracts_cost_once():
    inputs = _inputs()
    changed_inputs = []
    for config in inputs:
        tables = {kind: table.copy() for kind, table in config.tables.items()}
        trades = tables[RecordTable.EXECUTED_TRADE]
        trades.loc[0, "resolution"] = "scheduled_close"
        trades.loc[0, "exit_ticks"] = 104
        trades.loc[0, "realized_ticks"] = 4
        trades.loc[0, "scheduled_exit_deadline_ts_utc"] = trades.loc[0, "resolution_ts_utc"]
        trades.loc[0, "scheduled_exit_schedule_id"] = "frozen-schedule-id"
        changed_inputs.append(replace(config, tables=tables))
    reports = build_study_reports(changed_inputs)
    total = reports["comparison"].iloc[0]
    assert total.gross_points == -1.5  # +1 point scheduled exit, -2.5 point stop
    assert total.net_points == -2.5  # two round trips at 0.5 point each
    months = reports["comparison_by_month"]
    june = months.loc[months.configuration.eq("baseline") &
                      months.calendar_month.eq("2025-06")].iloc[0]
    assert june.gross_points == 1 and june.net_points_by_entry_month == .5


def test_executed_context_uses_honest_entry_time_even_when_emitted_at_later_resolution():
    inputs = _inputs()
    changed_inputs = []
    for config in inputs:
        tables = {kind: table.copy() for kind, table in config.tables.items()}
        trades = tables[RecordTable.EXECUTED_TRADE]
        trades.loc[0, "resolution_ts_utc"] = pd.Timestamp("2025-07-01T13:00:00Z")
        trades.loc[0, "envelope_ts_utc"] = trades.loc[0, "resolution_ts_utc"]
        changed_inputs.append(replace(config, tables=tables))
    reports = build_study_reports(changed_inputs)
    grouped = reports["baseline/executed_trade_metrics"].set_index("regime")
    assert grouped.loc["unknown", "trades"] == 1
    assert grouped.loc["unknown", "entry_session"] == "asia"
    assert grouped.loc["unknown", "slot_chicago"] == "outside_cash"
    assert grouped.loc["positive", "trades"] == 1
    months = reports["comparison_by_month"]
    june = months.loc[months.configuration.eq("baseline") &
                      months.calendar_month.eq("2025-06")].iloc[0]
    assert june.trades == 1 and june.net_points_by_entry_month == 2


def test_sunday_cash_purchase_keeps_event_date_without_monday_roll_attribution():
    inputs = _inputs()
    result = copy.deepcopy(inputs[0].funded_result)
    for event in result["tables"]["cash_ledger"]:
        if event["kind"] == "account_purchase" and event["amount_usd"] == 102:
            event["ts_utc"] = "2025-06-15T22:00:00Z"  # Sunday 17:00 Chicago
    reports = build_study_reports(tuple(replace(config, funded_result=result) for config in inputs))
    rolls = reports["roll_days"]
    monday = rolls.loc[rolls.trading_day.eq("2025-06-16")]
    assert monday.cash_events.eq(0).all()
    for firm in _FIRMS:
        assert monday[f"net_cash_by_cash_event_day_cents.{firm}"].eq(0).all()
        assert reports["comparison"][f"net_cash_cents.{firm}"].eq(7_501).all()


@pytest.mark.parametrize("stamp, accepted", [
    ("2025-06-15T23:00:00Z", True),
    ("2025-06-14T23:00:00Z", False),
])
def test_sunday_asia_entry_uses_monday_scope_and_roll_but_preserves_civil_export_date(
    tmp_path, stamp, accepted,
):
    entry = pd.Timestamp(stamp)
    logical_day = trading_day_for(entry.to_pydatetime())
    assert logical_day.isoformat() == ("2025-06-16" if accepted else "2025-06-15")
    inputs = []
    for config in _inputs():
        tables = {kind: frame.copy() for kind, frame in config.tables.items()}
        for kind in (RecordTable.ENTRY_CANDIDATE, RecordTable.ELIGIBLE_DECISION,
                     RecordTable.EXECUTED_TRADE):
            tables[kind].loc[0, "trading_day"] = logical_day
            tables[kind].loc[0, "envelope_trading_day"] = logical_day.isoformat()
            tables[kind].loc[0, "envelope_ts_utc"] = entry
        trade = tables[RecordTable.EXECUTED_TRADE]
        trade.loc[0, "entry_ts_utc"] = entry
        trade.loc[0, "resolution_ts_utc"] = entry + pd.Timedelta(minutes=2)
        trade.loc[0, "envelope_ts_utc"] = trade.loc[0, "resolution_ts_utc"]
        inputs.append(replace(config, tables=tables))
    if not accepted:
        with pytest.raises(ValueError, match="outside the evaluation calendar"):
            build_study_reports(inputs)
        return
    reports = build_study_reports(inputs)
    roll_rows = reports["roll_days"]
    monday = roll_rows.loc[roll_rows.trading_day.eq("2025-06-16")]
    assert monday.trades.eq(1).all() and monday.net_points.eq(2).all()
    assert reports["comparison"].roll_day_trades.eq(2).all()
    export = reports["baseline/context_export"]
    sunday = export.loc[export.entry_session.eq("asia") & export.trade_id.notna()].iloc[0]
    assert sunday.trading_date == date(2025, 6, 15)
    assert sunday.source_eod_date is None and bool(sunday.roll_flag)
    months = reports["comparison_by_month"]
    june = months.loc[months.configuration.eq("baseline") & months.calendar_month.eq("2025-06")]
    assert june.trades.eq(1).all() and june.net_points_by_entry_month.eq(2).all()
    paths = write_study_reports(tmp_path / "reports", inputs)
    context_path = paths["baseline/context_export"]
    written = pd.read_csv(context_path, comment="#")
    assert written.loc[written.trade_id.notna() & written.entry_session.eq("asia"),
                       "trading_date"].tolist() == ["2025-06-15"]
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    assert validation["context_trading_date"] == "civil_date_in_America_Chicago"


def test_parity_allows_only_proven_profile_ids_and_preserves_qualification_geometry_and_outcomes():
    left = _input_tables()[RecordTable.EXECUTED_TRADE]
    right = left.copy()
    for column in ("setup_id", "envelope_setup_id", "candidate_id", "decision_id", "trade_id"):
        right[column] = right[column].astype(str) + "-context"
    right["envelope_profile_hash"] = "c" * 64
    right["envelope_profile_name"] = "ifvg_search_profile_" + "c" * 16
    right["profile_name"] = right.envelope_profile_name
    assert_context_execution_parity(left, right)
    for column in ("realized_ticks", "entry_ticks", "mae_ticks", "envelope_qualification_mode"):
        changed = right.copy()
        changed.loc[0, column] = "custom_profile" if column.endswith(("mode", "name")) else 999
        with pytest.raises(ValueError, match="executed trades differ"):
            assert_context_execution_parity(left, changed)
    with pytest.raises(ValueError, match="executed trades differ"):
        assert_context_execution_parity(left, pd.concat([right, right.iloc[[0]]]))


def _profile_bound_seed_execution_pair():
    sections = [resolve_profile_config({
        "section_overrides": resolve_axis_overrides(dict(row.axis_value_ids)),
    }).section for row in task_b_configurations()[:2]]
    seed = IfvgDaySeed(
        schema_version=IFVG_SEED_SCHEMA_VERSION, profile_hash=ifvg_profile_hash(sections[0]),
        source_day=date(2025, 6, 30), registries=(), swings=SwingTracker().snapshot(), reducer=None,
    )
    context_seed = replace(seed, profile_hash=ifvg_profile_hash(sections[1]))
    assert seed.profile_hash != context_seed.profile_hash
    assert seed_hash(seed) != seed_hash(context_seed)
    baseline = _input_tables()[RecordTable.EXECUTED_TRADE]
    context = baseline.copy()
    baseline["entering_seed_hash"] = seed_hash(seed)
    context["entering_seed_hash"] = seed_hash(context_seed)
    return baseline, context


def test_parity_allows_complete_seed_digest_difference_from_only_the_core_profile_hash():
    baseline, context = _profile_bound_seed_execution_pair()
    assert_context_execution_parity(baseline, context)


@pytest.mark.parametrize("column", ["entry_ticks", "geometry_entry_bar_open_ticks",
                                     "entry_ts_utc", "resolution_ts_utc", "envelope_ts_utc",
                                     "unrecognized_source_hash"])
def test_seed_provenance_exclusion_keeps_execution_geometry_timing_and_other_hashes_strict(column):
    baseline, context = _profile_bound_seed_execution_pair()
    if column == "unrecognized_source_hash":
        baseline[column] = "a" * 64
        context[column] = "a" * 64
        context.loc[0, column] = "b" * 64
    elif column.endswith("ts_utc"):
        context.loc[0, column] += pd.Timedelta(nanoseconds=1)
    else:
        context.loc[0, column] += 1
    with pytest.raises(ValueError, match="executed trades differ"):
        assert_context_execution_parity(baseline, context)


@pytest.mark.parametrize("failure", ["unverified", "cash_mismatch", "inexact_cents",
                                     "wrong_mode", "missing_firm", "component_mismatch",
                                     "event_count_mismatch", "amount_fields_mismatch"])
def test_bad_funded_evidence_fails_before_any_report_file_is_written(tmp_path, failure):
    inputs = _inputs()
    result = copy.deepcopy(inputs[0].funded_result)
    if failure == "unverified":
        result["validation"]["passed"] = False
    elif failure == "cash_mismatch":
        result["summaries_cents"]["baseline|myfundedfutures"]["net_cash_earned_cents"] += 1
    elif failure == "inexact_cents":
        result["tables"]["cash_ledger"][0]["amount_usd"] = 102.001
    elif failure == "wrong_mode":
        result["mode"] = "old_campaign"
    elif failure == "component_mismatch":
        result["summaries_cents"]["baseline|myfundedfutures"]["payouts_received_cents"] += 1
    elif failure == "event_count_mismatch":
        result["summaries_cents"]["baseline|myfundedfutures"]["accounts_purchased"] += 1
    elif failure == "amount_fields_mismatch":
        result["tables"]["cash_ledger"][0]["amount_cents"] = 10_201
    else:
        result["summaries_cents"].pop("baseline|myfundedfutures")
    inputs = (replace(inputs[0], funded_result=result), *inputs[1:])
    with pytest.raises(ValueError):
        write_study_reports(tmp_path / "reports", inputs)
    assert not (tmp_path / "reports").exists()


def test_reports_refuse_missing_or_additional_configurations():
    inputs = _inputs()
    with pytest.raises(ValueError, match="thirteen owner-specified"):
        build_study_reports(inputs[:-1])
    with pytest.raises(ValueError, match="thirteen owner-specified"):
        build_study_reports((*inputs, replace(inputs[0], configuration="extra")))
    changed = (*inputs[:-1], replace(inputs[-1], evaluation_days=_DAYS[:-1]))
    with pytest.raises(ValueError, match="evaluation calendars differ"):
        build_study_reports(changed)


def test_report_writer_exports_baseline_review_context_and_explicit_parity_exclusions(tmp_path):
    inputs = _inputs()
    paths = write_study_reports(tmp_path / "reports", inputs)
    baseline_export = paths["baseline/context_export"]
    metadata = json.loads(baseline_export.read_text().splitlines()[0][2:])
    assert metadata["menthorq_context_version"] is None
    assert metadata["report_only"] is True and metadata["archival"] is False
    assert len(pd.read_csv(baseline_export, comment="#")) == 4
    validation = json.loads(paths["validation"].read_text())
    assert validation["passed"] and validation["configuration_count"] == 13
    assert not validation["cash_allocated_to_trades"] and not validation["roll_days_excluded"]
    assert validation["roll_points_day"] == "entry_logical_trading_date"
    assert validation["roll_cash_day"] == "cash_event_calendar_date_in_America_Chicago"
    assert validation["monthly_index_column"] == "calendar_month"
    assert validation["monthly_net_points_column"] == "net_points_by_entry_month"
    assert validation["monthly_net_cash_columns"] == "net_cash_by_cash_event_month_cents.<firm>"
    assert validation["monthly_trade_metrics_scope"] == "entry_calendar_month_in_America_Chicago"
    assert "candidate_id" in validation["execution_parity_exclusions"]
    assert "entering_seed_hash" in validation["execution_parity_exclusions"]
    assert "envelope_qualification_mode" not in validation["execution_parity_exclusions"]
    assert "canonicalize_section" in validation["execution_parity_exclusion_provenance"][
        "envelope_profile_name"]
    assert "complete Core IfvgDaySeed" in validation["execution_parity_exclusion_provenance"][
        "entering_seed_hash"]
    assert set(validation["execution_parity_exclusion_provenance"]) == set(
        validation["execution_parity_exclusions"])
    assert all(path.is_file() for path in paths.values())


def test_current_verified_funded_result_round_trips_through_study_reporting(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import _calendar
    from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600
    from alpha_lab.propsim.funded.comparison_result import (
        build_comparison_result,
        validate_comparison,
    )
    from alpha_lab.propsim.funded.pair_ledger import PairLedger
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    inputs = _inputs()
    schedule, start_ns, cutoff_ns = _calendar(inputs[0].section, _DAYS)
    outputs = []
    for config in inputs:
        pairs = {}
        for firm in FIRM_PROFILES.values():
            pair_id = f"{config.configuration}|{firm.firm_key}"
            ledger = PairLedger(
                pair_id=pair_id, configuration=config.configuration, profile=firm,
                processing=TWO_BUSINESS_DAYS_FED_1600, quantity=1, tick_value_cents=500,
                cost_per_side_cents=514, trading_days=schedule,
                start_ns=start_ns, cutoff_ns=cutoff_ns,
            )
            ledger.start()
            ledger.finish()
            pairs[firm.firm_key] = {
                "pair_id": pair_id, "ledger": ledger.snapshot(), "forced_flat": 0,
                "trades_not_in_reference": 0,
            }
        outputs.append({
            "configuration": config.configuration, "display_name": config.configuration,
            "settings_plain": [], "axes": {}, "pairs": pairs,
            "sizing": {"instrument_label": "NQ", "quantity": 1,
                       "tick_value_cents": 500, "cost_per_contract_mills": 5140},
            "strategy_trades_no_account": [],
            "reference": {"equivalent": True, "saved_study_trades": 0, "replayed_trades": 0},
            "resumed": {}, "prints": {"minutes_checked": 0, "minutes_rebuilt_exactly": 0,
                                       "missing_utc_days": [], "files": []},
        })
    result = build_comparison_result(
        context={"task_b_cutoff_ns": cutoff_ns}, outputs=outputs, failures=[],
        profiles=tuple(FIRM_PROFILES.values()), trading_days=schedule,
        start_ns=start_ns, cutoff_ns=cutoff_ns,
        settings={"tick_value_cents": 500, "cost_per_side_cents": 514},
        rank_results=False, resume_check_requested=False,
    )
    result["validation"] = validate_comparison(
        result, outputs, tuple(FIRM_PROFILES.values()), cutoff_ns, require_resume_check=False,
    )
    assert result["validation"]["passed"]
    assert len(result["summaries_cents"]) == 26
    assert all("rank_within_firm" not in summary for summary in result["summaries_cents"].values())
    assert all(row["resumed_run_identical"] is None and not row["resume_check_requested"]
               for row in result["tables"]["execution_evidence"])
    empty_inputs = tuple(replace(
        config, tables={kind: frame.iloc[:0].copy() for kind, frame in config.tables.items()},
        funded_result=result, cost_points=.514,
    ) for config in inputs)
    paths = write_study_reports(tmp_path / "reports", empty_inputs)
    comparison = pd.read_csv(paths["comparison"])
    months = pd.read_csv(paths["comparison_by_month"])
    rolls = pd.read_csv(paths["roll_days"])
    assert comparison.trades.eq(0).all()
    for firm in FIRM_PROFILES.values():
        cash = -firm.acquisition_cost_cents
        assert comparison[f"net_cash_cents.{firm.firm_key}"].eq(cash).all()
        monthly = months.groupby("configuration")[
            f"net_cash_by_cash_event_month_cents.{firm.firm_key}"].sum()
        assert monthly.eq(cash).all()
    assert rolls.cash_events.eq(0).all()  # genuine account purchases occurred Sunday
