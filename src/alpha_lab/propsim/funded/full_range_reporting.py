"""Saved descriptive tables for the single six-configuration full-range batch.

This module consumes executed worker records. It never runs an engine, opens
market inputs or allocates funded cash to entry groups. Call before the result
is hashed and saved; the screen and review exporter then read the same tables.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.search.entry_activity import (
    day_start,
    entry_activity_payload,
    entry_trading_day,
)
from alpha_lab.propsim.funded.campaign import display_time, machine_time
from alpha_lab.propsim.funded.comparison_result import strategy_metrics
from alpha_lab.propsim.funded.position_walk import fill_cost_cents

SCHEMA = "ifsm_six_configuration_full_range_reporting_v1"
TABLE_FILES = {
    "strategy_trades": "strategy_trades.csv",
    "daily_activity": "daily_activity.csv",
    "no_entry_spans": "no_entry_spans.csv",
    "context_breakdowns": "context_breakdowns.csv",
    "roll_days": "roll_days.csv",
    "period_results": "period_results.csv",
}
_CONTEXT_KEYS = (
    "entry_session", "regime", "slot_chicago", "context_available", "gate_status",
    "source_eod_date", "levels_source_eod_date", "regime_source_eod_date",
    "availability_ts_utc", "selected_instrument_id", "roll_flag",
    "context_asof_ts", "availability_window_status", "unavailable_reason", "source_file_sha256",
)


def _cents(value: Any) -> int:
    number = Decimal(str(value)) * 100
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("reporting money must be exact cents")
    return int(number)


def _stamp(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("reporting timestamps must be aware")
    return stamp.tz_convert("UTC")


def _month(value: Any) -> str:
    return _stamp(value).tz_convert("America/Chicago").strftime("%Y-%m")


def _side(value: Any) -> int:
    return 1 if str(value).lower().split(".")[-1] in {"long", "bullish"} else -1


def _context(row: dict) -> dict:
    context = dict(row.get("entry_context") or {})
    out = {key: context.get(key, row.get(key)) for key in _CONTEXT_KEYS}
    for key in ("entry_session", "regime", "slot_chicago", "gate_status"):
        if out[key] is None:
            out[key] = "unknown"
    out["context_evidence_status"] = "available" if context else "unavailable"
    return out


def _ordinary_trade(raw: dict, output: dict) -> dict:
    size = output["sizing"]
    quantity = int(raw.get("quantity", size["quantity"]))
    tick_value = int(size["tick_value_cents"])
    mills = int(size["cost_per_contract_mills"])
    scale = raw.get("scale_out_ticks")
    partial = quantity // 2 if scale is not None else 0
    if partial and quantity % 2:
        raise ValueError("half-exit reporting requires an even original quantity")
    final = quantity - partial
    entry, exit_ = int(raw["entry_ticks"]), raw.get("exit_ticks")
    if exit_ is None or raw.get("realized_r") is None:
        raise ValueError("ordinary trade is missing its final priced outcome")
    tick_quantities = _side(raw["direction"]) * (
        final * (int(exit_) - entry) + partial * (int(scale or entry) - entry)
    )
    gross = tick_quantities * tick_value
    points = tick_quantities * .25 / quantity
    core_points = float(raw["realized_r"]) * int(raw["risk_ticks"]) * .25
    if abs(points - core_points) > 1e-8:
        raise ValueError("ordinary weighted fills differ from Core's saved realized outcome")
    costs = fill_cost_cents(quantity, mills) + fill_cost_cents(final, mills)
    if partial:
        costs += fill_cost_cents(partial, mills)
    entry_stamp, exit_stamp = _stamp(raw["entry_ts_utc"]), _stamp(raw["resolution_ts_utc"])
    row = {
        **raw, "configuration": output["configuration"], "stream": "strategy",
        "firm_key": None, "instrument": size["instrument"], "quantity": quantity,
        "scale_out_quantity": partial, "final_exit_quantity": final,
        "tick_value_cents": tick_value, "cost_per_contract_mills": mills,
        "entry_utc": entry_stamp.isoformat(), "exit_utc": exit_stamp.isoformat(),
        "entry_chicago": display_time(int(entry_stamp.value)),
        "exit_chicago": display_time(int(exit_stamp.value)),
        "entry_trading_day": entry_trading_day(entry_stamp),
        "entry_calendar_date": entry_stamp.tz_convert("America/Chicago").date().isoformat(),
        "exit_calendar_date": exit_stamp.tz_convert("America/Chicago").date().isoformat(),
        "entry_calendar_month": _month(entry_stamp),
        "gross_pnl_cents": gross, "costs_cents": costs, "net_pnl_cents": gross - costs,
        "quantity_weighted_gross_points": points,
        "quantity_weighted_net_points": (gross - costs) / (tick_value * quantity) * .25,
        "execution_policy": "ordinary_completed_candle_exit_v1",
        "section_config_hash": output.get("section_config_hash"),
        "scale_out_timestamp_status": ("available" if raw.get("scale_out_ts_utc") else
                                       "unavailable_in_ordinary_record" if partial else
                                       "not_scaled_out"),
        **_context(raw),
    }
    if raw.get("scale_out_ts_utc"):
        partial_stamp = _stamp(raw["scale_out_ts_utc"])
        row["scale_out_utc"] = partial_stamp.isoformat()
        row["scale_out_chicago"] = display_time(int(partial_stamp.value))
    return row


_PRICE_EXTREMA = (
    "price_min", "price_max", "pre_target_price_min", "pre_target_price_max",
    "post_target_price_min", "post_target_price_max",
)


def _funded_trade(row: dict, sizes: dict[str, dict], *, reused: bool = False) -> dict:
    size = sizes[row["configuration"]]
    out = {
        **row, "stream": "funded", "instrument": size["instrument"],
        "tick_value_cents": size["tick_value_cents"],
        "cost_per_contract_mills": size["cost_per_contract_mills"],
        "entry_trading_day": entry_trading_day(row["entry_utc"]),
        "entry_calendar_month": _month(row["entry_utc"]),
        "entry_calendar_date": _stamp(row["entry_utc"]).tz_convert(
            "America/Chicago").date().isoformat(),
        "exit_calendar_date": _stamp(row["exit_utc"]).tz_convert(
            "America/Chicago").date().isoformat(),
        "execution_policy": "ordered_prints_stop_market_v2",
        **_context(row),
    }
    for key in ("gross_pnl", "costs", "net_pnl"):
        out[key + "_cents"] = _cents(row[key + "_usd"])
    risk = (int(row["initial_risk_cents"]) if row.get("initial_risk_cents") is not None
            else _cents(row["initial_risk_usd"])
            if row.get("initial_risk_usd") is not None else None)
    out["initial_risk_cents"] = risk
    out["r_status"] = "available" if risk is not None and risk > 0 else "unavailable_original_risk"
    if row["configuration"].startswith("MCB") and not reused and out["r_status"] != "available":
        raise ValueError("new MFFU funded trade lacks its original stop risk")
    out["r_basis"] = "trade_pnl_cents_divided_by_original_stop_risk_cents"
    out["gross_r"] = round(out["gross_pnl_cents"] / risk, 8) if risk and risk > 0 else None
    out["net_r"] = round(out["net_pnl_cents"] / risk, 8) if risk and risk > 0 else None
    for key in ("gross_r", "net_r"):
        if row.get(key) is not None and out[key] != round(float(row[key]), 8):
            raise ValueError(f"funded saved trade {key} differs from original-stop risk")
    price_status = row.get("price_excursion_status")
    if price_status is None:
        price_status = "unavailable_legacy_reuse" if reused else "unavailable_source_record"
    if row["configuration"].startswith("MCB") and not reused and price_status not in {
            "available", "available_entry_fill_only", "available_fills_only"}:
        raise ValueError("new MFFU funded trade lacks a complete price path")
    out["price_excursion_status"] = price_status
    out["price_excursion_fidelity"] = row.get("price_excursion_fidelity")
    out["htf_zone_id"] = row.get("htf_zone_id")
    for stem in _PRICE_EXTREMA:
        price = row.get(stem + "_ticks")
        ts_ns = row.get(stem + "_ns")
        if price_status.startswith("unavailable"):
            price = ts_ns = None
        if (price is None) != (ts_ns is None):
            raise ValueError(f"funded saved trade has incomplete {stem} evidence")
        out[stem + "_ticks"] = price
        out[stem + "_ns"] = ts_ns
        out[stem + "_utc"] = machine_time(int(ts_ns)) if ts_ns is not None else None
        out[stem + "_chicago"] = display_time(int(ts_ns)) if ts_ns is not None else None
    for key in ("favorable_excursion_ticks", "adverse_excursion_ticks"):
        out[key] = (None if price_status.startswith("unavailable") else row.get(key))
    if row["configuration"].startswith("MCB") and not reused:
        required = ("price_min", "price_max", "pre_target_price_min",
                    "pre_target_price_max")
        if row.get("scale_out_quantity"):
            required += ("post_target_price_min", "post_target_price_max")
        if any(out[stem + "_ticks"] is None for stem in required):
            raise ValueError("new MFFU funded trade has incomplete price extrema")
    denom = int(size["tick_value_cents"]) * int(row["quantity"])
    out["quantity_weighted_gross_points"] = out["gross_pnl_cents"] / denom * .25
    out["quantity_weighted_net_points"] = out["net_pnl_cents"] / denom * .25
    if out["gross_pnl_cents"] - out["costs_cents"] != out["net_pnl_cents"]:
        raise ValueError("funded saved trade money does not reconcile")
    if row.get("scale_out_ns") is not None:
        out["scale_out_utc"] = machine_time(int(row["scale_out_ns"]))
        out["scale_out_chicago"] = display_time(int(row["scale_out_ns"]))
    return out


def _metrics(rows: list[dict]) -> dict:
    gross = sum(row["gross_pnl_cents"] for row in rows)
    costs = sum(row["costs_cents"] for row in rows)
    net = sum(row["net_pnl_cents"] for row in rows)
    profits = sum(max(0, row["net_pnl_cents"]) for row in rows)
    losses = -sum(min(0, row["net_pnl_cents"]) for row in rows)
    return {
        "trades": len(rows), "gross_profit_cents": gross, "trading_costs_cents": costs,
        "profit_after_costs_cents": net, "profit_after_costs_usd": net / 100,
        "gross_points": sum(row["quantity_weighted_gross_points"] for row in rows),
        "net_points": sum(row["quantity_weighted_net_points"] for row in rows),
        "profit_factor_dollars": profits / losses if losses else None,
        "profit_factor_dollars_basis": (
            "positive_net_dollars_divided_by_absolute_negative_net_dollars"),
        "profit_factor_dollars_status": ("defined" if losses else "no_losses"
                                         if profits else "no_trades"),
        "wins_after_costs": sum(row["net_pnl_cents"] > 0 for row in rows),
    }


def _activity(rows: list[dict], dates: tuple[str, ...], cutoff: str) -> dict:
    trades = pd.DataFrame([{
        "trade_id": str(row.get("trade_id") or row.get("strategy_trade_id")
                        or row.get("trade_ref")) + ":" + str(row.get("seq", index)),
        "entry_ts_utc": row["entry_utc"], "resolution_ts_utc": row["exit_utc"],
    } for index, row in enumerate(rows)])
    activity = entry_activity_payload(trades, list(dates), cutoff_utc=cutoff)
    if activity["summary"]["entry_rows_outside_evaluation_dates"]:
        raise ValueError("actual evaluated trade entry lies outside the declared membership")
    return activity


def attach_full_range_reports(
    result: dict, outputs: list[dict], failures: list[dict], *,
    configuration_names: tuple[str, ...], evaluation_dates: tuple[str, ...],
    warmup_dates: tuple[str, ...], cutoff_utc: str, coverage_by_day: dict | None = None,
    expected_configuration_count: int = 6, expected_firm_count: int = 2,
) -> dict:
    """Attach saved/reconciled descriptive tables for independent streams.

    Worker ``daily_activity`` rows may supply observed status/count fields. Every
    declared date gets a row even on failure; absent evidence stays unavailable.
    Point-in-time ``entry_context`` belongs to each actual trade separately.
    """
    names, dates = tuple(configuration_names), tuple(evaluation_dates)
    if len(names) != expected_configuration_count or len(set(names)) != len(names):
        raise ValueError("the batch has the wrong number of unique configurations")
    if not dates or dates != tuple(sorted(set(dates))):
        raise ValueError("the evaluation calendar must be ordered and unique")
    if set(dates) & set(warmup_dates):
        raise ValueError("warmup and evaluation membership overlap")
    successful = {output["configuration"]: output for output in outputs}
    failed = {failure["configuration"]: failure for failure in failures}
    if (len(successful) != len(outputs) or len(failed) != len(failures) or
            set(successful) & set(failed) or set(successful) | set(failed) != set(names)):
        raise ValueError("every configuration must have exactly one complete or failed output")
    tables = result["tables"]
    summaries = result["summaries_cents"]
    firms = tuple(dict.fromkeys(summary["firm_key"] for summary in summaries.values()))
    if (len(firms) != expected_firm_count
            or set(summaries) != {f"{name}|{firm}" for name in names for firm in firms}):
        raise ValueError("the batch has the wrong number of separate configuration/firm results")
    sizes = {name: output["sizing"] for name, output in successful.items()}
    ordinary = [_ordinary_trade(raw, output) for output in outputs
                for raw in output.get("strategy_trades_no_account", [])]
    reused = {name for name, output in successful.items()
              if (output.get("reuse", {}).get("status") == "compatible_reused"
                  or output.get("correction_reuse", {}).get("status") in {
                      "reused_nonimpact", "reused_equivalent_after_verification"})}
    funded = [_funded_trade(row, sizes, reused=row["configuration"] in reused)
              for row in tables.get("trades", [])]
    tables["trades"] = funded
    tables["strategy_trades"] = ordinary
    old_metrics = {row["configuration"]: row for row in tables.get("strategy_metrics", [])}
    for key in TABLE_FILES:
        if key != "strategy_trades":
            tables[key] = []
    tables["strategy_metrics"] = []
    tables["monthly_results"] = []
    months = tuple(str(period) for period in pd.period_range(dates[0][:7], dates[-1][:7], freq="M"))
    coverage = dict(coverage_by_day or {})
    activity_summaries = {}
    for name in names:
        output = successful.get(name)
        observations = {}
        for row in (output or {}).get("daily_activity", []):
            key = (row["stream"], row.get("firm_key"), row["evaluation_date"])
            if key in observations:
                raise ValueError("duplicate daily stream observation")
            observations[key] = row
        for stream, firm in [("strategy", None), *[("funded", firm) for firm in firms]]:
            base = {"configuration": name, "stream": stream, "firm_key": firm,
                    "pair_id": None if firm is None else f"{name}|{firm}",
                    "firm": None if firm is None else summaries[f"{name}|{firm}"].get("firm")}
            stream_id = name + "|" + (firm or "strategy")
            status = "Completed" if output else "Not completed"
            rows = [row for row in (ordinary if stream == "strategy" else funded)
                    if row["configuration"] == name and row.get("firm_key") == firm
                    and not row.get("is_warmup", False)]
            activity = _activity(rows, dates, cutoff_utc) if output else None
            summary = activity["summary"] if activity else {}
            activity_summaries[stream_id] = {**base, "status": status, **summary}
            daily_counts = {row["evaluation_date"]: row["actual_entries"]
                            for row in activity["daily"]} if activity else {}
            unavailable = []
            for day in dates:
                evidence = {**coverage.get(day, {}), **observations.get((stream, firm, day), {})}
                source_status = evidence.get("source_status", "unavailable")
                if source_status != "available":
                    unavailable.append(day)
                lower = day_start(day)
                upper = lower + pd.Timedelta(days=1)
                held = any(_stamp(row["entry_utc"]) < upper
                           and _stamp(row["exit_utc"]) > lower for row in rows)
                open_status = ("position_open_during_day" if held else "flat_all_day")
                daily = {
                    **base, "evaluation_date": day, "status": status,
                    "actual_entries": (daily_counts.get(day)
                                       if output and source_status == "available" else None),
                    "source_status": source_status,
                    "source_selected_instrument_id": evidence.get("source_selected_instrument_id"),
                    "roll_flag": evidence.get("roll_flag"),
                    "open_position_status": evidence.get(
                        "open_position_status", open_status if output else "unavailable"),
                    "processing_status": evidence.get("processing_status", "unavailable"),
                    "stage_count_status": evidence.get(
                        "stage_count_status", "observed_existing_counters" if
                        evidence.get("stage_counters") is not None else "unavailable"),
                    "reason": failed.get(name, {}).get("reason"),
                }
                for key, value in evidence.items():
                    if (key.startswith("stage_") or key.startswith("entries_refused_") or
                            key in {"open_position_at_day_end", "account_status_at_end",
                                    "refused_entries", "payout_events_this_day"}):
                        daily[key] = value
                    elif key == "entries":
                        daily["observed_entry_signals"] = value
                tables["daily_activity"].append(daily)
                day_rows = [row for row in rows if row["entry_trading_day"] == day]
                tables["roll_days"].append({
                    **base, "trading_day": day, "status": status,
                    "selected_instrument_id": evidence.get("source_selected_instrument_id"),
                    "roll_flag": evidence.get("roll_flag"), "source_status": source_status,
                    **(_metrics(day_rows) if output and source_status == "available" else {}),
                    "roll_days_excluded": False,
                })
            complete_activity = bool(output and not unavailable)
            for gap in activity["gaps"] if activity and complete_activity else []:
                tables["no_entry_spans"].append({
                    **base, **gap, "is_tied_longest": gap["evaluation_dates_without_entry"]
                    == summary["longest_zero_entry_dates"],
                    "activity_evidence_status": "complete",
                })
            if stream == "strategy":
                prior_metrics = old_metrics.get(name, {})
                if output and not prior_metrics:
                    prior_metrics = strategy_metrics(output)
                metrics = {**prior_metrics, **base, "status": status,
                           "reason": failed.get(name, {}).get("reason"),
                           "evaluation_dates": len(dates),
                           "activity_evidence_status": (
                               "complete" if complete_activity else "unavailable"),
                           "unavailable_evaluation_dates": len(unavailable)}
                if output:
                    metrics["profit_factor_basis"] = "net_R_existing_canonical_protocol"
                    metrics.update(_metrics(rows))
                if complete_activity:
                    metrics.update({key: summary[key] for key in (
                        "active_entry_dates", "zero_entry_dates", "longest_zero_entry_dates")})
                    metrics["tied_longest_spans"] = summary["tied_longest_intervals"]
                tables["strategy_metrics"].append(metrics)
            elif complete_activity:
                pair = summaries[base["pair_id"]]
                for key in ("active_entry_dates", "zero_entry_dates", "longest_zero_entry_dates"):
                    pair[key] = summary[key]
                for row in tables.get("pair_results", []):
                    if row["pair_id"] == base["pair_id"]:
                        row.update({key: summary[key] for key in (
                            "active_entry_dates", "zero_entry_dates", "longest_zero_entry_dates")})
            cash = [event for event in tables.get("cash_ledger", [])
                    if stream == "funded" and event["pair_id"] == base["pair_id"]]
            for scope, scoped_dates in [("full_range", dates), *[
                (year, tuple(day for day in dates if day.startswith(year)))
                for year in tuple(dict.fromkeys(day[:4] for day in dates))]]:
                selected = [row for row in rows if row["entry_trading_day"] in scoped_dates]
                events = [event for event in cash if scope == "full_range"
                          or _month(event["ts_utc"]).startswith(scope)]
                tables["period_results"].append({
                    **base, "scope": scope, "status": status,
                    "evaluation_dates": len(scoped_dates),
                    **(_metrics(selected) if output else {}),
                    **(_cash_metrics(events) if stream == "funded" and output else {}),
                    "continuation_slice": scope != "full_range",
                })
            cumulative_cash = 0
            for month in months:
                selected = [row for row in rows if row["entry_calendar_month"] == month]
                events = [event for event in cash if _month(event["ts_utc"]) == month]
                cash_metrics = _cash_metrics(events) if firm and output else {}
                if cash_metrics:
                    cumulative_cash += _cents(cash_metrics["net_usd"])
                tables["monthly_results"].append({
                    **base, "month": month, "status": status,
                    "partial_month": month in (months[0], months[-1]),
                    "trade_attribution": "entry_calendar_month_America_Chicago",
                    "cash_attribution": "event_calendar_month_America_Chicago" if firm else None,
                    **(_metrics(selected) if output else {}),
                    **cash_metrics,
                    **({"cumulative_net_usd": cumulative_cash / 100} if firm and output else {}),
                })
            for dimension in ("entry_session", "regime", "slot_chicago", "availability"):
                groups = defaultdict(list)
                for row in rows:
                    label = row.get(dimension)
                    if dimension == "availability":
                        label = ("available" if row.get("context_available") is True else
                                 "outside_hours" if row.get("gate_status") ==
                                 "not_applicable_outside_hours" else "unknown_or_missing")
                    groups[str(label or "unknown")].append(row)
                for label, grouped in sorted(groups.items()):
                    sources = sorted({str(row[key]) for row in grouped for key in
                                      ("source_eod_date", "levels_source_eod_date",
                                       "regime_source_eod_date")
                                      if row.get(key) is not None})
                    tables["context_breakdowns"].append({
                        **base, "dimension": dimension, "bucket": label,
                        **_metrics(grouped), "source_eod_dates": sources,
                        "funded_cash_allocated": False,
                        "context_evidence_unavailable_trades": sum(
                            row["context_evidence_status"] != "available" for row in grouped),
                    })
    result["full_range_reporting"] = {
        "schema": (SCHEMA if expected_configuration_count == 6 and expected_firm_count == 2
                   else "ifsm_mffu_context_batch_reporting_v1"),
        "configuration_names": list(names), "firms": list(firms),
        "evaluation_dates": list(dates), "warmup_dates": list(warmup_dates),
        "cutoff_utc": cutoff_utc, "activity_summaries": activity_summaries,
        "point_definition": "quantity_weighted_per_original_contract",
        "cash_group_allocation": False,
    }
    validation = validate_full_range_reports(result)
    if not validation["passed"]:
        raise ValueError("full-range saved reporting reconciliation failed: " +
                         ", ".join(key for key, value in validation["checks"].items() if not value))
    result["full_range_reporting"]["validation"] = validation
    return result


def _cash_metrics(events: list[dict]) -> dict:
    receipts = sum(_cents(event["amount_usd"]) for event in events
                   if event["kind"] == "payout_received")
    costs = sum(_cents(event["amount_usd"]) for event in events
                if event["kind"] == "account_purchase")
    return {"received_usd": receipts / 100, "account_costs_usd": costs / 100,
            "net_usd": (receipts - costs) / 100,
            "payouts_received": sum(event["kind"] == "payout_received" for event in events),
            "accounts_bought": sum(event["kind"] == "account_purchase" for event in events)}


def validate_full_range_reports(result: dict) -> dict:
    """Reconcile saved activity, money and grouped outcomes without executing inputs."""
    spec, tables = result["full_range_reporting"], result["tables"]
    names, firms, dates = spec["configuration_names"], spec["firms"], spec["evaluation_dates"]
    streams = {(name, stream, firm) for name in names for stream, firm in
               [("strategy", None), *[("funded", firm) for firm in firms]]}
    daily = tables["daily_activity"]
    keys = [(row["configuration"], row["stream"], row.get("firm_key"), row["evaluation_date"])
            for row in daily]
    checks = {
        "strategy_rows_match_plan": len(tables["strategy_metrics"]) == len(names)
        and {row["configuration"] for row in tables["strategy_metrics"]} == set(names),
        "funded_rows_match_plan": len(tables["pair_results"]) == len(names) * len(firms),
        "all_declared_daily_rows": len(keys) == len(streams) * len(dates)
        and len(set(keys)) == len(keys) and set(keys) == {
            (*stream, day) for stream in streams for day in dates},
        "unavailable_is_not_zero": all(row["actual_entries"] is None for row in daily
                                        if row["source_status"] != "available"
                                        or row["status"] != "Completed"),
        "warmup_excluded_from_activity": not set(spec["warmup_dates"]) & set(dates),
        "cash_not_allocated_to_entry_groups": all(row.get("funded_cash_allocated") is False
                                                  for row in tables["context_breakdowns"]),
    }
    trade_reconciled = cash_reconciled = groups_reconciled = gaps_reconciled = True
    for name, stream, firm in streams:
        def match(row, expected=(name, stream, firm)):
            return (row["configuration"], row["stream"], row.get("firm_key")) == expected
        rows = [row for row in tables["strategy_trades" if stream == "strategy" else "trades"]
                if match(row) and not row.get("is_warmup", False)]
        days = [row for row in daily if match(row)]
        if any(row["status"] != "Completed" for row in days):
            continue
        if all(row["source_status"] == "available" for row in days):
            trade_reconciled &= sum(row["actual_entries"] for row in days) == len(rows)
            expected_counts = Counter(row["entry_trading_day"] for row in rows)
            trade_reconciled &= all(row["actual_entries"] == expected_counts[row["evaluation_date"]]
                                    for row in days)
            covered = [day for gap in tables["no_entry_spans"] if match(gap)
                       for day in dates if gap["first_evaluation_date"] <= day <=
                       gap["last_evaluation_date"]]
            gaps_reconciled &= set(covered) == {row["evaluation_date"] for row in days
                                               if row["actual_entries"] == 0}
            gaps_reconciled &= len(covered) == len(set(covered))
        monthly = [row for row in tables["monthly_results"] if match(row)]
        trade_reconciled &= sum(row["trades"] for row in monthly) == len(rows)
        trade_reconciled &= sum(row["profit_after_costs_cents"] for row in monthly) == sum(
            row["net_pnl_cents"] for row in rows)
        for dimension in ("entry_session", "regime", "slot_chicago", "availability"):
            groups = [row for row in tables["context_breakdowns"] if match(row)
                      and row["dimension"] == dimension]
            groups_reconciled &= sum(row["trades"] for row in groups) == len(rows)
            groups_reconciled &= sum(row["profit_after_costs_cents"] for row in groups) == sum(
                row["net_pnl_cents"] for row in rows)
        if firm:
            summary = result["summaries_cents"][f"{name}|{firm}"]
            cash_reconciled &= sum(_cents(row["net_usd"]) for row in monthly) == summary[
                "net_cash_earned_cents"]
            cash_reconciled &= sum(_cents(row["received_usd"]) for row in monthly) == summary[
                "payouts_received_cents"]
            cash_reconciled &= sum(_cents(row["account_costs_usd"]) for row in monthly) == summary[
                "account_costs_cents"]
    checks.update(activity_equals_actual_entries=bool(trade_reconciled),
                  cash_months_equal_saved_headlines=bool(cash_reconciled),
                  context_groups_equal_actual_outcomes=bool(groups_reconciled),
                  spans_partition_zero_entry_dates=bool(gaps_reconciled))
    return {"passed": all(checks.values()), "checks": checks,
            "activity_denominator": len(dates), "streams": len(streams)}
