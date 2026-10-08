"""Read-only 64-intent matrix from one saved MFFU plan and result.

The plan is the membership authority.  A missing or failed economic child has
no cash value; its status is never inferred from a zero trade count or filled
with zero dollars.  This view does not re-resolve policies or inspect market
data while the Lab renders.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.propsim.funded.mffu_batch_plan import PLAN_SCHEMA

_AXES = ("schedule", "daily_cap", "entry_context", "exit", "sizing", "overhead", "geometry")
_LABELS = {
    "schedule": "Schedule",
    "daily_cap": "Daily cap",
    "entry_context": "Entry context",
    "exit": "Exit",
    "sizing": "Sizing",
    "overhead": "Overhead",
    "geometry": "Geometry",
}
_DISPOSITION_LABELS = {
    "newly_completed": "Newly completed",
    "compatible_reused": "Compatible reused result",
    "reused_nonimpact": "Reused · no reviewed lifecycle impact",
    "reused_equivalent_after_verification": "Reused · repaired history verified equivalent",
    "failed": "Failed",
}
POLICY_WORDS = {
    "schedule": {"S0": "Original three windows", "S1": "All open-market hours"},
    "daily_cap": {"U": "Uncapped", "D1": "One actual entry per trading date"},
    "entry_context": {
        "F0": "No gamma admission filter",
        "FE": "Skip early positive",
        "FL": "Skip positive London",
        "FEL": "Skip early positive and positive London",
    },
    "exit": {
        "XP": "Half at 1R; remainder to entry stop or daily close",
        "XF": "Whole position at 1R",
        "XG": "At 1R: whole in positive gamma; half otherwise",
        "XE": "At 1R: whole only in early positive; half otherwise",
    },
    "sizing": {
        "Q10": "Ten MNQ micros",
        "Q6": "Six MNQ micros (60% of ten)",
        "QG": "Six MNQ micros in positive gamma; ten otherwise",
    },
    "overhead": {
        "O0": "No overhead admission filter",
        "O8": "Nearest of eight overhead levels; GEX above 300,000",
    },
    "geometry": {
        "G0": "Fixed 80 ticks / 20 points",
        "G05": "5% of implied half-range, frozen at parent lock",
        "G075": "7.5% of implied half-range, frozen at parent lock",
        "G10": "10% of implied half-range, frozen at parent lock",
    },
}


def policy_label(axis: str, code: str) -> str:
    words = POLICY_WORDS.get(axis, {}).get(code)
    return f"{words} ({code})" if words else str(code).replace("_", " ")


def is_mffu_plan(plan: Any) -> bool:
    return getattr(plan, "plan_schema", None) == PLAN_SCHEMA


def variant_settings(variant: Any) -> list[tuple[str, str]]:
    """All seven intended axes plus the actual source-bound worker settings."""
    intent = json.loads(variant.intent_json)
    section = json.loads(variant.effective_section_json)
    settings = [(_LABELS[axis], policy_label(axis, str(intent[axis]))) for axis in _AXES]
    if "htf_timeframes" in section:
        settings.extend(canonical_settings(variant))
    settings.extend(
        (
            ("Context version", str(section["ifsm_context_policy_version"])),
            ("Entry gate", str(section["entry_context_policy"])),
            ("Overhead rule", str(section["overhead_policy"])),
            ("Opposing distance rule", str(section["opposing_distance_policy"])),
            ("Effective exit policy", str(section["exit_policy"])),
            ("Maximum actual entries per day", str(section["max_executed_trades_per_day"])),
            ("Possible micro quantities", ", ".join(map(str, variant.possible_quantities))),
            ("Cost per micro per fill", "$0.514"),
            ("Effective section hash", variant.effective_section_config_hash),
            ("Effective behavior hash", variant.effective_behavior_hash),
        )
    )
    return settings


def canonical_settings(variant: Any) -> list[tuple[str, str]]:
    """Known mechanics from this variant's saved effective section, in UI terms."""
    section = json.loads(variant.effective_section_json)
    scheme = section.get("session_scheme") or {}
    windows = []
    source_zone = ZoneInfo(scheme.get("timezone", "America/New_York"))
    chicago = ZoneInfo("America/Chicago")
    for name, window in (scheme.get("sessions") or {}).items():
        if name not in section.get("enabled_entry_sessions", []):
            continue
        endpoints = []
        for endpoint in ("start", "end"):
            hour, minute = map(int, window[endpoint].split(":"))
            at = datetime(2026, 1, 15, hour, minute, tzinfo=source_zone).astimezone(chicago)
            endpoints.append(at.strftime("%I:%M %p").lstrip("0"))
        windows.append(name.title() + " " + "–".join(endpoints))
    hours = (
        "All saved permitted open-market hours"
        if section.get("entry_schedule_policy") == "all_open_market_v1"
        else "; ".join(windows) + " Chicago, end exclusive"
    )
    direction = (
        "Long and short"
        if section.get("enable_longs") and section.get("enable_shorts")
        else "Long only"
        if section.get("enable_longs")
        else "Short only"
    )
    descriptions = policy_descriptions(variant)
    size = descriptions["Position size"]
    close_time = section.get("daily_close_time", "15:55")
    hour, minute = map(int, close_time.split(":"))
    close = datetime(2026, 1, 15, hour, minute).strftime("%I:%M %p").lstrip("0")
    buffer = section.get("daily_close_buffer_minutes", 5)
    stop_buffer = section.get("sl_buffer_ticks", 0)
    minimum = section.get("opposing_min_gap_ticks")
    cap = section.get("max_executed_trades_per_day")
    return [
        ("Entry hours", hours),
        ("Direction", direction),
        (
            "Higher-timeframe gap charts",
            ", ".join(chart_words(v) for v in section.get("htf_timeframes") or []),
        ),
        (
            "Supporting (parent) charts",
            ", ".join(chart_words(v) for v in section.get("parent_timeframes") or []),
        ),
        (
            "Largest distance from the parent gap to the opposing gap",
            descriptions["Largest distance from the parent gap to the opposing gap"],
        ),
        (
            "Smallest opposing gap",
            f"{minimum} tick(s) / {minimum * 0.25:g} index points"
            if minimum is not None
            else "Not saved",
        ),
        ("Profit target", f"Initial {section.get('tp_r_multiple', 1):g}R checkpoint"),
        ("Stop", f"Structural stop plus {stop_buffer} tick(s) buffer"),
        ("Exit rule", descriptions["Exit rule"]),
        (
            "Trades per day",
            "Uncapped actual entries"
            if cap is None
            else f"At most {cap} actual entry per trading date",
        ),
        (
            "Daily close",
            f"{close} Chicago; shortened-session close minus {buffer} minutes when earlier",
        ),
        ("Position size", size + "; $0.514 per micro per actual fill, total rounded once to cents"),
        ("Entry family", "Fresh continuation; retest is diagnostic and unratified"),
        ("Traded product", descriptions["Traded product"]),
        ("Signal source", "NQ one-minute candles supply signal evidence"),
        ("Open-position marks and loss-limit checks", "NQ recorded exchange trades"),
        (
            "Execution prices",
            "NQ recorded exchange trades are a price proxy for MNQ fills; "
            "no actual MNQ trade data was used",
        ),
    ]


def policy_descriptions(variant: Any) -> dict[str, str]:
    """One UI/export description of the saved effective policies, never amended settings."""
    section = json.loads(variant.effective_section_json)
    intent = json.loads(variant.intent_json)
    quantity_policy = getattr(variant, "quantity_policy", intent["sizing"])
    sizes = {
        "Q10": "10 MNQ micros at every entry",
        "Q6": "6 MNQ micros at every entry",
        "QG": "6 MNQ micros in positive gamma at entry; 10 in negative, neutral or unknown gamma",
    }
    size = sizes[quantity_policy]
    partial = "half exits at first 1R; remainder moves to entry stop or the daily close"
    exits = {
        "fixed_target_v1": "Always whole position at first 1R",
        "scale_out_half_breakeven_hold_to_close_v1": "Always " + partial,
        "gamma_conditional_1r_v1": "At first 1R, positive gamma exits whole; "
        "negative, neutral or unknown gamma: "
        + partial,
        "early_positive_whole_1r_v1": "At first 1R, early positive gamma "
        "(known age 1–5 reports) exits whole; "
        "established positive, negative, neutral or unknown gamma/age: " + partial,
    }
    policy = section["opposing_distance_policy"]
    fallback = section["opposing_parent_distance_ticks_max"]
    distance = f"Fixed {fallback} ticks / {fallback * 0.25:g} points"
    fractions = {
        "implied_move_005_v1": "5%",
        "implied_move_0075_v1": "7.5%",
        "implied_move_010_v1": "10%",
    }
    if policy != "fixed_v1":
        distance = (
            f"(1D Max − 1D Min) / 2 × {fractions[policy]}, frozen at supporting-parent lock; "
            "divide points by 0.25, ROUND_HALF_UP to integer ticks, minimum 1 tick; "
            f"missing/stale/invalid-context fallback: {fallback} ticks / {fallback * 0.25:g} points"
        )
    return {
        "Traded product": "Micro E-mini Nasdaq-100 exposure (MNQ); " + size,
        "Position size": size,
        "Exit rule": exits[section["exit_policy"]],
        "Largest distance from the parent gap to the opposing gap": distance,
    }


def result_variants(result: dict) -> dict[str, Any]:
    """Read bound variants from the already verified saved result, without executing Core."""
    from alpha_lab.propsim.funded.mffu_batch_plan import MffuVariantRef

    plan = (result.get("mffu_batch") or {}).get("plan") or {}
    if plan.get("plan_schema") != PLAN_SCHEMA:
        return {}
    variants = {v["name"]: MffuVariantRef.model_validate(v) for v in plan["variants"]}
    for row in (result.get("tables") or {}).get("configurations", []):
        variant = variants[row["configuration"]]
        if row.get("axes") != json.loads(variant.intent_json):
            raise ValueError("configuration description has a different bound intent")
        sizing = ((result.get("settings") or {}).get("sizing_by_configuration") or {}).get(
            variant.name
        )
        if sizing and any(
            sizing.get(k) != getattr(variant, k)
            for k in ("instrument", "quantity_policy", "quantity", "exit_policy")
        ):
            raise ValueError("configuration description has different execution settings")
    return variants


def concise_policy_columns(variant: Any) -> tuple[tuple[str, str], ...]:
    """Readable chart key: every distinguishing axis survives, without opaque codes."""
    section = json.loads(variant.effective_section_json)
    intent = json.loads(variant.intent_json)
    geometry = {
        "fixed_v1": "fixed 20pt",
        "implied_move_005_v1": "5% half-range",
        "implied_move_0075_v1": "7.5% half-range",
        "implied_move_010_v1": "10% half-range",
    }
    exits = {
        "fixed_target_v1": "whole at 1R",
        "scale_out_half_breakeven_hold_to_close_v1": "half at 1R",
        "gamma_conditional_1r_v1": "1R: positive whole, else half",
        "early_positive_whole_1r_v1": "1R: early-positive whole, else half",
    }
    return (
        (
            "Schedule",
            "all open hours"
            if section["entry_schedule_policy"] == "all_open_market_v1"
            else "three windows",
        ),
        ("Exit", exits[section["exit_policy"]]),
        ("Geometry", geometry[section["opposing_distance_policy"]]),
        (
            "Size",
            {"Q10": "10 MNQ", "Q6": "6 MNQ", "QG": "6/10 MNQ by gamma"}[variant.quantity_policy],
        ),
        ("Cap", "uncapped" if section["max_executed_trades_per_day"] is None else "1 entry/day"),
        ("Entry filter", POLICY_WORDS["entry_context"][intent["entry_context"]]),
        (
            "Overhead",
            "no overhead filter"
            if section["overhead_policy"] == "off"
            else "8-level overhead filter",
        ),
    )


def chart_words(value: str) -> str:
    """Spell saved chart units for the existing readable chart-list presenter."""
    numbers = {"1": "one", "3": "three", "4": "four", "5": "five", "10": "ten",
               "15": "fifteen", "30": "thirty"}
    if value and value[:-1] in numbers and value[-1].lower() in {"h", "m"}:
        unit = "hour" if value[-1].lower() == "h" else "minute"
        return numbers[value[:-1]] + "-" + unit
    return value


def matrix_rows(study: Any) -> list[dict[str, Any]]:
    """Every saved plan intent with its exact saved result disposition."""
    if not is_mffu_plan(study.plan):
        return []
    failed = {
        str(row.get("configuration") or row.get("batch_id")): row
        for row in (study.result.get("full_range_batch") or {}).get(
            "failed_configurations", []
        )
    }
    dispositions = {
        str(row["variant_id"]): row
        for row in (study.result.get("mffu_batch") or {}).get("dispositions") or []
    }
    firm = study.plan.firm_profiles[0].firm_key
    output = []
    for variant in study.plan.configurations:
        intent = json.loads(variant.intent_json)
        section = json.loads(variant.effective_section_json)
        summary = study.summary(variant.name, firm) or {}
        failure = failed.get(variant.name) or {}
        disposition = dispositions.get(variant.name) or {}
        raw_status = str(disposition.get("status") or summary.get("status")
                         or failure.get("status") or "No result saved")
        status = _DISPOSITION_LABELS.get(raw_status, raw_status)
        reason = str(disposition.get("reason") or summary.get("reason")
                     or failure.get("reason") or "")
        net_cash = (int(summary["net_cash_earned_cents"])
                    if summary.get("status") == "Completed"
                    and "net_cash_earned_cents" in summary else None)
        output.append({
            "Intent": variant.variant_id,
            "Family": variant.family,
            "Schedule": intent["schedule"],
            "Daily cap": intent["daily_cap"],
            "Entry context": intent["entry_context"],
            "Exit": intent["exit"],
            "Sizing": intent["sizing"],
            "Overhead": intent["overhead"],
            "Geometry": intent["geometry"],
            "Effective context version": section["ifsm_context_policy_version"],
            "Effective entry gate": section["entry_context_policy"],
            "Effective overhead rule": section["overhead_policy"],
            "Effective opposing distance": section["opposing_distance_policy"],
            "Effective exit rule": section["exit_policy"],
            "Effective daily cap": (
                section["max_executed_trades_per_day"]
                if section["max_executed_trades_per_day"] is not None else "None (uncapped)"),
            "Possible micro quantities": ", ".join(map(str, variant.possible_quantities)),
            "Status": status,
            "Reason": reason,
            "Reused from": disposition.get("reused_from"),
            "Net cash cents": net_cash,
            "Section hash": variant.effective_section_config_hash,
            "Behavior hash": variant.effective_behavior_hash,
        })
    return output


def variant_name(variant: Any) -> tuple[str, str]:
    """A compact unique label for the ranking and detail views."""
    intent = json.loads(variant.intent_json)
    first = (f"{variant.variant_id} · {policy_label('schedule', intent['schedule'])} · "
             f"{policy_label('daily_cap', intent['daily_cap'])}")
    second = " · ".join(
        policy_label(axis, intent[axis])
        for axis in ("entry_context", "exit", "sizing", "overhead", "geometry")
    )
    return first, second


def saved_analysis_tables(study: Any) -> dict[str, list[dict[str, Any]]]:
    """Compact saved cash effects and receipt waits; no replay or inferred zeroes."""
    if not is_mffu_plan(study.plan):
        return {}
    analysis = study.result.get("mffu_analysis") or {}
    if analysis.get("schema") not in {"ifsm_mffu_batch_analysis_v1", "ifsm_mffu_batch_analysis_v2"}:
        return {}
    pairs = []
    for item in analysis.get("matched_pairs") or []:
        detail = item.get("trade_diagnostics") or {}
        pairs.append(
            {
                "Base": item["base_id"],
                "Challenger": item["challenger_id"],
                "Changed axis": item.get("changed_axis"),
                "Group": item.get("group"),
                "Status": item["status"],
                "Net cash effect": (
                    fmt.money_cents(item["delta_cents"], signed=True)
                    if item.get("delta_cents") is not None
                    else None
                ),
                "Gained funded entries": detail.get("gained_entries"),
                "Lost funded entries": detail.get("lost_entries"),
                "Unavailable IDs": ", ".join(item.get("unavailable_ids") or ()),
            }
        )
    interactions = []
    for name, group in (analysis.get("interactions") or {}).items():
        for cell in group.get("cells") or []:
            interactions.append(
                {
                    "Interaction": name,
                    **{
                        key.replace("_", " ").title(): value
                        for key, value in cell.items()
                        if key not in {"delta_cents", "status", "unavailable_ids"}
                    },
                    "Status": cell["status"],
                    "Net cash effect": (
                        fmt.money_cents(cell["delta_cents"], signed=True)
                        if cell.get("delta_cents") is not None
                        else None
                    ),
                    "Unavailable IDs": ", ".join(cell.get("unavailable_ids") or ()),
                }
            )
    waits = []
    for variant in study.plan.configurations:
        item = (analysis.get("waiting_by_variant") or {}).get(variant.variant_id) or {}
        for measure in ("initial", "max_between_receipts", "terminal", "max_no_receipt_interval"):
            interval = item.get(measure) or {}
            if interval:
                status = "available"
            elif item.get("status") != "available":
                status = item.get("reason") or "unavailable"
            elif measure == "max_between_receipts":
                status = item.get("between_status") or "unavailable"
            else:
                status = item.get("terminal_status") or "unavailable"
            waits.append(
                {
                    "Intent": variant.variant_id,
                    "Measure": measure.replace("_", " "),
                    "Status": status,
                    "Start (Chicago)": interval.get("start_chicago_date"),
                    "End (Chicago)": interval.get("end_chicago_date"),
                    "Calendar days": interval.get("calendar_days"),
                    "Evaluated trading days": interval.get("evaluated_trading_days"),
                    "Start censored": interval.get("censored_start"),
                    "End censored": interval.get("censored_end"),
                }
            )
    return {"matched_pairs": pairs, "interactions": interactions, "waiting": waits}
