"""Pure presenter for a completed funded-account payout simulation.

``present_funded_result`` turns the ONE saved funded result dict (see
:mod:`alpha_lab.propsim.funded.result`) into frozen, display-ready views for
the completed-results screen and the review-folder export. It only READS the
result: every money figure comes from ``result["summaries_cents"]`` (exact
integer cents) or the saved tables. Nothing here recomputes money a second
way; :func:`headline_figures` exposes the exact integers the screen shows so
the export can prove screen/export agreement.

All text is plain English with full firm names and Chicago 12-hour times. No
hashes, file paths, serialized settings or engine labels reach the view.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

__all__ = [
    "HEADLINE_SENTENCE",
    "HEADLINE_KEYS",
    "Card",
    "Fact",
    "WaitRow",
    "PayoutRow",
    "PathPoint",
    "TradeRow",
    "AccountRow",
    "MonthRow",
    "CashPoint",
    "FirmView",
    "ComparisonRow",
    "DecisionRow",
    "StatusLine",
    "FundedResultView",
    "present_funded_result",
    "headline_figures",
    "format_usd",
    "format_points",
    "chicago_text",
    "chicago_datetime",
    "close_reason_text",
]

HEADLINE_SENTENCE = "Two separate funded-account simulations on the same market period."

#: The integers each firm's primary cards show (cents for money, plain counts
#: for accounts). The export compares these with instance_results.csv.
HEADLINE_KEYS = (
    "net_cash_earned_cents",
    "payouts_received_cents",
    "largest_single_payout_cents",
    "acquisition_costs_cents",
    "accounts_lost_before_first_payout",
    "accounts_lost_after_a_payout",
)

_CHICAGO = ZoneInfo("America/Chicago")
_TICK_POINTS = Decimal("0.25")

_CLOSE_REASONS = {
    "target": "Profit target reached",
    "stop": "Protective stop hit",
    "scheduled_close": "Closed at the scheduled daily close",
    "account_failure": "Account loss limit reached — position closed and account lost",
}

_FUNDING_TEXT = {
    "initial_credit": "Initial account (monthly credit)",
    "replacement_credit": "Replacement (monthly credit)",
    "growth_wallet": "Growth block (paid from payout cash)",
}

_DECISION_STATUS = {
    "owner_confirmed": "Confirmed by the owner",
    "owner_approved_for_pilot": "Approved by the owner for the pilot",
    "assumption": "Assumption — not established by a published source",
}

_INSTRUMENT_WORDS = {"mini": ("mini", "minis"), "micro": ("micro", "micros")}

_DEFAULT_EVIDENCE_LABELS = {
    "ordered_trade_prints": "Ordered exchange trade prints (exact relative to recorded prints)",
    "minute_bars_adverse_first": "One-minute candles, losing side assumed first (approximation)",
    "minute_bars_favorable_first": "One-minute candles, winning side assumed first "
    "(approximation)",
}


# ── formatting helpers ────────────────────────────────────────────────────


def format_usd(cents: int | None, *, signed: bool = False) -> str:
    """Exact integer cents as ``$1,234.56`` (``-$102.00`` when negative)."""

    if cents is None:
        return "Not available"
    sign = "-" if cents < 0 else ("+" if signed and cents > 0 else "")
    whole, part = divmod(abs(int(cents)), 100)
    return f"{sign}${whole:,}.{part:02d}"


def _cents(usd: Any) -> int | None:
    """Dollar values saved in the result tables back to exact cents."""

    if usd is None or usd == "":
        return None
    return int((Decimal(str(usd)) * 100).quantize(Decimal(1)))


def format_points(ticks: int | None) -> str:
    """Integer ticks as full index points (ticks x 0.25) with two decimals."""

    if ticks is None:
        return "Not available"
    return f"{Decimal(int(ticks)) * _TICK_POINTS:,.2f}"


_CHICAGO_LABEL = re.compile(
    r"^(?P<month>[A-Za-z]+) (?P<day>\d{1,2}), (?P<year>\d{4}) "
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2}) (?P<ampm>AM|PM)(?: (?P<zone>[A-Z]{2,4}))?$"
)


def chicago_text(label: str | None) -> str:
    """Tidy the result's Chicago label: ``January 13, 2026 9:00 AM CST``."""

    if not label:
        return "—"
    match = _CHICAGO_LABEL.match(label.strip())
    if not match:
        return label
    parts = match.groupdict()
    zone = f" {parts['zone']}" if parts["zone"] else ""
    return (
        f"{parts['month']} {int(parts['day'])}, {parts['year']} "
        f"{int(parts['hour'])}:{parts['minute']} {parts['ampm']}{zone}"
    )


def chicago_datetime(utc_text: str | None) -> datetime | None:
    """A saved UTC machine time as a naive Chicago wall-clock datetime (charts)."""

    if not utc_text:
        return None
    moment = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(_CHICAGO).replace(tzinfo=None)


def _month_label(month: str, partial: bool) -> str:
    year, number = month.split("-")
    name = datetime(int(year), int(number), 1).strftime("%B")
    return f"{name} {year}" + (" (partial month)" if partial else "")


def close_reason_text(exit_kind: str | None) -> str:
    if not exit_kind:
        return "Not recorded"
    return _CLOSE_REASONS.get(exit_kind, exit_kind.replace("_", " ").capitalize())


def _account_label(account_id: str | None, numbers: dict[str, int]) -> str:
    if not account_id:
        return "—"
    number = numbers.get(account_id)
    return f"Account {number}" if number is not None else "Another account"


def _hours(value: Any) -> str:
    hours = float(value or 0)
    return f"{hours:,.1f} account-hours"


# ── view dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Card:
    """One primary card. ``figures`` holds the exact integers shown."""

    key: str
    label: str
    value: str
    note: str
    figures: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class Fact:
    label: str
    value: str
    note: str = ""


@dataclass(frozen=True)
class WaitRow:
    reason: str
    hours: float
    hours_text: str
    count: int | None
    count_text: str


@dataclass(frozen=True)
class PayoutRow:
    state: str
    secured: str
    requested: str
    paid_or_due: str
    gross: str
    firm_share: str
    after_split: str
    after_split_cents: int | None


@dataclass(frozen=True)
class PathPoint:
    ts_utc: str
    when: str
    balance_usd: float | None
    floor_usd: float | None
    what: str


@dataclass(frozen=True)
class TradeRow:
    account_id: str
    number: int
    label: str
    trading_day: str
    direction: str
    quantity: str
    entry_time: str
    exit_time: str
    entry_price: str
    exit_price: str
    stop_price: str
    points_moved: str
    gross_result: str
    costs: str
    net_result: str
    net_result_cents: int | None
    initial_risk: str
    lowest_open_equity: str
    close_reason: str
    price_evidence: str
    approximate: bool
    account_lost: bool


@dataclass(frozen=True)
class AccountRow:
    account_id: str
    number: int
    label: str
    slot: int
    funding: str
    created: str
    created_utc: str | None
    status: str
    status_key: str
    lost: bool
    lost_after_payout: bool
    failed: str
    failed_utc: str | None
    failure_reason: str
    replaces: str
    replaced_by: str
    replaces_id: str | None
    trades: int
    payouts_received: int
    received: str
    received_cents: int
    largest_payout: str
    gross_requested: str
    final_balance: str
    final_floor: str
    payouts: tuple[PayoutRow, ...]
    path: tuple[PathPoint, ...]
    receipts_utc: tuple[tuple[str, int], ...]
    processing_intervals_utc: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class MonthRow:
    firm: str
    month: str
    label: str
    partial: bool
    received_usd: float
    account_costs_usd: float
    net_usd: float
    cumulative_net_usd: float
    payouts_received: int
    accounts_bought: int


@dataclass(frozen=True)
class CashPoint:
    firm: str
    ts_utc: str
    received_usd: float
    account_costs_usd: float
    net_cash_usd: float


@dataclass(frozen=True)
class FirmView:
    firm_key: str
    firm: str
    cards: tuple[Card, ...]
    facts: tuple[Fact, ...]
    waiting: tuple[WaitRow, ...]
    notices: tuple[str, ...]
    accounts: tuple[AccountRow, ...]
    trades: tuple[TradeRow, ...]
    months: tuple[MonthRow, ...]
    cash: tuple[CashPoint, ...]


@dataclass(frozen=True)
class ComparisonRow:
    measure: str
    unit: str
    values: tuple[tuple[str, str], ...]
    difference: str


@dataclass(frozen=True)
class DecisionRow:
    decided_on: str
    subject: str
    decision: str
    status: str


@dataclass(frozen=True)
class StatusLine:
    level: str  # info | warning | error | success
    text: str


@dataclass(frozen=True)
class FundedResultView:
    headline: str
    question: str
    period_text: str
    period_start_utc: str | None
    period_end_utc: str | None
    months: tuple[str, ...]
    strategy_text: str
    settings_text: str
    status: tuple[StatusLine, ...]
    is_sample: bool
    approximation: bool
    validation_passed: bool
    firms: tuple[FirmView, ...]
    comparison: tuple[ComparisonRow, ...]
    comparison_note: str
    limitations: tuple[str, ...]
    decisions: tuple[DecisionRow, ...]


# ── presenter ─────────────────────────────────────────────────────────────


def present_funded_result(result: dict[str, Any]) -> FundedResultView:
    """Build the frozen screen view from the one saved funded result."""

    summaries: dict[str, dict[str, Any]] = result["summaries_cents"]
    tables: dict[str, list[dict[str, Any]]] = result.get("tables", {})
    settings = result.get("settings") or {}
    period = result.get("period") or {}
    evidence = result.get("price_evidence") or {}
    evidence_labels = {**_DEFAULT_EVIDENCE_LABELS, **(evidence.get("labels") or {})}
    instrument = settings.get("instrument", "mini")
    singular, plural = _INSTRUMENT_WORDS.get(instrument, (instrument, f"{instrument}s"))

    order = [row["firm_key"] for row in tables.get("instance_results", [])] or list(summaries)
    firms = tuple(
        _firm_view(key, summaries[key], tables, evidence_labels, singular, plural)
        for key in order
        if key in summaries
    )

    approx_trades = int(evidence.get("trades_with_minute_approximation") or 0)
    approx_failures = int(evidence.get("approximate_failures") or 0)
    approximate = approx_trades > 0 or approx_failures > 0 or any(
        t.approximate for f in firms for t in f.trades
    )
    is_sample = result.get("purpose") == "engineering_sample"
    validation = result.get("validation")
    validation_passed = bool(validation and validation.get("passed"))

    status: list[StatusLine] = [
        StatusLine("info", "Simulated result — no real trades, account purchases or "
                   "withdrawals took place.")
    ]
    if is_sample:
        status.append(StatusLine(
            "warning", "Engineering sample — not a historical result. The prices are "
            "hand-made test prices and prove only that the calculations work."))
    if approximate:
        status.append(StatusLine(
            "warning",
            f"Approximation: {approx_trades} trade(s) were checked against one-minute "
            "candles, where the order of prices inside each minute is unknown. "
            f"{approx_failures} account loss(es) depend on that assumed order. Those "
            "outcomes are approximations, not exact live-account behavior."))
    elif evidence.get("policy") == "synthetic_fixture":
        status.append(StatusLine(
            "info", "Loss limits were checked on every hand-made price point of each trade."))
    else:
        status.append(StatusLine(
            "info", "Loss limits were checked on ordered exchange trade prints — exact "
            "relative to the recorded prints, not proof of broker execution."))
    if validation is None:
        status.append(StatusLine(
            "error", "This result has no internal money check. Do not rely on these figures."))
    elif not validation_passed:
        status.append(StatusLine(
            "error", "The internal money checks did not pass. Do not rely on these figures "
            "until the result is corrected."))
    else:
        status.append(StatusLine(
            "success", "Internal money checks passed: receipts, account costs, credits and "
            "account balances reconcile for both firms."))

    comparison = tuple(
        _comparison_row(row, [f.firm for f in firms]) for row in tables.get("comparison", [])
    )

    source = result.get("source") or {}
    strategy_text = str(source.get("description") or "Strategy source not recorded")
    quantity = settings.get("quantity")
    cost_side = settings.get("cost_per_side_usd")
    clock = settings.get("processing_clock") or {}
    settings_bits = []
    if quantity is not None:
        settings_bits.append(
            f"{quantity} {singular if quantity == 1 else plural} per trade in both firms")
    if cost_side is not None:
        settings_bits.append(
            f"{format_usd(_cents(cost_side))} cost per {singular} at entry and again at exit")
    if clock.get("description"):
        settings_bits.append(f"payout processing: {clock['description']}")
    settings_text = "; ".join(settings_bits)

    decisions = tuple(
        DecisionRow(
            decided_on=_date_text(d.get("decided_on")),
            subject=str(d.get("subject", "")),
            decision=str(d.get("decision", "")),
            status=_DECISION_STATUS.get(d.get("status", ""), str(d.get("status", ""))),
        )
        for d in result.get("owner_decisions") or []
    )
    limitations = tuple(str(x) for x in result.get("limitations") or [])
    if not limitations:
        limitations = ("No material limitations were saved with this result; treat it as "
                       "incomplete evidence.",)

    start = chicago_text(period.get("start_chicago"))
    end = chicago_text(period.get("cutoff_chicago"))
    return FundedResultView(
        headline=HEADLINE_SENTENCE,
        question=str(result.get("question") or ""),
        period_text=f"{start} to {end} (Chicago time)",
        period_start_utc=period.get("start_utc"),
        period_end_utc=period.get("cutoff_utc"),
        months=tuple(period.get("months") or ()),
        strategy_text=strategy_text,
        settings_text=settings_text,
        status=tuple(status),
        is_sample=is_sample,
        approximation=approximate,
        validation_passed=validation_passed,
        firms=firms,
        comparison=comparison,
        comparison_note=(
            "Each firm is its own operation with its own five monthly credits, costs and "
            "payout cash. The columns are never added together. Five credits buy "
            + " versus ".join(
                format_usd(summaries[f.firm_key].get("next_five_cost_cents")) + f" ({f.firm})"
                for f in firms)
            + " of accounts, so compare the cost-adjusted row next to the absolute cash."
        ) if firms else "",
        limitations=limitations,
        decisions=decisions,
    )


def headline_figures(view: FundedResultView) -> dict[str, dict[str, int]]:
    """Exact integers per firm shown on the primary cards (screen/export check)."""

    out: dict[str, dict[str, int]] = {}
    for firm in view.firms:
        figures: dict[str, int] = {}
        for card in firm.cards:
            figures.update(dict(card.figures))
        out[firm.firm_key] = figures
    return out


def _date_text(value: Any) -> str:
    if not value:
        return "—"
    try:
        day = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    return f"{day.strftime('%B')} {day.day}, {day.year}"


def _comparison_row(row: dict[str, Any], firm_names: list[str]) -> ComparisonRow:
    unit = row.get("unit", "")

    def fmt(value: Any) -> str:
        if value is None:
            return "Not available"
        if unit == "USD":
            return format_usd(_cents(value))
        if unit == "ratio":
            return f"{float(value):,.2f} dollars per dollar of costs"
        return f"{value:,}" if isinstance(value, int) else str(value)

    diff = row.get("difference_first_minus_second")
    if diff is None:
        diff_text = "Not available"
    elif unit == "USD":
        diff_text = format_usd(_cents(diff), signed=True)
    elif unit == "ratio":
        diff_text = f"{float(diff):+,.2f}"
    else:
        diff_text = f"{diff:+,}" if isinstance(diff, int) else str(diff)
    unit_text = {"USD": "US dollars", "count": "accounts", "ratio": "ratio"}.get(unit, unit)
    return ComparisonRow(
        measure=str(row.get("measure", "")),
        unit=unit_text,
        values=tuple((name, fmt(row.get(name))) for name in firm_names),
        difference=diff_text,
    )


def _firm_view(
    key: str,
    s: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    evidence_labels: dict[str, str],
    singular: str,
    plural: str,
) -> FirmView:
    name = str(s["firm"])

    def mine(table: str) -> list[dict[str, Any]]:
        return [r for r in tables.get(table, []) if r.get("firm_key") == key]

    lost_before = int(s["accounts_lost_before_first_payout"])
    lost_after = int(s["accounts_lost_after_a_payout"])
    received_count = int(s["payouts_received_count"])
    cards = (
        Card("net_cash_earned_cents", "Net cash earned after all account costs",
             format_usd(s["net_cash_earned_cents"]),
             "Payouts received minus every account purchase. Pending payouts are not "
             "included.",
             (("net_cash_earned_cents", int(s["net_cash_earned_cents"])),)),
        Card("payouts_received_cents", "Payouts received after the firm's share",
             format_usd(s["payouts_received_cents"]),
             f"{received_count} payout(s) received; the trader keeps "
             f"{s['trader_share_pct']}% of each gross withdrawal.",
             (("payouts_received_cents", int(s["payouts_received_cents"])),)),
        Card("largest_single_payout_cents", "Largest single received payout",
             format_usd(s["largest_single_payout_cents"]) if received_count
             else "No payout received",
             "After the firm's share. Counts even if the account was later lost.",
             (("largest_single_payout_cents", int(s["largest_single_payout_cents"])),)),
        Card("acquisition_costs_cents", "Account costs",
             format_usd(s["acquisition_costs_cents"]),
             f"{s['accounts_bought']} account(s) at {format_usd(s['cost_per_account_cents'])} "
             f"each: {format_usd(s['acquisition_costs_from_credits_cents'])} with monthly "
             f"credits, {format_usd(s['acquisition_costs_from_wallet_cents'])} from payout "
             "cash.",
             (("acquisition_costs_cents", int(s["acquisition_costs_cents"])),)),
        Card("accounts_lost", "Accounts lost",
             f"{lost_before} before first payout · {lost_after} after a payout",
             "An account that paid out and was later lost still keeps its received payouts.",
             (("accounts_lost_before_first_payout", lost_before),
              ("accounts_lost_after_a_payout", lost_after))),
    )

    processing_n = int(s["payouts_processing_at_cutoff"])
    facts = (
        Fact("Accounts active at the end / account capacity",
             f"{s['accounts_active_at_end']} of {s['capacity_at_end']}",
             "Accounts paused for a payout still occupy a slot."),
        Fact("Unused purchase credits", str(s["unused_credits_at_end"]),
             "Monthly allowance for buying accounts; credits are not income."),
        Fact("Payout cash available (wallet)", format_usd(s["payout_wallet_at_end_cents"]),
             "Received payouts minus accounts bought with payout cash."),
        Fact("Payouts requested but still processing at the end",
             f"{format_usd(s['pending_gross_at_cutoff_cents'])} gross → "
             f"{format_usd(s['pending_after_split_at_cutoff_cents'])} after the firm's share "
             f"({processing_n} request(s))",
             "Not received by the end of the period and not counted in net cash."),
        Fact("Payouts secured but not yet requested at the end",
             str(s["secured_not_requested_at_cutoff"]),
             "Eligibility reached, but the end-of-day request had not happened yet."),
        Fact("Price of the next five accounts",
             format_usd(s["next_five_cost_cents"]),
             f"Growth needs a payout wallet of at least {format_usd(s['growth_threshold_cents'])} "
             "(the next five may cost at most 25% of it) and positive net cash."),
        Fact("Five-account growth purchases made", str(s["growth_purchases"])),
        Fact("Trading profit still inside funded accounts",
             format_usd(s["trading_profit_inside_accounts_cents"]),
             "NOT received cash — it stays in the accounts (including the retained $2,100 "
             "cushion) and is lost if an account fails."),
        Fact("Gross withdrawals requested over the whole period",
             format_usd(s["gross_withdrawals_requested_cents"]),
             f"Firm's share kept: {format_usd(s['firm_share_cents'])}."),
        Fact("Replacement slots waiting for a credit at the end",
             str(s["vacancies_waiting_at_end"])),
        Fact("Trades taken (all accounts)", f"{s['trades_taken']:,}",
             f"Position limit: {s['max_minis']:g} {plural} or equivalent."),
    )

    waiting = (
        WaitRow("Payout protection — finished for the day after securing a payout",
                float(s["hours_payout_protection"]), _hours(s["hours_payout_protection"]),
                int(s["entries_blocked_payout_protection"]),
                f"{s['entries_blocked_payout_protection']} signal(s) skipped"),
        WaitRow("Payout processing — trading paused until the payment arrives",
                float(s["hours_payout_processing"]), _hours(s["hours_payout_processing"]),
                int(s["entries_blocked_payout_processing"]),
                f"{s['entries_blocked_payout_processing']} signal(s) skipped"),
        WaitRow("Waiting for a monthly credit to replace a lost account",
                float(s["hours_waiting_for_credit"]), _hours(s["hours_waiting_for_credit"]),
                int(s["signals_missed_waiting_for_credit"]),
                f"{s['signals_missed_waiting_for_credit']} signal(s) missed by empty slots"),
        WaitRow("Ready, market closed or mandatory daily close / weekend lock",
                float(s.get("hours_ready_market_closed_or_locked", 0.0)),
                _hours(s.get("hours_ready_market_closed_or_locked", 0.0)), None, "—"),
        WaitRow("Ready, market open, no new strategy entry",
                float(s.get("hours_ready_market_open_no_new_signal",
                            s["hours_ready_without_position"])),
                _hours(s.get("hours_ready_market_open_no_new_signal",
                             s["hours_ready_without_position"])), None, "—"),
        WaitRow("In a trade", float(s["hours_trading"]), _hours(s["hours_trading"]),
                int(s["trades_taken"]), f"{s['trades_taken']} trade(s)"),
    )

    journeys = mine("account_journeys")
    numbers = {j["account_id"]: int(j["account_number"]) for j in journeys}
    payouts = mine("payout_events")
    events = mine("account_events")
    trades_raw = mine("trades")

    pending_due: dict[str, str] = {}
    received_ids = {r.get("request_id") for r in payouts if r.get("event") == "received"}
    for r in payouts:
        if r.get("event") == "requested" and r.get("request_id") not in received_ids:
            pending_due[r["account_id"]] = chicago_text(r.get("due_chicago"))

    replaced_by: dict[str, str] = {}
    for j in journeys:
        if j.get("replaces_account"):
            replaced_by[j["replaces_account"]] = _account_label(j["account_id"], numbers)

    trades_by_account: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in trades_raw:
        trades_by_account[t["account_id"]].append(t)

    accounts = tuple(
        _account_row(j, numbers, payouts, events, trades_by_account[j["account_id"]],
                     pending_due, replaced_by)
        for j in journeys
    )

    trade_rows: list[TradeRow] = []
    per_account_count: dict[str, int] = defaultdict(int)
    for t in trades_raw:
        per_account_count[t["account_id"]] += 1
        trade_rows.append(_trade_row(t, per_account_count[t["account_id"]], numbers,
                                     evidence_labels, singular, plural))

    months = tuple(
        MonthRow(
            firm=name, month=m["month"], label=_month_label(m["month"], bool(m["partial_month"])),
            partial=bool(m["partial_month"]),
            received_usd=float(m["received_usd"] or 0),
            account_costs_usd=float(m["account_costs_usd"] or 0),
            net_usd=float(m["net_usd"] or 0),
            cumulative_net_usd=float(m["cumulative_net_usd"] or 0),
            payouts_received=int(m["payouts_received"]),
            accounts_bought=int(m["accounts_bought"]),
        )
        for m in mine("monthly_results")
    )
    cash = tuple(
        CashPoint(name, c["ts_utc"], float(c["received_usd"]), float(c["account_costs_usd"]),
                  float(c["net_cash_usd"]))
        for c in mine("cash_over_time")
    )

    notices: list[str] = []
    if received_count == 0:
        notices.append("No payouts were received in this period.")
    if processing_n:
        notices.append(
            f"{processing_n} payout request(s) were still processing at the end of the "
            f"period ({format_usd(s['pending_after_split_at_cutoff_cents'])} after the firm's "
            "share). They are not counted as received.")
    if s["secured_not_requested_at_cutoff"]:
        notices.append(
            f"{s['secured_not_requested_at_cutoff']} account(s) had secured a payout that was "
            "not yet requested when the period ended. Not counted as received.")
    if int(s["accounts_active_at_end"]) == 0:
        notices.append("Every account was lost by the end of the period.")
    if int(s["net_cash_earned_cents"]) < 0:
        notices.append(
            f"Account costs exceeded received payouts: net cash is "
            f"{format_usd(s['net_cash_earned_cents'])}.")
    if not trades_raw:
        notices.append("No trades were taken by this firm's accounts in this period.")

    return FirmView(
        firm_key=key, firm=name, cards=cards, facts=facts, waiting=waiting,
        notices=tuple(notices), accounts=accounts, trades=tuple(trade_rows),
        months=months, cash=cash,
    )


def _status_key(label: str) -> str:
    for key, text in (
        ("failed", "Account lost"),
        ("processing", "Payout processing"),
        ("secured", "Payout secured"),
        ("in_trade", "Trading"),
        ("ready", "Ready to trade"),
    ):
        if label.startswith(text):
            return key
    return "ready"


def _account_row(
    j: dict[str, Any],
    numbers: dict[str, int],
    payouts: list[dict[str, Any]],
    events: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    pending_due: dict[str, str],
    replaced_by: dict[str, str],
) -> AccountRow:
    account_id = j["account_id"]
    label = _account_label(account_id, numbers)
    status_label = str(j.get("status_at_end") or "")
    status_key = _status_key(status_label)
    if status_key == "processing":
        due = pending_due.get(account_id)
        status_label = (f"Payout processing — trading paused until {due}" if due
                        else "Payout processing — trading paused")
    elif status_key == "secured":
        status_label = "Payout secured — finished for today"

    own_payouts = [p for p in payouts if p.get("account_id") == account_id]
    requests = {p.get("request_id"): p for p in own_payouts if p.get("event") == "requested"}
    received = {p.get("request_id"): p for p in own_payouts if p.get("event") == "received"}
    pending_ids = {p.get("request_id") for p in own_payouts
                   if p.get("event") == "processing_not_received_at_cutoff"}
    rows: list[PayoutRow] = []
    for request_id, req in requests.items():
        paid = received.get(request_id)
        if paid is not None:
            state = "Received"
            paid_or_due = f"Paid {chicago_text(paid.get('received_chicago'))}"
            after = _cents(paid.get("trader_usd"))
        elif request_id in pending_ids or req.get("after_cutoff"):
            state = "Processing, not received at the end"
            paid_or_due = f"Due {chicago_text(req.get('due_chicago'))}"
            after = _cents(req.get("trader_usd"))
        else:
            state = "Processing, not received at the end"
            paid_or_due = f"Due {chicago_text(req.get('due_chicago'))}"
            after = _cents(req.get("trader_usd"))
        rows.append(PayoutRow(
            state=state,
            secured=chicago_text(req.get("secured_chicago")),
            requested=chicago_text(req.get("requested_chicago")),
            paid_or_due=paid_or_due,
            gross=format_usd(_cents(req.get("gross_usd"))),
            firm_share=format_usd(_cents(req.get("firm_share_usd"))),
            after_split=format_usd(after),
            after_split_cents=after,
        ))
    for p in own_payouts:
        if p.get("event") == "secured_not_requested_at_cutoff":
            rows.append(PayoutRow(
                state="Secured, not requested at the end",
                secured=chicago_text(p.get("ts_chicago")), requested="—", paid_or_due="—",
                gross="—", firm_share="—", after_split="—", after_split_cents=None))
        elif p.get("event") == "request_shortfall":
            rows.append(PayoutRow(
                state="Request not possible at day end (balance fell below the minimum)",
                secured=chicago_text(p.get("ts_chicago")), requested="—", paid_or_due="—",
                gross="—", firm_share="—", after_split="—", after_split_cents=None))

    # Balance-versus-loss-limit path: every recorded state change, plus the
    # lowest open-position equity reached inside each trade.
    points: list[PathPoint] = []
    for e in events:
        if e.get("account_id") != account_id or e.get("event") in (
            "vacancy_opened", "vacancy_filled"):
            continue
        points.append(PathPoint(
            ts_utc=e["ts_utc"], when=chicago_text(e.get("ts_chicago")),
            balance_usd=e.get("balance_usd"), floor_usd=e.get("floor_usd"),
            what=_event_words(e)))
    for t in trades:
        if t.get("min_equity_utc") and t.get("min_equity_usd") is not None:
            points.append(PathPoint(
                ts_utc=t["min_equity_utc"], when=chicago_text(t.get("min_equity_chicago")),
                balance_usd=t["min_equity_usd"], floor_usd=None,
                what="Lowest open-position equity during the trade"))
    points.sort(key=lambda p: p.ts_utc)

    intervals: list[tuple[str, str]] = []
    start: str | None = None
    for e in events:
        if e.get("account_id") != account_id:
            continue
        after = e.get("status_after")
        if after == "processing" and start is None:
            start = e["ts_utc"]
        elif start is not None and after != "processing":
            intervals.append((start, e["ts_utc"]))
            start = None
    if start is not None:
        intervals.append((start, ""))

    receipts = tuple(
        (p["received_utc"], _cents(p.get("trader_usd")) or 0)
        for p in own_payouts if p.get("event") == "received" and p.get("received_utc")
    )
    lost = status_key == "failed"
    return AccountRow(
        account_id=account_id,
        number=int(j["account_number"]),
        label=label,
        slot=int(j.get("slot") or 0),
        funding=_FUNDING_TEXT.get(j.get("funding", ""), str(j.get("funding", ""))),
        created=chicago_text(j.get("created_chicago")),
        created_utc=j.get("created_utc"),
        status=status_label,
        status_key=status_key,
        lost=lost,
        lost_after_payout=bool(j.get("lost_after_payout")),
        failed=chicago_text(j.get("failed_chicago")) if j.get("failed_chicago") else "—",
        failed_utc=j.get("failed_utc"),
        failure_reason=(str(j["failure_reason"]).capitalize()
                        if j.get("failure_reason") else "—"),
        replaces=_account_label(j.get("replaces_account"), numbers)
        if j.get("replaces_account") else "—",
        replaced_by=replaced_by.get(account_id, "—"),
        replaces_id=j.get("replaces_account"),
        trades=int(j.get("trades") or 0),
        payouts_received=int(j.get("payouts_received") or 0),
        received=format_usd(_cents(j.get("received_usd"))),
        received_cents=_cents(j.get("received_usd")) or 0,
        largest_payout=format_usd(_cents(j.get("largest_payout_usd"))),
        gross_requested=format_usd(_cents(j.get("gross_requested_usd"))),
        final_balance=format_usd(_cents(j.get("final_balance_usd"))),
        final_floor=format_usd(_cents(j.get("final_floor_usd"))),
        payouts=tuple(rows),
        path=tuple(points),
        receipts_utc=receipts,
        processing_intervals_utc=tuple(intervals),
    )


def _event_words(e: dict[str, Any]) -> str:
    event = e.get("event", "")
    after = e.get("status_after")
    if event == "created":
        return "Account started fresh (no profit, $2,000 loss allowance)"
    if after == "in_trade":
        return "Entered a trade"
    if after == "secured":
        return "Payout secured — finished for today"
    if after == "processing":
        return "Full surplus requested; trading paused while the payout processes"
    if after == "failed":
        reason = e.get("reason") or "loss limit reached"
        return f"Account lost: {reason}"
    if e.get("status_before") == "processing":
        return "Payout received; trading may resume"
    if e.get("status_before") == "in_trade":
        return "Trade closed"
    return str(e.get("reason") or "Status changed").capitalize()


def _trade_row(
    t: dict[str, Any],
    index: int,
    numbers: dict[str, int],
    evidence_labels: dict[str, str],
    singular: str,
    plural: str,
) -> TradeRow:
    qty = int(t.get("quantity") or 0)
    direction = "Long" if t.get("direction") == "long" else "Short"
    entry, exit_ = t.get("entry_ticks"), t.get("exit_ticks")
    moved = None
    if entry is not None and exit_ is not None:
        sign = 1 if t.get("direction") == "long" else -1
        moved = Decimal(sign * (int(exit_) - int(entry))) * _TICK_POINTS
    evidence_key = str(t.get("price_evidence") or "")
    evidence_text = evidence_labels.get(evidence_key, "Price evidence not recorded")
    approximate = evidence_key.startswith("minute_bars")
    net = _cents(t.get("net_pnl_usd"))
    account = _account_label(t.get("account_id"), numbers)
    entry_time = chicago_text(t.get("entry_chicago"))
    label = (f"{account} · trade {index} · {entry_time} · {direction.lower()} "
             f"{qty} {singular if qty == 1 else plural} · {format_usd(net, signed=True)}")
    return TradeRow(
        account_id=str(t.get("account_id")),
        number=index,
        label=label,
        trading_day=_date_text(t.get("trading_day")),
        direction=direction,
        quantity=f"{qty} {singular if qty == 1 else plural}",
        entry_time=entry_time,
        exit_time=chicago_text(t.get("exit_chicago")),
        entry_price=format_points(entry),
        exit_price=format_points(exit_),
        stop_price=format_points(t.get("stop_ticks")),
        points_moved=f"{moved:+,.2f}" if moved is not None else "Not available",
        gross_result=format_usd(_cents(t.get("gross_pnl_usd")), signed=True),
        costs=format_usd(_cents(t.get("costs_usd"))),
        net_result=format_usd(net, signed=True),
        net_result_cents=net,
        initial_risk=format_usd(_cents(t.get("initial_risk_usd"))),
        lowest_open_equity=(
            f"{format_usd(_cents(t.get('min_equity_usd')))} at "
            f"{chicago_text(t.get('min_equity_chicago'))}"
            if t.get("min_equity_usd") is not None else "Not recorded"),
        close_reason=close_reason_text(t.get("exit_kind")),
        price_evidence=evidence_text,
        approximate=approximate,
        account_lost=bool(t.get("account_failed")),
    )
