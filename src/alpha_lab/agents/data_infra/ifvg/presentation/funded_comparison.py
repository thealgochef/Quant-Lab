"""Pure presenter for a completed funded configuration comparison.

``present_comparison`` and ``present_pair_detail`` turn the ONE saved
comparison result (``funded_comparison_result_v1``, built by
:mod:`alpha_lab.propsim.funded.comparison_result`) into frozen, display-ready
views for the completed-results screen and the review-folder export. They only
READ the result: headline money comes from ``result["summaries_cents"]``
(exact integer cents) and detail rows from the saved tables. Nothing here
recomputes money; it formats, selects and sorts.

:func:`comparison_headline_figures` exposes the exact integers the comparison
table shows, so the export can prove screen/export agreement.

Every configuration-and-firm pair is its own comparison: nothing here adds
configurations or firms together. Text is plain English with full firm names,
full chart-timeframe names (as saved in the configuration settings) and Chicago
12-hour times. No hashes, internal identities, file paths, serialized settings,
schema labels or developer counters reach the view.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.funded_results import (
    DecisionRow,
    StatusLine,
    chicago_datetime,
    chicago_text,
    close_reason_text,
    format_points,
    format_usd,
)

__all__ = [
    "COMPARISON_HEADLINE_KEYS",
    "FUTURE_NOTE",
    "ComparisonRow",
    "FirmTable",
    "ComparisonView",
    "Fact",
    "MonthRow",
    "CashPoint",
    "TimeRow",
    "PayoutRow",
    "BalancePoint",
    "AccountRow",
    "TradeRow",
    "PairDetail",
    "cash_chart_points",
    "comparison_headline_figures",
    "present_comparison",
    "present_pair_detail",
    "completed_configurations",
    "configuration_columns",
    "execution_sources",
    "proxy_micro_positions",
    "plain_configuration_label",
    "plain_reason",
]

#: The exact integers each comparison-table row shows (cents for money, plain
#: counts for accounts). The export compares configuration_results.csv with these.
COMPARISON_HEADLINE_KEYS = (
    "net_cash_earned_cents",
    "payouts_received_cents",
    "account_costs_cents",
    "largest_single_payout_cents",
    "accounts_purchased",
    "accounts_lost_before_first_payout",
    "accounts_lost_after_a_payout",
)

COMPLETED = "Completed"
NOT_COMPLETED = "Not completed"

FUTURE_NOTE = (
    "This is one simulated historical path. It compares configurations on that path only; "
    "it is not a probability of future payouts or of future success."
)

_DECISION_STATUS = {
    "owner_confirmed": "Confirmed by the owner",
    "owner_approved_for_pilot": "Approved by the owner",
    "assumption": "Assumption — not established by a published source",
}

_PURPOSE_TEXT = {
    "historical_comparison": "Owner-authorized historical comparison",
    "engineering_sample": "Engineering sample",
}

_TICK_POINTS = Decimal("0.25")
_EXCEPTION_PREFIX = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Warning)\s*:\s*")
_LOCATION = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/(?:Users|home|tmp)/)\S*")


# ── small helpers ─────────────────────────────────────────────────────────


def _cents(usd: Any) -> int | None:
    if usd is None or usd == "":
        return None
    return int((Decimal(str(usd)) * 100).quantize(Decimal(1)))


def _usd_text(usd: Any, *, signed: bool = False) -> str:
    return format_usd(_cents(usd), signed=signed)


def _hours_text(hours: Any) -> str:
    value = float(hours or 0)
    return f"{value:,.1f} hours"


def _count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n:,} {singular if n == 1 else (plural or singular + 's')}"


def plain_configuration_label(label: str | None) -> str:
    """The saved display name as plain text (``a | b`` becomes ``a · b``)."""

    text = str(label or "Unnamed configuration")
    return " · ".join(part.strip() for part in text.split("|") if part.strip())


def plain_reason(reason: str | None) -> str:
    """A failure reason without exception class names or file locations."""

    if not reason:
        return "No reason was recorded."
    text = _EXCEPTION_PREFIX.sub("", str(reason).strip().splitlines()[0])
    text = _LOCATION.sub("(internal location omitted)", text).strip()
    if not text:
        return "No reason was recorded."
    text = text[:1].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _month_label(month: str, partial: bool) -> str:
    year, number = month.split("-")
    name = datetime(int(year), int(number), 1).strftime("%B")
    return f"{name} {year}" + (" (partial month)" if partial else "")


def _date_text(value: Any) -> str:
    if not value:
        return "—"
    try:
        day = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    return f"{day.strftime('%B')} {day.day}, {day.year}"


def _firm_order(result: dict[str, Any]) -> list[tuple[str, str]]:
    """(firm_key, firm name) in the saved profile order, then any others."""

    order: list[tuple[str, str]] = []
    seen: set[str] = set()
    for profile in (result.get("settings") or {}).get("firm_profiles") or []:
        key = profile.get("firm_key")
        if key and key not in seen:
            seen.add(key)
            order.append((key, str(profile.get("firm_name") or key)))
    for summary in (result.get("summaries_cents") or {}).values():
        key = summary.get("firm_key")
        if key and key not in seen:
            seen.add(key)
            order.append((key, str(summary.get("firm") or key)))
    return order


# ── headline figures (screen/export agreement) ────────────────────────────


def comparison_headline_figures(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Exact integers per configuration-and-firm pair shown in the comparison.

    Keys are the result's internal pair keys (never displayed). A pair that did
    not complete carries only its status: it has no figures, never zeros.
    """

    out: dict[str, dict[str, Any]] = {}
    for pair_key, s in (result.get("summaries_cents") or {}).items():
        if s.get("status") == COMPLETED:
            out[pair_key] = {"status": COMPLETED,
                             **{k: int(s[k]) for k in COMPARISON_HEADLINE_KEYS}}
        else:
            out[pair_key] = {"status": NOT_COMPLETED}
    return out


# ── comparison view ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ComparisonRow:
    pair_key: str  # internal selection key, never displayed
    configuration: str  # internal selection key, never displayed
    firm_key: str
    firm: str
    completed: bool
    rank: int | None
    rank_text: str
    label: str
    net_cash: str
    payouts: str
    account_costs: str
    largest_payout: str
    accounts_purchased: str
    lost_before_payout: str
    lost_after_payout: str
    reason: str
    figures: tuple[tuple[str, int], ...]
    #: the settings that differ between configurations, as short readable columns
    columns: tuple[tuple[str, str], ...] = ()

    def display(self) -> dict[str, str]:
        """One row of the business comparison table (plain column names)."""

        return {
            "Rank": self.rank_text,
            "Net cash earned": self.net_cash,
            **(dict(self.columns) if self.columns else {"Configuration": self.label}),
            "Payouts received after the split": self.payouts,
            "Account costs": self.account_costs,
            "Largest payout": self.largest_payout,
            "Accounts purchased": self.accounts_purchased,
            "Lost before a payout": self.lost_before_payout,
            "Lost after a payout": self.lost_after_payout,
        }


@dataclass(frozen=True)
class FirmTable:
    firm_key: str
    firm: str
    rows: tuple[ComparisonRow, ...]
    note: str


@dataclass(frozen=True)
class ComparisonView:
    title: str
    question: str
    period_text: str
    period_start_utc: str | None
    period_end_utc: str | None
    firms_text: str
    size_text: str
    configurations_text: str
    evidence_text: str
    status: tuple[StatusLine, ...]
    is_sample: bool
    validation_passed: bool
    firm_tables: tuple[FirmTable, ...]
    limitations: tuple[str, ...]
    future_note: str
    decisions: tuple[DecisionRow, ...]


_SHORT_SETTING = {
    "Entry hours": "Entry hours",
    "Direction": "Direction",
    "Higher-timeframe gap charts": "Gap charts",
    "Supporting (parent) charts": "Parent charts",
    "Largest distance from the parent gap to the opposing gap": "Largest opposing distance",
    "Smallest opposing gap": "Smallest opposing gap",
    "Profit target": "Target",
    "Exit rule": "Exit",
    "Position size": "Size",
}


def _short_value(setting: str, value: str) -> str:
    if setting == "Entry hours":
        return value.split(": ", 1)[0]
    if setting == "Profit target":
        found = re.match(r"(\d+(?:\.\d+)?) times", value)
        return f"{found.group(1)}R" if found else ("1R" if "1 to 1" in value else value)
    if setting == "Higher-timeframe gap charts":
        return value if " and " in value else f"{value} only"
    if setting == "Supporting (parent) charts":
        if "one-minute" not in value:
            return "without one-minute"
        return ("with one- and three-minute" if "three-minute" in value
                else "with one-minute, no three-minute")
    if setting == "Exit rule":
        return "Half at 1R, rest held" if value.startswith("Half") else "Whole position"
    if setting == "Position size":
        found = re.match(r"^(\d+) x .*\((\w+)\)", value)
        return f"{found.group(1)} {found.group(2)}" if found else value
    return value


def configuration_columns(result: dict[str, Any]) -> dict[str, tuple[tuple[str, str], ...]]:
    """Per configuration: the settings that differ across this result, short and readable.

    Empty when fewer than two configurations were saved (the full label is used).
    """

    rows = (result.get("tables") or {}).get("configurations") or []
    settings = {c["configuration"]: {str(i.get("setting")): str(i.get("value"))
                                     for i in c.get("settings") or []} for c in rows}
    if len(settings) < 2:
        return {}
    names: list[str] = []
    for values in settings.values():
        for name in values:
            if name not in names:
                names.append(name)
    varying = [n for n in names if len({v.get(n) for v in settings.values()}) > 1]
    if not varying:
        return {}
    return {key: tuple((_SHORT_SETTING.get(n, n), _short_value(n, values.get(n, "—")))
                       for n in varying)
            for key, values in settings.items()}


def execution_sources(result: dict[str, Any], configuration: str) -> tuple[tuple[str, str], ...]:
    """Traded product, signal source, mark source and execution price source."""

    settings = result.get("settings") or {}
    sizing = (settings.get("sizing_by_configuration") or {}).get(configuration) or {
        "instrument": settings.get("instrument"),
        "instrument_label": settings.get("instrument_label"),
        "quantity": settings.get("quantity")}
    nq = "E-mini Nasdaq-100 (NQ)"
    micro = sizing.get("instrument") == "micro"
    traded = str(sizing.get("instrument_label") or nq)
    quantity = sizing.get("quantity")
    return (
        ("Traded product", traded + (f", {quantity} contracts" if quantity else "")),
        ("Signal source", f"{nq} one-minute candles (the strategy study's own data)"),
        ("Open-position marks and loss-limit checks", f"{nq} recorded exchange trades"),
        ("Execution prices", (
            f"{nq} recorded exchange trades used as a proxy for micro fills; no micro "
            "trade data was used (disclosed limitation)") if micro
         else f"{nq} recorded exchange trades (same product)"),
    )


def proxy_micro_positions(result: dict[str, Any]) -> int:
    """Funded positions traded in micros but priced on the mini's recorded trades."""

    sizing = (result.get("settings") or {}).get("sizing_by_configuration") or {}
    micro = {k for k, v in sizing.items() if v.get("instrument") == "micro"}
    return sum(1 for t in (result.get("tables") or {}).get("trades") or []
               if t.get("configuration") in micro)


def completed_configurations(result: dict[str, Any]) -> list[tuple[str, str]]:
    """(configuration key, plain label) for configurations with results, in saved order."""

    return [(c["configuration"], plain_configuration_label(c.get("configuration_label")))
            for c in (result.get("tables") or {}).get("configurations") or []]


def _comparison_row(pair_key: str, s: dict[str, Any], figures: dict[str, Any],
                    columns: dict[str, tuple[tuple[str, str], ...]] | None = None
                    ) -> ComparisonRow:
    label = plain_configuration_label(s.get("configuration_label"))
    names = next(iter((columns or {}).values()), ())
    own = (columns or {}).get(str(s.get("configuration")))
    base = {"pair_key": pair_key, "configuration": str(s.get("configuration")),
            "firm_key": str(s.get("firm_key")), "firm": str(s.get("firm")), "label": label,
            "columns": own or tuple((n, label if i == 0 else "—")
                                    for i, (n, _v) in enumerate(names))}
    if figures.get("status") != COMPLETED:
        reason = plain_reason(s.get("reason"))
        text = f"Not completed: {reason}"
        return ComparisonRow(
            **base, completed=False, rank=None, rank_text="—", net_cash=NOT_COMPLETED,
            payouts=NOT_COMPLETED, account_costs=NOT_COMPLETED, largest_payout=NOT_COMPLETED,
            accounts_purchased=NOT_COMPLETED, lost_before_payout=NOT_COMPLETED,
            lost_after_payout=NOT_COMPLETED, reason=text, figures=())
    received_count = int(s.get("payouts_received_count") or 0)
    rank = s.get("rank_within_firm")
    return ComparisonRow(
        **base, completed=True, rank=None if rank is None else int(rank),
        rank_text="—" if rank is None else str(rank),
        net_cash=format_usd(figures["net_cash_earned_cents"]),
        payouts=(f"{format_usd(figures['payouts_received_cents'])} "
                 f"({_count(received_count, 'payout')})"),
        account_costs=format_usd(figures["account_costs_cents"]),
        largest_payout=(format_usd(figures["largest_single_payout_cents"]) if received_count
                        else "No payout received"),
        accounts_purchased=f"{figures['accounts_purchased']:,}",
        lost_before_payout=f"{figures['accounts_lost_before_first_payout']:,}",
        lost_after_payout=f"{figures['accounts_lost_after_a_payout']:,}",
        reason="",
        figures=tuple((k, int(figures[k])) for k in COMPARISON_HEADLINE_KEYS),
    )


def present_comparison(result: dict[str, Any]) -> ComparisonView:
    """Build the frozen comparison view from the one saved comparison result."""

    summaries: dict[str, dict[str, Any]] = result.get("summaries_cents") or {}
    figures = comparison_headline_figures(result)
    settings = result.get("settings") or {}
    period = result.get("period") or {}
    evidence = result.get("price_evidence") or {}
    firms = _firm_order(result)
    columns = configuration_columns(result)

    tables = []
    for firm_key, firm_name in firms:
        rows = [_comparison_row(k, s, figures[k], columns) for k, s in summaries.items()
                if s.get("firm_key") == firm_key]
        rows.sort(key=lambda r: (not r.completed, r.rank if r.rank is not None else 10**9,
                                 r.label))
        done = sum(1 for r in rows if r.completed)
        note = (f"{_count(len(rows), 'configuration')} tested with {firm_name}; "
                f"{done:,} completed.")
        if done < len(rows):
            note += (f" {len(rows) - done:,} did not complete and "
                     "{} shown as not completed, never as zero.".format(
                         "is" if len(rows) - done == 1 else "are"))
        tables.append(FirmTable(firm_key=firm_key, firm=firm_name, rows=tuple(rows),
                                note=note))

    requested = int(result.get("configurations_requested") or 0)
    completed = int(result.get("configurations_completed") or 0)
    is_sample = result.get("purpose") == "engineering_sample"
    validation = result.get("validation")
    validation_passed = bool(validation and validation.get("passed"))

    checked = int(evidence.get("position_minutes_checked") or 0)
    exact = int(evidence.get("position_minutes_rebuilt_exactly_from_prints") or 0)
    approx = int(evidence.get("position_minutes_approximated") or max(0, checked - exact))
    missing_days = list(evidence.get("missing_print_days") or [])
    evidence_text = (
        f"{exact:,} of {checked:,} position minutes used recorded exchange trades; "
        f"{approx:,} used a labeled one-minute approximation (losing side first)."
    )
    if missing_days:
        evidence_text += f" Recorded trades were missing on {_count(len(missing_days), 'day')}."

    status: list[StatusLine] = [StatusLine(
        "info", "Simulated historical result — no real trades, account purchases or "
        "withdrawals took place.")]
    if is_sample:
        status.append(StatusLine(
            "warning", "Engineering sample — not an owner-approved study. It shows that the "
            "comparison works end to end; it is not the authorized historical result."))
    else:
        approval = result.get("approval") or {}
        purpose = _PURPOSE_TEXT.get(str(result.get("purpose")), "Historical comparison")
        if approval.get("approved_on"):
            status.append(StatusLine(
                "info", f"{purpose}, approved by the owner on "
                f"{_date_text(approval.get('approved_on'))}."))
        else:
            status.append(StatusLine("warning", f"{purpose}; no owner approval is recorded."))
    status.append(StatusLine("warning" if approx or missing_days else "info",
                             "Price evidence: " + evidence_text))
    if validation is None:
        status.append(StatusLine(
            "error", "This result has no internal money checks. Do not rely on these figures."))
    elif not validation_passed:
        status.append(StatusLine(
            "error", "The internal money checks did not pass. Do not rely on these figures "
            "until the result is corrected."))
    else:
        status.append(StatusLine(
            "success", "Internal money checks passed for every completed configuration and "
            "firm: receipts, account costs and account balances reconcile."))
    proxies = proxy_micro_positions(result)
    if proxies:
        status.append(StatusLine(
            "warning", f"Proxy micro executions: all {proxies:,} funded positions traded in "
            "Micro E-mini Nasdaq-100 (MNQ) contracts were priced on the E-mini Nasdaq-100 "
            "(NQ) recorded trades (signals, open-position marks and fills). No micro trade "
            "data was used. This is an accepted, disclosed limitation, not a fixed one."))
    for record in result.get("reporting_corrections") or []:
        status.append(StatusLine(
            "info", "Reporting correction (the saved result is unchanged): "
            + str(record.get("description", ""))))
    if requested > completed:
        status.append(StatusLine(
            "warning", f"{_count(requested - completed, 'configuration')} did not complete; "
            "each is listed as not completed with its reason, never as zero."))

    start = chicago_text(period.get("start_chicago"))
    end = chicago_text(period.get("cutoff_chicago"))
    quantity = settings.get("quantity")
    instrument = settings.get("instrument_label") or settings.get("instrument") or "contract"
    size_text = (f"{quantity} {instrument} contract{'s' if quantity != 1 else ''} per trade"
                 if quantity is not None else "Position size not recorded")
    if settings.get("size_text"):
        size_text = str(settings["size_text"])
    names = [name for _, name in firms]
    firms_text = (" and ".join(names) if len(names) <= 2
                  else ", ".join(names[:-1]) + " and " + names[-1])
    limitations = tuple(str(x) for x in result.get("limitations") or []) or (
        "No material limitations were saved with this result; treat it as incomplete "
        "evidence.",)
    decisions = tuple(
        DecisionRow(decided_on=_date_text(d.get("decided_on")),
                    subject=str(d.get("subject", "")), decision=str(d.get("decision", "")),
                    status=_DECISION_STATUS.get(d.get("status", ""), str(d.get("status", ""))))
        for d in result.get("owner_decisions") or [])
    return ComparisonView(
        title="Funded account comparison of strategy configurations",
        question=str(result.get("question") or ""),
        period_text=f"{start} to {end} (Chicago time)",
        period_start_utc=period.get("start_utc"),
        period_end_utc=period.get("cutoff_utc"),
        firms_text=(f"{firms_text}, each compared separately (one live funded account at a "
                    "time per configuration and firm)"),
        size_text=size_text,
        configurations_text=(f"{completed:,} of {_count(requested, 'configuration')} "
                             "completed"),
        evidence_text=evidence_text,
        status=tuple(status),
        is_sample=is_sample,
        validation_passed=validation_passed,
        firm_tables=tuple(tables),
        limitations=limitations,
        future_note=FUTURE_NOTE,
        decisions=decisions,
    )


# ── detail view (one configuration and firm) ──────────────────────────────


@dataclass(frozen=True)
class Fact:
    label: str
    value: str
    note: str = ""


@dataclass(frozen=True)
class MonthRow:
    month: str
    label: str
    partial: bool
    received_usd: float
    account_costs_usd: float
    net_usd: float
    cumulative_net_usd: float
    payouts_received: int
    accounts_bought: int

    def display(self) -> dict[str, str]:
        return {
            "Month": self.label,
            "Payouts received": _usd_text(self.received_usd),
            "Number of payouts": f"{self.payouts_received:,}",
            "Account costs": _usd_text(self.account_costs_usd),
            "Accounts bought": f"{self.accounts_bought:,}",
            "Net cash this month": _usd_text(self.net_usd),
            "Cumulative net cash": _usd_text(self.cumulative_net_usd),
        }


@dataclass(frozen=True)
class CashPoint:
    ts_utc: str
    when: str
    received_usd: float
    account_costs_usd: float
    net_cash_usd: float


@dataclass(frozen=True)
class TimeRow:
    reason: str
    hours: float
    hours_text: str
    note: str


@dataclass(frozen=True)
class PayoutRow:
    state: str
    secured: str
    requested: str
    paid_or_due: str
    gross: str
    firm_share: str
    after_split: str

    def display(self) -> dict[str, str]:
        return {"State": self.state, "Secured": self.secured, "Requested": self.requested,
                "Paid or due": self.paid_or_due, "Gross withdrawal": self.gross,
                "Firm's share": self.firm_share, "Received after the split": self.after_split}


@dataclass(frozen=True)
class BalancePoint:
    ts_utc: str
    when: str
    what: str
    balance_usd: float | None
    loss_limit_usd: float | None


@dataclass(frozen=True)
class AccountRow:
    number: int
    label: str
    created: str
    replaces: str
    status_at_end: str
    lost: bool
    lost_when: str
    failure_reason: str
    payout_outcome: str
    trades: int
    payouts_received: int
    received: str
    largest_payout: str
    final_balance: str
    final_loss_limit: str
    payouts: tuple[PayoutRow, ...]
    balance_path: tuple[BalancePoint, ...]

    def display(self) -> dict[str, str]:
        return {
            "Account": self.label,
            "Started": self.created,
            "Replaces": self.replaces,
            "Payouts received": f"{self.payouts_received:,}",
            "Received after the split": self.received,
            "Largest payout": self.largest_payout,
            "Trades": f"{self.trades:,}",
            "Lost": self.lost_when,
            "Why it was lost": self.failure_reason,
            "Outcome": self.payout_outcome,
            "Status at the end": self.status_at_end,
        }


@dataclass(frozen=True)
class TradeRow:
    number: int
    account: str
    direction: str
    quantity: str
    entry_time: str
    exit_time: str
    entry_price: str
    stop_price: str
    exit_price: str
    points_moved: str
    net_result: str
    initial_risk: str
    close_reason: str
    price_source: str
    approximate: bool
    stop_worse: str
    account_lost: bool

    def display(self) -> dict[str, str]:
        return {
            "Trade": str(self.number),
            "Account": self.account,
            "Direction": self.direction,
            "Quantity": self.quantity,
            "Entry (Chicago)": self.entry_time,
            "Exit (Chicago)": self.exit_time,
            "Entry price (points)": self.entry_price,
            "Stop price (points)": self.stop_price,
            "Exit price (points)": self.exit_price,
            "Points gained or lost": self.points_moved,
            "Result after costs": self.net_result,
            "Initial risk": self.initial_risk,
            "Exit reason": self.close_reason,
            "Prices from": self.price_source,
            "Stop filled worse than the stop": self.stop_worse,
        }


@dataclass(frozen=True)
class PairDetail:
    configuration: str  # internal selection key, never displayed
    firm_key: str
    firm: str
    label: str
    completed: bool
    reason: str
    rank_text: str
    settings: tuple[tuple[str, str], ...]
    headline: tuple[Fact, ...]
    facts: tuple[Fact, ...]
    months: tuple[MonthRow, ...]
    cash: tuple[CashPoint, ...]
    period_end_utc: str | None
    accounts: tuple[AccountRow, ...]
    time_split: tuple[TimeRow, ...]
    refused: tuple[Fact, ...]
    trades: tuple[TradeRow, ...]
    notices: tuple[str, ...]


def _instant(value: Any) -> int:
    """Sort key: the recorded instant in nanoseconds (never its text; repair R4).

    ISO text orders "17:43:00.67Z" before "17:43:00Z"; the instant does not.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import utc_instant

    instant = utc_instant(value) if value else None
    return -1 if instant is None else int(instant.value)


def _find_summary(result: dict[str, Any], configuration: str,
                  firm_key: str) -> tuple[str, dict[str, Any]] | None:
    for key, s in (result.get("summaries_cents") or {}).items():
        if s.get("configuration") == configuration and s.get("firm_key") == firm_key:
            return key, s
    return None


def present_pair_detail(result: dict[str, Any], configuration: str,
                        firm_key: str) -> PairDetail:
    """Detail view for one configuration and firm (read-only, exact figures)."""

    found = _find_summary(result, configuration, firm_key)
    if found is None:
        raise KeyError("this configuration and firm are not in the result")
    pair_key, s = found
    tables = result.get("tables") or {}
    firm_name = str(s.get("firm") or firm_key)
    label = plain_configuration_label(s.get("configuration_label"))
    config_row = next((c for c in tables.get("configurations") or []
                       if c.get("configuration") == configuration), None)
    plain_settings = tuple(
        (str(item.get("setting", "")), str(item.get("value", "")))
        for item in (config_row or {}).get("settings") or []) + execution_sources(
            result, configuration)
    period_end = (result.get("period") or {}).get("cutoff_utc")
    if s.get("status") != COMPLETED:
        return PairDetail(
            configuration=configuration, firm_key=firm_key, firm=firm_name, label=label,
            completed=False, reason=plain_reason(s.get("reason")), rank_text="—",
            settings=plain_settings, headline=(), facts=(), months=(), cash=(),
            period_end_utc=period_end, accounts=(), time_split=(), refused=(), trades=(),
            notices=(f"This configuration did not complete: {plain_reason(s.get('reason'))} "
                     "It has no figures; it is not a zero result.",))

    def mine(table: str) -> list[dict[str, Any]]:
        return [r for r in tables.get(table) or [] if r.get("pair_id") == pair_key]

    received_count = int(s["payouts_received_count"])
    headline = (
        Fact("Net cash earned", format_usd(s["net_cash_earned_cents"]),
             "Payouts received after the split minus every account purchase. Pending "
             "payouts are not included."),
        Fact("Payouts received after the split", format_usd(s["payouts_received_cents"]),
             _count(received_count, "payout") + " received."),
        Fact("Account costs", format_usd(s["account_costs_cents"]),
             _count(int(s["accounts_purchased"]), "account") + " purchased."),
        Fact("Largest payout", format_usd(s["largest_single_payout_cents"]) if received_count
             else "No payout received",
             "After the split. It counts even if the account was later lost."),
    )

    pending_n = int(s["payouts_processing_at_cutoff"])
    pending = (f"{format_usd(s['pending_after_split_at_cutoff_cents'])} after the split "
               f"({format_usd(s['pending_gross_at_cutoff_cents'])} gross, "
               f"{_count(pending_n, 'request')} still processing)") if pending_n else "None"
    worse = int(s["stop_exits_filled_worse_than_stop"])
    paid = sorted(_cents(p.get("trader_usd")) or 0 for p in mine("payout_events")
                  if p.get("event") == "received")
    middle = len(paid) // 2
    median = (None if not paid else paid[middle] if len(paid) % 2
              else (paid[middle - 1] + paid[middle]) // 2)
    first_payout = _summary_time(result, pair_key, "first_payout_chicago")
    shortfall_at = _summary_time(result, pair_key, "max_unrecovered_spending_chicago")
    facts = (
        Fact("First payout received",
             chicago_text(first_payout) if first_payout else "No payout received"),
        Fact("Largest unrecovered account spending",
             format_usd(s["max_unrecovered_spending_cents"])
             + (f" (reached {chicago_text(shortfall_at)})" if shortfall_at else ""),
             "The most account spending not yet covered by received payouts."),
        Fact("Median payout received",
             "No payout received" if median is None else format_usd(median),
             "The usual payment size; the largest payout is shown above."),
        Fact("Money pending at the cutoff", pending,
             "Requested but not received by the end of the period; not counted as cash."),
        Fact("Payout secured but not yet requested at the cutoff",
             "Yes" if s["secured_not_requested_at_cutoff"] else "No"),
        Fact("Trading profit or loss inside the live account at the end",
             format_usd(s["profit_inside_live_account_cents"]),
             "Not received cash; it would be lost if the account failed."),
        Fact("Live account at the end", str(s["live_account_status_at_end"])),
        Fact("Trades taken", f"{int(s['trades_taken']):,}"),
        Fact("Stop exits filled worse than the stop",
             f"{worse:,} of {int(s['stop_exits']):,}"
             + (f" ({format_usd(s['stop_slippage_cents'])} worse in total)" if worse else ""),
             "Descriptive: already inside each trade's result, counted for the contracts "
             "actually closed at that stop."),
        Fact("Trades whose account loss limit ended the trade",
             f"{int(s['account_liquidations_that_ended_a_strategy_trade']):,}"),
        Fact("Trades using the one-minute approximation",
             f"{int(s['trades_with_approximated_minutes']):,}"),
    )

    months = tuple(
        MonthRow(month=m["month"], label=_month_label(m["month"], bool(m.get("partial_month"))),
                 partial=bool(m.get("partial_month")),
                 received_usd=float(m.get("received_usd") or 0),
                 account_costs_usd=float(m.get("account_costs_usd") or 0),
                 net_usd=float(m.get("net_usd") or 0),
                 cumulative_net_usd=float(m.get("cumulative_net_usd") or 0),
                 payouts_received=int(m.get("payouts_received") or 0),
                 accounts_bought=int(m.get("accounts_bought") or 0))
        for m in mine("monthly_results"))
    cash = tuple(
        CashPoint(ts_utc=c["ts_utc"], when=chicago_text(c.get("ts_chicago")),
                  received_usd=float(c["received_usd"]),
                  account_costs_usd=float(c["account_costs_usd"]),
                  net_cash_usd=float(c["net_cash_usd"]))
        for c in mine("cash_over_time"))

    payouts = mine("payout_events")
    events = mine("account_events")
    accounts = tuple(_account_row(j, payouts, events) for j in sorted(
        mine("account_journeys"), key=lambda j: int(j["account_number"])))

    replacement = ((result.get("settings") or {}).get("execution_model") or {}).get(
        "replacement_policy")
    time_split = [
        TimeRow("In a trade", float(s["hours_trading"]), _hours_text(s["hours_trading"]), ""),
        TimeRow("Payout protection — finished for the day after securing a payout",
                float(s["hours_payout_protection"]), _hours_text(s["hours_payout_protection"]),
                "A deliberate pause, not a strategy drought."),
        TimeRow("Payout processing — paused until the payment arrives",
                float(s["hours_payout_processing"]), _hours_text(s["hours_payout_processing"]),
                "A deliberate pause, not a strategy drought."),
        TimeRow("Ready, but the market was closed or the daily close or weekend lock applied",
                float(s["hours_ready_market_closed_or_locked"]),
                _hours_text(s["hours_ready_market_closed_or_locked"]), ""),
        TimeRow("Ready, market open, no entry",
                float(s["hours_ready_no_strategy_signal"]),
                _hours_text(s["hours_ready_no_strategy_signal"]),
                "No strategy signal, or outside this configuration's entry hours."),
    ]
    if replacement == "immediate_fresh_account_same_configuration":
        time_split.append(TimeRow(
            "Waiting for a replacement account", 0.0, "None",
            "A replacement is assumed available at once (research simplification)."))
    refused = (
        Fact("Entries refused during payout protection",
             f"{int(s['entries_refused_payout_protection']):,}"),
        Fact("Entries refused during payout processing",
             f"{int(s['entries_refused_payout_processing']):,}"),
    )

    trades = tuple(_trade_row(i, t) for i, t in enumerate(
        sorted(mine("trades"), key=lambda t: (_instant(t.get("entry_utc")), t.get("seq") or 0)),
        start=1))

    notices: list[str] = []
    if received_count == 0:
        notices.append("No payouts were received in this period.")
    if int(s["net_cash_earned_cents"]) < 0:
        notices.append("Account costs exceeded received payouts: net cash is "
                       f"{format_usd(s['net_cash_earned_cents'])}.")
    if not trades:
        notices.append("No trades were taken in this period.")
    if pending_n:
        notices.append(f"{_count(pending_n, 'payout request')} still processing at the end "
                       f"({format_usd(s['pending_after_split_at_cutoff_cents'])} after the "
                       "split) — not counted as received.")
    rank = s.get("rank_within_firm")
    return PairDetail(
        configuration=configuration, firm_key=firm_key, firm=firm_name, label=label,
        completed=True, reason="", rank_text="—" if rank is None else str(rank),
        settings=plain_settings, headline=headline, facts=facts, months=months, cash=cash,
        period_end_utc=period_end, accounts=accounts, time_split=tuple(time_split),
        refused=refused, trades=trades, notices=tuple(notices))


def _summary_time(result: dict[str, Any], pair_key: str, field: str) -> str | None:
    return ((result.get("summaries") or {}).get(pair_key) or {}).get(field)


def _payout_rows(number: int, payouts: list[dict[str, Any]]) -> tuple[PayoutRow, ...]:
    own = [p for p in payouts if p.get("account_number") == number]
    received = {p.get("request_id"): p for p in own if p.get("event") == "received"}
    rows: list[PayoutRow] = []
    for req in own:
        if req.get("event") != "requested":
            continue
        paid = received.get(req.get("request_id"))
        if paid is not None:
            state, when = "Received", f"Paid {chicago_text(paid.get('received_chicago'))}"
        else:
            state = "Processing, not received at the end"
            when = f"Due {chicago_text(req.get('due_chicago'))}"
        rows.append(PayoutRow(
            state=state, secured=chicago_text(req.get("secured_chicago")),
            requested=chicago_text(req.get("requested_chicago")), paid_or_due=when,
            gross=_usd_text(req.get("gross_usd")), firm_share=_usd_text(req.get("firm_share_usd")),
            after_split=_usd_text(req.get("trader_usd"))))
    for p in own:
        if p.get("event") == "secured_not_requested_at_cutoff":
            rows.append(PayoutRow(
                state="Secured, not requested at the end",
                secured=chicago_text(p.get("ts_chicago")), requested="—", paid_or_due="—",
                gross="—", firm_share="—", after_split="—"))
    return tuple(rows)


def _event_words(e: dict[str, Any]) -> str:
    after, before = e.get("status_after"), e.get("status_before")
    if e.get("event") == "created":
        return "Account started fresh"
    if after == "in_trade":
        return "Entered a trade"
    if after == "secured":
        return "Payout secured — finished for the day"
    if after == "processing":
        return "Full surplus requested; paused while the payout processes"
    if after == "failed":
        return "Account lost: " + str(e.get("reason") or "loss limit reached")
    if before == "processing":
        return "Payout received; trading may resume"
    if before == "in_trade":
        return "Trade closed"
    return str(e.get("reason") or "Status changed").capitalize()


def _account_row(j: dict[str, Any], payouts: list[dict[str, Any]],
                 events: list[dict[str, Any]]) -> AccountRow:
    number = int(j["account_number"])
    lost = bool(j.get("failed_utc"))
    paid = int(j.get("payouts_received") or 0)
    if lost:
        outcome = "Lost after receiving a payout" if paid else "Lost before any payout"
    else:
        outcome = "Still open at the end"
    replaces = j.get("replaces_account_number")
    path = tuple(
        BalancePoint(ts_utc=e["ts_utc"], when=chicago_text(e.get("ts_chicago")),
                     what=_event_words(e), balance_usd=e.get("balance_usd"),
                     loss_limit_usd=e.get("floor_usd"))
        for e in sorted((e for e in events if e.get("account_number") == number),
                        key=lambda e: (_instant(e.get("ts_utc")), e.get("seq") or 0)))
    return AccountRow(
        number=number, label=f"Account {number}",
        created=chicago_text(j.get("created_chicago")),
        replaces=f"Replaces account {replaces}" if replaces else "First account",
        status_at_end=str(j.get("status_at_end") or "—"), lost=lost,
        lost_when=chicago_text(j.get("failed_chicago")) if lost else "Not lost",
        failure_reason=(str(j["failure_reason"])[:1].upper() + str(j["failure_reason"])[1:]
                        if j.get("failure_reason") else "—"),
        payout_outcome=outcome, trades=int(j.get("trades") or 0), payouts_received=paid,
        received=_usd_text(j.get("received_usd")),
        largest_payout=_usd_text(j.get("largest_payout_usd")) if paid else "No payout",
        final_balance=_usd_text(j.get("final_balance_usd")),
        final_loss_limit=_usd_text(j.get("final_floor_usd")),
        payouts=_payout_rows(number, payouts), balance_path=path)


def _trade_row(index: int, t: dict[str, Any]) -> TradeRow:
    qty = int(t.get("quantity") or 0)
    sign = 1 if t.get("direction") == "long" else -1
    entry, exit_, stop = t.get("entry_ticks"), t.get("exit_ticks"), t.get("stop_ticks")
    moved = (f"{Decimal(sign * (int(exit_) - int(entry))) * _TICK_POINTS:+,.2f}"
             if entry is not None and exit_ is not None else "Not available")
    scale = t.get("scale_out_ticks")
    if scale is not None and entry is not None and exit_ is not None:
        half = Decimal(sign * (int(scale) - int(entry))) * _TICK_POINTS
        moved = (f"{int(t.get('scale_out_quantity') or 0)} at {half:+,.2f}; "
                 f"{int(t.get('final_exit_quantity') or 0)} at {moved}")
    approx_minutes = int(t.get("minutes_approximated") or 0)
    print_minutes = int(t.get("minutes_on_prints") or 0)
    approximate = approx_minutes > 0 or bool(t.get("approximate_exit"))
    if approx_minutes:
        source = (f"One-minute approximation for {approx_minutes:,} of "
                  f"{approx_minutes + print_minutes:,} minutes")
    elif t.get("approximate_exit"):
        source = "Recorded trades; exit price approximated"
    else:
        source = "Recorded exchange trades"
    stop_worse = "—"
    if t.get("exit_kind") in ("stop", "breakeven_stop") and exit_ is not None:
        # the stop the closing contracts actually rested at (break-even after a half exit)
        reference = t.get("final_stop_ticks")
        reference = stop if reference is None else reference
        if reference is not None:
            gap = (int(exit_) - int(reference)) * sign
            stop_worse = (f"Yes, {Decimal(-gap) * _TICK_POINTS:,.2f} points worse"
                          if gap < 0 else "No")
    return TradeRow(
        number=index, account=f"Account {t.get('account_number')}",
        direction="Long" if sign == 1 else "Short",
        quantity=f"{qty:,} contract{'s' if qty != 1 else ''}",
        entry_time=chicago_text(t.get("entry_chicago")),
        exit_time=chicago_text(t.get("exit_chicago")),
        entry_price=format_points(entry), stop_price=format_points(stop),
        exit_price=format_points(exit_), points_moved=moved,
        net_result=_usd_text(t.get("net_pnl_usd"), signed=True),
        initial_risk=_usd_text(t.get("initial_risk_usd")),
        close_reason=_close_words(t), price_source=source,
        approximate=approximate, stop_worse=stop_worse,
        account_lost=bool(t.get("account_failed")))


def cash_chart_points(detail: PairDetail) -> list[tuple[datetime | None, float, float, float]]:
    """Chicago wall-clock points for the cumulative cash chart, held to the cutoff."""

    points = [(chicago_datetime(p.ts_utc), p.received_usd, p.account_costs_usd,
               p.net_cash_usd) for p in detail.cash]
    end = chicago_datetime(detail.period_end_utc)
    if points and end is not None and (points[-1][0] is None or points[-1][0] < end):
        points.append((end, *points[-1][1:]))
    return points


STRATEGY_METRICS_NOTE = (
    "Strategy measures come from each configuration's replay with no account limits "
    "(every signal taken), after a round-trip cost of 0.514 index points per contract, "
    "in multiples of the initial risk (R). They describe the strategy, not cash; the "
    "tables above are the funded-account results. The two use different execution rules: "
    "these measures follow Strategy-Core's candle rules (stop checked first; after a half "
    "exit the break-even stop is checked from the next candle), while the funded accounts "
    "follow the recorded exchange trades in order, so a half exit and a break-even stop can "
    "both happen inside one minute. R and cash are not two views of one execution path.")


def present_strategy_metrics(result: dict[str, Any]) -> list[dict[str, str]]:
    """Plain rows of the strategy-measures table, best net R first."""

    rows = list((result.get("tables") or {}).get("strategy_metrics") or [])
    rows.sort(key=lambda r: (-(r.get("net_r_after_costs") or -1e9), r["configuration"]))
    columns = configuration_columns(result)
    out = []
    for r in rows:
        pf = r.get("profit_factor")
        own = columns.get(r["configuration"])
        out.append({
            **(dict(own) if own else {
                "Configuration": plain_configuration_label(r.get("configuration_label"))}),
            "Trades": f"{r['trades']:,}",
            "Long / short": f"{r.get('long_trades', 0)} / {r.get('short_trades', 0)}",
            "Win rate": "—" if r.get("win_rate_pct") is None else f"{r['win_rate_pct']:.1f}%",
            "Net R after costs": f"{r['net_r_after_costs']:+.2f} R",
            "Average R per trade": ("—" if r.get("expectancy_r_per_trade") is None
                                    else f"{r['expectancy_r_per_trade']:+.3f} R"),
            "Profit factor": "—" if pf is None else f"{pf:.2f}",
            "Largest drawdown": f"{r['max_drawdown_r']:.2f} R",
            "Longest time under water": f"{r['longest_trading_days_under_water']} trading days",
        })
    return out


def _close_words(t: dict[str, Any]) -> str:
    kind = t.get("exit_kind")
    if t.get("scale_out_ticks") is None:
        return close_reason_text(kind)
    rest = {"breakeven_stop": "the rest at the break-even stop (entry price)",
            "scheduled_close": "the rest held to the daily close",
            "account_failure": "the rest closed when the account reached its loss limit",
            }.get(kind, f"the rest: {close_reason_text(kind).lower()}")
    return f"Half at the target, {rest}"
