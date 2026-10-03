"""Configuration detail · Payouts and accounts (mock 04).

Every earlier funded detail panel for ONE configuration at ONE firm, kept and
reorganized as the mock shows: payout timing and spending; where the account
time went (stacked bar and refused entries); payouts by month (bars and the full
month table with the running net); cumulative payouts, account costs and net
cash by day; the account replacement history; and "Show one account" — that
account's balance against its loss limit by trade, a plain explanation of how it
was lost, and its payouts.

Everything is read from the saved result (``present_pair_detail`` and the saved
tables). No stored money figure is recomputed: amounts are the saved cents or
saved dollar rows, formatted. The explanation of a lost account is derived from
the saved account events, trades and loss-limit evidence, never written by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import streamlit as st

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import css_var, palette

__all__ = [
    "AccountPoint",
    "AccountStory",
    "PayoutsView",
    "TimePart",
    "account_path",
    "account_story",
    "build_payouts_view",
    "loss_limit_words",
    "money_spans",
    "money_ticks",
    "month_label",
    "month_ticks",
    "render",
    "time_parts",
    "timing_rows",
]

KEY = "ifvg_lab_v1_"
_MINUS = "−"

#: legend colors of "Where the account time went" (mock 04 source), as palette KEYS:
#: the value is looked up when the card or chart is drawn, so the cached view holds
#: no color and the page's own theme (light or dark) picks it
_TIME_COLOR_KEYS = {
    "trading": "ink",
    "protection": "blue_line",
    "processing": "blue",
    "closed": "control_border",
    "no_signal": "light_rule",
}

_v = css_var  # a palette color as its CSS variable, for inline styles


# ── pure pieces (tested without Streamlit) ────────────────────────────────


def _t(value: Any) -> Any:
    """A stored instant kept to the microsecond for display (the stored text is unchanged)."""

    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import utc_instant

    stamp = utc_instant(value) if value else None
    return None if stamp is None else stamp.floor("us")


def _long(value: Any) -> str:
    """``April 12, 2026, 7:07 PM`` (Chicago)."""

    return fmt.chicago_long(_t(value))


def _short(value: Any) -> str:
    """``Apr 12, 7:07 PM`` (Chicago)."""

    return fmt.chicago_short(_t(value))


def _usd(value: Any) -> float | None:
    return None if value is None or value == "" else float(value)


def _long_no_year(utc: Any) -> str:
    """``February 11, 10:27 AM`` (Chicago) — the year is shown elsewhere on the page."""

    text = _long(utc)
    if text == fmt.MISSING:
        return text
    month_day, _year, clock = text.split(", ", 2)
    return f"{month_day}, {clock}"


def _mono(text: str) -> str:
    return f'<span class="lab-mono">{h.esc(text)}</span>'


_MONEY = re.compile(r"[−+]?\$[\d,]+(?:\.\d{2})?")


def money_spans(text: str) -> h.Markup:
    """Escape ``text`` and set each amount in mono, never broken across lines."""

    escaped = str(h.esc(text))
    return h.Markup(_MONEY.sub(
        lambda m: f'<span class="lab-mono" style="white-space:nowrap">{m.group(0)}</span>',
        escaped))


def timing_rows(summary_cents: dict[str, Any], summary_usd: dict[str, Any],
                median_payout_cents: int | None) -> list[tuple[str, h.Markup]]:
    """"Payout timing and spending" rows, from the saved summary (exact cents)."""

    s, u = summary_cents, summary_usd
    first = u.get("first_payout_utc")
    worst_at = u.get("max_unrecovered_spending_utc")
    spending = _mono(fmt.money_cents(s.get("max_unrecovered_spending_cents")))
    if worst_at:
        spending += f", reached {h.esc(_long_no_year(worst_at))}"
    pending_n = int(s.get("payouts_processing_at_cutoff") or 0)
    pending = ("None" if not pending_n else
               f"{_mono(fmt.money_cents(s.get('pending_after_split_at_cutoff_cents')))} after "
               f"the split ({_mono(fmt.money_cents(s.get('pending_gross_at_cutoff_cents')))} "
               f"gross, {fmt.count(pending_n, 'request')} still processing)")
    worse = int(s.get("stop_exits_filled_worse_than_stop") or 0)
    stops = int(s.get("stop_exits") or 0)
    stop_text = f"{worse:,} of {stops:,}"
    if worse:
        stop_text += f" · {_mono(fmt.money_cents(s.get('stop_slippage_cents')))} in total"
    inside = s.get("profit_inside_live_account_cents")
    return [
        ("First payout received",
         h.Markup(h.esc(_long(first)) if first else "No payout received")),
        ("Most account spending not yet paid back", h.Markup(spending)),
        ("Median payout", h.Markup(_mono(fmt.money_cents(median_payout_cents,
                                                          missing="No payout received"))
                                   if median_payout_cents is not None
                                   else "No payout received")),
        ("Money still processing at the end", h.Markup(pending)),
        ("Payout secured but not requested at the end",
         h.Markup("Yes" if s.get("secured_not_requested_at_cutoff") else "No")),
        ("Profit still inside the live account",
         h.Markup(f"{_mono(fmt.money_cents(inside))} · not cash, lost if the account fails"
                  if inside is not None else h.placeholder("Not in this study's export"))),
        ("Live account at the end", h.esc(s.get("live_account_status_at_end") or fmt.MISSING)),
        ("Trades taken", h.Markup(_mono(f"{int(s.get('trades_taken') or 0):,}"))),
        ("Stop exits filled worse than the stop", h.Markup(stop_text)),
        ("Trades ended by the account loss limit", h.Markup(_mono(
            f"{int(s.get('account_liquidations_that_ended_a_strategy_trade') or 0):,}"))),
        ("Trades using the one-minute approximation", h.Markup(_mono(
            f"{int(s.get('trades_with_approximated_minutes') or 0):,}"))),
    ]


@dataclass(frozen=True)
class TimePart:
    key: str
    label: str
    hours: float
    color_key: str  # a palette key (``theme.COLORS``), never a color value


def time_parts(summary_usd: dict[str, Any]) -> list[TimePart]:
    """The five saved hour totals, in the mock's order (hours as stored)."""

    s, keys = summary_usd, _TIME_COLOR_KEYS
    return [
        TimePart("trading", "In a trade", float(s.get("hours_trading") or 0), keys["trading"]),
        TimePart("protection", "Done for the day after securing a payout",
                 float(s.get("hours_payout_protection") or 0), keys["protection"]),
        TimePart("processing", "Paused while a payout processed",
                 float(s.get("hours_payout_processing") or 0), keys["processing"]),
        TimePart("closed", "Market closed, daily close or weekend lock",
                 float(s.get("hours_ready_market_closed_or_locked") or 0), keys["closed"]),
        TimePart("no_signal", "Ready, market open, no signal",
                 float(s.get("hours_ready_no_strategy_signal") or 0), keys["no_signal"]),
    ]


def month_label(month: str, partial: bool, *, with_year: bool = False) -> str:
    """``January (partial)``; the year is added when the months span years."""

    year, number = month.split("-")
    names = ("January", "February", "March", "April", "May", "June", "July", "August",
             "September", "October", "November", "December")
    text = names[int(number) - 1] + (f" {year}" if with_year else "")
    return text + (" (partial)" if partial else "")


def loss_limit_words(profile: dict[str, Any] | None) -> str:
    """How the firm's loss limit moves, from its saved firm rules (chart label)."""

    update = (profile or {}).get("threshold_update")
    lock = int((profile or {}).get("floor_lock_cents") or 0)
    if update == "intraday_peak_equity":
        return "Loss limit (trails the account's high point)"
    if update == "session_close_balance":
        text = "Loss limit (rises with each day's closing balance"
        if lock:
            text += f", stops at {fmt.money_cents(lock, signed=True)}"
        return text + ")"
    return "Loss limit"


@dataclass(frozen=True)
class AccountPoint:
    label: str  # "Start", "Trade 1", …
    balance: float
    limit: float
    trade_result: float | None = None
    when_utc: str | None = None


def account_path(created: dict[str, Any] | None,
                 trades: list[dict[str, Any]]) -> list[AccountPoint]:
    """Balance and loss limit at the start and after each of the account's trades (saved)."""

    points: list[AccountPoint] = []
    if created is not None and created.get("balance_usd") is not None:
        points.append(AccountPoint("Start", float(created["balance_usd"]),
                                   float(created.get("floor_usd") or 0.0),
                                   when_utc=created.get("ts_utc")))
    elif trades:
        first = trades[0]
        points.append(AccountPoint("Start", float(first.get("balance_before_usd") or 0.0),
                                   float(first.get("floor_before_usd") or 0.0)))
    for number, trade in enumerate(trades, start=1):
        points.append(AccountPoint(f"Trade {number}", float(trade.get("balance_after_usd") or 0),
                                   float(trade.get("floor_after_usd") or 0),
                                   trade_result=_usd(trade.get("net_pnl_usd")),
                                   when_utc=trade.get("exit_utc")))
    return points


@dataclass(frozen=True)
class AccountStory:
    """The plain explanation beside one account's chart (derived from saved records)."""

    headline: str
    detail: str
    started: str


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _touch_word(comparator: str | None) -> str:
    return "went below" if comparator == "below" else "touched"


def account_story(journey: dict[str, Any], trades: list[dict[str, Any]],
                  events: list[dict[str, Any]], failure: dict[str, Any] | None,
                  profile: dict[str, Any] | None) -> AccountStory:
    """How this account ended, in two short paragraphs, from its saved records.

    ``trades`` are the account's funded trades in execution order; ``events`` its
    saved account events; ``failure`` its saved ``account_failure`` loss-limit
    evidence row (None when it was not lost or the row is absent).
    """

    replaces = journey.get("replaces_account_number")
    started = (f"Started {_long(journey.get('created_utc'))}"
               + (f", replacing Account {replaces}." if replaces else ", the first account."))
    balance = fmt.money(journey.get("final_balance_usd"))
    limit = fmt.money(journey.get("final_floor_usd"))
    paid = int(journey.get("payouts_received") or 0)
    if not journey.get("failed_utc"):
        status = str(journey.get("status_at_end") or "open").strip()
        headline = (f"Still open at the end ({status[:1].lower() + status[1:]}): balance "
                    f"{balance} against a loss limit of {limit}.")
        if paid:
            detail = (f"It received {fmt.count(paid, 'payout')}, "
                      f"{fmt.money(journey.get('received_usd'))} after the split; the largest "
                      f"was {fmt.money(journey.get('largest_payout_usd'))}.")
        else:
            detail = "It has not received a payout yet."
        return AccountStory(headline, detail, started)

    outcome = "Lost after receiving a payout" if paid else "Lost before any payout"
    headline = f"{outcome}: final balance {balance} against a loss limit of {limit}."
    failed_event = next((e for e in events if e.get("status_after") == "failed"), None)
    ref = (failed_event or {}).get("trade_ref") or (failure or {}).get("trade_ref")
    last = (next((t for t in trades if ref and t.get("trade_ref") == ref), None)
            or next((t for t in trades if t.get("account_failed")), None)
            or (trades[-1] if trades else None))
    reason = str(journey.get("failure_reason") or "").strip()
    if last is None:
        detail = (f"It was lost {_long(journey.get('failed_utc'))} before any trade"
                  + (f": {reason}." if reason else "."))
        return AccountStory(headline, detail, started)
    net = fmt.money(last.get("net_pnl_usd"), signed=True)
    first = f"Its last trade finished {net}."
    if reason and reason != "open-position equity reached the loss limit":
        return AccountStory(headline, f"{first} {_cap(reason)}.", started)
    touched = _touch_word((failure or {}).get("comparator"))
    half = (last.get("scale_out_ns") is not None
            and int(last.get("scale_out_quantity") or 0) > 0)
    floor_before = _usd(last.get("floor_before_usd"))
    floor_after = _usd(last.get("floor_after_usd"))
    raised = (floor_before is not None and floor_after is not None
              and floor_after > floor_before + 0.005)
    if raised:
        update = (profile or {}).get("threshold_update")
        how = ("During it the account's high point rose, the limit trailed up behind it"
               if update == "intraday_peak_equity"
               else "During it a daily close raised the limit")
        pull = ("the pullback on the remaining half" if half else "the pullback")
        return AccountStory(headline, f"{first} {how}, and {pull} {touched} the new limit.",
                            started)
    before = _usd(last.get("balance_before_usd"))
    cushion = (fmt.money(before - floor_before) if before is not None
               and floor_before is not None else None)
    if cushion is None:
        second = f"The open loss {touched} the loss limit."
    elif half:
        second = (f"It started {cushion} above the limit; after the half exit, the open loss "
                  f"on the remaining half {touched} it.")
    else:
        second = f"It started {cushion} above the limit, and the open loss {touched} it."
    return AccountStory(headline, f"{first} {second}", started)


@dataclass(frozen=True)
class AccountView:
    number: int
    lost: bool
    started_utc: str | None
    lost_utc: str | None
    payouts: int
    received_usd: float | None
    largest_usd: float | None
    trades: int
    reason: str
    status_at_end: str
    story: AccountStory
    path: tuple[AccountPoint, ...]
    payout_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PayoutsView:
    """Everything the tab shows for one configuration at one firm (read from the result)."""

    firm: str
    timing: tuple[tuple[str, h.Markup], ...]
    parts: tuple[TimePart, ...]
    waiting: str | None
    refused_protection: int
    refused_processing: int
    months: tuple[Any, ...]
    cash: tuple[Any, ...]
    period_end_utc: str | None
    accounts: tuple[AccountView, ...]
    limit_words: str
    notices: tuple[str, ...]
    received_cents: int
    costs_cents: int
    net_cents: int
    first_positive_utc: str | None = None
    #: every earlier fact with its note, in full: (label, value as saved, note)
    fact_notes: tuple[tuple[str, str, str], ...] = ()


def _payout_rows(number: int, payouts: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """The account's payout requests (the earlier per-account payouts table), in order."""

    own = [p for p in payouts if p.get("account_number") == number]
    received = {p.get("request_id"): p for p in own if p.get("event") == "received"}
    rows = []
    for req in own:
        if req.get("event") != "requested":
            continue
        paid = received.get(req.get("request_id"))
        rows.append({
            "state": "Received" if paid else "Processing, not received at the end",
            "secured": _short(req.get("secured_utc")),
            "requested": _short(req.get("requested_utc")),
            "paid": (f"Paid {_short(paid.get('received_utc'))}" if paid
                     else f"Due {_short(req.get('due_utc'))}"),
            "gross": fmt.money(req.get("gross_usd")),
            "firm_share": fmt.money(req.get("firm_share_usd")),
            "after_split": fmt.money(req.get("trader_usd")),
        })
    for p in own:
        if p.get("event") == "secured_not_requested_at_cutoff":
            rows.append({"state": "Secured, not requested at the end",
                         "secured": _short(p.get("ts_utc")), "requested": "—",
                         "paid": "—", "gross": "—", "firm_share": "—", "after_split": "—"})
    return tuple(rows)


def _instant_key(row: dict[str, Any], field_name: str) -> tuple[int, int]:
    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import utc_instant

    stamp = utc_instant(row.get(field_name)) if row.get(field_name) else None
    return (-1 if stamp is None else int(stamp.value), int(row.get("seq") or 0))


def build_payouts_view(study, configuration: str, firm_key: str) -> PayoutsView:
    """Read the saved result for one configuration at one firm (never recomputes money)."""

    from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (
        present_pair_detail,
    )
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import funded_row

    result = study.result
    detail = present_pair_detail(result, configuration, firm_key)
    summary = study.summary(configuration, firm_key) or {}
    summary_usd = study.summary_usd(configuration, firm_key) or {}
    row = funded_row(study, configuration, firm_key)
    settings = result.get("settings") or {}
    profile = next((p for p in settings.get("firm_profiles") or []
                    if p.get("firm_key") == firm_key), None)
    replacement = (settings.get("execution_model") or {}).get("replacement_policy")
    waiting = ("none · assumed instant"
               if replacement == "immediate_fresh_account_same_configuration" else None)

    trades = list(ordered_trades(study, configuration, firm_key))
    events = sorted(study.rows("account_events", configuration, firm_key),
                    key=lambda e: _instant_key(e, "ts_utc"))
    failures = [r for r in study.rows("rule_boundary_evidence", configuration, firm_key)
                if r.get("check") == "account_failure"]
    payouts = sorted(study.rows("payout_events", configuration, firm_key),
                     key=lambda p: _instant_key(p, "ts_utc"))
    journeys = sorted(study.rows("account_journeys", configuration, firm_key),
                      key=lambda j: int(j.get("account_number") or 0))
    accounts = []
    for journey in journeys:
        number = int(journey.get("account_number") or 0)
        own_trades = [t for t in trades if t.get("account_number") == number]
        own_events = [e for e in events if e.get("account_number") == number]
        created = next((e for e in own_events if e.get("event") == "created"), None)
        failure = next((f for f in failures if f.get("account_number") == number), None)
        paid = int(journey.get("payouts_received") or 0)
        accounts.append(AccountView(
            number=number, lost=bool(journey.get("failed_utc")),
            started_utc=journey.get("created_utc"), lost_utc=journey.get("failed_utc"),
            payouts=paid, received_usd=_usd(journey.get("received_usd")),
            largest_usd=_usd(journey.get("largest_payout_usd")) if paid else None,
            trades=int(journey.get("trades") or 0),
            reason=_cap(str(journey.get("failure_reason") or "")),
            status_at_end=str(journey.get("status_at_end") or ""),
            story=account_story(journey, own_trades, own_events, failure, profile),
            path=tuple(account_path(created, own_trades)),
            payout_rows=_payout_rows(number, payouts)))
    first_positive = next((c.ts_utc for c in detail.cash if c.net_cash_usd > 0), None)
    return PayoutsView(
        firm=detail.firm,
        timing=tuple(timing_rows(summary, summary_usd, row.median_payout_cents)),
        parts=tuple(time_parts(summary_usd)), waiting=waiting,
        refused_protection=int(summary.get("entries_refused_payout_protection") or 0),
        refused_processing=int(summary.get("entries_refused_payout_processing") or 0),
        months=detail.months, cash=detail.cash, period_end_utc=detail.period_end_utc,
        accounts=tuple(accounts), limit_words=loss_limit_words(profile),
        notices=tuple(fmt.display_words(n) for n in detail.notices),
        received_cents=int(summary.get("payouts_received_cents") or 0),
        costs_cents=int(summary.get("account_costs_cents") or 0),
        net_cents=int(summary.get("net_cash_earned_cents") or 0),
        first_positive_utc=first_positive,
        fact_notes=tuple((f.label, f.value, f.note)
                         for f in (*detail.headline, *detail.facts) if f.note))


# ── HTML panels ───────────────────────────────────────────────────────────


def _timing_card(view: PayoutsView) -> str:
    rows = "".join(
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:9px 0;'
        f'border-top:1px solid {_v("light_rule")};font-size:14px;line-height:1.45">'
        f'<div style="color:{_v("body_2")}">{h.esc(label)}</div><div>{h.esc(value)}</div></div>'
        for label, value in view.timing)
    return (f'<section class="lab lab-card" style="gap:4px">'
            '<div class="lab-card-title" style="margin-bottom:10px">Payout timing and spending'
            f"</div>{rows}</section>")


def _time_card(view: PayoutsView) -> str:
    total = sum(p.hours for p in view.parts)
    bar = "".join(
        f'<div title="{h.esc(p.label)}" style="width:{p.hours / total * 100:.2f}%;'
        f'background:{_v(p.color_key)}"></div>' for p in view.parts if total > 0 and p.hours > 0)
    legend = []
    for part in view.parts:
        border = (f";border:1px solid {_v('control_border')};box-sizing:border-box"
                  if part.key == "no_signal" else "")
        legend.append(
            '<div style="display:grid;grid-template-columns:18px 1fr 110px;gap:10px;'
            'align-items:center;font-size:14px">'
            f'<div style="width:14px;height:14px;background:{_v(part.color_key)};'
            f'border-radius:3px{border}"></div><div>{h.esc(part.label)}</div>'
            '<div class="lab-mono" style="text-align:right">'
            f"{h.esc(f'{part.hours:,.1f} h')}</div></div>")
    waiting = (h.esc(view.waiting) if view.waiting
               else h.placeholder("Not in this study's export"))
    legend.append(
        '<div style="display:grid;grid-template-columns:18px 1fr 110px;gap:10px;'
        f'align-items:center;font-size:14px;color:{_v("muted")}"><div></div>'
        f'<div>Waiting for a replacement account</div><div style="text-align:right">{waiting}'
        "</div></div>")
    refused = (
        f'<div style="border-top:1px solid {_v("light_rule")};padding-top:12px;display:flex;'
        'flex-direction:column;gap:8px;font-size:14px">'
        '<div style="font-weight:600">Entries refused</div>'
        '<div style="display:flex;justify-content:space-between">'
        '<span>While done for the day after a payout</span>'
        f'<span class="lab-mono">{view.refused_protection:,}</span></div>'
        '<div style="display:flex;justify-content:space-between">'
        '<span>While a payout processed</span>'
        f'<span class="lab-mono">{view.refused_processing:,}</span></div></div>')
    return (
        '<section class="lab lab-card" style="gap:12px">'
        '<div class="lab-card-title">Where the account time went</div>'
        f'<div style="font-size:14px;color:{_v("body_2")}">Summed over this configuration\'s '
        f"accounts, {total:,.0f} hours in all.</div>"
        '<div style="display:flex;height:28px;border-radius:6px;overflow:hidden;'
        f'background:{_v("grid")}">{bar}</div>{"".join(legend)}{refused}</section>')


def top_cards(view: PayoutsView) -> h.Markup:
    return h.Markup(
        '<div class="lab lab-grid" style="grid-template-columns:repeat(2,minmax(0,1fr));'
        f'gap:16px;align-items:start">{_timing_card(view)}{_time_card(view)}</div>')


def _month_bars(months) -> str:
    top = max((m.received_usd for m in months), default=0.0)
    cells, names = [], []
    for m in months:
        bold = ";font-weight:600" if top > 0 and m.received_usd == top else ""
        if m.received_usd > 0 and top > 0:
            height = max(2.0, m.received_usd / top * 150)
            bar = (f'<div style="width:100%;height:{height:.0f}px;background:{_v("blue")}">'
                   "</div>")
        else:
            bar = f'<div style="width:100%;height:2px;background:{_v("control_border")}"></div>'
        cells.append(
            '<div style="display:flex;flex-direction:column;align-items:center;gap:6px">'
            f'<div class="lab-mono" style="font-size:11px{bold}">'
            f"{h.esc(fmt.money_whole(m.received_usd))}</div>{bar}</div>")
        names.append(f"<div>{h.esc(month_label(m.month, False)[:3])}</div>")
    count = max(1, len(months))
    return (
        f'<div style="display:grid;grid-template-columns:repeat({count},minmax(0,1fr));'
        f'gap:10px;align-items:end;height:190px">{"".join(cells)}</div>'
        f'<div style="display:grid;grid-template-columns:repeat({count},minmax(0,1fr));'
        f'gap:10px;font-size:12px;color:{_v("body_2")};text-align:center">'
        f'{"".join(names)}</div>')


def months_card(view: PayoutsView) -> h.Markup:
    months = view.months
    if not months:
        return h.card(h.placeholder("No months in this period"), title="Payouts by month")
    years = {m.month[:4] for m in months}
    columns = [h.Column("month", "Month", width="22%"),
               h.Column("received", "Received", "right"),
               h.Column("payouts", "Payouts", "right"),
               h.Column("costs", "Account costs", "right"),
               h.Column("bought", "Accounts bought", "right"),
               h.Column("net", "Net this month", "right"),
               h.Column("so_far", "Net so far", "right")]
    rows = []
    for index, m in enumerate(months):
        so_far = fmt.money(m.cumulative_net_usd)
        rows.append(h.Row({
            "month": h.Markup('<span style="font-family:var(--lab-sans);white-space:nowrap">'
                              f"{h.esc(month_label(m.month, m.partial, with_year=len(years) > 1))}"
                              "</span>"),
            "received": fmt.money(m.received_usd), "payouts": f"{m.payouts_received:,}",
            "costs": fmt.money(m.account_costs_usd), "bought": f"{m.accounts_bought:,}",
            "net": fmt.money(m.net_usd),
            "so_far": h.Markup(f"<b>{h.esc(so_far)}</b>") if index == len(months) - 1
            else so_far}))
    table = h.table(columns, rows, plain=True, wrap=False)
    best = max(months, key=lambda m: m.received_usd)
    caption = (f"{month_label(best.month, False)} brought the most, "
               f"{fmt.money(best.received_usd)} from {fmt.count(best.payouts_received, 'payout')}"
               "; partial months are the first and last of the study period."
               if best.received_usd > 0 else "No payout was received in any month.")
    return h.Markup(
        '<section class="lab lab-card lab-grid" style="display:grid;'
        'grid-template-columns:minmax(0,340px) minmax(0,1fr);gap:28px">'
        '<div style="display:flex;flex-direction:column;gap:14px">'
        '<div class="lab-card-title">Payouts by month</div>'
        f'{_month_bars(months)}<div class="lab-line">{h.esc(caption)}</div></div>'
        f"<div>{table}</div></section>")


def account_table(view: PayoutsView) -> h.Markup:
    columns = [h.Column("account", "Account", width="90px"), h.Column("started", "Started"),
               h.Column("payouts", "Payouts", "right"), h.Column("received", "Received", "right"),
               h.Column("largest", "Largest", "right"), h.Column("trades", "Trades", "right"),
               h.Column("lost", "Lost"), h.Column("why", "Why it was lost")]
    rows = []
    for a in view.accounts:
        def bold(text: Any, live: bool = not a.lost) -> Any:
            return h.Markup(f"<b>{h.esc(text)}</b>") if live else text

        rows.append(h.Row({
            "account": bold(f"Account {a.number}"),
            "started": bold(_short(a.started_utc)),
            "payouts": bold(f"{a.payouts:,}"),
            "received": bold(fmt.money(a.received_usd)),
            "largest": bold(fmt.money(a.largest_usd)) if a.largest_usd is not None
            else h.Markup(f'<span style="color:{_v("muted")}">—</span>'),
            "trades": bold(f"{a.trades:,}"),
            "lost": _short(a.lost_utc) if a.lost else bold("Not lost"),
            "why": a.reason if a.lost and a.reason
            else h.Markup(f'<span style="color:{_v("muted")}">—</span>'),
        }))
    return h.table(columns, rows, plain=True, wrap=False)


def payout_table(account: AccountView) -> h.Markup:
    if not account.payout_rows:
        return h.note(f"Account {account.number} requested no payouts.")
    columns = [h.Column("state", "State"), h.Column("secured", "Secured"),
               h.Column("requested", "Requested"), h.Column("paid", "Paid or due"),
               h.Column("gross", "Gross withdrawal", "right"),
               h.Column("firm_share", "Firm's share", "right"),
               h.Column("after_split", "Received after the split", "right")]
    return h.table(columns, [h.Row(r) for r in account.payout_rows], plain=True, wrap=False)


# ── charts ────────────────────────────────────────────────────────────────


def cash_figure(view: PayoutsView):
    import plotly.graph_objects as go

    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import chicago_wall
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import style_chart

    stamps = [c.ts_utc for c in view.cash]
    received = [c.received_usd for c in view.cash]
    costs = [c.account_costs_usd for c in view.cash]
    net = [c.net_cash_usd for c in view.cash]
    if view.period_end_utc and stamps:
        stamps.append(view.period_end_utc)
        received.append(received[-1])
        costs.append(costs[-1])
        net.append(net[-1])
    x = [chicago_wall(stamp) for stamp in stamps]  # mixed stored precisions, one by one
    c = palette()  # the active theme's colors, read when the figure is built
    fig = go.Figure()
    hover = "%{x|%b %-d, %Y, %-I:%M %p}<br>%{fullData.name}: %{customdata}<extra></extra>"
    for name, values, line in (
            ("Payouts received", received, {"color": c["blue"], "width": 2.5}),
            ("Net cash", net, {"color": c["ink"], "width": 2}),
            ("Account costs", costs, {"color": c["orange"], "width": 2, "dash": "dash"})):
        fig.add_trace(go.Scatter(x=x, y=values, name=name, mode="lines",
                                 line={**line, "shape": "hv"},
                                 customdata=[fmt.money(v) for v in values],
                                 hovertemplate=hover))
    style_chart(fig, height=300, x_title="Date (Chicago)", y_title="Cumulative dollars")
    yticks = money_ticks(received + costs + net)
    fig.update_yaxes(tickvals=[v for v, _ in yticks], ticktext=[t for _, t in yticks])
    ticks = month_ticks(x)
    fig.update_xaxes(tickvals=[t for t, _ in ticks], ticktext=[label for _, label in ticks],
                     tickfont={"family": "'IBM Plex Sans', sans-serif", "size": 12})
    return fig


def month_ticks(stamps: list[Any]) -> list[tuple[Any, str]]:
    """One tick per month shown: the month's first moment, or the first point in a partial
    first month; labeled with the month's name."""

    import pandas as pd

    points = [pd.Timestamp(t) for t in stamps if t is not None]
    if not points:
        return []
    first, last = min(points), max(points)
    ticks = [(first, f"{first:%B}")]
    month = (first + pd.offsets.MonthBegin(1)).normalize()
    while month <= last:
        ticks.append((month, f"{month:%B}"))
        month = (month + pd.offsets.MonthBegin(1)).normalize()
    return ticks


def money_ticks(values: list[float], *, target: int = 5) -> list[tuple[float, str]]:
    """Round money ticks covering ``values``, labeled like ``−$1.5k`` (minus before $)."""

    import math

    low, high = min(values + [0.0]), max(values + [0.0])
    span = max(high - low, 1.0)
    raw = span / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    start = math.floor(low / step) * step
    ticks = []
    value = start
    while value <= high + step * 0.5:
        ticks.append((round(value, 2), fmt.money_short(value)))
        value += step
    return ticks


def account_figure(account: AccountView, limit_words: str):
    import plotly.graph_objects as go

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import style_chart

    points = account.path
    x = list(range(len(points)))
    labels = [p.label for p in points]
    balance = [p.balance for p in points]
    limit = [p.limit for p in points]
    custom = [[p.label, fmt.money(p.trade_result, signed=True) if p.trade_result is not None
               else "—", _short(p.when_utc), fmt.money(p.balance), fmt.money(p.limit)]
              for p in points]
    c = palette()  # the active theme's colors, read when the figure is built
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=limit, name=limit_words, mode="lines",
        line={"color": c["orange"], "width": 2, "dash": "dash", "shape": "hv"},
        customdata=custom,
        hovertemplate="%{customdata[0]}: loss limit %{customdata[4]}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=balance, name="Account balance", mode="lines",
        line={"color": c["ink"], "width": 2.5, "shape": "hv"}, customdata=custom,
        hovertemplate=("%{customdata[0]} (%{customdata[2]}): balance %{customdata[3]}"
                       "<br>Trade result %{customdata[1]}<extra></extra>")))
    if account.lost and points:
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[balance[-1]], mode="markers", name="Lost here",
            marker={"color": c["orange"], "size": 11},
            hovertemplate=(f"Lost {_short(account.lost_utc)}: balance "
                           f"{fmt.money(balance[-1])}<extra></extra>")))
    trades = len(points) - 1
    step = 1 if trades <= 10 else next(s for s in (2, 5, 10, 20, 25, 50, 100, 10**6)
                                       if trades / s <= 8)
    ticks = [i for i in x if i == 0 or i % step == 0]
    style_chart(fig, height=280, x_title="Trades on this account",
                y_title="Balance and loss limit (dollars)")
    yticks = money_ticks(balance + limit)
    fig.update_yaxes(tickvals=[v for v, _ in yticks], ticktext=[t for _, t in yticks])
    fig.update_xaxes(tickvals=ticks, ticktext=[labels[i] for i in ticks], showgrid=False,
                     tickangle=0, tickfont={"family": "'IBM Plex Sans', sans-serif", "size": 12})
    if points:
        middle = x[len(x) // 2] if len(x) > 2 else x[-1]
        fig.add_annotation(x=middle, y=limit[middle], text=limit_words, showarrow=False,
                           yshift=-12, font={"color": c["orange"], "size": 12},
                           xanchor="center")
    return fig


# ── Streamlit ─────────────────────────────────────────────────────────────


def _view(ctx) -> PayoutsView:
    return _cached_view(ctx.store_root, ctx.result_id, ctx.configuration, ctx.firm_key)


@st.cache_resource(show_spinner="Reading the saved payouts and accounts…", max_entries=32)
def _cached_view(store_root: str, result_id: str, configuration: str,
                 firm_key: str) -> PayoutsView:
    """One immutable view per configuration and firm of a saved result (read only)."""

    from ifvg_lab_ui import funded_study

    return build_payouts_view(funded_study(store_root, result_id), configuration, firm_key)


def _earlier_time_words(text: str) -> str:
    """``February 11, 2026 10:27 AM CST`` → ``February 11, 2026, 10:27 AM`` (rule 11)."""

    import re

    return re.sub(r"(\d{4}) 0?(\d{1,2}:\d{2} [AP]M) C[SD]T", r"\1, \2", text)


def fact_notes_panel(view: PayoutsView) -> h.Markup:
    """The earlier screen's fact notes, word for word (rule 12: nothing disappears)."""

    rows = "".join(
        '<div style="display:grid;grid-template-columns:minmax(180px,1fr) minmax(120px,0.6fr) '
        f'2.4fr;gap:14px;padding:9px 0;border-top:1px solid {_v("light_rule")};font-size:14px;'
        f'line-height:1.5"><div style="font-weight:500">{h.esc(label)}</div>'
        f'<div class="lab-mono">{h.esc(_earlier_time_words(value))}</div>'
        f'<div style="color:{_v("body")}">{h.esc(note.replace("**", ""))}</div></div>'
        for label, value, note in view.fact_notes)
    return h.Markup(f'<div class="lab">{rows}</div>')


def _cash_caption(view: PayoutsView) -> str:
    text = (f"Payouts received reached {fmt.money_cents(view.received_cents)} against "
            f"{fmt.money_cents(view.costs_cents)} of account costs, leaving "
            f"{fmt.money_cents(view.net_cents)} net cash at the end")
    if view.first_positive_utc:
        text += (f"; net cash first rose above $0 on "
                 f"{_long(view.first_positive_utc).rsplit(', ', 1)[0]}")
    return text + "."


def _account_picker(st_module, ctx, view: PayoutsView) -> AccountView:
    numbers = [a.number for a in view.accounts]
    pair = f"{ctx.configuration}|{ctx.firm_key}"
    remembered = ctx.context.get("account") or {}
    index = (numbers.index(remembered["number"])
             if remembered.get("pair") == pair and remembered.get("number") in numbers else 0)
    chosen = st_module.selectbox(
        "Show one account", numbers, index=index, format_func=lambda n: f"Account {n}",
        key=f"{KEY}account_{ctx.result_id[:16]}_{ctx.configuration}_{ctx.firm_key}",
        help="Shows that account's balance against its loss limit after each trade, how it "
             f"ended and its payouts. The choice is kept for this configuration at {ctx.firm} "
             "only.")
    ctx.context["account"] = {"pair": pair, "number": chosen}
    return view.accounts[numbers.index(chosen)]


def render(st_module, ctx) -> None:
    from ifvg_lab_ui import show

    view = _view(ctx)
    for notice in view.notices:
        show(h.note(notice), st_module)
    show(top_cards(view), st_module)
    show(months_card(view), st_module)
    try:
        _cumulative_and_accounts(st_module, ctx, view)
    finally:
        _more(st_module, view)


def _more(st_module, view: PayoutsView) -> None:
    from ifvg_lab_ui import show

    if not view.fact_notes:
        return
    with st_module.expander("More: what each figure means (the earlier notes, in full)"):
        show(h.Markup('<div class="lab-line">How each payout and account figure is measured, '
                      "as the earlier results screen explained it.</div>"), st_module)
        show(fact_notes_panel(view), st_module)


def _cumulative_and_accounts(st_module, ctx, view: PayoutsView) -> None:
    from ifvg_lab_ui import plot, show

    with st_module.container(key="ifvg_lab_card_payouts_cumulative"):
        legend = (
            f'<div style="display:flex;gap:18px;font-size:13px;color:{_v("body_2")};'
            'flex-wrap:wrap">'
            '<span style="display:flex;align-items:center;gap:6px"><span style="width:18px;'
            f'height:3px;background:{_v("blue")}"></span>Payouts received</span>'
            '<span style="display:flex;align-items:center;gap:6px"><span style="width:18px;'
            f'height:3px;background:{_v("ink")}"></span>Net cash</span>'
            '<span style="display:flex;align-items:center;gap:6px"><span style="width:18px;'
            f'height:0;border-top:3px dashed {_v("orange")}"></span>Account costs</span></div>')
        show(h.Markup(
            '<div class="lab" style="display:flex;justify-content:space-between;'
            'align-items:baseline;gap:16px;flex-wrap:wrap"><div class="lab-card-title">'
            f"Cumulative payouts, account costs and net cash</div>{legend}</div>"), st_module)
        if view.cash:
            plot(cash_figure(view), key=f"cash_{ctx.configuration}_{ctx.firm_key}",
                 st_module=st_module)
            show(h.Markup(f'<div class="lab-line">{h.esc(_cash_caption(view))}</div>'),
                 st_module)
        else:
            show(h.placeholder("No cash movements were recorded for this configuration."),
                 st_module)

    with st_module.container(key="ifvg_lab_card_payouts_accounts"):
        show(h.Markup('<div class="lab-card-title">Account replacement history</div>'),
             st_module)
        if not view.accounts:
            show(h.placeholder("No accounts were purchased."), st_module)
            return
        show(account_table(view), st_module)
        show(h.Markup(f'<div style="border-top:1px solid {_v("light_rule")};margin-top:4px">'
                      "</div>"), st_module)
        left, right = st_module.columns([1, 2.3], gap="large")
        with left:
            account = _account_picker(st_module, ctx, view)
            story = account.story
            show(h.Markup(
                '<div class="lab" style="display:flex;flex-direction:column;gap:10px;'
                f'font-size:14px;line-height:1.55;color:{_v("body")}">'
                f"<div>{money_spans(story.headline)}</div><div>{money_spans(story.detail)}</div>"
                f'<div class="lab-line">{h.esc(story.started)}</div></div>'), st_module)
        with right:
            show(h.Markup(
                '<div class="lab" style="display:flex;gap:18px;font-size:13px;'
                f'color:{_v("body_2")};flex-wrap:wrap;justify-content:flex-end">'
                '<span style="display:flex;align-items:center;gap:6px"><span style="width:18px;'
                f'height:3px;background:{_v("ink")}"></span>Account balance</span>'
                '<span style="display:flex;align-items:center;gap:6px"><span style="width:18px;'
                f'height:0;border-top:3px dashed {_v("orange")}"></span>'
                f"{h.esc(view.limit_words)}</span>"
                + ('<span style="display:flex;align-items:center;gap:6px"><span style="width:'
                   f'10px;height:10px;border-radius:50%;background:{_v("orange")}"></span>'
                   "Where it was lost</span>" if account.lost else "") + "</div>"), st_module)
            plot(account_figure(account, view.limit_words),
                 key=f"account_{ctx.configuration}_{ctx.firm_key}_{account.number}",
                 st_module=st_module)
            show(h.Markup(
                f'<div class="lab-line">Account {account.number}: balance and loss limit after '
                f"each of its {fmt.count(len(account.path) - 1, 'trade')}; payouts come out "
                "between trades.</div>"), st_module)
        show(h.Markup(f'<div class="lab-h3" style="font-size:18px;margin-top:8px">'
                      f"Account {account.number} payouts</div>"), st_module)
        show(payout_table(account), st_module)
