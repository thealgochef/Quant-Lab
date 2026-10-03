"""Trade review (mocks 09, 09b): the words and values of one funded trade, Streamlit-free.

Everything comes from the saved funded trade row (prices in ticks, instants as
recorded) and, for the setup, from its linked saved setup record
(:mod:`.setup_records`). Point in time: every builder takes an optional
``moment`` (a UTC instant) and leaves out whatever happened after it —
fills, the exit, the result, balances, later setup steps, a later loss-limit
check and a replacement account (correction A7).

Setup timing (closeout review follow-up): every formation event has one
availability instant (:class:`SetupEvent`) that the steps text, the setup chart's
key and the moment picker share. A gap is known at its confirmation; a tap or a
close through the opposing gap is known when its candle closes, since the strategy
records both at the close and saves no earlier observation. A candle's opening
minute only names it ("the 7:05 PM candle").

Setup identity (correction A6): only the configuration's own record
(``record.identity_established``) names this configuration's formation history,
its first review step and its point-in-time moments. A related record (another
configuration's record of the same entry) is labelled as related context and
never changes the recorded fills, stops, exits, result or balances.

Price evidence (correction A10): :func:`evidence_note` adds the published
approximated-minute companion when it links to exactly this trade
(:mod:`.minute_companion`) and says honestly when it conflicts or is unavailable.

Review vocabulary: the overall choices map one-to-one onto the review ledger's
verdict keys (``review_vocabulary.VERDICT_LABELS``); the four step-by-step
judgments map onto the ledger's existing verdict fields; the tags map onto
ledger tag keys. Nothing here writes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import time, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import CHICAGO, utc_instant
from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab.minute_companion import (
    COMPANION_FILE,
    MinuteLink,
    short_cause,
)
from alpha_lab.agents.data_infra.ifvg.presentation.lab.setup_records import NO_RECORD, SetupRecord

__all__ = [
    "context_line",
    "known_accounts",
    "known_trades",
    "OVERALL_CHOICES",
    "clock",
    "short",
    "STEP_CHOICES",
    "STEPS",
    "TAGS",
    "Moment",
    "RecordedLine",
    "SetupEvent",
    "SetupStep",
    "STEP_TIMES_NOTE",
    "TradeView",
    "OWN_RECORD_HELP",
    "OWN_RECORD_STEPS",
    "day_close",
    "default_moment",
    "disabled_steps",
    "evidence_note",
    "exit_words",
    "failure_lines",
    "five_largest",
    "formed_card_title",
    "gap_legend",
    "moments",
    "recorded_lines",
    "related_source_line",
    "review_verdicts",
    "setup_card_title",
    "setup_events",
    "setup_key",
    "setup_steps",
    "step_label",
    "timeframe_words",
    "trade_label",
    "when",
]

# ── review vocabulary ─────────────────────────────────────────────────────

#: (ledger verdict key, label) in ledger order — the mock's five overall choices
OVERALL_CHOICES: tuple[tuple[str, str], ...] = (
    ("correct", "Correct — matches how I'd trade it"),
    ("incorrect", "Incorrect — I wouldn't take this"),
    ("questionable", "Needs investigation"),
    ("insufficient_evidence", "Unclear"),
    ("not_applicable", "Not applicable"),
)
#: step-by-step choice → ledger verdict ("Not reviewed" saves nothing for that field)
STEP_CHOICES: dict[str, str | None] = {
    "Not reviewed": None,
    "Agree": "correct",
    "Disagree": "incorrect",
    "Unclear": "insufficient_evidence",
}
#: (step, ledger verdict fields it fills)
STEPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("htf", ("htf_verdict",)),
    ("Parent gap and retest are right", ("parent_verdict",)),
    ("Entry and stop are right", ("entry_verdict", "stop_verdict")),
    ("Exit handling is right", ("outcome_verdict",)),
)
#: (label, ledger tag key)
TAGS: tuple[tuple[str, str], ...] = (
    ("A+ setup", "a_plus_setup"),
    ("Late entry", "late_entry"),
    ("Stop too tight", "stop_too_tight"),
    ("Stale parent", "stale_parent"),
    ("News time", "news_time"),
)


def _own(record: SetupRecord | None) -> bool:
    """The record is this configuration's own (identity established, correction A6)."""

    return record is not None and record.identity_established


def step_label(step: str, record: SetupRecord | None) -> str:
    """The first step names the higher-timeframe gap of this configuration's own record.

    A related record (identity not established) never names it: the step stays
    the generic "Higher-timeframe gap is valid" (correction A6).
    """

    if step != "htf":
        return step
    if _own(record) and record.htf is not None and record.htf.timeframe_seconds:
        return f"{timeframe_words(record.htf.timeframe_seconds).capitalize()} gap is valid"
    return "Higher-timeframe gap is valid"


#: steps that judge the setup's formation: they need this configuration's own record (A6)
OWN_RECORD_STEPS: tuple[str, ...] = ("htf", "Parent gap and retest are right")
OWN_RECORD_HELP = ("Needs this configuration's own setup record. The zones shown are related "
                   "context from another configuration, so this step can't be judged here and "
                   "nothing is saved for it.")


def disabled_steps(record: SetupRecord | None) -> tuple[str, ...]:
    """Steps that can't be judged: the formation steps when only a related record exists.

    With a related record (another configuration's, identity not established) the
    gap and parent steps would judge that configuration's setup; entry and stop and
    the exit remain this trade's recorded facts (correction A6). Without any record
    nothing is disabled.
    """

    if record is not None and not record.identity_established:
        return OWN_RECORD_STEPS
    return ()


def review_verdicts(overall: str | None, steps: Mapping[str, str], *,
                    disabled: Sequence[str] = ()) -> dict[str, str]:
    """Ledger verdicts from the form: the overall key plus each reviewed step's fields.

    Correction A8: a step left at "Not reviewed" writes nothing, so the saved
    value stays unknown (never "incorrect"). Entry and stop are written together
    only by an explicit choice on their combined step; showing that step writes
    nothing, and an earlier review's separate entry and stop values are never
    read back into the form. A ``disabled`` step (:func:`disabled_steps`) writes
    nothing whatever its control holds.
    """

    verdicts: dict[str, str] = {}
    if overall is not None:
        verdicts["overall_verdict"] = overall
    fields = dict(STEPS)
    for step, choice in steps.items():
        value = STEP_CHOICES.get(choice)
        if value is None or step in disabled:
            continue
        for name in fields.get(step, ()):
            verdicts[name] = value
    return verdicts


# ── the trade ─────────────────────────────────────────────────────────────


def _int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


def _float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


@dataclass(frozen=True)
class TradeView:
    """One saved funded trade, as the review screens read it (never changed)."""

    seq: int
    account: int | None
    direction: str
    quantity: int
    entry_utc: pd.Timestamp
    entry_ticks: int
    stop_ticks: int | None
    target_ticks: int | None
    half_utc: pd.Timestamp | None
    half_ticks: int | None
    half_quantity: int | None
    exit_utc: pd.Timestamp
    exit_ticks: int | None
    exit_quantity: int
    exit_kind: str
    final_stop_ticks: int | None
    net: float | None
    costs: float | None
    balance_before: float | None
    balance_after: float | None
    initial_risk: float | None
    trading_day: str
    minutes_on_prints: int | None
    minutes_approximated: int | None
    approximate_exit: bool
    account_failed: bool

    @property
    def long(self) -> bool:
        return self.direction != "short"

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> TradeView:
        half_ticks = _int(row.get("scale_out_ticks"))
        quantity = int(row.get("quantity") or 0)
        final_quantity = _int(row.get("final_exit_quantity"))
        return cls(
            seq=int(row.get("seq") or 0), account=_int(row.get("account_number")),
            direction=str(row.get("direction") or "long").lower(), quantity=quantity,
            entry_utc=utc_instant(row.get("entry_utc")), entry_ticks=int(row["entry_ticks"]),
            stop_ticks=_int(row.get("stop_ticks")), target_ticks=_int(row.get("target_ticks")),
            half_utc=utc_instant(row.get("scale_out_ns")) if half_ticks is not None else None,
            half_ticks=half_ticks,
            half_quantity=_int(row.get("scale_out_quantity")) if half_ticks is not None else None,
            exit_utc=utc_instant(row.get("exit_utc")), exit_ticks=_int(row.get("exit_ticks")),
            exit_quantity=final_quantity if final_quantity else quantity,
            exit_kind=str(row.get("exit_kind") or ""),
            final_stop_ticks=_int(row.get("final_stop_ticks")),
            net=_float(row.get("net_pnl_usd")), costs=_float(row.get("costs_usd")),
            balance_before=_float(row.get("balance_before_usd")),
            balance_after=_float(row.get("balance_after_usd")),
            initial_risk=_float(row.get("initial_risk_usd")),
            trading_day=str(row.get("trading_day") or ""),
            minutes_on_prints=_int(row.get("minutes_on_prints")),
            minutes_approximated=_int(row.get("minutes_approximated")),
            approximate_exit=bool(row.get("approximate_exit")),
            account_failed=bool(row.get("account_failed")),
        )

    def r_multiple(self, ticks: int | None) -> float | None:
        if ticks is None or self.stop_ticks is None or self.stop_ticks == self.entry_ticks:
            return None
        risk = abs(self.entry_ticks - self.stop_ticks)
        move = (ticks - self.entry_ticks) if self.long else (self.entry_ticks - ticks)
        return move / risk


def _known(at: Any, moment: pd.Timestamp | None) -> bool:
    return moment is None or (at is not None and at <= moment)


def _price(ticks: Any) -> str:
    return fmt.points(ticks, from_ticks=True)


def _mono(text: str, *, bold: bool = False) -> h.Markup:
    weight = ";font-weight:600" if bold else ""
    return h.Markup(f'<span class="lab-mono" style="white-space:nowrap{weight}">{h.esc(text)}'
                    "</span>")


def _local_date(at: pd.Timestamp) -> Any:
    return at.tz_convert(CHICAGO).date()


def _us(at: Any) -> Any:
    """Display precision: recorded nanoseconds are kept for comparisons, not for words."""

    return at.floor("us") if isinstance(at, pd.Timestamp) else at


def clock(at: Any) -> str:
    """``7:07 PM`` (Chicago)."""

    return fmt.chicago_clock(_us(at))


def short(at: Any) -> str:
    """``Apr 12, 7:07 PM`` (Chicago)."""

    return fmt.chicago_short(_us(at))


def when(at: pd.Timestamp | None, reference: pd.Timestamp | None = None) -> str:
    """``7:07 PM`` on the reference's Chicago date, else ``Apr 13, 3:55 PM``."""

    if at is None:
        return fmt.MISSING
    if reference is not None and _local_date(at) == _local_date(reference):
        return clock(at)
    return short(at)


def _r_text(value: float | None) -> str:
    if value is None:
        return ""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{text}R"


def exit_words(view: TradeView) -> str:
    """Why the (last part of the) position closed, in plain words."""

    kind = view.exit_kind
    if kind == "scheduled_close":
        return "scheduled daily close"
    if kind == "breakeven_stop":
        return "stop at entry (break-even)"
    if kind == "target":
        return "target reached"
    if kind == "account_failure":
        return "account loss limit reached"
    if kind == "stop":
        moved = (view.final_stop_ticks is not None and view.stop_ticks is not None
                 and view.final_stop_ticks != view.stop_ticks)
        return "moved stop hit" if moved else "stopped out"
    return kind.replace("_", " ") or "exit"


def _units(instrument: str | None, quantity: int) -> str:
    if instrument == "micro":
        return "micro" if quantity == 1 else "micros"
    if instrument == "mini":
        return "E-mini contract" if quantity == 1 else "E-mini contracts"
    return "contract" if quantity == 1 else "contracts"


def _moved_stop_words(view: TradeView) -> str:
    if view.final_stop_ticks is None:
        return "stop on the rest unchanged"
    if view.final_stop_ticks == view.entry_ticks:
        return "stop on the rest moved to entry"
    return f"stop on the rest moved to {_price(view.final_stop_ticks)}"


@dataclass(frozen=True)
class RecordedLine:
    label: str
    text: h.Markup
    at: pd.Timestamp | None = None


def recorded_lines(view: TradeView, *, instrument: str | None = None,
                   moment: pd.Timestamp | None = None,
                   extra: Sequence[tuple[str, str]] = ()) -> list[RecordedLine]:
    """"What was recorded": every fill with its Chicago time, the stop, result and balance.

    With a ``moment``, only what happened at or before it is returned; the exit,
    the result and the balance after the trade are left out.
    """

    buy, sell = ("Bought", "Sold") if view.long else ("Sold short", "Bought back")
    lines: list[RecordedLine] = []
    if not _known(view.entry_utc, moment):
        return lines
    lines.append(RecordedLine(
        when(view.entry_utc, view.entry_utc),
        h.Markup(f"{buy} {view.quantity:,} {_units(instrument, view.quantity)} at "
                 f"{_mono(_price(view.entry_ticks))}"), view.entry_utc))
    risk = (f" · risk {fmt.money(view.initial_risk)}" if view.initial_risk is not None else "")
    lines.append(RecordedLine("Initial stop", h.Markup(
        f"{_mono(_price(view.stop_ticks))}{h.esc(risk)}")))
    if view.half_utc is not None and _known(view.half_utc, moment):
        lines.append(RecordedLine(
            when(view.half_utc, view.entry_utc),
            h.Markup(f"{sell} {view.half_quantity or 0:,} at {_mono(_price(view.half_ticks))}"
                     f" · target reached, {h.esc(_moved_stop_words(view))}"), view.half_utc))
    if not _known(view.exit_utc, moment):
        return lines
    lines.append(RecordedLine(
        when(view.exit_utc, view.entry_utc),
        h.Markup(f"{sell} {view.exit_quantity:,} at {_mono(_price(view.exit_ticks))} · "
                 f"{h.esc(exit_words(view))}"), view.exit_utc))
    net = (_mono(fmt.money(view.net, signed=True), bold=True) if view.net is not None
           else h.placeholder("Not in this study's export"))
    costs = (f" after {fmt.money(view.costs)} costs" if view.costs is not None else "")
    lines.append(RecordedLine("Result", h.Markup(f"{net}{h.esc(costs)}")))
    if view.balance_before is not None and view.balance_after is not None:
        lines.append(RecordedLine(
            f"Account {view.account} balance" if view.account is not None else "Balance",
            _mono(f"{fmt.money(view.balance_before)} → {fmt.money(view.balance_after)}")))
    for label, text in extra:
        lines.append(RecordedLine(label, h.esc(text)))
    return lines


def failure_lines(tables: Mapping[str, Any], row: Mapping[str, Any], *,
                  moment: pd.Timestamp | None = None) -> list[tuple[str, str]]:
    """When the account's loss limit ended the trade: the recorded check and the replacement.

    Found exactly as the earlier funded reviewer found them (same pair, account and trade
    reference); written in this screen's words and Chicago times. With a ``moment``
    (point in time), a check or a replacement account after it is left out: a later
    account's start is not known yet (correction A7).
    """

    pair, account = row.get("pair_id"), row.get("account_number")
    entry = utc_instant(row.get("entry_utc"))
    lines: list[tuple[str, str]] = []
    failure = next((r for r in tables.get("rule_boundary_evidence") or []
                    if r.get("pair_id") == pair and r.get("account_number") == account
                    and r.get("check") == "account_failure"
                    and r.get("trade_ref") in (None, row.get("trade_ref"))), None)
    if failure and moment is not None and not _known(utc_instant(failure.get("ts_utc")), moment):
        failure = None
    if failure:
        comparator = {"at_or_below": "at or below", "below": "below"}.get(
            failure.get("comparator"), "against")
        text = when(utc_instant(failure.get("ts_utc")), entry)
        if failure.get("equity_usd") is not None and failure.get("floor_usd") is not None:
            text += (f" · equity {fmt.money(failure['equity_usd'])} {comparator} the loss "
                     f"limit {fmt.money(failure['floor_usd'])}")
        if failure.get("price_ticks") is not None:
            text += f", at {_price(failure['price_ticks'])}"
        if failure.get("detail"):
            text += f" ({failure['detail']})"
        lines.append(("Loss limit", text))
    replacement = next((e for e in tables.get("account_events") or []
                        if e.get("pair_id") == pair and e.get("event") == "created"
                        and e.get("replaces") in (account, f"{pair}#{account}")), None)
    if replacement and moment is not None and not _known(
            utc_instant(replacement.get("ts_utc")), moment):
        replacement = None
    if replacement:
        lines.append(("Replacement", f"Account {replacement.get('account_number')} started "
                                     f"{short(utc_instant(replacement.get('ts_utc')))}"))
    return lines


def five_largest(trades: Sequence[Mapping[str, Any]]) -> set[int]:
    """Sequences of the configuration's five largest trades by net result (ties by order)."""

    ranked = sorted(enumerate(trades), key=lambda item: (-float(item[1].get("net_pnl_usd") or 0),
                                                         item[0]))
    return {int(t.get("seq") or 0) for _, t in ranked[:5]}


def _minute_words(at: pd.Timestamp | None) -> str:
    """``April 13, 2:03 AM`` (Chicago)."""

    if at is None:
        return fmt.MISSING
    local = _us(at).tz_convert(CHICAGO)
    return f"{local:%B} {local.day}, {clock(at)}"


def _linked_minutes(link: MinuteLink) -> str:
    """Each linked minute: when, a short cause, and what its ordering can decide."""

    parts = []
    for row, at in zip(link.rows, link.minutes, strict=True):
        cause = short_cause(row.get("why_recorded_trades_did_not_rebuild_the_candle"))
        text = f"{_minute_words(at)} — {cause}." if cause else f"{_minute_words(at)}."
        decide = str(row.get("what_the_minute_can_decide") or "").strip().rstrip(".")
        parts.append(text + (f" {decide}." if decide else ""))
    return " ".join(parts)


def evidence_note(view: TradeView, link: MinuteLink | None = None) -> str:
    """The trade's price-evidence sentence: its saved row and, with ``link``, the
    published approximated-minute companion (correction A10).

    ``link`` is :func:`.minute_companion.link_from_review_folder` for this trade:
    ``linked`` names each approximated minute (Chicago), a short cause, what the
    ordering inside it can decide and the hash-checked source; ``conflict`` and
    ``unavailable`` say why no minute is shown; ``none`` (no approximated minute)
    leaves the approximation out. Without ``link`` the saved row alone is used.
    A trade row without its approximated-minute count never reads as zero.
    """

    on = view.minutes_on_prints
    approx = view.minutes_approximated
    linked = link is not None and link.status == "linked" and bool(link.rows)
    if on is None and approx is None and not linked:
        return "Price evidence for this trade is not in this study's export."
    count = approx if approx is not None else (len(link.rows) if linked else 0)
    text = f"{fmt.count(on or 0, 'minute')} on recorded exchange trades"
    unrecorded = "the number of approximated minutes isn't recorded for this trade"
    if approx is None and not linked and (link is None or link.status != "conflict"):
        if link is None:
            text += f"; {unrecorded}."
        elif link.status == "none":
            text += (f"; {unrecorded}, and the hash-checked published minute record lists none "
                     "for it.")
        else:
            text += (f"; {unrecorded}, and no hash-checked published minute record is "
                     "available for this result.")
    elif link is None:
        text += (f", {approx:,} approximated from the one-minute bar (which minute, and why, "
                 "isn't in this study's export)." if approx else ".")
    elif linked:
        source = link.source  # only a folder read through its manifest hash carries one
        where = (f"the published review folder (export {source.version} of this result), "
                 f"{COMPANION_FILE}, hash-checked" if source is not None
                 else f"{COMPANION_FILE} (not hash-checked)")
        text += (f", {count:,} approximated from the one-minute bar: {_linked_minutes(link)} "
                 f"Source: {where}.")
    elif link.status == "conflict":
        mismatch = (f"published minute record doesn't match this trade exactly "
                    f"({link.reason}), so it isn't shown.")
        if approx is None:  # an unrecorded count is said so, never read as zero (re-review)
            text += f"; {unrecorded}, and the {mismatch}"
        else:
            text += (f", {count:,} approximated from the one-minute bar; the {mismatch}"
                     if count else f". The {mismatch}")
    elif link.status == "unavailable" and count:
        text += (f", {count:,} approximated from the one-minute bar; no hash-checked published "
                 "minute record is available for this result.")
    else:
        text += "."
    text += (" The exit used the one-minute approximation." if view.approximate_exit
             else " The exit was priced from recorded trades.")
    return text


# ── the setup ─────────────────────────────────────────────────────────────

_TIMEFRAMES = {60: "one-minute", 180: "three-minute", 300: "five-minute", 600: "ten-minute",
               900: "fifteen-minute", 1800: "thirty-minute", 3600: "one-hour",
               7200: "two-hour", 14400: "four-hour"}
_NUMERIC = {60: "1-minute", 180: "3-minute", 300: "5-minute", 600: "10-minute",
            900: "15-minute", 1800: "30-minute", 3600: "1-hour", 14400: "4-hour"}


def timeframe_words(seconds: int | None, *, numeric: bool = False) -> str:
    if not seconds:
        return "gap-chart"
    table = _NUMERIC if numeric else _TIMEFRAMES
    return table.get(int(seconds), f"{int(seconds) // 60}-minute")


def _range(low: int, high: int) -> str:
    return f"{_price(low)}–{_price(high)}"


def gap_legend(record: SetupRecord | None) -> str | None:
    """The whole-trade legend's zone name; a related record's zone is labelled context.

    ``None`` when no gap zone is saved (no record, or a record without its
    higher-timeframe gap): the legend then shows no gap swatch.
    """

    if record is None or record.htf is None:
        return None
    name = "four-hour"
    if record.htf.timeframe_seconds:
        name = timeframe_words(record.htf.timeframe_seconds)
    if not _own(record):
        return f"Related {name} gap (context)"
    return f"{name.capitalize()} gap"


def setup_card_title(record: SetupRecord | None) -> str:
    """The setup chart's title: a related record is context, not this setup (A6)."""

    if record is not None and not _own(record):
        return "Related setup context · 1-minute candles"
    return "The setup · 1-minute candles"


def formed_card_title(record: SetupRecord | None) -> str:
    if record is not None and not _own(record):
        return "Related setup context"
    return "How the setup formed"


def related_source_line(record: SetupRecord | None) -> str | None:
    """Under a related record's steps: whose history they are (``None`` for the own record)."""

    if record is None or _own(record):
        return None
    return (f"From configuration {record.configuration}'s record; not this configuration's own "
            "formation history.")


@dataclass(frozen=True)
class SetupEvent:
    """One saved formation event: when it became known, and the candle that shows it.

    ``known_at`` is the event's availability instant, the one instant the steps
    text, the setup chart's key and markers and the moment picker all use. A gap
    is known at its confirmation (its third candle's close). A tap (a candle's
    high/low reaching the gap) and a close through the opposing gap (a candle's
    body close) are known only when that candle closes: the strategy records them
    at the close and saves no earlier observation. ``candle_open`` only names the
    candle ("the 7:05 PM candle"); it never decides what is known. ``known_at`` is
    ``None`` when the completion was not recorded: the event is then never known at
    a point in time and offers no moment.
    """

    role: str  # "htf", "tap", "parent", "opposing" or "inversion", in formation order
    known_at: pd.Timestamp | None
    candle_open: pd.Timestamp | None = None


def setup_events(record: SetupRecord | None) -> list[SetupEvent]:
    """The record's saved formation events in formation order (see :class:`SetupEvent`)."""

    if record is None:
        return []
    events: list[SetupEvent] = []
    if record.htf is not None:
        events.append(SetupEvent("htf", record.htf.confirmed_utc))
    for role, bar, gap in (("tap", "tap_bar", None), ("parent", None, record.parent),
                           ("opposing", None, record.opposing),
                           ("inversion", "inversion_bar", None)):
        if bar is not None:
            opened, closed = record.bar_open(bar), record.bar_close(bar)
            # a tap belongs to the higher-timeframe gap: listed with it even when its candle's
            # times were not recorded (never known at a moment then); a close-through only
            # when its candle was recorded
            if opened is not None or closed is not None or (role == "tap"
                                                              and record.htf is not None):
                events.append(SetupEvent(role, closed, opened))
        elif gap is not None:
            events.append(SetupEvent(role, gap.confirmed_utc))
    return events


def _candle(event: SetupEvent, reference: pd.Timestamp | None, *, first: bool = False) -> str:
    """``the 7:05 PM candle`` (named by its opening minute), ``a candle`` without one."""

    words = ("a candle" if event.candle_open is None
             else f"the {when(event.candle_open, reference)} candle")
    return words[:1].upper() + words[1:] if first else words


#: under "How the setup formed": what the step times mean
STEP_TIMES_NOTE = ("Each time is when that step became known: a gap when its third candle "
                   "closed, a candle's touch or close-through when that candle closed.")
_NOT_RECORDED = " (its close was not recorded)"


@dataclass(frozen=True)
class SetupStep:
    number: int
    text: str
    at: pd.Timestamp | None
    time_text: str
    bold: bool = False


def setup_steps(record: SetupRecord | None, view: TradeView, *, distance_cap: int | None = None,
                moment: pd.Timestamp | None = None) -> list[SetupStep]:
    """"How the setup formed": the saved chain, oldest first, ending at the entry.

    Each step is dated, and shown at a point in time, by the instant it became
    known (:class:`SetupEvent`): a candle's tap or close-through at that candle's
    close, never its opening minute, which only names the candle. A step whose
    completion was not recorded is never shown at a point in time; in full
    history it is listed without a time.

    With a related record (identity not established, correction A6) the chain is
    that record's, shown under "Related setup context"; this configuration's
    distance limit is then not claimed for it.
    """

    ref = view.entry_utc
    raw: list[tuple[str, pd.Timestamp | None, bool, bool]] = []  # text, at, dated, bold
    for event in setup_events(record):
        missing = _NOT_RECORDED if event.known_at is None else ""
        if event.role == "htf":
            gap = record.htf
            raw.append((f"{timeframe_words(gap.timeframe_seconds).capitalize()} gap, "
                        f"{_range(gap.low_ticks, gap.high_ticks)}", event.known_at, True, False))
        elif event.role == "tap":
            target = ("it" if record.htf is not None
                      else "the higher-timeframe gap")
            raw.append((f"{_candle(event, ref, first=True)} taps into {target}{missing}",
                        event.known_at, True, False))
        elif event.role == "parent":
            raw.append((f"{timeframe_words(record.parent.timeframe_seconds).capitalize()} parent "
                        "gap forms", event.known_at, False, False))
        elif event.role == "opposing":
            gap = record.opposing
            size = gap.size_ticks if gap.size_ticks is not None else gap.high_ticks - gap.low_ticks
            cap = (f", inside the {distance_cap}-tick limit"
                   if distance_cap is not None and _own(record) else "")
            raw.append((f"Opposing {timeframe_words(gap.timeframe_seconds, numeric=True)} gap, "
                        f"{fmt.count(size, 'tick')}{cap}", event.known_at, False, False))
        elif event.role == "inversion":
            raw.append((f"{_candle(event, ref, first=True)} closes through it{missing}",
                        event.known_at, False, False))
    raw.append(("Entry", view.entry_utc, False, True))
    steps = []
    for number, (text, at, dated, bold) in enumerate(raw, start=1):
        if moment is not None and (at is None or at > moment):
            continue
        time_text = (short(at) if dated or at is None
                     else when(at, view.entry_utc))
        steps.append(SetupStep(number, text, at, time_text, bold))
    return steps


def setup_key(record: SetupRecord | None, view: TradeView, *,
              moment: pd.Timestamp | None = None) -> list[tuple[str, str, pd.Timestamp | None]]:
    """The setup chart's numbered key: (role, text, instant) in marker order."""

    items: list[tuple[str, str, pd.Timestamp | None]] = []
    ref = view.entry_utc
    if record is not None:
        if record.parent is not None:
            gap = record.parent
            items.append(("parent", f"{timeframe_words(gap.timeframe_seconds).capitalize()} "
                                    f"parent gap, {_range(gap.low_ticks, gap.high_ticks)}, "
                                    f"confirmed {when(gap.confirmed_utc, ref)}",
                          gap.confirmed_utc))
        if record.opposing is not None:
            gap = record.opposing
            items.append(("opposing", f"Opposing {timeframe_words(gap.timeframe_seconds)} gap, "
                                      f"{_range(gap.low_ticks, gap.high_ticks)}, confirmed "
                                      f"{when(gap.confirmed_utc, ref)}", gap.confirmed_utc))
        for event in setup_events(record):  # known at its candle's close (SetupEvent)
            if event.role == "inversion":
                missing = _NOT_RECORDED if event.known_at is None else ""
                items.append(("inversion", f"{_candle(event, ref, first=True)} closes "
                                           f"through it{missing}", event.known_at))
    items.append(("entry", f"Entry {_price(view.entry_ticks)} at {when(view.entry_utc, ref)} · "
                           f"initial stop {_price(view.stop_ticks)}", view.entry_utc))
    if view.half_utc is not None:
        r = _r_text(view.r_multiple(view.half_ticks))
        items.append(("half", f"Half out at {_price(view.half_ticks)}"
                              + (f" ({r})" if r else "")
                              + f", {when(view.half_utc, ref)} · "
                              + _moved_stop_words(view).replace("moved", "moves"),
                      view.half_utc))
    else:
        items.append(("exit", f"{exit_words(view).capitalize()}: out at "
                              f"{_price(view.exit_ticks)}, {when(view.exit_utc, ref)}",
                      view.exit_utc))
    if moment is not None:
        items = [item for item in items if item[2] is not None and item[2] <= moment]
    return items


# ── point in time ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Moment:
    at: pd.Timestamp
    label: str


#: the point-in-time clock: every 5 minutes for the first hour after the entry, then hourly
_FINE_STEP = timedelta(minutes=5)
_FINE_SPAN = timedelta(hours=1)
_COARSE_STEP = timedelta(hours=1)


def day_close(view: TradeView) -> pd.Timestamp:
    """The 4:00 PM close of the trade's trading day (named by its closing date), in UTC.

    Without a saved trading day, the day is found from the entry: an entry at or
    after the 5:00 PM open belongs to the next day's session.
    """

    text = str(view.trading_day or "")[:10]
    stamp = pd.Timestamp(text) if text else pd.NaT
    if pd.isna(stamp):
        local = view.entry_utc.tz_convert(CHICAGO)
        closing = (local + timedelta(days=1)).date() if local.hour >= 17 else local.date()
    else:
        closing = stamp.date()
    return pd.Timestamp.combine(closing, time(16, 0)).tz_localize(CHICAGO).tz_convert("UTC")


def _clock_grid(view: TradeView) -> list[pd.Timestamp]:
    """Fixed clock marks after the entry up to the day's close (never the fills or exit).

    Five-minute marks (on the clock: 7:10, 7:15 …) for the first hour after the
    entry, then whole hours to the 4:00 PM close. Flooring the UTC instant equals
    flooring the Chicago clock because Chicago's offsets are whole hours.
    """

    entry, close = view.entry_utc, day_close(view)
    marks: list[pd.Timestamp] = []
    at = entry.floor("5min") + _FINE_STEP
    while at <= entry + _FINE_SPAN and at <= close:
        marks.append(at)
        at += _FINE_STEP
    at = (marks[-1] if marks else entry).floor("h") + _COARSE_STEP
    while at <= close:
        marks.append(at)
        at += _COARSE_STEP
    return marks


def moments(view: TradeView, record: SetupRecord | None) -> list[Moment]:
    """The moments a reviewer can stop the clock at, in time order.

    Built only from what was known by the entry, so the list itself never hints
    at the result: this configuration's own saved setup steps before the entry
    (tap, parent gap, opposing gap, the candle that closed through it), the
    entry, then fixed clock marks to the trading day's 4:00 PM close
    (:func:`_clock_grid`), named by their time alone. The half exit, the exit and
    the account's fate are never used — a trade stopped out in minutes offers the
    same clock as one held to the close. A related record (another
    configuration's, identity not established) adds no step (correction A6).
    """

    ref = view.entry_utc
    found: list[Moment] = []
    if _own(record):
        gap = (timeframe_words(record.htf.timeframe_seconds) if record.htf is not None
               else "higher-timeframe")
        what = {"tap": f"has tapped the {gap} gap", "parent": "the parent gap has formed",
                "opposing": "the opposing gap has formed",
                "inversion": "has closed through the opposing gap"}
        for event in setup_events(record):  # each at the instant it became known
            if event.known_at is None or event.role not in what:
                continue  # no recorded completion: never a moment
            words = (f"{_candle(event, ref)} {what[event.role]}" if event.role in
                     ("tap", "inversion") else what[event.role])
            found.append(Moment(event.known_at, f"{when(event.known_at, ref)} · {words}"))
        found = [m for m in found if m.at <= view.entry_utc]  # the setup, as known by entry
    found.append(Moment(view.entry_utc, f"{when(view.entry_utc, ref)} · entry"))
    found += [Moment(at, when(at, ref)) for at in _clock_grid(view)]
    unique: dict[pd.Timestamp, Moment] = {}
    for moment in sorted(found, key=lambda m: m.at):  # stable: a named step wins a tie
        unique.setdefault(moment.at, moment)
    return list(unique.values())


def default_moment(view: TradeView) -> pd.Timestamp:
    """The first five-minute clock mark after the entry (7:10 PM for a 7:07 PM entry).

    The entry itself when the entry is too close to the day's close for a mark.
    """

    marks = _clock_grid(view)
    return marks[0] if marks else view.entry_utc


def known_accounts(accounts: list[int], opened: dict[int, pd.Timestamp], moment: pd.Timestamp,
                   current: int | None, *, trades: Sequence[TradeView] | None = None
                   ) -> list[int]:
    """Point in time: only the accounts opened at or before ``moment``.

    A later account's number would show that an earlier one was lost. The
    reviewed trade's own account is always kept: the trade happened in it.
    An account whose opening time isn't saved is left out. With ``trades``
    (what the Account picker uses), an account is listed only when one of its
    trades had been entered by the moment, so every listed account has a trade
    to open (an account opened minutes before the moment with no trade yet is
    left out; its opening is still shown with the loss that caused it).
    """

    traded = None if trades is None else {
        v.account for v in trades if v.account is not None and v.entry_utc <= moment}
    return [a for a in accounts
            if a == current or (a in opened and opened[a] <= moment
                                and (traded is None or a in traded))]


def known_trades(views: list[TradeView], moment: pd.Timestamp, current: int | None
                 ) -> list[TradeView]:
    """Point in time: only the trades entered at or before ``moment`` (and the reviewed one)."""

    return [v for v in views if v.seq == current or v.entry_utc <= moment]


def context_line(parts: list[str], account: int | None, number: int, total: int, *,
                 point_in_time: bool) -> str:
    """``… · Account 6 · trade 77 of 114 at this firm``; point in time leaves out the total."""

    where = f"trade {number} at this firm" if point_in_time else (
        f"trade {number} of {total} at this firm")
    return " · ".join([*parts, f"Account {account}", where])


def trade_label(view: TradeView, *, hide_result: bool = False, account: bool = False) -> str:
    """``Apr 12, 7:07 PM · Account 6 · +$6,254.72``.

    Point in time (``hide_result``) leaves out the result AND the account: a
    later trade's new account number would show that this trade's account was
    lost.
    """

    parts = [short(view.entry_utc)]
    if account and not hide_result and view.account is not None:
        parts.append(f"Account {view.account}")
    parts.append("result hidden" if hide_result else fmt.money(view.net, signed=True))
    return " · ".join(parts)


def hidden_sentence(moment: pd.Timestamp, reference: pd.Timestamp | None = None) -> str:
    return f"Hidden: everything after {when(moment, reference)}"


def points_between(low: int, high: int) -> Decimal:
    return Decimal(high - low) * fmt.TICK_POINTS


def no_record_sentence() -> str:
    return NO_RECORD
