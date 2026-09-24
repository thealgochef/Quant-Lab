"""Funded comparison trades in the existing Trade review (repair R3; no screen code).

A funded trade is identified exactly: saved result, configuration, firm,
account and the strategy trade id (with the ledger sequence), never by its
timestamp. This module lists a pair's recorded trades in time order, turns one
trade row into the plain facts of its recorded path (entry, initial quantity
and stop, half fill, remaining quantity, moved stop, final exit, account
liquidation), gives the chart markers at their exact recorded instants, and
builds the review-ledger keys that keep funded judgments apart from every other
account, firm, result and the reference strategy review.

The strategy evidence for bars and gap geometry is the plan's own bound study
package (run id and manifest hash recorded in the plan), read only and verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import (
    chicago_label,
    utc_instant,
)

__all__ = [
    "FundedTradeFacts",
    "Marker",
    "StopSegment",
    "funded_review_sources",
    "funded_trade_facts",
    "pair_trades",
    "plan_strategy_package",
    "review_keys",
    "strategy_trade_links",
    "trade_option_label",
]

_TICK = Decimal("0.25")


def _usd(value: Any, *, signed: bool = False) -> str:
    amount = float(value or 0)
    sign = "-" if amount < 0 else ("+" if signed and amount > 0 else "")
    return f"{sign}${abs(amount):,.2f}"


def _points(ticks: Any) -> str:
    return "Not recorded" if ticks is None else f"{Decimal(int(ticks)) * _TICK:,.2f}"


#: statuses whose saved result the results page shows (and so may send to review)
REVIEWABLE_STATUSES = {
    "Completed": "funded accounts",
    "Incomplete": "funded accounts; some configurations did not complete",
    "Failed": "funded accounts; failed its money checks, review only",
}


def funded_review_sources(studies) -> list[Any]:
    """Unarchived funded comparisons with a saved result the results page shows.

    Completed, Incomplete and Failed-with-a-result comparisons all have a
    verified saved result on the results page, so each can open its trades.
    """

    return [s for s in studies if getattr(s, "kind", None) == "funded_comparison"
            and not getattr(s, "archived", False)
            and getattr(s, "status", None) in REVIEWABLE_STATUSES
            and (getattr(s, "state", None) or {}).get("result_id")]


def funded_source_label(study) -> str:
    return f"{study.name} ({REVIEWABLE_STATUSES[study.status]})"


def pair_trades(result: dict[str, Any], configuration: str, firm_key: str) -> list[dict]:
    """The pair's recorded trades ordered by the entry instant, then ledger sequence."""

    pair = f"{configuration}|{firm_key}"
    rows = [t for t in (result.get("tables") or {}).get("trades") or []
            if t.get("pair_id") == pair]
    return sorted(rows, key=lambda t: (utc_instant(t.get("entry_utc")), int(t.get("seq") or 0)))


def trade_option_label(index: int, trade: dict[str, Any]) -> str:
    direction = "Long" if trade.get("direction") == "long" else "Short"
    return (f"{index}. {chicago_label(trade.get('entry_utc'), short=True)} · Account "
            f"{trade.get('account_number')} · {direction} · {_exit_words(trade)} · "
            f"entry {_points(trade.get('entry_ticks'))}")


def _exit_words(trade: dict[str, Any]) -> str:
    kind = trade.get("exit_kind")
    rest = {"target": "target", "stop": "stop", "breakeven_stop": "break-even stop",
            "scheduled_close": "daily close", "account_failure": "account loss limit"
            }.get(kind, str(kind or "exit").replace("_", " "))
    if trade.get("scale_out_ticks") is not None:
        return f"half at target, rest at {rest}"
    return rest


@dataclass(frozen=True)
class Marker:
    label: str
    ts_utc: Any  # an instant (ISO text or integer nanoseconds), exactly as recorded
    price_ticks: int | None


@dataclass(frozen=True)
class StopSegment:
    label: str
    start: Any
    end: Any
    price_ticks: int


@dataclass(frozen=True)
class FundedTradeFacts:
    rows: tuple[tuple[str, str], ...]
    markers: tuple[Marker, ...]
    stops: tuple[StopSegment, ...]
    target_ticks: int | None
    entry_ticks: int | None
    notices: tuple[str, ...]


def funded_trade_facts(result: dict[str, Any], trade: dict[str, Any], *,
                       instrument: str | None = None) -> FundedTradeFacts:
    """The recorded funded path of one trade, in plain words, with exact instants."""

    tables = result.get("tables") or {}
    pair, account = trade.get("pair_id"), trade.get("account_number")
    quantity = int(trade.get("quantity") or 0)
    unit = {"micro": "Micro E-mini Nasdaq-100 (MNQ)", "mini": "E-mini Nasdaq-100 (NQ)"}.get(
        instrument or "", "contracts")
    half = trade.get("scale_out_ticks") is not None
    rows: list[tuple[str, str]] = [
        ("Firm and account", f"{trade.get('firm')} — account {account}"),
        ("Saved trading day", str(trade.get("trading_day"))),
        ("Direction", "Long" if trade.get("direction") == "long" else "Short"),
        ("Entry", f"{chicago_label(trade.get('entry_utc'), seconds=True)} at "
                  f"{_points(trade.get('entry_ticks'))}"),
        ("Initial quantity", f"{quantity:,} x {unit}"),
        ("Initial stop", _points(trade.get("stop_ticks"))),
        ("Target", _points(trade.get("target_ticks"))),
    ]
    markers = [Marker("Entry", trade.get("entry_utc"), trade.get("entry_ticks"))]
    stops: list[StopSegment] = []
    stop_end = trade.get("exit_utc")
    if half:
        half_at = trade.get("scale_out_ns")
        rows += [
            ("Half exit (partial fill)",
             f"{int(trade.get('scale_out_quantity') or 0):,} at "
             f"{_points(trade.get('scale_out_ticks'))}, "
             f"{chicago_label(half_at, seconds=True)}"),
            ("Remaining quantity", f"{int(trade.get('final_exit_quantity') or 0):,}"),
            ("Stop after the half exit",
             f"{_points(trade.get('final_stop_ticks'))} (moved at the half exit)"),
        ]
        markers.append(Marker("Half exit", half_at, trade.get("scale_out_ticks")))
        stops.append(StopSegment("Initial stop", trade.get("entry_utc"), half_at,
                                 int(trade["stop_ticks"])))
        if trade.get("final_stop_ticks") is not None:
            stops.append(StopSegment("Moved stop", half_at, stop_end,
                                     int(trade["final_stop_ticks"])))
    elif trade.get("stop_ticks") is not None:
        stops.append(StopSegment("Stop", trade.get("entry_utc"), stop_end,
                                 int(trade["stop_ticks"])))
    exit_label = "Account liquidation" if trade.get("account_failed") else "Final exit"
    rows += [
        (exit_label, f"{chicago_label(trade.get('exit_utc'), seconds=True)} at "
                     f"{_points(trade.get('exit_ticks'))}"
                     + (f" ({int(trade.get('final_exit_quantity') or 0):,} contracts)"
                        if half else "")),
        ("Exit reason", _exit_words(trade).capitalize()
         + (f" — {trade['exit_basis']}" if trade.get("exit_basis") else "")),
        ("Result after costs", f"{_usd(trade.get('net_pnl_usd'), signed=True)} "
                               f"(costs {_usd(trade.get('costs_usd'))})"),
        ("Prices from", "One-minute approximation for some minutes"
         if int(trade.get("minutes_approximated") or 0) or trade.get("approximate_exit")
         else "Recorded exchange trades"),
    ]
    markers.append(Marker(exit_label, trade.get("exit_utc"), trade.get("exit_ticks")))
    notices: list[str] = []
    if trade.get("account_failed"):
        failure = next((r for r in tables.get("rule_boundary_evidence") or []
                        if r.get("pair_id") == pair and r.get("account_number") == account
                        and r.get("check") == "account_failure"
                        and r.get("trade_ref") in (None, trade.get("trade_ref"))), None)
        if failure:
            rows.append(("Loss limit check", _failure_words(failure)))
        replacement = next((e for e in tables.get("account_events") or []
                            if e.get("pair_id") == pair and e.get("event") == "created"
                            and e.get("replaces") in (account, f"{pair}#{account}")), None)
        if replacement:
            rows.append(("Replacement", f"Account {replacement.get('account_number')} started "
                                        f"{chicago_label(replacement.get('ts_utc'))}"))
    if instrument == "micro":
        notices.append("Micro positions are priced on the E-mini Nasdaq-100 (NQ) recorded "
                       "trades; the micro contract's own trades were not used.")
    return FundedTradeFacts(rows=tuple(rows), markers=tuple(markers), stops=tuple(stops),
                            target_ticks=trade.get("target_ticks"),
                            entry_ticks=trade.get("entry_ticks"), notices=tuple(notices))


def _failure_words(row: dict[str, Any]) -> str:
    parts = [chicago_label(row.get("ts_utc"), seconds=True)]
    if row.get("price_ticks") is not None:
        parts.append(f"price {_points(row['price_ticks'])}")
    if row.get("equity_usd") is not None and row.get("floor_usd") is not None:
        comparator = {"at_or_below": "at or below", "below": "below"}.get(
            row.get("comparator"), str(row.get("comparator") or "against"))
        parts.append(f"equity {_usd(row['equity_usd'])} {comparator} the loss limit "
                     f"{_usd(row['floor_usd'])}")
    if row.get("detail"):
        parts.append(str(row["detail"]))
    return "; ".join(parts)


def review_keys(result_id: str, plan_id: str, trade: dict[str, Any]
                ) -> tuple[str, str, dict[str, str]]:
    """(chart key, case key, pair reference) for the shared review ledger.

    The case key names the result, the configuration-and-firm pair, the account
    and the strategy trade, so a judgment can never land on another firm's,
    another account's, another result's or the reference strategy's review.
    """

    pair = str(trade.get("pair_id"))
    account = str(trade.get("account_number"))
    strategy_trade = str(trade.get("strategy_trade_id") or trade.get("trade_ref"))
    chart_key = f"funded_comparison_result:{result_id}"
    case_key = f"funded:{result_id}:{pair}#{account}:{strategy_trade}"
    pair_ref = {
        "funded_comparison_result_id": result_id,
        "funded_comparison_plan_id": plan_id,
        "pair_id": pair,
        "configuration": str(trade.get("configuration")),
        "firm_key": str(trade.get("firm_key")),
        "account_number": account,
        "funded_trade_seq": str(trade.get("seq")),
        "strategy_trade_id": strategy_trade,
    }
    return chart_key, case_key, pair_ref


def plan_strategy_package(plan: Any, archive_root: Path) -> Path | None:
    """The plan's bound strategy package, only if its run id and manifest hash match."""

    from alpha_lab.propsim.funded.comparison_source import open_comparison_source

    for candidate in sorted(Path(archive_root).glob(f"*/{plan.source.package_root_name}")):
        try:
            source = open_comparison_source(candidate)
        except Exception:
            continue
        if (source.package.run_id == plan.source.package_run_id
                and source.package.manifest_sha256 == plan.source.package_manifest_sha256):
            return candidate
    return None


def strategy_trade_links(plan: Any, source: Any, configuration: str
                         ) -> tuple[str | None, str | None]:
    """(core replay for exact geometry, core replay for the day's bars) of one configuration.

    Only a configuration that is a member of the verified study has saved
    strategy records, so only it can show gap geometry. Every configuration's
    bars come from a verified member whose charts include its own.
    """

    members = source.by_name
    variants = {v.name: v for v in getattr(plan, "variants", ()) or ()}
    if configuration in variants:
        variant = variants[configuration]
        geometry = (members[configuration].core_replay_id
                    if variant.in_verified_study and configuration in members else None)
        bars = members.get(variant.cache_configuration)
        return geometry, bars.core_replay_id if bars else None
    member = members.get(configuration)
    return (member.core_replay_id, member.core_replay_id) if member else (None, None)
