"""Conserving decomposition of saved fills; no fees are reposted."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from alpha_lab.propsim.funded.position_walk import fill_cost_cents

VERSION = "posted_fill_proportional_entry_allocation_v1"


def money_cents(row: Mapping[str, Any], stem: str) -> int:
    if row.get(f"{stem}_cents") is not None:
        value = Decimal(str(row[f"{stem}_cents"]))
    else:
        value = Decimal(str(row[f"{stem}_usd"])) * 100
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(f"{stem} must be posted whole cents")
    return int(value)


def allocate_entry_fee(posted_cents: int, quantity: int, first_quantity: int) -> tuple[int, int]:
    """Floor proportional first-leg share; final leg receives the residual."""
    if quantity <= 0 or not 0 <= first_quantity < quantity or posted_cents < 0:
        raise ValueError("invalid entry fee allocation quantities")
    first = posted_cents * first_quantity // quantity
    return first, posted_cents - first


@dataclass(frozen=True)
class TradeLegs:
    had_partial: bool
    first_cents: int
    remaining_cents: int
    net_cents: int
    entry_fee_cents: int
    first_exit_fee_cents: int
    final_exit_fee_cents: int
    first_entry_fee_cents: int
    remaining_entry_fee_cents: int
    outcome: str


def trade_legs(row: Mapping[str, Any], *, tick_value_cents: int | None = None,
               mills: int | None = None) -> TradeLegs:
    """Use saved individual fees when present, else reconcile frozen fill policy."""
    tick_value = int(row.get("tick_value_cents") or tick_value_cents or 0)
    rate = row.get("cost_per_contract_mills", mills)
    if tick_value <= 0 or rate is None:
        raise ValueError("trade leg sizing or frozen fee policy unavailable")
    quantity = int(row["quantity"])
    first_quantity = int(row.get("scale_out_quantity") or 0)
    partial = first_quantity > 0
    if partial and row.get("scale_out_ticks") is None:
        raise ValueError("partial quantity lacks its saved fill price")
    final_quantity = int(row.get("final_exit_quantity") or quantity - first_quantity)
    if quantity <= 0 or final_quantity <= 0 or first_quantity + final_quantity != quantity:
        raise ValueError("saved trade fill quantities do not conserve position")
    fees = []
    for stem, qty in (("entry_fee", quantity), ("scale_out_fee", first_quantity),
                      ("final_exit_fee", final_quantity)):
        expected = fill_cost_cents(qty, int(rate))
        posted = (money_cents(row, stem) if row.get(f"{stem}_cents") is not None
                  or row.get(f"{stem}_usd") is not None else expected)
        if posted != expected:
            raise ValueError(f"posted {stem} disagrees with frozen per-fill fee policy")
        fees.append(posted)
    if sum(fees) != money_cents(row, "costs"):
        raise ValueError("saved total costs do not reconcile to actual execution fills")
    first_entry_fee, remaining_entry_fee = allocate_entry_fee(fees[0], quantity, first_quantity)
    direction = str(row["direction"]).lower().split(".")[-1]
    if direction not in {"long", "short"}:
        raise ValueError("saved trade direction unavailable")
    sign = 1 if direction == "long" else -1
    first = ((int(row["scale_out_ticks"]) - int(row["entry_ticks"])) * sign
             * tick_value * first_quantity - first_entry_fee - fees[1]) if partial else 0
    remaining = ((int(row["exit_ticks"]) - int(row["entry_ticks"])) * sign * tick_value
                 * final_quantity - remaining_entry_fee - fees[2])
    net = money_cents(row, "net_pnl")
    if first + remaining != net:
        raise ValueError("saved leg P&L does not reconcile to whole-trade net")
    outcome = ("no_partial" if not partial else
               "partial_to_scheduled_close" if row.get("exit_kind") == "scheduled_close" else
               "partial_to_breakeven_stop" if row.get("exit_kind") == "breakeven_stop" else
               "partial_other")
    return TradeLegs(partial, first, remaining, net, *fees, first_entry_fee,
                     remaining_entry_fee, outcome)
