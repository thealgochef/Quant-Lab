"""v2 executed-trade → account-trade adapter + the order-sensitive stream hash.

Ticks map to instrument points via the cost policy's tick size. The gross
stream hash is a STABLE digest of the exact resolved trade SEQUENCE in
canonical entry-time order (ties by trade id): any addition, removal, value
change, or change of execution order (different entry times) produces a
different hash, while a pure row-shuffle of the same frame canonicalizes to
the same hash — the SEQUENCE is pinned, not the frame's storage order. Rows
hash as named column→value records over the REQUIRED column set (a frame
missing any hash column is refused, never silently under-pinned).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime

import pandas as pd

from alpha_lab.propsim.account import AccountTrade

__all__ = [
    "account_trades_from_executed_frame",
    "gross_trade_stream_hash",
    "group_account_trades_by_day",
]

_HASH_COLUMNS = (
    "trade_id",
    "trading_day",
    "entry_ts_utc",
    "resolution_ts_utc",
    "direction",
    "entry_ticks",
    "stop_ticks",
    "target_ticks",
    "risk_ticks",
    "realized_ticks",
    "mfe_ticks",
    "mae_ticks",
)


def account_trades_from_executed_frame(
    trades: pd.DataFrame,
    *,
    tick_size: float,
    cost_points_round_turn: float = 0.0,
) -> tuple[AccountTrade, ...]:
    """Ordered account trades (entry-time order, ties by trade id).

    ``points`` is the NET per-contract point result (realized − cost);
    excursions stay gross magnitudes in points. Only resolved executed trades
    may enter (the caller passes the validated EXECUTED_TRADE table).
    """

    if trades.empty:
        return ()
    work = trades.copy()
    work["_entry"] = pd.to_datetime(work["entry_ts_utc"], utc=True)
    work = work.sort_values(["_entry", "trade_id"], kind="mergesort")
    out: list[AccountTrade] = []
    for row in work.to_dict("records"):
        realized_points = float(row["realized_ticks"]) * tick_size
        risk_points = float(row["risk_ticks"]) * tick_size
        mfe = row.get("mfe_ticks")
        mae = row.get("mae_ticks")
        entry_ts = row["entry_ts_utc"]
        if isinstance(entry_ts, datetime | pd.Timestamp):
            entry_ts = pd.Timestamp(entry_ts).isoformat()
        resolution_ts = row["resolution_ts_utc"]
        if isinstance(resolution_ts, datetime | pd.Timestamp):
            resolution_ts = pd.Timestamp(resolution_ts).isoformat()
        out.append(
            AccountTrade(
                day=date.fromisoformat(str(row["trading_day"])),
                entry_ts_utc=str(entry_ts),
                resolution_ts_utc=str(resolution_ts),
                points=realized_points - cost_points_round_turn,
                risk_points=risk_points,
                mfe_pts=(
                    abs(float(mfe)) * tick_size
                    if mfe is not None and not pd.isna(mfe)
                    else None
                ),
                mae_pts=(
                    abs(float(mae)) * tick_size
                    if mae is not None and not pd.isna(mae)
                    else None
                ),
                trade_id=str(row["trade_id"]),
                decision_id=str(row.get("decision_id")) if row.get("decision_id") else None,
                candidate_id=(
                    str(row.get("candidate_id")) if row.get("candidate_id") else None
                ),
                setup_id=str(row.get("setup_id")) if row.get("setup_id") else None,
            )
        )
    return tuple(out)


def gross_trade_stream_hash(trades: pd.DataFrame) -> str:
    """Sequence-pinning SHA-256 over the exact resolved trade stream.

    Refuses a frame missing any required hash column (fail-closed: a
    silently shorter row would under-pin the stream), and hashes named
    column→value records so two frames with different schemas can never
    collide positionally.
    """

    if trades.empty:
        return hashlib.sha256(b"empty_gross_trade_stream_v1").hexdigest()
    missing = [column for column in _HASH_COLUMNS if column not in trades.columns]
    if missing:
        raise ValueError(
            f"gross_trade_stream_hash requires columns {missing} — refusing "
            "to hash an under-specified stream"
        )
    work = trades.copy()
    work["_entry"] = pd.to_datetime(work["entry_ts_utc"], utc=True)
    work = work.sort_values(["_entry", "trade_id"], kind="mergesort")
    rows = []
    for row in work.to_dict("records"):
        rows.append({column: str(row.get(column)) for column in _HASH_COLUMNS})
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def group_account_trades_by_day(
    trades: tuple[AccountTrade, ...],
) -> tuple[tuple[date, tuple[AccountTrade, ...]], ...]:
    """Chronological (day, trades-in-entry-order) blocks for walking/resampling."""

    by_day: dict[date, list[AccountTrade]] = {}
    for trade in trades:
        by_day.setdefault(trade.day, []).append(trade)
    return tuple(
        (day, tuple(sorted(block, key=lambda t: (t.entry_ts_utc, t.trade_id))))
        for day, block in sorted(by_day.items())
    )
