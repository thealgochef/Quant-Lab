"""Runtime MenthorQ review exports and reconciled descriptive reports (Task A1).

Nothing here extends an immutable record family or provides ML inputs. Entry
context is looked up anew at each record's entry availability; economics remain
the existing validated execution projection. Unknowns remain explicit buckets.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from strategy_core.strategies.ifvg_smc.menthorq_levels import (
    MenthorqDerivedValues,
    MenthorqLevelSnapshot,
    derive_menthorq_values,
    evaluate_menthorq_entry_gates,
    slot_chicago_for,
)

from .contracts import RecordTable
from .trade_stats import _validate_and_normalize_executed_trades

GROUP_COLUMNS = ("entry_session", "regime", "slot_chicago")
POOLING_POLICY = (
    "All post-warmup executions are pooled for totals; session, regime and "
    "Chicago slot are reported jointly. Unknown buckets are retained. "
    "Roll-flagged rows remain in grouped totals and have separate rows; "
    "comparisons exclude only known roll-flagged days."
)
NOT_PRODUCED = "not_produced_in_a1"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _dataclass_values(value: Any) -> dict:
    # MappingProxyType deliberately cannot be deep-copied by dataclasses.asdict.
    return {field.name: getattr(value, field.name) for field in fields(value)}


def record_context(records: pd.DataFrame, *, provider, section, tick_size: float) -> pd.DataFrame:
    """Curated entry-time values only; never copy labels or outcomes to the view."""
    rows = []
    if not math.isfinite(tick_size) or tick_size <= 0:
        raise ValueError("entry context requires a positive instrument tick size")
    for record in records.to_dict("records"):
        # Core candidates/decisions carry final-entry availability in their
        # envelope, not an entry_ts_utc field. The flattened union still has
        # that column as a structural null. Trade envelopes are emitted at
        # resolution, so trades must use their explicit immutable entry time.
        trade_id = record.get("trade_id")
        timestamp_column = (
            "entry_ts_utc" if trade_id is not None and not pd.isna(trade_id)
            else "envelope_ts_utc"
        )
        ts = pd.Timestamp(record.get(timestamp_column))
        if pd.isna(ts) or ts.tzinfo is None:
            raise ValueError("entry context requires an aware availability timestamp")
        ts = ts.tz_convert("UTC").to_pydatetime()
        price = float(record["entry_ticks"]) * tick_size
        snapshot = provider.snapshot(ts)
        bar_open = record.get("geometry_entry_bar_open_ticks")
        bar_open = None if bar_open is None or pd.isna(bar_open) else float(bar_open) * tick_size
        derived = derive_menthorq_values(
            snapshot, price, ts, bar_open_points=bar_open,
            prior_cash_close_points=provider.prior_cash_close_for(ts),
        )
        gates = evaluate_menthorq_entry_gates(
            snapshot, price, ts,
            regime_gate_policy=section.regime_gate_policy,
            regime_unknown_policy=section.regime_unknown_policy,
            nearest_support_gex1_block=section.nearest_support_gex1_block,
            enable_shorts=section.enable_shorts,
        )
        row = {
            "candidate_id": record["candidate_id"],
            "availability_ts_utc": ts.isoformat(),
            "entry_price_points": price, "tick_size": tick_size,
            **_dataclass_values(snapshot), **_dataclass_values(derived),
            "regime_gate_blocked": gates.regime_gate_blocked,
            "nearest_support_gate_blocked": gates.nearest_support_gate_blocked,
            "gate_status": gates.gate_status,
            "entry_session": record.get("entry_session"),
        }
        row["levels"] = dict(snapshot.levels)
        if row["entry_session"] is None or pd.isna(row["entry_session"]):
            session = record.get("envelope_entry_session")
            row["entry_session"] = "unknown" if session is None or pd.isna(session) else session
        rows.append(row)
    columns = ["candidate_id", "availability_ts_utc", "entry_price_points", "tick_size",
               *[field.name for field in fields(MenthorqLevelSnapshot)],
               *[field.name for field in fields(MenthorqDerivedValues)],
               "regime_gate_blocked", "nearest_support_gate_blocked", "gate_status",
               "entry_session"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def _links(frame: pd.DataFrame, key: str) -> dict:
    if frame.empty:
        return {}
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"review export has multiple {key} links per candidate")
    return dict(zip(frame["candidate_id"].astype(str), frame[key], strict=True))


def write_context_export(
    path: Path, candidates: pd.DataFrame, decisions: pd.DataFrame, trades: pd.DataFrame,
    *, provider, section, tick_size: float, run_identity: dict,
) -> Path | None:
    """One CSV row per run candidate, with a JSON comment carrying provenance."""
    if section.menthorq_context_version is None:
        return None
    context = record_context(candidates, provider=provider, section=section, tick_size=tick_size)
    decision_ids, trade_ids = _links(decisions, "decision_id"), _links(trades, "trade_id")
    context.insert(1, "decision_id", context["candidate_id"].astype(str).map(decision_ids))
    context.insert(2, "trade_id", context["candidate_id"].astype(str).map(trade_ids))
    if "levels" in context:
        context["levels"] = context["levels"].map(
            lambda value: json.dumps(value, ensure_ascii=False)
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_identity": run_identity, "formula_version": provider.formula_version,
        "schema_version": provider.schema_version,
        "source_file_sha256": dict(provider.source_file_sha256),
        "archival": False, "enable_shorts": section.enable_shorts,
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("# " + json.dumps(metadata, sort_keys=True, default=str) + "\n")
        context.to_csv(stream, index=False)
    return path


def _metrics(frame: pd.DataFrame) -> dict:
    count = len(frame)
    wins = int(frame["_label"].eq("win").sum()) if count else 0
    gross = float(frame["_realized_pts"].sum()) if count else 0.0
    net = float(frame["_net_points"].sum()) if count else 0.0
    return {
        "trades": count, "wins": wins, "win_rate": wins / count if count else None,
        "gross_points": gross, "net_points": net,
        "expectancy_points": net / count if count else None,
    }


def _grouped(frame: pd.DataFrame, columns: tuple[str, ...], *, metrics: bool) -> list[dict]:
    rows = []
    for keys, group in frame.groupby(list(columns), dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(columns, keys, strict=True))
        rows.append({**row, **(_metrics(group) if metrics else {"count": len(group)})})
    return rows


def build_menthorq_reports(
    tables: dict[RecordTable, pd.DataFrame], *, provider, section,
    tick_size: float, cost_points: float,
) -> dict:
    """Build direct report views; preserve all warmup rows in the separate export."""
    from .experiment import _post_warmup_report_views  # noqa: PLC0415

    views, scope = _post_warmup_report_views(tables)
    contexts = {
        kind: record_context(views[kind], provider=provider, section=section, tick_size=tick_size)
        for kind in (RecordTable.ENTRY_CANDIDATE, RecordTable.ELIGIBLE_DECISION,
                     RecordTable.EXECUTED_TRADE)
    }
    trades = views[RecordTable.EXECUTED_TRADE]
    resolved = trades.loc[trades["status"].eq("resolved")].copy() if len(trades) else trades.copy()
    if len(resolved):
        normalized = _validate_and_normalize_executed_trades(resolved, tick_size=tick_size)
        normalized["_net_points"] = normalized["_realized_pts"] - cost_points
        normalized = normalized.join(contexts[RecordTable.EXECUTED_TRADE].set_index("candidate_id")
                                     [list(GROUP_COLUMNS) + ["roll_flag"]],
                                     on="candidate_id", rsuffix="_menthorq")
        # Prefer the explicitly retained runtime session if the raw table has it.
        for column in GROUP_COLUMNS:
            derived_column = column + "_menthorq"
            if derived_column in normalized:
                normalized[column] = normalized.pop(derived_column)
    else:
        normalized = pd.DataFrame(columns=[*GROUP_COLUMNS, "roll_flag", "_label",
                                          "_realized_pts", "_net_points"])
    roll = normalized["roll_flag"].eq(True).fillna(False)
    comparison = normalized.loc[~roll]
    groups = _grouped(normalized, GROUP_COLUMNS, metrics=True)
    total = _metrics(normalized)
    if sum(row["trades"] for row in groups) != total["trades"] or any(
        not math.isclose(sum(row[column] for row in groups), total[column], abs_tol=1e-9)
        for column in ("gross_points", "net_points")
    ):
        raise ValueError("MenthorQ grouped execution totals do not reconcile")
    counts = {}
    for kind, context in contexts.items():
        rows = _grouped(context, GROUP_COLUMNS, metrics=False)
        if sum(row["count"] for row in rows) != len(views[kind]):
            raise ValueError("MenthorQ grouped record counts do not reconcile")
        counts[kind.value] = rows
    return {
        "pooling_policy": POOLING_POLICY, "evaluation_scope": scope,
        "record_counts": counts, "executed_trade_groups": groups, "total": total,
        "open_unresolved_trades": len(trades) - len(resolved),
        "roll_days": sorted({str(value) for value in normalized.loc[roll, "trading_day"]})
        if "trading_day" in normalized else [],
        "roll_trade_groups": _grouped(normalized.loc[roll], GROUP_COLUMNS, metrics=True),
        "comparison": _metrics(comparison),
        "comparison_by_regime": _grouped(comparison, ("regime",), metrics=True),
        "comparison_by_slot": _grouped(comparison, ("slot_chicago",), metrics=True),
        # The source-selected instrument is explicitly unreachable in A1's
        # existing seam, including when this particular cohort has no trades.
        "roll_flag_unavailable": True,
        "net_cash_cents": NOT_PRODUCED,
        "net_cash_note": "A new verified funded result requires the existing gated plan path.",
    }


def build_comparison_table(configurations: dict[str, dict]) -> pd.DataFrame:
    """One row per provided configuration; never invent an unexecuted arm."""
    rows = []
    for name, report in configurations.items():
        row = {"configuration": name, **report["comparison"], "net_cash_cents": NOT_PRODUCED,
               "roll_flag_unavailable": report["roll_flag_unavailable"]}
        for dimension in ("regime", "slot"):
            key = "regime" if dimension == "regime" else "slot_chicago"
            labels = ("positive", "negative", "unknown") if dimension == "regime" else (
                "s1_0830_1000", "s2_1000_1200", "s3_1200_1330", "s4_1330_1510", "outside_cash",
            )
            splits = {split[key]: split for split in report[f"comparison_by_{dimension}"]}
            for label in labels:
                split = splits.get(label, {key: label, **_metrics(pd.DataFrame())})
                label = split["regime" if dimension == "regime" else "slot_chicago"]
                for metric, value in split.items():
                    if metric not in ("regime", "slot_chicago"):
                        row[f"{dimension}.{label}.{metric}"] = value
                row[f"{dimension}.{label}.net_cash_cents"] = NOT_PRODUCED
        rows.append(row)
    return pd.DataFrame(rows)


def build_funded_cash_groups(result: dict, *, provider) -> dict:
    """Consume the one saved verified funded result without recomputing economics.

    Cash receipts and purchases have cash-event timestamps, not entry signals.
    Their session is therefore unknown; regimes/slots describe those timestamps.
    No cash is apportioned to trades. This follows the existing monthly ledger
    aggregation and explicitly distinguishes cash-event pooling from entry views.
    """
    if not result.get("validation", {}).get("passed"):
        raise ValueError("funded cash reporting requires a verified saved result")
    buckets: dict[tuple, dict] = {}
    totals: dict[str, int] = {firm: 0 for firm in result["summaries_cents"]}
    for event in result["tables"]["cash_ledger"]:
        if event["kind"] not in ("payout_received", "account_purchase"):
            continue
        ts = pd.Timestamp(event["ts_utc"]).to_pydatetime()
        snapshot = provider.snapshot(ts)
        key = (event["firm_key"], "unknown", snapshot.regime, slot_chicago_for(ts))
        decimal_cents = Decimal(str(event["amount_usd"])) * 100
        if decimal_cents != decimal_cents.to_integral_value():
            raise ValueError("saved cash amount is not exact cents")
        amount = int(decimal_cents) * (1 if event["kind"] == "payout_received" else -1)
        bucket = buckets.setdefault(key, {"events": 0, "net_cash_cents": 0})
        bucket["events"] += 1
        bucket["net_cash_cents"] += amount
        totals[event["firm_key"]] = totals.get(event["firm_key"], 0) + amount
    for firm, summary in result["summaries_cents"].items():
        if totals.get(firm, 0) != summary["net_cash_earned_cents"]:
            raise ValueError("grouped cash does not reconcile to the saved result")
    return {
        "pooling_policy": (
            "Cash events grouped by their own timestamp; entry session unknown; "
            "no trade allocation."
        ),
        "groups": [{**dict(zip(("firm_key", *GROUP_COLUMNS), key, strict=True)), **value}
                   for key, value in sorted(buckets.items())],
        "totals_cents": totals,
    }
