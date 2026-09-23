"""Evidence-only costed exports from verified, already executed trade tables.

This module reads no market data, runs no replay, and writes only the requested
report directory. Source tables/metrics remain authoritative and unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import (
    research_trades,
)
from alpha_lab.agents.data_infra.ifvg.trade_stats import (
    DOLLARS_PER_POINT,
    _validate_and_normalize_executed_trades,
)
from alpha_lab.agents.data_infra.ifvg.working_artifacts import external_working_output


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, value):
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _pair(frame, out_dir, stem):
    frame.to_parquet(out_dir / f"{stem}.parquet", index=False)
    _write_json(out_dir / f"{stem}.json", frame.to_dict(orient="records"))


def _sign(value):
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _summary(frame):
    net = frame["derived_net_r"].astype(float)
    gross = frame["derived_gross_r"].astype(float)
    positive = float(net[net > 0].sum())
    negative = float(-net[net < 0].sum())
    equity = net.cumsum()
    drawdown = equity.cummax().clip(lower=0) - equity
    days = frame["trading_day"].astype(str)
    underwater_days = set()
    longest = 0
    for day, depth in zip(days, drawdown, strict=True):
        if depth > 1e-12:
            underwater_days.add(day)
            longest = max(longest, len(underwater_days))
        else:
            underwater_days = set()
    return {
        "executed_trades": len(frame),
        "independent_days": int(days.nunique()),
        "target_resolutions": int(frame["resolution"].eq("target").sum()),
        "stop_resolutions": int(frame["resolution"].eq("stop").sum()),
        "scheduled_close_resolutions": int(frame["resolution"].eq("scheduled_close").sum()),
        "net_positive_trades": int((net > 0).sum()),
        "net_negative_trades": int((net < 0).sum()),
        "net_zero_trades": int((net == 0).sum()),
        "total_gross_r": float(gross.sum()),
        "total_cost_r": float(frame["derived_cost_r"].sum()),
        "total_net_r": float(net.sum()),
        "total_gross_usd": float(frame["derived_gross_usd"].sum()),
        "total_cost_usd": float(frame["derived_cost_usd"].sum()),
        "total_net_usd": float(frame["derived_net_usd"].sum()),
        "gross_expectancy_r": float(gross.mean()) if len(frame) else None,
        "net_expectancy_r": float(net.mean()) if len(frame) else None,
        "profit_factor": positive / negative if negative > 0 else None,
        "max_drawdown_r": float(drawdown.max()) if len(frame) else None,
        "time_under_water_days": longest if len(frame) else None,
    }


def generate_costed_exports(
    *,
    executed_trades: pd.DataFrame,
    stored_metrics,
    cost_policy,
    core_replay_id: str,
    costed_evaluation_id: str,
    evaluation_dates: Sequence[str],
    out_dir: Path,
    include_raw_evidence: bool = False,
    repo_root: Path | None = None,
) -> dict:
    """Create compact audit tables after reconciling every saved metric.

    Caller must obtain ``executed_trades`` and metrics through their verified
    readers. The default exports one CSV per execution/equity table and one JSON
    per summary. ``include_raw_evidence=True`` explicitly requests the historical
    full-column JSON/Parquet pairs and original saved trade-statistics payload.
    Raw exports must target an external working directory; ``repo_root`` defaults
    to this source checkout. Compact exports may target curated ``reports``.
    All original values are retained in that mode. Warmup exclusion and execution
    validation call the same functions used by saved strategy metrics. In
    particular, a post-warmup open/unresolved trade raises, never gains invented
    realized P&L or silently disappears from the saved metric cohort.
    """
    out_dir = Path(out_dir)
    if include_raw_evidence:
        out_dir = external_working_output(
            repo_root if repo_root is not None else Path(__file__).resolve().parents[6],
            out_dir,
        )
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("costed export directory must be empty; preserve prior exports separately")
    metrics = (
        stored_metrics.model_dump(mode="python")
        if hasattr(stored_metrics, "model_dump")
        else dict(stored_metrics)
    )
    policy = (
        cost_policy.model_dump(mode="python")
        if hasattr(cost_policy, "model_dump")
        else dict(cost_policy)
    )
    stats = dict(metrics["trade_stats"])
    dates = [str(day) for day in evaluation_dates]
    if len(dates) != len(set(dates)) or dates != sorted(dates):
        raise ValueError("evaluation_dates must be unique and chronological")
    if not executed_trades.index.is_unique:
        raise ValueError("raw execution row index must be unique")
    if any(str(column).startswith("derived_") for column in executed_trades):
        raise ValueError("source columns collide with reserved derived_ namespace")

    cohort = research_trades(executed_trades)
    normalized = _validate_and_normalize_executed_trades(
        cohort, tick_size=float(policy["tick_size"])
    ).sort_values(["resolution_ts_utc", "trade_id"], kind="mergesort")
    if not set(normalized["trading_day"].astype(str)).issubset(dates):
        raise ValueError("post-warmup trade lies outside the frozen evaluation dates")
    cost_points = float(policy["cost_points_round_turn"])
    dollars_per_point = float(policy["dollars_per_point"])
    if dollars_per_point != DOLLARS_PER_POINT:
        raise ValueError("frozen dollar conversion differs from saved trade_stats convention")
    if not all(math.isfinite(v) and v >= 0 for v in (cost_points, dollars_per_point)):
        raise ValueError("invalid cost policy")

    # Select from the raw frame, not the normalizer's altered direction values.
    costed = executed_trades.loc[normalized.index].copy().reset_index(drop=True)
    risk = normalized["_risk_points"].astype(float).reset_index(drop=True)
    realized = normalized["_realized_pts"].astype(float).reset_index(drop=True)
    costed["derived_core_replay_id"] = core_replay_id
    costed["derived_costed_evaluation_id"] = costed_evaluation_id
    costed["derived_closed_trade_ordinal"] = np.arange(1, len(costed) + 1)
    costed["derived_risk_points"] = risk
    costed["derived_risk_usd"] = risk * dollars_per_point
    costed["derived_gross_points"] = realized
    costed["derived_cost_points"] = cost_points
    costed["derived_net_points"] = realized - cost_points
    costed["derived_gross_r"] = realized / risk
    costed["derived_cost_r"] = cost_points / risk
    costed["derived_net_r"] = (realized - cost_points) / risk
    costed["derived_gross_usd"] = realized * dollars_per_point
    costed["derived_cost_usd"] = cost_points * dollars_per_point
    costed["derived_net_usd"] = (realized - cost_points) * dollars_per_point
    costed["derived_minutes_in_trade"] = (
        pd.to_datetime(costed["resolution_ts_utc"], utc=True)
        - pd.to_datetime(costed["entry_ts_utc"], utc=True)
    ).dt.total_seconds() / 60
    session_column = next((c for c in ("entry_session", "session_doc") if c in costed), None)
    if session_column is not None:
        costed["derived_session_group"] = costed[session_column].astype(str)
    overall = _summary(costed)

    # The exact gate uses thirds of traded days, floor(n/3), not thirds of 107 dates.
    traded_days = sorted(costed["trading_day"].astype(str).unique())
    thirds = max(1, len(traded_days) // 3)
    blocks = [traded_days[:thirds], traded_days[thirds : 2 * thirds], traded_days[2 * thirds :]]
    block_rows = []
    costed["derived_chronological_third"] = pd.Series(pd.NA, index=costed.index, dtype="Int64")
    for index, block in enumerate(blocks, 1):
        sub = costed.loc[costed["trading_day"].astype(str).isin(block)]
        summary = _summary(sub)
        block_rows.append(
            {
                "block": index,
                "trading_days": block,
                "first_traded_day": block[0] if block else None,
                "last_traded_day": block[-1] if block else None,
                "included_in_sign_consistency": bool(block),
                "net_sign": _sign(summary["total_net_r"]) if block else None,
                **summary,
            }
        )
        costed.loc[sub.index, "derived_chronological_third"] = index
    costed["derived_chronological_third"] = costed["derived_chronological_third"].astype("Int64")
    signs = [row["net_sign"] for row in block_rows if row["included_in_sign_consistency"]]
    consistency = max(signs.count(1), signs.count(-1)) / len(signs) if signs else None

    session_rows = []
    overall_sign = _sign(overall["total_net_r"])
    if session_column is not None:
        for label, sub in costed.groupby("derived_session_group", sort=True):
            summary = _summary(sub)
            sign = _sign(summary["total_net_r"])
            session_rows.append(
                {
                    "session_source_column": session_column,
                    "session_group": label,
                    "null_source_rows": int(sub[session_column].isna().sum()),
                    "net_sign": sign,
                    "agrees_with_overall_sign": sign == overall_sign if overall_sign else None,
                    **summary,
                }
            )
    stability = (
        sum(row["agrees_with_overall_sign"] for row in session_rows) / len(session_rows)
        if session_rows and overall_sign
        else None
    )

    equity = costed[
        [
            "trade_id",
            "setup_id",
            "candidate_id",
            "decision_id",
            "trading_day",
            "entry_ts_utc",
            "resolution_ts_utc",
            "derived_closed_trade_ordinal",
            "derived_net_r",
            "derived_net_usd",
        ]
    ].copy()
    underwater_days = set()
    running_underwater = []
    for unit in ("r", "usd"):
        curve = costed[f"derived_net_{unit}"].cumsum()
        equity[f"closed_equity_before_{unit}"] = curve.shift(1, fill_value=0)
        equity[f"closed_equity_after_{unit}"] = curve
        equity[f"zero_based_peak_{unit}"] = curve.cummax().clip(lower=0)
        equity[f"drawdown_{unit}"] = equity[f"zero_based_peak_{unit}"] - curve
    for day, depth in zip(costed["trading_day"].astype(str), equity["drawdown_r"], strict=True):
        if depth > 1e-12:
            underwater_days.add(day)
        else:
            underwater_days = set()
        running_underwater.append(len(underwater_days))
    equity["underwater_distinct_traded_days_current_episode"] = running_underwater

    daily_rows = []
    for day in dates:
        sub = costed.loc[costed["trading_day"].astype(str).eq(day)]
        daily_rows.append({"trading_day": day, **_summary(sub)})
    checks = []

    def check(name, actual, expected):
        if actual is None or expected is None:
            passed = actual is None and expected is None
        else:
            passed = math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-11)
        checks.append({"check": name, "exported": actual, "stored": expected, "passed": passed})
        if not passed:
            raise ValueError(f"costed evidence mismatch: {name}: {actual!r} != {expected!r}")

    for field in (
        "executed_trades",
        "independent_days",
        "gross_expectancy_r",
        "net_expectancy_r",
        "profit_factor",
        "max_drawdown_r",
        "time_under_water_days",
    ):
        check(field, overall[field], metrics[field])
    check("time_block_sign_consistency", consistency, metrics["time_block_sign_consistency"])
    check("session_stability_score", stability, metrics["session_stability_score"])
    check("trade_stats.n", len(costed), stats["n"])
    for unit in ("r", "usd"):
        if costed.empty:
            continue
        block = stats["equity"][unit]
        check(f"total_net_{unit}", overall[f"total_net_{unit}"], block["net"])
        expected_curve = block["equity"]
        curve = equity[f"closed_equity_after_{unit}"].tolist()
        if len(curve) != len(expected_curve) or not np.allclose(
            curve, expected_curve, rtol=1e-11, atol=1e-11
        ):
            raise ValueError(f"costed evidence mismatch: full {unit} equity vector")
        actual_ts = pd.to_datetime(equity["resolution_ts_utc"], utc=True)
        expected_ts = pd.to_datetime(block["timestamps"], utc=True)
        if not np.array_equal(actual_ts.to_numpy(), expected_ts.to_numpy()):
            raise ValueError("costed evidence mismatch: equity resolution timestamps")
        checks.append(
            {"check": f"full_trade_linked_equity_{unit}", "rows": len(curve), "passed": True}
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if include_raw_evidence:
        _pair(costed, out_dir, "costed_trades_post_warmup")
        _pair(equity, out_dir, "closed_trade_equity")
        _pair(pd.DataFrame(session_rows), out_dir, "session_breakdown")
        _pair(pd.DataFrame(block_rows), out_dir, "chronological_thirds_breakdown")
        _pair(pd.DataFrame(daily_rows), out_dir, "evaluation_day_breakdown")
        _write_json(out_dir / "original_saved_trade_stats.json", stats)
    else:
        audit_columns = {
            "trade_id",
            "setup_id",
            "candidate_id",
            "decision_id",
            "trading_day",
            "direction",
            "entry_ts_utc",
            "resolution_ts_utc",
            "entry_ticks",
            "stop_ticks",
            "target_ticks",
            "exit_ticks",
            "resolution",
            "mfe_ticks",
            "mae_ticks",
            "scheduled_exit_deadline_ts_utc",
            "scheduled_exit_schedule_id",
        }
        selected = [
            column for column in costed if column in audit_columns or column.startswith("derived_")
        ]
        costed[selected].to_csv(out_dir / "costed_trades_post_warmup.csv", index=False)
        equity.to_csv(out_dir / "closed_trade_equity.csv", index=False)
        _write_json(out_dir / "session_breakdown.json", session_rows)
        _write_json(out_dir / "chronological_thirds_breakdown.json", block_rows)
        _write_json(out_dir / "evaluation_day_breakdown.json", daily_rows)
    _write_json(
        out_dir / "original_saved_uncertainty.json",
        {
            "net_expectancy_bootstrap_ci95": metrics["net_expectancy_bootstrap_ci95"],
            "cluster_bootstrap_ci95": stats.get("cluster_bootstrap_ci95"),
            "note": (
                "Original saved uncertainty; no resampling or independent confirmation performed."
            ),
        },
    )
    metadata = {
        "schema_version": 2,
        "delivery_mode": "full_raw_evidence" if include_raw_evidence else "compact_audit",
        "evidence_type": "derived_from_verified_saved_executions",
        "core_replay_id": core_replay_id,
        "costed_evaluation_id": costed_evaluation_id,
        "cost_policy": policy,
        "evaluation_dates": dates,
        "captured_execution_rows": len(executed_trades),
        "excluded_warmup_rows": len(executed_trades) - len(cohort),
        "post_warmup_resolved_execution_rows": len(costed),
        "original_trade_columns": list(executed_trades.columns),
        "original_trade_dtypes": {str(k): str(v) for k, v in executed_trades.dtypes.items()},
        "derived_columns": [c for c in costed if c.startswith("derived_")],
        "normalization": (
            "research_trades then _validate_and_normalize_executed_trades; "
            "open/unresolved rows rejected"
        ),
        "accounting": (
            "1 contract; realized_ticks * tick_size; net_points = gross_points - frozen "
            "round-turn cost; R divides by actual trade risk_points; USD uses frozen "
            "dollars_per_point"
        ),
        "equity": (
            "Resolved executions ordered by resolution_ts_utc then trade_id; initial "
            "peak 0; no mark-to-market or invented intrabar path"
        ),
        "session_grouping": (
            f"{session_column}.astype(str) exactly as saved strategy metrics; nulls stay "
            "explicit; no clock-based session imputation"
            if session_column
            else "unavailable: no recorded entry_session or session_doc column"
        ),
        "time_blocks": (
            "Sorted distinct traded days; thirds=max(1,n_days//3), slices [:thirds], "
            "[thirds:2*thirds], [2*thirds:]; empty blocks displayed but omitted from score"
        ),
        "daily_breakdown": (
            "All frozen evaluation dates, including zero-trade dates; grouped by original "
            "entry trading_day; per-group drawdown resets to zero and is not a contribution "
            "to full-study drawdown"
        ),
        "original_components": (
            ["original_saved_trade_stats.json", "original_saved_uncertainty.json"]
            if include_raw_evidence
            else ["original_saved_uncertainty.json"]
        ),
        "summary": overall,
        "time_block_sign_consistency": consistency,
        "session_stability_score": stability,
        "checks": checks,
        "all_checks_passed": all(row["passed"] for row in checks),
    }
    _write_json(out_dir / "costed_export_receipt.json", metadata)
    metadata["files"] = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(out_dir.iterdir())
        if path.is_file()
    }
    return _jsonable(metadata)
