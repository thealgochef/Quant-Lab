"""Source loaders normalizing trades into ``TradePath`` rows.

Three sources:

- ``load_executions_trades`` — a Trade-Lab executions directory (per-trading-day
  JSONL written by the paper-execution tracker; ``close`` rows are
  self-contained and carry BOTH fill columns) joined with prediction-journal
  ``outcome`` rows on ``prediction_id`` for MFE/MAE.
- ``load_journal_trades`` — EVIDENCE MODE: every resolved prediction-journal
  outcome treated as a 1-contract trade (includes outcomes whose predictions
  were serving-ineligible — stated in the notes). Points derive from the
  resolution barriers (tp_hit → +tp, sl_hit → −sl); the conservative column
  mirrors the Trade-Lab tracker's fill model (entry 1 tick adverse; SL exit 1
  tick worse; TP exits at the barrier in both columns).
- ``load_oos_trades`` — a bundle's ``oos_predictions.parquet``. Post-P1 files
  carry ``max_mfe_pts``/``max_mae_pts``/``entry_price``/``resolution_type``;
  pre-P1 files degrade to realized-only with a stated reason. OOS fills are
  idealized: the conservative column EQUALS the optimistic one (no synthetic
  slippage model is invented), which the notes state.

Trading-day dating mirrors Trade-Lab's canonical 18:00 America/New_York roll
(``trade_lab.domain.trading_day``); trades are dated by ENTRY.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from alpha_lab.propsim.models import TradePath

_ET = ZoneInfo("America/New_York")
_TRADING_DAY_BOUNDARY = time(18, 0)

#: The ratified label -> resolution mapping (mirrors Trade-Lab serving and the
#: PROP-SIM P1 OOS writer) for pre-P1 parquets lacking ``resolution_type``.
_RESOLUTION_TYPE_BY_LABEL = {
    "tradeable_reversal": "tp_hit",
    "trap_reversal": "sl_hit",
    "aggressive_blowthrough": "sl_hit",
}


def trading_day_for(ts_utc: datetime) -> date:
    """The trading day a UTC instant belongs to: ``[prev 18:00 ET, 18:00 ET)``."""
    local = ts_utc.astimezone(_ET)
    day = local.date()
    if local.time() >= _TRADING_DAY_BOUNDARY:
        day += timedelta(days=1)
    return day


@dataclass(frozen=True)
class LoadedTrades:
    """Normalized trades + provenance for the report layer."""

    trades: list[TradePath]
    source: str
    excursions_available: bool
    degradation_reason: str | None
    notes: list[str] = field(default_factory=list)


def _parse_ts(value: object) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _finite(value: object) -> float | None:
    """A finite float, or None (rejects NaN/inf/non-numeric)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _iter_jsonl_rows(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _journal_outcomes(journal_dir: Path) -> dict[str, dict]:
    """``prediction_id`` -> outcome row (last write wins)."""
    outcomes: dict[str, dict] = {}
    for row in _iter_jsonl_rows(journal_dir):
        if row.get("type") == "outcome" and row.get("prediction_id"):
            outcomes[str(row["prediction_id"])] = row
    return outcomes


def load_executions_trades(
    executions_dir: Path,
    journal_dir: Path | None = None,
) -> LoadedTrades:
    """Trades from tracker ``close`` rows, excursions joined from the journal."""
    if not executions_dir.is_dir():
        msg = f"executions directory not found: {executions_dir}"
        raise FileNotFoundError(msg)
    rows = _iter_jsonl_rows(executions_dir)
    outcomes = _journal_outcomes(journal_dir) if journal_dir is not None else {}

    opens: set[str] = set()
    closes: list[dict] = []
    resets = 0
    for row in rows:
        row_type = row.get("type")
        if row_type == "open" and row.get("prediction_id"):
            opens.add(str(row["prediction_id"]))
        elif row_type == "close":
            closes.append(row)
        elif row_type == "reset":
            resets += 1

    # Executions files are append-only across replay re-runs: replaying the same
    # day again journals the SAME physical trades under fresh uuids (reset rows
    # mark run boundaries but carry no epoch key). Dedup closes by their physical
    # fill signature, last-write-wins, and surface the drop count.
    deduped_closes: dict[tuple, dict] = {}
    for row in closes:
        signature = (
            row.get("entry_ts_utc"),
            row.get("entry_price"),
            row.get("exit_price"),
            row.get("direction"),
            row.get("reason"),
        )
        deduped_closes[signature] = row
    duplicate_closes = len(closes) - len(deduped_closes)

    trades: list[TradePath] = []
    skipped_unparseable = 0
    missing_excursions = 0
    for row in deduped_closes.values():
        entry_ts = _parse_ts(row.get("entry_ts_utc")) or _parse_ts(row.get("ts_utc"))
        points = _finite(row.get("points"))
        points_cons = _finite(row.get("points_conservative"))
        if entry_ts is None or points is None or points_cons is None:
            skipped_unparseable += 1
            continue
        outcome = outcomes.get(str(row.get("prediction_id") or ""))
        mfe = _finite(outcome.get("max_mfe_pts")) if outcome else None
        mae = _finite(outcome.get("max_mae_pts")) if outcome else None
        if mfe is None or mae is None:
            missing_excursions += 1
        trades.append(
            TradePath(
                day=trading_day_for(entry_ts),
                entry_ts=entry_ts,
                points_optimistic=points,
                points_conservative=points_cons,
                mfe_pts=mfe,
                mae_pts=mae,
                resolution=row.get("reason"),
            )
        )

    unmatched_opens = opens - {
        str(row.get("prediction_id") or "") for row in closes
    }
    notes = [
        f"executions dir: {executions_dir}",
        f"rows: {len(rows)} total, {len(closes)} closes, {len(opens)} opens, {resets} resets",
        f"replay re-run duplicate closes dropped (fill-signature dedup, "
        f"last-write-wins): {duplicate_closes}",
        f"opens without a close (skipped — never completed): {len(unmatched_opens)}",
        f"closes skipped as unparseable (missing/non-finite fields): {skipped_unparseable}",
    ]
    degradation: str | None = None
    if journal_dir is None:
        degradation = (
            "no --journal directory supplied — MFE/MAE unavailable; "
            "unrealized_adverse_first degrades to realized-only"
        )
        notes.append(degradation)
    else:
        notes.append(
            f"journal join: {len(trades) - missing_excursions}/{len(trades)} trades "
            f"carry MFE/MAE from journal outcomes"
        )
        if trades and missing_excursions == len(trades):
            degradation = (
                "no execution close matched a journal outcome — "
                "unrealized_adverse_first degrades to realized-only"
            )
    excursions_available = bool(trades) and missing_excursions < len(trades)
    return LoadedTrades(
        trades=trades,
        source="executions",
        excursions_available=excursions_available,
        degradation_reason=degradation,
        notes=notes,
    )


def load_journal_trades(
    journal_dir: Path,
    *,
    tp_points: float,
    sl_points: float,
    tick_size: float = 0.25,
) -> LoadedTrades:
    """EVIDENCE MODE: every resolved journal outcome as a 1-contract trade."""
    if not journal_dir.is_dir():
        msg = f"journal directory not found: {journal_dir}"
        raise FileNotFoundError(msg)
    rows = _iter_jsonl_rows(journal_dir)
    predictions: dict[str, dict] = {}
    outcome_rows: list[dict] = []
    for row in rows:
        if row.get("type") == "prediction" and row.get("prediction_id"):
            predictions[str(row["prediction_id"])] = row
        elif row.get("type") == "outcome":
            outcome_rows.append(row)

    # Warm restarts re-predict the SAME physical touch under fresh
    # prediction_ids/touch_ids (the WARM-PERF journal-dupe class; files were not
    # retroactively cleaned) — a 1-contract account cannot fill one touch N
    # times. Dedup outcomes by the physical touch signature (the prediction's
    # instant + level + direction), last-write-wins, and surface the drop count.
    deduped_outcomes: dict[tuple, dict] = {}
    for row in outcome_rows:
        prediction = predictions.get(str(row.get("prediction_id") or ""))
        if prediction is not None:
            signature = (
                "prediction",
                prediction.get("ts_utc"),
                prediction.get("level_kind"),
                prediction.get("level_price_ticks"),
                prediction.get("direction"),
            )
        else:
            signature = ("outcome", row.get("ts_utc"), row.get("entry_price"), None, None)
        deduped_outcomes[signature] = row
    duplicates_dropped = len(outcome_rows) - len(deduped_outcomes)

    trades: list[TradePath] = []
    skipped = 0
    eligible = 0
    missing_prediction = 0
    for row in deduped_outcomes.values():
        resolution = row.get("resolution_type")
        if resolution == "tp_hit":
            points = tp_points
            points_cons = tp_points - tick_size
        elif resolution == "sl_hit":
            points = -sl_points
            points_cons = -(sl_points + 2.0 * tick_size)
        else:
            skipped += 1
            continue
        prediction = predictions.get(str(row.get("prediction_id") or ""))
        if prediction is None:
            missing_prediction += 1
            entry_ts = _parse_ts(row.get("ts_utc"))
        else:
            entry_ts = _parse_ts(prediction.get("ts_utc")) or _parse_ts(row.get("ts_utc"))
            if prediction.get("is_eligible"):
                eligible += 1
        if entry_ts is None:
            skipped += 1
            continue
        trades.append(
            TradePath(
                day=trading_day_for(entry_ts),
                entry_ts=entry_ts,
                points_optimistic=float(points),
                points_conservative=float(points_cons),
                mfe_pts=_finite(row.get("max_mfe_pts")),
                mae_pts=_finite(row.get("max_mae_pts")),
                resolution=str(resolution),
            )
        )
    notes = [
        f"journal dir: {journal_dir}",
        "EVIDENCE MODE: every resolved journal outcome treated as a 1-contract "
        "trade — includes outcomes of serving-INELIGIBLE predictions",
        f"warm-restart duplicate outcomes dropped (touch-signature dedup, "
        f"last-write-wins): {duplicates_dropped}",
        f"outcomes: {len(outcome_rows)} rows -> {len(trades)} trades "
        f"({skipped} skipped: unresolved/unparseable)",
        f"eligible predictions among trades: {eligible}; outcomes without a "
        f"prediction row (dated by resolved ts): {missing_prediction}",
        f"points model: tp_hit -> +{tp_points}, sl_hit -> -{sl_points}; "
        f"conservative mirrors the tracker fill model (entry 1 tick adverse, "
        f"SL exit 1 tick worse, tick={tick_size})",
    ]
    with_excursions = sum(1 for t in trades if t.mae_pts is not None)
    degradation = None
    if trades and with_excursions == 0:
        degradation = (
            "journal outcomes carry no MFE/MAE — unrealized_adverse_first "
            "degrades to realized-only"
        )
    return LoadedTrades(
        trades=trades,
        source="journal",
        excursions_available=with_excursions > 0,
        degradation_reason=degradation,
        notes=notes,
    )


def load_oos_trades(
    parquet_path: Path,
    *,
    tp_points: float,
    sl_points: float,
    gate_column: str | None = None,
) -> LoadedTrades:
    """Trades from a bundle's OOS predictions parquet."""
    if not parquet_path.is_file():
        msg = f"OOS parquet not found: {parquet_path}"
        raise FileNotFoundError(msg)
    frame = pd.read_parquet(parquet_path)
    total_rows = len(frame)
    if gate_column is not None:
        if gate_column not in frame.columns:
            msg = (
                f"gate column {gate_column!r} not in {parquet_path.name} "
                f"(columns: {', '.join(frame.columns)})"
            )
            raise ValueError(msg)
        frame = frame[frame[gate_column].astype(bool)]
    frame = frame[frame["label"].notna()]

    has_excursions = (
        "max_mfe_pts" in frame.columns
        and "max_mae_pts" in frame.columns
        and bool(frame["max_mae_pts"].notna().any())
    )
    degradation = None
    if not has_excursions:
        degradation = (
            "pre-P1 OOS parquet: max_mfe_pts/max_mae_pts absent or empty — "
            "unrealized_adverse_first degrades to realized-only"
        )

    trades: list[TradePath] = []
    skipped = 0
    for row in frame.itertuples(index=False):
        label = str(row.label)
        resolution = getattr(row, "resolution_type", None)
        if not isinstance(resolution, str) or not resolution:
            resolution = _RESOLUTION_TYPE_BY_LABEL.get(label)
        if resolution == "tp_hit":
            points = tp_points
        elif resolution == "sl_hit":
            points = -sl_points
        else:
            skipped += 1
            continue
        ts = pd.Timestamp(row.timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        entry_ts = ts.to_pydatetime()
        mfe = getattr(row, "max_mfe_pts", None)
        mae = getattr(row, "max_mae_pts", None)
        mfe = None if mfe is None or (isinstance(mfe, float) and math.isnan(mfe)) else float(mfe)
        mae = None if mae is None or (isinstance(mae, float) and math.isnan(mae)) else float(mae)
        trades.append(
            TradePath(
                day=trading_day_for(entry_ts),
                entry_ts=entry_ts,
                points_optimistic=points,
                points_conservative=points,
                mfe_pts=mfe,
                mae_pts=mae,
                resolution=resolution,
            )
        )
    notes = [
        f"OOS parquet: {parquet_path}",
        f"rows: {total_rows} total; gate: "
        + (f"{gate_column}" if gate_column else "none (all labeled rows are trades)")
        + f"; trades: {len(trades)} ({skipped} skipped: unmapped label/resolution)",
        f"points model: idealized barrier fills, tp_hit -> +{tp_points}, "
        f"sl_hit -> -{sl_points}; conservative column EQUALS optimistic "
        "(no synthetic slippage model)",
    ]
    if degradation:
        notes.append(degradation)
    return LoadedTrades(
        trades=trades,
        source="oos",
        excursions_available=has_excursions,
        degradation_reason=degradation,
        notes=notes,
    )
