"""Separately identified, cutoff-safe counterfactual IFVG label derivations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .context_experiment_contracts import (
    IfvgContextLabelConfig,
    canonical_contract_sha256,
)

__all__ = [
    "ContextLabelDerivation",
    "derive_context_candidate_labels",
]


@dataclass(frozen=True, slots=True)
class ContextLabelDerivation:
    derivation_id: str
    config_hash: str
    cutoff_ts_utc: pd.Timestamp
    labels: pd.DataFrame


def _column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame:
            return name
    raise ValueError(f"required label source column is missing: {names}")


def _integer(value: Any, label: str) -> int:
    if pd.isna(value):
        raise ValueError(f"{label} is unavailable")
    numeric = int(value)
    if numeric != value:
        raise ValueError(f"{label} is not an integer tick value")
    return numeric


def _direction(value: Any) -> str:
    normalized = str(value).lower()
    if normalized in {"long", "bullish"}:
        return "long"
    if normalized in {"short", "bearish"}:
        return "short"
    raise ValueError(f"unsupported candidate direction {value!r}")


def _bar_evidence_hash(frame: pd.DataFrame, *, id_col: str) -> str:
    payload = []
    for row in frame.to_dict("records"):
        payload.append(
            {
                "bar_id": str(row[id_col]),
                "close_ts_utc": pd.Timestamp(row["_close_ts"]).isoformat(),
                "high_ticks": int(row["_high_ticks"]),
                "low_ticks": int(row["_low_ticks"]),
            }
        )
    return canonical_contract_sha256(payload)


def derive_context_candidate_labels(
    candidates: pd.DataFrame,
    bars_1m: pd.DataFrame,
    config: IfvgContextLabelConfig,
    *,
    cutoff_ts_utc: str | pd.Timestamp = "2026-06-10T21:00:00Z",
) -> ContextLabelDerivation:
    """Derive one barrier family without reusing another family's path metrics.

    Eligible bars satisfy ``entry_ts < close_ts < cutoff``.  Same-bar stop and
    target collisions are resolved stop-first, matching the pinned v2 resolver.
    """

    if candidates["candidate_id"].isna().any() or candidates["candidate_id"].duplicated().any():
        raise ValueError("candidate labels require unique non-null candidate IDs")
    cutoff = pd.Timestamp(cutoff_ts_utc)
    if cutoff.tzinfo is None:
        raise ValueError("label cutoff must be timezone-aware")
    cutoff = cutoff.tz_convert("UTC")
    entry_ts_col = _column(candidates, "entry_ts_utc", "feature_as_of_ts")
    entry_col = _column(candidates, "entry_ticks", "geometry_entry_ticks")
    stop_col = _column(candidates, "proposed_stop_ticks", "stop_ticks", "geometry_stop_ticks")
    direction_col = _column(candidates, "direction")
    close_col = _column(bars_1m, "close_ts_utc", "logical_close_ts_utc", "availability_ts_utc")
    high_col = _column(bars_1m, "high_ticks")
    low_col = _column(bars_1m, "low_ticks")
    id_col = _column(bars_1m, "bar_id")

    bars = bars_1m.copy()
    bars["_close_ts"] = pd.to_datetime(bars[close_col], utc=True, errors="raise")
    bars["_high_ticks"] = pd.to_numeric(bars[high_col], errors="raise").astype("int64")
    bars["_low_ticks"] = pd.to_numeric(bars[low_col], errors="raise").astype("int64")
    if (bars["_high_ticks"] < bars["_low_ticks"]).any():
        raise ValueError("label source bar high is below low")
    bars = bars.sort_values(["_close_ts", id_col], kind="mergesort")
    if bars[id_col].astype(str).duplicated().any():
        raise ValueError("label source bar IDs are duplicated")

    config_hash = config.identity
    results: list[dict[str, Any]] = []
    for candidate in candidates.sort_values("candidate_id", kind="mergesort").to_dict("records"):
        candidate_id = str(candidate["candidate_id"])
        entry_ts_value = candidate[entry_ts_col]
        if pd.isna(entry_ts_value) and "feature_as_of_ts" in candidate:
            entry_ts_value = candidate["feature_as_of_ts"]
        if pd.isna(entry_ts_value):
            raise ValueError("candidate entry timestamp is unavailable")
        entry_ts = pd.Timestamp(entry_ts_value)
        if entry_ts.tzinfo is None:
            raise ValueError("candidate entry timestamp must be timezone-aware")
        entry_ts = entry_ts.tz_convert("UTC")
        entry = _integer(candidate[entry_col], "entry_ticks")
        source_stop = _integer(candidate[stop_col], "stop_ticks")
        direction = _direction(candidate[direction_col])
        source_risk = entry - source_stop if direction == "long" else source_stop - entry
        if source_risk < 1:
            raise ValueError("candidate stop is on the wrong side or has zero risk")
        if config.label_family == "fixed_sl_tp":
            risk = int(config.fixed_stop_ticks or 0)
            reward_ticks = int(config.fixed_target_ticks or 0)
        else:
            risk = source_risk
            reward_ticks = math.ceil(source_risk * config.reward_r)
        stop = entry - risk if direction == "long" else entry + risk
        target = entry + reward_ticks if direction == "long" else entry - reward_ticks

        eligible = bars[(bars["_close_ts"] > entry_ts) & (bars["_close_ts"] < cutoff)]
        if config.horizon_bars is not None:
            eligible = eligible.head(config.horizon_bars)
        resolution = None
        resolution_ts = None
        resolution_bar_id = None
        bars_after = None
        path_rows: list[dict[str, Any]] = []
        max_favorable = 0
        max_adverse = 0
        for ordinal, bar in enumerate(eligible.to_dict("records"), start=1):
            high = int(bar["_high_ticks"])
            low = int(bar["_low_ticks"])
            favorable = high - entry if direction == "long" else entry - low
            adverse = entry - low if direction == "long" else high - entry
            max_favorable = max(max_favorable, favorable)
            max_adverse = max(max_adverse, adverse)
            path_rows.append(bar)
            stop_hit = low <= stop if direction == "long" else high >= stop
            target_hit = high >= target if direction == "long" else low <= target
            if stop_hit or target_hit:
                resolution = "stop" if stop_hit else "target"
                resolution_ts = pd.Timestamp(bar["_close_ts"])
                resolution_bar_id = str(bar[id_col])
                bars_after = ordinal
                break
        scanned = pd.DataFrame(path_rows, columns=eligible.columns)
        if resolution == "target":
            label = "win"
            gross_r = reward_ticks / risk
        elif resolution == "stop":
            label = "loss"
            gross_r = -1.0
        else:
            label = "censored"
            gross_r = None
        censor_reason = None
        if resolution is None:
            censor_reason = (
                "horizon_exhausted"
                if config.horizon_bars is not None and len(eligible) >= config.horizon_bars
                else "censored_protected_boundary"
            )
        evidence_hash = _bar_evidence_hash(scanned, id_col=id_col)
        label_id = canonical_contract_sha256(
            {
                "candidate_id": candidate_id,
                "label_config_hash": config_hash,
                "entry_ts_utc": entry_ts.isoformat(),
                "entry_ticks": entry,
                "stop_ticks": stop,
                "target_ticks": target,
                "cutoff_ts_utc": cutoff.isoformat(),
                "bar_evidence_hash": evidence_hash,
            }
        )
        results.append(
            {
                "context_label_id": label_id,
                "candidate_id": candidate_id,
                "setup_id": candidate.get("setup_id"),
                "trading_day": candidate.get("trading_day"),
                "label_config_hash": config_hash,
                "label_family": config.label_family,
                "reward_r": config.reward_r,
                "cost_per_trade_r": config.cost_per_trade_r,
                "direction": direction,
                "entry_ts_utc": entry_ts,
                "entry_ticks": entry,
                "stop_ticks": stop,
                "target_ticks": target,
                "risk_ticks": risk,
                "label": label,
                "binary_target": None if label == "censored" else int(label == "win"),
                "resolution": resolution,
                "resolution_ts_utc": resolution_ts,
                "resolution_bar_id": resolution_bar_id,
                "bars_after_entry_to_resolution": bars_after,
                "eligible_bar_count": len(eligible),
                "scanned_bar_count": len(scanned),
                "first_eligible_close_ts_utc": (
                    eligible.iloc[0]["_close_ts"] if not eligible.empty else None
                ),
                "last_evaluated_close_ts_utc": (
                    scanned.iloc[-1]["_close_ts"] if not scanned.empty else None
                ),
                "mfe_ticks": max_favorable,
                "mae_ticks": max_adverse,
                "mfe_r": max_favorable / risk,
                "mae_r": max_adverse / risk,
                "gross_r": gross_r,
                "net_r": (
                    None if gross_r is None else gross_r - config.cost_per_trade_r
                ),
                "censored": resolution is None,
                "censor_reason": censor_reason,
                "entry_available": True,
                "resolution_available": resolution is not None,
                "cutoff_ts_utc": cutoff,
                "bar_evidence_hash": evidence_hash,
            }
        )
    labels = pd.DataFrame(results)
    derivation_id = canonical_contract_sha256(
        {
            "label_config_hash": config_hash,
            "cutoff_ts_utc": cutoff.isoformat(),
            "label_ids": labels["context_label_id"].tolist(),
        }
    )
    return ContextLabelDerivation(
        derivation_id=derivation_id,
        config_hash=config_hash,
        cutoff_ts_utc=cutoff,
        labels=labels,
    )
