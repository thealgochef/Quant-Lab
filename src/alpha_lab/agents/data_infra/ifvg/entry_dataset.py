"""IFVG candidate-label composition.

The v2 path emits long-form, candidate-ID-keyed counterfactual labels. It
persists setup direction, asserts stop side, uses each label family's own
barrier/path, and never turns a candidate label into executed P&L.

The lower half retains the original wide v1 gate-dataset composer for explicit
``legacy_v1`` reproduction only. Its setup-ID pivots, derived direction, and
modeled candidate outcomes are not accepted by a v2 execution/report path.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pandas as pd
from strategy_core.strategies.ifvg_smc.labels import resolve_ifvg_outcome
from strategy_core.types import Bar, Direction

from .config import IfvgCaptureConfig
from .day_artifacts import load_day_artifacts

__all__ = [
    "build_entry_dataset",
    "build_candidate_label_rows",
    "build_candidate_labels_from_tables",
    "assert_no_all_nan_columns",
    "NQ_COST_POINTS_ROUND_TURN",
]

_LABEL_NAMESPACE = uuid.UUID("9fbe0114-a6d3-4a24-92c9-3cb945933d65")


def build_candidate_label_rows(
    *,
    candidate_id: str,
    entry_ticks: int,
    stop_ticks: int,
    direction: Direction,
    entry_bar: Bar,
    forward_bars_1m,
    tick_size: float,
    r_multiples: tuple[float, ...] = (1.0, 1.5, 2.0),
) -> list[dict]:
    """Long-form, family-qualified counterfactual labels.

    Each family owns its barrier, resolution horizon, and MFE/MAE path. No
    generic R10 metrics are copied onto R15/R20 rows.
    """
    if not candidate_id:
        raise ValueError("candidate_id is required")
    rows: list[dict] = []
    for r_multiple in r_multiples:
        outcome = resolve_ifvg_outcome(
            entry_ticks=entry_ticks,
            stop_ticks=stop_ticks,
            direction=direction,
            entry_bar=entry_bar,
            forward_bars_1m=forward_bars_1m,
            tick_size=tick_size,
            r_multiple=r_multiple,
        )
        label_family = f"static_r_{r_multiple:g}_next_bar_stop_first_v1"
        label_id = str(
            uuid.uuid5(
                _LABEL_NAMESPACE,
                json.dumps(
                    [candidate_id, label_family],
                    separators=(",", ":"),
                ),
            )
        )
        rows.append(
            {
                "record_table": "candidate_label",
                "candidate_label_schema_version": 2,
                "candidate_label_id": label_id,
                "candidate_id": candidate_id,
                "label_family": label_family,
                "r_multiple": r_multiple,
                "label": outcome.label,
                "kernel_label": outcome.kernel_label,
                "target_ticks": outcome.target_ticks,
                "bars_to_resolution_generic_zero_based": (
                    outcome.bars_to_resolution
                ),
                "bars_after_entry_to_resolution": (
                    outcome.bars_after_entry_to_resolution
                ),
                "resolution_bar_id": outcome.resolution_bar_id,
                "mfe_r": outcome.mfe_r,
                "mae_r": outcome.mae_r,
                "censored": outcome.bars_after_entry_to_resolution is None,
                "censor_reason": (
                    "candidate_label_window_exhausted"
                    if outcome.bars_after_entry_to_resolution is None
                    else None
                ),
            }
        )
    return rows


def build_candidate_labels_from_tables(
    candidates: pd.DataFrame,
    *,
    bars_by_day: dict[str, tuple[Bar, ...]],
    tick_size: float,
    resolved_profile,
    r_multiples: tuple[float, ...] = (1.0, 1.5, 2.0),
) -> pd.DataFrame:
    """Resolve candidate-specific, trading-day-censored long-form labels.

    Geometry-incomplete candidates remain in the quarantine stream and do not
    receive a fabricated label. Direction comes from the setup record; stop
    side is asserted and never used to infer direction.
    """
    rows: list[dict] = []
    if candidates.empty:
        return pd.DataFrame()
    required = {
        "candidate_id",
        "setup_id",
        "direction",
        "entry_ticks",
        "proposed_stop_ticks",
        "trading_day",
        "entry_family",
        "trigger_cursor",
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"entry_candidate table is missing label evidence {missing}")

    ordered = candidates.sort_values(
        ["trading_day", "candidate_id"], kind="mergesort"
    )
    for candidate in ordered.to_dict("records"):
        geometry_entry_id = candidate.get("geometry_entry_bar_bar_id")
        geometry_cursor = candidate.get("geometry_feature_as_of_cursor")
        if geometry_entry_id is None or pd.isna(geometry_entry_id) or not geometry_entry_id:
            continue
        day = str(candidate["trading_day"])[:10]
        bars_1m = [
            bar
            for bar in bars_by_day.get(day, ())
            if bar.timeframe_ticks == 60
        ]
        entry_matches = [
            bar for bar in bars_1m if bar.bar_id == str(geometry_entry_id)
        ]
        if len(entry_matches) != 1:
            raise ValueError(
                "candidate geometry entry bar does not resolve uniquely: "
                f"{candidate['candidate_id']}"
            )
        entry_bar = entry_matches[0]
        direction = Direction(str(candidate["direction"]))
        entry_ticks = int(candidate["entry_ticks"])
        stop_ticks = int(candidate["proposed_stop_ticks"])
        if direction is Direction.LONG and stop_ticks >= entry_ticks:
            raise ValueError("LONG candidate stop must be below entry")
        if direction is Direction.SHORT and stop_ticks <= entry_ticks:
            raise ValueError("SHORT candidate stop must be above entry")
        forward = tuple(
            bar
            for bar in bars_1m
            if bar.availability_ts_utc > entry_bar.availability_ts_utc
        )
        label_rows = build_candidate_label_rows(
            candidate_id=str(candidate["candidate_id"]),
            entry_ticks=entry_ticks,
            stop_ticks=stop_ticks,
            direction=direction,
            entry_bar=entry_bar,
            forward_bars_1m=forward,
            tick_size=tick_size,
            r_multiples=r_multiples,
        )
        common = {
            "setup_id": str(candidate["setup_id"]),
            "trading_day": day,
            "strategy_id": candidate.get("strategy_id"),
            "strategy_version": candidate.get("strategy_version"),
            "profile_hash": candidate.get("profile_hash"),
            "profile_name": candidate.get("profile_name"),
            "qualification_mode": candidate.get("qualification_mode"),
            "section_config_hash": candidate.get("section_config_hash"),
            "evaluation_config_hash": resolved_profile.evaluation_config_hash,
            "entry_family": candidate.get("entry_family"),
            "entry_session": candidate.get("entry_session", "none"),
            "anchor_policy": candidate.get("anchor_policy"),
            "resolver_policy": candidate.get("resolver_policy"),
            "causality_parent": candidate.get("causality_parent"),
            "causality_opposing": candidate.get("causality_opposing"),
            "causality_entry": candidate.get("causality_entry"),
            "timeout_policy": candidate.get("timeout_policy"),
            "label_censor_policy": "trading_day_end_counterfactual_v1",
            "feature_as_of_cursor": (
                geometry_cursor
                if geometry_cursor is not None
                and not pd.isna(geometry_cursor)
                and str(geometry_cursor)
                else candidate["trigger_cursor"]
            ),
            "entry_bar_id": entry_bar.bar_id,
            "entry_cursor": candidate["trigger_cursor"],
            "is_warmup": bool(candidate.get("is_warmup", False)),
            "days_of_htf_history": int(
                candidate.get("days_of_htf_history", 0)
            ),
        }
        rows.extend({**row, **common} for row in label_rows)
    return pd.DataFrame(rows)


def assert_no_all_nan_columns(frame: pd.DataFrame) -> None:
    """FAIL CLOSED on silent pivot/union breakage: a fully-all-NaN column means
    a join key or a pivot rename collided (the ``parent_fvg_id`` lesson) —
    raise with the column names, never emit."""
    if frame.empty:
        return
    dead = [c for c in frame.columns if frame[c].isna().all()]
    if dead:
        raise ValueError(f"entry dataset emitted fully-all-NaN columns: {dead}")

#: NQ round-turn friction in POINTS: commission $2.64/side x2 = $5.28 -> /$20
#: per point = 0.264 pts, plus 0.5 tick slippage per side = 0.25 pts.
NQ_COST_POINTS_ROUND_TURN = 0.514

_R_FAMILIES = (1.0, 1.5, 2.0)

_STAGE_FEATURES = {
    "htf_tap": (
        "htf_tf_seconds",
        "fvg_size_ticks",
        "penetration_ticks",
        "ce_reached",
        "htf_age_seconds",
        "remaining_fraction",
        "registry_live_count",
        "nearest_level_kind",
        "nearest_level_distance_ticks",
    ),
    "parent_candidate": (
        "parent_tf_seconds",
        "fvg_fvg_id",
        "fvg_size_ticks",
        "distance_to_htf_ticks",
        "elapsed_1m_bars_since_tap",
        "confirmed_after",
        "fully_formed_after",
    ),
    "parent_lock": ("penetration_ticks", "ce_reached", "elapsed_1m_bars_since_selection"),
    "opposing": (
        "fvg_size_ticks",
        "distance_to_parent_ticks",
        "elapsed_1m_bars_since_lock",
        "confirmed_after",
        "fully_formed_after",
    ),
    "inversion": (
        "close_through_margin_ticks",
        "bars_armed_to_inversion",
        "sweep_sweep_confirmed",
        "sweep_swept_kinds",
        "sweep_max_penetration_ticks",
        "sweep_nearest_unswept_distance_ticks",
    ),
}

_STAGE_PREFIX = {
    "htf_tap": "tap",
    "parent_candidate": "parent",
    "parent_lock": "lock",
    "opposing": "opp",
    "inversion": "inv",
}

#: Pivot names that are explicit, not mechanical ``{prefix}_{column}``: the
#: parent's own gap id must emerge as ``parent_fvg_id`` (the join name every
#: downstream reader expects), not ``parent_fvg_fvg_id``.
_STAGE_RENAME_OVERRIDES: dict[str, dict[str, str]] = {
    "parent_candidate": {"fvg_fvg_id": "parent_fvg_id"},
}


def _atr14_ticks(bars_1m: list[Bar], before: datetime) -> float | None:
    """Mean true range (ticks) over the last 14 COMPLETE 1m bars closing
    strictly before ``before``."""
    window: list[int] = []
    prev_close: int | None = None
    for bar in bars_1m:
        if bar.close_ts_utc >= before:
            break
        tr = bar.high_ticks - bar.low_ticks
        if prev_close is not None:
            tr = max(tr, abs(bar.high_ticks - prev_close), abs(bar.low_ticks - prev_close))
        window.append(tr)
        prev_close = bar.close_ticks
    if len(window) < 14:
        return None
    return sum(window[-14:]) / 14.0


def build_entry_dataset(capture: pd.DataFrame, cfg: IfvgCaptureConfig) -> pd.DataFrame:
    """Compose the per-entry-candidate research dataset from a capture frame."""
    if capture.empty:
        return pd.DataFrame()
    entries = capture[capture["kind"] == "entry_candidate"].copy()
    if entries.empty:
        return pd.DataFrame()

    # The capture frame is a UNION of per-kind record columns, so entry rows
    # carry other kinds' fields as all-NA remnants (parent_fvg_id from
    # parent_lock/resolution rows, the resolution timestamps, ...). Those
    # namespaces belong exclusively to the stage pivot below — drop the
    # remnants first so the pivot output is authoritative and no colliding
    # all-NaN column leaks through the prefix passthrough.
    stage_namespaces = tuple(f"{p}_" for p in _STAGE_PREFIX.values())
    entries = entries.drop(columns=[c for c in entries.columns if c.startswith(stage_namespaces)])

    # ── stage-feature pivots (selected rows only, keyed by setup_id) ─────────
    for kind, columns in _STAGE_FEATURES.items():
        stage = capture[capture["kind"] == kind]
        if "selected" in stage.columns and kind in ("htf_tap", "parent_candidate", "opposing"):
            stage = stage[stage["selected"] == True]  # noqa: E712 — pandas mask
        stage = stage.drop_duplicates(subset=["envelope_setup_id"], keep="last")
        prefix = _STAGE_PREFIX[kind]
        overrides = _STAGE_RENAME_OVERRIDES.get(kind, {})
        available = [c for c in columns if c in stage.columns]
        renamed = stage[["envelope_setup_id", *available]].rename(
            columns={c: overrides.get(c, f"{prefix}_{c}") for c in available}
        )
        entries = entries.merge(renamed, on="envelope_setup_id", how="left")

    rows = []
    bars_cache: dict[str, list[Bar]] = {}
    for row in entries.itertuples(index=False):
        day = row.envelope_trading_day
        day_str = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]
        if day_str not in bars_cache:
            artifacts = load_day_artifacts(day_str, cfg, expected_seeds=None)
            bars_cache[day_str] = (
                [b for b in artifacts.bars if b.timeframe_ticks == 60] if artifacts else []
            )
        bars_1m = bars_cache[day_str]
        entry_ts = row.envelope_ts_utc
        entry_ticks = int(row.entry_ticks)
        stop_ticks = int(row.stop_ticks)
        direction = Direction.LONG if stop_ticks < entry_ticks else Direction.SHORT
        sign = 1 if direction is Direction.LONG else -1
        forward = [b for b in bars_1m if b.close_ts_utc > entry_ts]
        risk_pts = abs(entry_ticks - stop_ticks) * cfg.tick_size

        out = {
            "setup_id": row.envelope_setup_id,
            "trading_day": day_str,
            "entry_ts_utc": entry_ts,
            "entry_family": row.entry_family,
            # ifvg_retest rows carry a PLACEHOLDER entry model (bar close; no
            # CE/boundary entry references yet — recon A1). False until the
            # real entry references land; not comparable to the fresh family.
            "entry_model_final": row.entry_family == "fresh_fvg_continuation",
            "selected": bool(row.selected),
            "drop_reason": row.drop_reason,
            "direction": direction.value,
            "entry_ticks": entry_ticks,
            "stop_ticks": stop_ticks,
            "risk_ticks": int(row.risk_ticks),
            "risk_points": risk_pts,
            "session_engine": row.in_engine_session,
            "session_doc": row.in_doc_session,
            "bars_since_inversion": row.bars_since_inversion,
            "entry_to_parent_ticks": row.entry_to_parent_ticks,
            "is_warmup": bool(row.is_warmup),
            "days_of_htf_history": int(row.days_of_htf_history),
            "profile_hash": row.envelope_profile_hash,
            "strategy_version": row.envelope_strategy_version,
        }
        for col in entries.columns:
            if col.split("_", 1)[0] in _STAGE_PREFIX.values() and "_" in col:
                out[col] = getattr(row, col, None)

        atr = _atr14_ticks(bars_1m, entry_ts)
        out["atr14_1m_ticks"] = atr
        out["risk_atr"] = (abs(entry_ticks - stop_ticks) / atr) if atr else None
        highs = [b.high_ticks for b in bars_1m if b.close_ts_utc <= entry_ts]
        lows = [b.low_ticks for b in bars_1m if b.close_ts_utc <= entry_ts]
        if highs and lows and max(highs) > min(lows):
            day_hi, day_lo = max(highs), min(lows)
            out["day_range_ticks"] = day_hi - day_lo
            out["position_in_day_range"] = (entry_ticks - day_lo) / (day_hi - day_lo)
            out["dist_day_high_ticks"] = day_hi - entry_ticks
            out["dist_day_low_ticks"] = entry_ticks - day_lo
        else:
            out["day_range_ticks"] = out["position_in_day_range"] = None
            out["dist_day_high_ticks"] = out["dist_day_low_ticks"] = None

        # ── labels through the shared kernel, per R family ───────────────────
        if forward:
            for r in _R_FAMILIES:
                outcome = resolve_ifvg_outcome(
                    entry_ticks=entry_ticks,
                    stop_ticks=stop_ticks,
                    direction=direction,
                    forward_bars_1m=forward,
                    tick_size=cfg.tick_size,
                    r_multiple=r,
                )
                tag = f"r{str(r).replace('.', '')}"
                out[f"label_{tag}"] = outcome.label
                out[f"bars_to_res_{tag}"] = outcome.bars_to_resolution
                if r == 1.0:
                    out["mfe_r"] = outcome.mfe_r
                    out["mae_r"] = outcome.mae_r
                # Realized points: barrier for win/loss, day-end close for timeout.
                if outcome.label == "win":
                    realized = r * risk_pts
                elif outcome.label == "loss":
                    realized = -risk_pts
                else:
                    realized = sign * (forward[-1].close_ticks - entry_ticks) * cfg.tick_size
                out[f"realized_pts_{tag}"] = realized
                out[f"realized_r_net_{tag}"] = (
                    (realized - NQ_COST_POINTS_ROUND_TURN) / risk_pts if risk_pts else None
                )
            out["entry_slippage_next_open_pts"] = (
                sign * (forward[0].open_ticks - entry_ticks) * cfg.tick_size
            )
            out["label_window_end"] = forward[-1].close_ts_utc
        else:
            for r in _R_FAMILIES:
                tag = f"r{str(r).replace('.', '')}"
                out[f"label_{tag}"] = "no_forward"
            out["label_window_end"] = entry_ts
        rows.append(out)
    dataset = pd.DataFrame(rows)
    assert_no_all_nan_columns(dataset)
    return dataset
