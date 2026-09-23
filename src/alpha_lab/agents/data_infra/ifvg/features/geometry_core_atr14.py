"""Geometry ratios using the existing registered Core arithmetic ATR14 scale.

This corrects the prefit inventory: registered sweep features already use this
scale. The original ATR20 preparation stays readable; no model results informed
the correction. The numerical scale is called directly from Strategy-Core.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry_features import (
    _materialize_geometry_features,
    _normalized_bars,
    load_geometry_features,
    save_geometry_features,
)

__all__ = [
    "GEOMETRY_BLOCK_KEY", "GEOMETRY_BUNDLE_KEY", "GEOMETRY_FORMULA_VERSION",
    "GEOMETRY_FEATURES", "GEOMETRY_FORMULAS", "compute_core_atr14",
    "materialize_geometry_features", "load_geometry_features", "save_geometry_features",
]

GEOMETRY_BLOCK_KEY = "IFVG_GEOMETRY_CORE_ATR14_V1"
GEOMETRY_BUNDLE_KEY = "B0_GEOMETRY_CORE_ATR14_V1"
GEOMETRY_FORMULA_VERSION = "ifvg_geometry_core_arithmetic_atr14_1m_v1"
GEOMETRY_FEATURES = (
    "geo_inversion_clearance_atr14", "geo_parent_htf_distance_atr14",
    "geo_opposing_parent_size_ratio",
)
ATR_PERIOD = 14
GEOMETRY_FORMULAS = {
    GEOMETRY_FEATURES[0]: "close_through_margin_ticks / decision_atr14_ticks",
    GEOMETRY_FEATURES[1]: "distance_to_htf_ticks / decision_atr14_ticks",
    GEOMETRY_FEATURES[2]: "geometry_opposing_size_ticks / geometry_parent_size_ticks",
    "volatility": (
        "Existing Core ContextFeatureConfig.atr_scale_period=14 and "
        "EqualLevelPoolTracker.on_source_bar/atr_at(60). Complete TIME 60s bars only, "
        "strict availability/bar-ID order; original source-chain state carries across "
        "trading-day and named-session boundaries, including original warmup. "
        "Arithmetic mean of trailing14 TR, TR=max(H-L,abs(H-prevC),abs(L-prevC)); "
        "prevC is preceding complete source bar (including before the 14-bar window); "
        "at chain startup first previous close is own close, so firstTR=H-L. "
        "First13 complete source-chain bars null. No fabricated minutes or gap reset. "
        "Core retains21 source bars and restores these through day seeds. "
        "The completed entry bar updates the scale before reducer/candidate emission. "
        "Zero scale yields null ratios. All input prices/ATR are ticks; ratios dimensionless."
    ),
    "authority": (
        "strategy_core/structures/equal_levels.py::EqualLevelPoolTracker._scales; "
        "context_config.py::_schema_payload.observational_scale_periods; "
        "registered ctx_sweep_sweep_depth_normalized, "
        "ctx_sweep_reclaim_close_distance_normalized, ctx_sweep_distance_at_lock_normalized"
    ),
}


def _normalized_core_bars(bars: pd.DataFrame) -> pd.DataFrame:
    frame = _normalized_bars(bars)
    if "kind" not in bars:
        raise ValueError("Core ATR14 source requires explicit TIME-bar kind")
    kinds = bars.set_index("bar_id")["kind"].map(lambda value: getattr(value, "value", value))
    frame["kind"] = frame.bar_id.map(kinds)
    if not frame.kind.eq("time").all():
        raise ValueError("Core ATR14 source includes a non-TIME bar")
    return frame.sort_values(["availability_ts_utc", "bar_id"]).reset_index(drop=True)


def compute_core_atr14(bars: pd.DataFrame) -> pd.DataFrame:
    """Use Core's actual tracker, maintaining exactly its source-chain scale state."""
    from strategy_core.strategies.ifvg_smc.context_config import (  # noqa: PLC0415
        ContextFeatureConfig,
        build_context_identity,
    )
    from strategy_core.structures.equal_levels import EqualLevelPoolTracker  # noqa: PLC0415

    from ..search.research_data import frame_to_bars  # noqa: PLC0415

    frame = _normalized_core_bars(bars)
    original = bars.set_index("bar_id", drop=False).loc[frame.bar_id].reset_index(drop=True)
    source_bars = frame_to_bars(original)
    identity = build_context_identity(ContextFeatureConfig(), symbol="NQ")
    tracker = EqualLevelPoolTracker(identity=identity, atr_period=ATR_PERIOD)
    frame["decision_atr14_ticks"] = np.nan
    frame["atr_completed_bar_count"] = 0
    frame["atr_missing_reason"] = "insufficient_source_chain_completed_bars"
    completed = 0
    for index, bar in enumerate(source_bars):
        if pd.Timestamp(bar.availability_ts_utc) != frame.iloc[index].availability_ts_utc:
            raise ValueError("Core ATR14 bar adapter changed the authoritative availability")
        tracker.on_source_bar(bar)
        if not bar.is_complete:
            frame.at[index, "atr_missing_reason"] = "decision_bar_incomplete"
            frame.at[index, "atr_completed_bar_count"] = completed
            continue
        completed += 1
        frame.at[index, "atr_completed_bar_count"] = completed
        value = tracker.atr_at(60)
        if value is not None:
            frame.at[index, "decision_atr14_ticks"] = value
            frame.at[index, "atr_missing_reason"] = "zero_atr_denominator" if value == 0 else ""
    return frame


def materialize_geometry_features(view, bars_1m: pd.DataFrame, *, source_reference: dict):
    """Bind the saved source configuration and use its registered Core scale."""
    from ..search.research_data import CONTEXT_STORE, ResearchContextEnvelope  # noqa: PLC0415
    from ..search.store import load_verified_envelope  # noqa: PLC0415

    source_path = Path(source_reference["path"]).resolve()
    context = load_verified_envelope(
        source_path.parent.parent.parent, CONTEXT_STORE, source_reference["artifact_id"],
        ResearchContextEnvelope,
    )
    config = json.loads(context.payload.context_config_json)
    if (config.get("atr_scale_period") != ATR_PERIOD
            or config.get("require_complete_time_bars") is not True):
        raise ValueError("geometry Core ATR14 differs from the saved context convention")
    return _materialize_geometry_features(
        view, bars_1m, source_reference=source_reference, feature_names=GEOMETRY_FEATURES,
        formula_version=GEOMETRY_FORMULA_VERSION, formula_contract=GEOMETRY_FORMULAS,
        volatility_field="decision_atr14_ticks", compute_volatility=compute_core_atr14,
        normalize_bars=_normalized_core_bars,
    )
