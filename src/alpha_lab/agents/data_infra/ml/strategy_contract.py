"""
Strategy contract emission for runtime (Trade-Lab) consumption.

A trained model bundle (model.cbm + metadata.json + evaluation.json) describes
*what was trained*. It does NOT, on its own, fully describe the *strategy
semantics* a runtime needs to reproduce features and labels live: the session
scheme, touch rule, level scheme, feature windows, and label policy are only
implicit in how the dashboard-utility builder computed the dataset.

`strategy.json` makes those semantics explicit and versioned so a downstream
runtime (Trade-Lab) can be driven by the contract instead of hardcoding one
strategy. When research changes the strategy (sessions, touch rule, windows,
thresholds, feature set), it ships a new bundle with a new strategy.json and
the runtime adapts via config rather than a code rewrite.

Contract version + engine version are imported from ``strategy_core`` (not
restated here).

Only the production-aligned `dashboard_utility` training mode emits a full
contract. Other modes emit a minimal record (the runtime is not expected to
serve them).

NO RESTATED LITERALS. Every STRUCTURAL field of the contract (the values that
describe the *engine/strategy semantics*, as opposed to the per-training-run
inputs that legitimately vary with config) is single-sourced from
``strategy_core.constants`` -- the one place the engine computes with them -- so
the description (this emitter) and the computation (the engine) can never drift
apart. The standing no-drift test
(``tests/agents/test_strategy_contract_nodrift.py``) asserts each structural
field equals its constant AND that no new contract field can appear without a
constant or an allow-listed per-run config source.
"""

from __future__ import annotations

import logging
from datetime import time

from strategy_core import CONTRACT_VERSION, ENGINE_VERSION
from strategy_core import constants as k

from alpha_lab.agents.data_infra.ml.config import (
    LIVE_APPROACH_FEATURES,
    LIVE_INTERACTION_FEATURES,
    MLPipelineConfig,
)

logger = logging.getLogger(__name__)


def _hhmm(t: time) -> str:
    return t.strftime("%H:%M")


def _session_block(name: str) -> dict:
    """Emit a session's start/end (+crosses_midnight) from the engine scheme."""
    w = k.RESEARCH_SESSION_SCHEME.sessions[name]
    block = {"start": _hhmm(w.start), "end": _hhmm(w.end)}
    if w.crosses_midnight:
        block["crosses_midnight"] = True
    return block


def build_strategy_contract(
    config: MLPipelineConfig,
    selected_features: list[str] | None,
    *,
    strategy_id: str,
) -> dict | None:
    """Build a versioned strategy contract dict for runtime consumption.

    Every STRUCTURAL field is sourced from ``strategy_core.constants`` (or the
    engine package); only per-training-run scalars (tp/sl/trap, the interaction/
    approach windows, bar_type, tick_size, instrument, model.*, provenance hash,
    strategy_id, selected feature names) come from ``config`` -- those are inputs,
    not engine semantics.

    Args:
        config: The full pipeline config used for this training run.
        selected_features: Ordered feature names the model was trained on
            (RFECV-selected subset, contractual order). Falls back to the
            full live feature set when not provided.
        strategy_id: Identifier for this bundle (typically the output dir name).

    Returns:
        A JSON-serialisable contract dict, or None if the training mode is
        unknown/unsupported.
    """
    mode = getattr(config, "training_mode", "unknown")

    feature_names = list(selected_features) if selected_features else list(LIVE_INTERACTION_FEATURES)

    if mode != "dashboard_utility":
        # Minimal record only; the runtime is not expected to serve non
        # dashboard-utility strategies. Keep enough to identify the bundle.
        logger.info("strategy.json: minimal contract for training_mode=%s", mode)
        return {
            "contract_version": CONTRACT_VERSION,
            "engine_version": ENGINE_VERSION,
            "strategy_id": strategy_id,
            "training_mode": mode,
            "supported_by_runtime": False,
            "instrument": config.instrument,
            "tick_size": config.tick_size,
            "feature_set": {"names": feature_names, "order_is_contractual": True},
        }

    du = config.dashboard_utility

    interaction = [f for f in feature_names if f in LIVE_INTERACTION_FEATURES]
    approach = [f for f in feature_names if f in LIVE_APPROACH_FEATURES]

    # class_map keyed by stringified index for stable JSON (0/1/2 -> label),
    # single-sourced from the engine's CLASS_NAMES.
    class_map = {str(idx): name for idx, name in sorted(k.CLASS_NAMES.items())}

    return {
        "contract_version": CONTRACT_VERSION,
        "engine_version": ENGINE_VERSION,
        "strategy_id": strategy_id,
        "training_mode": mode,
        "supported_by_runtime": True,
        "instrument": config.instrument,
        "tick_size": config.tick_size,
        "point_value": k.POINT_VALUE.get(config.instrument),
        "model": {
            "type": config.model.model_type,
            "loss_function": config.model.loss_function,
            "file": "model.cbm",
        },
        "feature_set": {
            "names": feature_names,
            "order_is_contractual": True,
            "interaction_features": interaction,
            "approach_features": approach,
            # Missing feature values are passed through as NaN; the loaded
            # CatBoost model applies its trained nan_mode. (engine NAN_POLICY)
            "nan_policy": k.NAN_POLICY,
        },
        "class_map": class_map,
        "session_scheme": {
            "timezone": k.SESSION_TIMEZONE,
            "trading_day_boundary": _hhmm(k.TRADING_DAY_BOUNDARY),
            "sessions": {
                "asia": _session_block("asia"),
                "london": _session_block("london"),
                "ny": _session_block("ny"),
            },
        },
        "level_scheme": {
            "pdh_pdl_source": k.PDH_PDL_SOURCE,
            "session_levels": list(k.SESSION_LEVELS),
            "available_from_guard": k.LEVEL_AVAILABLE_FROM_GUARD,
        },
        "touch_rule": {
            "type": k.TOUCH_TYPE,
            "bar_type": du.bar_type,
            "zone_proximity_pts": k.ZONE_PROXIMITY_PTS,
            "zone_representative_price": k.ZONE_REPRESENTATIVE_PRICE,
            "scope": k.TOUCH_SCOPE,
            # low touch -> long, high touch -> short, single-sourced from the
            # engine's DIRECTION_FROM_SIDE (Side enum -> Direction enum).
            "direction_from_side": {
                side.value.lower(): direction.value.lower()
                for side, direction in k.DIRECTION_FROM_SIDE.items()
            },
        },
        "feature_windows": {
            "interaction_window_minutes": du.interaction_window_minutes,
            "approach_window_minutes": du.approach_window_minutes,
            "within_band_pts": k.WITHIN_BAND_PTS,
            "level_proximity_pts": du.level_proximity_pts,
            "large_trade_threshold": k.LARGE_TRADE_THRESHOLD,
            # Engine-single-sourced. The engine standardizes the 3 interaction
            # features on the TRADE PRINT price ("trade_price"), which DIVERGES
            # from the legacy book-mid training path; the engine_version binding
            # above is what makes Trade-Lab fail-close on a model trained under the
            # old book-mid feature definition (a retrain is required, next phase).
            "mid_price_source": k.MID_PRICE_SOURCE,
        },
        "label_policy": {
            "resolution": k.LABEL_RESOLUTION,
            # Engine-single-sourced honest-entry re-anchor (engine v2): the label is
            # measured from the realistic price at the DECISION INSTANT
            # (touch + decision_offset_minutes), matching the Trade-Lab executor.
            "entry_reference": k.LABEL_ENTRY_REFERENCE,
            "decision_offset_minutes": k.DECISION_OFFSET_MINUTES,
            "tp_points": du.tp_points,
            "sl_points": du.sl_points,
            "trap_mfe_min": du.trap_mfe_min,
            "forward_bar_type": du.bar_type,
            "forward_cutoff": k.LABEL_FORWARD_CUTOFF,
            "no_resolution_dropped": k.LABEL_NO_RESOLUTION_DROPPED,
        },
        "inference": {
            "eligible_class": k.TRADEABLE_REVERSAL,
            "eligible_session": k.INFERENCE_ELIGIBLE_SESSION,
            "confidence_gate": k.DEFAULT_CONFIDENCE_GATE,
        },
        "data_requirements": {
            "min_book_level": k.MIN_BOOK_LEVEL,
            "live_schemas": list(k.LIVE_SCHEMAS),
            "replay_schemas": list(k.REPLAY_SCHEMAS),
            "depth_usage": k.DEPTH_USAGE,
        },
        "provenance": {
            "dataset_config_hash": config.dataset_config_hash(),
            "catboost": {
                "iterations": config.model.iterations,
                "depth": config.model.depth,
                "learning_rate": config.model.learning_rate,
                "auto_class_weights": config.model.auto_class_weights,
            },
        },
    }
