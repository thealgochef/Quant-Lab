"""
Strategy contract emission for Strategy-Core v3 research/runtime parity.

A trained model bundle (model.cbm + metadata.json + evaluation.json) describes
*what was trained*. It does NOT, on its own, fully describe the *strategy
semantics* a runtime needs to reproduce features and labels live: the session
scheme, touch rule, level scheme, feature windows, and label policy are only
implicit in how the dashboard-utility builder computed the dataset.

`strategy.json` makes those semantics explicit and versioned so a downstream
runtime can be driven by the contract instead of hardcoding one strategy.

Contract version + platform version are imported from ``strategy_core`` (not
restated here). ``strategy_id`` is the REGISTRY ROUTER KEY (E2): it must resolve
via ``strategy_core.strategies.registry.get_strategy``, which also supplies the
plugin's ``strategy_version`` — an unknown id fail-closes EMISSION. Bundle
identity remains the output directory name (passed separately by the caller).

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

Contract v3 (E3 — the envelope/section split): the emitter builds the
plugin-consumed ``section`` subtree FROM a configured instance of the plugin's
own ``SectionModel`` (``TouchReversalSection``): the plugin's canonical
``default_touch_reversal_section()`` supplies every structural value (already
constants-sourced inside Strategy-Core) and ONLY the per-run config values are
overridden onto it — so the section the runtime validates is the very model the
plugin owns, not a restated dict. The envelope keeps the platform-consumed flat
keys. The feature-partition cross-check (section partition == envelope
``feature_set.names``) runs HERE at emission (the first of the two validation
sites; TL activation is the second).
"""

from __future__ import annotations

import logging

# The explicit registration import (touch_reversal/__init__ registers the plugin)
# so get_strategy can route; the registry is intentionally empty on a bare
# `import strategy_core`.
import strategy_core.strategies.touch_reversal  # noqa: F401
from strategy_core import CONTRACT_VERSION, PLATFORM_VERSION
from strategy_core import constants as k
from strategy_core.strategies.registry import get_strategy
from strategy_core.strategies.touch_reversal.section import (
    TouchReversalSection,
    default_touch_reversal_section,
    validate_feature_partition,
)

from alpha_lab.agents.data_infra.ml.config import (
    LIVE_APPROACH_FEATURES,
    LIVE_INTERACTION_FEATURES,
    MLPipelineConfig,
)

logger = logging.getLogger(__name__)


def _build_touch_reversal_section(
    config: MLPipelineConfig,
    interaction: list[str],
    approach: list[str],
) -> TouchReversalSection:
    """Build the configured ``TouchReversalSection`` INSTANCE for this run (E3).

    Starts from the plugin's canonical ``default_touch_reversal_section()`` —
    every structural value is already single-sourced from ``strategy_core``
    inside that builder — and overrides ONLY the per-run config values
    (``du.bar_type``, the two windows, ``level_proximity_pts``, the selected
    feature partition, and the run's session-experiment scope).
    ``direction_from_side`` is sourced from the plugin section VERBATIM
    (ratified §3, honored in W1 P4c: the plugin owns the lowercase wire
    vocabulary; the platform-constants re-derivation is gone).
    """
    du = config.dashboard_utility
    base = default_touch_reversal_section()
    return TouchReversalSection(
        session_scheme=base.session_scheme,
        level_scheme=base.level_scheme,
        touch_rule=base.touch_rule.model_copy(
            update={
                "bar_type": du.bar_type,
            }
        ),
        feature_windows=base.feature_windows.model_copy(
            update={
                "interaction_window_minutes": du.interaction_window_minutes,
                "approach_window_minutes": du.approach_window_minutes,
                "level_proximity_pts": du.level_proximity_pts,
            }
        ),
        interaction_features=tuple(interaction),
        approach_features=tuple(approach),
        research_session_experiment=config.session_experiment.model_dump(),
    )


def build_strategy_contract(
    config: MLPipelineConfig,
    selected_features: list[str] | None,
    *,
    strategy_id: str,
) -> dict | None:
    """Build a versioned strategy contract dict for v3 parity/runtime handoff.

    Every STRUCTURAL field is sourced from ``strategy_core.constants`` (or the
    engine package); only per-training-run scalars (tp/sl/trap, the interaction/
    approach windows, decision offset derived from the interaction window, bar_type,
    tick_size, instrument, model.*, provenance hash, strategy_id, selected feature
    names) come from ``config`` -- those are inputs, not engine semantics.

    Args:
        config: The full pipeline config used for this training run.
        selected_features: Ordered feature names the model was trained on
            (RFECV-selected subset, contractual order). Falls back to the
            full live feature set when not provided.
        strategy_id: The REGISTRY ROUTER id of the strategy this bundle runs
            (e.g. ``"touch_reversal"``) — resolved fail-closed via
            ``get_strategy``; NOT the bundle/output-dir name.

    Returns:
        A JSON-serialisable contract dict, or None if the training mode is
        unknown/unsupported.

    Raises:
        ContractError: when ``strategy_id`` is unknown to the registry (the
            emission fail-closes rather than stamping an unroutable contract).
    """
    mode = getattr(config, "training_mode", "unknown")

    # E2: resolve the router id against the registry — an unknown id raises
    # ContractError here, so an unroutable contract is never written to disk.
    strategy_version = get_strategy(strategy_id).strategy_version

    feature_names = (
        list(selected_features) if selected_features else list(LIVE_INTERACTION_FEATURES)
    )

    if mode != "dashboard_utility":
        # Minimal record only; the runtime is not expected to serve non
        # dashboard-utility strategies. Keep enough to identify the bundle.
        # (Still schema-incomplete by design — it is not loadable by the runtime.)
        logger.info("strategy.json: minimal contract for training_mode=%s", mode)
        return {
            "contract_version": CONTRACT_VERSION,
            "platform_version": PLATFORM_VERSION,
            "strategy_id": strategy_id,
            "strategy_version": strategy_version,
            "training_mode": mode,
            "supported_by_runtime": False,
            "instrument": config.instrument,
            "tick_size": config.tick_size,
            "feature_set": {"names": feature_names, "order_is_contractual": True},
        }

    du = config.dashboard_utility

    interaction = [f for f in feature_names if f in LIVE_INTERACTION_FEATURES]
    approach = [f for f in feature_names if f in LIVE_APPROACH_FEATURES]

    # E3: the section subtree is generated FROM the plugin's SectionModel instance
    # (structural values via the plugin's own default; per-run values overridden),
    # then the emission-site feature-partition cross-check runs against the
    # envelope's names — an inconsistent bundle is never written to disk.
    section = _build_touch_reversal_section(config, interaction, approach)
    validate_feature_partition(feature_names, section)

    # class_map keyed by stringified index for stable JSON (0/1/2 -> label),
    # single-sourced from the engine's CLASS_NAMES.
    class_map = {str(idx): name for idx, name in sorted(k.CLASS_NAMES.items())}

    return {
        "contract_version": CONTRACT_VERSION,
        "platform_version": PLATFORM_VERSION,
        "strategy_id": strategy_id,
        "strategy_version": strategy_version,
        "training_mode": mode,
        # The blocking condition was satisfied at the C/D windows: Trade-Lab is
        # repointed onto Strategy-Core (C1) and serves the honest resolver (D1b),
        # so a full-contract bundle is servable; TL activation REFUSES False (E2).
        "supported_by_runtime": True,
        "instrument": config.instrument,
        "tick_size": config.tick_size,
        "point_value": k.POINT_VALUE.get(config.instrument),
        "model": {
            "type": config.model.model_type,
            "loss_function": config.model.loss_function,
            "file": "model.cbm",
        },
        "class_map": class_map,
        "feature_set": {
            # The envelope SHELL (v3): the interaction/approach partition lives in
            # the section (plugin semantics), cross-checked above.
            "names": feature_names,
            "order_is_contractual": True,
            # Missing feature values are passed through as NaN; the loaded
            # CatBoost model applies its trained nan_mode. (engine NAN_POLICY)
            "nan_policy": k.NAN_POLICY,
        },
        "label_policy": {
            "resolution": k.LABEL_RESOLUTION,
            # E3: how tp/sl/trap are interpreted — sourced from the PLUGIN's own
            # label-policy declaration (LabelPolicySpec), not restated.
            "barrier_mode": get_strategy(strategy_id).label_policy().barrier_mode,
            # Engine-single-sourced honest-entry re-anchor (engine v2/v3): the label
            # is measured from the realistic price at the DECISION INSTANT. The
            # decision offset is the configured interaction window because the
            # engine cannot decide until those post-touch features are available.
            "entry_reference": k.LABEL_ENTRY_REFERENCE,
            "decision_offset_minutes": du.interaction_window_minutes,
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
        # The ONE strategy-owned subtree (wire shape: flat envelope keys + this).
        # Dumped from the typed instance built above; exclude_none drops the
        # optional closed_window when absent (the research scheme has none).
        "section": section.model_dump(mode="json", exclude_none=True),
    }
