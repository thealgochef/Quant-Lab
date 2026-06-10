"""Phase-5 Part-2 NO-DRIFT test: the contract emitter cannot silently re-grow literals.

This is the real closure of the contract drift surface. It proves:

1. Every STRUCTURAL field of the emitted ``strategy.json`` equals its
   ``strategy_core.constants`` value (the one place the engine computes with it),
   so the description (the emitter) and the mechanism (the engine) cannot drift.
2. A COVERAGE GUARD that enumerates *every leaf field* of the ``StrategyContract``
   pydantic model and asserts each is covered by EITHER the structural constant-map
   OR an explicit per-run config allow-list. A NEW contract field added with no
   constant/allow-listed source FAILS this test -- the drift surface cannot reopen.
3. The emitted contract loads cleanly through the shared SC loader with the
   platform-version binding (fail-closed on mismatch; the engine axis renamed
   at E1) and carries the registry-resolved strategy axis (E2).

No data dependency: the emitter is pure config -> dict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from strategy_core import CONTRACT_VERSION, PLATFORM_VERSION
from strategy_core import constants as k
from strategy_core.contract.loader import load_strategy_contract
from strategy_core.contract.schema import StrategyContract
from strategy_core.strategies.registry import get_strategy

from alpha_lab.agents.data_infra.ml.config import MLPipelineConfig
from alpha_lab.agents.data_infra.ml.strategy_contract import build_strategy_contract

# The production-aligned feature set (3 interaction + 3 runtime approach, RFECV order).
MODEL_FEATURES = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
    "app_avg_trade_size",
    "app_large_trade_vol_pct",
    "app_max_spread",
]


@pytest.fixture
def config() -> MLPipelineConfig:
    """Representative production config: dashboard utility, NQ, 147t, 5m/30m."""
    return MLPipelineConfig(
        training_mode="dashboard_utility",
        instrument="NQ",
        tick_size=0.25,
        dashboard_utility=dict(
            bar_type="147t",
            interaction_window_minutes=5,
            approach_window_minutes=30,
            include_approach_features=True,
            tp_points=15.0,
            sl_points=30.0,
            trap_mfe_min=5.0,
            level_proximity_pts=0.5,
        ),
    )


@pytest.fixture
def contract(config: MLPipelineConfig) -> dict:
    # E2: strategy_id is the REGISTRY ROUTER id; an unregistered id fail-closes.
    return build_strategy_contract(config, MODEL_FEATURES, strategy_id="touch_reversal")


# ─────────────────────────────────────────────────────────────────────────────
# (1)+(2) The structural constant-map: leaf dotted-path -> the engine constant
# value it MUST equal. This is the single registry the coverage guard cross-checks
# against the pydantic model, so every structural leaf is pinned to a constant.
# ─────────────────────────────────────────────────────────────────────────────
def _sessions_projection() -> dict[str, dict]:
    """The sessions block as the emitter builds it from RESEARCH_SESSION_SCHEME."""
    out: dict[str, dict] = {}
    for name in ("asia", "london", "ny"):
        w = k.RESEARCH_SESSION_SCHEME.sessions[name]
        block = {"start": w.start.strftime("%H:%M"), "end": w.end.strftime("%H:%M")}
        if w.crosses_midnight:
            block["crosses_midnight"] = True
        out[name] = block
    return out


def _structural_constant_map(config: MLPipelineConfig) -> dict[str, Any]:
    class_map = {str(idx): name for idx, name in sorted(k.CLASS_NAMES.items())}
    direction_from_side = {
        side.value.lower(): direction.value.lower()
        for side, direction in k.DIRECTION_FROM_SIDE.items()
    }
    return {
        # top-level version stamps: the two-axis binding (decision 9.3). The
        # platform axis is the package constant; the strategy axis is
        # single-sourced from the REGISTERED plugin (registry-resolved), the one
        # place that version is declared.
        "contract_version": CONTRACT_VERSION,
        "platform_version": PLATFORM_VERSION,
        "strategy_version": get_strategy("touch_reversal").strategy_version,
        # point_value is the engine table projected onto the run's instrument
        "point_value": k.POINT_VALUE[config.instrument],
        # feature_set structural descriptor
        "feature_set.nan_policy": k.NAN_POLICY,
        # class map
        "class_map": class_map,
        # session scheme (windows projected from RESEARCH_SESSION_SCHEME). The
        # schema types ``sessions`` as a single dict leaf, so it is pinned whole
        # to the projection of the engine scheme's named windows.
        "session_scheme.timezone": k.SESSION_TIMEZONE,
        "session_scheme.trading_day_boundary": k.TRADING_DAY_BOUNDARY.strftime("%H:%M"),
        "session_scheme.sessions": _sessions_projection(),
        # level scheme
        "level_scheme.pdh_pdl_source": k.PDH_PDL_SOURCE,
        "level_scheme.session_levels": list(k.SESSION_LEVELS),
        "level_scheme.available_from_guard": k.LEVEL_AVAILABLE_FROM_GUARD,
        # touch rule
        "touch_rule.type": k.TOUCH_TYPE,
        "touch_rule.zone_proximity_pts": k.ZONE_PROXIMITY_PTS,
        "touch_rule.zone_representative_price": k.ZONE_REPRESENTATIVE_PRICE,
        "touch_rule.scope": k.TOUCH_SCOPE,
        "touch_rule.direction_from_side": direction_from_side,
        # feature windows
        "feature_windows.within_band_pts": k.WITHIN_BAND_PTS,
        "feature_windows.large_trade_threshold": k.LARGE_TRADE_THRESHOLD,
        "feature_windows.mid_price_source": k.MID_PRICE_SOURCE,
        # label policy (engine v2/v3: structural semantics are constant-sourced;
        # decision_offset_minutes is intentionally per-run config because the
        # engine uses dashboard_utility.interaction_window_minutes as the offset.)
        "label_policy.resolution": k.LABEL_RESOLUTION,
        "label_policy.entry_reference": k.LABEL_ENTRY_REFERENCE,
        "label_policy.forward_cutoff": k.LABEL_FORWARD_CUTOFF,
        "label_policy.no_resolution_dropped": k.LABEL_NO_RESOLUTION_DROPPED,
        # inference
        "inference.eligible_class": k.TRADEABLE_REVERSAL,
        "inference.eligible_session": k.INFERENCE_ELIGIBLE_SESSION,
        "inference.confidence_gate": k.DEFAULT_CONFIDENCE_GATE,
        # data requirements
        "data_requirements.min_book_level": k.MIN_BOOK_LEVEL,
        "data_requirements.live_schemas": list(k.LIVE_SCHEMAS),
        "data_requirements.replay_schemas": list(k.REPLAY_SCHEMAS),
        "data_requirements.depth_usage": k.DEPTH_USAGE,
    }


# Per-training-run inputs that LEGITIMATELY come from config, not from an engine
# constant. These are the strategy's tunable inputs (and the bundle identity),
# allow-listed so the coverage guard does not demand a constant for them.
_CONFIG_INPUT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "strategy_id",
        "training_mode",
        "supported_by_runtime",
        "instrument",
        "tick_size",
        "model.type",
        "model.loss_function",
        "model.file",
        "feature_set.names",
        "feature_set.order_is_contractual",
        "feature_set.interaction_features",
        "feature_set.approach_features",
        "touch_rule.bar_type",
        "feature_windows.interaction_window_minutes",
        "feature_windows.approach_window_minutes",
        "feature_windows.level_proximity_pts",
        "label_policy.tp_points",
        "label_policy.sl_points",
        "label_policy.trap_mfe_min",
        "label_policy.decision_offset_minutes",
        "label_policy.forward_bar_type",
        "provenance.dataset_config_hash",
        "provenance.catboost",
        "research_session_experiment",
    }
)


def _model_leaf_paths(model_cls: type, prefix: str = "") -> set[str]:
    """Enumerate every leaf dotted-path of a pydantic contract model.

    A field whose annotation is itself a nested ``_ContractModel`` recurses; any
    other field (str/float/bool/tuple/dict) is a leaf. ``ClassMap`` is treated as a
    single leaf (``class_map``) because its inner ``mapping`` IS the class map dict.
    """
    from strategy_core.contract.schema import ClassMap

    leaves: set[str] = set()
    for name, field in model_cls.model_fields.items():
        dotted = f"{prefix}{name}"
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, ClassMap):
            leaves.add(dotted)  # class_map is one leaf
            continue
        if isinstance(ann, type) and issubclass(ann, _is_contract_section()):
            leaves |= _model_leaf_paths(ann, prefix=f"{dotted}.")
        else:
            leaves.add(dotted)
    return leaves


def _is_contract_section():
    from strategy_core.contract.schema import _ContractModel

    return _ContractModel


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
def test_platform_version_is_stamped(contract: dict) -> None:
    assert contract["platform_version"] == PLATFORM_VERSION
    assert contract["contract_version"] == CONTRACT_VERSION


def test_mid_price_source_is_trade_price(contract: dict) -> None:
    # The ratified divergence from the legacy book-mid path: now describes the engine.
    assert contract["feature_windows"]["mid_price_source"] == "trade_price"
    assert contract["feature_windows"]["mid_price_source"] == k.MID_PRICE_SOURCE


def test_platform_version_is_v1_and_strategy_axis_is_registry_sourced(contract: dict) -> None:
    # E1: the platform axis renames engine v3; E2: the strategy axis comes from
    # the registered plugin, and the flag flips servable.
    assert contract["platform_version"] == "strategy_core_platform_v1"
    assert contract["platform_version"] == PLATFORM_VERSION
    assert contract["strategy_id"] == "touch_reversal"
    assert contract["strategy_version"] == get_strategy("touch_reversal").strategy_version == "1"
    assert contract["supported_by_runtime"] is True


def test_unknown_strategy_id_fails_emission_closed(config: MLPipelineConfig) -> None:
    from strategy_core.contract.schema import ContractError

    with pytest.raises(ContractError, match="unknown strategy_id"):
        build_strategy_contract(config, MODEL_FEATURES, strategy_id="not_a_registered_id")


def test_label_policy_honest_entry_reanchor_tracks_config_window(
    contract: dict, config: MLPipelineConfig
) -> None:
    """Engine v2/v3 honest-entry re-anchor: entry_reference constant, offset config-sourced.

    ``entry_reference`` is a structural engine semantic. ``decision_offset_minutes``
    is a per-run field because engine_decision uses the configured interaction
    window as the offset that starts the forward label window.
    """
    label = contract["label_policy"]
    assert label["entry_reference"] == k.LABEL_ENTRY_REFERENCE == "realistic_at_decision"
    assert label["decision_offset_minutes"] == config.dashboard_utility.interaction_window_minutes
    assert (
        label["decision_offset_minutes"]
        == contract["feature_windows"]["interaction_window_minutes"]
    )
    # The representative production fixture uses the Strategy-Core default offset/window.
    assert label["decision_offset_minutes"] == k.DECISION_OFFSET_MINUTES
    assert (
        config.dashboard_utility.interaction_window_minutes == k.DEFAULT_INTERACTION_WINDOW_MINUTES
    )
    # tp/sl/trap and the MAE-first ladder are UNCHANGED by the re-anchor.
    assert label["resolution"] == k.LABEL_RESOLUTION == "mae_first"


def _resolve(contract: dict, dotted: str) -> Any:
    """Resolve a dotted path against the contract dict, mapping class_map specially."""
    if dotted == "class_map":
        return contract["class_map"]
    node: Any = contract
    for part in dotted.split("."):
        node = node[part]
    return node


def test_every_structural_field_equals_its_engine_constant(
    contract: dict, config: MLPipelineConfig
) -> None:
    """Each structural leaf in the contract equals its strategy_core constant."""
    for dotted, expected in _structural_constant_map(config).items():
        actual = _resolve(contract, dotted)
        assert actual == expected, f"{dotted}: contract={actual!r} != constant={expected!r}"


def test_coverage_guard_no_structural_field_without_a_source(
    contract: dict, config: MLPipelineConfig
) -> None:
    """COVERAGE GUARD: every leaf of the StrategyContract pydantic model is covered.

    Each leaf must be sourced by EITHER the structural constant-map OR the explicit
    per-run config allow-list. A new contract field with no constant/allow-listed
    source has NO home here and trips this assertion -- so the drift surface that
    Phase 5 just closed cannot silently reopen.
    """
    model_leaves = _model_leaf_paths(StrategyContract)
    structural = set(_structural_constant_map(config).keys())
    covered = structural | _CONFIG_INPUT_ALLOWLIST

    uncovered = model_leaves - covered
    assert not uncovered, (
        "StrategyContract leaf field(s) with NO constant or allow-listed config source "
        f"(drift surface reopened): {sorted(uncovered)}"
    )

    # And nothing in the structural map is stale (points at a field the model dropped).
    stale_structural = structural - model_leaves
    assert not stale_structural, (
        f"structural constant-map references field(s) not on the model: {sorted(stale_structural)}"
    )
    # The allow-list must also only name real leaves.
    stale_allow = _CONFIG_INPUT_ALLOWLIST - model_leaves
    assert not stale_allow, (
        f"config allow-list references field(s) not on the model: {sorted(stale_allow)}"
    )

    # Structural map and config allow-list must be disjoint (a field is one or the other).
    overlap = structural & _CONFIG_INPUT_ALLOWLIST
    assert not overlap, (
        f"field(s) claimed by BOTH structural-map and config allow-list: {sorted(overlap)}"
    )


def test_contract_loads_through_engine_loader_with_version_binding(
    contract: dict, tmp_path: Path
) -> None:
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(contract, default=str), encoding="utf-8")

    loaded = load_strategy_contract(path, expected_platform_version=PLATFORM_VERSION)

    assert loaded.platform_version == PLATFORM_VERSION
    assert loaded.strategy_version == "1"
    assert loaded.contract_version == CONTRACT_VERSION
    assert loaded.feature_count == 6
    assert loaded.point_value == k.POINT_VALUE["NQ"]
    assert loaded.feature_windows.mid_price_source == k.MID_PRICE_SOURCE
    assert loaded.class_map.labels == (
        k.CLASS_NAMES[0],
        k.CLASS_NAMES[1],
        k.CLASS_NAMES[2],
    )


def test_loader_fails_closed_on_platform_version_mismatch(contract: dict, tmp_path: Path) -> None:
    from strategy_core.contract.schema import ContractError

    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(contract, default=str), encoding="utf-8")

    with pytest.raises(ContractError):
        load_strategy_contract(path, expected_platform_version="strategy_core_platform_v999")


def test_platform_v1_bundle_loads_against_v1_and_rejects_old_engine_axis(
    contract: dict, config: MLPipelineConfig, tmp_path: Path
) -> None:
    """The E1 axis rename is a real fail-close boundary.

    A platform_v1 bundle LOADS against platform_v1 and is REJECTED against the
    retired engine-axis literal (a runtime still expecting the old axis value
    refuses it), exactly as the old v2->v3 engine bump was a boundary.
    """
    from strategy_core.contract.schema import ContractError

    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(contract, default=str), encoding="utf-8")

    # Loads against platform v1 (== PLATFORM_VERSION).
    loaded = load_strategy_contract(path, expected_platform_version="strategy_core_platform_v1")
    assert loaded.platform_version == "strategy_core_platform_v1" == PLATFORM_VERSION
    assert loaded.label_policy.entry_reference == k.LABEL_ENTRY_REFERENCE
    assert (
        loaded.label_policy.decision_offset_minutes
        == config.dashboard_utility.interaction_window_minutes
    )
    assert loaded.level_scheme.pdh_pdl_source == k.PDH_PDL_SOURCE == "prior_day_full"
    assert loaded.inference.eligible_session == k.INFERENCE_ELIGIBLE_SESSION == "ny"

    # Rejected against the retired engine-axis literal (the rename is fail-closed).
    with pytest.raises(ContractError):
        load_strategy_contract(path, expected_platform_version="strategy_core_engine_v3")
