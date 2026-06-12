"""Phase-5 Part-2 standing test: the contract emitter is literal-free + version-stamped.

Proves ``build_strategy_contract`` no longer RESTATES the strategy literals but
single-sources them from ``strategy_core.constants`` (the one place the engine
computes with them), carries the two-axis binding (``platform_version`` +
registry-resolved ``strategy_version``, E1/E2), and round-trips cleanly through
the shared ``strategy_core`` schema/loader with the platform-version binding.

No data dependency: the emitter is pure config -> dict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import strategy_core as sc
from strategy_core import PLATFORM_VERSION
from strategy_core.contract.loader import load_strategy_contract

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig, MLPipelineConfig
from alpha_lab.agents.data_infra.ml.strategy_contract import build_strategy_contract

MODEL_FEATURES = [
    "int_time_beyond_level",
    "int_time_within_2pts",
    "int_absorption_ratio",
    "app_avg_trade_size",
    "app_large_trade_vol_pct",
    "app_max_spread",
]


@pytest.fixture
def config():
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
def contract(config):
    # E2: strategy_id must be a REGISTERED router id (unknown ids fail emission).
    return build_strategy_contract(config, MODEL_FEATURES, strategy_id="touch_reversal")


def test_dashboard_utility_default_bar_type_is_v3_production_147t():
    """Default dashboard-utility bundles should use the Strategy-Core v3 production bar."""
    cfg = MLPipelineConfig(training_mode="dashboard_utility", instrument="NQ")
    c = build_strategy_contract(cfg, None, strategy_id="touch_reversal")

    assert c is not None
    assert cfg.dashboard_utility.bar_type == "147t"
    assert c["section"]["touch_rule"]["bar_type"] == "147t"
    assert c["label_policy"]["forward_bar_type"] == "147t"


def test_decision_offset_tracks_configured_interaction_window():
    """strategy.json must describe the same decision offset the engine used for labels."""
    cfg = MLPipelineConfig(
        training_mode="dashboard_utility",
        instrument="NQ",
        tick_size=0.25,
        dashboard_utility=DashboardUtilityConfig(
            bar_type="147t",
            interaction_window_minutes=11,
            approach_window_minutes=30,
            include_approach_features=True,
        ),
    )
    c = build_strategy_contract(cfg, MODEL_FEATURES, strategy_id="touch_reversal")

    assert c is not None
    assert c["section"]["feature_windows"]["interaction_window_minutes"] == 11
    assert c["label_policy"]["decision_offset_minutes"] == 11


def test_contract_is_platform_version_stamped(contract):
    assert contract["platform_version"] == PLATFORM_VERSION
    assert contract["strategy_id"] == "touch_reversal"
    assert contract["strategy_version"] == "1"


def test_dashboard_utility_contract_advertises_runtime_servable(contract):
    """E2: the blocking condition cleared at C/D — full bundles are servable."""
    assert contract["supported_by_runtime"] is True


def test_contract_records_research_session_experiment(contract):
    """strategy.json should audit train/eval/gate session scope for experiments.

    (v3: the research scope is plugin-section-bound — it rides the section.)
    """
    scope = contract["section"]["research_session_experiment"]
    assert scope["training_sessions"] == ["asia", "london", "ny"]
    assert scope["evaluation_sessions"] == ["asia", "london", "ny"]
    assert scope["production_gate_sessions"] == ["ny"]
    assert scope["report_session_breakdowns"] is True


def test_literals_are_single_sourced_from_engine(contract):
    """Every value the emitter used to restate now equals the engine constant.

    (v3: the section-bound values are read through the ``section`` subtree —
    sourced from the plugin's SectionModel default, itself constants-sourced.)
    """
    section = contract["section"]
    assert section["touch_rule"]["zone_proximity_pts"] == sc.constants.ZONE_PROXIMITY_PTS
    assert section["feature_windows"]["within_band_pts"] == sc.constants.WITHIN_BAND_PTS
    assert (
        section["feature_windows"]["large_trade_threshold"]
        == sc.constants.LARGE_TRADE_THRESHOLD
    )
    assert section["feature_windows"]["mid_price_source"] == sc.constants.MID_PRICE_SOURCE
    assert contract["point_value"] == sc.constants.POINT_VALUE["NQ"]
    assert section["session_scheme"]["timezone"] == sc.constants.SESSION_TIMEZONE
    assert section["session_scheme"][
        "trading_day_boundary"
    ] == sc.constants.TRADING_DAY_BOUNDARY.strftime("%H:%M")


def test_sessions_match_engine_scheme(contract):
    sessions = contract["section"]["session_scheme"]["sessions"]
    scheme = sc.constants.RESEARCH_SESSION_SCHEME.sessions
    for name in ("asia", "london", "ny"):
        assert sessions[name]["start"] == scheme[name].start.strftime("%H:%M")
        assert sessions[name]["end"] == scheme[name].end.strftime("%H:%M")
        assert sessions[name]["crosses_midnight"] == scheme[name].crosses_midnight


def test_class_map_matches_engine(contract):
    expected = {str(idx): name for idx, name in sorted(sc.constants.CLASS_NAMES.items())}
    assert contract["class_map"] == expected


def test_direction_from_side_matches_engine(contract):
    # W1 P4c (ratified §3): sourced from the plugin section verbatim — the plugin
    # owns the lowercase wire vocabulary; the platform constant is gone.
    from strategy_core.strategies.touch_reversal.section import (
        default_touch_reversal_section,
    )

    expected = default_touch_reversal_section().touch_rule.direction_from_side
    assert contract["section"]["touch_rule"]["direction_from_side"] == expected
    assert expected == {"low": "long", "high": "short"}


def test_contract_round_trips_through_engine_loader(contract, tmp_path: Path):
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(contract, default=str), encoding="utf-8")

    # v3: the section hook types the subtree against the plugin's SectionModel.
    loaded = load_strategy_contract(
        path,
        expected_platform_version=PLATFORM_VERSION,
        validate_section_via_registry=True,
    )

    assert loaded.platform_version == PLATFORM_VERSION
    assert loaded.strategy_version == "1"
    assert loaded.feature_count == 6
    assert loaded.section_model.research_session_experiment is not None
    assert loaded.section_model.research_session_experiment.production_gate_sessions == ("ny",)
    assert loaded.class_map.labels == (
        "tradeable_reversal",
        "trap_reversal",
        "aggressive_blowthrough",
    )


def test_minimal_contract_is_platform_version_stamped():
    cfg = MLPipelineConfig(training_mode="extrema_rebound_crossing", instrument="NQ")
    c = build_strategy_contract(cfg, None, strategy_id="touch_reversal")
    assert c["platform_version"] == PLATFORM_VERSION
    assert c["strategy_version"] == "1"
    # The minimal record stays non-servable by design (schema-incomplete; it
    # cannot load) — the E2 flag flip applies to the FULL contract only.
    assert c["supported_by_runtime"] is False
