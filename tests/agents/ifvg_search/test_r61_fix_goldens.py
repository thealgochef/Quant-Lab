"""R6.1-FIX golden identities (plan §3 / §3.10): the identities the fix
release must NOT move — pinned from the R6.1 HEAD ``6c0b60a`` before any
edit (``../R6.1-FIX/PRE_R6_1_FIX_BASELINE.md``). The regime fit id and the
frozen M0 CatBoost hash are pinned by their own suites
(``test_regime_service.R6_GOLDEN_FOLD0_FIT_ID``,
``test_catboost_bundle_model._M0_GOLDEN_RESOLVED_HASH``).
"""

from __future__ import annotations

from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (
    feature_block_registry_hash,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import resolve_kmeans_protocol
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    CoreStrategyReplayIdentity,
    CoreStrategyReplayPayload,
)
from alpha_lab.propsim.simulation import AccountSimulationEnvelope, AccountSimulationPayload
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import REGIME_INPUT_FEATURES

GOLDEN_B0_BUNDLE_ID = "668f6fa60afe73f63d0036952874bfdd8319c328570041c25cbdcfdf7bc428cb"
GOLDEN_CANDIDATE_PROTOCOL_ID = (
    "a9b7888ad1ff801ad343422248bdf5b1951ddff9121a972ff23ca3814598159b"
)
GOLDEN_CORE_REPLAY_ID = "46b7148c6aee55c1cf62c2524a77a75f4b9b833f2ee423333a5f96ef07362249"
GOLDEN_ACCOUNT_SIMULATION_ID = (
    "96229ec06622182cf22408beb4ead9d64b8743b3729f652800a0dc07575976dc"
)
GOLDEN_FEATURE_BLOCK_REGISTRY_HASH = (
    "4ebbe7ccbb16e897b1b6b208049f1570b226ce31aa5705c625259e5f0e514d5d"
)


def test_r61_fix_golden_identities() -> None:
    assert resolve_bundle("B0_CORE").resolved_feature_bundle_id == GOLDEN_B0_BUNDLE_ID
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=GOLDEN_B0_BUNDLE_ID,
        resolved_input_features=REGIME_INPUT_FEATURES,
    )
    assert protocol.resolved_regime_protocol_id == GOLDEN_CANDIDATE_PROTOCOL_ID
    core = CoreStrategyReplayIdentity.from_payload(
        CoreStrategyReplayPayload(
            replay_input_bundle_id="1" * 64,
            quant_lab_replay_source_identity="e" * 64,
            strategy_core_commit="f" * 40,
            strategy_core_source_identity="f" * 64,
            resolved_section_config_hash="2" * 64,
            canonical_profile_id="ifvg_v2_doc_default_fresh_static_1r",
            warmup_seed_identity="synthetic_cold_start_v1",
            anchor_policy="trading_day_18et_elapsed_v1",
            capture_schema_version=2,
            record_schema_version=2,
        )
    )
    assert core.core_replay_id == GOLDEN_CORE_REPLAY_ID
    simulation = AccountSimulationEnvelope.from_payload(
        AccountSimulationPayload(
            core_replay_id="a" * 64,
            gross_trade_stream_hash="1" * 64,
            costed_evaluation_id="2" * 64,
            trade_path_bundle_id="3" * 64,
            trade_path_bundle_manifest_sha256="4" * 64,
            path_capability_report_id="5" * 64,
            account_policy_set_id="6" * 64,
            simulation_mode="historical_closed_trade",
            intrabar_scenario_policy_id=None,
            bootstrap_protocol_id=None,
            stress_scenario_id=None,
            seed=11,
            n_paths=1,
        )
    )
    assert simulation.account_simulation_id == GOLDEN_ACCOUNT_SIMULATION_ID
    assert feature_block_registry_hash() == GOLDEN_FEATURE_BLOCK_REGISTRY_HASH
