"""Search-bridge mode bridging tests (DEV-R3-11 closure; DEV-R4-17 writers).

The R3 default (single `historical_closed_trade` mode, unlabeled specs, no
persistence) is byte-preserved by the existing suites; these rows prove the
ADDITIVE seam: mode validation fail-closed, the scenario path built from
real bar observations under a registered intrabar policy, the multi-mode
label scheme, and the store writers for the policy-set and simulation
envelopes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    load_sidecar_bytes,
)
from alpha_lab.propsim.search_bridge import make_prop_simulator
from alpha_lab.propsim.trade_path import OhlcBarPathObservation
from tests.agents.ifvg_search.conftest import (
    SYNTHETIC_DAYS,
    make_resolved_trades_frame,
)
from tests.agents.ifvg_search.pipeline_fixture import firm_simulation_specs

_TICK = 0.25


def _result():
    frame = make_resolved_trades_frame(SYNTHETIC_DAYS)
    return frame, SimpleNamespace(
        tables={RecordTable.EXECUTED_TRADE: frame},
        gross_trade_stream_hash=canonical_contract_sha256({"bridge": "modes"}),
    )


def _outcome():
    outcome = SimpleNamespace()
    outcome.core_replay_id = "a" * 64
    return outcome


def _observations_for(frame):
    per_trade = {}
    for row in frame.to_dict("records"):
        entry = float(row["entry_ticks"]) * _TICK
        exit_price = entry + float(row["realized_ticks"]) * _TICK
        low = min(entry, exit_price) - _TICK
        high = max(entry, exit_price) + _TICK
        per_trade[str(row["trade_id"])] = (
            OhlcBarPathObservation(
                open_ts_utc=str(row["entry_ts_utc"]),
                close_ts_utc=str(row["resolution_ts_utc"]),
                open_price=entry,
                high_price=high,
                low_price=low,
                close_price=exit_price,
            ),
        )
    return lambda _core_id: per_trade


def _make(**kwargs):
    return make_prop_simulator(
        firm_simulation_specs(),
        tick_size=_TICK,
        costed_evaluation_id_for=lambda core_id: canonical_contract_sha256(
            {"costed": core_id}
        ),
        **kwargs,
    )


def test_mode_validation_fails_closed():
    with pytest.raises(ValueError, match="reserved for actual ordered"):
        _make(simulation_modes=("historical_ordered_event_replay",))
    with pytest.raises(ValueError, match="registered intrabar scenario policy"):
        _make(simulation_modes=("historical_1m_scenario",))
    with pytest.raises(ValueError, match="never fabricated"):
        _make(
            simulation_modes=("historical_1m_scenario",),
            intrabar_scenario_policy_id="bar_adverse_extreme_first_v1",
        )
    with pytest.raises(ValueError, match="day_block_bootstrap_h<N>_v1"):
        _make(simulation_modes=("day_block_bootstrap",))
    with pytest.raises(ValueError, match="at least one registered scenario id"):
        _make(simulation_modes=("stress",))
    with pytest.raises(ValueError, match="unregistered simulation modes"):
        _make(simulation_modes=("made_up_mode",))


def test_single_mode_labels_stay_r3_identical():
    frame, result = _result()
    simulator = _make()
    vectors = simulator(outcome=_outcome(), result=result)
    assert set(vectors) == {"synthetic_fixture_firm"}


def test_multi_mode_labels_and_scenario_identity():
    frame, result = _result()
    simulator = _make(
        simulation_modes=("historical_closed_trade", "historical_1m_scenario"),
        intrabar_scenario_policy_id="bar_adverse_extreme_first_v1",
        bar_observations_for=_observations_for(frame),
    )
    vectors = simulator(outcome=_outcome(), result=result)
    assert set(vectors) == {
        "synthetic_fixture_firm:historical_closed_trade",
        "synthetic_fixture_firm:historical_1m_scenario",
    }


def test_scenario_mode_requires_observations_for_every_trade():
    frame, result = _result()
    simulator = _make(
        simulation_modes=("historical_1m_scenario",),
        intrabar_scenario_policy_id="bar_adverse_extreme_first_v1",
        bar_observations_for=lambda _core_id: {},
    )
    with pytest.raises(ValueError, match="every executed trade"):
        simulator(outcome=_outcome(), result=result)


def test_store_writers_persist_policy_sets_and_simulations(tmp_path):
    """DEV-R4-17: real AccountPolicySetEnvelopes + AccountSimulationEnvelopes
    (with the trader-UI sidecars) reach the immutable stores."""

    import json
    import os

    frame, result = _result()
    store_root = tmp_path / "store"
    simulator = _make(
        simulation_modes=("historical_closed_trade", "day_block_bootstrap"),
        bootstrap_protocol_id="day_block_bootstrap_h90_v1",
        n_paths=16,
        store_root=store_root,
    )
    vectors = simulator(outcome=_outcome(), result=result)
    assert len(vectors) == 2
    for spec in firm_simulation_specs():
        assert has_envelope(
            store_root,
            "account_policy_sets",
            spec.policy_set_envelope().account_policy_set_id,
        )
    simulation_ids = [
        name
        for name in os.listdir(store_root / "account_simulations")
        if len(name) == 64
    ]
    assert len(simulation_ids) == 2  # one per mode
    modes_seen = set()
    for simulation_id in simulation_ids:
        summary = json.loads(
            load_sidecar_bytes(
                store_root, "account_simulations", simulation_id, "walk_summary.json"
            )
        )
        modes_seen.add(summary["simulation_mode"])
        if summary["simulation_mode"] == "historical_closed_trade":
            events = json.loads(
                load_sidecar_bytes(
                    store_root,
                    "account_simulations",
                    simulation_id,
                    "account_events.json",
                )
            )
            assert events, "historical simulations persist their event stream"
    assert modes_seen == {"historical_closed_trade", "day_block_bootstrap"}
