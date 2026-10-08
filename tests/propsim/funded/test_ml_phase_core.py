"""Run the actual task-only Core seam in a fresh interpreter, with no import leak."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_phase_core_admission_and_inactive_family_isolation():
    repo = Path(__file__).resolve().parents[3]
    core = Path(
        os.environ.get(
            "IFSM_ML_TEST_CORE",
            str(repo.parent / "Claude-Quant-Lab-Research-Artifacts/ifsm-mffu-ml-phase-v01/core"),
        )
    )
    reducer = core / "src/strategy_core/strategies/ifvg_smc/reducer.py"
    if not reducer.is_file() and "IFSM_ML_TEST_CORE" not in os.environ:
        pytest.skip("the task-only ML Core source is not installed")
    assert reducer.is_file(), f"the selected ML Core source is unavailable: {core}"
    script = r"""
import sys
sys.path.insert(0, sys.argv[1])
from dataclasses import replace
from datetime import timedelta
from strategy_core.strategies.ifvg_smc.reducer import IfvgReducer, IfvgReducerConfig
from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section
from strategy_core.strategies.ifvg_smc.ifsm_policy_context import IfsmPolicyContext
from strategy_core.structures.fvg import GapDirection
from tests.propsim.funded import ml_core_fixture as fixture
original = fixture._step
def step(bar, **kwargs):
    context = lambda ts: IfsmPolicyContext(
        decision_ts_utc=ts, total_net_gex=-1., gamma_status="available",
        levels_status="available", gamma_eligible_from_utc=ts-timedelta(hours=1))
    return replace(original(bar, **kwargs),
        policy_context_open=context(bar.open_ts_utc),
        policy_context_close=context(bar.availability_ts_utc))
fixture._step = step
values = default_ifvg_smc_section().model_dump()
values.update(holding_policy="scheduled_daily_close_v1",
              entry_schedule_policy="all_open_market_v1",
              ifsm_context_policy_version="mq_eod_asof_nominal_2200_chicago_v01")
section = default_ifvg_smc_section().model_validate(values)
cfg = IfvgReducerConfig.from_section(section, tick_size=.25,
                                    strategy_id="ifvg_smc", strategy_version="2")
def prepared():
    reducer = IfvgReducer(cfg)
    fixture._drive_to_inversion(reducer)
    return reducer
bar = fixture._bar(5, 10015, 10019, 9985, 10016)
gap = fixture._fvg(60, GapDirection.BULLISH, 10014, 10015,
                   confirmed=bar.availability_ts_utc, ident="entry")
inp = fixture._step(bar, new_fvgs={60: (gap,)})
plain = prepared()
baseline = plain.step(inp)
calls = []
scored = prepared()
scored._research_entry_decision = lambda **kw: calls.append(kw) or None
same = scored.step(inp)
assert same == baseline
assert len(calls) == 1
assert calls[0]["geometry"].entry_ticks == 10016
rejected = prepared()
rejected._research_entry_decision = lambda **kw: "ml_entry_negative_return"
emissions = rejected.step(inp)
assert not any(e.kind == "eligible_decision" for e in emissions)
assert rejected._setup is None
assert rejected._executions_by_day.get(bar.trading_day, 0) == 0
assert any(e.kind == "entry_candidate" and e.record.block_reasons ==
           ("ml_entry_negative_return",) for e in emissions)
# The same callback must never run for a strategy-blocked opportunity.
blocked = prepared()
blocked.execution_block_reasons = lambda *a, **kw: ("account_pause",)
blocked._research_entry_decision = lambda **kw: (_ for _ in ()).throw(
    AssertionError("scored an already blocked diagnostic"))
blocked.step(inp)
assert blocked._setup is not None and blocked._setup.phase == "S4"
print("actual Core: baseline parity, negative rejection and guarded scoring passed")

from types import SimpleNamespace
import numpy as np
from alpha_lab.propsim.funded.ml_phase.stream import PhaseStream
from alpha_lab.propsim.funded.ml_phase.protocol import load_contracts
from alpha_lab.propsim.funded.pair_engine import DayInput
from alpha_lab.propsim.funded.campaign import TradingDay
from alpha_lab.propsim.funded.clock import to_ns
from alpha_lab.propsim.funded.position_walk import MinuteObservations
from pathlib import Path
values["exit_policy"] = "scale_out_half_breakeven_hold_to_close_v1"
partial_section = section.model_validate(values)
partial_cfg = IfvgReducerConfig.from_section(partial_section, tick_size=.25,
                                    strategy_id="ifvg_smc", strategy_version="2")
real = IfvgReducer(partial_cfg)
fixture._drive_to_inversion(real)
class Context:
    def policy_context_fields(self, ts):
        return dict(decision_ts_utc=ts, total_net_gex=-1., gamma_status="available",
                    levels_status="available", gamma_eligible_from_utc=ts-timedelta(hours=1))
    def snapshot(self, ts):
        return dict(policy_id="synthetic", decision_time_utc=ts.isoformat(), gamma={}, levels={})
contracts = load_contracts(Path("docs/ifsm-mffu-ml-phase-v01"))
stream = PhaseStream(reference="MCB025", stream_id="SHADOW_MCB025", section=partial_section,
                     context_index=Context(), definitions=contracts["FEATURE_DEFINITIONS"]["rows"],
                     source_id="synthetic-source")
bars = [bar, fixture._bar(6,10050,10060,10050,10060),
        fixture._bar(7,10080,10090,10080,10090), fixture._bar(8,10030,10030,10015,10015)]
def begin(by_tf, levels):
    stream.driver.completed_bars = []
    real._research_entry_decision = stream.driver._score_candidate
    stream.driver._orch = SimpleNamespace(_reducer=real,
       on_decision_bar=lambda b: real.step(fixture._step(
           b, new_fvgs={60:(gap,)} if b is bar else {})))
    return bars
stream.driver.begin_day = begin
stream.driver.end_day = lambda *a, **kw: []
class Prints:
    def minute(self, b, sign):
        prices=np.array([b.open_ticks,b.close_ticks])
        opened=to_ns(b.logical_open_ts_utc)
        return MinuteObservations(opened,to_ns(b.availability_ts_utc),b.close_ticks,
          np.array([opened+1,opened+2]),prices,np.zeros(2,dtype=bool),"ordered_trade_prints")
deadline=to_ns(bars[-1].availability_ts_utc)
day=DayInput("2026-01-13",{60:bars},None,True,
             TradingDay("2026-01-13",deadline+300000000000,deadline+3600000000000,deadline))
stream.run_day(day,lambda:Prints())
assert len(stream.datasets["ENTRY"]) == len(stream.datasets["CONTINUATION"]) == 1
entry=stream.datasets["ENTRY"][0]
continuation=stream.datasets["CONTINUATION"][0]
assert entry["label_status"] == continuation["label_status"] == "exact"
parts=entry["label_components"]
assert parts["entry_cost_cents"]+parts["partial_cost_cents"]+parts["exit_cost_cents"] == 1028
assert entry["label"] == parts["net_pnl_cents"]/parts["initial_risk_cents"]
assert continuation["label"] < 0
assert continuation["label_available_ns"] >= continuation["decision_ns"]
print("actual Core and shared stream: exact shadow labels, next-print branch and fees passed")

"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(core / "src")],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
