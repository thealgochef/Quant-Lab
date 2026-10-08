"""Matched synthetic account replay through the actual Core admission/exit seams.

The selected task Core's existing deterministic reducer helpers supply pattern
events. No market files or alternative financial engine are involved.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection

from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600, to_ns
from alpha_lab.propsim.funded.pair_engine import DayInput, PairRun, run_one_day
from alpha_lab.propsim.funded.pair_ledger import PairLedger
from alpha_lab.propsim.funded.position_walk import MinuteObservations
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES
from alpha_lab.propsim.funded.strategy_driver import CoreStrategyDriver

pytestmark = pytest.mark.skipif(
    "exit_policy" not in IfvgSmcSection.model_fields,
    reason="requires the explicitly selected compatible task Core",
)
SCALE = "scale_out_half_breakeven_hold_to_close_v1"
# Core trade UUIDs include the section/profile identity. Trade references must
# still join consistently within each run; the annotation changes that identity.
EXCLUDED_IDENTITY_FIELDS = {"trade_id", "strategy_trade_id", "trade_ref"}


@pytest.fixture
def core_helpers(monkeypatch):
    selected = os.environ.get("IFSM_RESEARCH_CORE")
    if not selected:
        pytest.skip("task Core helper fixtures were not explicitly selected")
    monkeypatch.syspath_prepend(str(Path(selected) / "tests"))
    return (
        importlib.import_module("test_ifvg_daily_close"),
        importlib.import_module("test_ifvg_v2_characterization"),
        importlib.import_module("test_ifvg_menthorq_levels"),
    )


def _economic(value):
    if isinstance(value, dict):
        return {key: _economic(item) for key, item in value.items()
                if key not in EXCLUDED_IDENTITY_FIELDS}
    if isinstance(value, (tuple, list)):
        return [_economic(item) for item in value]
    return value


class _Prints:
    def __init__(self, prices):
        self.prices = prices

    def minute(self, bar, _sign):
        prices = self.prices[bar.bar_index]
        opened = to_ns(bar.logical_open_ts_utc)
        closed = to_ns(bar.logical_close_ts_utc)
        return MinuteObservations(
            open_ns=opened, close_ns=closed, close_ticks=bar.close_ticks,
            ts_ns=np.arange(len(prices), dtype=np.int64) + opened + 1,
            price_ticks=np.array(prices, dtype=np.int64),
            continuous=np.zeros(len(prices), dtype=bool), fidelity="ordered_trade_prints",
        )


class _SeededCoreDriver(CoreStrategyDriver):
    """Inject known pattern events, preserving the real Core driver and reducer."""

    def __init__(self, section, reducer, script, dc, snapshot):
        super().__init__(section, tick_size=0.25)
        self.fixture_reducer = reducer
        self.script = script
        self.dc = dc
        self.snapshot = snapshot

    def begin_day(self, bars_by_tf, _levels_for):
        reducer = self.fixture_reducer
        original = type(reducer).execution_block_reasons

        def admission(execution_ts, *, session_doc):
            return original(reducer, execution_ts, session_doc=session_doc) + (
                (self._gate,) if self._gate else ())

        reducer.execution_block_reasons = admission

        def emit(bar):
            inp = self.dc.entry_input() if bar.bar_index == 5 else self.script._step(bar)
            return reducer.step(replace(inp, menthorq=self.snapshot))

        self._orch = SimpleNamespace(_reducer=reducer, on_decision_bar=emit)
        return list(bars_by_tf[60])

    def end_day(self, _day, *, dataset_exhausted=False):
        assert not self.in_position
        self.seed = self.fixture_reducer.snapshot()
        self._orch = None
        return []


def _run(monkeypatch, helpers, *, exit_policy, context, scenario, firm, enabled):
    dc, script, mq = helpers
    start = "2026-01-13T15:48" if scenario == "mandatory_close" else (
        "2026-01-13T05:48" if context == "outside" else "2026-01-13T09:54")
    changes = {"exit_policy": exit_policy}
    if enabled:
        changes.update(menthorq_context_version="menthorq_eod_v1", regime_gate_policy="off",
                       nearest_support_gex1_block=False, regime_unknown_policy="allow",
                       nearest_support_universe="all_19")
    section = dc.section(**changes)
    reducer = dc.prepare(monkeypatch, start=start, **changes)
    snapshot = None
    if enabled:
        snapshot = mq.snapshot(
            regime="negative" if context == "conflicting" else "unknown",
            context_available=context == "conflicting",
            unavailable_reason=(None if context == "conflicting" else
                                "before_0600" if context == "outside" else "no_level_row"),
        )
    entry, stop, target = 10016, 9984, 10048
    price_path = {
        "same_minute_stop": {6: [entry, target, entry], 7: [entry, entry + 1, entry]},
        "later_stop": {6: [entry, target, target], 7: [target, entry, entry]},
        "protective_exit": {6: [entry, stop - 1, entry + 1], 7: [entry, entry + 1, entry]},
        "mandatory_close": {6: [entry, target, target + 8]},
    }[scenario]
    bars = [dc.entry_input().bar_1m]
    bars.extend(script._bar(index, prices[0], max(prices), min(prices), prices[-1])
                for index, prices in price_path.items())
    from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import _calendar

    schedule, start_ns, cutoff_ns = _calendar(section, ("2026-01-13",))
    scaled = exit_policy == SCALE
    ledger = PairLedger(
        pair_id="fixture", configuration="fixture", profile=FIRM_PROFILES[firm],
        processing=TWO_BUSINESS_DAYS_FED_1600, quantity=10 if scaled else 1,
        tick_value_cents=50 if scaled else 500, cost_per_side_cents=0,
        cost_per_contract_mills=514 if scaled else 5140, scale_out=scaled,
        trading_days=schedule, start_ns=start_ns, cutoff_ns=cutoff_ns,
    )
    driver = _SeededCoreDriver(section, reducer, script, dc, snapshot)
    funded = PairRun("fixture", driver, ledger)
    run_one_day(funded, DayInput("2026-01-13", {60: bars}, None, True, schedule[0]),
                lambda: _Prints(price_path))
    ledger.finish()
    assert len(ledger.trades) == 1
    trade = ledger.trades[0]
    refs = {row["trade_ref"] for row in ledger.account_events if row.get("trade_ref")}
    assert refs == {trade["trade_ref"]}
    assert trade["quantity"] == (10 if scaled else 1)
    assert trade["costs_cents"] == 1028
    assert trade["scale_out_quantity"] == (5 if scaled and scenario != "protective_exit" else 0)
    assert trade["final_exit_quantity"] == (5 if scaled and scenario != "protective_exit"
                                             else 10 if scaled else 1)
    return ledger.snapshot(), funded.strategy_trades, driver.forced_flat, driver.discarded_setups


@pytest.mark.parametrize("exit_policy", ["fixed_target_v1", SCALE])
@pytest.mark.parametrize("context", ["missing", "outside", "conflicting"])
@pytest.mark.parametrize(
    "scenario", ["same_minute_stop", "later_stop", "protective_exit", "mandatory_close"]
)
@pytest.mark.parametrize("firm", ["takeprofittrader", "myfundedfutures"])
def test_passive_context_preserves_actual_funded_core_events(
    monkeypatch, core_helpers, exit_policy, context, scenario, firm,
):
    kwargs = dict(exit_policy=exit_policy, context=context, scenario=scenario, firm=firm)
    off = _run(monkeypatch, core_helpers, enabled=False, **kwargs)
    on = _run(monkeypatch, core_helpers, enabled=True, **kwargs)
    assert _economic(off) == _economic(on)
