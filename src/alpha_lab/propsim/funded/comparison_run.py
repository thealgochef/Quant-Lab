"""Run ONE configuration for every selected firm (one worker process each).

For one resolved configuration this runs, over the same trusted day inputs:

* the no-account REFERENCE replay, which must reproduce the saved study's
  trades exactly (equivalence of this driver with the normal replay on this
  exact path);
* one account-driven replay per selected firm, each with its own Strategy-Core
  state and its own one-account-at-a-time ledger;
* a checkpoint at the middle evaluation day and a resumed run from it, which
  must end in the identical account and strategy state.

The day's candles, levels and prints are loaded once and read by all of these
runs; nothing mutable is shared. The return value is plain JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from alpha_lab.propsim.funded.clock import ProcessingClockPolicy
from alpha_lab.propsim.funded.comparison_describe import describe_section
from alpha_lab.propsim.funded.comparison_source import (
    ComparisonConfiguration,
    ComparisonSource,
    core_trade_key,
    iter_days,
    normal_replay_trades,
    reference_trades,
    resolve_configuration,
)
from alpha_lab.propsim.funded.pair_engine import DayInput, PairRun, run_days
from alpha_lab.propsim.funded.pair_ledger import PairLedger
from alpha_lab.propsim.funded.price_evidence import DATA_ROOT
from alpha_lab.propsim.funded.print_minutes import DayPrints, NoPrints, bar_window_ns
from alpha_lab.propsim.funded.profiles import INSTRUMENTS, FundedFirmProfile
from alpha_lab.propsim.funded.strategy_driver import CoreStrategyDriver

__all__ = ["run_configuration", "pair_id_for", "PrintStats"]


def pair_id_for(configuration: str, firm_key: str) -> str:
    return f"{configuration}|{firm_key}"


class PrintStats:
    def __init__(self, data_root: Path | None) -> None:
        self.data_root = data_root
        self.files: dict[str, dict] = {}
        self.missing: set[str] = set()
        self.minutes_checked = 0
        self.minutes_matched = 0

    def factory(self, day: DayInput):
        holder: dict[str, Any] = {}

        def load():
            if "prints" not in holder:
                if self.data_root is None:
                    holder["prints"] = NoPrints()
                else:
                    bars = day.bars_by_tf.get(60, [])
                    first = bar_window_ns(bars[0])[0]
                    last = bar_window_ns(bars[-1])[1]
                    prints = DayPrints(first, last, data_root=self.data_root)
                    for item in prints.files:
                        self.files[item["file"]] = item
                    self.missing.update(prints.missing)
                    holder["prints"] = prints
            return holder["prints"]

        def release():
            prints = holder.pop("prints", None)
            if prints is not None:
                self.minutes_checked += prints.minutes_checked
                self.minutes_matched += prints.minutes_matched

        load.release = release
        return load


def _roundtrip(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def run_configuration(*, source: ComparisonSource, configuration: ComparisonConfiguration,
                      profiles: tuple[FundedFirmProfile, ...], instrument: str, quantity: int,
                      cost_per_side_cents: int, processing: ProcessingClockPolicy,
                      data_root: Path | None = DATA_ROOT, check_resume: bool = True,
                      cost_per_contract_mills: int | None = None,
                      cache_configuration: ComparisonConfiguration | None = None
                      ) -> dict[str, Any]:
    """``cache_configuration``: a verified configuration whose cached charts are a
    superset of this one's (variants outside the verified study)."""
    started = time.monotonic()
    section, cfg = resolve_configuration(configuration.axis_value_ids)
    if cfg.profile_hash != configuration.resolved_section_config_hash:
        raise PermissionError(
            f"{configuration.name}: the configurator resolves a different section than "
            "the approved one")
    tick_value = INSTRUMENTS[instrument].tick_value_cents
    scale_out = getattr(section, "exit_policy", "fixed_target_v1") != "fixed_target_v1"
    cache_cfg = None
    if cache_configuration is not None:
        _cache_section, cache_cfg = resolve_configuration(cache_configuration.axis_value_ids)
        if cache_cfg.profile_hash != cache_configuration.resolved_section_config_hash:
            raise PermissionError("the cache configuration no longer resolves to its approval")

    def new_ledger(profile: FundedFirmProfile) -> PairLedger:
        return PairLedger(
            pair_id=pair_id_for(configuration.name, profile.firm_key),
            configuration=configuration.name, profile=profile, processing=processing,
            quantity=quantity, tick_value_cents=tick_value,
            cost_per_side_cents=cost_per_side_cents, trading_days=source.trading_days,
            start_ns=source.start_ns, cutoff_ns=source.cutoff_ns,
            cost_per_contract_mills=cost_per_contract_mills, scale_out=scale_out)

    def new_driver() -> CoreStrategyDriver:
        return CoreStrategyDriver(section, tick_size=cfg.tick_size)

    reference = PairRun("reference", new_driver(), None)
    runs = [PairRun(pair_id_for(configuration.name, p.firm_key), new_driver(), new_ledger(p))
            for p in profiles]
    days = list(iter_days(source, configuration, cfg=cfg, cache_cfg=cache_cfg))
    evaluation_index = [i for i, d in enumerate(days) if d.is_evaluation]
    middle = evaluation_index[len(evaluation_index) // 2]
    checkpoints: dict[str, dict] = {}
    stats = PrintStats(data_root)

    def on_day_start(index: int, _day: DayInput) -> None:
        if index == middle:
            for run in runs:
                checkpoints[run.pair_id] = _roundtrip(run.to_state())

    run_days([reference, *runs], days, stats.factory, on_day_start=on_day_start)
    for run in runs:
        run.ledger.finish()
    counted = (stats.minutes_checked, stats.minutes_matched)

    if configuration.name in source.package.profiles:
        expected = reference_trades(source, configuration.name)
        equivalence_target = "saved study trades"
    else:
        expected = normal_replay_trades(days, section, cfg)
        equivalence_target = "Strategy-Core's normal day-by-day replay of the same days"
    mine = [core_trade_key(t) for t in reference.strategy_trades]
    first_difference = next(
        (index for index, (a, b) in enumerate(zip(mine, expected, strict=False)) if a != b),
        None if len(mine) == len(expected) else min(len(mine), len(expected)))
    reference_report = {
        "saved_study_trades": len(expected), "replayed_trades": len(mine),
        "equivalent": mine == expected, "first_difference_index": first_difference,
        "warmup_trades": reference.warmup_trades, "compared_with": equivalence_target,
    }
    reference_keys = {k[:3] for k in mine}

    resume_report: dict[str, Any] = {}
    if check_resume:
        resumed_runs = []
        for profile in profiles:
            pair_id = pair_id_for(configuration.name, profile.firm_key)
            state = checkpoints[pair_id]
            driver = new_driver()
            driver.restore(state["driver"])
            ledger = new_ledger(profile)
            ledger.restore(state["ledger"])
            resumed_runs.append(PairRun(pair_id, driver, ledger,
                                        strategy_trades=list(state["strategy_trades"]),
                                        warmup_trades=state["warmup_trades"],
                                        also_blocked=state["also_blocked"]))
        run_days(resumed_runs, days[middle:], stats.factory)
        stats.minutes_checked, stats.minutes_matched = counted  # count the straight run once
        for straight, resumed in zip(runs, resumed_runs, strict=True):
            resumed.ledger.finish()
            identical = (
                json.dumps(_roundtrip(straight.ledger.snapshot()), sort_keys=True)
                == json.dumps(_roundtrip(resumed.ledger.snapshot()), sort_keys=True)
                and straight.driver.seed_hash() == resumed.driver.seed_hash()
                and straight.strategy_trades == resumed.strategy_trades)
            resume_report[straight.ledger.profile.firm_key] = {
                "checkpoint_before_day": days[middle].trading_day, "identical": identical}

    pairs = {}
    for run in runs:
        ledger = run.ledger
        taken = [t for t in ledger.trades]
        same_as_reference = sum(
            1 for t in taken
            if (_iso_ns(t["entry_ns"]), t["entry_ticks"], t["stop_ticks"]) in reference_keys)
        pairs[ledger.profile.firm_key] = {
            "pair_id": run.pair_id,
            "entry_candidates_also_blocked_by_strategy": run.also_blocked,
            "ledger": _roundtrip(ledger.snapshot()),
            "strategy_trades": run.strategy_trades,
            "forced_flat": run.driver.forced_flat,
            "discarded_refused_setups": run.driver.discarded_setups,
            "final_seed_hash": run.driver.seed_hash(),
            "trades_matching_reference_entries": same_as_reference,
            "trades_not_in_reference": len(taken) - same_as_reference,
        }
    return {
        "configuration": configuration.name,
        "display_name": configuration.display_name,
        "axes": dict(configuration.axes),
        "settings_plain": describe_section(section),
        "section_config_hash": cfg.profile_hash,
        "sizing": {"instrument": instrument, "instrument_label": INSTRUMENTS[instrument].label,
                   "quantity": quantity, "tick_value_cents": tick_value,
                   "cost_per_contract_mills": (cost_per_contract_mills
                                               if cost_per_contract_mills is not None
                                               else cost_per_side_cents * 10)},
        "exit_policy": getattr(section, "exit_policy", "fixed_target_v1"),
        "strategy_trades_no_account": reference.strategy_trades,
        "pairs": pairs,
        "reference": reference_report,
        "resumed": resume_report,
        "prints": {"minutes_checked": stats.minutes_checked,
                   "minutes_rebuilt_exactly": stats.minutes_matched,
                   "missing_utc_days": sorted(stats.missing),
                   "files": sorted(stats.files.values(), key=lambda f: f["file"])},
        "seconds": round(time.monotonic() - started, 1),
    }


def _iso_ns(ts_ns: int) -> str:
    import pandas as pd

    return pd.Timestamp(ts_ns, unit="ns", tz="UTC").isoformat()
