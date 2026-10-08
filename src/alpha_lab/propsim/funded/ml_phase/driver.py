"""Core lifecycle adapter for entry scoring and causal geometry capture."""

from __future__ import annotations

from dataclasses import replace

from alpha_lab.propsim.funded.clock import to_ns
from alpha_lab.propsim.funded.mffu_batch_driver import MffuCoreDriver

from .features import structural_snapshot


class MlCoreDriver(MffuCoreDriver):
    def __init__(self, *args, entry_scorer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.entry_scorer = entry_scorer
        self.decisions = []
        self.entries = {}
        self.completed_bars = []
        self._candidate_records = {}
        self._ml_refused_bar = False

    def begin_day(self, bars_by_tf, levels_for):
        bars = super().begin_day(bars_by_tf, levels_for)
        self.completed_bars = []
        self.reducer._research_entry_decision = self._score_candidate
        return bars

    def _score_candidate(self, *, setup, geometry, bar, candidate_id):
        structure = structural_snapshot(setup, geometry, tick_size=self.tick_size)
        if self.section.opposing_distance_policy == "fixed_v1":
            structure["frozen_max_distance_points"] = (
                self.section.opposing_parent_distance_ticks_max * self.tick_size
            )
        record = {
            "candidate_id": candidate_id,
            "setup_id": setup.setup_id,
            "decision_ns": to_ns(bar.availability_ts_utc),
            "trading_day": bar.trading_day.isoformat(),
            "structure": structure,
        }
        # Second executable family on a rejection candle must not resurrect a setup.
        if self._ml_refused_bar:
            raise AssertionError("another executable candidate after phase refusal")
        decision = (
            {"action": "baseline", "score": None, "reason": "model_disabled"}
            if self.entry_scorer is None
            else self.entry_scorer(record, self)
        )
        record["prediction"] = decision
        self.decisions.append(record)
        self._candidate_records[candidate_id] = record
        if decision["action"] == "change":
            self._ml_refused_bar = True
            return "ml_entry_negative_return"
        return None

    def step(self, bar, gate, *, target_decision_override=None):
        self._ml_refused_bar = False
        self.completed_bars.append(bar)
        out = super().step(bar, gate, target_decision_override=target_decision_override)
        if out.entry is not None:
            candidate_id = self.reducer._setup.candidate_id
            record = self._candidate_records[candidate_id]
            record["trade_id"] = out.entry.trade_id
            self.entries[out.entry.trade_id] = record
        if self._ml_refused_bar:
            if out.entry is not None:
                raise AssertionError("phase rejection still opened a position")
            out = replace(out, refused=(*out.refused, "ml_entry_negative_return"))
        return out

    def checkpoint(self):
        return {**super().checkpoint(), "ml_decisions": self.decisions, "ml_entries": self.entries}

    def restore(self, state):
        super().restore(state)
        self.decisions = state["ml_decisions"]
        self.entries = state["ml_entries"]
        self._candidate_records = {r["candidate_id"]: r for r in self.decisions}
