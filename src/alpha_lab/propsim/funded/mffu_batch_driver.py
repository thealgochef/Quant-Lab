"""Independent Core stream with causal IFSM policy evidence for the MFFU batch."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import date

from alpha_lab.propsim.funded.clock import from_ns
from alpha_lab.propsim.funded.full_range_batch import _plain
from alpha_lab.propsim.funded.strategy_driver import CoreStrategyDriver

_POLICY_REFUSAL_REASONS = frozenset({
    "ifsm_early_positive", "ifsm_positive_london",
    "ifsm_overhead_gex", "daily_execution_cap",
})


class MffuCoreDriver(CoreStrategyDriver):
    """A stream-local reducer; the shared EOD index contains immutable inputs."""

    def __init__(self, section, *, tick_size: float, context_index,
                 quantity_policy: str = "Q10") -> None:
        from strategy_core.strategies.ifvg_smc.ifsm_policy_context import IfsmPolicyContext

        def context_for(ts_utc):
            return IfsmPolicyContext(**context_index.policy_context_fields(ts_utc))

        super().__init__(
            section, tick_size=tick_size, decision_context_for=context_for,
        )
        self.context_index = context_index
        if quantity_policy not in {"Q10", "Q6", "QG"}:
            raise ValueError("unsupported MFFU quantity policy")
        self.quantity_policy = quantity_policy
        self.entry_contexts: dict[str, dict] = {}
        self.entry_quantities: dict[str, int] = {}
        self.core_exit_annotations: dict[str, dict] = {}
        self.policy_decisions: list[dict] = []
        self.day_entries = 0
        self.day_refused = 0
        self.day_policy_reasons: Counter[str] = Counter()
        self.day_open_position = False
        self.last_day_evidence: dict = {}

    def begin_day(self, bars_by_tf, levels_for):
        self.day_entries = self.day_refused = 0
        self.day_policy_reasons.clear()
        self.day_open_position = False
        return super().begin_day(bars_by_tf, levels_for)

    def target_override_from(self, decision: dict):
        """Bridge the funded print's first target observation into Core's slot."""
        from strategy_core.strategies.ifvg_smc.ifsm_policy_context import (
            IfsmPolicyContext,
            IfsmTargetOverride,
            target_action,
        )

        ts = from_ns(int(decision["decision_ns"]))
        context = IfsmPolicyContext(**self.context_index.policy_context_fields(ts))
        observed = decision.get("context")
        if (not isinstance(observed, dict)
                or observed.get("asof") != self.context_index.snapshot(ts)
                or observed.get("receipt") != context.receipt()):
            raise ValueError("funded target context differs from the same as-of Core source")
        action = target_action(self.section.exit_policy, context)
        if action != decision["action"]:
            raise ValueError("funded target decision differs from the same as-of Core context")
        return IfsmTargetOverride(action=action, context=context)

    def step(self, bar, gate, *, target_decision_override=None):
        from strategy_core.decisions.sessions import classify_session
        from strategy_core.strategies.ifvg_smc.replay import _runtime_scheme

        out = super().step(bar, gate, target_decision_override=target_decision_override)
        self.day_refused += len(out.refused)
        self.day_open_position = self.day_open_position or self.in_position
        for emission in out.emissions:
            if emission.kind != "ifsm_policy_decision":
                continue
            decision = _plain(asdict(emission.record))
            decision["stream_trading_day"] = bar.trading_day.isoformat()
            self.policy_decisions.append(decision)
            if decision["event"] == "entry_admission" and decision["action"] == "reject":
                self.day_policy_reasons.update(
                    reason for reason in decision["reasons"]
                    if reason in _POLICY_REFUSAL_REASONS
                )
        for record in out.core_exits:
            self.core_exit_annotations[str(record.trade_id)] = {
                "scale_out_ts_utc": (
                    record.scale_out_ts_utc.isoformat()
                    if getattr(record, "scale_out_ts_utc", None) else None
                ),
                "scale_out_fraction": getattr(record, "scale_out_fraction", None),
                "exit_policy": getattr(record, "exit_policy", self.section.exit_policy),
            }
        if out.entry is not None:
            self.day_entries += 1
            snapshot = self.context_index.snapshot(bar.availability_ts_utc)
            gamma = snapshot.get("gamma") or {}
            levels = snapshot.get("levels") or {}
            quantity = 6 if self.quantity_policy == "Q6" or (
                self.quantity_policy == "QG" and gamma.get("sign") == "positive"
            ) else 10
            self.entry_quantities[out.entry.trade_id] = quantity
            entry_session = classify_session(
                bar.availability_ts_utc, _runtime_scheme(self.section.session_scheme)
            ).session
            self.entry_contexts[out.entry.trade_id] = {
                "policy_id": snapshot["policy_id"],
                "decision_time_utc": snapshot["decision_time_utc"],
                "historical_publication_verified": False,
                "gamma": gamma,
                "levels": {key: value for key, value in levels.items() if key != "items"},
                "entry_session": entry_session,
                "regime": gamma.get("sign", "unknown"),
                "context_available": gamma.get("status") == "selected",
                "gate_status": "asof_available" if gamma.get("status") == "selected"
                else "asof_unknown",
                "source_eod_date": gamma.get("report_date"),
                "regime_source_eod_date": gamma.get("report_date"),
                "levels_source_eod_date": levels.get("report_date"),
                "context_asof_ts": snapshot["decision_time_utc"],
                "availability_ts_utc": snapshot["decision_time_utc"],
                "availability_window_status": gamma.get("status", "unknown"),
                "quantity_policy": self.quantity_policy,
                "selected_quantity": quantity,
            }
        return out

    def end_day(self, trading_day: date, *, dataset_exhausted: bool = False):
        self.last_day_evidence = {
            "entries": self.day_entries,
            "refused_entries": self.day_refused,
            "open_position_status": (
                "observed_open" if self.day_open_position else "flat_all_day"
            ),
            "open_position_at_day_end": self.in_position,
            "stage_counters": self._orch._reducer.funnel_counters(),
            **{f"entries_refused_{key}": value for key, value in self.day_policy_reasons.items()},
        }
        records = super().end_day(trading_day, dataset_exhausted=dataset_exhausted)
        for record in records:
            self.core_exit_annotations[str(record.trade_id)] = {
                "scale_out_ts_utc": (
                    record.scale_out_ts_utc.isoformat()
                    if getattr(record, "scale_out_ts_utc", None) else None
                ),
                "scale_out_fraction": getattr(record, "scale_out_fraction", None),
                "exit_policy": getattr(record, "exit_policy", self.section.exit_policy),
            }
        return records

    def checkpoint(self):
        return {
            **super().checkpoint(),
            "entry_contexts": self.entry_contexts,
            "entry_quantities": self.entry_quantities,
            "core_exit_annotations": self.core_exit_annotations,
            "policy_decisions": self.policy_decisions,
            "breakeven_cleared": getattr(self, "breakeven_cleared", 0),
        }

    def restore(self, state):
        super().restore(state)
        self.entry_contexts = state["entry_contexts"]
        self.entry_quantities = state["entry_quantities"]
        self.core_exit_annotations = state["core_exit_annotations"]
        self.policy_decisions = state["policy_decisions"]
        self.breakeven_cleared = state.get("breakeven_cleared", 0)
