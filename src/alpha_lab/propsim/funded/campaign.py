"""The funded campaign: both firm instances on one shared market clock.

Each firm instance is replayed through its OWN time-ordered event queue over
the SAME strategy executions, price paths and trading schedule. Instances
share market inputs only; no money, credit or account crosses them. The run
can stop at any event boundary, be serialized to JSON, and resume to an
identical result (checkpoint equivalence is part of the run's validation).
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from alpha_lab.propsim.funded.clock import (
    ProcessingClockPolicy,
    chicago_date,
    chicago_label,
    iso_utc,
    month_grant_times,
    to_ns,
)
from alpha_lab.propsim.funded.instance import FirmInstance
from alpha_lab.propsim.funded.paths import ExecutionPath, StrategyExecution
from alpha_lab.propsim.funded.profiles import INSTRUMENTS, FundedFirmProfile

__all__ = [
    "TradingDay",
    "CampaignInputs",
    "FirmCampaign",
    "run_campaign",
    "PRIORITY",
    "SettingsError",
]

PRIORITY = {
    "position_closed": 1,
    "processing_complete": 2,
    "growth_review": 3,
    "monthly_grant": 4,
    "day_end": 5,
    "day_release": 6,
    "signal": 7,
}


class SettingsError(ValueError):
    """The requested scenario cannot run as stated (never silently changed)."""


@dataclass(frozen=True)
class TradingDay:
    trading_day: str
    day_end_ns: int  # scheduled market close (payout request window)
    reopen_ns: int  # next permitted opening
    deadline_ns: int  # mandatory flat deadline


@dataclass(frozen=True)
class CampaignInputs:
    executions: tuple[StrategyExecution, ...]
    paths: Mapping[str, ExecutionPath]
    trading_days: tuple[TradingDay, ...]
    start_ns: int
    cutoff_ns: int
    instrument: str
    quantity: int
    cost_per_side_cents: int
    processing: ProcessingClockPolicy
    profiles: tuple[FundedFirmProfile, ...]


def validate_inputs(inputs: CampaignInputs) -> None:
    spec = INSTRUMENTS.get(inputs.instrument)
    if spec is None:
        raise SettingsError(f"unknown instrument {inputs.instrument!r}")
    if inputs.quantity <= 0:
        raise SettingsError("position size must be a positive whole number of contracts")
    exposure = spec.mini_equivalent_tenths * inputs.quantity
    for profile in inputs.profiles:
        if exposure > profile.max_mini_equivalent_tenths:
            raise SettingsError(
                f"{inputs.quantity} {spec.label} exceeds the {profile.firm_name} limit of "
                f"{profile.max_minis_label}; choose a supported size (the scenario is "
                "refused, never clipped)"
            )
    ends = sorted(day.day_end_ns for day in inputs.trading_days)
    for execution in inputs.executions:
        if execution.is_warmup:
            continue
        if execution.trade_id not in inputs.paths:
            raise SettingsError(f"no price path for execution {execution.trade_id}")
        entry, exit_ = to_ns(execution.entry_ts_utc), to_ns(execution.exit_ts_utc)
        # funded accounts require flat positions at every session close
        if any(entry < end < exit_ for end in ends):
            raise SettingsError(
                "the strategy source holds positions through a session close; a funded "
                "simulation requires the mandatory daily-close holding policy"
            )


class FirmCampaign:
    """One firm's event loop (resumable)."""

    def __init__(self, inputs: CampaignInputs, profile: FundedFirmProfile) -> None:
        self.inputs = inputs
        self.profile = profile
        self.by_id = {e.trade_id: e for e in inputs.executions}
        self.day_of_exec = {e.trade_id: e.trading_day for e in inputs.executions}
        self.instance = FirmInstance(
            profile=profile, processing=inputs.processing, quantity=inputs.quantity,
            tick_value_cents=INSTRUMENTS[inputs.instrument].tick_value_cents,
            cost_per_side_cents=inputs.cost_per_side_cents, start_ns=inputs.start_ns,
        )
        self.queue: list[tuple] = []
        self.seq = 0
        self.processed = 0
        self.growth_pending: set[int] = set()
        self.started = False

    # ── queue ─────────────────────────────────────────────────────────────
    def push(self, ts_ns: int, kind: str, *payload: Any) -> None:
        self.seq += 1
        heapq.heappush(self.queue, (ts_ns, PRIORITY[kind], self.seq, kind, list(payload)))

    def _seed(self) -> None:
        inputs = self.inputs
        self.instance.start()
        for execution in inputs.executions:
            if execution.is_warmup:
                continue
            entry = to_ns(execution.entry_ts_utc)
            if inputs.start_ns < entry <= inputs.cutoff_ns:
                self.push(entry, "signal", execution.trade_id)
        for grant_id, at in month_grant_times(inputs.start_ns, inputs.cutoff_ns):
            self.push(at, "monthly_grant", grant_id)
        for day in inputs.trading_days:
            if inputs.start_ns < day.day_end_ns <= inputs.cutoff_ns:
                self.push(day.day_end_ns, "day_end", day.trading_day)
            if inputs.start_ns < day.reopen_ns <= inputs.cutoff_ns:
                self.push(day.reopen_ns, "day_release")
        self.started = True

    def run(self, *, stop_after_events: int | None = None) -> bool:
        """Process events; returns True when the queue is exhausted."""

        if not self.started:
            self._seed()
        inputs = self.inputs
        instance = self.instance
        while self.queue:
            if stop_after_events is not None and self.processed >= stop_after_events:
                return False
            ts, _prio, _seq, kind, payload = heapq.heappop(self.queue)
            if ts > inputs.cutoff_ns:
                # beyond the cash horizon: leave pending money unreceived
                self.queue.clear()
                break
            self.processed += 1
            if kind == "signal":
                execution = self.by_id[payload[0]]
                instance.on_signal(
                    ts, execution, inputs.paths[execution.trade_id],
                    lambda at, _firm, account_id, tid=execution.trade_id: self.push(
                        at, "position_closed", account_id, tid
                    ),
                )
            elif kind == "position_closed":
                account_id, trade_id = payload
                execution = self.by_id[trade_id]
                instance.on_position_closed(
                    ts, account_id, execution, inputs.paths[trade_id],
                    self.day_of_exec[trade_id],
                )
            elif kind == "day_end":
                before = {a.account_id: a.pending_payout for a in instance.accounts}
                instance.end_of_trading_day(ts, payload[0], inputs.cutoff_ns)
                for account in instance.accounts:
                    pending = account.pending_payout
                    if (pending and pending.get("state") == "processing"
                            and before.get(account.account_id) is not pending):
                        self.push(pending["due_ns"], "processing_complete",
                                  account.account_id, pending["request_id"])
            elif kind == "processing_complete":
                instance.processing_complete(ts, payload[0], payload[1])
                if ts not in self.growth_pending:
                    self.growth_pending.add(ts)
                    self.push(ts, "growth_review")
            elif kind == "growth_review":
                self.growth_pending.discard(ts)
                instance.review_growth(ts, chicago_date(ts).isoformat())
            elif kind == "monthly_grant":
                instance.monthly_grant(ts, payload[0])
            elif kind == "day_release":
                instance.next_day_release(ts)
            else:  # pragma: no cover
                raise ValueError(kind)
        instance.close_durations(inputs.cutoff_ns)
        return True

    # ── checkpoint ────────────────────────────────────────────────────────
    def checkpoint(self) -> str:
        state = {
            "firm_key": self.profile.firm_key,
            "queue": [list(item) for item in sorted(self.queue)],
            "seq": self.seq,
            "processed": self.processed,
            "growth_pending": sorted(self.growth_pending),
            "started": self.started,
            "instance": self.instance.snapshot(),
        }
        return json.dumps(state, sort_keys=True, default=_json_default)

    @classmethod
    def resume(cls, inputs: CampaignInputs, profile: FundedFirmProfile,
               checkpoint: str) -> FirmCampaign:
        state = json.loads(checkpoint)
        if state["firm_key"] != profile.firm_key:
            raise ValueError("checkpoint belongs to another firm instance")
        campaign = cls(inputs, profile)
        campaign.queue = [tuple(item) for item in state["queue"]]
        heapq.heapify(campaign.queue)
        campaign.seq = state["seq"]
        campaign.processed = state["processed"]
        campaign.growth_pending = set(state["growth_pending"])
        campaign.started = state["started"]
        campaign.instance.restore(state["instance"])
        return campaign


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(type(value))


def run_campaign(inputs: CampaignInputs) -> dict[str, FirmInstance]:
    validate_inputs(inputs)
    out: dict[str, FirmInstance] = {}
    for profile in inputs.profiles:
        campaign = FirmCampaign(inputs, profile)
        campaign.run()
        out[profile.firm_key] = campaign.instance
    return out


def resumed_equivalence(inputs: CampaignInputs, *, fraction: float = 0.5) -> dict[str, Any]:
    """Run each firm straight through AND via a mid-run JSON checkpoint."""

    report: dict[str, Any] = {}
    for profile in inputs.profiles:
        straight = FirmCampaign(inputs, profile)
        straight.run()
        total = straight.processed
        stop = max(1, int(total * fraction))
        first = FirmCampaign(inputs, profile)
        first.run(stop_after_events=stop)
        saved = first.checkpoint()
        resumed = FirmCampaign.resume(inputs, profile, saved)
        resumed.run()
        a = json.dumps(straight.instance.snapshot(), sort_keys=True, default=_json_default)
        b = json.dumps(resumed.instance.snapshot(), sort_keys=True, default=_json_default)
        report[profile.firm_key] = {
            "events": total,
            "checkpoint_after_events": stop,
            "identical": a == b,
        }
    return report


def display_time(ts_ns: int | None) -> str | None:
    return None if ts_ns is None else chicago_label(ts_ns)


def machine_time(ts_ns: int | None) -> str | None:
    return None if ts_ns is None else iso_utc(ts_ns)


def month_keys(start_ns: int, end_ns: int) -> list[str]:
    first = chicago_date(start_ns)
    last = chicago_date(end_ns)
    keys = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        keys.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return keys


def executions_sequence(executions: Sequence[StrategyExecution]) -> list[str]:
    return [e.trade_id for e in executions if not e.is_warmup]
