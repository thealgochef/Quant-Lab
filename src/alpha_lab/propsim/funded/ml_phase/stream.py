"""One causal phase stream, shared by account-independent labels and policies.

Only the shadow removes account constraints. Funded streams delegate every
account purchase, fee, floor, pause and payout to the inherited PairLedger.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from datetime import date

import numpy as np

from alpha_lab.propsim.funded.clock import from_ns
from alpha_lab.propsim.funded.pair_engine import EngineConsistencyError
from alpha_lab.propsim.funded.position_walk import open_position, walk_minute
from alpha_lab.propsim.funded.print_minutes import bar_window_ns
from alpha_lab.propsim.funded.profiles import MYFUNDEDFUTURES_PROFILE

from .driver import MlCoreDriver
from .features import build_features
from .protocol import digest


class PhaseStream:
    def __init__(
        self,
        *,
        reference: str,
        stream_id: str,
        section,
        context_index,
        definitions: list[dict],
        source_id: str,
        ledger=None,
        scorer=None,
    ):
        self.reference, self.stream_id = reference, stream_id
        self.definitions, self.source_id = definitions, source_id
        self.context_index, self.ledger, self.scorer = context_index, ledger, scorer
        self.driver = MlCoreDriver(
            section, tick_size=0.25, context_index=context_index, entry_scorer=self._entry
        )
        self.shadow_position = None
        self.entry_rows = {}
        self.checkpoint_rows = {}
        self.close_positions = {}
        self.close_results = {}
        self.shadow_trades = []
        self.datasets = {"ENTRY": [], "CONTINUATION": []}
        self.day = None
        self.day_records = []

    @property
    def is_shadow(self):
        return self.ledger is None

    @property
    def position(self):
        return self.shadow_position if self.is_shadow else self.ledger.position

    def _features(self, record, job, *, decision_ns, price, checkpoint=None):
        if self.day.schedule is None:
            raise ValueError("phase shadow requires the frozen warmup/evaluation calendar")
        return build_features(
            job=job,
            decision_ns=decision_ns,
            entry_ns=record["decision_ns"],
            decision_price=price,
            trading_day=self.day.trading_day,
            deadline_ns=self.day.schedule.deadline_ns,
            structure=record["structure"],
            completed_bars=self.driver.completed_bars,
            context=self.context_index.snapshot(from_ns(decision_ns)),
            definitions=self.definitions,
            checkpoint=checkpoint,
        )

    def _identity(self, record, job, stamp, ordinal):
        episode = digest(
            {"reference": self.reference, "setup": record["setup_id"], "source": self.source_id}
        )
        shared = digest(
            {
                "htf": record["structure"]["htf_id"],
                "entry_ns": record["decision_ns"],
                "entry": record["structure"]["entry_price"],
            }
        )
        key = {
            "reference": self.reference,
            "stream": self.stream_id,
            "job": job,
            "candidate": record["candidate_id"],
            "decision_ns": stamp,
            "event_ordinal": ordinal,
            "source": self.source_id,
        }
        return {
            **key,
            "row_id": digest(key),
            "episode_id": episode,
            "cross_reference_episode_id": shared,
            "trading_day": self.day.trading_day,
        }

    def _entry(self, record, driver):
        features = self._features(
            record,
            "ENTRY",
            decision_ns=record["decision_ns"],
            price=record["structure"]["entry_price"],
        )
        row = {
            **self._identity(record, "ENTRY", record["decision_ns"], "completed_bar"),
            **features["values"],
            "feature_provenance": features["provenance"],
            "structure": record["structure"],
        }
        prediction = self._score("ENTRY", row)
        row["prediction"] = prediction
        self.entry_rows[record["candidate_id"]] = row
        return prediction

    def _score(self, job, row):
        if self.scorer is None:
            return {"action": "baseline", "score": None, "reason": "model_disabled"}
        return self.scorer(job, row)

    def _continuation(self, pos, visible, index):
        record = self.driver.entries[pos.trade_ref]
        stamp = int(visible.ts_ns[index])
        ordinal = (
            int(visible.event_ordinals[index]) if visible.event_ordinals is not None else index
        )
        key = self._identity(record, "CONTINUATION", stamp, ordinal)
        if visible.fidelity != "ordered_trade_prints":
            row = {
                **key,
                "label_status": "approximate_checkpoint",
                "feature_provenance": {},
                **{r["name"]: None for r in self.definitions if "CONTINUATION" in r["jobs"]},
            }
            prediction = {
                "action": "baseline_unavailable",
                "score": None,
                "reason": "approximate_checkpoint",
            }
        else:
            features = self._features(
                record,
                "CONTINUATION",
                decision_ns=stamp,
                price=int(visible.price_ticks[index]) * 0.25,
                checkpoint={
                    "event_id": key["row_id"],
                    "observed_min_price": pos.price_min_ticks * 0.25,
                    "target_price": pos.target_ticks * 0.25,
                },
            )
            row = {**key, **features["values"], "feature_provenance": features["provenance"]}
            prediction = self._score("CONTINUATION", row)
        row["prediction"] = prediction
        row["checkpoint_fidelity"] = visible.fidelity
        self.checkpoint_rows[pos.trade_ref] = row
        if self.is_shadow and visible.fidelity == "ordered_trade_prints":
            branch = copy.deepcopy(pos)
            branch.continuation_decision = {
                "action": "change",
                "state": "pending",
                "reason": "counterfactual_close_label",
                "decision_ns": stamp,
                "decision_minute_ns": visible.open_ns,
                "decision_ordinal": ordinal,
            }
            self.close_positions[pos.trade_ref] = branch
        return prediction

    def _walk_close_labels(self, observation, deadline):
        for trade_id, position in list(self.close_positions.items()):
            intent = position.continuation_decision
            ordinals = (
                np.arange(len(observation.ts_ns))
                if observation.event_ordinals is None
                else observation.event_ordinals
            )
            mask = observation.ts_ns >= intent["decision_ns"]
            if observation.open_ns == intent["decision_minute_ns"]:
                mask &= ordinals >= intent["decision_ordinal"]
            clipped = replace(
                observation,
                ts_ns=observation.ts_ns[mask],
                price_ticks=observation.price_ticks[mask],
                continuous=observation.continuous[mask],
                event_ordinals=ordinals[mask],
            )
            result = walk_minute(
                profile=MYFUNDEDFUTURES_PROFILE,
                pos=position,
                obs=clipped,
                deadline_minute=deadline,
                _count_minute=observation.open_ns != intent["decision_minute_ns"],
            )
            if result is not None:
                self.close_results[trade_id] = {
                    "exit": asdict(result),
                    "position": position.to_json(),
                }
                del self.close_positions[trade_id]

    def _finish_shadow(self, pos, outcome):
        record = self.driver.entries[pos.trade_ref]
        entry = self.entry_rows[record["candidate_id"]]
        risk = pos.initial_risk_cents
        if risk <= 0:
            raise ValueError("shadow initial risk must be positive")
        trade = {
            "trade_id": pos.trade_ref,
            "entry": entry,
            "position": pos.to_json(),
            "exit": asdict(outcome),
            "initial_risk_cents": risk,
            "net_pnl_cents": outcome.balance_after_cents - pos.balance_before_cents,
            "is_warmup": not self.day.is_evaluation,
        }
        self.shadow_trades.append(trade)
        if not self.day.is_evaluation:
            return
        components = {
            "entry_cost_cents": pos.entry_cost_cents,
            "partial_gross_cents": pos.partial_gross_cents,
            "partial_cost_cents": pos.partial_cost_cents,
            "exit_cost_cents": pos.exit_cost_cents,
            "gross_pnl_cents": outcome.gross_pnl_cents,
            "net_pnl_cents": trade["net_pnl_cents"],
            "initial_risk_cents": risk,
        }
        status = "exact" if pos.minutes_approximated == 0 else "minute_approximation"
        common = {
            "label_start_ns": pos.entry_ns,
            "label_end_ns": outcome.ts_ns,
            "label_available_ns": outcome.ts_ns,
            "trade_id": pos.trade_ref,
        }
        self.datasets["ENTRY"].append(
            {
                **entry,
                **common,
                "label_status": status,
                "label": trade["net_pnl_cents"] / risk if status == "exact" else None,
                "label_components": components,
            }
        )
        checkpoint = self.checkpoint_rows.get(pos.trade_ref)
        if checkpoint is not None:
            close = self.close_results.get(pos.trade_ref)
            continuation_status = status
            if close is None:
                continuation_status = "close_branch_unavailable"
            elif close["position"]["minutes_approximated"]:
                continuation_status = "minute_approximation"
            close_value = None if close is None else close["exit"]["balance_after_cents"]
            end = max(outcome.ts_ns, close["exit"]["ts_ns"]) if close else outcome.ts_ns
            self.datasets["CONTINUATION"].append(
                {
                    **checkpoint,
                    **common,
                    "label_start_ns": checkpoint["decision_ns"],
                    "label_end_ns": end,
                    "label_available_ns": end,
                    "label_status": continuation_status,
                    "label": (
                        (outcome.balance_after_cents - close_value) / risk
                        if continuation_status == "exact"
                        else None
                    ),
                    "label_components": {
                        **components,
                        "hold_balance_cents": outcome.balance_after_cents,
                        "close_balance_cents": close_value,
                        "close_branch": close,
                    },
                }
            )

    def run_day(self, day, prints):
        self.day = day
        bars = self.driver.begin_day(day.bars_by_tf, day.levels_for)
        if self.ledger is not None and not self.ledger.started:
            self.ledger.start()
        before_entries, before_trades = len(self.driver.entries), len(self.shadow_trades)
        for bar in bars:
            open_ns, close_ns = bar_window_ns(bar)
            deadline = close_ns == day.schedule.deadline_ns
            outcome, opening = None, self.position
            if self.ledger is not None:
                self.ledger.run_until(open_ns)
            if opening is not None:
                observations = prints().minute(bar, opening.sign)
                if self.is_shadow:
                    outcome = walk_minute(
                        profile=MYFUNDEDFUTURES_PROFILE,
                        pos=opening,
                        obs=observations,
                        deadline_minute=deadline,
                        continuation_selector=self._continuation,
                    )
                    self._walk_close_labels(observations, deadline)
                    if outcome is not None:
                        self._finish_shadow(opening, outcome)
                        self.shadow_position = None
                else:
                    outcome = self.ledger.on_minute(
                        observations,
                        deadline_minute=deadline,
                        trading_day=day.trading_day,
                        continuation_selector=self._continuation,
                    )
            gate = None if self.is_shadow else self.ledger.gate()
            step = self.driver.step(bar, gate)
            if step.core_exits and self.position is not None:
                raise EngineConsistencyError("Core resolved a phase position still open on prints")
            if outcome is not None and self.driver.in_position:
                allowed = (
                    outcome.account_failed
                    or outcome.kind == "ml_continuation_close"
                    or (outcome.kind == "breakeven_stop" and outcome.scaled_in_exit_minute)
                )
                if not allowed:
                    raise EngineConsistencyError(f"unexplained phase close: {outcome.kind}")
                self.driver.force_flat(counted=outcome.account_failed)
            if step.refused and gate is not None and gate in step.refused:
                self.ledger.note_blocked(close_ns, gate, "entry refused and discarded")
                self.driver.discard_refused_setup()
            if step.entry is not None:
                if outcome is not None or gate is not None:
                    raise EngineConsistencyError("entry on a resolution or refused candle")
                signal = step.entry
                if self.is_shadow:
                    self.shadow_position, failure = open_position(
                        profile=MYFUNDEDFUTURES_PROFILE,
                        trade_ref=signal.trade_id,
                        direction=signal.direction,
                        entry_ns=close_ns,
                        entry_ticks=signal.entry_ticks,
                        stop_ticks=signal.stop_ticks,
                        target_ticks=signal.target_ticks,
                        quantity=10,
                        tick_value_cents=50,
                        cost_per_side_cents=0,
                        cost_per_contract_mills=514,
                        balance_cents=0,
                        floor_cents=0,
                        peak_cents=0,
                        scale_out=True,
                        enforce_account_floor=False,
                    )
                    if failure is not None:
                        raise AssertionError("label shadow unexpectedly has account failure")
                else:
                    self.ledger.open(
                        ts_ns=close_ns,
                        trade_ref=signal.trade_id,
                        direction=signal.direction,
                        entry_ticks=signal.entry_ticks,
                        stop_ticks=signal.stop_ticks,
                        target_ticks=signal.target_ticks,
                        trading_day=day.trading_day,
                        strategy={
                            "trade_id": signal.trade_id,
                            "entry_chart": signal.entry_chart,
                            "htf_zone_id": signal.htf_zone_id,
                        },
                    )
                    if self.ledger.position is None:
                        self.driver.force_flat()
        if self.position is not None or self.close_positions:
            raise EngineConsistencyError("phase exposure remains after mandatory daily close")
        tail = self.driver.end_day(
            date.fromisoformat(day.trading_day), dataset_exhausted=day.exhausted
        )
        if tail:
            raise EngineConsistencyError("phase Core position remained at day end")
        self.day_records.append(
            {
                "trading_day": day.trading_day,
                "entries": len(self.driver.entries) - before_entries,
                "shadow_resolutions": len(self.shadow_trades) - before_trades,
            }
        )

    def snapshot(self):
        if self.position is not None:
            raise ValueError("daily checkpoint requires flat position")
        return {
            "driver": self.driver.checkpoint(),
            "entry_rows": self.entry_rows,
            "checkpoint_rows": self.checkpoint_rows,
            "close_results": self.close_results,
            "shadow_trades": self.shadow_trades,
            "datasets": self.datasets,
            "day_records": self.day_records,
            "ledger": None if self.ledger is None else self.ledger.snapshot(),
        }

    def restore(self, state):
        self.driver.restore(state["driver"])
        for name in (
            "entry_rows",
            "checkpoint_rows",
            "close_results",
            "shadow_trades",
            "datasets",
            "day_records",
        ):
            setattr(self, name, state[name])
        if self.ledger is not None:
            self.ledger.restore(state["ledger"])
