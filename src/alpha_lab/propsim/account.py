"""The deterministic prop-account walk: typed state + totally ordered events.

Extends the proven evaluation semantics (``alpha_lab.propsim.engine``, which
stays untouched) to the FULL account lifecycle (CS §5.3/§5.4): evaluation →
funded → payouts/fees/replacement, driven by the ordered trade-path bundle
under ONE account policy set. Every state change is a typed
:class:`PropAccountEventEnvelope` in one strictly ordered stream
(``event_order_policy_id = "prop_account_event_order_v1"``: emission follows
the deterministic day walk; ties at one timestamp resolve by the fixed kind
precedence baked into the walk's emission order, surfaced through the global
``event_ordinal``). At one fixed contract with fees/payouts disabled the
evaluation phase reproduces :class:`~alpha_lab.propsim.engine.EvaluationWalk`
EXACTLY (the §16.4 parity row).
"""

from __future__ import annotations

import copy as _copy
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import ClassVar, Literal

from pydantic import Field

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    register_identity_pair,
)
from alpha_lab.propsim.calendar import (
    HISTORICAL_CLOCK_POLICY,
    DayCountBasis,
    DurationRule,
    SimulatedClock,
    SimulatedClockPolicy,
    UnsupportedCalendarRuleError,
)
from alpha_lab.propsim.firm_contracts import (
    AccountPhase,
    PhaseRules,
    PropFirmContractPayload,
)
from alpha_lab.propsim.risk import (
    PropRiskPolicyPayload,
    SizingDecision,
    size_position,
)
from alpha_lab.propsim.withdrawal import (
    WithdrawalPolicyPayload,
    plan_withdrawal_request,
)

__all__ = [
    "EVENT_ORDER_POLICY_V1",
    "BREACH_MODES",
    "UnsupportedFirmRuleError",
    "AccountTrade",
    "PropAccountState",
    "PropFeeEvent",
    "PropPayoutEvent",
    "PropBreachEvent",
    "PropReplacementEvent",
    "PropPhaseTransitionEvent",
    "PropThresholdRatchetEvent",
    "PropDailyHaltEvent",
    "PropEquityUpdateEvent",
    "PropAccountEventEnvelope",
    "AccountPolicySetPayload",
    "AccountPolicySetEnvelope",
    "AccountWalk",
    "AccountWalkResult",
]

EVENT_ORDER_POLICY_V1 = "prop_account_event_order_v1"

#: The three breach-observation modes the walk implements. The two unrealized
#: modes are SCENARIOS over MFE/MAE magnitudes: adverse-first tests the
#: position-adverse excursion against the CURRENT floor before any favorable
#: excursion can raise it; favorable-first raises the intraday-trail peak from
#: the favorable excursion FIRST, then tests the adverse excursion against the
#: (possibly raised) floor. On order-sensitive trades under an intraday trail
#: the two modes produce different results — that difference IS the scenario.
BREACH_MODES: tuple[str, ...] = (
    "realized_only",
    "unrealized_adverse_first",
    "unrealized_favorable_first",
)


class UnsupportedFirmRuleError(ValueError):
    """The walk cannot honor a rule the firm contract declares (fail-closed)."""


@dataclass(frozen=True)
class AccountTrade:
    """One resolved trade, in dollars-per-contract terms, with exact links."""

    day: date
    entry_ts_utc: str
    resolution_ts_utc: str
    points: float
    risk_points: float
    mfe_pts: float | None
    mae_pts: float | None
    trade_id: str
    decision_id: str | None = None
    candidate_id: str | None = None
    setup_id: str | None = None
    entry_path_event_id: str | None = None
    exit_path_event_id: str | None = None


@dataclass
class PropAccountState:
    phase: AccountPhase
    balance: float
    equity: float
    high_water_mark: float
    drawdown_floor: float
    daily_realized_pnl: float
    daily_unrealized_pnl: float
    available_buffer: float
    contract_allowance: int
    winning_day_count: int
    consistency_ok: bool
    payout_eligible: bool
    payout_available: float
    post_payout_floor: float | None
    fees_paid_total: float
    breached: bool
    breach_reason: str | None
    account_age_days: int
    days_in_phase: int


class PropFeeEvent(FrozenContract):
    fee_kind: Literal["evaluation", "activation", "recurring", "reset"]
    amount: float


class PropPayoutEvent(FrozenContract):
    requested_amount: float
    approved_amount: float
    trader_amount: float
    firm_amount: float


class PropBreachEvent(FrozenContract):
    breach_reason: str
    threshold_value: float
    observed_equity: float


class PropReplacementEvent(FrozenContract):
    prior_account_id: str
    replacement_account_id: str
    reset_fee: float


class PropPhaseTransitionEvent(FrozenContract):
    from_phase: AccountPhase
    to_phase: AccountPhase
    reason: str


class PropThresholdRatchetEvent(FrozenContract):
    prior_floor: float
    new_floor: float
    reference_equity: float


class PropDailyHaltEvent(FrozenContract):
    halt_reason: str
    halt_until_trading_day: str


class PropEquityUpdateEvent(FrozenContract):
    prior_equity: float
    new_equity: float
    realized_delta: float
    unrealized_delta: float


PropAccountEventBody = (
    PropFeeEvent
    | PropPayoutEvent
    | PropBreachEvent
    | PropReplacementEvent
    | PropPhaseTransitionEvent
    | PropThresholdRatchetEvent
    | PropDailyHaltEvent
    | PropEquityUpdateEvent
)


class PropAccountEventEnvelope(FrozenContract):
    event_id: str = Field(pattern=SHA256_PATTERN)
    event_ts_utc: str
    event_ordinal: int = Field(ge=0)
    path_instance_id: str
    account_id: str
    account_ordinal: int = Field(ge=0)
    firm_contract_id: str
    account_phase: AccountPhase
    source_trade_id: str | None
    source_decision_id: str | None
    source_candidate_id: str | None
    source_setup_id: str | None
    source_path_event_id: str | None
    event_type: Literal[
        "fee",
        "payout",
        "breach",
        "replacement",
        "phase_transition",
        "threshold_ratchet",
        "daily_halt",
        "equity_update",
    ]
    event_order_policy_id: str
    payload: PropAccountEventBody


class AccountPolicySetPayload(FrozenContract):
    firm_contract_id: str = Field(pattern=SHA256_PATTERN)
    risk_policy_id: str = Field(pattern=SHA256_PATTERN)
    withdrawal_policy_id: str = Field(pattern=SHA256_PATTERN)
    replacement_policy: Literal["none", "auto_replace_up_to_n"]
    max_replacements: int = Field(ge=0)
    clock_policy_id: str


class AccountPolicySetEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "account_policy_set_id"

    account_policy_set_id: str = Field(pattern=SHA256_PATTERN)
    payload: AccountPolicySetPayload


@dataclass(frozen=True)
class AccountWalkResult:
    events: tuple[PropAccountEventEnvelope, ...]
    final_state: PropAccountState
    accounts_used: int
    verdict: str  # "funded_alive" | "breached_out" | "expired" | "retired" | "evaluation_alive"
    total_trader_payouts: float
    total_fees: float
    per_account_verdicts: tuple[str, ...]


def _event_id(fields: dict) -> str:
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _excursion(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return abs(value)


class AccountWalk:
    """Deterministic lifecycle walk over ordered day blocks of trades.

    ``breach_mode`` mirrors the proven engine semantics: ``realized_only``
    (closed-trade evidence) or ``unrealized_adverse_first`` (the registered
    adverse-first SCENARIO over MFE/MAE — a conservative approximation, never
    exact history). Fees/payouts/replacement apply per the policy set;
    ``enable_fees=False`` with one fixed contract makes the evaluation phase
    the exact EvaluationWalk parity surface.
    """

    def __init__(
        self,
        *,
        firm: PropFirmContractPayload,
        firm_contract_id: str,
        risk_policy: PropRiskPolicyPayload,
        withdrawal_policy: WithdrawalPolicyPayload,
        policy_set: AccountPolicySetPayload,
        path_instance_id: str,
        breach_mode: str = "realized_only",
        clock_policy: SimulatedClockPolicy = HISTORICAL_CLOCK_POLICY,
        enable_fees: bool = True,
        bundle_event_ids: frozenset[str] | None = None,
        account_namespace: str | None = None,
        account_open_ts_utc: str | None = None,
    ) -> None:
        if breach_mode not in BREACH_MODES:
            raise ValueError(f"unknown breach mode {breach_mode!r}")
        self._firm = firm
        self._firm_contract_id = firm_contract_id
        self._risk = risk_policy
        self._withdrawal = withdrawal_policy
        self._policy_set = policy_set
        self._path_instance_id = path_instance_id
        self._mode = breach_mode
        self._clock_policy = clock_policy
        self._enable_fees = enable_fees
        self._bundle_event_ids = bundle_event_ids
        self._account_namespace = account_namespace
        self._validate_firm_rules()
        self._events: list[PropAccountEventEnvelope] = []
        self._ordinal = 0
        self._account_ordinal = 0
        self._per_account_verdicts: list[str] = []
        self._total_trader_payouts = 0.0
        self._total_fees = 0.0
        self._start_account(open_ts=account_open_ts_utc)

    def _validate_firm_rules(self) -> None:
        """Fail closed on any declared rule this walk cannot honor (P0-14/CS §5.2).

        Every duration rule's basis must be representable under the active
        clock; a firm requiring unrealized breach observation refuses a
        realized-only walk (the reverse — walking a MORE conservative
        unrealized scenario than the firm mandates — is permitted and labeled
        by the breach mode itself); payout processing delays are unimplemented
        and refuse rather than silently settle instantly.
        """

        firm = self._firm
        duration_rules: list[tuple[str, DurationRule | None]] = [
            ("evaluation.min_days", firm.evaluation.min_days),
            ("evaluation.max_eval_days", firm.evaluation.max_eval_days),
            ("fees.recurring_period", firm.fees.recurring_period),
            ("account_expiration", firm.account_expiration),
        ]
        if firm.funded is not None:
            duration_rules += [
                ("funded.min_days", firm.funded.min_days),
                ("funded.max_eval_days", firm.funded.max_eval_days),
            ]
        if firm.payout is not None:
            duration_rules += [
                ("payout.waiting_period", firm.payout.waiting_period),
                ("payout.min_between_payouts", firm.payout.min_between_payouts),
                ("payout.min_winning_days", firm.payout.min_winning_days),
            ]
        if self._withdrawal.behavior == "fixed_cadence":
            duration_rules.append(("withdrawal.cadence", self._withdrawal.cadence))
        for rule_path, rule in duration_rules:
            if rule is None:
                continue
            if not self._clock_policy.supports(rule.basis):
                raise UnsupportedCalendarRuleError(
                    f"{rule_path}: day-count basis {rule.basis.value!r} is not "
                    f"representable under clock policy "
                    f"{self._clock_policy.policy_id!r} — refusing to guess"
                )
        if (
            firm.payout is not None
            and firm.payout.payout_processing is not None
            and firm.payout.payout_processing.count > 0
        ):
            raise UnsupportedFirmRuleError(
                "payout.payout_processing delays are not implemented by this "
                "walk; refusing to silently settle payouts instantly"
            )
        for phase_path, phase in (
            ("evaluation", firm.evaluation),
            ("funded", firm.funded),
        ):
            if phase is None:
                continue
            if (
                phase.unrealized_equity_counts_for_breach
                and self._mode == "realized_only"
            ):
                raise UnsupportedFirmRuleError(
                    f"{phase_path}: the firm counts unrealized equity for "
                    "breach but the walk is realized-only — under-observation "
                    "would understate breach risk (fail-closed)"
                )

    # ── account lifecycle plumbing ──────────────────────────────────────────

    def _account_id(self) -> str:
        prefix = f"{self._account_namespace}-" if self._account_namespace else ""
        return (
            f"{prefix}acct-{self._account_ordinal:03d}-"
            f"{canonical_contract_sha256(self._policy_set)[:12]}"
        )

    def _start_account(self, *, open_ts: str | None) -> None:
        rules = self._firm.evaluation
        self._open_ts = open_ts
        self._phase_rules = rules
        self._clock = SimulatedClock(policy=self._clock_policy)
        self._phase_start_index = 0
        self._last_payout_index: int | None = None
        self._funded_fee_marker = 0
        self._payouts_taken = 0
        self._consecutive_losses = 0
        self._recovery_wins = 0
        self._payout_base = rules.starting_balance
        self._peak = rules.starting_balance
        self._phase_entry_balance = rules.starting_balance
        self._floor_locked_at: float | None = None
        self._day_pnls: list[float] = []
        self.state = PropAccountState(
            phase=AccountPhase.EVALUATION,
            balance=rules.starting_balance,
            equity=rules.starting_balance,
            high_water_mark=rules.starting_balance,
            drawdown_floor=self._compute_floor(rules),
            daily_realized_pnl=0.0,
            daily_unrealized_pnl=0.0,
            available_buffer=0.0,
            contract_allowance=rules.max_contracts,
            winning_day_count=0,
            consistency_ok=True,
            payout_eligible=False,
            payout_available=0.0,
            post_payout_floor=None,
            fees_paid_total=self._total_fees,
            breached=False,
            breach_reason=None,
            account_age_days=0,
            days_in_phase=0,
        )
        self._refresh_buffer()
        # the initial account may not know its open timestamp yet: defer the
        # evaluation fee to the first played day (a *_ts_utc field must never
        # carry a sentinel); replacement accounts always pass a real ts
        self._pending_evaluation_fee = bool(
            self._enable_fees and self._firm.fees.evaluation_fee
        )
        if self._pending_evaluation_fee and self._open_ts is not None:
            self._pending_evaluation_fee = False
            self._emit_fee("evaluation", self._firm.fees.evaluation_fee, self._open_ts)

    def _compute_floor(self, rules: PhaseRules) -> float:
        floor = self._peak - rules.trail_amount
        if rules.trail_locks_at_start:
            floor = min(floor, rules.starting_balance)
        return floor

    def _refresh_buffer(self) -> None:
        self.state.available_buffer = self.state.balance - self.state.drawdown_floor
        self.state.equity = self.state.balance

    # ── event emission (the ONE strictly ordered stream) ────────────────────

    def _emit(
        self,
        event_type: str,
        payload: PropAccountEventBody,
        *,
        ts: str,
        trade: AccountTrade | None = None,
        path_event_id: str | None = None,
    ) -> None:
        if (
            path_event_id is not None
            and self._bundle_event_ids is not None
            and path_event_id not in self._bundle_event_ids
        ):
            raise ValueError(
                f"account event cites unknown path event {path_event_id[:12]}…"
            )
        fields = {
            "path_instance_id": self._path_instance_id,
            "account_namespace": self._account_namespace,
            "account_ordinal": self._account_ordinal,
            "event_ordinal": self._ordinal,
            "event_type": event_type,
            "ts": ts,
            "payload": payload.model_dump(mode="json"),
        }
        self._events.append(
            PropAccountEventEnvelope(
                event_id=_event_id(fields),
                event_ts_utc=ts,
                event_ordinal=self._ordinal,
                path_instance_id=self._path_instance_id,
                account_id=self._account_id(),
                account_ordinal=self._account_ordinal,
                firm_contract_id=self._firm_contract_id,
                account_phase=self.state.phase,
                source_trade_id=trade.trade_id if trade else None,
                source_decision_id=trade.decision_id if trade else None,
                source_candidate_id=trade.candidate_id if trade else None,
                source_setup_id=trade.setup_id if trade else None,
                source_path_event_id=path_event_id,
                event_type=event_type,  # type: ignore[arg-type]
                event_order_policy_id=EVENT_ORDER_POLICY_V1,
                payload=payload,
            )
        )
        self._ordinal += 1

    def _emit_fee(self, kind: str, amount: float, ts: str) -> None:
        self._total_fees += amount
        self.state.fees_paid_total = self._total_fees
        self._emit("fee", PropFeeEvent(fee_kind=kind, amount=amount), ts=ts)  # type: ignore[arg-type]

    # ── the day walk ────────────────────────────────────────────────────────

    def play_day(self, day: date, trades: Sequence[AccountTrade]) -> str | None:
        """Play one trading day; returns a terminal verdict or None."""

        if self.state.phase in (
            AccountPhase.BREACHED,
            AccountPhase.EXPIRED,
            AccountPhase.RETIRED,
        ):
            raise RuntimeError(f"account already terminal: {self.state.phase}")
        rules = self._phase_rules
        eod_ts = f"{day.isoformat()}T23:59:59+00:00"
        if self._pending_evaluation_fee:
            self._pending_evaluation_fee = False
            self._emit_fee(
                "evaluation",
                self._firm.fees.evaluation_fee,
                f"{day.isoformat()}T00:00:00+00:00",
            )

        # account expiration (per account; day max+1 pattern for trading days)
        expiration = self._firm.account_expiration
        if expiration is not None:
            expired = (
                self.state.account_age_days >= expiration.count
                if expiration.basis is DayCountBasis.TRADING_DAY
                else self._clock.played_days > 0
                and self._clock.elapsed_since(0, expiration.basis) >= expiration.count
            )
            if expired:
                return self._terminal_expired(eod_ts, reason="account_expiration")

        # phase day-budget expiry lands on day max+1 (engine parity for the
        # trading-day basis; other bases go through the typed clock)
        if rules.max_eval_days is not None:
            budget = rules.max_eval_days
            exhausted = (
                self.state.days_in_phase >= budget.count
                if budget.basis is DayCountBasis.TRADING_DAY
                else self._clock.played_days > 0
                and self._clock.elapsed_since(self._phase_start_index, budget.basis)
                >= budget.count
            )
            if exhausted:
                return self._terminal_expired(eod_ts)

        day_start_balance = self.state.balance
        self.state.daily_realized_pnl = 0.0
        halted = False
        skipped = 0
        for trade in trades:
            if halted:
                skipped += 1
                continue
            sizing = self._size(trade)
            if sizing.skipped:
                skipped += 1
                continue
            verdict = self._play_trade(
                trade, sizing, day_start_balance=day_start_balance, eod_ts=eod_ts
            )
            if verdict == "halted":
                halted = True
                self._emit(
                    "daily_halt",
                    PropDailyHaltEvent(
                        halt_reason="daily_loss_limit_soft",
                        halt_until_trading_day=day.isoformat(),
                    ),
                    ts=trade.resolution_ts_utc,
                    trade=trade,
                )
                continue
            if verdict is not None:
                return verdict

        # ── end of day ──────────────────────────────────────────────────────
        day_pnl = self.state.balance - day_start_balance
        self._day_pnls.append(day_pnl)
        winning_threshold = (
            self._firm.payout.winning_day_min_pnl
            if self._firm.payout is not None
            and self._firm.payout.winning_day_min_pnl is not None
            else 0.0
        )
        winning = day_pnl > winning_threshold
        if winning:
            self.state.winning_day_count += 1
        self._clock.advance(day, winning=winning)
        self.state.account_age_days += 1
        self.state.days_in_phase += 1
        # post-loss bookkeeping (day granularity: a losing day counts a loss)
        if day_pnl < 0:
            self._consecutive_losses += 1
            self._recovery_wins = 0
        elif day_pnl > 0:
            if self._consecutive_losses:
                self._recovery_wins += 1
                if (
                    self._risk.post_loss_adjustment is not None
                    and self._recovery_wins
                    >= self._risk.post_loss_adjustment.recovery_wins
                ):
                    self._consecutive_losses = 0
                    self._recovery_wins = 0

        if self.state.balance <= self.state.drawdown_floor:
            return self._terminal_breach(
                "trailing_floor", self.state.balance, eod_ts, None
            )

        if self.state.phase is AccountPhase.EVALUATION:
            verdict = self._evaluate_pass(eod_ts)
            if verdict == "evaluation_passed":
                return verdict  # terminal: no funded phase exists
            # "funded" transitions and the walk continues
        else:
            self._funded_eod(eod_ts)

        # trail ratchet (EOD styles bank the EOD balance; intraday already
        # trailed in real time; static never ratchets) — engine parity.
        prior_floor = self.state.drawdown_floor
        raise_emits_itself = False
        if self._floor_locked_at is not None:
            # a post-payout lock is permanent: the floor neither trails up
            # nor resets until the account ends
            self.state.drawdown_floor = self._floor_locked_at
        elif rules.trail_style == "eod_floor_realtime_breach":
            self._peak = max(self._peak, self.state.balance)
            self.state.drawdown_floor = max(
                self.state.drawdown_floor, self._compute_floor(rules)
            )
        elif rules.trail_style == "intraday_peak_trail":
            raise_emits_itself = True  # _raise_peak events its own ratchets
            self._raise_peak(self.state.balance, rules, ts=eod_ts)
        if not raise_emits_itself and self.state.drawdown_floor > prior_floor:
            self._emit(
                "threshold_ratchet",
                PropThresholdRatchetEvent(
                    prior_floor=prior_floor,
                    new_floor=self.state.drawdown_floor,
                    reference_equity=self.state.balance,
                ),
                ts=eod_ts,
            )
        self._refresh_buffer()
        return None

    # ── per-trade mechanics (engine-parity arithmetic, scaled by contracts) ─

    def _size(self, trade: AccountTrade) -> SizingDecision:
        rules = self._phase_rules
        return size_position(
            self._risk,
            risk_points=trade.risk_points,
            start_buffer=rules.starting_balance - self._compute_floor_static(rules),
            current_buffer=self.state.available_buffer,
            day_realized_pnl=self.state.daily_realized_pnl,
            consecutive_losses=self._consecutive_losses,
            recovery_wins_since=self._recovery_wins,
            payouts_taken=self._payouts_taken,
            phase_max_contracts=rules.max_contracts,
            micro_scaling_table=rules.micro_scaling_table,
        )

    @staticmethod
    def _compute_floor_static(rules: PhaseRules) -> float:
        floor = rules.starting_balance - rules.trail_amount
        if rules.trail_locks_at_start:
            floor = min(floor, rules.starting_balance)
        return floor

    def _raise_peak(
        self,
        equity: float,
        rules: PhaseRules,
        *,
        ts: str,
        trade: AccountTrade | None = None,
    ) -> None:
        if rules.trail_style != "intraday_peak_trail":
            return
        if self._floor_locked_at is not None:
            return  # a post-payout lock is permanent
        if equity > self._peak:
            self._peak = equity
            prior_floor = self.state.drawdown_floor
            self.state.drawdown_floor = max(
                self.state.drawdown_floor, self._compute_floor(rules)
            )
            if self.state.drawdown_floor > prior_floor:
                self._emit(
                    "threshold_ratchet",
                    PropThresholdRatchetEvent(
                        prior_floor=prior_floor,
                        new_floor=self.state.drawdown_floor,
                        reference_equity=equity,
                    ),
                    ts=ts,
                    trade=trade,
                )

    def _play_trade(
        self,
        trade: AccountTrade,
        sizing: SizingDecision,
        *,
        day_start_balance: float,
        eod_ts: str,
    ) -> str | None:
        rules = self._phase_rules
        dollars_per_point = sizing.point_value * sizing.contracts
        pnl = trade.points * dollars_per_point
        unrealized = self._mode in (
            "unrealized_adverse_first",
            "unrealized_favorable_first",
        )
        mae = _excursion(trade.mae_pts) if unrealized else None
        mfe = _excursion(trade.mfe_pts) if unrealized else None
        if self._mode == "unrealized_favorable_first" and mfe is not None:
            # the favorable-first SCENARIO: the favorable excursion raises the
            # intraday-trail peak BEFORE the adverse excursion is tested
            # against the (now possibly higher) floor
            self._raise_peak(
                self.state.balance + mfe * dollars_per_point,
                rules,
                ts=trade.resolution_ts_utc,
                trade=trade,
            )
        if mae is not None:
            adverse_equity = self.state.balance - mae * dollars_per_point
            dll_level = (
                day_start_balance - rules.dll_amount
                if rules.dll_amount is not None
                else None
            )
            floor_crossed = adverse_equity <= self.state.drawdown_floor
            dll_crossed = dll_level is not None and adverse_equity <= dll_level
            if floor_crossed and (
                not dll_crossed or self.state.drawdown_floor >= dll_level
            ):
                return self._terminal_breach(
                    "trailing_floor", adverse_equity, trade.resolution_ts_utc, trade
                )
            if dll_crossed:
                # force-close AT the DLL level; the deeper excursion never
                # happens because the position is flat (engine parity)
                prior = self.state.balance
                self.state.balance = dll_level
                self.state.daily_realized_pnl = self.state.balance - day_start_balance
                self._emit(
                    "equity_update",
                    PropEquityUpdateEvent(
                        prior_equity=prior,
                        new_equity=self.state.balance,
                        realized_delta=self.state.balance - prior,
                        unrealized_delta=0.0,
                    ),
                    ts=trade.resolution_ts_utc,
                    trade=trade,
                    path_event_id=trade.exit_path_event_id,
                )
                self._refresh_buffer()
                if rules.dll_hard:
                    return self._terminal_breach(
                        "daily_loss_limit",
                        dll_level,
                        trade.resolution_ts_utc,
                        trade,
                    )
                return "halted"
            if self._mode == "unrealized_adverse_first" and mfe is not None:
                self._raise_peak(
                    self.state.balance + mfe * dollars_per_point,
                    rules,
                    ts=trade.resolution_ts_utc,
                    trade=trade,
                )

        prior = self.state.balance
        self.state.balance += pnl
        self.state.daily_realized_pnl = self.state.balance - day_start_balance
        self._emit(
            "equity_update",
            PropEquityUpdateEvent(
                prior_equity=prior,
                new_equity=self.state.balance,
                realized_delta=pnl,
                unrealized_delta=0.0,
            ),
            ts=trade.resolution_ts_utc,
            trade=trade,
            path_event_id=trade.exit_path_event_id,
        )
        self._refresh_buffer()
        if self.state.balance <= self.state.drawdown_floor:
            return self._terminal_breach(
                "trailing_floor", self.state.balance, trade.resolution_ts_utc, trade
            )
        if (
            rules.dll_amount is not None
            and (self.state.balance - day_start_balance) <= -rules.dll_amount
        ):
            if rules.dll_hard:
                return self._terminal_breach(
                    "daily_loss_limit",
                    self.state.balance,
                    trade.resolution_ts_utc,
                    trade,
                )
            return "halted"
        self._raise_peak(
            self.state.balance, rules, ts=trade.resolution_ts_utc, trade=trade
        )
        return None

    # ── phase transitions, payouts, fees ────────────────────────────────────

    def _evaluate_pass(self, eod_ts: str) -> str | None:
        rules = self._phase_rules
        if rules.profit_target is None:
            return None
        total = self.state.balance - rules.starting_balance
        if total < rules.profit_target:
            return None
        if rules.min_days is not None:
            if rules.min_days.basis is DayCountBasis.WINNING_DAY:
                if self.state.winning_day_count < rules.min_days.count:
                    return None
            elif rules.min_days.basis is DayCountBasis.TRADING_DAY:
                if self.state.days_in_phase < rules.min_days.count:
                    return None
            elif (
                self._clock.elapsed_since(self._phase_start_index, rules.min_days.basis)
                < rules.min_days.count
            ):
                return None
        if rules.consistency_pct is not None and self._day_pnls:
            best = max(self._day_pnls)
            self.state.consistency_ok = best <= (rules.consistency_pct / 100.0) * total
            if not self.state.consistency_ok:
                return None
        if self._firm.funded is None:
            self._emit(
                "phase_transition",
                PropPhaseTransitionEvent(
                    from_phase=AccountPhase.EVALUATION,
                    to_phase=AccountPhase.RETIRED,
                    reason="evaluation_passed_no_funded_phase",
                ),
                ts=eod_ts,
            )
            self.state.phase = AccountPhase.RETIRED
            self._per_account_verdicts.append("evaluation_passed")
            return "evaluation_passed"
        self._emit(
            "phase_transition",
            PropPhaseTransitionEvent(
                from_phase=AccountPhase.EVALUATION,
                to_phase=AccountPhase.FUNDED,
                reason="evaluation_passed",
            ),
            ts=eod_ts,
        )
        self.state.phase = AccountPhase.FUNDED
        funded = self._firm.funded
        # the funded phase re-anchors its own trail around the funded start
        self._phase_rules = funded
        self._peak = self.state.balance
        floor = self.state.balance - funded.trail_amount
        if funded.trail_locks_at_start:
            floor = min(floor, funded.starting_balance)
        self.state.drawdown_floor = floor
        self._payout_base = max(self.state.balance, funded.starting_balance)
        self._phase_entry_balance = self.state.balance
        self.state.days_in_phase = 0
        self.state.winning_day_count = 0  # payout winning-day counters are phase-scoped
        self._day_pnls = []
        self._phase_start_index = max(0, self._clock.played_days - 1)
        self._refresh_buffer()
        # fees are EXTERNAL costs — they never touch the account balance
        if self._enable_fees and self._firm.fees.activation_fee:
            self._emit_fee("activation", self._firm.fees.activation_fee, eod_ts)
        return "funded"

    def _funded_eod(self, eod_ts: str) -> None:
        payout = self._firm.payout
        if payout is None:
            return
        rules = self._phase_rules
        if self._enable_fees and self._firm.fees.recurring_period is not None:
            period = self._firm.fees.recurring_period
            if period.count > 0:
                # every basis routes through the typed clock (P0-14); the
                # constructor already refused bases this clock cannot
                # represent, so this never guesses
                periods_due = (
                    self._clock.elapsed_since(self._phase_start_index, period.basis)
                    // period.count
                )
                while self._funded_fee_marker < periods_due:
                    self._funded_fee_marker += 1
                    self._emit_fee(
                        "recurring", self._firm.fees.recurring_fee, eod_ts
                    )
        eligible = True
        if not self._clock.satisfied(
            payout.waiting_period, since_index=self._phase_start_index
        ):
            eligible = False
        if eligible and self._last_payout_index is not None and not self._clock.satisfied(
            payout.min_between_payouts, since_index=self._last_payout_index
        ):
            eligible = False
        if (
            eligible
            and payout.min_winning_days is not None
            and self.state.winning_day_count < payout.min_winning_days.count
        ):
            eligible = False
        if eligible and rules.consistency_pct is not None and self._day_pnls:
            # the funded phase's consistency rule gates payout eligibility
            # (the evaluation-phase rule gates passing in _evaluate_pass)
            funded_profit = self.state.balance - self._phase_entry_balance
            if funded_profit > 0:
                best = max(self._day_pnls)
                self.state.consistency_ok = best <= (
                    rules.consistency_pct / 100.0
                ) * funded_profit
                if not self.state.consistency_ok:
                    eligible = False
        available = max(0.0, self.state.balance - self._payout_base)
        if payout.payout_cap_per_period is not None:
            available = min(available, payout.payout_cap_per_period)
        if payout.max_payout is not None:
            available = min(available, payout.max_payout)
        self.state.payout_eligible = eligible
        self.state.payout_available = available if eligible else 0.0
        if not eligible or available <= 0:
            return
        cadence = self._withdrawal.cadence
        cadence_basis = (
            cadence.basis if cadence is not None else DayCountBasis.TRADING_DAY
        )
        elapsed_since_last = (
            None
            if self._last_payout_index is None
            else self._clock.elapsed_since(self._last_payout_index, cadence_basis)
        )
        requested = plan_withdrawal_request(
            self._withdrawal,
            payout_available=available,
            current_buffer=self.state.available_buffer,
            elapsed_since_last_payout=elapsed_since_last,
        )
        if requested <= 0:
            return
        if payout.min_payout is not None and requested < payout.min_payout:
            return
        approved = min(requested, available)
        trader_amount = approved * payout.split_pct_trader / 100.0
        firm_amount = approved - trader_amount
        prior_balance = self.state.balance
        self.state.balance -= approved
        self._total_trader_payouts += trader_amount
        self._payouts_taken += 1
        self._last_payout_index = max(0, self._clock.played_days - 1)
        if payout.withdrawal_reduces_threshold:
            self._payout_base = max(0.0, self._payout_base - approved)
        if payout.post_payout_buffer_rule == "threshold_resets_to_balance_minus_trail":
            self.state.drawdown_floor = self.state.balance - rules.trail_amount
            self._peak = self.state.balance
            self.state.post_payout_floor = self.state.drawdown_floor
        elif payout.post_payout_buffer_rule == "locked_at_starting_balance":
            self.state.drawdown_floor = rules.starting_balance
            self._floor_locked_at = rules.starting_balance
            self.state.post_payout_floor = self.state.drawdown_floor
        self._emit(
            "payout",
            PropPayoutEvent(
                requested_amount=requested,
                approved_amount=approved,
                trader_amount=trader_amount,
                firm_amount=firm_amount,
            ),
            ts=eod_ts,
        )
        self._emit(
            "equity_update",
            PropEquityUpdateEvent(
                prior_equity=prior_balance,
                new_equity=self.state.balance,
                realized_delta=-approved,
                unrealized_delta=0.0,
            ),
            ts=eod_ts,
        )
        self._refresh_buffer()

    # ── terminals + replacement ─────────────────────────────────────────────

    def _terminal_breach(
        self,
        reason: str,
        observed_equity: float,
        ts: str,
        trade: AccountTrade | None,
    ) -> str:
        threshold = (
            self.state.drawdown_floor
            if reason == "trailing_floor"
            else observed_equity
        )
        self._emit(
            "breach",
            PropBreachEvent(
                breach_reason=reason,
                threshold_value=threshold,
                observed_equity=observed_equity,
            ),
            ts=ts,
            trade=trade,
            path_event_id=trade.exit_path_event_id if trade else None,
        )
        prior_phase = self.state.phase
        self._emit(
            "phase_transition",
            PropPhaseTransitionEvent(
                from_phase=prior_phase,
                to_phase=AccountPhase.BREACHED,
                reason=reason,
            ),
            ts=ts,
        )
        self.state.phase = AccountPhase.BREACHED
        self.state.breached = True
        self.state.breach_reason = reason
        self._per_account_verdicts.append(f"breached:{reason}")
        if (
            self._policy_set.replacement_policy == "auto_replace_up_to_n"
            and self._account_ordinal < self._policy_set.max_replacements
        ):
            prior_account = self._account_id()
            self._account_ordinal += 1
            replacement_account = (
                f"acct-{self._account_ordinal:03d}-"
                f"{canonical_contract_sha256(self._policy_set)[:12]}"
            )
            self._start_account(open_ts=ts)
            self._emit(
                "replacement",
                PropReplacementEvent(
                    prior_account_id=prior_account,
                    replacement_account_id=replacement_account,
                    reset_fee=self._firm.fees.reset_fee,
                ),
                ts=ts,
            )
            if self._enable_fees and self._firm.fees.reset_fee:
                self._emit_fee("reset", self._firm.fees.reset_fee, ts)
            return "replaced"
        return "breached_out"

    def _terminal_expired(
        self, ts: str, *, reason: str = "phase_day_budget_exhausted"
    ) -> str:
        self._emit(
            "phase_transition",
            PropPhaseTransitionEvent(
                from_phase=self.state.phase,
                to_phase=AccountPhase.EXPIRED,
                reason=reason,
            ),
            ts=ts,
        )
        self.state.phase = AccountPhase.EXPIRED
        self._per_account_verdicts.append("expired")
        return "expired"

    # ── results ─────────────────────────────────────────────────────────────

    def result(self) -> AccountWalkResult:
        if self.state.phase in (AccountPhase.EVALUATION, AccountPhase.FUNDED):
            verdict = (
                "funded_alive"
                if self.state.phase is AccountPhase.FUNDED
                else "evaluation_alive"
            )
        elif self.state.phase is AccountPhase.BREACHED:
            verdict = "breached_out"
        elif self.state.phase is AccountPhase.EXPIRED:
            verdict = "expired"
        else:
            verdict = "retired"
        return AccountWalkResult(
            events=tuple(self._events),
            # a defensive copy: continuing the walk must never mutate an
            # already-returned result (deep-immutability doctrine)
            final_state=_copy.copy(self.state),
            accounts_used=self._account_ordinal + 1,
            verdict=verdict,
            total_trader_payouts=self._total_trader_payouts,
            total_fees=self._total_fees,
            per_account_verdicts=tuple(self._per_account_verdicts),
        )


register_identity_pair(
    name="AccountPolicySet",
    envelope_cls=AccountPolicySetEnvelope,
    payload_cls=AccountPolicySetPayload,
    id_field="account_policy_set_id",
    example_factory=lambda: AccountPolicySetPayload(
        firm_contract_id="a" * 64,
        risk_policy_id="b" * 64,
        withdrawal_policy_id="c" * 64,
        replacement_policy="none",
        max_replacements=0,
        clock_policy_id="historical_calendar_clock_v1",
    ),
)
