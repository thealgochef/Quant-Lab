"""Reporting-only verification of saved checkpoint links; never execution authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

CONDITIONAL = {"gamma_conditional_1r_v1", "early_positive_whole_1r_v1"}
EXECUTED = {"executed_saved_context", "executed_saved_decision_receipt"}


def ns(value: Any) -> int:
    at = pd.Timestamp(value)
    if at.tzinfo is None:
        raise ValueError("checkpoint evidence requires a timezone")
    return at.value


def receipt_matches(context: Mapping, snapshot: Mapping) -> bool:
    """Match the saved receipt vocabulary to its independently selected source."""
    gamma, levels = snapshot.get("gamma") or {}, snapshot.get("levels") or {}
    age = gamma.get("positive_run_age")
    lower = gamma.get("positive_run_lower_bound", 0) or 0
    bounded = age is None and lower >= 6
    expected = {
        "gamma_report_date": gamma.get("report_date"),
        "gamma_sign": gamma.get("sign", "unknown"),
        "gamma_source_id": gamma.get("source_sha256"),
        "gamma_status": "available" if gamma.get("status") == "selected" else gamma.get("status"),
        "positive_run_age": lower if bounded else age,
        "positive_run_age_is_lower_bound": bounded,
        "level_report_date": levels.get("report_date"),
        "level_requested_date": levels.get("requested_date"),
        "level_source_id": levels.get("level_set_id"),
        "levels_status": "available"
        if levels.get("status") == "selected"
        else levels.get("status"),
    }
    if any(key not in context or context[key] != value for key, value in expected.items()):
        return False
    for key, value in (
        ("gamma_eligible_from_utc", gamma.get("nominal_eligible_from_utc")),
        ("level_eligible_from_utc", levels.get("nominal_eligible_from_utc")),
    ):
        actual = context.get(key)
        if (actual is None) != (value is None) or actual and ns(actual) != ns(value):
            return False
        if actual and ns(actual) > ns(context["decision_ts_utc"]):
            return False
    return snapshot.get("status", "within_cutoff") == "within_cutoff"


def checkpoint_origin(
    *,
    trade: Mapping,
    population: str,
    candidates: Sequence[Mapping],
    checkpoint: Any,
    section: Mapping,
    index: Any,
    unique_membership: bool = True,
) -> dict:
    """Exact scoped identity first, then the saved stream's time/source contract.

    Funded clocks truncate integer nanoseconds to Python datetime microseconds
    (clock.from_ns). That conversion is checked only AFTER the embedded target
    decision and observation equal the actual event, never as a temporal join.
    Ordinary completed-candle receipts name the opening minute; availability is
    its close one minute later under ordinary_completed_candle_exit_v1.
    """
    embedded = trade.get("target_decision") or {}
    receipt = candidates[0] if len(candidates) == 1 else None
    context = (receipt or {}).get("context") or {}
    at = context.get("decision_ts_utc")
    info = {
        "status": "unresolved",
        "reason": "missing or contradictory saved evidence",
        "match_method": None,
        "observation_utc": checkpoint,
        "observation_ns": ns(checkpoint) if checkpoint else None,
        "receipt_context_utc": at,
        "reporting_evaluation_utc": at if population == "strategy" and at else checkpoint,
        "decision_ns": embedded.get("decision_ns"),
        "precision_difference_ns": ns(checkpoint) - ns(at) if checkpoint and at else None,
        "context_usage": "unavailable",
        "role": "unresolved_checkpoint_evidence",
    }

    def reject(reason):
        return dict(info, reason=reason, reporting_evaluation_utc=None)

    if not checkpoint:
        reached = bool(trade.get("scale_out_quantity")) or trade.get("exit_kind") == "target"
        return dict(
            info,
            status="unavailable" if reached else "not_reached",
            role="unavailable" if reached else "not_reached",
            reason="checkpoint time not saved" if reached else "checkpoint not reached",
        )
    policy = section.get("exit_policy")
    if not candidates and not embedded:
        return dict(
            info,
            status="annotation",
            reason="no saved target-context decision",
            role="reporting_annotation_at_actual_funded_fill"
            if population == "funded"
            else "reporting_annotation_at_ordinary_candle_time",
            context_usage="reporting_only",
        )
    if len(candidates) != 1 or not unique_membership:
        return reject("target receipt or scoped trade membership is absent or ambiguous")
    ref = trade.get("trade_ref") or trade.get("strategy_trade_id") or trade.get("trade_id")
    required = {
        "configuration": trade.get("configuration"),
        "stream": population,
        "trade_id": ref,
        "event": "first_target",
        "policy": policy,
    }
    if any(receipt.get(k) != v for k, v in required.items()):
        return reject("receipt trade/configuration/stream/policy identity mismatch")
    for key in ("firm_key", "account_number", "account_id"):
        if receipt.get(key) is not None and receipt[key] != trade.get(key):
            return reject("receipt firm/account identity mismatch")
    if trade.get("stream", population) != population or policy not in CONDITIONAL:
        return reject("receipt is not a conditional decision for this stream/policy")
    if not at or ns(checkpoint) < ns(trade["entry_utc"]):
        return reject("missing receipt time or checkpoint precedes entry")
    action = "partial" if trade.get("scale_out_quantity") else "whole"
    if receipt.get("action") != action:
        return reject("saved action contradicts the actual target branch")
    if population == "funded":
        event_ns = ns(checkpoint)
        if (
            embedded.get("decision_ns") != event_ns
            or embedded.get("target_observation_ns") != event_ns
        ):
            return reject("embedded exact decision/observation does not identify the actual fill")
        if ns(at) != (event_ns // 1000) * 1000:
            return reject("receipt time is not the saved integer datetime conversion")
        if (
            embedded.get("action") != action
            or (embedded.get("context") or {}).get("receipt") != context
        ):
            return reject("embedded decision and scoped policy receipt disagree")
        full = index.snapshot(pd.Timestamp(at).to_pydatetime(warn=False))
        if (embedded.get("context") or {}).get("asof") != full:
            return reject("embedded source snapshot differs from the bound context source")
        method = "scoped_embedded_exact_decision_and_fill; integer_ns_to_microsecond_datetime"
    else:
        if trade.get("execution_policy") != "ordinary_completed_candle_exit_v1":
            return reject("ordinary candle timing contract unavailable")
        if ns(at) % 60_000_000_000 or ns(checkpoint) != ns(at) + 60_000_000_000:
            return reject("receipt does not name the actual completed candle checkpoint")
        full = index.snapshot(pd.Timestamp(at).to_pydatetime(warn=False))
        method = "scoped_trade_receipt; declared_1m_candle_open_to_close"
    if section.get("ifsm_context_policy_version") != full.get("policy_id"):
        return reject("context policy differs from the effective section")
    if not receipt_matches(context, full):
        return reject("receipt source/eligibility/value metadata differs from bound context")
    sign, age = context.get("gamma_sign"), context.get("positive_run_age")
    positive = sign == "positive"
    whole = (
        positive
        if policy == "gamma_conditional_1r_v1"
        else (
            positive
            and age is not None
            and 1 <= age <= 5
            and not context.get("positive_run_age_is_lower_bound")
        )
    )
    if action != ("whole" if whole else "partial"):
        return reject("saved action contradicts the bound conditional policy context")
    return dict(
        info,
        status="verified",
        reason="unique scoped decision, actual event and causal source verified",
        role="executed_saved_decision_receipt",
        match_method=method,
        context_usage="used_by_conditional_target_policy",
    )
