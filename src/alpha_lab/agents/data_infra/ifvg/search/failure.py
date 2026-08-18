"""Typed failure attribution for search children (CONTRACTS_AND_SCHEMAS.md §12)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from .identities import SHA256_PATTERN, FrozenContract

__all__ = ["FailureReason", "ChildFailureRecord", "sanitize_failure_message"]


class FailureReason(StrEnum):
    """The complete 15-value failure vocabulary."""

    REPLAY = "replay"
    INVARIANT = "invariant"
    INSUFFICIENT_DAYS = "insufficient_days"
    INSUFFICIENT_TRADES = "insufficient_trades"
    NEGATIVE_EXPECTANCY = "negative_expectancy"
    DRAWDOWN = "drawdown"
    FUNDED_SURVIVAL = "funded_survival"
    BREACH = "breach"
    FEES = "fees"
    ONE_FIRM = "one_firm"
    STRESS = "stress"
    KNIFE_EDGE = "knife_edge"
    UNVERIFIED_CONTRACT = "unverified_contract"
    BLOCKED_AXIS = "blocked_axis"
    CANCELLED = "cancelled"


_SENSITIVE_TOKENS = ("C:\\", "c:\\", "/Users/", "\\Users\\", "Traceback (most recent")


def sanitize_failure_message(message: str, *, limit: int = 400) -> str:
    """Strip local paths and raw tracebacks from user-facing failure text."""

    cleaned_lines = []
    for line in str(message).splitlines():
        if any(token in line for token in _SENSITIVE_TOKENS):
            continue
        cleaned_lines.append(line.strip())
    cleaned = " ".join(part for part in cleaned_lines if part)
    if not cleaned:
        cleaned = "failure details withheld (sanitized)"
    return cleaned[:limit]


class ChildFailureRecord(FrozenContract):
    core_replay_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    parent_search_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    child_ordinal: int | None = Field(default=None, ge=0)
    reason: FailureReason
    gate_id: str | None = None
    sanitized_message: str
    evidence_refs: tuple[str, ...] = ()
