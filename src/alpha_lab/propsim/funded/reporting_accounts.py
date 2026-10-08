"""Scoped account attribution at the read boundary; never changes ledgers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

VERSION = "scoped_reporting_account_v1"


def _token(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"invalid reporting {field}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"empty reporting {field}")
    if text.isdecimal():
        if int(text) <= 0:
            raise ValueError(f"invalid reporting {field}")
        return str(int(text))
    return text


def account_token(row: Mapping[str, Any]) -> str | None:
    """Accept account_number or legacy account_id; conflicting aliases fail."""
    old = _token(row.get("account_id"), "account_id")
    number = _token(row.get("account_number"), "account_number")
    if old is not None and number is not None and old != number:
        raise ValueError("account_identity_conflict: account_id and account_number differ")
    return number if number is not None else old


@dataclass(frozen=True)
class ReportingAccount:
    result_id: str
    configuration: str
    firm: str
    account: str

    @property
    def key(self) -> str:
        # JSON encodes components unambiguously, including arbitrary legacy IDs.
        return json.dumps([self.result_id, self.configuration, self.firm, self.account],
                          separators=(",", ":"))


def resolve_account(row: Mapping[str, Any], *, result_id: str,
                    configuration: str | None = None,
                    firm: str | None = None) -> ReportingAccount | None:
    account = account_token(row)
    if account is None:
        return None
    configuration = configuration or row.get("configuration") or row.get("variant_id")
    firm = firm or row.get("firm_key")
    values = (result_id, configuration, firm)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("reporting account requires economic result, configuration and firm")
    return ReportingAccount(result_id, configuration, firm, account)


def account_row(row: Mapping[str, Any], *, result_id: str) -> dict[str, Any]:
    """A derived copy with shared attribution, preserving original money/events."""
    account = resolve_account(row, result_id=result_id)
    output = dict(row)
    if account is not None:
        output["account_id"] = account.account
        output["reporting_account_key"] = account.key
        if account.account.isdecimal():
            output["account_number"] = int(account.account)
    return output
