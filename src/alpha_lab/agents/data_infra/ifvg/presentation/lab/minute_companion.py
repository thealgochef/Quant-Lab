"""The published approximated-minute companion, linked to one funded trade (correction A10).

A saved funded comparison's review folder
(``reports/funded_comparison/funded_comparison_<result>_export_v<N>/``) publishes
``approximated_minutes.csv``: one row per one-minute bar whose recorded exchange
trades did not rebuild the candle exactly, per funded trade row it touches. Each
row names the trade by configuration, firm, account number, Chicago entry and exit
texts (``April 12, 2026 07:07 PM CDT``), exit kind and net result, and the minute by
its UTC opening instant and Chicago text. It has no trade id or sequence column.

This module links those rows to ONE recorded funded trade — never by re-running
anything and never by re-formatting the trade's times:

- identity: configuration + firm name + account number + entry minute + exit minute
  + exit kind + net result. The companion's Chicago texts are PARSED to instants
  (the zone abbreviation settles the repeated fall-back hour) and compared with the
  trade's recorded instants floored to the minute; the result is compared in cents;
- checks: exactly one trade of the configuration-and-firm pair has that identity
  (``identity_unique``); every linked minute opens inside [entry minute, exit instant]
  (``minutes_inside_trade``); each minute's Chicago text is the same instant as its
  UTC opening (``minute_times_agree``); no minute is listed twice for the trade
  (``no_duplicate_minutes``); the number of linked rows equals the trade's
  ``minutes_approximated`` when that field is saved (``count_matches``); every row of
  this configuration, firm and account can be read (``times_readable``); no row names
  this trade's account, entry and exit minutes with an exit kind or result that no
  trade of the pair recorded (``no_mismatched_rows``).

Any failed check gives ``"conflict"`` with its reason; a trade with approximated
minutes and no row is a conflict too; a trade without approximated minutes and no
row (and no mismatched row) is ``"none"``. The file is trusted only through the
published folder's manifest hash (``ifvg_lab_detail_settings.ReviewFolder.read``
returns bytes only when their SHA-256 matches); no folder, an unreadable file or a
hash mismatch is ``"unavailable"``. Nothing here reads a file or writes anything.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import CHICAGO, utc_instant

__all__ = [
    "COMPANION_FILE",
    "REQUIRED_COLUMNS",
    "STATUSES",
    "CompanionSource",
    "MinuteLink",
    "link_from_review_folder",
    "link_trade",
    "parse_chicago_text",
    "parse_companion",
    "short_cause",
    "trade_identity",
]

COMPANION_FILE = "approximated_minutes.csv"
STATUSES = ("linked", "none", "conflict", "unavailable")
REQUIRED_COLUMNS = ("configuration", "firm", "account_number", "trade_entry_chicago",
                    "trade_exit_chicago", "trade_exit_kind", "trade_net_result_usd",
                    "minute_open_utc", "minute_chicago")
#: the order the checks are reported in (the first failure is the reason given)
CHECKS = ("times_readable", "no_mismatched_rows", "identity_unique", "minutes_inside_trade",
          "minute_times_agree", "no_duplicate_minutes", "count_matches")

_MONTHS = {name: number for number, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August", "September",
     "October", "November", "December"), start=1)}
_MONTHS.update({name[:3]: number for name, number in list(_MONTHS.items())})
_CHICAGO_TEXT = re.compile(
    r"^\s*(?P<month>[A-Za-z]+)\.? (?P<day>\d{1,2}), (?P<year>\d{4}),? "
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))? (?P<half>AM|PM) "
    r"(?P<zone>CST|CDT)\s*$")
_CENT = Decimal("0.01")


@dataclass(frozen=True)
class CompanionSource:
    """Where the linked rows come from: the published folder and the file's manifest hash."""

    folder: str
    version: int
    result_id: str
    sha256: str


@dataclass(frozen=True)
class MinuteLink:
    """The companion evidence for one trade: ``status`` is one of :data:`STATUSES`."""

    status: str
    rows: tuple[dict[str, str], ...] = ()
    reason: str = ""
    checks: dict[str, bool] = field(default_factory=dict)
    source: CompanionSource | None = None

    @property
    def minutes(self) -> list[pd.Timestamp | None]:
        """Each linked row's minute opening instant (UTC), in row order."""

        return [_utc(row.get("minute_open_utc")) for row in self.rows]


# ── reading the file and its texts ────────────────────────────────────────


def parse_companion(data: bytes) -> list[dict]:
    """The companion's rows (all values as the published text); refuses a wrong header."""

    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or ())]
    if missing:
        raise ValueError(f"{COMPANION_FILE} lacks the columns {missing}")
    return [dict(row) for row in reader]


def parse_chicago_text(text: Any) -> pd.Timestamp | None:
    """``April 13, 2026 2:03:00 AM CDT`` → the UTC instant; ``None`` if it can't be read.

    The abbreviation must be the zone Chicago actually used at that wall time (it
    settles the repeated hour when clocks fall back); a wall time that didn't exist
    or a wrong abbreviation is unreadable, never guessed.
    """

    found = _CHICAGO_TEXT.match(str(text or ""))
    if not found:
        return None
    month = _MONTHS.get(found["month"].capitalize())
    hour = int(found["hour"])
    if month is None or not 1 <= hour <= 12:
        return None
    hour = hour % 12 + (12 if found["half"] == "PM" else 0)
    try:
        wall = pd.Timestamp(year=int(found["year"]), month=month, day=int(found["day"]),
                            hour=hour, minute=int(found["minute"]),
                            second=int(found["second"] or 0))
        local = wall.tz_localize(CHICAGO, ambiguous=found["zone"] == "CDT",
                                 nonexistent="raise")
    except Exception:  # a date that doesn't exist, or a wall time skipped by the clock change
        return None
    if local.tzname() != found["zone"]:
        return None
    return local.tz_convert("UTC")


def _utc(value: Any) -> pd.Timestamp | None:
    try:
        return utc_instant(value)
    except (TypeError, ValueError):
        return None


def _cents(value: Any) -> Decimal | None:
    """Money text (``6,254.72``, ``-190.28``, ``−190.28``, ``(190.28)``) or a number, in cents."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float, Decimal)):
        text = str(value)
    else:
        text = str(value).strip().replace("$", "").replace(",", "").replace(" ", "")
        text = text.replace("−", "-")
        if text.startswith("(") and text.endswith(")"):
            text = "-" + text[1:-1]
    try:
        return Decimal(text).quantize(_CENT, rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError):
        return None


def _account(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def short_cause(text: Any, *, limit: int = 160) -> str:
    """The companion's cause in one short clause: first sentence, no bracketed detail."""

    cause = re.sub(r"\s*\([^()]*\)", "", str(text or "")).strip()
    # a sentence ends at a period followed by a space (a decimal point never is)
    cause = re.split(r"\.\s+", cause, maxsplit=1)[0].strip().rstrip(".")
    if len(cause) > limit:
        cause = cause[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    return cause


# ── identity and linking ──────────────────────────────────────────────────


def trade_identity(trade: Any) -> tuple:
    """(account, entry minute, exit minute, exit kind, result in cents) of a recorded trade.

    ``trade`` is a review ``TradeView`` (or anything with ``account``, ``entry_utc``,
    ``exit_utc``, ``exit_kind`` and ``net``); instants are floored to the minute, the
    recorded instants themselves are never changed.
    """

    entry, exit_ = getattr(trade, "entry_utc", None), getattr(trade, "exit_utc", None)
    return (getattr(trade, "account", None),
            entry.floor("min") if entry is not None else None,
            exit_.floor("min") if exit_ is not None else None,
            str(getattr(trade, "exit_kind", "") or ""),
            _cents(getattr(trade, "net", None)))


def _row_identity(row: Mapping[str, Any]) -> tuple | None:
    parts = (_account(row.get("account_number")),
             parse_chicago_text(row.get("trade_entry_chicago")),
             parse_chicago_text(row.get("trade_exit_chicago")),
             str(row.get("trade_exit_kind") or ""), _cents(row.get("trade_net_result_usd")))
    return None if any(p is None for p in parts) else parts


def _readable(row: Mapping[str, Any]) -> bool:
    return (_row_identity(row) is not None and _utc(row.get("minute_open_utc")) is not None
            and parse_chicago_text(row.get("minute_chicago")) is not None)


def link_trade(rows: Iterable[Mapping[str, Any]], *, trade: Any, pair_trades: Sequence[Any],
               configuration: str, firm_name: str,
               source: CompanionSource | None) -> MinuteLink:
    """Link the companion rows of exactly this trade, or say why they can't be linked."""

    own = [dict(r) for r in rows
           if str(r.get("configuration")) == str(configuration)
           and str(r.get("firm")) == str(firm_name)]
    target = trade_identity(trade)
    account = target[0]
    mine = [r for r in own if _account(r.get("account_number")) in (account, None)]
    linked = [r for r in mine if _row_identity(r) == target]
    minutes = [_utc(r.get("minute_open_utc")) for r in linked]
    order = sorted(range(len(linked)), key=lambda i: (minutes[i] is None, minutes[i] or 0))
    linked, minutes = [linked[i] for i in order], [minutes[i] for i in order]
    approx = getattr(trade, "minutes_approximated", None)
    recorded = {trade_identity(t) for t in (pair_trades or [trade])} | {target}
    same = [t for t in (pair_trades or [trade]) if trade_identity(t) == target]
    entry_minute, exit_at = target[1], getattr(trade, "exit_utc", None)
    # a row naming this trade's account, entry and exit minutes with an exit or result no
    # trade of the pair recorded: the record disagrees with this trade, never "no minute"
    mismatched = [r for r in mine if (ident := _row_identity(r)) is not None
                  and ident[:3] == target[:3] and ident not in recorded]
    checks = {
        "times_readable": all(_readable(r) for r in mine),
        "no_mismatched_rows": not mismatched,
        "identity_unique": len(same) == 1,
        "minutes_inside_trade": all(m is not None and entry_minute is not None
                                    and exit_at is not None and entry_minute <= m <= exit_at
                                    for m in minutes),
        "minute_times_agree": all(parse_chicago_text(r.get("minute_chicago")) == m
                                  for r, m in zip(linked, minutes, strict=True)),
        "no_duplicate_minutes": len(set(minutes)) == len(minutes),
        "count_matches": approx is None or len(linked) == int(approx),
    }
    if not linked and not approx and checks["times_readable"] and checks["no_mismatched_rows"]:
        return MinuteLink("none", (), "", checks, source)
    for name in CHECKS:
        if not checks[name]:
            reason = _reason(name, linked=len(linked), approx=approx)
            return MinuteLink("conflict", tuple(linked), reason, checks, source)
    return MinuteLink("linked", tuple(linked), "", checks, source)


def _reason(check: str, *, linked: int, approx: Any) -> str:
    if check == "count_matches" and not linked:
        return "it lists no minute for this trade"
    if check == "count_matches":
        return (f"it lists {linked:,} {'minute' if linked == 1 else 'minutes'} where this trade "
                f"recorded {int(approx):,}")
    return {
        "times_readable": "a time listed for this account can't be read",
        "no_mismatched_rows": ("it lists a minute for this account, entry and exit time with a "
                               "different exit or result"),
        "identity_unique": ("another trade of this configuration at this firm has the same "
                            "account, entry and exit minutes, exit and result"),
        "minutes_inside_trade": "a listed minute falls outside this trade's entry-to-exit time",
        "minute_times_agree": "a listed minute's Chicago time and UTC time disagree",
        "no_duplicate_minutes": "a minute is listed twice for this trade",
    }[check]


def link_from_review_folder(folder: Any, *, result_id: str, trade: Any,
                            pair_trades: Sequence[Any], configuration: str,
                            firm_name: str) -> MinuteLink:
    """The link through the published review folder of ``result_id`` (hash-checked).

    ``folder`` is what ``ifvg_lab_detail_settings.latest_review_folder(repo_root,
    result_id)`` returned: a folder whose manifest names this exact result, or ``None``.
    Its ``read`` returns the file's bytes only when they match the manifest's SHA-256.
    """

    if folder is None:
        return MinuteLink("unavailable", reason="no published review folder names this result")
    expected = (getattr(folder, "files", None) or {}).get(COMPANION_FILE)
    source = CompanionSource(folder=str(folder.name), version=int(folder.version),
                             result_id=str(result_id), sha256=str(expected or ""))
    if not expected:
        return MinuteLink("unavailable", reason=f"the published folder doesn't list "
                                                f"{COMPANION_FILE}", source=source)
    try:
        data = folder.read(COMPANION_FILE)
    except OSError:
        data = None
    if data is None:
        return MinuteLink("unavailable", reason=f"{COMPANION_FILE} is missing or failed its "
                                                "hash check", source=source)
    try:
        rows = parse_companion(data)
    except (ValueError, UnicodeDecodeError, csv.Error):
        return MinuteLink("unavailable", reason=f"{COMPANION_FILE} can't be read",
                          source=source)
    return link_trade(rows, trade=trade, pair_trades=pair_trades, configuration=configuration,
                      firm_name=firm_name, source=source)
