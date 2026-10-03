"""One way to write money, percentages, prices and Chicago times on screen.

Design system "Words": money like ``$30,781.88`` in tables and ``$30.8k`` on
axes; times like ``April 12, 2026, 7:07 PM`` and, in tables, ``Apr 12, 7:07 PM``.
Every time is converted from the stored instant with the shared repair-R4 helper
(:mod:`..chicago_time`); stored values are never changed. The zone name is left
out because every screen carries the note "All times Chicago, 12-hour"
(decision log, rule 11).

Missing values are never shown as zero: callers pass ``None`` and get the
placeholder text they choose.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import CHICAGO, utc_instant

__all__ = [
    "MISSING",
    "TICK_POINTS",
    "chicago_clock",
    "chicago_long",
    "chicago_short",
    "count",
    "date_long",
    "date_words",
    "date_range",
    "date_short",
    "DISPLAY_WORDS",
    "display_words",
    "money",
    "money_cents",
    "money_k",
    "money_short",
    "money_whole",
    "number",
    "percent",
    "points",
    "ticks_to_points",
    "trading_day_label",
]

MISSING = "—"
TICK_POINTS = Decimal("0.25")
_MINUS = "−"  # typographic minus, as in the mocks


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return isinstance(value, float) and math.isnan(value)
    except TypeError:
        return False


def _decimal(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def money(usd: Any, *, signed: bool = False, missing: str = MISSING, cents: bool = True) -> str:
    """``$30,781.88``; ``−$215.28`` when negative; ``+`` only when ``signed``."""

    if _is_missing(usd):
        return missing
    value = _decimal(usd).quantize(Decimal("0.01") if cents else Decimal(1),
                                   rounding=ROUND_HALF_UP)
    if value == 0:
        value = abs(value)
    sign = _MINUS if value < 0 else ("+" if signed and value > 0 else "")
    text = f"{abs(value):,.2f}" if cents else f"{abs(value):,.0f}"
    return f"{sign}${text}"


def money_cents(cents: Any, *, signed: bool = False, missing: str = MISSING) -> str:
    """Exact integer cents as money (``3078188`` → ``$30,781.88``)."""

    if _is_missing(cents):
        return missing
    return money(Decimal(int(cents)) / 100, signed=signed, missing=missing)


def money_whole(usd: Any, *, signed: bool = False, missing: str = MISSING) -> str:
    """Whole dollars (``$3,661``) for measures that are not exact cash."""

    return money(usd, signed=signed, missing=missing, cents=False)


def money_short(usd: Any, *, signed: bool = False, missing: str = MISSING) -> str:
    """Axis and callout form: ``$30.8k``, ``$950``, ``−$2.2k``."""

    if _is_missing(usd):
        return missing
    value = float(usd)
    sign = _MINUS if value < 0 else ("+" if signed and value > 0 else "")
    magnitude = abs(value)
    if magnitude >= 1000:
        text = f"{magnitude / 1000:,.1f}k"
        if text.endswith(".0k"):
            text = text[:-3] + "k"
    else:
        text = f"{magnitude:,.0f}"
    return f"{sign}${text}"


def money_k(usd: Any, *, missing: str = MISSING) -> str:
    """Thousands with one decimal, every value alike: ``−$0.7k``, ``$30.0k``, ``$59.5k``.

    For a table row that must read in one format (money_short drops ``.0`` and writes
    small values in dollars).
    """

    if _is_missing(usd):
        return missing
    value = float(usd)
    text = f"{abs(value) / 1000:,.1f}k"
    sign = _MINUS if value < 0 and text != "0.0k" else ""
    return f"{sign}${text}"


def number(value: Any, *, decimals: int = 2, missing: str = MISSING, signed: bool = False) -> str:
    if _is_missing(value):
        return missing
    text = f"{abs(float(value)):,.{decimals}f}"
    if float(value) < 0 and float(text.replace(",", "")) != 0:
        return _MINUS + text
    return ("+" if signed and float(value) > 0 else "") + text


def percent(share: Any, *, decimals: int = 0, missing: str = MISSING,
            already_percent: bool = False) -> str:
    """A share (``0.584``) as ``58%``; ``already_percent`` when given ``58.4``."""

    if _is_missing(share):
        return missing
    value = float(share) if already_percent else float(share) * 100
    text = f"{abs(value):.{decimals}f}%"
    return (_MINUS if value < 0 and float(text[:-1]) != 0 else "") + text


def ticks_to_points(ticks: Any) -> Decimal | None:
    if _is_missing(ticks):
        return None
    return Decimal(int(ticks)) * TICK_POINTS


def points(value: Any, *, missing: str = MISSING, from_ticks: bool = False) -> str:
    """Index points with two decimals (``24,971.00``)."""

    if from_ticks:
        value = ticks_to_points(value)
    if _is_missing(value):
        return missing
    return f"{_decimal(value):,.2f}"


def count(n: int, singular: str, plural: str | None = None) -> str:
    return f"{n:,} {singular if n == 1 else (plural or singular + 's')}"


def _local(value: Any) -> datetime | None:
    """Chicago wall time of a stored instant (a zone-less time is refused)."""

    instant = utc_instant(value)
    if instant is None:
        return None
    # displayed to the minute: drop sub-microsecond digits before leaving pandas
    return instant.tz_convert(CHICAGO).floor("us").to_pydatetime()


def _clock(local: datetime) -> str:
    hour = local.hour % 12 or 12
    return f"{hour}:{local.minute:02d} {'AM' if local.hour < 12 else 'PM'}"


def chicago_clock(value: Any, *, missing: str = MISSING) -> str:
    """``7:07 PM`` (Chicago)."""

    local = _local(value)
    return missing if local is None else _clock(local)


def chicago_long(value: Any, *, missing: str = MISSING) -> str:
    """``April 12, 2026, 7:07 PM`` (Chicago)."""

    local = _local(value)
    if local is None:
        return missing
    return f"{local:%B} {local.day}, {local.year}, {_clock(local)}"


def chicago_short(value: Any, *, missing: str = MISSING, year: bool = False) -> str:
    """``Apr 12, 7:07 PM`` (Chicago); ``year`` adds ``, 2026``."""

    local = _local(value)
    if local is None:
        return missing
    return f"{local:%b} {local.day}" + (f", {local.year}" if year else "") + f", {_clock(local)}"


def _as_date(value: Any) -> date | None:
    if _is_missing(value) or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def date_long(value: Any, *, missing: str = MISSING) -> str:
    """``January 13, 2026`` from a calendar date (``2026-01-13``)."""

    day = _as_date(value)
    return missing if day is None else f"{day:%B} {day.day}, {day.year}"


def date_short(value: Any, *, missing: str = MISSING) -> str:
    """``Jan 13`` from a calendar date."""

    day = _as_date(value)
    return missing if day is None else f"{day:%b} {day.day}"


def date_range(first: Any, last: Any, *, missing: str = MISSING) -> str:
    """``January 13 – June 10, 2026`` (years repeated only when they differ)."""

    start, end = _as_date(first), _as_date(last)
    if start is None or end is None:
        return missing
    if start.year == end.year:
        return f"{start:%B} {start.day} – {end:%B} {end.day}, {end.year}"
    return f"{date_long(start)} – {date_long(end)}"


def trading_day_label(day: Any, *, missing: str = MISSING) -> str:
    """A trading day by its closing date: ``April 13, 2026`` (runs 5:00 PM → 4:00 PM)."""

    return date_long(day, missing=missing)


# ── display wording for system-written texts ──────────────────────────────

#: (code name, words shown) applied in order to system-written texts only —
#: limitations, correction descriptions, notices and engine problems. Owner decision
#: texts and saved approvals are shown word for word, and no saved file is edited.
DISPLAY_WORDS: tuple[tuple[str, str], ...] = (
    (" (NQ)", ""),
    (" (MNQ)", ""),
    ("the Strategy-Core version", "the engine version"),
    ("the pinned Strategy-Core", "the default engine"),
    ("the research Strategy-Core checkout", "the half-exit engine's folder"),
    ("The shared pinned engine", "The default engine"),
    ("the pinned Core", "the default engine"),
    ("pinned Core", "the default engine"),
    ("The Strategy-Core", "The strategy engine"),
    ("the Strategy-Core", "the strategy engine"),
    ("Strategy-Core", "the strategy engine"),
)


def display_words(text: Any) -> str:
    """A system-written sentence with its code names replaced (``DISPLAY_WORDS``)."""

    out = str(text or "")
    for code, words in DISPLAY_WORDS:
        out = out.replace(code, words)
    # a sentence that now starts with "the strategy engine" is capitalized
    parts = [part[:1].upper() + part[1:] if part.startswith("the strategy engine") else part
             for part in _sentences(out)]
    return "".join(parts)


def _sentences(text: str) -> list[str]:
    import re

    pieces = re.split(r"((?:^|(?<=[.!?])\s+))", text)
    return [p for p in pieces if p]


def date_words(value: Any, *, missing: str = MISSING) -> str:
    """``September 24, 2026`` — a date in words, for new default study names."""

    day = _as_date(value)
    return f"{day:%B} {day.day}, {day.year}" if day is not None else missing
